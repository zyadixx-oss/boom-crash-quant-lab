"""Independent auditor boundary and tamper fixtures; no historical prices."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

P = Path(__file__).resolve().parents[2] / "scripts/verify_zone_excursions.py"
SPEC = importlib.util.spec_from_file_location("region_excursion_auditor", P)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def fixture(endpoint="ISSUE_DELAYED"):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    issue = start + pd.Timedelta(minutes=1)
    anchor = issue + pd.Timedelta(seconds=61 if endpoint == "ISSUE_DELAYED" else 62)
    times = np.arange(A.second(start), A.second(start) + 3601)
    prices = np.full(len(times), 100.)
    row = {"issue_time": issue, "anchor_time": anchor, "atr": 2., "side": 1}
    return times, prices, row, A.second(start), A.second(start) + 3601


@pytest.mark.parametrize("endpoint", ["ISSUE_DELAYED", "ENTRY_CONDITIONAL"])
@pytest.mark.parametrize("side", [-1, 1])
def test_original_quote_first_crossing_and_closed_endpoint(endpoint, side):
    t, p, r, start, end = fixture(endpoint)
    r["side"] = side
    a = A.second(r["anchor_time"])
    p[t == a - 1] = 100 + side * 100  # Never label the earlier move.
    p[t == a + 300] = 100 + side * 4
    out = A.oracle(t, p, r, start, end, endpoint)
    assert out["y_a2_h05"] == 1 and out["tts_a2_h05"] == 300
    assert out["y_a3_h05"] == 0 and out["tts_a3_h05"] is None
    assert out["anchor_price"] == 100.


def test_missing_after_early_hit_still_unknown_all_horizon():
    t, p, r, start, end = fixture()
    a = A.second(r["anchor_time"])
    p[t == a + 1] = 110
    keep = t != a + 301
    out = A.oracle(t[keep], p[keep], r, start, end, "ISSUE_DELAYED")
    assert out["y_a3_h05"] == 1 and out["tts_a3_h05"] == 1
    assert out["y_a1.5_h10"] == -1 and out["tts_a1.5_h10"] is None
    assert out["reason_h10"] == "incomplete_full_horizon"


@pytest.mark.parametrize("endpoint,purge", [("ISSUE_DELAYED", 1920), ("ENTRY_CONDITIONAL", 2760)])
def test_fixed_planned_purge_all_definitions(endpoint, purge):
    t, p, r, start, end = fixture(endpoint)
    end = A.second(r["issue_time"]) + purge - 1
    out = A.oracle(t, p, r, start, end, endpoint)
    assert all(out[f"y_a{a:g}_h{h:02d}"] == -1 for a, h in A.DEFINITIONS)
    assert all(out[f"reason_h{h:02d}"] == "planned_purge" for h in (5, 10, 15, 30))


def test_missing_anchor_is_unknown_without_carry_forward():
    t, p, r, start, end = fixture()
    keep = t != A.second(r["anchor_time"])
    out = A.oracle(t[keep], p[keep], r, start, end, "ISSUE_DELAYED")
    assert out["anchor_price"] is None and out["reason_h05"] == "missing_anchor"
    assert out["y_a1.5_h05"] == -1


def test_audit_detects_tampered_label_and_numeric_value():
    audit = A.Audit()
    audit.equal(0, 1, "changed label")
    audit.equal(.25, .30, "changed precision")
    audit.equal(None, None, "valid unknown")
    assert audit.checks == 3 and len(audit.errors) == 2


def test_membership_detects_past_population_change():
    f = pd.DataFrame({"issue_time": ["2026-01-01T00:00Z"],
        "anchor_time": ["2026-01-01T00:01:01Z"], "atr": [2.], "side": [1]})
    changed = f.copy()
    changed["atr"] = 3.
    audit = A.Audit()
    A.check_membership(audit, changed, f, "fixture")
    assert audit.errors == ["fixture/atr"]


def test_empty_entry_schema_is_a_valid_zero_population():
    ledger = pd.DataFrame(columns=["signal_time", "entry_time", "atr", "side"])
    expected = A.expected_entry(ledger)
    audit = A.Audit()
    A.check_membership(audit, expected.copy(), expected, "empty")
    assert not audit.errors


def test_zero_reference_lift_is_undefined_and_holm_monotone():
    assert np.isnan(A.fraction(np.array([1.]), np.array([0.]))[0])
    assert A.interval(np.array([np.nan])) == [None, None]
    assert A.holm([.01, .04, .03]).tolist() == pytest.approx([.03, .06, .06])


def test_no_production_engine_import():
    text = P.read_text()
    assert "from app." not in text and "from scripts." not in text
