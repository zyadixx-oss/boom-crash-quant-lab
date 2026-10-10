"""Independent scalar implementation checked on known synthetic and adversarial inputs."""
import ast
import math

import pandas as pd
import pytest

from app.research.tick_tail import FixedTailDetector
from app.research.tail_successor import collect_successor_days, summarize_successor
from scripts import verify_tail_successor as verifier

BASE = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())


def test_verifier_does_not_import_research_helper_or_study_runner():
    tree = ast.parse(verifier.Path(verifier.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("app.", "scripts."))


def test_scalar_response_excludes_event_jump_and_first_next_move():
    rows = [(BASE + i, q) for i, q in enumerate([100., 200., 201., 202., 203.])]
    d = verifier.reconstruct_days(rows, 1, .01, BASE, BASE + 5)[0]
    assert d["event_eligible"] == 1
    assert d["event_response_sum"] == math.log1p(1 / 201)
    assert d["reference_eligible"] == 2
    assert d["boundary_excluded_anchors"] == 2


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("missing", [None, 2, 3, 5, 11])
def test_epoch_lookup_and_adjacent_implementations_agree_on_synthetic_gaps(side, missing):
    quotes = [100., 200., 201., 402., 403., 404., 808., 809., 810., 811., 812., 813., 814.]
    if side == -1:
        quotes = [1e5 / q for q in quotes]
    rows = [(BASE + i, q) for i, q in enumerate(quotes) if i != missing]
    frame = pd.DataFrame({"quote": [r[1] for r in rows]}, index=pd.to_datetime([r[0] for r in rows], unit="s", utc=True))
    daily = collect_successor_days(frame, FixedTailDetector(side, .01), BASE + 1, BASE + 13)
    expected = verifier.reconstruct_days(rows, side, .01, BASE + 1, BASE + 13)
    verifier.Audit().compare(daily, expected)
    verifier.Audit().compare(summarize_successor(daily), verifier.reconstruct_summary(expected))


def test_independent_midnight_and_partition_start_context():
    rows = [(BASE + 86398 + i, q) for i, q in enumerate([100., 200., 400., 401., 402., 403., 404.])]
    daily = verifier.reconstruct_days(rows, 1, .01, BASE + 86399, BASE + 86405)
    assert daily[0]["boundary_excluded_events"] == 1
    assert daily[1]["detector_pair_exclusions"] == 1
    assert daily[1]["detected_event_anchors"] == 0


@pytest.mark.parametrize("actual,expected", [(True, 1), (1, True), (0., None), ({"x": 1, "extra": 2}, {"x": 1}),
    ([1], [1, 2]), (float("nan"), 1.), (1.01, 1.)])
def test_audit_rejects_type_schema_nonfinite_and_numeric_tampering(actual, expected):
    with pytest.raises(ValueError):
        verifier.Audit().compare(actual, expected)


@pytest.mark.parametrize("key", [*verifier.FLAGS, "goal_achieved", "historical_strategy_candidate", "strategy_eligible", "physical_spike_census"])
def test_audit_rejects_any_promoted_or_execution_gate(key):
    with pytest.raises(ValueError):
        verifier.Audit().safety({"nested": [{key: True}]})


@pytest.mark.parametrize("flag", verifier.FLAGS)
def test_environment_guard_precedes_historical_file_access(monkeypatch, flag, tmp_path):
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match="flags"):
        verifier.verify(tmp_path, results_ready=True)


def test_explicit_results_ready_precedes_historical_file_access(tmp_path):
    with pytest.raises(ValueError, match="results-ready"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize("body", ["epoch,quote\n1,100,extra\n", "epoch,quote\n01,100\n", "epoch,quote\n1,nan\n",
    "epoch,quote\n1,0\n", "epoch,quote\n1,100\n1,101\n", "epoch,price\n1,100\n"])
def test_invalid_quote_csv_refused(tmp_path, body):
    path = tmp_path / "ticks.csv"; path.write_text(body)
    with pytest.raises(ValueError):
        verifier.read_quotes(path, {"start_epoch": 0, "end_exclusive_epoch": 86400,
            "rows": 1, "missing_seconds": 86399, "gap_free": False})


@pytest.mark.parametrize("label", ["../x", "./x", "/tmp/x", "x//y", "", 1])
def test_independent_path_guard_refuses_alias_or_traversal(label):
    with pytest.raises(ValueError):
        verifier.safe_path(label)


def test_scalar_shared_bootstrap_preserves_unknown_and_zero_event_day_counts():
    rows = [(BASE + t, 100. if t == 0 else 200. + t) for t in [0, 1, 3, 4, 5, 6]]
    rows += [(BASE + 86400 + t, 100. + .01 * t) for t in range(7)]
    days = verifier.reconstruct_days(rows, 1, .01, BASE, BASE + 86407)
    value = verifier.reconstruct_summary(days)
    assert value["totals"]["event_unknown"] == 1
    assert value["observed_day_clusters"] == 2
    assert value["bootstrap"]["event_mean"]["undefined_draws"] == 9999
    assert value["conditional_positive_excess_rejected"] is False
