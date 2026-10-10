import importlib.util
from pathlib import Path

import numpy as np
import pytest

P = Path(__file__).resolve().parents[2] / "scripts/verify_representation_availability.py"
S = importlib.util.spec_from_file_location("representation_availability_oracle", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def source():
    times = np.arange(24 * 300, dtype=np.int64)
    prices = 100 * np.exp(-times * 1e-5)
    prices[17 * 300 + 150:] *= 1.05
    return times, prices


def test_removed_native_move_does_not_invent_a_known_age():
    t, p = source(); r = M.native_increments(t, p)
    frame, details = M.independent_m5(t, r, .001, 1, 0, len(t))
    assert details["tail_increments_removed"] == 1
    assert details["large_transformed_m5_bars"] == 0
    assert np.isnan(frame["log1p_large_bar_age"]).all()
    assert details["known_increments"] == len(t) - 1


def test_actual_large_bar_initializes_then_increments_age():
    t, p = source(); r = M.native_increments(t, p)
    f, d = M.independent_m5(t, r, 1., 1, 0, len(t))
    assert d["tail_increments_removed"] == 0
    assert f["large_completed_bar"][17] == 1
    assert f["log1p_large_bar_age"][17] == 0
    assert f["log1p_large_bar_age"][18] == pytest.approx(np.log(2))


def test_gap_resets_age_and_partial_bar_stays_unknown():
    t, p = source(); keep = t != 20 * 300 + 100
    t, p = t[keep], p[keep]
    f, d = M.independent_m5(t, M.native_increments(t, p), 1., 1, 0, 24 * 300)
    assert d["runs"] == 2 and np.isnan(f["close"][20])
    assert np.isnan(f["log1p_large_bar_age"][21:]).all()


def test_future_quotes_cannot_change_training_scale():
    t, p = source()
    base = M.training_scale(t, M.native_increments(t, p), 0, 3600)
    p[3600:] *= 20
    assert M.training_scale(t, M.native_increments(t, p), 0, 3600) == base


def test_training_scale_excludes_boundary_straddling_increment():
    t = np.arange(0, 12, dtype=np.int64)
    p = np.exp(np.arange(12) * .01)
    p[5:] *= 10
    scale = M.training_scale(t, M.native_increments(t, p), 5, 12)
    assert scale == pytest.approx(.01)


def test_oracle_keeps_planned_empty_grid():
    f, d = M.independent_m5(np.array([], dtype=np.int64), np.array([], dtype=float), .001, 1, 0, 601)
    assert len(f["close"]) == 3 and np.isnan(f["close"]).all()
    assert d["known_increments"] == 0


def test_numeric_tampering_is_detected_but_matching_unknowns_are_valid():
    a = M.Audit()
    a.array([np.nan, 1.], [np.nan, 1.], "good", close=True)
    a.array([np.nan, 2.], [np.nan, 1.], "tampered", close=True)
    assert a.errors == ["tampered"]


def test_independent_oracle_imports_no_research_engine():
    text = P.read_text()
    assert "from app." not in text and "from scripts." not in text
