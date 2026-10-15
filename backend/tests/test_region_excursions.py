"""Known excursions, unknown paths, chronology and denominator regressions."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts.zone_excursions import (
    DEFINITIONS, bootstrap_weights, compare_labels, label_anchors,
)


BASE = pd.Timestamp("2026-01-01T00:00Z")
SECONDS = BASE.value // 10**9


def source(changes=None, missing=()):
    t, p = SECONDS + np.arange(4000, dtype=np.int64), np.full(4000, 100.)
    for i, v in (changes or {}).items(): p[i] = v
    good = ~np.isin(t - SECONDS, missing)
    return t[good], p[good]


def anchors(endpoint="ISSUE_DELAYED", side=1, **overrides):
    r = {"issue_time": BASE, "anchor_time": BASE + pd.Timedelta(seconds=61 if endpoint == "ISSUE_DELAYED" else 62),
         "atr": 1., "side": side}
    r.update(overrides)
    return pd.DataFrame([r])


def labels(changes=None, missing=(), endpoint="ISSUE_DELAYED", **kwargs):
    t, p = source(changes, missing)
    return label_anchors(t, p, anchors(endpoint, **kwargs), start=BASE,
                         end=BASE + pd.Timedelta(seconds=4000), endpoint=endpoint)


def test_all_twelve_definitions_and_first_exact_second_crossings():
    r = labels({100: 101.5, 361: 102., 662: 103.}).iloc[0]
    assert len(DEFINITIONS) == 12
    assert r["y_a1.5_h05"] == 1
    assert r["tts_a1.5_h05"] == 39
    assert r.y_a2_h05 == 1 and r.tts_a2_h05 == 300
    assert r.y_a3_h10 == 0 and r.y_a3_h15 == 1 and r.tts_a3_h15 == 601


def test_hit_after_horizon_is_not_counted_and_prior_jump_is_not_an_entry_catch():
    r = labels({360: 101., 362: 103.}).iloc[0]
    assert r.y_a3_h05 == 0 and r.y_a3_h10 == 1
    r = labels({61: 200., 62: 200.}, endpoint="ENTRY_CONDITIONAL").iloc[0]
    assert r.anchor_price == 200. and r.y_a3_h30 == 0


@pytest.mark.parametrize("gap", [62, 100, 361])
def test_gap_anywhere_in_horizon_is_unknown_even_after_early_success(gap):
    r = labels({63: 110.}, missing=[gap]).iloc[0]
    assert r.y_a3_h05 == -1 and pd.isna(r.tts_a3_h05)
    assert r.reason_h05 == "incomplete_full_horizon"


def test_gap_after_short_horizon_does_not_censor_shorter_target():
    r = labels({63: 110.}, missing=[362]).iloc[0]
    assert r.y_a3_h05 == 1 and r.y_a3_h10 == -1


def test_missing_exact_anchor_is_not_replaced_with_a_later_quote():
    r = labels({62: 100.}, missing=[61]).iloc[0]
    assert r.reason_h30 == "missing_anchor" and pd.isna(r.anchor_price)


@pytest.mark.parametrize("endpoint,purge", [("ISSUE_DELAYED", 32), ("ENTRY_CONDITIONAL", 46)])
def test_fixed_purge_cannot_be_rescued_by_an_early_hit(endpoint, purge):
    t, p = source({63: 110.})
    result = label_anchors(t, p, anchors(endpoint), start=BASE,
        end=BASE + pd.Timedelta(minutes=purge) - pd.Timedelta(seconds=1), endpoint=endpoint)
    assert result.iloc[0].reason_h05 == "planned_purge" and result.iloc[0].y_a3_h05 == -1


def test_future_quote_values_outside_partition_do_not_change_labels():
    t, p = source({100: 103.})
    expected = label_anchors(t, p, anchors(), start=BASE, end=BASE + pd.Timedelta(seconds=3900), endpoint="ISSUE_DELAYED")
    p[t >= SECONDS + 3900] = np.nan
    actual = label_anchors(t, p, anchors(), start=BASE, end=BASE + pd.Timedelta(seconds=3900), endpoint="ISSUE_DELAYED")
    assert_frame_equal(expected, actual, check_exact=True)


def test_native_crash_direction_uses_falls_not_rises():
    r = labels({100: 97.}, side=-1).iloc[0]
    assert r.y_a3_h05 == 1 and r.tts_a3_h05 == 39
    assert labels({100: 103.}, side=-1).iloc[0].y_a3_h05 == 0


def frame(values):
    issue = pd.date_range(BASE, periods=len(values), freq="D")
    return pd.DataFrame({"issue_time": issue, "anchor_time": issue + pd.Timedelta(seconds=61),
        "y_a2_h15": values, "tts_a2_h15": [60. if v == 1 else np.nan for v in values],
        "reason_h15": ["known_full_horizon" if v >= 0 else "incomplete_full_horizon" for v in values]})


def test_precision_recall_and_unknown_bounds_use_distinct_denominators():
    base = frame([1, 0, 1, -1])
    model = base.iloc[[0, 3]]
    days, w = bootstrap_weights(BASE, BASE + pd.Timedelta(days=4), repeats=99)
    r = compare_labels(model, base, 2., 15, days=days, weights=w, opportunity_recall=True)
    assert r["precision"] == 1. and r["base_rate"] == 2 / 3 and r["lift"] == 1.5
    assert r["opportunity_recall"] == .5 and r["unique_event_recall"] is None
    assert r["model"]["unknown"] == 1 and r["model"]["rate_bounds_with_unknown"] == [.5, 1.]
    assert r["median_time_to_excursion_minutes_from_issue"] == pytest.approx(121 / 60)
    assert r["QUALIFIED"] is False


def test_entry_conditional_rate_does_not_claim_identified_event_recall():
    days, w = bootstrap_weights(BASE, BASE + pd.Timedelta(days=8), repeats=99)
    r = compare_labels(frame([1, 0]), frame([1, 1, 0]), 2., 15,
                       days=days, weights=w, opportunity_recall=False)
    assert r["opportunity_recall"] is None and r["unique_event_recall"] is None


def test_inconsistent_subset_labels_are_refused():
    days, w = bootstrap_weights(BASE, BASE + pd.Timedelta(days=8), repeats=19)
    with pytest.raises(ValueError, match="subset"):
        compare_labels(frame([0]), frame([1]), 2., 15, days=days, weights=w, opportunity_recall=True)


def test_empty_labels_never_create_infinite_lift_or_significance():
    days, w = bootstrap_weights(BASE, BASE + pd.Timedelta(days=8), repeats=19)
    r = compare_labels(frame([]), frame([]), 2., 15, days=days, weights=w, opportunity_recall=True)
    assert r["precision"] is None and r["lift"] is None and r["p"] == 1.
