"""Synthetic causality, exact successor availability and shared-day inference."""
import math
import pandas as pd
import pytest

from app.research.tick_tail import FixedTailDetector, _log_return
from app.research.tail_successor import COUNT_FIELDS, collect_successor_days, summarize_successor


BASE = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())


def frame(quotes, seconds=None):
    if seconds is None:
        seconds = range(len(quotes))
    return pd.DataFrame({"quote": quotes}, index=pd.to_datetime([BASE + t for t in seconds], unit="s", utc=True))


def collect(quotes, seconds=None, start=0, end=None, side=1, median=.01):
    data = frame(quotes, seconds)
    if end is None:
        end = int(data.index[-1].timestamp()) - BASE + 1
    return collect_successor_days(data, FixedTailDetector(side, median), BASE + start, BASE + end)


def day(date="2026-01-01", events=100, known=None, event_sum=-.001, refs=1000,
        reference_sum=0., unknown_refs=0):
    if known is None:
        known = events
    return {"date": date, "anchor_rows": refs, "detector_pair_exclusions": 0,
        "detector_eligible_anchors": refs, "detected_event_anchors": events,
        "boundary_excluded_anchors": 0, "boundary_excluded_events": 0,
        "reference_eligible": refs, "reference_known": refs - unknown_refs,
        "reference_unknown": unknown_refs, "event_eligible": events, "event_known": known,
        "event_unknown": events - known, "overlapping_consecutive_event_windows": 0,
        "reference_response_sum": reference_sum, "event_response_sum": event_sum}


def test_response_excludes_event_jump_and_first_successor_move():
    result = collect([100., 200., 201., 202., 203.])[0]
    assert result["event_eligible"] == result["event_known"] == 1
    assert result["event_response_sum"] == _log_return(201., 202.)
    assert result["event_response_sum"] != _log_return(100., 200.)
    assert result["event_response_sum"] != _log_return(200., 201.)
    assert result["reference_eligible"] == 2
    assert result["reference_response_sum"] == math.fsum([_log_return(201., 202.), _log_return(202., 203.)])


def test_reference_includes_events_and_all_consecutive_events_are_retained():
    result = collect([100., 200., 400., 800., 1600., 3200.])[0]
    assert result["event_eligible"] == result["reference_eligible"] == 3
    assert result["event_response_sum"] == result["reference_response_sum"]
    assert result["overlapping_consecutive_event_windows"] == 2
    assert result["boundary_excluded_events"] == 2


@pytest.mark.parametrize("missing", [2, 3])
def test_missing_successor_remains_unknown_in_eligible_cohort(missing):
    secs = [t for t in range(7) if t != missing]
    quotes = [100. if t == 0 else 200. + t for t in secs]
    result = collect(quotes, secs, start=1, end=7)[0]
    assert result["event_eligible"] == result["event_unknown"] == 1
    assert result["event_known"] == 0 and result["event_response_sum"] == 0.
    assert result["reference_unknown"] >= 1
    assert result["reference_eligible"] == result["reference_known"] + result["reference_unknown"]


def test_missing_current_past_pair_does_not_create_event_or_reference_anchor():
    result = collect([100., 200., 201., 202.], [0, 2, 3, 4], start=2, end=5)[0]
    assert result["detector_pair_exclusions"] == 1
    assert result["detected_event_anchors"] == 0


def test_partition_start_may_use_same_day_preceding_quote():
    result = collect([100., 200., 201., 202., 203.], start=1, end=5)[0]
    assert result["detector_pair_exclusions"] == 0
    assert result["event_eligible"] == 1


def test_forward_endpoint_is_strictly_inside_partition():
    result = collect([100., 200., 400., 800., 1600., 3200.], start=1, end=4)[0]
    assert result["reference_eligible"] == 1
    assert result["boundary_excluded_anchors"] == result["boundary_excluded_events"] == 2


def test_midnight_pair_is_excluded_even_with_contiguous_public_quotes():
    data = frame([100., 200., 400., 401., 402., 403., 404.], range(86398, 86405))
    result = collect_successor_days(data, FixedTailDetector(1, .01), BASE + 86398, BASE + 86405)
    assert len(result) == 2
    assert result[0]["reference_eligible"] == 0
    assert result[0]["boundary_excluded_events"] == 1
    assert result[1]["detector_pair_exclusions"] == 1
    assert result[1]["detected_event_anchors"] == 0


def test_future_changes_never_change_anchor_event_membership():
    a = collect([100., 200., 201., 202., 203.], start=1, end=2)[0]
    b = collect([100., 200., 1e20, 1e30, 1e40], start=1, end=2)[0]
    assert a == b
    assert a["detected_event_anchors"] == 1


def test_crash_native_sign_is_used_for_event_and_response():
    result = collect([200., 100., 99., 98., 97.], side=-1)[0]
    assert result["event_eligible"] == 1
    assert result["event_response_sum"] == -_log_return(99., 98.)


def test_two_second_event_windows_count_overlap_but_three_seconds_do_not():
    result = collect([100., 200., 201., 402., 403., 404., 808., 809., 810.])[0]
    assert result["event_eligible"] == 3
    assert result["overlapping_consecutive_event_windows"] == 1


def test_day_matched_reference_uses_event_exposure_not_calendar_or_reference_count():
    days = [day(events=100, refs=1000, event_sum=2., reference_sum=10.),
            day("2026-01-02", events=300, refs=10000, event_sum=12., reference_sum=300.)]
    value = summarize_successor(days)
    assert value["event_mean_known_subset"] == pytest.approx(14. / 400)
    assert value["day_matched_reference_mean_known_subset"] == pytest.approx((100 * .01 + 300 * .03) / 400)
    assert value["excess_mean_known_subset"] == pytest.approx(.01)
    assert value["observed_day_clusters"] == 2


def test_unknown_event_response_does_not_change_reference_exposure_weights():
    days = [day(events=100, known=50, refs=1000, unknown_refs=50, event_sum=1., reference_sum=9.5),
            day("2026-01-02", events=300, refs=10000, event_sum=12., reference_sum=300.)]
    value = summarize_successor(days)
    assert value["day_matched_reference_mean_known_subset"] == pytest.approx(.025)
    assert value["complete_responses"] is False
    assert value["conditional_positive_excess_rejected"] is False


def test_observed_zero_event_day_retained_and_undefined_draws_reported():
    days = [day(), day("2026-01-02", events=0, event_sum=0.)]
    value = summarize_successor(days)
    assert value["observed_day_clusters"] == 2
    assert value["bootstrap"]["excess_mean"]["undefined_draws"] > 0
    assert value["bootstrap"]["excess_mean"]["finite_draws"] + value["bootstrap"]["excess_mean"]["undefined_draws"] == 9999
    assert value["conditional_positive_excess_rejected"] is False


def test_zero_event_cohort_has_null_point_and_all_undefined_draws():
    value = summarize_successor([day(events=0, event_sum=0.)])
    assert value["event_mean_known_subset"] is None and value["excess_mean_known_subset"] is None
    assert value["bootstrap"]["excess_mean"] == {"finite_draws": 0, "undefined_draws": 9999, "conditional_ci95": None}


def test_single_complete_negative_day_can_only_reject_the_specified_excess():
    value = summarize_successor([day()])
    assert value["conditional_positive_excess_rejected"] is True
    assert value["historical_strategy_candidate"] is False and value["strategy_eligible"] is False
    assert value["profit_factor"] is None and value["prospective_paper"] == "NOT TESTED"


@pytest.mark.parametrize("change", ["small", "unknown", "repeats", "seed"])
def test_rejection_requires_all_fixed_gates(change):
    d = day(); options = {}
    if change == "small": d = day(events=99)
    if change == "unknown": d = day(unknown_refs=1)
    if change == "repeats": options["repeats"] = 10
    if change == "seed": options["seed"] = 1
    assert summarize_successor([d], **options)["conditional_positive_excess_rejected"] is False


def test_positive_excess_never_promotes_strategy():
    value = summarize_successor([day(event_sum=1.)])
    assert value["excess_mean_known_subset"] > 0
    assert value["conditional_positive_excess_rejected"] is False
    assert value["historical_strategy_candidate"] is False and value["profit_factor"] is None


def test_unobserved_zero_day_cannot_be_padded_into_bootstrap():
    d = day(events=0, event_sum=0., refs=0)
    with pytest.raises(ValueError, match="observed"):
        summarize_successor([d])


def test_shared_resampling_preserves_exact_equal_event_and_reference_mean():
    days = [day(event_sum=1., reference_sum=10.), day("2026-01-02", event_sum=3., reference_sum=30.)]
    value = summarize_successor(days)
    assert value["bootstrap"]["excess_mean"]["conditional_ci95"] == pytest.approx([0., 0.], abs=1e-17)


@pytest.mark.parametrize("field", COUNT_FIELDS)
def test_negative_or_boolean_counts_rejected(field):
    d = day(); d[field] = True
    with pytest.raises(ValueError): summarize_successor([d])


@pytest.mark.parametrize("mutation", ["duplicate", "unsorted", "schema", "identity", "nonfinite"])
def test_invalid_daily_statistics_are_rejected(mutation):
    days = [day(), day("2026-01-02")]
    if mutation == "duplicate": days[1]["date"] = days[0]["date"]
    if mutation == "unsorted": days.reverse()
    if mutation == "schema": days[0]["extra"] = 1
    if mutation == "identity": days[0]["reference_known"] -= 1
    if mutation == "nonfinite": days[0]["event_response_sum"] = float("nan")
    with pytest.raises(ValueError): summarize_successor(days)


@pytest.mark.parametrize("args", [(True, BASE + 10), (BASE, BASE), (BASE, BASE - 1)])
def test_invalid_partition_bounds_rejected(args):
    with pytest.raises(ValueError): collect_successor_days(frame([100., 101.]), FixedTailDetector(1, .01), *args)
