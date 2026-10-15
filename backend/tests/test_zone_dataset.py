"""Synthetic streaming chronology, complete-population and missing-data tests."""

import numpy as np
import pandas as pd
import pytest

from app.research.zone_dataset import build_zone_dataset


START = pd.Timestamp("2026-01-01 23:58", tz="UTC")
END = START + pd.Timedelta(minutes=6)


def ticks(start=START, periods=360):
    index = pd.date_range(start, periods=periods, freq="s")
    step = np.arange(periods, dtype=float)
    return pd.DataFrame({"quote": 100 + step / 1000 + np.sin(step / 3)}, index=index)


def build(chunks, start=START, end=END):
    return build_zone_dataset(chunks, start=start, end=end)


def assert_equal(actual, expected):
    pd.testing.assert_frame_equal(actual.m1, expected.m1)
    pd.testing.assert_frame_equal(actual.coverage, expected.coverage)
    assert actual.summary == expected.summary


@pytest.mark.parametrize("boundaries", [
    [1, 7, 59, 60, 61, 119, 120, 179, 359],
    [30, 31, 33, 48, 52, 58, 59, 60, 121, 181, 237, 300],
    [60, 120, 180, 240, 300],
    list(range(1, 360)),
])
def test_arbitrary_chunk_boundaries_match_batch_exactly(boundaries):
    source = ticks()
    offsets = [0, *boundaries, len(source)]
    chunks = (source.iloc[left:right] for left, right in zip(offsets[:-1], offsets[1:]))
    actual = build(chunks)
    expected = build([source])
    assert_equal(actual, expected)
    # Independent direct observed-population OHLC, with no builder internals.
    for stamp in expected.m1.index:
        population = source.loc[(source.index >= stamp) &
                                (source.index < stamp + pd.Timedelta(minutes=1)), "quote"]
        np.testing.assert_array_equal(expected.m1.loc[stamp].to_numpy(),
                                      [population.iloc[0], population.max(),
                                       population.min(), population.iloc[-1]])
    assert actual.coverage.minute_valid.all()
    assert actual.coverage.observed_seconds.eq(60).all()
    assert actual.coverage.run_id.eq(0).all()
    assert actual.summary["known_consecutive_increments"] == 359


def test_continuous_midnight_does_not_reset_even_when_source_chunks_are_days():
    source = ticks()
    midnight = pd.Timestamp("2026-01-02", tz="UTC")
    actual = build([source.loc[source.index < midnight], source.loc[source.index >= midnight]])
    assert actual.summary["runs"] == 1
    assert actual.coverage.run_id.eq(0).all()
    assert_equal(actual, build([source]))


@pytest.mark.parametrize("missing", [119, 120, 121])
def test_actual_second_gap_around_midnight_resets_only_its_run(missing):
    source = ticks().drop(ticks().index[missing])
    actual = build([source.iloc[:119], source.iloc[119:121], source.iloc[121:]])
    assert_equal(actual, build([source]))
    assert actual.summary["runs"] == 2
    assert actual.summary["missing_seconds"] == 1
    assert actual.summary["known_consecutive_increments"] == 357
    affected = (START + pd.Timedelta(seconds=missing)).floor("min")
    assert actual.coverage.loc[affected, "observed_seconds"] == 59
    assert not actual.coverage.loc[affected, "minute_valid"]
    assert actual.m1.loc[affected].isna().all()
    assert actual.coverage.iloc[0].run_id == 0
    assert actual.coverage.iloc[-1].run_id == 1
    if missing == 121:
        assert pd.isna(actual.coverage.loc[affected, "run_id"])


def test_interior_gap_within_a_pending_minute_preserves_mixed_run_unknown():
    source = ticks(periods=180).drop(ticks(periods=180).index[30])
    actual = build([source.iloc[:15], source.iloc[15:40], source.iloc[40:]],
                   end=START + pd.Timedelta(minutes=3))
    assert actual.coverage.observed_seconds.tolist() == [59, 60, 60]
    assert actual.coverage.minute_valid.tolist() == [False, True, True]
    assert pd.isna(actual.coverage.run_id.iloc[0])
    assert actual.coverage.run_id.iloc[1:].eq(1).all()
    assert actual.m1.iloc[0].isna().all()


def test_full_grid_retains_missing_leading_day_middle_day_and_trailing_day():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = start + pd.Timedelta(days=5)
    first = ticks(start + pd.Timedelta(days=1), periods=60)
    second = ticks(start + pd.Timedelta(days=3), periods=60)
    actual = build([first, second], start, end)
    assert len(actual.m1) == 5 * 1440
    assert actual.m1.index[0] == start
    assert actual.m1.index[-1] == end - pd.Timedelta(minutes=1)
    assert actual.coverage.minute_valid.sum() == 2
    assert actual.coverage.observed_seconds.sum() == 120
    assert actual.summary["runs"] == 2
    empty = actual.coverage.observed_seconds.eq(0)
    assert actual.m1.loc[empty].isna().all().all()
    assert actual.coverage.loc[empty, "run_id"].isna().all()
    assert actual.coverage.loc[first.index[0], "run_id"] == 0
    assert actual.coverage.loc[second.index[0], "run_id"] == 1


@pytest.mark.parametrize("chunks", [[], [pd.DataFrame({"quote": []}, index=pd.DatetimeIndex([], tz="UTC"))]])
def test_entire_empty_cohort_preserves_full_unknown_grid(chunks):
    actual = build(chunks)
    assert len(actual.m1) == 6
    assert actual.m1.isna().all().all()
    assert actual.coverage.observed_seconds.eq(0).all()
    assert not actual.coverage.minute_valid.any()
    assert actual.coverage.run_id.isna().all()
    assert str(actual.coverage.run_id.dtype) == "Int64"
    assert actual.summary["runs"] == actual.summary["observed_seconds"] == 0
    assert actual.summary["missing_seconds"] == 360
    assert actual.summary["first_observed"] is None
    assert actual.summary["last_observed"] is None


def test_missing_final_minute_seconds_are_not_invented():
    source = ticks(periods=140)
    actual = build([source.iloc[:125], source.iloc[125:]])
    assert actual.coverage.observed_seconds.tolist() == [60, 60, 20, 0, 0, 0]
    assert actual.coverage.minute_valid.tolist() == [True, True, False, False, False, False]
    assert actual.coverage.run_id.iloc[2] == 0
    assert actual.m1.iloc[2:].isna().all().all()


def test_future_and_past_poison_values_are_excluded_before_value_validation():
    source = ticks(START - pd.Timedelta(minutes=1), periods=480)
    clean = build([source])
    poisoned = source.astype(object)
    outside = (poisoned.index < START) | (poisoned.index >= END)
    poison = [None, "unreadable", float("nan"), -1, 0, True, complex(1, 2), object()]
    poisoned.loc[outside, "quote"] = [poison[i % len(poison)] for i in range(int(outside.sum()))]
    assert_equal(build([poisoned]), clean)
    assert_equal(build([poisoned.iloc[:60], poisoned.iloc[60:430], poisoned.iloc[430:]]), clean)
    assert clean.summary["observed_seconds"] == 360
    assert clean.summary["first_observed"] == START.isoformat()
    assert clean.summary["last_observed"] == (END - pd.Timedelta(seconds=1)).isoformat()


def test_only_outside_poisoned_rows_produce_an_empty_cohort():
    source = pd.DataFrame({"quote": [object(), -1]},
                          index=pd.DatetimeIndex([START - pd.Timedelta(seconds=1), END]))
    assert_equal(build([source]), build([]))


@pytest.mark.parametrize("value", [None, "bad", float("nan"), float("inf"), -1, 0, True, 1j])
def test_inside_poison_values_are_refused(value):
    source = ticks(periods=1).astype(object)
    source.iloc[0, 0] = value
    with pytest.raises((ValueError, TypeError)):
        build([source])


def test_chronology_outside_planned_bounds_is_still_required():
    future = ticks(END, periods=3)
    with pytest.raises(ValueError, match="sorted and unique"):
        build([future.iloc[::-1]])
    with pytest.raises(ValueError, match="without overlap"):
        build([future, future.iloc[:1]])
    with pytest.raises(ValueError, match="without overlap"):
        build([future, ticks()])
    past = ticks(START - pd.Timedelta(days=1), periods=3)
    with pytest.raises(ValueError, match="without overlap"):
        build([past, past.iloc[-1:]])


@pytest.mark.parametrize("kind", ["duplicate", "unordered", "naive", "non_utc", "subsecond", "nat"])
def test_bad_source_timestamp_schema_refused(kind):
    source = ticks(periods=3)
    if kind == "duplicate":
        source.index = pd.DatetimeIndex([START, START, START + pd.Timedelta(seconds=1)])
    elif kind == "unordered":
        source = source.iloc[[1, 0, 2]]
    elif kind == "naive":
        source.index = source.index.tz_localize(None)
    elif kind == "non_utc":
        source.index = source.index.tz_convert("Asia/Riyadh")
    elif kind == "subsecond":
        source.index = source.index + pd.Timedelta(milliseconds=1)
    elif kind == "nat":
        source.index = pd.DatetimeIndex([START, pd.NaT, START + pd.Timedelta(seconds=2)])
    with pytest.raises(ValueError):
        build([source])


@pytest.mark.parametrize("kwargs", [
    {"start": START.tz_localize(None)},
    {"end": END.tz_convert("Asia/Riyadh")},
    {"start": START + pd.Timedelta(seconds=1)},
    {"end": END + pd.Timedelta(milliseconds=1)},
    {"start": END},
    {"start": END + pd.Timedelta(minutes=1)},
    {"end": pd.NaT},
    {"start": "2026-01-01 23:58:00+00:00"},
])
def test_bad_bounds_refused(kwargs):
    bounds = {"start": START, "end": END, **kwargs}
    with pytest.raises(ValueError):
        build_zone_dataset([], **bounds)


def test_empty_chunks_and_index_names_do_not_change_population():
    source = ticks()
    expected = build([source])
    source.index.name = "source_timestamp"
    empty = source.iloc[:0]
    assert_equal(build([empty, source.iloc[:73], empty, source.iloc[73:], empty]), expected)


def test_source_input_remains_unmodified():
    source = ticks()
    original = source.copy(deep=True)
    build([source])
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("source", [
    [1, 2],
    pd.DataFrame({"close": [100.]}, index=pd.DatetimeIndex([START])),
    pd.DataFrame([[100., 101.]], columns=["quote", "quote"], index=pd.DatetimeIndex([START])),
])
def test_invalid_chunk_or_quote_schema_refused(source):
    with pytest.raises((TypeError, ValueError)):
        build([source])
