"""Synthetic causal/coverage checks only; no historical prices or model fit."""

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.research.jump_representation import (
    JumpRepresentationState,
    RepresentationArithmeticError,
    complete_m1_bars,
    jump_representation,
)
from app.research.tick_tail import FixedTailDetector, _log_return


def ticks(prices, seconds=None, start="2026-01-01T00:00:00Z"):
    seconds = range(len(prices)) if seconds is None else seconds
    return pd.DataFrame({"quote": prices}, index=pd.DatetimeIndex(
        [pd.Timestamp(start) + pd.Timedelta(seconds=int(value)) for value in seconds]
    ).as_unit("ns"))


def detector(side=1, scale=0.001):
    return FixedTailDetector(side, scale)


@pytest.mark.parametrize("side", (1, -1))
def test_native_tail_is_removed_but_ordinary_signed_returns_are_kept(side):
    prices = [100.0]
    for change in (-side * 0.001, side * 0.05, -side * 0.002, side * 0.001):
        prices.append(prices[-1] * math.exp(change))
    result = JumpRepresentationState(detector(side)).consume(ticks(prices))
    assert result.tail_removed.iloc[1:].tolist() == [False, True, False, False]
    assert result.transformed_quote.iloc[0] == 1.0
    assert result.transformed_quote.iloc[2] == result.transformed_quote.iloc[1]
    for i in (1, 3, 4):
        expected = _log_return(prices[i-1], prices[i])
        assert result.retained_log_return.iloc[i] == expected
        assert math.log(result.transformed_quote.iloc[i] / result.transformed_quote.iloc[i-1]) == pytest.approx(expected)
    assert result.log_transformed_quote.iloc[-1] == pytest.approx(-side * 0.002)
    assert pd.isna(result.tail_removed.iloc[0])
    assert not result.increment_known.iloc[0]
    assert math.isnan(result.retained_log_return.iloc[0])


@pytest.mark.parametrize("side,prices", ((1, [1.0, 2.0]), (-1, [2.0, 1.0])))
def test_exact_threshold_is_kept_and_strictly_greater_is_removed(side, prices):
    directed = side * _log_return(*prices)
    equal = FixedTailDetector(side, directed / 10.0)
    assert equal.threshold == directed
    kept = JumpRepresentationState(equal).consume(ticks(prices))
    assert not kept.tail_removed.iloc[1]
    lower = FixedTailDetector(side, math.nextafter(directed / 10.0, 0.0))
    assert lower.threshold < directed
    removed = JumpRepresentationState(lower).consume(ticks(prices))
    assert removed.tail_removed.iloc[1]
    assert removed.transformed_quote.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("side,prices", ((1, [1.0, math.exp(-0.5)]), (-1, [1.0, math.exp(0.5)])))
def test_large_opposite_side_increment_is_not_removed(side, prices):
    result = JumpRepresentationState(detector(side)).consume(ticks(prices))
    assert not result.tail_removed.iloc[1]
    assert result.transformed_quote.iloc[1] == pytest.approx(prices[1])


def test_gap_resets_anchor_and_retains_unknown_increment():
    result = JumpRepresentationState(detector()).consume(ticks([100.0, 99.0, 800.0, 792.0], [0, 1, 5, 6]))
    assert result.run_id.tolist() == [0, 0, 1, 1]
    assert result.increment_known.tolist() == [False, True, False, True]
    assert result.transformed_quote.tolist() == pytest.approx([1.0, 0.99, 1.0, 0.99])
    assert pd.isna(result.tail_removed.iloc[2])
    assert math.isnan(result.log_return.iloc[2])


@pytest.mark.parametrize("size", (1, 2, 7, 31, 60, 100))
def test_chunk_boundaries_do_not_change_chain_or_minute_coverage(size):
    seconds = np.delete(np.arange(240), [65, 150])
    prices = 100.0 * np.exp(-0.0001 * seconds + 0.03 * (seconds >= 30))
    source = ticks(prices, seconds)
    batch, expected_bars = jump_representation(source, detector())
    state = JumpRepresentationState(detector())
    chunks = [state.consume(source.iloc[i:i+size]) for i in range(0, len(source), size)]
    combined = pd.concat(chunks)
    assert_frame_equal(combined, batch, check_exact=True)
    assert state.snapshot == _snapshot_after(source, detector())
    actual_bars = complete_m1_bars(combined)
    assert_frame_equal(actual_bars.raw, expected_bars.raw, check_exact=True)
    assert_frame_equal(actual_bars.transformed, expected_bars.transformed, check_exact=True)
    assert_frame_equal(actual_bars.coverage, expected_bars.coverage, check_exact=True)


def _snapshot_after(source, supplied_detector):
    state = JumpRepresentationState(supplied_detector)
    state.consume(source)
    return state.snapshot


def test_future_perturbation_does_not_change_past_quotes_or_completed_bars():
    source = ticks(100.0 * np.exp(-np.arange(240) * 0.0001))
    changed = source.copy(deep=True)
    changed.loc[changed.index[120]:, "quote"] *= 50.0
    original, original_bars = jump_representation(source, detector())
    perturbed, perturbed_bars = jump_representation(changed, detector())
    assert_frame_equal(original.iloc[:120], perturbed.iloc[:120], check_exact=True)
    assert_frame_equal(original_bars.raw.iloc[:2], perturbed_bars.raw.iloc[:2], check_exact=True)
    assert_frame_equal(original_bars.transformed.iloc[:2], perturbed_bars.transformed.iloc[:2], check_exact=True)
    assert_frame_equal(original_bars.coverage.iloc[:2], perturbed_bars.coverage.iloc[:2], check_exact=True)


def test_partial_empty_and_mixed_run_minutes_remain_entirely_unknown():
    seconds = [*range(5, 60), *range(60, 89), *range(90, 120), *range(180, 240), 240]
    source = ticks([100.0 + i * 0.0001 for i in range(len(seconds))], seconds)
    description, bars = jump_representation(source, detector())
    assert bars.coverage.observed_seconds.tolist() == [55, 59, 0, 60, 1]
    assert bars.coverage.minute_valid.tolist() == [False, False, False, True, False]
    assert bars.raw.iloc[[0, 1, 2, 4]].isna().all().all()
    assert bars.transformed.iloc[[0, 1, 2, 4]].isna().all().all()
    assert pd.isna(bars.coverage.run_id.iloc[1])
    assert pd.isna(bars.coverage.run_id.iloc[2])
    selected = description.loc[pd.Timestamp("2026-01-01T00:03:00Z"):pd.Timestamp("2026-01-01T00:03:59Z")]
    for field, name in (("quote", "raw"), ("transformed_quote", "transformed")):
        row = getattr(bars, name).iloc[3]
        expected = [selected[field].iloc[0], selected[field].max(), selected[field].min(), selected[field].iloc[-1]]
        assert row.tolist() == expected


def test_midnight_continuity_is_not_a_gap():
    source = ticks([100.0] * 120, start="2026-01-01T23:59:00Z")
    description, bars = jump_representation(source, detector())
    assert description.run_id.eq(0).all()
    assert bars.coverage.minute_valid.tolist() == [True, True]
    assert bars.coverage.run_id.tolist() == [0, 0]
    assert description.increment_known.iloc[60]


def test_original_input_and_description_are_never_mutated():
    source = ticks([100.0 + i for i in range(120)])
    source["unrelated"] = "unchanged"
    before = source.copy(deep=True)
    description, bars = jump_representation(source, detector())
    assert_frame_equal(source, before)
    assert description.quote.tolist() == before.quote.tolist()
    saved = description.copy(deep=True)
    complete_m1_bars(description)
    assert_frame_equal(description, saved)
    bars.raw.iloc[0, 0] = 12345.0
    assert_frame_equal(source, before)
    assert_frame_equal(description, saved)


def test_price_scale_does_not_change_transformed_representation():
    source = ticks([1.0, 0.5, 1.0, 2.0, 1.0, 0.5])
    multiplied = source * 8.0
    a = JumpRepresentationState(detector()).consume(source)
    b = JumpRepresentationState(detector()).consume(multiplied)
    assert_frame_equal(a.drop(columns="quote"), b.drop(columns="quote"), check_exact=True)


def test_neumaier_accumulation_preserves_small_residual(monkeypatch):
    # Exercise cancellation independently of the mathematical price/return
    # relationship, without fabricating a market sample.
    changes = iter([700.0, 1e-14, -700.0])
    monkeypatch.setattr("app.research.jump_representation._log_return", lambda previous, current: next(changes))
    result = JumpRepresentationState(detector(scale=100.0)).consume(ticks([1.0] * 4))
    assert result.log_transformed_quote.iloc[-1] == math.fsum([700.0, 1e-14, -700.0])


@pytest.mark.parametrize("prices,side", (([1e-320, 1e308], -1), ([1e308, 1e-320], 1)))
def test_exponentiation_failure_is_explicit_and_does_not_commit(prices, side):
    state = JumpRepresentationState(detector(side))
    before = state.snapshot
    with pytest.raises(RepresentationArithmeticError):
        state.consume(ticks(prices))
    assert state.snapshot == before


@pytest.mark.parametrize("value", (None, {}, 1, True))
def test_detector_is_explicit_and_typed(value):
    with pytest.raises(TypeError):
        JumpRepresentationState(value)


def test_snapshot_is_immutable_and_read_only():
    state = JumpRepresentationState(detector())
    with pytest.raises(FrozenInstanceError):
        state.snapshot.run_id = 1
    with pytest.raises(AttributeError):
        state.detector = detector(-1)


@pytest.mark.parametrize("prices", ([0.0], [-1.0], [math.inf], [math.nan], [True], ["1.0"], [1+0j]))
def test_invalid_quotes_are_refused_before_state_change(prices):
    state = JumpRepresentationState(detector())
    before = state.snapshot
    with pytest.raises((TypeError, ValueError)):
        state.consume(ticks(prices))
    assert state.snapshot == before


@pytest.mark.parametrize("kind", ("duplicate", "reverse", "subsecond", "naive", "nonutc", "nat", "empty", "duplicate_column", "missing_quote"))
def test_invalid_source_schema_or_clock_is_refused(kind):
    source = ticks([100.0, 99.0])
    if kind == "duplicate":
        source.index = pd.DatetimeIndex([source.index[0], source.index[0]])
    elif kind == "reverse":
        source = source.iloc[::-1]
    elif kind == "subsecond":
        source.index += pd.Timedelta(milliseconds=1)
    elif kind == "naive":
        source.index = source.index.tz_localize(None)
    elif kind == "nonutc":
        source.index = source.index.tz_convert("Asia/Riyadh")
    elif kind == "nat":
        source.index = pd.DatetimeIndex([source.index[0], pd.NaT])
    elif kind == "empty":
        source = source.iloc[:0]
    elif kind == "duplicate_column":
        source = pd.concat([source, source], axis=1)
    elif kind == "missing_quote":
        source = source.rename(columns={"quote": "price"})
    with pytest.raises((TypeError, ValueError)):
        JumpRepresentationState(detector()).consume(source)


@pytest.mark.parametrize("seconds", ([0], [-1], [1, 2]))
def test_overlapping_or_earlier_chunks_are_refused_atomically(seconds):
    state = JumpRepresentationState(detector())
    state.consume(ticks([100.0, 99.0], [0, 1]))
    before = state.snapshot
    with pytest.raises(ValueError, match="strictly later"):
        state.consume(ticks([100.0] * len(seconds), seconds))
    assert state.snapshot == before


@pytest.mark.parametrize("kind", ("missing_transformed", "missing_run", "negative_run", "bool_run", "text_run", "jump_run", "bridged_gap", "bad_transformed", "huge_run"))
def test_invalid_minute_description_is_refused(kind):
    source = ticks([100.0, 99.0, 98.0], [0, 1, 3])
    result = JumpRepresentationState(detector()).consume(source)
    if kind == "missing_transformed":
        result = result.drop(columns="transformed_quote")
    elif kind == "missing_run":
        result = result.drop(columns="run_id")
    elif kind == "negative_run":
        result["run_id"] = -1
    elif kind == "bool_run":
        result["run_id"] = True
    elif kind == "text_run":
        result["run_id"] = "0"
    elif kind == "jump_run":
        result["run_id"] = [0, 1, 2]
    elif kind == "bridged_gap":
        result["run_id"] = 0
    elif kind == "bad_transformed":
        result["transformed_quote"] = math.nan
    elif kind == "huge_run":
        result["run_id"] = [2**80] * 3
    with pytest.raises((TypeError, ValueError)):
        complete_m1_bars(result)


def test_bad_input_after_valid_chunk_does_not_destroy_existing_state():
    state = JumpRepresentationState(detector())
    state.consume(ticks([100.0, 99.0]))
    before = state.snapshot
    bad = ticks([98.0, math.nan], [2, 3])
    with pytest.raises(ValueError):
        state.consume(bad)
    assert state.snapshot == before
    continued = state.consume(ticks([98.0], [2]))
    assert continued.run_id.iloc[0] == 0
    assert continued.transformed_quote.iloc[0] == pytest.approx(0.98)


def test_minute_run_identity_never_rounds_via_missing_minute_float_conversion():
    source = ticks([100.0] * 120, [*range(60), *range(120, 180)])
    description = JumpRepresentationState(detector()).consume(source)
    first = 2**53 + 1
    description["run_id"] += first
    bars = complete_m1_bars(description)
    assert bars.coverage.run_id.iloc[0] == first
    assert pd.isna(bars.coverage.run_id.iloc[1])
    assert bars.coverage.run_id.iloc[2] == first + 1
