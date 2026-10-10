"""Synthetic native-second closed-frame pairing and gap-reset checks."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from app.research.jump_multiframe import _stable_bb_width, causal_jump_multiframe_inputs
from app.research.jump_representation import PairedMinuteBars, jump_representation
from app.research.multiframe_signal import FEATURE_NAMES, causal_multiframe_inputs
from app.research.tick_tail import FixedTailDetector


BB_DEPENDENT_COLUMNS = ("bb_width_over_prior_median_100", "bb_squeeze", "feature_valid")


def _assert_non_bb_exact(actual, expected):
    pd.testing.assert_frame_equal(actual.drop(columns=list(BB_DEPENDENT_COLUMNS)),
                                  expected.drop(columns=list(BB_DEPENDENT_COLUMNS)))


def _decimal_bb(frame, position):
    """Independent50-digit arithmetic directly on original floating closes."""
    with localcontext() as context:
        context.prec = 50
        values = [Decimal.from_float(float(value))
                  for value in frame.close.iloc[position - 119:position + 1]]
        assert len(values) == 120 and all(value.is_finite() and value > 0 for value in values)
        widths = []
        for end in range(19, 120):
            window = values[end - 19:end + 1]
            mean = sum(window) / Decimal(20)
            variance = sum((value - mean) ** 2 for value in window) / Decimal(20)
            widths.append(Decimal(4) * variance.sqrt() / mean)
        current, previous = widths[-1], sorted(widths[:-1])
        median = (previous[49] + previous[50]) / Decimal(2)
        q35 = previous[34] * Decimal("0.35") + previous[35] * Decimal("0.65")
        return float(current / median), current <= q35


def _assert_selected_decimal_bb(frame):
    valid = np.flatnonzero(frame.feature_valid.to_numpy(bool))
    assert len(valid) >= 3
    for position in (valid[0], valid[len(valid) // 2], valid[-1]):
        expected, squeeze = _decimal_bb(frame, position)
        assert frame.bb_width_over_prior_median_100.iloc[position] == pytest.approx(expected, abs=1e-11, rel=0)
        assert bool(frame.bb_squeeze.iloc[position]) == squeeze


@pytest.fixture(scope="module")
def synthetic_ticks():
    # Six continuous days supply the unchanged H4 ATR/structure warmup. Single
    # native tails are removed; ordinary120-second volatility bursts stay below
    # the fixed detector threshold and make transformed large-bar age observable.
    n = 6 * 86400
    phase = np.arange(n, dtype=float)
    increments = 1e-6 * (0.7 * np.sin(phase / 41) + 0.4 * np.cos(phase / 83)) + 2e-7
    burst = np.arange(n) % 21600
    increments[(burst >= 1200) & (burst < 1320)] += 4e-5
    increments[5000::10000] += 1e-3
    increments[0] = 0
    quotes = 1000 * np.exp(np.cumsum(increments))
    index = pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC")
    return pd.DataFrame({"quote": quotes}, index=index)


def _direction_ticks(ticks, direction):
    if direction == "boom":
        return ticks.copy()
    return pd.DataFrame({"quote": 1_000_000 / ticks.quote}, index=ticks.index.copy())


def _detector(direction):
    return FixedTailDetector(1 if direction == "boom" else -1, 1e-5)


@pytest.fixture(scope="module")
def source_bars(synthetic_ticks):
    return {direction: jump_representation(_direction_ticks(synthetic_ticks, direction),
                                          _detector(direction))[1]
            for direction in ("boom", "crash")}


@pytest.fixture(scope="module")
def paired(source_bars):
    return {direction: causal_jump_multiframe_inputs(source_bars[direction], direction)
            for direction in ("boom", "crash")}


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_same44_closed_features_same_exact_populations_and_separate_atrs(source_bars, paired, direction):
    bars, result = source_bars[direction], paired[direction]
    assert len(result.feature_names) == 44 and result.feature_names == FEATURE_NAMES
    expected_index = pd.date_range("2026-01-01", "2026-01-06 23:55", freq="5min", tz="UTC").as_unit("ns")
    pd.testing.assert_index_equal(result.raw.index, expected_index)
    pd.testing.assert_index_equal(result.transformed.index, expected_index)
    assert bars.coverage.minute_valid.all() and bars.coverage.observed_seconds.eq(60).all()
    assert result.availability.common_feature_valid.any()
    for side, minute in ((result.raw, bars.raw), (result.transformed, bars.transformed)):
        expected, names = causal_multiframe_inputs(minute, direction)
        assert tuple(names) == result.feature_names
        _assert_non_bb_exact(side.drop(columns="representation_run_id"), expected)
        assert side.loc[side.feature_valid, names].notna().all().all()
        _assert_selected_decimal_bb(side)
    pd.testing.assert_series_equal(result.availability.raw_execution_atr, result.raw.atr,
                                   check_names=False)
    pd.testing.assert_series_equal(result.availability.transformed_feature_atr,
                                   result.transformed.atr, check_names=False)
    assert not np.allclose(result.availability.raw_execution_atr.dropna(),
                           result.availability.transformed_feature_atr.dropna())
    expected = result.raw.feature_valid & result.transformed.feature_valid
    pd.testing.assert_series_equal(result.availability.common_feature_valid, expected,
                                   check_names=False)
    assert not result.availability.common_feature_valid.iloc[:12 * 12].any()


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_tick_population_ohlc_uses_same_exact60_seconds(synthetic_ticks, source_bars, direction):
    ticks, bars = _direction_ticks(synthetic_ticks, direction), source_bars[direction]
    description, _ = jump_representation(ticks.iloc[:360], _detector(direction))
    for stamp in bars.raw.index[:5]:
        population = description.loc[(description.index >= stamp) &
                                     (description.index < stamp + pd.Timedelta(minutes=1))]
        assert len(population) == 60
        for quote, actual in ((population.quote, bars.raw.loc[stamp]),
                              (population.transformed_quote, bars.transformed.loc[stamp])):
            np.testing.assert_array_equal(actual.to_numpy(),
                                          [quote.iloc[0], quote.max(), quote.min(), quote.iloc[-1]])


@pytest.mark.parametrize("direction", ["boom", "crash"])
@pytest.mark.parametrize("issue_text", ["2026-01-04 12:00", "2026-01-04 12:05"])
def test_at_or_after_issue_tick_perturbation_leaves_every_prior44_unchanged(
        synthetic_ticks, paired, direction, issue_text):
    issue = pd.Timestamp(issue_text, tz="UTC")
    ticks = _direction_ticks(synthetic_ticks, direction)
    ticks.loc[ticks.index >= issue, "quote"] *= 1.7
    _, bars = jump_representation(ticks, _detector(direction))
    changed = causal_jump_multiframe_inputs(bars, direction)
    completed = paired[direction].raw.index + pd.Timedelta(minutes=5) <= issue
    for name in ("raw", "transformed", "availability"):
        pd.testing.assert_frame_equal(getattr(paired[direction], name).loc[completed],
                                      getattr(changed, name).loc[completed])


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_positive_multiplicative_scale_invariance_with_explicit_1e6_tolerance(
        synthetic_ticks, paired, direction):
    ticks = _direction_ticks(synthetic_ticks, direction)
    ticks.quote *= 17.0
    _, bars = jump_representation(ticks, _detector(direction))
    scaled = causal_jump_multiframe_inputs(bars, direction)
    original = paired[direction]
    pd.testing.assert_series_equal(original.availability.common_feature_valid,
                                   scaled.availability.common_feature_valid)
    # These are dimensionless features. The explicit absolute tolerance permits
    # only floating arithmetic differences, not a changed feature definition.
    for name in ("raw", "transformed"):
        left, right = getattr(original, name), getattr(scaled, name)
        valid = left.feature_valid & right.feature_valid
        assert valid.any()
        np.testing.assert_allclose(left.loc[valid, FEATURE_NAMES], right.loc[valid, FEATURE_NAMES],
                                   rtol=0, atol=1e-6)
    np.testing.assert_allclose(scaled.availability.raw_execution_atr,
                               original.availability.raw_execution_atr * 17,
                               rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("direction", ["boom", "crash"])
@pytest.mark.parametrize("gauge", [1e-9, 1e9])
def test_transformed_feature_gauge_invariance_directly_on_all44(
        source_bars, paired, direction, gauge):
    # Re-anchoring ticks always starts Q at1. Scale transformed OHLC directly
    # here to check the unchanged44 formulas across eighteen orders of gauge.
    # This finite six-day fixture does not certify arbitrary long-chain ranges.
    bars, original = source_bars[direction], paired[direction]
    scaled = causal_jump_multiframe_inputs(replace(bars, transformed=bars.transformed * gauge),
                                           direction)
    pd.testing.assert_series_equal(original.availability.common_feature_valid,
                                   scaled.availability.common_feature_valid)
    valid = original.transformed.feature_valid
    assert valid.any()
    left = original.transformed.loc[valid, FEATURE_NAMES].to_numpy()
    right = scaled.transformed.loc[valid, FEATURE_NAMES].to_numpy()
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-6)
    for frame in (original.raw, original.transformed, scaled.transformed):
        _assert_selected_decimal_bb(frame)
    pd.testing.assert_series_equal(original.availability.raw_execution_atr,
                                   scaled.availability.raw_execution_atr)
    np.testing.assert_allclose(scaled.availability.transformed_feature_atr,
                               original.availability.transformed_feature_atr * gauge,
                               rtol=1e-9, atol=1e-18, equal_nan=True)


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_180_day_transformed_drift_gauge_conditioning_on_all44(direction):
    # Pure synthetic M1 populations avoid creating15million unnecessary second
    # rows. The ordinary drift spans Q=exp(-26) or exp(+26), with observed small
    # oscillations and periodic volatility bursts keeping original age semantics.
    n = 180 * 1440
    phase = np.arange(n + 1, dtype=float)
    sign = -1 if direction == "boom" else 1
    log_price = sign * 26 * phase / n + 0.001 * np.sin(phase / 20)
    opening, closing = np.exp(log_price[:-1]), np.exp(log_price[1:])
    wick = np.full(n, 3e-5)
    burst = np.arange(n) % 1440
    wick[(burst >= 150) & (burst < 153)] = 0.004
    index = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC").as_unit("ns")
    transformed = pd.DataFrame({"open": opening,
                                "high": np.maximum(opening, closing) * np.exp(wick),
                                "low": np.minimum(opening, closing) * np.exp(-wick),
                                "close": closing}, index=index)
    coverage = pd.DataFrame({"observed_seconds": np.full(n, 60, dtype=np.int64),
                             "minute_valid": np.ones(n, dtype=bool),
                             "run_id": pd.array(np.zeros(n, dtype=np.int64), dtype="Int64")},
                            index=index)
    bars = PairedMinuteBars(transformed * 1000, transformed, coverage)
    original = causal_jump_multiframe_inputs(bars, direction)
    scaled = causal_jump_multiframe_inputs(replace(bars, transformed=transformed * 1e9), direction)
    pd.testing.assert_series_equal(original.transformed.feature_valid,
                                   scaled.transformed.feature_valid)
    valid = original.transformed.feature_valid
    assert valid.any()
    left = original.transformed.loc[valid, FEATURE_NAMES].to_numpy()
    right = scaled.transformed.loc[valid, FEATURE_NAMES].to_numpy()
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-6)
    print(f"180-day {direction} all44 max absolute gauge difference: {np.max(np.abs(left - right)):.17g}")
    for frame in (original.raw, original.transformed, scaled.transformed):
        _assert_selected_decimal_bb(frame)


@pytest.mark.parametrize("gap_text", ["2026-01-02 09:03:00", "2026-01-02 09:03:30"])
def test_missing_second_inside_m5_resets_every_run_without_fragment_aggregation(
        synthetic_ticks, gap_text):
    missing = pd.Timestamp(gap_text, tz="UTC")
    _, bars = jump_representation(synthetic_ticks.drop(missing), _detector("boom"))
    result = causal_jump_multiframe_inputs(bars, "boom")
    minute = missing.floor("min")
    assert not bars.coverage.loc[minute, "minute_valid"]
    assert bars.raw.loc[minute].isna().all() and bars.transformed.loc[minute].isna().all()
    row = missing.floor("5min")
    assert not result.availability.loc[row, "common_feature_valid"]
    for frame in (result.raw, result.transformed):
        assert frame.loc[row, ["open", "high", "low", "close"]].isna().all()
        assert pd.isna(frame.loc[row, "atr"])
        # A prior complete H1/H4 is forbidden after the new run starts. Neither
        # context lookup nor first TR can use a different chain's price anchor.
        assert not frame.loc[row, "h1_row_valid"]
        assert not frame.loc[row, "h4_row_valid"]
        assert pd.isna(frame.loc[row, "log1p_large_bar_age"])
        h1_open = pd.Timestamp("2026-01-02 09:55", tz="UTC")
        assert not frame.loc[h1_open, "h1_row_valid"]
        assert frame.loc[h1_open, "h1_closed_at"] == pd.Timestamp("2026-01-02 10:00", tz="UTC")
        assert pd.isna(frame.loc[h1_open, "h1_range_atr"])
        first_complete_h1 = pd.Timestamp("2026-01-02 10:55", tz="UTC")
        assert frame.loc[first_complete_h1, "h1_row_valid"]
        assert pd.isna(frame.loc[first_complete_h1, "h1_range_atr"])
    # Recompute the post-gap run in isolation. Its prices/context must exactly
    # match the corresponding source-run calculation, including all NaNs.
    run = bars.coverage.run_id.max()
    mask = bars.coverage.run_id.eq(run).fillna(False)
    for field, source in (("raw", bars.raw), ("transformed", bars.transformed)):
        isolated, _ = causal_multiframe_inputs(source.loc[mask], "boom")
        _assert_non_bb_exact(getattr(result, field).loc[isolated.index].drop(
            columns="representation_run_id"), isolated)
    assert (result.raw.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()
    assert result.availability.common_feature_valid.iloc[-100:].any()


def test_wholly_missing_m5_rows_remain_unknown_without_older_context(synthetic_ticks):
    start, end = pd.Timestamp("2026-01-02 09:00", tz="UTC"), pd.Timestamp("2026-01-02 10:00", tz="UTC")
    ticks = synthetic_ticks.loc[(synthetic_ticks.index < start) | (synthetic_ticks.index >= end)]
    _, bars = jump_representation(ticks, _detector("boom"))
    result = causal_jump_multiframe_inputs(bars, "boom")
    missing = (result.raw.index >= start) & (result.raw.index < end)
    assert missing.sum() == 12
    assert not result.availability.loc[missing, "common_feature_valid"].any()
    assert result.availability.loc[missing, "run_id"].isna().all()
    for frame in (result.raw, result.transformed):
        assert frame.loc[missing, FEATURE_NAMES].isna().all().all()
        assert not frame.loc[missing, "h4_row_valid"].any()
        assert frame.loc[missing, "h4_closed_at"].isna().all()


def test_large_bar_age_stays_unknown_and_all44_stay_present(source_bars):
    bars = source_bars["boom"]
    # A synthetic constant ordinary drift has no transformed large bar. It
    # cannot borrow raw events or replace unknown age with an arbitrary value.
    phase = np.arange(len(bars.raw), dtype=float)
    opening = np.exp(phase * 6e-5)
    close = opening * np.exp(5.9e-5)
    monotone = pd.DataFrame({"open": opening, "high": close, "low": opening,
                             "close": close}, index=bars.raw.index)
    result = causal_jump_multiframe_inputs(replace(bars, transformed=monotone), "boom")
    assert result.raw.feature_valid.any()
    assert len(result.feature_names) == 44
    assert result.transformed.log1p_large_bar_age.isna().all()
    assert not result.transformed.feature_valid.any()
    assert not result.availability.common_feature_valid.any()
    assert len(result.raw) == len(result.transformed) == 6 * 288


def test_mixed_run_minute_with_no_usable_population_retains_unknown_m5():
    index = pd.DatetimeIndex(["2026-01-01 00:00:00", "2026-01-01 00:00:02"], tz="UTC")
    _, bars = jump_representation(pd.DataFrame({"quote": [1000.0, 1001.0]}, index=index),
                                 _detector("boom"))
    result = causal_jump_multiframe_inputs(bars, "boom")
    assert len(result.raw) == 1
    assert result.raw.loc[:, FEATURE_NAMES].isna().all().all()
    assert result.transformed.loc[:, FEATURE_NAMES].isna().all().all()
    assert not result.availability.common_feature_valid.any()


def test_source_tables_are_not_mutated(source_bars):
    bars = source_bars["boom"]
    before = [frame.copy(deep=True) for frame in (bars.raw, bars.transformed, bars.coverage)]
    result = causal_jump_multiframe_inputs(bars, "boom")
    result.raw.iloc[-1, 0] = 12345
    result.availability.iloc[-1, 0] = False
    for original, snapshot in zip((bars.raw, bars.transformed, bars.coverage), before, strict=True):
        pd.testing.assert_frame_equal(original, snapshot)


def test_stable_bb_constant_and_missing_windows_preserve_zero_and_unknown_semantics():
    index = pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC")
    close = pd.Series(3.14, index=index)
    width = _stable_bb_width(close)
    assert width.iloc[:19].isna().all()
    assert width.iloc[19:].eq(0).all()
    close.iloc[25] = np.nan
    missing = _stable_bb_width(close)
    assert missing.iloc[25:45].isna().all()
    assert missing.iloc[45:].eq(0).all()


def test_repaired_validity_does_not_retain_old_bb_flag(source_bars, paired, monkeypatch):
    from app.research import jump_multiframe

    original = jump_multiframe.causal_multiframe_inputs

    def old_bb_unavailable(minute, direction):
        frame, names = original(minute, direction)
        frame["bb_width_over_prior_median_100"] = np.nan
        frame["bb_squeeze"] = False
        frame["feature_valid"] = False
        return frame, names

    monkeypatch.setattr(jump_multiframe, "causal_multiframe_inputs", old_bb_unavailable)
    repaired = causal_jump_multiframe_inputs(source_bars["boom"], "boom")
    assert repaired.availability.common_feature_valid.any()
    pd.testing.assert_frame_equal(repaired.raw, paired["boom"].raw)
    pd.testing.assert_frame_equal(repaired.transformed, paired["boom"].transformed)


@pytest.mark.parametrize("field", ["raw", "transformed", "coverage"])
def test_mismatched_populations_are_refused(source_bars, field):
    bars = source_bars["boom"]
    with pytest.raises(ValueError, match="populations|minute grid"):
        causal_jump_multiframe_inputs(replace(bars, **{field: getattr(bars, field).iloc[1:]}), "boom")


def test_partial_ohlc_and_noncomplete_validity_are_refused(source_bars):
    bars = source_bars["boom"]
    partial = bars.transformed.copy()
    partial.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="OHLC populations"):
        causal_jump_multiframe_inputs(replace(bars, transformed=partial), "boom")
    coverage = bars.coverage.copy()
    coverage.iloc[-1, coverage.columns.get_loc("observed_seconds")] = 59
    with pytest.raises(ValueError, match="Complete minutes"):
        causal_jump_multiframe_inputs(replace(bars, coverage=coverage), "boom")


def test_full_count_known_run_cannot_be_marked_invalid(source_bars):
    bars = source_bars["boom"]
    coverage = bars.coverage.copy()
    coverage.iloc[-1, coverage.columns.get_loc("minute_valid")] = False
    with pytest.raises(ValueError, match="Complete minutes"):
        causal_jump_multiframe_inputs(replace(bars, coverage=coverage), "boom")


def test_empty_minute_cannot_carry_a_run_id(source_bars):
    bars = source_bars["boom"]
    coverage = bars.coverage.copy()
    coverage.iloc[-1, coverage.columns.get_loc("observed_seconds")] = 0
    coverage.iloc[-1, coverage.columns.get_loc("minute_valid")] = False
    with pytest.raises(ValueError, match="Empty minutes"):
        causal_jump_multiframe_inputs(replace(bars, coverage=coverage), "boom")


def test_invalid_direction_or_pair_type_are_refused(source_bars):
    with pytest.raises(ValueError, match="direction"):
        causal_jump_multiframe_inputs(source_bars["boom"], "both")
    with pytest.raises(TypeError, match="PairedMinuteBars"):
        causal_jump_multiframe_inputs(source_bars["boom"].raw, "boom")
