"""Independent synthetic region/next-tick path checks; no historical outcomes."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from app.research.zone_replay import (
    FIELDS, TIME_FIELDS, ZoneReplayConfig, replay_zones, zone_trades,
)


BASE = pd.Timestamp("2026-01-01T00:00:00Z")


def stamp(seconds):
    return BASE + pd.Timedelta(seconds=seconds)


def region(issue=0, identity="z0", side=1, **overrides):
    value = {"signal_time": stamp(issue), "variant": "fixed-region", "zone_id": identity,
             "side": side, "zone_low": 96. if side == 1 else 102.,
             "zone_high": 98. if side == 1 else 104.,
             "invalidation": 94. if side == 1 else 106., "issue_price": 100., "atr": 1.}
    value.update(overrides)
    return value


def quotes(changes=None, missing=(), count=2400, side=1):
    values = np.full(count, 100., dtype=float)
    for second, value in (changes or {}).items():
        values[second] = value
    if side == -1:
        values = 200. - values
    frame = pd.DataFrame({"quote": values}, index=pd.date_range(BASE, periods=count, freq="s"))
    return frame.drop(index=[stamp(i) for i in missing])


def short_config(**overrides):
    # Short synthetic waits reduce test size; the production defaults are
    # separately asserted and the required31-minute purge stays in force.
    values = {"wait_minutes": 3, "hold_minutes": 2}
    values.update(overrides)
    return ZoneReplayConfig(**values)


def replay(frame, regions=None, config=None, start=0, end=2400):
    return replay_zones(frame, [region()] if regions is None else regions,
                        short_config() if config is None else config, stamp(start), stamp(end))


def single(frame, **kwargs):
    events, audit = replay(frame, **kwargs)
    assert len(events) == 1
    return events.iloc[0], audit


def test_default_protocol_constants_are_preserved():
    config = ZoneReplayConfig()
    assert config.activation_delay_minutes == 1
    assert config.wait_minutes == config.hold_minutes == 15
    assert config.stop_atr == 2. and config.cost_atr == .10
    assert config.purge_minutes == 31
    with pytest.raises(FrozenInstanceError): config.wait_minutes = 10


@pytest.mark.parametrize("side", [1, -1])
def test_fill_uses_actual_next_quote_beyond_zone_and_stop_loss_is_uncapped(side):
    events, audit = replay(quotes({61: 97., 62: 99., 63: 96., 64: 90.}, side=side),
                           regions=[region(side=side)])
    row = events.iloc[0]
    assert row.status == "completed" and row.reason == "sl"
    assert row.touch_time == stamp(61) and row.entry_time == stamp(62)
    assert row.entry == (99. if side == 1 else 101.)
    assert not row.zone_low <= row.entry <= row.zone_high
    assert row.trigger_time == stamp(63) and row.exit_time == stamp(64)
    assert row.exit == (90. if side == 1 else 110.)
    assert row.gross_R == -4.5 and row.net_R == pytest.approx(-4.55)
    assert row.holding_minutes == pytest.approx(2 / 60)
    assert not row.censored and not row.ambiguous
    assert audit["filled"] == 1 and audit["unknown"] == 0


def test_fill_after_observed_touch_is_not_repriced_or_cancelled_at_zone_boundary():
    row, _ = single(quotes({61: 97., 62: 90.}))
    assert row.status == "completed" and row.reason == "time"
    assert row.entry == 90. and row.entry < row.invalidation
    assert row.exit == 100. and row.net_R == pytest.approx(4.95)


@pytest.mark.parametrize("side", [1, -1])
def test_jump_through_zone_without_observed_touch_does_not_create_fill(side):
    row, audit = single(quotes({61: 95.}, side=side), regions=[region(side=side)])
    assert row.status == "expired" and row.reason == "no_observed_touch"
    assert pd.isna(row.touch_time) and pd.isna(row.entry_time)
    assert row.exit_time == stamp(180)
    assert pd.isna(row.net_R) and audit["filled"] == 0


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("second,price", [(1, 93.), (30, 94.), (61, 93.), (180, 94.)])
def test_invalidation_is_immediate_inclusive_and_precedes_fabricated_crossing(side, second, price):
    row, audit = single(quotes({second: price}, side=side), regions=[region(side=side)])
    assert row.status == "invalidated" and row.reason == "invalidation_before_touch"
    assert row.exit_time == stamp(second)
    assert pd.isna(row.touch_time) and pd.isna(row.entry_time) and not row.censored
    assert audit["filled"] == audit["unknown"] == 0


def test_touch_at_activation_is_ignored_but_next_second_is_eligible():
    ignored, _ = single(quotes({60: 97.}))
    accepted, _ = single(quotes({60: 97., 61: 97.}))
    assert ignored.status == "expired"
    assert accepted.touch_time == stamp(61) and accepted.entry_time == stamp(62)


@pytest.mark.parametrize("level", [96., 98.])
def test_observed_touch_includes_both_zone_edges(level):
    row, _ = single(quotes({61: level}))
    assert row.touch_time == stamp(61) and row.status == "completed"


def test_touch_at_wait_deadline_is_allowed_but_later_quote_is_not():
    accepted, _ = single(quotes({180: 97.}))
    expired, _ = single(quotes({181: 97.}))
    assert accepted.touch_time == stamp(180) and accepted.entry_time == stamp(181)
    assert accepted.exit_time == stamp(301)
    assert expired.status == "expired" and expired.exit_time == stamp(180)


@pytest.mark.parametrize("missing", [1, 30, 60, 61, 180])
def test_unknown_before_touch_is_censored_without_later_substitution(missing):
    row, audit = single(quotes(missing=[missing]))
    assert row.status == "waiting_gap" and row.reason == "unknown_before_touch"
    assert row.missing_time == stamp(missing) and row.censored
    assert pd.isna(row.entry_time) and pd.isna(row.net_R)
    assert audit["unknown"] == 1 and audit["filled"] == 0


def test_unknown_immediate_entry_remains_unfilled_even_if_next_later_quote_exists():
    row, audit = single(quotes({61: 97., 63: 97.}, missing=[62]))
    assert row.status == "entry_gap" and row.reason == "unknown_next_quote_entry"
    assert row.touch_time == stamp(61) and row.missing_time == stamp(62)
    assert pd.isna(row.entry_time) and pd.isna(row.entry) and pd.isna(row.net_R)
    assert row.censored and audit["filled"] == 0


def test_gap_after_entry_censors_path_and_never_creates_zero_payoff():
    row, audit = single(quotes({61: 97., 62: 99.}, missing=[63]))
    assert row.status == "path_gap" and row.reason == "unknown_after_entry"
    assert row.entry_time == stamp(62) and row.missing_time == stamp(63)
    assert row.exit_time == stamp(181) and pd.isna(row.exit)
    assert pd.isna(row.gross_R) and pd.isna(row.net_R) and row.censored
    assert audit["filled"] == audit["unknown"] == 1


def test_gap_at_required_stop_fill_does_not_use_trigger_or_later_quote():
    row, _ = single(quotes({61: 97., 62: 99., 63: 96., 65: 80.}, missing=[64]))
    assert row.status == "path_gap" and row.reason == "unknown_stop_fill"
    assert row.trigger_time == stamp(63) and row.missing_time == stamp(64)
    assert pd.isna(row.exit) and pd.isna(row.net_R)


def test_stop_trigger_exactly_at_expiry_takes_priority_over_timeout():
    row, _ = single(quotes({61: 97., 62: 99., 181: 97., 182: 90.}))
    assert row.status == "completed" and row.reason == "sl"
    assert row.trigger_time == stamp(181) and row.exit_time == stamp(182)
    assert row.entry_time == stamp(62) and row.holding_minutes == 2.
    assert row.exit == 90. and row.gross_R == -4.5


def test_first_quote_after_expiry_is_timeout_even_if_it_crosses_stop():
    row, _ = single(quotes({61: 97., 62: 99., 182: 90.}))
    assert row.status == "completed" and row.reason == "time"
    assert pd.isna(row.trigger_time) and row.exit_time == stamp(182)
    assert row.exit == 90. and row.net_R == pytest.approx(-4.55)


def test_missing_timeout_quote_is_unknown_even_when_later_quote_exists():
    row, _ = single(quotes({61: 97., 62: 99., 183: 110.}, missing=[182]))
    assert row.status == "path_gap" and row.missing_time == stamp(182)
    assert pd.isna(row.net_R)


def test_prices_and_gaps_after_completed_exit_cannot_change_completed_path():
    source = quotes({61: 97., 62: 99.})
    changed = source.copy(); changed.loc[stamp(183):, "quote"] = 1e6
    changed = changed.drop(index=stamp(190))
    before, _ = replay(source); after, _ = replay(changed)
    assert_frame_equal(before, after, check_exact=True)


def test_issued_levels_and_input_quotes_are_not_mutated():
    source = quotes({61: 97.}); regions = [region()]
    old_frame, old_regions = source.copy(deep=True), deepcopy(regions)
    replay(source, regions=regions)
    assert_frame_equal(source, old_frame)
    assert regions == old_regions


def test_outside_partition_invalid_numeric_values_do_not_change_results():
    source = quotes({61: 97.})
    outside = pd.DataFrame({"quote": [float("nan"), float("inf"), -1.]},
                           index=[stamp(-1), stamp(2400), stamp(2500)])
    expected, _ = replay(source)
    actual, _ = replay(pd.concat([source, outside]).sort_index())
    assert_frame_equal(actual, expected, check_exact=True)


def test_outside_partition_string_cannot_poison_valid_numeric_object_column():
    source = quotes({61: 97.})
    outside = pd.DataFrame({"quote": ["future invalid value"]}, index=[stamp(2400)])
    poisoned = pd.concat([source, outside])
    assert poisoned.quote.dtype == object
    expected, _ = replay(source); actual, _ = replay(poisoned)
    assert_frame_equal(actual, expected, check_exact=True)


def test_common_prefix_events_do_not_depend_on_valid_future_quotes():
    source = quotes({61: 97., 62: 99.})
    changed = source.copy(); changed.loc[stamp(600):, "quote"] *= 2.
    original, _ = replay(source); perturbed, _ = replay(changed)
    assert_frame_equal(original, perturbed, check_exact=True)


def test_31_minute_purge_is_planned_and_early_exit_cannot_rescue_it():
    config = ZoneReplayConfig()
    frame = quotes({1: 93.}, count=2000)
    purged, _ = single(frame, config=config, end=1859)
    allowed, _ = single(frame, config=config, end=1860)
    assert purged.status == "purged" and purged.reason == "planned_partition_purge"
    assert pd.isna(purged.exit_time)
    assert allowed.status == "invalidated" and allowed.exit_time == stamp(1)


def test_maximum_planned_end_must_be_strictly_inside_partition():
    config = short_config(purge_minutes=0)
    frame = quotes({1: 93.}, count=500)
    purged, _ = single(frame, config=config, end=301)
    allowed, _ = single(frame, config=config, end=302)
    assert purged.maximum_end == stamp(301) and purged.status == "purged"
    assert allowed.status == "invalidated"


def test_one_pending_region_or_position_blocks_new_issues_until_actual_exit():
    regions = [region(240, "later"), region(180, "position"), region(60, "pending"), region()]
    events, audit = replay(quotes({61: 97., 62: 99., 301: 97.}), regions=regions)
    assert events.zone_id.tolist() == ["z0", "pending", "position", "later"]
    assert events.status.tolist() == ["completed", "overlap_skipped", "overlap_skipped", "completed"]
    assert audit["filled"] == 2 and audit["issued"] == 4


def test_new_region_can_issue_at_exact_prior_invalidation_time():
    events, _ = replay(quotes({60: 94., 121: 97.}), regions=[region(), region(60, "next")])
    assert events.status.tolist() == ["invalidated", "completed"]
    assert events.iloc[0].exit_time == events.iloc[1].signal_time == stamp(60)


def test_new_region_can_issue_at_exact_untouched_expiry_time():
    events, _ = replay(quotes({241: 97.}), regions=[region(), region(180, "next")])
    assert events.status.tolist() == ["expired", "completed"]
    assert events.iloc[1].touch_time == stamp(241)


@pytest.mark.parametrize("kind", ["waiting", "entry", "path"])
def test_unknown_region_reserves_maximum_planned_occupancy(kind):
    changes, missing = {}, [10]
    if kind == "entry": changes, missing = {61: 97.}, [62]
    if kind == "path": changes, missing = {61: 97., 62: 99.}, [63]
    events, _ = replay(quotes(changes, missing=missing),
                       regions=[region(), region(300, "blocked"), region(360, "allowed")])
    assert events.iloc[0].censored and events.iloc[0].maximum_end == stamp(301)
    assert events.iloc[1].status == "overlap_skipped"
    assert events.iloc[2].status == "expired"


def test_same_time_regions_have_deterministic_identity_priority():
    a, b = region(identity="a"), region(identity="b")
    left, _ = replay(quotes(), regions=[b, a]); right, _ = replay(quotes(), regions=[a, b])
    assert_frame_equal(left, right, check_exact=True)
    assert left.zone_id.tolist() == ["a", "b"]
    assert left.status.tolist() == ["expired", "overlap_skipped"]


def test_outside_partition_regions_are_retained_without_occupancy():
    events, audit = replay(quotes(), regions=[region(-60, "before"), region(), region(2400, "after")])
    assert events.status.tolist() == ["outside_partition", "expired", "outside_partition"]
    assert audit["issued"] == 3 and audit["statuses"]["outside_partition"] == 2


def test_empty_sources_and_empty_zone_lists_remain_explicit():
    frame = quotes(count=0)
    missing, audit = replay(frame)
    assert missing.iloc[0].status == "waiting_gap" and audit["unknown"] == 1
    empty, audit = replay(frame, regions=[])
    assert empty.empty and tuple(empty.columns) == FIELDS
    assert audit["issued"] == audit["filled"] == audit["unknown"] == 0
    for name in TIME_FIELDS: assert str(empty[name].dtype) == "datetime64[ns, UTC]"


def test_trade_view_retains_filled_unknown_paths_and_does_not_mutate_events():
    events, _ = replay(quotes({61: 97., 62: 99.}, missing=[63]),
                       regions=[region(), region(300, "unfilled"), region(360, "expired")])
    saved = events.copy(deep=True); ledger = zone_trades(events)
    assert len(ledger) == 1 and ledger.iloc[0].status == "path_gap"
    assert ledger.iloc[0].censored and pd.isna(ledger.iloc[0].net_R)
    ledger.iloc[0, ledger.columns.get_loc("entry")] = 12345.
    assert_frame_equal(events, saved)


@pytest.mark.parametrize("key,value", [("activation_delay_minutes", True), ("activation_delay_minutes", -1),
    ("wait_minutes", 0), ("hold_minutes", 0), ("purge_minutes", -1), ("purge_minutes", 1.5),
    ("stop_atr", 0), ("stop_atr", math.inf), ("stop_atr", True), ("cost_atr", -1),
    ("cost_atr", float("nan")), ("cost_atr", "0.1"), ("activation_delay_minutes", 15)])
def test_invalid_configuration_is_rejected(key, value):
    with pytest.raises(ValueError): ZoneReplayConfig(**{key: value})


@pytest.mark.parametrize("change", [{"side": True}, {"side": 1.}, {"side": 0}, {"variant": ""},
    {"zone_id": ""}, {"zone_low": 98.}, {"zone_high": 100.}, {"invalidation": 96.},
    {"zone_low": -1.}, {"atr": 0.}, {"atr": True}, {"atr": math.inf},
    {"issue_price": "100"}, {"signal_time": BASE + pd.Timedelta(seconds=1)},
    {"signal_time": BASE.tz_localize(None)}])
def test_invalid_region_contract_is_rejected(change):
    with pytest.raises(ValueError): replay(quotes(), regions=[region(**change)])


@pytest.mark.parametrize("kind", ["duplicate_id", "mixed_variant", "mixed_side", "missing", "nonlist", "notdict"])
def test_stream_identity_and_required_region_schema(kind):
    regions = [region(), region(60, "z1")]
    if kind == "duplicate_id": regions[1]["zone_id"] = "z0"
    elif kind == "mixed_variant": regions[1]["variant"] = "other"
    elif kind == "mixed_side": regions[1] = region(60, "z1", side=-1)
    elif kind == "missing": del regions[0]["atr"]
    elif kind == "nonlist": regions = tuple(regions)
    elif kind == "notdict": regions[0] = 1
    with pytest.raises((ValueError, TypeError)): replay(quotes(), regions=regions)


@pytest.mark.parametrize("kind", ["boolean", "complex", "text", "nan", "zero", "duplicate", "reverse", "subsecond", "naive", "columns"])
def test_invalid_in_partition_tick_schema_is_rejected(kind):
    frame = quotes()
    if kind == "boolean": frame["quote"] = True
    elif kind == "complex": frame["quote"] = frame.quote.astype(complex)
    elif kind == "text": frame["quote"] = frame.quote.astype(str)
    elif kind == "nan": frame.iloc[1, 0] = float("nan")
    elif kind == "zero": frame.iloc[1, 0] = 0.
    elif kind == "duplicate": frame = pd.concat([frame.iloc[:1], frame])
    elif kind == "reverse": frame = frame.iloc[::-1]
    elif kind == "subsecond": frame.index += pd.Timedelta(milliseconds=1)
    elif kind == "naive": frame.index = frame.index.tz_localize(None)
    elif kind == "columns": frame = pd.concat([frame, frame], axis=1)
    with pytest.raises((ValueError, TypeError)): replay(frame)


@pytest.mark.parametrize("flag", ["LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"])
def test_enabled_execution_gate_blocks_before_path_validation(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    with pytest.raises((ValueError, RuntimeError)):
        replay_zones(None, None, None, None, None)


def test_audit_identifies_only_assumed_quote_proxy_costs_and_disabled_execution():
    _, audit = replay(quotes({61: 97.}))
    assert audit["quote_proxy_only"] is True and audit["broker_execution"] == "NOT TESTED"
    assert set(audit["safety"]) == {"LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"}
    assert all(value is False for value in audit["safety"].values())
