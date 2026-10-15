"""Hand-calculated oracles for the independent region verifier."""
import numpy as np
import pandas as pd
import pytest

from scripts.verify_zone_study import TIME_FIELDS, independent_path, independent_pf_intervals, read_events


CFG = {"activation_delay_minutes": 1, "wait_minutes": 3, "hold_minutes": 2,
       "stop_atr": 2., "cost_atr": .10, "purge_minutes": 31}


def oracle(changes=None, missing=(), side=1, end=2400, busy=0, **cfg):
    times = np.arange(2400, dtype=np.int64)
    prices = np.full(2400, 100.)
    for t, p in (changes or {}).items(): prices[t] = p
    if side == -1: prices = 200. - prices
    keep = ~np.isin(times, missing)
    z = {"signal_time": "1970-01-01T00:00Z", "side": side, "atr": 1.,
         "zone_low": 96. if side == 1 else 102., "zone_high": 98. if side == 1 else 104.,
         "invalidation": 94. if side == 1 else 106.}
    return independent_path(times[keep], prices[keep], z, {**CFG, **cfg}, 0, end, busy)


@pytest.mark.parametrize("side", [1, -1])
def test_next_quote_loss_is_uncapped_and_expiry_is_touch_based(side):
    r, busy = oracle({61: 97., 62: 99., 63: 96., 64: 90.}, side=side)
    assert r["touch_time"] == 61 and r["entry_time"] == 62
    assert r["trigger_time"] == 63 and r["exit_time"] == busy == 64
    assert r["gross_R"] == -4.5 and r["net_R"] == -4.55


@pytest.mark.parametrize("changes,status", [({}, "expired"), ({61: 95.}, "expired"),
                                          ({60: 97.}, "expired"), ({180: 94.}, "invalidated")])
def test_invalidation_and_no_fabricated_zone_touch(changes, status):
    r, _ = oracle(changes)
    assert r["status"] == status and r["entry_time"] is None


@pytest.mark.parametrize("missing,status", [(30, "waiting_gap"), (62, "entry_gap"),
                                          (63, "path_gap"), (182, "path_gap")])
def test_each_required_missing_second_is_preserved(missing, status):
    r, busy = oracle({61: 97., 62: 99.}, missing=[missing])
    assert r["status"] == status and r["missing_time"] == missing
    assert r["net_R"] is None and r["censored"] and busy == 301


def test_stop_at_expiry_precedes_timeout_and_fill_is_next_quote():
    r, _ = oracle({61: 97., 62: 99., 181: 97., 182: 90.})
    assert r["reason"] == "sl" and r["trigger_time"] == 181 and r["exit_time"] == 182
    r, _ = oracle({61: 97., 62: 99., 182: 90.})
    assert r["reason"] == "time" and r["trigger_time"] is None and r["net_R"] == -4.55


def test_purge_and_occupied_regions_precede_any_price_outcome():
    r, _ = oracle({1: 93.}, end=1859)
    assert r["status"] == "purged"
    r, busy = oracle({1: 93.}, busy=1)
    assert r["status"] == "overlap_skipped" and busy == 1


def test_constant_gain_loss_days_produce_exact_PF_intervals():
    values = np.tile([1., -.5], 14)
    days = np.repeat(np.arange(14) * 86400, 2)
    r = independent_pf_intervals(values, days, 0, 14 * 86400, 199, 20261008)
    assert r == {"day_profit_factor_ci95": [2., 2.], "weekly_profit_factor_ci95": [2., 2.]}


def test_absent_losses_and_absent_trades_have_unknown_PF_intervals():
    for values, days in ((np.array([1.]), np.array([0])), (np.array([]), np.array([], dtype=int))):
        r = independent_pf_intervals(values, days, 0, 7 * 86400, 19, 7)
        assert r == {"day_profit_factor_ci95": [None, None], "weekly_profit_factor_ci95": [None, None]}


def test_empty_ledger_csv_has_same_semantic_schema_as_empty_event_subset(tmp_path):
    fields = (*TIME_FIELDS, "variant", "zone_id", "side", "zone_low", "zone_high", "invalidation",
              "issue_price", "atr", "entry", "exit", "gross_R", "net_R", "holding_minutes",
              "status", "reason", "censored", "ambiguous")
    record = dict.fromkeys(fields, "")
    record.update(signal_time="1970-01-01T00:00Z", variant="v", zone_id="z", side=1,
                  zone_low=96, zone_high=98, invalidation=94, issue_price=100, atr=1,
                  status="expired", reason="no_observed_touch", censored=False, ambiguous=False)
    frame = pd.DataFrame([record])
    frame.to_csv(tmp_path / "events.csv", index=False)
    frame.iloc[:0].to_csv(tmp_path / "ledger.csv", index=False)
    events, ledger = read_events(tmp_path / "events.csv"), read_events(tmp_path / "ledger.csv")
    assert ledger.equals(events.iloc[:0])
