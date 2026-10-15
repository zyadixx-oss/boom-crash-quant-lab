"""Adversarial accounting fixtures only; these are not historical results."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import hashlib
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("ledger_audit", Path(__file__).with_name("audit_ledger.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

START = int(datetime(2026, 5, 31, 23, 59, tzinfo=timezone.utc).timestamp())


def fixture():
    trade = {"entry_atr": 10., "entry_price": 100., "exit_price": 120.,
             "direction": 1, "risk_fraction": .005, "equity_before": 10000.,
             "risk_cash": 50., "quantity": 5., "round_trip_cost_price": 1.,
             "stop_price": 90., "target_price": 120., "gross_R": 2., "net_R": 1.9,
             "net_pnl": 95., "equity_after": 10095., "signal_epoch": START,
             "entry_epoch": START, "exit_epoch": START + 120,
             "planned_end_epoch": START + 120, "signal_bar_index": 0, "entry_bar_index": 1,
             "exit_reason": "TARGET", "ambiguous_candle": False,
             "execution_mode": "assumed_candle_stop_first"}
    result = {"symbol": "BOOM500", "timeframe": "M1", "strategy_id": "fixture_only",
              "partition": "final", "start": START, "end": START + 300,
              "strategy": {"family": "crt", "stop_atr": 1., "target_atr": 2.,
                           "max_hold_bars": 2, "trailing_stop": False},
              "costs": {"round_trip_atr": .1, "round_trip_price": 0.},
              "safety": {k: False for k in mod.FLAGS}, "trades": [trade],
              "equity": [{"epoch": START, "equity": 10000., "kind": "initial"},
                         {"epoch": START + 120, "equity": 10200., "kind": "favorable"},
                         {"epoch": START + 120, "equity": 10020., "kind": "adverse"},
                         {"epoch": START + 120, "equity": 10095., "kind": "realized"}],
              "metrics": {"initial_equity": 10000., "final_equity": 10095., "net_profit": 95.,
                          "net_return": .0095, "completed_trades": 1, "sample_size": 1,
                          "profit_factor": None, "win_rate": 1., "mean_net_R": 1.9,
                          "median_holding_seconds": 120., "active_days": 1,
                          "period_days": 300 / 86400, "max_drawdown": 180 / 10200},
              "monthly": [{"month": "2026-06", "completed_trades": 1, "net_profit": 95.,
                           "sum_net_R": 1.9, "equity_start": 10000., "equity_end": 10095.,
                           "net_return": .0095}], "reference": {},
              "dispositions": [{"status": "COMPLETED", "trade_index": 0,
                                "signal_epoch": START, "entry_epoch": START,
                                "planned_end_epoch": START + 120}],
              "unknown_paths": 0, "purged_signals": 0, "ambiguous_candles": 0}
    result["strategy"]["id"] = "fixture_only"
    result["id"] = hashlib.sha256(json.dumps({k: result[k] for k in
        ("strategy", "symbol", "timeframe", "start", "end", "partition", "costs")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"final": result, "spec": deepcopy(result["strategy"]), "symbol": "BOOM500",
            "timeframe": "M1", "identity": {"symbol": "BOOM500", "timeframe": "M1",
            "phase": "final", "split": {"validation_end": START, "end": START + 300}}}


def check(value):
    c = mod.Checks()
    mod.audit_result(value, c)
    return c


def test_hand_computed_long_and_UTC_exit_month():
    assert check(fixture()).error_count == 0


@pytest.mark.parametrize("field", ["quantity", "round_trip_cost_price", "risk_cash", "risk_fraction",
                                  "equity_before", "equity_after", "stop_price", "target_price",
                                  "gross_R", "net_R", "net_pnl"])
def test_single_trade_amount_corruption_rejected(field):
    value = fixture()
    value["final"]["trades"][0][field] += .01
    assert check(value).error_count > 0


@pytest.mark.parametrize("field", ["net_profit", "net_return", "final_equity", "mean_net_R",
                                  "win_rate", "median_holding_seconds", "max_drawdown"])
def test_metric_corruption_rejected(field):
    value = fixture()
    value["final"]["metrics"][field] += .01
    assert check(value).error_count > 0


def test_long_and_native_short_are_symmetric():
    value = fixture()
    r = value["final"]
    r["symbol"] = "CRASH500"
    value["symbol"] = value["identity"]["symbol"] = "CRASH500"
    r["trades"][0].update(direction=-1, exit_price=80., stop_price=110., target_price=80.)
    r["id"] = hashlib.sha256(json.dumps({k: r[k] for k in
        ("strategy", "symbol", "timeframe", "start", "end", "partition", "costs")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert check(value).error_count == 0


def test_no_loss_profit_factor_must_be_null():
    value = fixture()
    value["final"]["metrics"]["profit_factor"] = 99.
    assert check(value).error_count > 0


def test_month_of_entry_is_not_exit_month():
    value = fixture()
    value["final"]["monthly"][0]["month"] = "2026-05"
    assert check(value).error_count > 0


def test_duplicate_epoch_adverse_mark_cannot_be_dropped():
    value = fixture()
    del value["final"]["equity"][2]
    assert check(value).error_count > 0


def test_completed_disposition_duplicates_rejected():
    value = fixture()
    value["final"]["dispositions"].append(deepcopy(value["final"]["dispositions"][0]))
    assert check(value).error_count > 0


def test_unknown_path_is_not_a_completed_zero_return_trade():
    value = fixture()
    value["final"]["dispositions"].append({"status": "UNKNOWN_MISSING_HOLDING_BARS"})
    assert check(value).error_count > 0
    value["final"]["unknown_paths"] = 1
    assert check(value).error_count == 0


def test_empty_ledger_retains_initial_cash_and_nulls():
    value = fixture()
    r = value["final"]
    r.update(trades=[], monthly=[], dispositions=[], equity=r["equity"][:1])
    r["metrics"].update(final_equity=10000., net_profit=0., net_return=0., completed_trades=0,
                         sample_size=0, win_rate=None, profit_factor=None, mean_net_R=None,
                         median_holding_seconds=None, active_days=0, max_drawdown=0.)
    assert check(value).error_count == 0


def test_gap_loss_is_not_clamped_to_initial_capital():
    value = fixture()
    r = value["final"]
    r["trades"][0].update(entry_atr=1., entry_price=1000., exit_price=10., stop_price=999.,
                           target_price=1002., quantity=50., round_trip_cost_price=.1,
                           gross_R=-990., net_R=-990.1, net_pnl=-49505., equity_after=-39505.,
                           exit_reason="GAP_STOP")
    r["equity"] = [r["equity"][0], {"epoch": START + 120, "equity": -39505., "kind": "realized"}]
    r["metrics"].update(final_equity=-39505., net_profit=-49505., net_return=-4.9505,
                         profit_factor=0., win_rate=0., mean_net_R=-990.1, max_drawdown=4.9505)
    r["monthly"][0].update(net_profit=-49505., sum_net_R=-990.1, equity_end=-39505., net_return=-4.9505)
    assert check(value).error_count == 0


def test_PF_uses_cash_not_R_when_risk_compounds():
    value = fixture()
    r = value["final"]
    second = deepcopy(r["trades"][0])
    second.update(entry_epoch=START + 120, signal_epoch=START + 120,
                  planned_end_epoch=START + 240, exit_epoch=START + 180,
                  equity_before=10095., risk_cash=50.475, quantity=5.0475,
                  exit_price=90., gross_R=-1., net_R=-1.1, net_pnl=-55.5225,
                  equity_after=10039.4775, exit_reason="STOP")
    r["trades"].append(second)
    r["equity"].append({"epoch": START + 180, "equity": 10039.4775, "kind": "realized"})
    r["dispositions"].append({"status": "COMPLETED", "trade_index": 1,
                               "signal_epoch": START + 120, "entry_epoch": START + 120,
                               "planned_end_epoch": START + 240})
    r["metrics"].update(final_equity=10039.4775, net_profit=39.4775,
                         net_return=.00394775, completed_trades=2, sample_size=2,
                         profit_factor=95. / 55.5225, win_rate=.5, mean_net_R=.4,
                         median_holding_seconds=90., active_days=2)
    r["monthly"][0].update(completed_trades=2, net_profit=39.4775,
                           sum_net_R=.8, equity_end=10039.4775, net_return=.00394775)
    assert check(value).error_count == 0
    r["metrics"]["profit_factor"] = 1.9 / 1.1
    assert check(value).error_count > 0


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "95"])
def test_nonnumeric_or_nonfinite_cash_rejected(bad):
    value = fixture()
    value["final"]["trades"][0]["net_pnl"] = bad
    with pytest.raises((TypeError, ValueError)) if isinstance(bad, str) else __import__("contextlib").nullcontext():
        assert check(value).error_count > 0


def test_ledger_cannot_enable_live_gate():
    value = fixture()
    value["final"]["safety"]["LIVE_ALLOWED"] = True
    assert check(value).error_count > 0


@pytest.mark.parametrize("field", ["strategy_id", "symbol", "timeframe", "start", "end", "partition", "id"])
def test_nested_selected_identity_cannot_be_substituted(field):
    value = fixture()
    r = value["final"]
    if field == "timeframe":
        r[field] = "M2"
    elif field in {"start", "end"}:
        r[field] += 1
    elif field == "symbol":
        r[field] = "CRASH500"
    else:
        r[field] = "WRONG_SELECTED_POLICY"
    assert check(value).error_count > 0


@pytest.mark.parametrize("field", ["completed_trades", "sample_size", "active_days"])
def test_boolean_metric_counts_are_not_integer_samples(field):
    value = fixture()
    value["final"]["metrics"][field] = True
    assert check(value).error_count > 0


def test_boolean_month_count_rejected():
    value = fixture()
    value["final"]["monthly"][0]["completed_trades"] = True
    assert check(value).error_count > 0
