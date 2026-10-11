"""Unit-only synthetic candles and ticks; never historical performance."""
from dataclasses import replace
import inspect
import json
import unittest
import numpy as np
import pandas as pd

from research_campaign.engine import (CostModel, SplitPlan, block_uncertainty,
    evaluate_final, finalize_multiplicity, risk_adjusted_score, run_backtest,
    validate_bars)
from research_campaign.strategies import candidates_for_round, prepare_context


def context(n=120):
    frame=pd.DataFrame({"epoch":np.arange(n)*60,"open":100.,"high":100.1,"low":99.9,"close":100.})
    ctx=prepare_context(frame,"M1");ctx.arrays={k:v.copy() for k,v in ctx.arrays.items()};ctx.arrays["atr"][:]=1
    return ctx


def run(ctx, entries=(30,), **kwargs):
    s=np.zeros(len(ctx.bars),dtype=np.int8);s[list(entries)]=1
    spec=replace(candidates_for_round(0)[0],stop_atr=1,target_atr=2,max_hold_bars=4)
    return run_backtest(ctx,s,spec,"BOOM500","M1",1200,6000,costs=CostModel(round_trip_atr=.1),**kwargs)


class EngineTests(unittest.TestCase):
    def test_calendar_split_is_shared_across_frames(self):
        p=SplitPlan.from_calendar(0,1000)
        self.assertEqual(p.bounds("train"),(0,600))
        self.assertEqual(p.bounds("validation"),(600,800))
        self.assertEqual(p.bounds("final"),(800,1000))
        self.assertEqual(SplitPlan.from_dict(p.to_dict()),p)

    def test_signal_at_close_entry_next_open(self):
        ctx=context();ctx.arrays["open"][31]=101;ctx.arrays["high"][31]=101.1;ctx.arrays["low"][31]=100.9;ctx.arrays["close"][31]=101
        r=run(ctx);t=r["trades"][0]
        self.assertEqual(t["signal_epoch"],1860)
        self.assertEqual(t["entry_epoch"],1860)
        self.assertEqual(t["entry_price"],101)
        self.assertEqual(t["stop_price"],100)

    def test_fixed_risk_and_cost_accounting(self):
        r=run(context());t=r["trades"][0]
        self.assertAlmostEqual(t["risk_cash"],50)
        self.assertAlmostEqual(t["quantity"],50)
        self.assertAlmostEqual(t["net_R"],-.1)
        self.assertAlmostEqual(t["net_pnl"],-5)
        self.assertAlmostEqual(r["metrics"]["net_profit"],-5)
        self.assertFalse(r["gates"]["actual_contract_costs_verified"])
        self.assertFalse(r["qualified"])

    def test_ambiguous_bar_stop_first(self):
        ctx=context();ctx.arrays["high"][31]=103;ctx.arrays["low"][31]=98
        r=run(ctx);t=r["trades"][0]
        self.assertEqual(t["exit_reason"],"AMBIGUOUS_STOP_FIRST")
        self.assertEqual(t["exit_price"],99)
        self.assertAlmostEqual(t["net_R"],-1.1)
        self.assertEqual(r["ambiguous_candles"],1)

    def test_gap_loss_is_uncapped(self):
        ctx=context();ctx.arrays["open"][32]=95;ctx.arrays["high"][32]=95.1;ctx.arrays["low"][32]=94.9;ctx.arrays["close"][32]=95
        r=run(ctx);t=r["trades"][0]
        self.assertEqual(t["exit_reason"],"GAP_STOP")
        self.assertEqual(t["exit_price"],95)
        self.assertAlmostEqual(t["net_R"],-5.1)
        self.assertLess(t["net_pnl"],-t["risk_cash"])

    def test_one_open_position(self):
        r=run(context(),entries=(30,31,32,33,34))
        self.assertEqual(len(r["trades"]),2)
        self.assertEqual(sum(x["status"]=="OCCUPIED" for x in r["dispositions"]),3)

    def test_full_horizon_purge_even_if_early_target_would_hit(self):
        ctx=context();ctx.arrays["high"][98]=105
        r=run(ctx,entries=(97,))
        self.assertEqual(r["metrics"]["completed_trades"],0)
        self.assertEqual(r["purged_signals"],1)

    def test_missing_next_bar_never_fabricates_entry(self):
        ctx=context();ctx.bars=ctx.bars.drop(index=31).reset_index(drop=True)
        for key in ctx.arrays: ctx.arrays[key]=np.delete(ctx.arrays[key],31)
        r=run(ctx)
        self.assertEqual(len(r["trades"]),0)
        self.assertEqual(r["unknown_paths"],1)
        self.assertFalse(r["gates"]["data_quality"])

    def test_missing_holding_bar_unknown_occupancy(self):
        ctx=context();ctx.bars=ctx.bars.drop(index=33).reset_index(drop=True)
        for key in ctx.arrays: ctx.arrays[key]=np.delete(ctx.arrays[key],33)
        r=run(ctx,entries=(30,31,32))
        self.assertFalse(r["trades"])
        self.assertTrue(r["unknown_paths"])

    def test_tick_mode_requires_exact_complete_path(self):
        ctx=context();epoch=np.arange(1860,2100)
        ticks=pd.DataFrame({"epoch":epoch,"quote":100.})
        r=run(ctx,ticks=ticks)
        self.assertEqual(r["trades"][0]["execution_mode"],"original_tick_observations")
        bad=ticks.drop(index=5)
        r=run(ctx,ticks=bad)
        self.assertEqual(r["unknown_paths"],1)
        self.assertFalse(r["trades"])

    def test_tick_stop_actual_quote_not_stop_threshold(self):
        ctx=context();epoch=np.arange(1860,2100);quotes=np.full(len(epoch),100.)
        quotes[10:]=97
        r=run(ctx,ticks=pd.DataFrame({"epoch":epoch,"quote":quotes}))
        self.assertEqual(r["trades"][0]["exit_price"],97)
        self.assertAlmostEqual(r["trades"][0]["net_R"],-3.1)

    def test_no_final_selection_leak(self):
        r=run(context(),partition="final",bootstrap_repetitions=99)
        with self.assertRaises(ValueError): risk_adjusted_score(r)
        self.assertEqual(risk_adjusted_score(run(context())),-1e12)

    def test_cost_verification_requires_evidence_and_tick_path(self):
        with self.assertRaises(ValueError): CostModel(round_trip_atr=-.1)
        with self.assertRaises(ValueError): CostModel(mapping_verified=True)
        costs=CostModel(mapping_verified=True,historical_costs_verified=True,execution_verified=True,evidence=("unit fixture",))
        self.assertEqual(CostModel.from_dict(costs.to_dict()),costs)
        r=run(context());r["costs"]=costs.to_dict()
        self.assertFalse(evaluate_final(r)["original_tick_execution_verified"])

    def test_block_bootstrap_and_holm_are_deterministic(self):
        trades=[{"entry_epoch":i*86400,"net_R":2. if i%3 else -1.,"net_pnl":2. if i%3 else -1.} for i in range(120)]
        a=block_uncertainty(trades,19,199);b=block_uncertainty(trades,19,199)
        self.assertEqual(a,b)
        self.assertEqual(a["status"],"TESTED")
        rows=[]
        for p in [.01,.04,.2]:
            r=run(context(),partition="final",bootstrap_repetitions=99)
            r["id"]=str(p);r["uncertainty"]["p_value"]=p;rows.append(r)
        finalize_multiplicity(rows)
        np.testing.assert_allclose([r["uncertainty"]["holm_p_value"] for r in rows],[.03,.08,.2])

    def test_safe_json_and_all_live_gates_false(self):
        r=run(context())
        json.dumps(r,allow_nan=False)
        self.assertTrue(all(v is False for v in r["safety"].values()))
        self.assertIsNone(run(context(),entries=())["metrics"]["profit_factor"])

    def test_native_and_long_reference_retained(self):
        r=run(context())
        self.assertIn("buy_and_hold",r["reference"])
        self.assertIn("native_direction_hold",r["reference"])
        self.assertFalse(r["reference"]["appropriate_for_cfd"])

    def test_reference_open_uses_previous_closed_atr(self):
        ctx=context();ctx.arrays["atr"][20]=1000
        r=run(ctx)
        self.assertAlmostEqual(r["reference"]["buy_and_hold"]["round_trip_cost_price"],.1)

    def test_default_bootstrap_resolution_supports_1300_hypotheses(self):
        draws=inspect.signature(block_uncertainty).parameters["repetitions"].default
        self.assertLessEqual(1/(draws+1),.05/1300)
        self.assertEqual(inspect.signature(run_backtest).parameters["bootstrap_repetitions"].default,draws)

    def test_data_validation_and_wrong_direction(self):
        ctx=context();self.assertTrue(validate_bars(ctx.bars,"M1")["ok"])
        signals=np.zeros(len(ctx.bars),dtype=np.int8);signals[30]=-1
        with self.assertRaises(ValueError): run_backtest(ctx,signals,candidates_for_round(0)[0],"BOOM500","M1",0,6000)


if __name__=="__main__": unittest.main()
