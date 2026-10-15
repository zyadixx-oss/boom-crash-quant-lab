"""Synthetic unit fixtures test causality; they are not market evidence."""
import unittest
import numpy as np
import pandas as pd

from research_campaign.strategies import (FAMILIES, TIMEFRAMES, StrategySpec,
    StrategyFit, candidates_for_round, direction_for_symbol, fit_strategy,
    generate_signals, normalize_bars, prepare_context)


def candles(n=500):
    i=np.arange(n); c=100+.01*i+np.sin(i*.19)
    o=np.r_[c[0],c[:-1]]
    return pd.DataFrame({"epoch":i*60,"open":o,"high":np.maximum(o,c)+.2,
                         "low":np.minimum(o,c)-.2,"close":c})


class StrategyTests(unittest.TestCase):
    def test_declared_finite_search(self):
        specs=[s for r in range(32) for s in candidates_for_round(r)]
        self.assertEqual(len(specs),416)
        self.assertEqual(len({s.id for s in specs}),416)
        self.assertEqual({s.family for s in specs},set(FAMILIES))
        self.assertEqual(len(TIMEFRAMES),10)
        self.assertTrue(all(not s.trailing_stop for s in specs))

    def test_round_outside_declaration_rejected(self):
        for value in [-1,32,1.5,True]:
            with self.assertRaises(ValueError): candidates_for_round(value)

    def test_serialization_identity(self):
        for s in candidates_for_round(31):
            self.assertEqual(StrategySpec.from_dict(s.to_dict()),s)
        bad=candidates_for_round(0)[0].to_dict(); bad["id"]="tampered"
        with self.assertRaises(ValueError): StrategySpec.from_dict(bad)

    def test_scientific_cache_key_excludes_unused_parameter(self):
        self.assertEqual(candidates_for_round(0)[0].scientific_id,candidates_for_round(2)[0].scientific_id)
        self.assertNotEqual(candidates_for_round(0)[1].scientific_id,candidates_for_round(2)[1].scientific_id)

    def test_all_signals_prefix_invariant(self):
        full=candles(); prefix=full.iloc[:350]
        for r in (0,31):
            for spec in candidates_for_round(r):
                with self.subTest(round=r,family=spec.family):
                    fit=fit_strategy(full.iloc[:200],spec,"BOOM500")
                    a=generate_signals(full,spec,"BOOM500",fit)
                    b=generate_signals(prefix,spec,"BOOM500",fit)
                    np.testing.assert_array_equal(a[:350],b)
                    self.assertTrue(np.isin(a,[0,1]).all())

    def test_crash_direction_is_independent(self):
        b=candles()
        for spec in candidates_for_round(0):
            fit=fit_strategy(b.iloc[:200],spec,"CRASH500")
            signals=generate_signals(b,spec,"CRASH500",fit)
            self.assertTrue(np.isin(signals,[0,-1]).all())
        with self.assertRaises(ValueError): direction_for_symbol("BTC")

    def test_gaps_restart_all_indicators(self):
        full=candles(); gapped=full.drop(index=250).reset_index(drop=True)
        ctx=prepare_context(gapped,"M1")
        self.assertTrue(np.isnan(ctx.arrays["atr"][250:263]).all())
        np.testing.assert_allclose(ctx.arrays["atr"][249],prepare_context(full.iloc[:250],"M1").arrays["atr"][-1])
        for spec in candidates_for_round(0)[:-1]:
            self.assertEqual(int(np.count_nonzero(generate_signals(ctx,spec,"BOOM500")[250:285])),0)

    def test_fit_does_not_cross_missing_timestamp(self):
        full=candles(500); gapped=full.drop(index=250).reset_index(drop=True)
        s=candidates_for_round(0)[-1]
        fit=fit_strategy(prepare_context(gapped,"M1"),s,"BOOM500")
        # Two contiguous segments each lose their own indicator warmup and
        # their own final unobserved response.
        f1=fit_strategy(prepare_context(full.iloc[:250],"M1"),s,"BOOM500")
        f2=fit_strategy(prepare_context(full.iloc[251:],"M1"),s,"BOOM500")
        self.assertEqual(fit.training_rows,f1.training_rows+f2.training_rows)
        self.assertEqual(sum(fit.state_counts),fit.training_rows)

    def test_fit_frozen_identity_and_availability(self):
        b=candles(); s=candidates_for_round(0)[-1]
        fit=fit_strategy(b.iloc[:200],s,"BOOM500")
        self.assertEqual(StrategyFit.from_dict(fit.to_dict()),fit)
        self.assertFalse(generate_signals(b,s,"BOOM500",fit)[:199].any())
        with self.assertRaises(ValueError): generate_signals(b,s,"BOOM500")
        with self.assertRaises(ValueError): generate_signals(b,s,"CRASH500",fit)

    def test_bad_bars_not_silently_repaired(self):
        b=candles(5)
        with self.assertRaises(ValueError): normalize_bars(b.iloc[::-1])
        with self.assertRaises(ValueError): normalize_bars(pd.concat([b,b.iloc[-1:]]))
        bad=b.copy();bad.loc[0,"high"]=bad.loc[0,"low"]-1
        with self.assertRaises(ValueError): normalize_bars(bad)
        alias=b.rename(columns={"epoch":"timestamp"})
        self.assertIn("epoch",normalize_bars(alias))
        bad=b.copy();bad["timestamp"]=bad["epoch"]+1
        with self.assertRaises(ValueError): normalize_bars(bad)


if __name__=="__main__": unittest.main()
