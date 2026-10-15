"""Synthetic unit fixtures test replay integrity, never historical profitability."""
import copy
import gzip
import json
from pathlib import Path
import platform
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_campaign import campaign as c
from research_campaign import replay as r
from research_campaign.data import FRAMES, load_m1
from research_campaign.engine import CostModel, SplitPlan
from research_campaign.state import SAFETY, atomic_json, digest, encoded, read_json
from research_campaign.strategies import candidates_for_round


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
        n=1800;epochs=1728000000+np.arange(n)*60
        close=100+np.sin(np.arange(n)/7)+np.arange(n)*.0002
        opening=np.r_[close[0],close[:-1]]
        frame=pd.DataFrame({"epoch":epochs,"open":opening,"high":np.maximum(opening,close)+.08,
                            "low":np.minimum(opening,close)-.08,"close":close})
        self.source=self.directory/"unit_synthetic_not_market.csv"
        frame.to_csv(self.source,index=False,float_format="%.17g")
        self.frame=load_m1(self.source)
        manifest=self.directory/"unit_manifest.json"
        atomic_json(manifest,{"source":"SYNTHETIC UNIT TEST ONLY; NOT HISTORICAL EVIDENCE"})
        split=SplitPlan.from_calendar(int(epochs[0]),int(epochs[-1]+60))
        self.declaration={"split":split.to_dict(),"costs":CostModel().to_dict(),
            "source_pins":c.source_pins(),"source_manifest":str(manifest),"source_manifest_sha256":digest(manifest),
            "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
            "timeframes":dict(FRAMES),"candidate_rounds":[[s.to_dict() for s in candidates_for_round(i)] for i in range(32)],
            "final_bootstrap_repetitions":99,
            "datasets":{"BOOM500":{"requested_file":str(self.source),"requested_sha256":digest(self.source),
                "manifest_file":str(manifest),"api_symbol":"BOOM500","quality":{"missing_minutes":0}}}}
        atomic_json(self.directory/"declaration.json",self.declaration)
        self.state={"declaration_sha256":digest(self.directory/"declaration.json"),"target_cycles":32,
                    "interval_seconds":900,"safety":SAFETY,"completed_cycles":0,"cycles":[],
                    "current_cycle":1,"started_epoch":time.time(),"final_accessed":False}
        atomic_json(self.directory/"state.json",self.state)
        self.spec=candidates_for_round(0)[2]
        self.saved,self.artifact,_=c.measure(self.directory,self.declaration,"BOOM500","M1",self.spec,self.frame)

    def tearDown(self): self.temp.cleanup()

    def rewrite_saved(self,mutator):
        value=copy.deepcopy(self.saved);mutator(value)
        path=self.directory/"artifacts"/"tampered_but_rehashed.json.gz"
        path.write_bytes(gzip.compress(encoded(value),mtime=0))
        return {"path":str(path.relative_to(self.directory)),"sha256":digest(path)}

    def save_declaration(self):
        atomic_json(self.directory/"declaration.json",self.declaration)
        self.state["declaration_sha256"]=digest(self.directory/"declaration.json")
        atomic_json(self.directory/"state.json",self.state)

    def authorize_final(self,spec=None):
        spec=spec or self.spec
        selection={"source":self.state["declaration_sha256"],"final_data_accessed":False,
                   "selection":{f"BOOM500/M1/{spec.family}":{"status":"TESTED","strategy_id":spec.id,"spec":spec.to_dict()}}}
        atomic_json(self.directory/"selection_frozen.json",selection)
        self.state.update(final_accessed=True,completed_cycles=32,started_epoch=1,current_cycle=None,
                          cycles=[{"number":i,"status":"COMPLETED"} for i in range(1,33)],
                          selection_sha256=digest(self.directory/"selection_frozen.json"))
        atomic_json(self.directory/"state.json",self.state)

    def test_fresh_replay_matches_actual_ledgers_without_measure_cache(self):
        self.assertGreater(len(self.saved["validation"]["trades"]),0)
        original=r.run_backtest
        with patch.object(c,"measure",side_effect=AssertionError("Replay used measurement cache")),patch.object(r,"run_backtest",wraps=original) as calls:
            report=r.replay_measurement(self.directory,self.artifact,self.declaration)
        self.assertEqual(calls.call_count,2)
        self.assertEqual(report["status"],"PASS")
        self.assertEqual(report["matched_partitions"],2)
        self.assertEqual(report["matched_trades"],sum(len(self.saved[p]["trades"]) for p in ("train","validation")))
        self.assertEqual(report["matched_dispositions"],report["matched_signal_epochs"])
        self.assertTrue(report["full_signal_fingerprint_matched"])
        self.assertFalse(report["cache_used"])

    def test_raw_artifact_byte_tamper_is_rejected(self):
        (self.directory/self.artifact["path"]).write_bytes(gzip.compress(b"{}"))
        with self.assertRaisesRegex(ValueError,"artifact changed"):
            r.replay_measurement(self.directory,self.artifact,self.declaration)

    def test_rehashed_numerical_tamper_is_caught_by_actual_rerun(self):
        artifact=self.rewrite_saved(lambda x:x["validation"]["metrics"].__setitem__("net_profit",123456.))
        with self.assertRaisesRegex(r.ReplayMismatch,"net_profit"):
            r.replay_measurement(self.directory,artifact,self.declaration)

    def test_rehashed_trade_price_tamper_is_caught(self):
        artifact=self.rewrite_saved(lambda x:x["validation"]["trades"][0].__setitem__("entry_price",1.))
        with self.assertRaisesRegex(r.ReplayMismatch,"entry_price"):
            r.replay_measurement(self.directory,artifact,self.declaration)

    def test_rehashed_effective_signal_tamper_is_caught(self):
        artifact=self.rewrite_saved(lambda x:x["validation"]["dispositions"][0].__setitem__("signal_epoch",1))
        with self.assertRaisesRegex(r.ReplayMismatch,"effective signal epochs"):
            r.replay_measurement(self.directory,artifact,self.declaration)

    def test_full_vector_hash_tamper_is_caught(self):
        artifact=self.rewrite_saved(lambda x:x["signal_fingerprint"].__setitem__("sha256","0"*64))
        with self.assertRaisesRegex(r.ReplayMismatch,"signal fingerprint"):
            r.replay_measurement(self.directory,artifact,self.declaration)

    def test_price_file_tamper_is_rejected_before_measurement(self):
        changed=self.frame.copy();changed.loc[300,"close"]+=.01
        changed.to_csv(self.source,index=False)
        with patch.object(r,"run_backtest",side_effect=AssertionError("Used changed prices")):
            with self.assertRaisesRegex(ValueError,"price source changed"):
                r.replay_measurement(self.directory,self.artifact,self.declaration)

    def test_code_and_runtime_pins_are_rechecked(self):
        self.declaration["source_pins"]["engine.py"]="0"*64;self.save_declaration()
        with self.assertRaisesRegex(ValueError,"code changed"):
            r.replay_measurement(self.directory,self.artifact,self.declaration)
        self.declaration["source_pins"]=c.source_pins();self.declaration["runtime"]["numpy"]="wrong";self.save_declaration()
        with self.assertRaisesRegex(ValueError,"runtime changed"):
            r.replay_measurement(self.directory,self.artifact,self.declaration)

    def test_frozen_declaration_requires_authoritative_state_hash(self):
        atomic_json(self.directory/"declaration.json",{**self.declaration,"unexpected":True})
        with self.assertRaisesRegex(ValueError,"authoritative frozen state"):
            r.replay_measurement(self.directory,self.artifact,self.declaration)

    def test_development_never_constructs_final_indicator_values(self):
        changed=self.frame.copy();end=self.declaration["split"]["validation_end"]
        for name in ("open","high","low","close"):
            changed.loc[changed.epoch>=end,name]*=10000
        original=r.prepare_context
        def guarded(frame,timeframe=None):
            self.assertTrue((frame.epoch+FRAMES[timeframe]<=end).all())
            return original(frame,timeframe)
        with patch.object(r,"load_m1",return_value=changed),patch.object(r,"prepare_context",side_effect=guarded):
            report=r.replay_measurement(self.directory,self.artifact,self.declaration)
        self.assertEqual(report["status"],"PASS")

    def test_lookahead_signal_is_detected_by_prefix_test(self):
        original=r.generate_signals
        def leaky(ctx,spec,symbol,fit=None):
            signals=original(ctx,spec,symbol,fit).copy()
            signals[100]=int(len(ctx.bars)>1400)
            return signals
        with patch.object(r,"generate_signals",side_effect=leaky):
            with self.assertRaisesRegex(r.ReplayMismatch,"prefix invariance"):
                r.replay_measurement(self.directory,self.artifact,self.declaration)

    def test_final_flag_cannot_open_authoritatively_sealed_holdout(self):
        _,artifact,_=c.measure(self.directory,self.declaration,"BOOM500","M1",self.spec,self.frame,phase="final")
        with patch.object(r,"load_m1",side_effect=AssertionError("Opened final source")):
            with self.assertRaisesRegex(ValueError,"holdout remains sealed"):
                r.replay_measurement(self.directory,artifact,self.declaration,final_access_allowed=True)

    def test_final_replay_requires_explicit_flag_and_pinned_exact_selection(self):
        _,artifact,_=c.measure(self.directory,self.declaration,"BOOM500","M1",self.spec,self.frame,phase="final")
        self.authorize_final()
        with self.assertRaisesRegex(ValueError,"holdout remains sealed"):
            r.replay_measurement(self.directory,artifact,self.declaration)
        result=r.replay_measurement(self.directory,artifact,self.declaration,final_access_allowed=True)
        self.assertEqual(result["matched_partitions"],1)
        self.assertEqual(result["phase"],"final")
        self.authorize_final(candidates_for_round(0)[1])
        with self.assertRaisesRegex(ValueError,"not selected"):
            r.replay_measurement(self.directory,artifact,self.declaration,final_access_allowed=True)

    def test_final_replay_requires_real_eight_hour_elapsed_window(self):
        _,artifact,_=c.measure(self.directory,self.declaration,"BOOM500","M1",self.spec,self.frame,phase="final")
        self.authorize_final();self.state["started_epoch"]=time.time();atomic_json(self.directory/"state.json",self.state)
        with self.assertRaisesRegex(ValueError,"eight hours"):
            r.replay_measurement(self.directory,artifact,self.declaration,final_access_allowed=True)

    def test_training_fit_metadata_cannot_be_moved_into_future(self):
        spec=candidates_for_round(0)[-1]
        saved,artifact,_=c.measure(self.directory,self.declaration,"BOOM500","M1",spec,self.frame)
        self.saved=saved
        artifact=self.rewrite_saved(lambda x:x["fit"].__setitem__("trained_until_epoch",self.declaration["split"]["validation_end"]))
        with self.assertRaisesRegex(r.ReplayMismatch,"training-only fit"):
            r.replay_measurement(self.directory,artifact,self.declaration)

    def test_campaign_limit_deduplication_and_actual_matched_counts(self):
        second=candidates_for_round(0)[1]
        _,artifact2,_=c.measure(self.directory,self.declaration,"BOOM500","M1",second,self.frame)
        atomic_json(self.directory/"cycles"/"01.json",{"number":1,"status":"RUNNING", "results":[
            {"artifact":self.artifact},{"artifact":self.artifact},{"artifact":artifact2}]})
        report=r.replay_campaign(self.directory,self.state,self.declaration,limit=1)
        self.assertEqual(report["available_unique_measurements"],2)
        self.assertEqual(report["duplicate_artifacts_skipped"],1)
        self.assertEqual(report["matched_measurements"],1)
        self.assertFalse(report["all_available_replayed"])
        full=r.replay_campaign(self.directory,self.state,self.declaration)
        self.assertEqual(full["matched_measurements"],2)
        self.assertTrue(full["all_available_replayed"])
        empty=r.replay_campaign(self.directory,self.state,self.declaration,limit=0)
        self.assertEqual(empty["status"],"NOT_RUN")
        self.assertEqual(empty["matched_measurements"],0)

    def test_legacy_fingerprint_absence_is_disclosed_not_claimed_matched(self):
        artifact=self.rewrite_saved(lambda x:x.pop("signal_fingerprint"))
        report=r.replay_measurement(self.directory,artifact,self.declaration)
        self.assertFalse(report["full_signal_fingerprint_matched"])
        self.assertIn("legacy",report["signal_check_scope"])

    def test_artifact_path_cannot_escape_campaign(self):
        artifact={"path":"../elsewhere.json.gz","sha256":"0"*64}
        with self.assertRaisesRegex(ValueError,"inside its campaign"):
            r.replay_measurement(self.directory,artifact,self.declaration)


if __name__=="__main__": unittest.main()
