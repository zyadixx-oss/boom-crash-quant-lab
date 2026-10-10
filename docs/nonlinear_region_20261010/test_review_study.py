"""Synthetic independent audit checks; never reads historical quote data."""
from copy import deepcopy
import hashlib
import sys
import unittest

import numpy as np
import pandas as pd

import review_study as M


def model_fixture(split=False):
    x = np.zeros((1000, 2))
    y = np.full(1000, .25)
    if split:
        x[500:, 0] = 1.
        y[:500], y[500:] = -.25, .75
    cuts = M.expected_boundaries(x)
    score = np.full(1000, float(y.mean()))
    trees = []
    for _ in range(100):
        residual = y - score
        total = float(residual.sum())
        root = {"count": 1000, "residual_sum": total, "value": total / 1020.}
        if split:
            children = []
            for indices in (np.arange(500), np.arange(500, 1000)):
                s = float(residual[indices].sum())
                children.append({"count": 500, "residual_sum": s, "value": s / 520.})
            gain = sum(c["residual_sum"] * (c["residual_sum"] / 520.) for c in children) - total * (total / 1020.)
            root.update(split_feature=0, split_bin=0, gain=gain, left=children[0], right=children[1])
            score[:500] += .05 * children[0]["value"]
            score[500:] += .05 * children[1]["value"]
        else:
            score += .05 * root["value"]
        trees.append(root)
    q = float(np.quantile(score, .75))
    model = {"version": 1, "feature_names": ["a", "b"], "parameters": deepcopy(M.PARAMETERS),
             "defaults": deepcopy(M.PARAMETERS), "fit_rows": 1000,
             "initial_prediction": float(y.mean()), "bin_boundaries": cuts, "trees": trees,
             "training_score_quantile": q, "threshold": max(0., q), **M.PROVENANCE}
    model["integrity_sha256"] = M.model_digest(model)
    return x, y, model


def gate_fixture():
    index = {}
    for family in M.FAMILIES:
        for part in M.PARTS:
            metrics = {"completed": 1200, "active_days": 100, "censored": 0, "invalid_uncensored": 0,
                       "mean_net_R": .02, "profit_factor": 1.01, "selection_score": .001,
                       "closed_trade_max_drawdown": .01, "equity_ruin": False}
            row = {"symbol": "BOOM600", "variant": family, "partition": part,
                   "metrics": metrics, "replay_audit": {"unknown": 0},
                   "day_inference": {"mean_net_R_ci95": [.001, .1], "baseline_difference_ci95": [.001, .1]},
                   "weekly_inference": {"weekly_mean_net_R_ci95": [.001, .1], "weekly_difference_ci95": [.001, .1]},
                   "profit_factor_inference": {"day_profit_factor_ci95": [1.001, 1.2], "weekly_profit_factor_ci95": [1.001, 1.2]},
                   "holm_p": .01, "thirds": [{"metrics": {"completed": 400, "mean_net_R": .02}} for _ in range(3)],
                   "double_cost_metrics": {"mean_net_R": .001}}
            if family in M.NEW:
                row["target_comparisons"] = {ref: {"day": deepcopy(row["day_inference"]),
                                                  "week": deepcopy(row["weekly_inference"])}
                                              for ref in M.REFERENCES[family][1:]}
            index[("BOOM600", family, part)] = row
    return index[("BOOM600", M.NEW[1], "later180")], index


def state_fixture():
    x, y, model = model_fixture(True)
    issues = pd.date_range("2026-01-01", periods=1000, freq="30min", tz="UTC")
    labels = pd.DataFrame({"signal_time": issues, "net_R": y, "completed": True,
                          "known_nonfill": False, "unknown_path": False, "exposure_skipped": False,
                          "status": "completed", "planned_end": issues + pd.Timedelta(minutes=30, seconds=1)})
    raw = pd.DataFrame(x, columns=["a", "b"]); hybrid = raw.copy()
    target = M.R.target_hash(issues, y)
    matrix = hashlib.sha256(target.encode()); matrix.update(np.asarray(x, dtype=">f8").tobytes())
    models = {family: deepcopy(model) for family in M.NEW}
    end = issues[-1] + pd.Timedelta(minutes=31)
    meta = {"version": 1, "symbol": "BOOM600", "native_side": 1, "algorithm": "app.research.nonlinear_signal.fit_histogram_boost",
            "parameters": deepcopy(M.PARAMETERS), "standardization": False, "feature_names": ["a", "b"],
            "training_start": issues[0].isoformat(), "training_end": end.isoformat(),
            "completed": 1000, "known_nonfill": 0, "unknown_path": 0, "exposure_skipped": 0,
            "opportunities": 1000, "quantile": .75, "safety": M.R.safety(),
            "target": "conditional_completed_original_quote_CLOCK_region_net_R",
            "training_is_independent_trade_sample": False, "label_selection_is_conditional_on_CLOCK_fill": True,
            "target_sha256": target, "matrix_sha256": {k: matrix.hexdigest() for k in M.NEW},
            "model_sha256": {k: M.R.fingerprint(v) for k, v in models.items()}}
    meta["metadata_sha256"] = M.R.fingerprint(meta)
    state = {"models": models, "model_metadata": meta, "status": "FITTED", "error": None,
             "completed": 1000, "known_nonfill": 0, "unknown_path": 0, "exposure_skipped": 0,
             "training_dispositions": {"completed": 1000}, "safety": M.R.safety(), "QUALIFIED": False}
    return [state, labels, raw, hybrid, np.arange(1000), ["a", "b"], "BOOM600", [issues[0], end]]


class ModelChecks(unittest.TestCase):
    def test_synthetic_production_fit_checked_by_independent_arithmetic(self):
        # Differential synthetic fixture only; the auditor never imports this.
        sys.path.insert(0, str(M.ROOT / "backend"))
        from app.research.nonlinear_signal import fit_histogram_boost
        random = np.random.default_rng(121)
        x = random.normal(size=(1200, 4))
        y = .1 + .5 * ((x[:, 0] > 0) != (x[:, 1] > 0)) + .1 * x[:, 2]
        names = ["a", "b", "c", "d"]
        model = fit_histogram_boost(x, y, names)
        audit = M.Audit(); M.check_boost(audit, x, y, model, names)
        self.assertFalse(audit.errors, audit.errors)

    def test_entire_model_state_target_matrix_and_counts(self):
        args = state_fixture(); audit = M.Audit()
        self.assertEqual(set(M.check_model(audit, *args)), set(M.NEW))
        self.assertFalse(audit.errors, audit.errors)

    def test_model_state_tampering_detected_after_metadata_rehash(self):
        for field in ("target_sha256", "matrix_sha256", "model_sha256"):
            args = state_fixture(); meta = args[0]["model_metadata"]
            if field == "target_sha256": meta[field] = "0" * 64
            else: meta[field][M.NEW[0]] = "0" * 64
            meta["metadata_sha256"] = M.R.fingerprint({k: v for k, v in meta.items() if k != "metadata_sha256"})
            audit = M.Audit(); M.check_model(audit, *args)
            with self.subTest(field=field): self.assertTrue(audit.errors)

    def test_constant_training_and_stage_scores(self):
        x, y, model = model_fixture()
        audit = M.Audit()
        scores = M.check_boost(audit, x, y, model, ["a", "b"])
        self.assertFalse(audit.errors, audit.errors)
        np.testing.assert_array_equal(scores, y)

    def test_routed_residual_statistics_for_nonconstant_tree(self):
        x, y, model = model_fixture(True)
        before = deepcopy(model); audit = M.Audit()
        scores = M.check_boost(audit, x, y, model, ["a", "b"])
        self.assertFalse(audit.errors, audit.errors)
        self.assertTrue(np.all(scores[:500] < scores[500:]))
        self.assertEqual(model, before)

    def test_boundary_equality_routes_right_and_tails_are_not_refitted(self):
        _, _, model = model_fixture(True)
        values = np.array([[-1e8, 0], [.49999, 0], [.5, 0], [1e8, 0]])
        scores = M.predict(values, model, ["a", "b"])
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores[2], scores[3])
        self.assertLess(scores[1], scores[2])

    def test_training_only_cuts_constant_low_cardinality_and_quantiles(self):
        x = np.c_[np.zeros(32), np.arange(32) % 2, np.arange(32)]
        cuts = M.expected_boundaries(x)
        self.assertEqual(cuts[0], [])
        self.assertEqual(cuts[1], [.5])
        np.testing.assert_array_equal(cuts[2], np.quantile(x[:, 2], np.arange(1, 16) / 16))

    def test_declared_cut_change_even_rehashed_is_detected(self):
        x, y, model = model_fixture(True); model["bin_boundaries"][0][0] = .4
        model["integrity_sha256"] = M.model_digest(model)
        audit = M.Audit(); M.check_boost(audit, x, y, model, ["a", "b"])
        self.assertTrue(audit.errors)

    def test_node_arithmetic_tampering_even_rehashed_is_detected(self):
        for field, value in (("value", .2), ("residual_sum", 20.), ("count", 900)):
            with self.subTest(field=field):
                x, y, model = model_fixture(); model["trees"][0][field] = value
                model["integrity_sha256"] = M.model_digest(model)
                audit = M.Audit(); M.check_boost(audit, x, y, model, ["a", "b"])
                self.assertTrue(audit.errors)

    def test_quantile_threshold_tampering_is_detected(self):
        for field in ("threshold", "training_score_quantile"):
            with self.subTest(field=field):
                x, y, model = model_fixture(); model[field] = .3
                model["integrity_sha256"] = M.model_digest(model)
                audit = M.Audit(); M.check_boost(audit, x, y, model, ["a", "b"])
                self.assertTrue(audit.errors)

    def test_parameter_provenance_hash_order_and_tree_count_rejected(self):
        for case in ("parameter", "provenance", "hash", "order", "tree_count", "boolean_version"):
            with self.subTest(case=case):
                x, y, model = model_fixture()
                if case == "parameter": model["parameters"]["learning_rate"] = .1
                elif case == "provenance": model["features_clipped"] = True
                elif case == "hash": model["initial_prediction"] += 1
                elif case == "order": model["feature_names"].reverse()
                elif case == "tree_count": model["trees"].pop()
                else: model["version"] = True
                if case != "hash": model["integrity_sha256"] = M.model_digest(model)
                with self.assertRaises(ValueError): M.check_boost(M.Audit(), x, y, model, ["a", "b"])

    def test_nonfinite_training_target_and_small_sample_rejected(self):
        x, y, model = model_fixture()
        for xx, yy in ((x[:-1], y[:-1]), (x, np.full(1000, np.nan)), (x * np.nan, y)):
            with self.assertRaises(ValueError): M.check_boost(M.Audit(), xx, yy, model, ["a", "b"])

    def test_empty_prediction_keeps_schema(self):
        _, _, model = model_fixture()
        self.assertEqual(M.predict(np.zeros((0, 2)), model, ["a", "b"]).shape, (0,))
        with self.assertRaises(ValueError): M.predict(np.zeros((0, 1)), model, ["a", "b"])

    def test_boolean_node_and_invalid_split_rejected(self):
        for key, value in (("count", True), ("split_feature", True), ("split_bin", 99)):
            x, y, model = model_fixture(True)
            model["trees"][0][key] = value; model["integrity_sha256"] = M.model_digest(model)
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.check_boost(M.Audit(), x, y, model, ["a", "b"])


class GateChecks(unittest.TestCase):
    def test_positive_pf_below_one_point_one_can_pass(self):
        row, index = gate_fixture()
        self.assertEqual(M.gate_outcome(row, index)[:2], (True, True))

    def test_strict_pf_boundary_unknown_and_nonfinite_rejected(self):
        for value in (1., .999, None, float("nan"), float("inf"), True):
            row, index = gate_fixture(); row["metrics"]["profit_factor"] = value
            with self.subTest(value=value): self.assertFalse(M.gate_outcome(row, index)[1])

    def test_nonpositive_mean_sample_and_day_shortfalls_rejected(self):
        for key, value in (("mean_net_R", 0.), ("mean_net_R", -.01), ("completed", 999), ("active_days", 59)):
            row, index = gate_fixture(); row["metrics"][key] = value
            with self.subTest(key=key, value=value): self.assertFalse(M.gate_outcome(row, index)[1])

    def test_every_heldout_reference_unknown_vetoes(self):
        for ref in M.REFERENCES[M.NEW[1]]:
            row, index = gate_fixture(); index[("BOOM600", ref, "later180")]["replay_audit"]["unknown"] = 1
            with self.subTest(ref=ref): self.assertFalse(M.gate_outcome(row, index)[1])

    def test_development_unknown_and_negative_fold_vetoes(self):
        for ref in M.REFERENCES[M.NEW[1]]:
            row, index = gate_fixture(); index[("BOOM600", ref, "wf2")]["replay_audit"]["unknown"] = 1
            self.assertEqual(M.gate_outcome(row, index)[:2], (False, False))
        row, index = gate_fixture(); index[("BOOM600", M.NEW[1], "wf1")]["metrics"]["mean_net_R"] = 0
        self.assertEqual(M.gate_outcome(row, index)[:2], (False, False))

    def test_reference_confidence_interval_vetoes_independent_of_p(self):
        for ref in M.REFERENCES[M.NEW[1]][1:]:
            for level, key in (("day", "mean_net_R_ci95"), ("day", "baseline_difference_ci95"),
                               ("week", "weekly_mean_net_R_ci95"), ("week", "weekly_difference_ci95")):
                row, index = gate_fixture(); row["target_comparisons"][ref][level][key] = [-.001, .1]
                with self.subTest(ref=ref, key=key): self.assertFalse(M.gate_outcome(row, index)[1])

    def test_weekly_pf_ci_holm_drawdown_thirds_and_double_cost_vetoes(self):
        for change in ("week", "holm", "drawdown", "third", "cost", "ruin"):
            row, index = gate_fixture()
            if change == "week": row["profit_factor_inference"]["weekly_profit_factor_ci95"] = [.99, 1.1]
            elif change == "holm": row["holm_p"] = .05
            elif change == "drawdown": row["metrics"]["closed_trade_max_drawdown"] = .101
            elif change == "third": row["thirds"][0]["metrics"]["completed"] = 199
            elif change == "cost": row["double_cost_metrics"]["mean_net_R"] = 0
            else: row["metrics"]["equity_ruin"] = True
            with self.subTest(change=change): self.assertFalse(M.gate_outcome(row, index)[1])

    def test_undefined_reversed_boolean_confidence_intervals_rejected(self):
        for ci in (None, [None, .1], [True, 2], [.2, .1], [float("nan"), .1], [0., .1]):
            with self.subTest(ci=ci): self.assertFalse(M.ci_positive(ci))

    def test_missing_required_reference_is_error(self):
        row, index = gate_fixture(); row["target_comparisons"].pop("HYBRID_REGION")
        with self.assertRaises(KeyError): M.gate_outcome(row, index)


class ArtifactAndMetricChecks(unittest.TestCase):
    def test_each_chronological_third_retains_inherited_31minute_purge(self):
        issues = pd.to_datetime(["2026-01-01T00:29Z", "2026-01-01T00:40Z",
                                 "2026-01-01T01:00Z", "2026-01-01T02:00Z"])
        ledger = pd.DataFrame({"signal_time": issues.astype(str),
            "entry_time": (issues + pd.Timedelta(seconds=62)).astype(str),
            "exit_time": (issues + pd.Timedelta(minutes=10)).astype(str),
            "net_R": [1., 100., -.5, .25], "censored": False, "holding_minutes": 8.})
        start, end = pd.Timestamp("2026-01-01T00:00Z"), pd.Timestamp("2026-01-01T03:00Z")
        row = {"start": start, "end_exclusive": end, "metrics": M.secondary_metrics(ledger, start, end), "thirds": []}
        # 00:29+31min is exactly the bound and remains; 00:40 is purged even
        # though its actual early exit is before the first third ends.
        for i, positions in enumerate(([0], [2], [3])):
            left, right = start + pd.Timedelta(hours=i), start + pd.Timedelta(hours=i + 1)
            row["thirds"].append({"start": left, "end": right,
                "metrics": M.secondary_metrics(ledger.iloc[positions], left, right)})
        audit = M.Audit(); M.check_secondary(audit, row, ledger)
        self.assertFalse(audit.errors, audit.errors)

    def test_type_sensitive_identity_and_reference_fields(self):
        self.assertFalse(M.exact({"x": True}, {"x": 1}))
        row = {"symbol": "BOOM600", "variant": "RAW_REGION", "partition": "wf1",
               "metrics": {"profit_factor": .8}, "qualification": {}, "holm_p": .9,
               "required_reference_checks": {}, "target_comparisons": {}}
        result = M.reference_copy(row)
        self.assertNotIn("qualification", result); self.assertEqual(result["metrics"], row["metrics"])
        self.assertEqual(result["reference_origin"]["result_sha256"], M.PARENT_PINS["results.json"])
        self.assertIn("qualification", row)

    def test_full_point_statistics_include_day_clusters_and_drawdown(self):
        ledger = pd.DataFrame({"signal_time": ["2026-01-01T00:00Z", "2026-01-02T00:00Z", "2026-01-03T00:00Z"],
            "entry_time": ["2026-01-01T00:02Z", "2026-01-02T00:02Z", "2026-01-03T00:02Z"],
            "exit_time": ["2026-01-01T00:10Z", "2026-01-02T00:10Z", "2026-01-03T00:10Z"],
            "net_R": [1., -.5, 0.], "censored": [False, False, False], "holding_minutes": [8., 8., 8.]})
        m = M.secondary_metrics(ledger, "2026-01-01T00:00Z", "2026-01-04T00:00Z")
        self.assertEqual(m["completed"], 3); self.assertEqual(m["active_days"], 3)
        self.assertEqual(m["profit_factor"], 2.); self.assertAlmostEqual(m["mean_net_R"], 1 / 6)
        self.assertAlmostEqual(m["closed_trade_max_drawdown"], .00125)
        self.assertAlmostEqual(m["closed_trade_return"], 1.0025 * .99875 - 1)
        self.assertLess(m["selection_score"], m["mean_net_R"])

    def test_empty_ledger_metrics_stay_undefined(self):
        ledger = pd.DataFrame({k: pd.Series(dtype=t) for k, t in {
            "signal_time": str, "entry_time": str, "exit_time": str, "net_R": float,
            "censored": bool, "holding_minutes": float}.items()})
        m = M.secondary_metrics(ledger, "2026-01-01T00:00Z", "2026-01-04T00:00Z")
        self.assertIsNone(m["mean_net_R"]); self.assertIsNone(m["profit_factor"])
        self.assertEqual(m["completed"], 0); self.assertFalse(m["equity_ruin"])


if __name__ == "__main__":
    unittest.main()
