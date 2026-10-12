"""Boundary and semantic tests for the descriptive validation analyzer."""
import csv
import io
import json
import math
from pathlib import Path
import unittest

import analyse as a


def raw(family="crt", round_index=0, metrics=None):
    actual_round = round_index & ~2 if family in a.INACTIVE_THRESHOLD else round_index
    actual = a.spec_for(family, actual_round)
    attempted = a.spec_for(family, round_index)
    metrics = metrics or {"completed_trades": 10, "profit_factor": 1.2, "net_profit": round_index - 10.0, "max_drawdown": .1}
    if family in a.INACTIVE_THRESHOLD:
        metrics = {**metrics, "net_profit": actual_round - 10.0}
    values = {"cycle": str(round_index + 1), "symbol": "BOOM500", "timeframe": "M5", "family": family,
              "strategy_id": actual["id"], "attempted_strategy_id": attempted["id"], "status": "TESTED",
              "cache_hit": str(actual_round != round_index), "settings_json": json.dumps(actual),
              "artifact": f"artifacts/{actual['scientific_id']}.json.gz", "metrics_partition": "validation"}
    values.update({key: "" if value is None else str(value) for key, value in metrics.items()})
    return values


def group(family="crt", skip=None):
    rows = [a.parse_row(raw(family, number)) for number in range(32) if number != skip]
    return a.analyse_group(f"BOOM500/M5/{family}", rows)


class Tests(unittest.TestCase):
    def test_actual_frozen_id_examples(self):
        self.assertEqual(a.spec_for("crt", 0)["id"], "crt.r00.5c605e892bb5")
        self.assertEqual(a.spec_for("ict_liquidity_mss", 0)["scientific_id"], "2833c7cb3e9401d513ab16277c9ca79716cb46b3247edf659d580302b270aaa7")

    def test_five_dimensional_factorial_not_2d(self):
        specs = [a.spec_for("crt", number) for number in range(32)]
        self.assertEqual(len({tuple(spec[k] for k in a.AXES) for spec in specs}), 32)
        self.assertEqual([sorted({spec[k] for spec in specs}) for k in a.AXES], [list(x) for x in a.AXES.values()])
        summary, points, science, edges = group()
        self.assertEqual((len(points), len(science), len(edges)), (32, 32, 80))
        self.assertEqual(summary["effective_dimensions"], 5)

    def test_inactive_threshold_aliases_and_edge_duplicates(self):
        summary, points, science, edges = group("trend")
        self.assertEqual((summary["measured_attempts"], summary["cache_alias_attempts"]), (16, 16))
        self.assertEqual(len(science), 16)
        self.assertEqual(summary["edge_status_counts"], {"EFFECTIVE_NEIGHBOR": 32, "DUPLICATE_EFFECTIVE_EDGE": 32, "INACTIVE_AXIS_ALIAS": 16})
        threshold = next(row for row in summary["axes"] if row["axis"] == "threshold")
        self.assertFalse(threshold["axis_active"])
        self.assertEqual(threshold["metric_high_minus_low"]["net_profit"]["points"], 0)

    def test_missing_point_stays_missing_and_not_zero(self):
        summary, points, science, edges = group(skip=31)
        self.assertEqual(summary["missing_attempts"], 1)
        self.assertEqual(summary["missing_scientific_points"], 1)
        self.assertEqual(points[31]["coverage_status"], "MISSING")
        self.assertIsNone(points[31]["net_profit"])
        self.assertEqual(sum(x["comparison_status"] == "MISSING_ENDPOINT" for x in edges), 5)

    def test_null_pf_not_zero_or_infinity(self):
        data = raw(metrics={"completed_trades": 0, "profit_factor": None, "net_profit": 0., "max_drawdown": 0.})
        row = a.parse_row(data)
        self.assertIsNone(row["metrics"]["profit_factor"])
        self.assertEqual(row["metrics"]["net_profit"], 0)
        d = a.distribution([None, 0., -2., 4.])
        self.assertEqual((d["null"], d["zero"], d["positive"], d["negative"], d["median"]), (1, 1, 1, 1, 0.))
        self.assertEqual(a.distribution([None, None])["min"], None)

    def test_neighbor_high_minus_low(self):
        summary, points, science, edges = group()
        lookback = next(row for row in summary["axes"] if row["axis"] == "lookback")
        self.assertEqual(lookback["metric_high_minus_low"]["net_profit"]["min"], 1.)
        hold = next(row for row in summary["axes"] if row["axis"] == "max_hold_bars")
        self.assertEqual(hold["metric_high_minus_low"]["net_profit"]["max"], 16.)
        for edge in edges:
            low, high = a.spec_for("crt", edge["low_round"]), a.spec_for("crt", edge["high_round"])
            self.assertEqual([key for key in a.AXES if low[key] != high[key]], [edge["axis"]])

    def test_final_fields_never_accessed(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key in a.FORBIDDEN_COLUMNS:
                    raise AssertionError("Forbidden final/ranking column read")
                return super().__getitem__(key)
        first = raw()
        first.update({key: object() for key in a.FORBIDDEN_COLUMNS})
        self.assertEqual(a.parse_row(Poison(first)), a.parse_row(raw()))

    def test_duplicate_attempt_rejected(self):
        row = a.parse_row(raw())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            a.analyse_group("BOOM500/M5/crt", [row, row])

    def test_alias_conflict_rejected(self):
        rows = [a.parse_row(raw("trend", number)) for number in range(32)]
        rows[2]["metrics"]["net_profit"] = 999.
        with self.assertRaisesRegex(ValueError, "Alias conflicts"):
            a.analyse_group("BOOM500/M5/trend", rows)

    def test_orphan_alias_rejected(self):
        with self.assertRaisesRegex(ValueError, "measured cache origin"):
            a.analyse_group("BOOM500/M5/trend", [a.parse_row(raw("trend", 2))])

    def test_active_parameter_alias_rejected(self):
        row = raw("crt", 2)
        spec = a.spec_for("crt", 0)
        row.update(settings_json=json.dumps(spec), strategy_id=spec["id"], cache_hit="True")
        with self.assertRaisesRegex(ValueError, "effective scientific parameter"):
            a.parse_row(row)

    def test_bad_metrics_and_bool_spelling_rejected(self):
        for field, value in [("net_profit", "NaN"), ("profit_factor", "inf"), ("max_drawdown", "-0.1"),
                             ("completed_trades", "True"), ("completed_trades", "1.0"), ("cache_hit", "false"),
                             ("metrics_partition", "final"), ("cycle", "33")]:
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, TypeError)):
                row = raw(); row[field] = value; a.parse_row(row)

    def test_insufficient_data_preserved(self):
        row = raw(metrics=dict.fromkeys(a.METRICS)); row["status"] = "INSUFFICIENT_DATA"
        parsed = a.parse_row(row)
        self.assertTrue(all(value is None for value in parsed["metrics"].values()))
        row["net_profit"] = "0.0"
        with self.assertRaisesRegex(ValueError, "invented"):
            a.parse_row(row)

    def test_settings_types_and_ids_rejected(self):
        for field, value in [("trailing_stop", 0), ("atr_period", 14.0), ("round_index", False), ("lookback", 99), ("threshold", 1.0)]:
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                row = raw(); spec = json.loads(row["settings_json"]); spec[field] = value
                row["settings_json"] = json.dumps(spec); a.parse_row(row)

    def test_csv_roundtrip_preserves_none_and_empty_fields(self):
        data = raw(metrics={"completed_trades": 0, "profit_factor": None, "net_profit": 0., "max_drawdown": 0.})
        stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(data)); writer.writeheader(); writer.writerow(data)
        stream.seek(0)
        self.assertIsNone(a.parse_row(next(csv.DictReader(stream)))["metrics"]["profit_factor"])

    def test_only_operational_output_directory_allowed(self):
        with self.assertRaisesRegex(ValueError, "Outputs must remain"):
            a.run(Path("unused-input.csv"), a.ROOT / "data/single_frame_campaign_20261011")


if __name__ == "__main__":
    unittest.main()
