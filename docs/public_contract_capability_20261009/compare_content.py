#!/usr/bin/env python3
"""Descriptive comparison of two saved public captures; never connects."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def messages(folder):
    return [json.loads(json.loads(line)["text"], parse_float=Decimal)
            for line in (folder / "received_text_messages.jsonl").read_text().splitlines()]


def compare():
    first, second = messages(HERE / "capture01"), messages(HERE / "capture02")
    catalogues = [{r["echo_req"]["contracts_for"]: r["contracts_for"]["available"]
                   for r in rows if "contracts_for" in r} for rows in (first, second)]
    offset_rows = []
    for symbol in ("BOOM500", "CRASH500", "BOOM600", "CRASH600", "R_100"):
        arrays = [[r["growth_rate_barrier_offsets"] for r in cat[symbol]
                   if r["contract_type"] == "ACCU"] for cat in catalogues]
        if any(len(rows) != 1 for rows in arrays):
            raise ValueError("Expected one saved accumulator record per symbol/capture")
        offset_rows.append({"symbol": symbol, "capture01": [str(v) for v in arrays[0][0]],
                            "capture02": [str(v) for v in arrays[1][0]],
                            "equal_at_these_two_observations_only": arrays[0] == arrays[1]})
    quotes = []
    for response in second:
        if "proposal" not in response:
            continue
        obj, request = response["proposal"], response["echo_req"]
        values = {key: str(obj[key]) if obj.get(key) is not None else None
                  for key in ("ask_price", "commission", "multiplier", "spot", "spot_time", "payout")}
        quotes.append({"symbol": request["underlying_symbol"],
                       "contract_type": request["contract_type"], "observed_values": values})
    assessment_path = HERE.parent / "stable_profit_assessment_20261009/assessment.json"
    assessment = json.loads(assessment_path.read_text())
    models = [r for r in assessment["rows"] if r["role"] == "model_or_rule"]
    model_positive = sum(r["necessary_conditions"]["strict_net_PF_gt_1"]
                         and r["necessary_conditions"]["strict_mean_net_R_gt_0"] for r in models)
    paths = [HERE / name / file for name in ("capture01", "capture02")
             for file in ("received_text_messages.jsonl", "result.json", "declaration.json")]
    paths += [assessment_path]
    return {"scope": "saved descriptive contents; no predictive or economic recomputation",
            "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths},
            "offset_comparison": offset_rows, "capture02_proposals": quotes,
            "prior_assessment_model_or_rule_rows": len(models),
            "prior_assessment_strict_positive_model_or_rule_rows": model_positive,
            "prior_assessment_necessary_conjunction_passed": assessment["summary"]["necessary_conjunction_passed"],
            "predictive_tested": False, "profit_tested": False,
            "actual_fills_verified": False, "historical_cfd_costs_verified": False,
            "permanently_constant_offsets_proven": False,
            "safety": dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"saved_quotes": len(result["capture02_proposals"]),
                      "two_snapshot_offset_equalities": sum(r["equal_at_these_two_observations_only"]
                                                            for r in result["offset_comparison"]),
                      "model_point_positive": result["prior_assessment_strict_positive_model_or_rule_rows"],
                      "necessary_conjunction_passed": result["prior_assessment_necessary_conjunction_passed"]}))
