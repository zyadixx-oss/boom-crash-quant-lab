"""Check pinned saved-statistic pointers and necessary conditions, without imports.

This is not a new quote-path audit, full qualification implementation or proof
about every strategy. It reads only the pinned saved JSON files and source.
"""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
ASSESSMENT_SHA = "b684b1b7d5af33bd16bc9584225f8991b25ce0432414da03d1962c5b673ab17c"
SOURCE_SHA = "c890ff5437882960cc5a4e6f7eaa30c7d8ad0fed75824de0ff5e26a0a8be0923"
FLAGS = {"LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"}


def main():
    checks = 0

    def require(value):
        nonlocal checks
        checks += 1
        if not value:
            raise ValueError("Saved-pointer review mismatch at check " + str(checks))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def at(document, pointer):
        value = document
        for token in pointer.split("/")[1:]:
            key = token.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value

    require(sha(FOLDER / "assessment.json") == ASSESSMENT_SHA)
    require(sha(FOLDER / "screen_saved_results.py") == SOURCE_SHA)
    assessment = json.loads((FOLDER / "assessment.json").read_text())
    require(assessment["screen_source_sha256"] == SOURCE_SHA)
    require(set(assessment["safety"]) == FLAGS)
    require(all(value is False for value in assessment["safety"].values()))
    require(assessment["QUALIFIED"] is False and assessment["fresh_out_of_sample"] is False)
    require(assessment["uncertainty_and_development_requirements_waived"] is False)
    sources = {}
    for source in assessment["sources"]:
        require(sha(ROOT / source["path"]) == source["sha256"])
        document = json.loads((ROOT / source["path"]).read_text())
        require(document["safety"] == assessment["safety"])
        sources[source["path"]] = document
    require(len(sources) == 12)
    identities = set()
    roles, source_counts = Counter(), Counter()
    positive = large = large_days = unknown_pf = passed = 0
    role_summary = {role: Counter() for role in ("model_or_rule", "clock_control")}
    for row in assessment["rows"]:
        identity = row["source_path"], row["row_pointer"]
        require(identity not in identities)
        identities.add(identity)
        source_counts[row["source_path"]] += 1
        roles[row["role"]] += 1
        original = at(sources[row["source_path"]], row["metrics_pointer"])
        require(row["point"] == {key: original.get(key) for key in row["point"]})
        pf, mean = original["profit_factor"], original["mean_net_R"]
        pf_positive = pf is not None and Decimal(str(pf)) > Decimal(1)
        mean_positive = mean is not None and Decimal(str(mean)) > Decimal(0)
        conditions = {
            "strict_net_PF_gt_1": pf_positive,
            "strict_mean_net_R_gt_0": mean_positive,
            "completed_ge_1000": original["completed"] >= 1000,
            "active_days_ge_60": original["active_days"] >= 60,
        }
        require(conditions == row["necessary_conditions"])
        require(row["necessary_conjunction_passed"] is all(conditions.values()))
        require(row["QUALIFIED"] is False and row["prior_gate_results_unchanged"] is True)
        require(row["development"]["gate_reimplemented"] is False)
        positive += pf_positive and mean_positive
        large += conditions["completed_ge_1000"]
        large_days += conditions["completed_ge_1000"] and conditions["active_days_ge_60"]
        unknown_pf += pf is None
        passed += all(conditions.values())
        summary = role_summary[row["role"]]
        summary.update({"rows": 1, "undefined_PF": int(pf is None),
                        "positive_point": int(pf_positive and mean_positive),
                        "completed_ge1000": int(conditions["completed_ge_1000"]),
                        "completed_ge1000_days_ge60": int(conditions["completed_ge_1000"] and conditions["active_days_ge_60"]),
                        "necessary_passes": int(all(conditions.values()))})
    require(len(identities) == 132)
    require(dict(roles) == {"model_or_rule": 96, "clock_control": 36})
    require(dict(source_counts) == {s["path"]: s["included_stored_rows"] for s in assessment["sources"] if s["included_stored_rows"]})
    summary = assessment["summary"]
    for key, value in {"undefined_PF_rows": unknown_pf, "strict_positive_point_rows": positive,
                       "rows_with_ge1000_completed": large,
                       "rows_with_ge1000_completed_and_ge60_days": large_days,
                       "necessary_conjunction_passed": passed}.items():
        require(summary[key] == value)
    diagnostics = assessment["nonzero_cost_positive_diagnostics"]
    require(len(diagnostics) == 4)
    for row in diagnostics:
        source = sources[row["source_path"]]
        original = at(source, row["json_pointer"])
        require(original == row["full_saved_sensitivity"])
        for key in ("completed", "active_days", "profit_factor", "mean_net_R", "round_trip_cost_atr",
                    "fill_mode", "entry_delay_minutes", "selection_score"):
            require(original[key] == row[key])
        require(at(source, row["development_eligible_pointer"]) is False)
        require(row["excluded_from_primary_screen"] is True and row["QUALIFIED"] is False)
        require(original["selection_score"] < 0)
        require(row["positive_weekly_uncertainty_evidence_saved"] is False)
    print(json.dumps({"status": "PASS", "checks": checks, "sources": len(sources),
        "stored_rows": len(identities), "roles": dict(roles),
        "role_summaries": {key: dict(value) for key, value in role_summary.items()},
        "nonzero_cost_diagnostics": len(diagnostics), "necessary_passes": passed,
        "assessment_sha256": ASSESSMENT_SHA, "screen_source_sha256": SOURCE_SHA,
        "review_source_sha256": sha(Path(__file__)),
        "scope": "pinned source pointers, copied statistics, strict Decimal point/sample predicates and diagnostic identity only",
        "original_quote_paths_or_full_gates_reaudited": False,
        "independent_agent_final_review_completed": False,
        "universal_profit_impossibility_proven": False,
        "QUALIFIED": False, "safety": assessment["safety"]}, indent=2))


if __name__ == "__main__":
    main()
