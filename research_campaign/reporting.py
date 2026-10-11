"""Render only recorded campaign evidence; absent final results stay null."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import tempfile

from .state import SAFETY, atomic_json, next_due, read_json, utc

LABELS = {
    "ict_liquidity_mss": "سحب السيولة وتحول الهيكل",
    "crt": "استعادة نطاق الشمعة CRT", "support_resistance": "الدعم والمقاومة",
    "false_break": "الاختراق الكاذب", "bollinger_squeeze": "ضغط بولينجر",
    "atr_compression_expansion": "ضغط وتوسع ATR", "rsi_momentum": "زخم RSI",
    "pre_spike": "فرضية ما قبل الانفجار", "post_spike": "متابعة ما بعد الانفجار",
    "mean_reversion": "العودة إلى المتوسط", "trend": "اتباع الاتجاه",
    "statistical_volatility": "الزخم المعدل بالتذبذب",
    "empirical_conditional_return": "حالات العائد الشرطي المتعلمة",
}


def metric_view(result, *, details=False):
    if not result:
        return None
    metrics = result["metrics"]
    value = {**metrics, "net_profit_unit": "وحدات حساب افتراضي؛ تكاليف وتنفيذ CFD غير موثقين",
             "uncertainty": result.get("uncertainty"),
             "reasons": result.get("rejection_reasons", []),
             "references": [
                 {"name": "الشراء والاحتفاظ" if name == "buy_and_hold" else "الاحتفاظ باتجاه القفزات",
                  **row, "status": "ASSUMED_NOT_VERIFIED", "reason": "مقارنة أسعار بتكلفة مفترضة؛ لا تثبت تنفيذ CFD"}
                 for name, row in result.get("reference", {}).items() if isinstance(row, dict)
             ]}
    if details:
        equity = result.get("equity", [])
        stride = max(1, len(equity) // 500)
        sample = equity[::stride]
        if equity and sample[-1] != equity[-1]:
            sample.append(equity[-1])
        value["equity"] = [{"time": utc(r["epoch"]), "value": r["equity"]} for r in sample]
        peak = metrics["initial_equity"]
        drawdown = []
        for r in equity:
            peak = max(peak, r["equity"])
            drawdown.append({"time": utc(r["epoch"]), "value": (peak - r["equity"]) / peak})
        value["drawdown"] = drawdown[::stride]
        if drawdown and value["drawdown"][-1] != drawdown[-1]:
            value["drawdown"].append(drawdown[-1])
        value["trades"] = [
            {**r, "id": f'{result["id"]}:{i}', "entry_time": utc(r["entry_epoch"]),
             "exit_time": utc(r["exit_epoch"]), "side": "BUY" if r.get("direction", 1) == 1 else "SELL",
             "entry_price": r.get("entry_price"), "exit_price": r.get("exit_price"),
             "net_profit": r["net_pnl"], "status": "COMPLETED"}
            for i, r in enumerate(result.get("trades", []))
        ]
        value["monthly"] = [{**r, "trades": r["completed_trades"]} for r in result.get("monthly", [])]
        value["dispositions"] = result.get("dispositions", [])
    return value


def publish_detail(public, artifact, measured):
    name = artifact["sha256"] + ".json"
    value = {"development": {"training": metric_view(measured.get("train")),
                              "validation": metric_view(measured.get("validation"), details=True)},
             "oos": metric_view(measured.get("final"), details=True)}
    atomic_json(public / "details" / name, value)
    return "/campaign/details/" + name


def _finals(directory, state):
    if not state.get("final_artifact"):
        return {}
    from .campaign import read_artifact
    result = read_artifact(directory, state["final_artifact"])
    return {r["selection_group"]: r for r in result["results"]}


def publish_snapshot(directory: Path, public: Path, state: dict):
    declaration = read_json(directory / "declaration.json")
    finals = _finals(directory, state)
    strategies = []
    for group, winner in state["winners"].items():
        final = finals.get(group)
        strategy = {
            "id": group + "/" + winner["strategy_id"], "symbol": winner["symbol"],
            "timeframe": winner["timeframe"], "name": LABELS.get(winner["family"], winner["family"]),
            "family": winner["family"], "settings": winner["spec"],
            "tested_at": final["measured_at"] if final else winner["tested_at"],
            "status": "QUALIFIED" if final and final["qualified"] else "REJECTED" if final else
                      "INSUFFICIENT_DATA" if winner["status"] == "INSUFFICIENT_DATA" else "NOT_TESTED",
            "reasons": final["rejection_reasons"] if final else ["FINAL_HOLDOUT_SEALED", "ACTUAL_CFD_COSTS_NOT_VERIFIED"],
            "development": {"training": winner.get("training_metrics"), "validation": winner.get("validation_metrics"),
                            "validation_score": winner.get("validation_score")},
            "oos": metric_view(final), "source": "Deriv public historical candles",
            "rank_score": final["metrics"]["net_return"] / max(final["metrics"]["max_drawdown"], .005) if final else None,
            "period_start": utc(declaration.get("splits", {}).get(winner["symbol"], declaration["split"])["start"]),
            "period_end": utc(declaration.get("splits", {}).get(winner["symbol"], declaration["split"])["end"]),
            "details_url": winner.get("details_url"),
        }
        if final:
            # Publish final details once; all full virtual ledgers stay in their source artifacts.
            detail_name = final["id"] + ".json"
            target = public / "details" / detail_name
            if not target.exists():
                from .campaign import read_artifact
                development = read_artifact(directory, winner["artifact"])
                atomic_json(target, {"development": {"training": metric_view(development.get("train")),
                                  "validation": metric_view(development.get("validation"), details=True)},
                                     "oos": metric_view(final, details=True)})
            strategy["details_url"] = "/campaign/details/" + detail_name
            strategy["oos"]["stability"] = [
                {"label": r["label"], "status": r["status"],
                 "value": r.get("metrics", {}).get("profit_factor"),
                 "reason": "فحص ثبات في بيانات التطوير بعد اختيار الإعدادات؛ ليس اختبارًا مستقلًا خارج العينة"}
                for r in final.get("development_walk_forward", [])]
        strategies.append(strategy)
    symbols = []
    for symbol, data in declaration["symbol_catalogue"].items():
        eligible = [r for r in strategies if r["symbol"] == symbol and r["status"] == "QUALIFIED"]
        best = max(eligible, key=lambda r: (r["oos"]["net_return"] /
                   max(r["oos"]["max_drawdown"], .005))) if eligible else None
        symbols.append({"symbol": symbol, "label": data.get("display_name", symbol),
                        "api_symbol": data.get("api_symbol"),
                        "status": "QUALIFIED" if best else "NO_QUALIFIED_STRATEGY" if finals else "RESEARCH_RUNNING",
                        "reason": "لا توجد استراتيجية مؤهلة لهذا المؤشر." if finals and not best else
                                  "الاختبار النهائي محجوب حتى اكتمال البحث." if not finals else None,
                        "best_strategy_id": best["id"] if best else None})
    next_epoch = next_due(state)
    quality_notes = []
    for symbol, row in declaration["datasets"].items():
        audit = row["study_quality"]
        quality_notes.append(f'{row.get("display_name", symbol)}: {row["requested_rows"]:,} شمعة M1، '
                             f'{audit["internal_missing_minutes"]} دقيقة مفقودة داخل الفترة، '
                             f'{audit["internal_invalid_rows"]} صف غير صالح داخل الفترة.')
    snapshot = {
        "schema_version": 1, "generated_at": utc(), "safety": dict(SAFETY),
        "campaign": {k: state.get(k) for k in ("id", "status", "started_at", "completed_cycles", "target_cycles",
                                             "current_cycle", "deadline_at", "failure", "completed_at")} | {
            "next_cycle_at": utc(next_epoch) if state["completed_cycles"] < state["target_cycles"] else None,
            "interval_minutes": 15, "final_accessed": state["final_accessed"]},
        "source": {"provider": "Deriv public API", "period_start": utc(declaration["split"]["start"]),
                   "period_end": utc(declaration["split"]["end"]), "updated_at": utc(declaration["split"]["end"]),
                   "cost_provenance": "تكلفة اختبار مفترضة 0.10 ATR؛ السبريد والانزلاق والرسوم التاريخية غير موثقة",
                   "contract_status": "CFD_MAPPING_NOT_VERIFIED", "data_notes": [
                       "كل استراتيجية تستخدم فريمًا واحدًا فقط.",
                       "تقسيم زمني 60% تدريب / 20% تحقق / 20% اختبار نهائي.",
                       "الأسعار تاريخية حقيقية؛ التنفيذ والأرباح بوحدات حساب افتراضي.",
                       "بعض هذا التاريخ مكشوف في أبحاث سابقة؛ لا يُعرض كاختبار مستقبلي جديد.",
                       "تحديثات السوق تحفظ منفصلة ولا تغير التقسيم المجمد.",
                       "فحوص Walk-Forward في التطوير تعتمد على اختيار الإعدادات ولا تمثل اختبارًا مستقلًا جديدًا.",
                   ] + quality_notes},
        "symbols": symbols, "strategies": strategies, "cycles": state["cycles"],
        "coverage": state["coverage"],
        "reproduction": {"commands": [".venv/bin/python -m research_campaign status",
                                      ".venv/bin/python -m research_campaign verify",
                                      ".venv/bin/python -m research_campaign replay",
                                      ".venv/bin/python -m research_campaign run"],
                         "report_url": "/campaign/latest.json", "search_csv_url": "/campaign/search.csv"},
    }
    atomic_json(public / "latest.json", snapshot)
    atomic_json(directory / "snapshot.json", snapshot)
    # All evaluated settings and rejected rows, not just the current winners.
    if state.get("current_cycle") is None:
        output = io.StringIO()
        columns = ["cycle", "symbol", "timeframe", "family", "strategy_id", "attempted_strategy_id",
                   "status", "cache_hit", "selection_disposition", "reasons", "settings_json", "validation_score",
                   "completed_trades", "profit_factor", "net_profit", "max_drawdown", "artifact"]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for cycle in state["cycles"]:
            record = read_json(directory / "cycles" / f'{cycle["number"]:02d}.json')
            for r in record["results"]:
                row = {c: r.get(c) for c in columns}
                row.update(cycle=cycle["number"], artifact=r["artifact"]["path"])
                group = f'{r["symbol"]}/{r["timeframe"]}/{r["family"]}'
                selected = state["winners"].get(group, {}).get("strategy_id") == r["strategy_id"]
                final = finals.get(group) if selected else None
                if r["status"] != "TESTED":
                    disposition, reasons = "INSUFFICIENT_DATA", ["INSUFFICIENT_CLOSED_TRAINING_BARS"]
                elif final:
                    disposition = "QUALIFIED" if final["qualified"] else "REJECTED_FINAL"
                    reasons = final["rejection_reasons"]
                elif selected:
                    disposition, reasons = "SELECTED_VALIDATION_CANDIDATE", ["FINAL_HOLDOUT_SEALED"]
                else:
                    disposition, reasons = "NOT_SELECTED", ["NOT_BEST_VALIDATION_SCORE_WITHIN_FAMILY"]
                row.update(selection_disposition=disposition, reasons=";".join(reasons),
                           settings_json=json.dumps(r["spec"], sort_keys=True, ensure_ascii=False))
                row.update({c: (r.get("validation_metrics") or {}).get(c) for c in (
                    "completed_trades", "profit_factor", "net_profit", "max_drawdown")})
                writer.writerow(row)
        public.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="search.", suffix=".tmp", dir=public)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(output.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, public / "search.csv")
        finally:
            temporary.unlink(missing_ok=True)
