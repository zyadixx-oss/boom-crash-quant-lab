#!/usr/bin/env python3
"""POST-HOC explanation of target order; local saved inputs only, no selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.research.spike_hunter import assert_offline, load_m1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in ("true", "false"):
        return str(value).lower() == "true"
    raise ValueError(f"Invalid ledger boolean: {value!r}")


def inspect_symbol(symbol: str, study: dict, directory: Path) -> dict:
    cohort = study["symbols"][symbol]["cohorts"]["external_older_replication"]
    primary = cohort["models"]["primary"]
    cfg = primary["config"]
    expected = {"stop_atr": 1.0, "target_atr": 2.0, "max_hold_minutes": 15,
                "entry_delay_minutes": 1, "fill_mode": "adverse_extreme",
                "same_bar_policy": "stop_first"}
    if any(cfg[k] != v for k, v in expected.items()) or primary["variant"] != "SR_ALIGNMENT":
        raise ValueError("This diagnostic requires the frozen primary SR_ALIGNMENT configuration")
    path = ROOT / "data/spike_payoff_external" / f"{symbol.lower()}_m1_180d_clean.csv"
    m1, audit = load_m1(path)
    if audit["sha256"] != cohort["audit"]["sha256"]:
        raise ValueError("External data differs from the evaluated source")
    ledger_path = directory / f"{symbol}_external_older_replication_primary_trades.csv"
    ledger = pd.read_csv(ledger_path, float_precision="round_trip")
    for col in ("signal_time", "entry_time", "exit_time", "planned_end"):
        ledger[col] = pd.to_datetime(ledger[col], utc=True)
    if len(ledger) != primary["audit"]["filled"]:
        raise ValueError("Saved ledger does not match evaluated trade count")
    side = 1 if symbol.startswith("BOOM") else -1
    counts = {k: 0 for k in (
        "complete_original_horizons", "incomplete_original_horizons",
        "unrestricted_target_touches", "complete_stop_horizons",
        "stop_target_first_touched_same_minute", "stop_target_first_touched_strictly_later_minute",
        "stop_target_touched_in_any_later_minute", "stop_target_touched_anytime_in_original_horizon",
        "stop_target_never_touched", "same_minute_ambiguous_stop_target",
        "same_minute_target_after_opening_stop_gap", "ledger_ambiguous_stop_exits")}
    later_minutes = []
    for row in ledger.itertuples(index=False):
        if row.entry_time != row.signal_time + pd.Timedelta(minutes=1):
            raise ValueError("Unexpected entry delay in saved ledger")
        if row.planned_end != row.entry_time + pd.Timedelta(minutes=15):
            raise ValueError("Unexpected saved planned horizon")
        if boolean(row.censored):
            raise ValueError("External primary ledger contains censored positions")
        if row.reason not in ("tp", "sl", "time"):
            raise ValueError("Unexpected primary exit reason")
        if row.reason == "sl":
            counts["ledger_ambiguous_stop_exits"] += int(boolean(row.ambiguous))
        times = pd.date_range(row.entry_time, periods=15, freq="min")
        path_data = m1.reindex(times)
        if not np.isfinite(path_data[["open", "high", "low", "close"]].to_numpy()).all():
            counts["incomplete_original_horizons"] += 1
            continue
        counts["complete_original_horizons"] += 1
        if float(path_data.open.iloc[0]) != row.entry:
            raise ValueError("Ledger entry differs from the exact source M1 open")
        target = row.entry + side * cfg["target_atr"] * row.atr
        stop = row.entry - side * cfg["stop_atr"] * row.atr
        hits = ((path_data.high.to_numpy() >= target) if side == 1
                else (path_data.low.to_numpy() <= target))
        positions = np.flatnonzero(hits)
        any_target = len(positions) > 0
        counts["unrestricted_target_touches"] += int(any_target)
        if row.reason == "tp" and not any_target:
            raise ValueError("Recorded TP is absent from the original horizon")
        if row.reason == "time" and any_target:
            raise ValueError("Recorded timeout contains a target touch")
        if row.reason != "sl":
            continue
        counts["complete_stop_horizons"] += 1
        # Primary adverse-extreme SL timestamps are at the stop minute's close.
        stop_minute = row.exit_time - pd.Timedelta(minutes=1)
        stop_i = int((stop_minute - row.entry_time) / pd.Timedelta(minutes=1))
        if stop_minute not in times or not 0 <= stop_i < 15:
            raise ValueError("Stop minute is outside its original planned horizon")
        if any_target and int(positions[0]) < stop_i:
            raise ValueError("Target touched in an earlier minute than recorded stop")
        same_minute = bool(hits[stop_i])
        opening_stop = side * (float(path_data.open.iloc[stop_i]) - stop) <= 0
        if boolean(row.ambiguous) != (same_minute and not opening_stop):
            raise ValueError("Saved ambiguity differs from source stop-minute ordering")
        counts["same_minute_ambiguous_stop_target"] += int(same_minute and not opening_stop)
        counts["same_minute_target_after_opening_stop_gap"] += int(same_minute and opening_stop)
        counts["stop_target_first_touched_same_minute"] += int(same_minute)
        later = any_target and int(positions[0]) > stop_i
        counts["stop_target_first_touched_strictly_later_minute"] += int(later)
        counts["stop_target_touched_in_any_later_minute"] += int(bool(hits[stop_i + 1:].any()))
        counts["stop_target_touched_anytime_in_original_horizon"] += int(any_target)
        counts["stop_target_never_touched"] += int(not any_target)
        if later:
            later_minutes.append(int(positions[0]) - stop_i)
    reasons = {reason: int((ledger.reason == reason).sum()) for reason in ("sl", "tp", "time")}
    n, complete = len(ledger), counts["complete_original_horizons"]
    if not counts["incomplete_original_horizons"]:
        assert counts["unrestricted_target_touches"] == reasons["tp"] + counts["stop_target_touched_anytime_in_original_horizon"]
    return {
        "model": primary["id"], "config": cfg, "data_sha256": audit["sha256"],
        "ledger_sha256": digest(ledger_path), "trades": n, "exit_counts": reasons,
        "exit_fractions": {k: probability(v, n) for k, v in reasons.items()},
        **counts,
        "primary_target_before_stop_fraction": probability(reasons["tp"], n),
        "unrestricted_target_touch_fraction_complete_horizons": probability(counts["unrestricted_target_touches"], complete),
        "stops_with_later_minute_first_target_fraction": probability(counts["stop_target_first_touched_strictly_later_minute"], counts["complete_stop_horizons"]),
        "stops_with_any_original_horizon_target_fraction": probability(counts["stop_target_touched_anytime_in_original_horizon"], counts["complete_stop_horizons"]),
        "median_minutes_from_stop_minute_to_strictly_later_first_target_minute": float(np.median(later_minutes)) if later_minutes else None,
    }


def markdown(result: dict) -> str:
    lines = ["# تشخيص لاحق لترتيب الهدف والوقف — POST-HOC", "",
             "قراءة تفسيرية بعد الاطلاع على النتائج، وليست فرضية جديدة أو تعديلًا للاختيار. استُخدمت سجلات SR_ALIGNMENT الخارجية الأصلية: وقف1ATR، هدف2ATR، دخول متأخر دقيقة ومدة15 دقيقة من الدخول.", "",
             "| المؤشر | الصفقات | وقف | هدف قبل الوقف وفق السجل | انتهاء المدة | لمس الهدف خلال كامل المدة بغض النظر عن الوقف |",
             "|---|---:|---:|---:|---:|---:|"]
    for symbol, s in result["symbols"].items():
        e = s["exit_counts"]
        lines.append(f"| {symbol} | {s['trades']} | {e['sl']} | {e['tp']} ({100*s['primary_target_before_stop_fraction']:.2f}%) | {e['time']} | {s['unrestricted_target_touches']} ({100*s['unrestricted_target_touch_fraction_complete_horizons']:.2f}%) |")
    lines += ["", "| المؤشر | وصل الهدف لأول مرة في دقيقة لاحقة بعد الوقف | لمس الهدف في دقيقة الوقف نفسها | منها ترتيب مجهول داخل الدقيقة | منها وقف عند الافتتاح قبل اللمس | لم يصل الهدف بعد الوقف حتى نهاية المدة الأصلية |", "|---|---:|---:|---:|---:|---:|"]
    for symbol, s in result["symbols"].items():
        lines.append(f"| {symbol} | {s['stop_target_first_touched_strictly_later_minute']} | {s['stop_target_first_touched_same_minute']} | {s['same_minute_ambiguous_stop_target']} | {s['same_minute_target_after_opening_stop_gap']} | {s['stop_target_never_touched']} |")
    lines += ["", "- احتساب اللمس خلال كامل15 دقيقة يتجاهل خروج الصفقة السابق عند الوقف؛ لذلك يمكنه إظهار احتمال حركة أعلى مع بقاء عائد الصفقة خاسرًا. الوصول بعد الخروج ليس ربحًا محققًا.",
              "- اللمس في دقيقة الوقف نفسها مفصول عن الوصول في دقيقة لاحقة. إذا كان الافتتاح داخل الحاجزين، لا تخبرنا OHLC لدقيقة واحدة أيهما لُمس أولًا؛ السجل الأساسي يضع الوقف أولًا. الوقف عند افتتاح يتجاوز الحاجز يسبق حركة تلك الدقيقة.",
              "- هذا القياس يبدأ من سعر الدخول المتأخر ويحافظ على ATR الخاص بالإشارة. يختلف عن تعريف دراسة الحركة السابقة الذي بدأ من إغلاق الإشارة؛ لا يمثل إعادة تقدير مباشرة لذلك lift.",
              "- لا يُمدد الأفق بعد النهاية الأصلية، ولا يُغيّر الوقف أو الهدف، ولا تُحسب عوائد جديدة أو تُنشأ استراتيجية. الأعلام الأربعة false. البيانات عامة، وليست تنفيذًا أو ربح حساب.",
              "", "إعادة الإنتاج: `.venv/bin/python scripts/diagnose_spike_target_order.py`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "docs/spike_payoff_20261004")
    args = parser.parse_args()
    safety = assert_offline()
    selection_path = args.study_dir / "selection.json"
    selection_hash = digest(selection_path)
    if selection_hash != (args.study_dir / "selection.sha256").read_text().strip():
        raise ValueError("Frozen selection hash mismatch")
    frozen = json.loads(selection_path.read_text())
    for source, expected in frozen["code_hashes"].items():
        if digest(ROOT / source) != expected:
            raise ValueError("Frozen research source changed")
    results_path = args.study_dir / "results.json"
    study = json.loads(results_path.read_text())
    if study["selection_sha256"] != selection_hash:
        raise ValueError("Evaluation and frozen selection differ")
    result = {
        "label": "POST-HOC_DIAGNOSTIC_NOT_SELECTION_OR_CONFIRMATORY_TEST",
        "created_utc": datetime.now(timezone.utc).isoformat(), "safety": safety,
        "selection_sha256": selection_hash, "results_sha256": digest(results_path),
        "diagnostic_script_sha256": digest(Path(__file__)),
        "method": "Original delayed entry and ATR; exact original 15 M1 candles; no horizon extension. Stop-minute OHLC cannot order intraminute barrier touches unless opening already crossed stop.",
        "symbols": {symbol: inspect_symbol(symbol, study, args.study_dir) for symbol in ("BOOM500", "CRASH500")},
    }
    (args.study_dir / "target_order_diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.study_dir / "target_order_diagnostic.ar.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in x.items() if not k.endswith("sha256") and k != "config"} for s, x in result["symbols"].items()}, indent=2))


if __name__ == "__main__":
    main()
