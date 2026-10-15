#!/usr/bin/env python3
"""POST-HOC descriptive ATR matching; never changes the original research gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.research.spike_hunter import (
    VARIANTS, assert_offline, bootstrap_context, eligible, features, interval,
    load_m1, outcomes, partitions, signals,
)


def fit_atr_edges(normalized_atr: np.ndarray, development: np.ndarray) -> np.ndarray:
    """Edges use development exposure only; no labels enter their estimation."""
    values = np.asarray(normalized_atr, float)[np.asarray(development, bool)]
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Development ATR exposure must be nonempty, finite and positive")
    return np.unique(np.quantile(values, np.arange(1, 10) / 10))


def assign_atr_bins(normalized_atr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(normalized_atr, float)
    assigned = np.searchsorted(edges, values, side="right")
    assigned[~np.isfinite(values) | (values <= 0)] = -1
    return assigned


def matched_estimate(baseline_n, baseline_k, signal_n, signal_k) -> dict:
    """Standardize baseline rates to each model's signal-bin exposure mixture."""
    bn, bk, sn, sk = [np.asarray(v, float) for v in (baseline_n, baseline_k, signal_n, signal_k)]
    if not (bn.shape == bk.shape == sn.shape == sk.shape):
        raise ValueError("Count arrays must share their bin dimensions")
    if any((v < 0).any() for v in (bn, bk, sn, sk)) or (bk > bn).any() or (sk > sn).any():
        raise ValueError("Invalid hit/exposure counts")
    total = sn.sum(axis=-1)
    missing = ((sn > 0) & (bn == 0)).any(axis=-1)
    available = (total > 0) & ~missing
    rates = np.divide(bk, bn, out=np.zeros_like(bk), where=bn > 0)
    precision = np.divide(sk.sum(axis=-1), total, out=np.full_like(total, np.nan), where=total > 0)
    baseline = np.divide((rates * sn).sum(axis=-1), total,
                         out=np.full_like(total, np.nan), where=available)
    lift = np.divide(precision, baseline, out=np.full_like(total, np.nan),
                     where=available & (baseline > 0))
    difference = np.where(available, precision - baseline, np.nan)
    return {"precision": precision, "matched_baseline": baseline,
            "matched_lift": lift, "difference": difference,
            "available": available, "missing_baseline_exposure": missing}


def day_bin_counts(indices: np.ndarray, bins: np.ndarray, ctx: dict, n_bins: int) -> np.ndarray:
    keys = ctx["ids"][indices] * n_bins + bins[indices]
    return np.bincount(keys, minlength=ctx["blocks"] * n_bins).reshape(ctx["blocks"], n_bins)


def finite_scalar(value):
    value = float(value)
    return value if np.isfinite(value) else None


def fmt(value, percentage=False):
    if value is None:
        return "—"
    return f"{100 * value:.2f}%" if percentage else f"{value:.3f}"


def fmt_ci(values, percentage=False):
    return "—" if values[0] is None else f"{fmt(values[0], percentage)}–{fmt(values[1], percentage)}"


def report(result: dict) -> str:
    lines = ["# تشخيص لاحق: مطابقة توزيع ATR", "",
             "**POST-HOC — فحص استكشافي اختير بعد الاطلاع على Lift أولي 1.412 لـSR_ALIGNMENT على Crash500.**",
             "لا يمثل اختبار تأكيد مستقلًا، ولا يعدّل البوابة الأصلية أو يرقّي أي نموذج. الربحية: **NOT TESTED**.", "",
             "التعريف كما في الدراسة الأصلية: وصول اتجاهي إلى 2×ATR من M5 خلال 15 دقيقة، بعد الإشارة.",
             "قُسم ATR/Close إلى أعشار من فرص أول 70% فقط، ثم جُمّدت الحدود لاختبار آخر 30%.",
             "الخط الأساسي المطابق = مجموع معدل نجاح كل فئة في الاختبار × وزن الفئة بين إشارات النموذج.",
             "Precision / الخط الأساسي المطابق يعطي Lift مطابقًا؛ الفئات التي تفتقد تعرضًا أساسيًا تصبح غير متاحة.",
             "CI95 يستخدم 9,999 سحبًا لأيام UTC نفسها، مع إعادة حساب المعدلات والأوزان لكل سحب. الفترات فردية واستكشافية.", "",
             "`LIVE_TRADING=false`، `READY_FOR_LIVE=false`، `LIVE_ALLOWED=false`، `OPENED_TRADES=false`.", ""]
    for symbol, item in result["symbols"].items():
        lines += [f"## {symbol}", "",
                  f"فرص نهائية: {item['final_opportunities']:,}؛ فئات فعلية: {item['effective_bins']}.",
                  f"وسيط ATR/Close للفرص النهائية: {100 * item['baseline_median_normalized_atr']:.5f}%.", "",
                  "| النموذج | نجاح/إشارات | Precision | Base الخام | Base المطابق | Lift الخام | Lift المطابق | CI95 المطابق | CI95 الفرق | ATR وسيط/الأساس |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in VARIANTS:
            row = item["models"][name]
            lines.append(f"| {name} | {row['hits']}/{row['signals']} | {fmt(row['precision'], True)} | {fmt(row['raw_base_rate'], True)} | {fmt(row['matched_base_rate'], True)} | {fmt(row['raw_lift'])} | {fmt(row['matched_lift'])} | {fmt_ci(row['block_matched_lift_ci95'])} | {fmt_ci(row['block_difference_ci95'], True)} | {fmt(row['median_normalized_atr_vs_baseline'])} |")
        lines += ["", "### الحالات محل الفحص", ""]
        for name in ("CRT", "SR_ALIGNMENT", "ATR_BB_COMPRESSION"):
            row = item["models"][name]
            fraction = row["descriptive_fraction_of_raw_gap_removed"]
            removal = f"؛ إزالة وصفية لـ{fmt(fraction, True)} من فرق Precision الخام" if fraction is not None else ""
            lines.append(f"- **{name}**: Lift الخام {fmt(row['raw_lift'])} → المطابق {fmt(row['matched_lift'])}، CI95 {fmt_ci(row['block_matched_lift_ci95'])}{removal}. وسيط ATR النسبي {fmt(row['median_normalized_atr_vs_baseline'])}× وسيط الأساس؛ حصة الإشارات في أدنى3 فئات {fmt(row['share_lowest_three_bins'], True)}.")
        lines += ["", "### معدلات الأساس حسب حدود التطوير المجمدة", "",
                  "| الفئة | الحد الأدنى ATR/Close | الحد الأعلى ATR/Close | فرص نهائية | Base نهائي |",
                  "|---:|---:|---:|---:|---:|"]
        for cell in item["bin_audit"]:
            lower = f"{100 * cell['lower_edge']:.5f}%" if cell['lower_edge'] is not None else "−∞"
            upper = f"{100 * cell['upper_edge']:.5f}%" if cell['upper_edge'] is not None else "+∞"
            lines.append(f"| {cell['bin'] + 1} | {lower} | {upper} | {cell['opportunities']} | {fmt(cell['base_rate'], True)} |")
        lines.append("")
    lines += ["## حدود التفسير", "",
              "- المطابقة تشخص تركيب ATR المقاس؛ لا تثبت أن ATR سبب الفارق، ولا تضبط كل خصائص المسار أو شكل الشموع.",
              "- حتى لو بقي Lift أعلى من1، الاختيار اللاحق والقياس على البيانات نفسها يمنعان اعتباره تأكيدًا جديدًا.",
              "- التقسيم العشري تقريبي، وقد يبقى اختلاف داخل الفئة؛ نحتاج بيانات جديدة قبل اختيار قواعد من هذا الفحص.",
              "- انخفاض ATR يقلل مسافة الوصول المطلقة للتعريف2×ATR، لذلك قد يرتفع Base عنده دون قدرة مستقلة على توقيت انفجار.",
              "- ملف نتائج الدراسة الأصلية وبوابتها لم يتغيرا. لا تداول أو محاكاة أرباح أو تكاليف في هذا التشخيص.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_hunter_20261004")
    parser.add_argument("--bootstrap", type=int, default=9999)
    args = parser.parse_args()
    safety = assert_offline()
    if args.bootstrap < 2000:
        parser.error("Use at least 2000 paired day bootstrap replicates")
    original = json.loads((args.output / "results.json").read_text())
    result = {"run_utc": datetime.now(timezone.utc).isoformat(), "status": "POST_HOC_DIAGNOSTIC",
              "selection_context": "Chosen after observing initial Crash500 SR_ALIGNMENT lift1.412",
              "not_independent_confirmation": True, "original_gate_unchanged": True,
              "profitability": "NOT TESTED", "safety": safety, "bootstrap_repeats": args.bootstrap,
              "definition": "2x signal-time M5 ATR14 in15min; normalizedATR=ATR/Close",
              "bin_fit": "Development0-70% eligible exposure only; decile edges frozen for final30%",
              "symbols": {}}
    flat = []
    for symbol in ("BOOM500", "CRASH500"):
        m1, audit = load_m1(ROOT / "data/spike_hunter" / f"{symbol.lower()}_m1_clean.csv")
        direction = "boom" if symbol.startswith("BOOM") else "crash"
        m5 = features(m1, direction)
        issued = signals(m5, direction)
        hit, _, valid15 = outcomes(m1, m5, direction, 2, 15)
        _, _, valid30 = outcomes(m1, m5, direction, 2, 30)
        part = partitions(m1)
        dev = eligible(m5, valid30, *part["development"])
        mask = eligible(m5, valid30, *part["final_test"])
        if (mask & ~valid15).any() or mask.sum() < 100:
            raise RuntimeError("Invalid shared final exposure")
        normalized = (m5.atr / m5.close).to_numpy()
        edges = fit_atr_edges(normalized, dev)
        bins, n_bins = assign_atr_bins(normalized, edges), len(edges) + 1
        if (bins[mask] < 0).any():
            raise RuntimeError("Eligible opportunity lacks causal ATR")
        ctx = bootstrap_context(m5, mask, *part["final_test"], args.bootstrap)
        selected_base = np.flatnonzero(mask)
        base_n = day_bin_counts(selected_base, bins, ctx, n_bins)
        base_k = day_bin_counts(np.flatnonzero(mask & hit), bins, ctx, n_bins)
        bn, bk = base_n.sum(axis=0), base_k.sum(axis=0)
        bootstrap_bn, bootstrap_bk = ctx["weights"] @ base_n, ctx["weights"] @ base_k
        baseline_median = float(np.median(normalized[mask]))
        item = {"audit": audit, "development_opportunities": int(dev.sum()),
                "final_opportunities": int(mask.sum()), "effective_bins": n_bins,
                "development_decile_edges": edges.tolist(), "day_blocks": ctx["evaluated_blocks"],
                "baseline_median_normalized_atr": baseline_median,
                "models": {}, "bin_audit": []}
        for b in range(n_bins):
            item["bin_audit"].append({"bin": b, "lower_edge": float(edges[b - 1]) if b else None,
                                      "upper_edge": float(edges[b]) if b < len(edges) else None,
                                      "opportunities": int(bn[b]), "hits": int(bk[b]),
                                      "base_rate": float(bk[b] / bn[b]) if bn[b] else None})
        result["symbols"][symbol] = item
        for name in VARIANTS:
            selected = issued[name][mask[issued[name]]]
            signal_n = day_bin_counts(selected, bins, ctx, n_bins)
            signal_k = day_bin_counts(selected[hit[selected]], bins, ctx, n_bins)
            sn, sk = signal_n.sum(axis=0), signal_k.sum(axis=0)
            observed = matched_estimate(bn, bk, sn, sk)
            boot = matched_estimate(bootstrap_bn, bootstrap_bk,
                                    ctx["weights"] @ signal_n, ctx["weights"] @ signal_k)
            precision, matched_base = finite_scalar(observed["precision"]), finite_scalar(observed["matched_baseline"])
            raw_base = float(bk.sum() / bn.sum())
            raw_gap = precision - raw_base if precision is not None else None
            row = {"symbol": symbol, "variant": name, "status": "AVAILABLE" if observed["available"] else "UNAVAILABLE",
                   "signals": int(sn.sum()), "hits": int(sk.sum()), "precision": precision,
                   "raw_base_rate": raw_base, "raw_lift": precision / raw_base if precision is not None and raw_base else None,
                   "matched_base_rate": matched_base, "matched_lift": finite_scalar(observed["matched_lift"]),
                   "difference": finite_scalar(observed["difference"]),
                   "block_precision_ci95": interval(boot["precision"]),
                   "block_matched_base_ci95": interval(boot["matched_baseline"]),
                   "block_matched_lift_ci95": interval(boot["matched_lift"]),
                   "block_difference_ci95": interval(boot["difference"]),
                   "bootstrap_available_replicates": int(boot["available"].sum()),
                   "missing_baseline_exposure": bool(observed["missing_baseline_exposure"]),
                   "signal_counts_by_bin": sn.astype(int).tolist(), "signal_hits_by_bin": sk.astype(int).tolist(),
                   "median_normalized_atr": float(np.median(normalized[selected])) if len(selected) else None,
                   "median_normalized_atr_vs_baseline": float(np.median(normalized[selected]) / baseline_median) if len(selected) else None,
                   "share_lowest_three_bins": float(sn[:3].sum() / sn.sum()) if sn.sum() else None,
                   "descriptive_fraction_of_raw_gap_removed": (matched_base - raw_base) / raw_gap if raw_gap is not None and raw_gap > 0 and matched_base is not None else None}
            reference = original["symbols"][symbol]["definitions"]["2.0ATR_15m"][name]["final_test"]
            if reference["signals"] != row["signals"] or reference["hits"] != row["hits"]:
                raise RuntimeError(f"Original opportunity definition drifted for {symbol}/{name}")
            item["models"][name] = row
            flat.append(row)
        print(f"{symbol}: 16 models matched; final opportunities={mask.sum():,}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "atr_matching_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    records = []
    for row in flat:
        record = {key: value for key, value in row.items() if not isinstance(value, list)}
        for key in ("block_precision_ci95", "block_matched_base_ci95", "block_matched_lift_ci95", "block_difference_ci95"):
            record[key + "_low"], record[key + "_high"] = row[key]
        records.append(record)
    pd.DataFrame(records).to_csv(args.output / "atr_matching_metrics.csv", index=False)
    (args.output / "atr_matching_REPORT.ar.md").write_text(report(result), encoding="utf-8")
    print("DONE: POST-HOC diagnostic only; original gate unchanged; all 4 live flags false.", flush=True)


if __name__ == "__main__":
    main()
