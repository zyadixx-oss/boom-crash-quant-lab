#!/usr/bin/env python3
"""Posthoc covariate/score audit of frozen transfer models; never fit or trade.

The z distances use the saved training scaler. The 3/5 standard-deviation
cutoffs are descriptive only and introduce no signal filter or probability.
No outcome labels, payoff ledgers, recalibration or hyperparameter search enter
this diagnostic. It cannot promote a model to a trading recommendation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.research.learned_signal import predict_ridge
from app.research.spike_hunter import assert_offline
from scripts.run_spike_learned_study import hashes, prepare, TRANSFER


QUANTILES = (0., .01, .05, .25, .50, .75, .95, .99, 1.)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray) -> dict:
    return {f"q{q:g}": float(value) for q, value in zip(
        QUANTILES, np.quantile(values, QUANTILES), strict=True,
    )}


def diagnose(rows, model: dict) -> dict:
    names = model["feature_names"]
    values = rows.loc[:, names].to_numpy(float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Transfer clock features must be nonempty and finite")
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["std"], dtype=float)
    coefficients = np.asarray(model["coefs"], dtype=float)
    standardized = (values - means) / scales
    scores = predict_ridge(rows.loc[:, names], model)
    selected = (scores >= model["threshold"]) & (scores > 0)
    z_mean = standardized.mean(axis=0)
    z_std = standardized.std(axis=0, ddof=0)
    contributions = z_mean * coefficients
    per_feature = [{
        "feature": name, "training_mean": float(means[i]), "training_std": float(scales[i]),
        "external_mean": float(values[:, i].mean()),
        "external_std": float(values[:, i].std(ddof=0)),
        "standardized_external_mean": float(z_mean[i]),
        "standardized_external_std": float(z_std[i]),
        "abs_z_above3_fraction": float((np.abs(standardized[:, i]) > 3).mean()),
        "abs_z_above5_fraction": float((np.abs(standardized[:, i]) > 5).mean()),
        "coefficient": float(coefficients[i]),
        "mean_score_contribution": float(contributions[i]),
    } for i, name in enumerate(names)]
    reconstruction = float(model["intercept"] + contributions.sum())
    return {
        "feature_clock_rows": len(rows), "training_completed_labels": model["training_completed_labels"],
        "training_fit_rows": model["fit_rows"], "training_intercept": model["intercept"],
        "frozen_threshold": model["threshold"],
        "score_kind": model["score_kind"],
        "score_mean": float(scores.mean()), "score_quantiles": quantiles(scores),
        "issuance_rows_before_partition_purge": int(selected.sum()),
        "issuance_fraction_before_partition_purge": float(selected.mean()),
        "positive_score_fraction": float((scores > 0).mean()),
        "all_features": per_feature,
        "largest_absolute_mean_score_contributions": sorted(
            per_feature, key=lambda item: -abs(item["mean_score_contribution"]),
        )[:5],
        "rows_with_any_abs_z_above3_fraction": float((np.abs(standardized) > 3).any(axis=1).mean()),
        "rows_with_any_abs_z_above5_fraction": float((np.abs(standardized) > 5).any(axis=1).mean()),
        "all_cells_abs_z_above3_fraction": float((np.abs(standardized) > 3).mean()),
        "all_cells_abs_z_above5_fraction": float((np.abs(standardized) > 5).mean()),
        "row_max_abs_z_quantiles": quantiles(np.abs(standardized).max(axis=1)),
        "mean_score_reconstructed_from_contributions": reconstruction,
        "mean_score_identity_residual": float(scores.mean() - reconstruction),
        "interpretation": (
            "Cross-symbol standardized-feature distances diagnose covariate shift. "
            "Shifted score signs reflect the frozen linear model's extrapolation, "
            "not calibrated probabilities or evidence for new trading signals. "
            "Mean contributions are an algebraic score decomposition, not causal "
            "attribution of future returns."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_learned_20261005")
    args = parser.parse_args()
    output = args.study / "transfer_shift.json"
    if output.exists():
        raise ValueError("Refusing to overwrite the saved posthoc diagnostic")
    safety = assert_offline()
    selected_path = args.study / "selection.json"
    expected_selection_hash = (args.study / "selection.sha256").read_text().strip()
    if digest(selected_path) != expected_selection_hash:
        raise ValueError("Frozen learned selection changed")
    selection = json.loads(selected_path.read_text())
    current_hashes = hashes()
    if selection["code_hashes"] != current_hashes:
        raise ValueError("Frozen learned protocol/source changed")
    # Confirm all declared acquisition hashes before deriving any new features.
    for sources in selection["sources"].values():
        for source in sources.values():
            if digest(ROOT / source["path"]) != source["sha256"]:
                raise ValueError(f"Frozen acquisition changed: {source['path']}")
    result = {
        "stage": "posthoc_frozen_transfer_covariate_and_score_diagnostic",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": expected_selection_hash, "code_hashes": current_hashes,
        "diagnostic_script_sha256": digest(Path(__file__).resolve()), "safety": safety,
        "model_refitted": False, "threshold_refitted": False, "outcomes_used": False,
        "features_recalibrated": False, "new_hyperparameters": False,
        "no_promotion_from_diagnostic": True,
        "population": "Every finite feature-valid M5 close at UTC00/30 before execution/purge filtering",
        "model_support": "Cross-symbol score interpretation remains unsupported by prospective evidence",
        "standard_deviation_cutoffs_are_descriptive_only": [3, 5],
        "symbols": {},
    }
    for training_symbol, sources in selection["sources"].items():
        transfer_symbol = TRANSFER[training_symbol]
        _, rows, names, audit, _ = prepare(ROOT / sources["transfer"]["path"], transfer_symbol)
        models = {}
        for mode, candidate in selection["symbols"][training_symbol]["models"].items():
            model = candidate["final_model"]
            if names != model["feature_names"]:
                raise ValueError("Transfer feature schema differs from frozen training")
            report = diagnose(rows, model)
            report.update(development_eligible=candidate["development_eligible"],
                          matrix_and_target_sha256=model["matrix_and_target_sha256"])
            models[mode] = report
            top = report["largest_absolute_mean_score_contributions"][0]
            print(f"{training_symbol}->{transfer_symbol}/{mode}: "
                  f"mean_score={report['score_mean']:.6f}, "
                  f"issued={report['issuance_rows_before_partition_purge']}/{len(rows)}, "
                  f"top_shift={top['feature']} zmean={top['standardized_external_mean']:.3f} "
                  f"contribution={top['mean_score_contribution']:.6f}", flush=True)
        result["symbols"][training_symbol] = {
            "transfer_symbol": transfer_symbol, "source": sources["transfer"],
            "feature_audit": audit, "models": models,
        }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"DESCRIPTIVE_TRANSFER_DIAGNOSTIC_SAVED {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
