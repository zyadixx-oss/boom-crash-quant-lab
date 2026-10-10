#!/usr/bin/env python3
"""Independent first-fold ridge and original-target check during the full run."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_hybrid_event_study import (Audit, FLAGS, FOLDER, array, clock_rows,
    decode_symbol, epoch, independent_ridge, independent_timed, sha)


def main():
    audit = Audit()
    d = json.loads(audit.pin(FOLDER / "declaration.json").read_text())
    spec = d["specification"]; symbol, fold = "BOOM600", "fit40"
    audit.pin(__file__); audit.pin("scripts/verify_hybrid_event_study.py")
    model = json.loads(audit.pin(FOLDER / f"{symbol}_{fold}_model.json").read_text())
    frames = [pd.read_csv(audit.pin(f"data/hybrid_event_regions_20261008/{symbol}_{fold}_{v}_features.csv.gz"), float_precision="round_trip")
              for v in ("RAW44", "HYBRID44")]
    raw, hybrid = frames
    labels = pd.read_csv(audit.pin(FOLDER / f"{symbol}_{fold}_training_labels.csv.gz"), float_precision="round_trip")
    start, end = spec["folds"][fold]["train"]
    positions, issues = clock_rows(raw, hybrid, start, end)
    array(audit, pd.DatetimeIndex(pd.to_datetime(labels.signal_time, utc=True)).as_unit("ns").asi8, issues.asi8, "training clock")
    times, prices = decode_symbol(audit, d["sources"]["symbols"][symbol])
    busy, target = epoch(start), []
    for pos, issue in zip(positions, issues, strict=True):
        signal = {"signal_time": issue, "atr": raw.atr.iloc[pos], "side": 1, "variant": "CLOCK_SPIKE"}
        path, busy, status = independent_timed(times, prices, signal, spec["timed_config"], epoch(start), epoch(end), busy)
        target.append(np.nan if path is None or path["censored"] else path["net_R"])
    array(audit, labels.net_R, target, "independent native targets", close=True)
    completed = labels.completed.to_numpy(bool)
    audit.check(np.array_equal(completed, np.isfinite(target)), "completed target population")
    for arm, features in (("RAW44", raw), ("TRANSFORMED44", hybrid)):
        oracle = independent_ridge(features.loc[positions[completed], spec["feature_names"]], labels.net_R.to_numpy(float)[completed])
        for field in ("means", "std", "coefs", "intercept", "threshold"):
            array(audit, model["models"][arm][field], oracle[field], "independent " + arm + field, close=True)
    for path, fingerprint in tuple(audit.pins.items()): audit.equal(sha(ROOT / path), fingerprint, "final stable pin")
    output = FOLDER / "independent_precheck.json"
    report = {"stage": "independent_hybrid_firstfold_precheck", "run_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not audit.errors, "checks": audit.checks, "errors": audit.errors, "input_sha256": audit.pins,
        "training_rows": len(labels), "completed": int(completed.sum()), "models_checked": 2,
        "declaration_sha256": sha(FOLDER / "declaration.json"), "safety": dict.fromkeys(FLAGS, False), "QUALIFIED": False}
    with output.open("x") as f: f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with output.with_suffix(".sha256").open("x") as f: f.write(sha(output) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "errors", "completed")}))
    return int(bool(audit.errors))


if __name__ == "__main__": sys.exit(main())
