#!/usr/bin/env python3
"""Frozen one-minute versus fifteen-minute ridge targets on known native600 history.

Both models are evaluated with the same one-minute execution policy. Scores
are uncalibrated continuous values, and all ledgers are hypothetical OHLC paths.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.learned_signal import fit_ridge, predict_ridge
from app.research.multiframe_signal import causal_multiframe_inputs, FEATURE_NAMES, FEATURE_FORMULAS
from app.research.payoff_timed import TimedExitConfig, replay_timed
from app.research.payoff_metrics import summarize, paired_inference
from app.research.learned_metrics import profit_factor_inference
from app.research.spike_hunter import assert_offline, load_m1, partitions
from scripts import run_spike_nonlinear_study as round6
from scripts.run_spike_native300_study import check_prepared
from scripts.run_spike_payoff_study import holm, window
from scripts.run_spike_timed_study import weekly_inference, tail_metrics, thirds
from scripts.run_spike_learned_study import development_eligible as inherited_eligible

SYMBOLS = ("BOOM600", "CRASH600")
MODES = ("SPIKE", "DRIFT")
FAMILIES = ("SHORT44", "LONG44")
TARGET_MINUTES = {"SHORT44": 1, "LONG44": 15}
WALK_FORWARD = (("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3"))
CFG = TimedExitConfig(stop_atr=2., max_hold_minutes=1)
TARGET_CONFIGS = {family: TimedExitConfig(**{**asdict(CFG), "max_hold_minutes": minutes})
                  for family, minutes in TARGET_MINUTES.items()}
PURGE, MIN_TRAIN, BOOTSTRAP, SEED = 31, 1000, 9999, 20261005
ROUND6 = ROOT / "docs/spike_nonlinear_20261005"
OUTPUT = ROOT / "docs/spike_short_target_20261006"
NEW_SOURCES = ("docs/SPIKE_SHORT_TARGET_PROTOCOL.md", "scripts/run_spike_short_target.py",
               "backend/tests/test_spike_short_target.py")
SOURCES = tuple(dict.fromkeys((*round6.SOURCES, *NEW_SOURCES)))
CONFIG = {"execution": asdict(CFG), "target_configs": {k: asdict(v) for k, v in TARGET_CONFIGS.items()},
          "target_names": {"SHORT44": "SHORT1", "LONG44": "LONG15"},
          "purge_minutes": PURGE, "clock_minutes": [0, 30], "ridge_penalty": .1,
          "training_quantile": .75, "minimum_common_training_labels": MIN_TRAIN,
          "bootstrap_repeats": BOOTSTRAP, "bootstrap_seed": SEED,
          "primary_holm_family": 4, "only_short_is_candidate": True,
          "shared_training_rows": "intersection_of_completed_finite_one_and_fifteen_minute_targets",
          "shared_scaler": "same_unclipped44_training_matrix_means_and_population_std",
          "unknown_outcomes_are_zero": False, "undefined_draws_block_bounded_inference": True}
HISTORY = {"known_prices": True, "known_fifteen_minute_payoffs": True, "price_oos": False,
           "new_label_holdout": False, "prospective": False,
           "adaptive_history": "round6_and_later_results_seen_before_short_target_declaration"}


def stamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("A valid timezone-aware timestamp is required")
    return value.tz_convert("UTC")


def canonical(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def json_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_for(label):
    if not isinstance(label, str) or not label or "\\" in label:
        raise ValueError("Artifact paths must be canonical relative POSIX strings")
    lexical = PurePosixPath(label)
    if lexical.is_absolute() or ".." in lexical.parts or str(lexical) != label:
        raise ValueError("Artifact path traversal or absolute aliases are forbidden")
    path = ROOT / label
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Artifact symlink escapes repository")
    return path


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(canonical(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_frozen(path, value):
    save(path, value)
    with Path(path).with_suffix(".sha256").open("x") as stream:
        stream.write(digest(path) + "\n")


def hashes():
    return {name: digest(path_for(name)) for name in SOURCES}


def offline_document(document):
    for flag in assert_offline():
        if document.get("safety", {}).get(flag) is not False:
            raise ValueError("All four saved safety values must be false")


def metadata_lineage():
    """Only hashes/manifests/prior saved evidence; no price or feature decoding."""
    assert_offline()
    selection, fingerprint = round6.frozen(ROUND6)
    declaration = json.loads((ROUND6 / "declaration.json").read_text())
    result = json.loads((ROUND6 / "results.json").read_text())
    audit = json.loads((ROUND6 / "independent_audit.json").read_text())
    for value in (selection, declaration, result, audit):
        offline_document(value)
    if (audit.get("status") != "PASS" or audit.get("pass") is not True or audit.get("errors") or
            audit.get("selection_sha256") != fingerprint or
            audit.get("results_sha256") != digest(ROUND6 / "results.json") or
            audit.get("script_sha256") != digest(path_for("scripts/verify_spike_nonlinear.py")) or
            result.get("selection_sha256") != fingerprint or result.get("adaptive_round") != 6):
        raise ValueError("Unchanged independently passing round6 evidence is required")
    pins = {name: digest(ROUND6 / name) for name in
            ("declaration.json", "declaration.sha256", "selection.json", "selection.sha256",
             "results.json", "independent_audit.json")}
    files = {str((ROUND6 / name).relative_to(ROOT)): value for name, value in pins.items()}
    files.update({"scripts/verify_spike_nonlinear.py": audit["script_sha256"], **audit["helper_sha256"]})
    if any(digest(path_for(name)) != value for name, value in files.items()):
        raise ValueError("Round6 verifier or inherited evidence changed")
    sources, intervals = {}, {}
    for symbol in SYMBOLS:
        pair = {"old": round6.source("data/spike_nonlinear600_old", symbol),
                "fresh": round6.source("data/spike_nonlinear600_recent", symbol)}
        if json_hash(pair) != json_hash(selection["sources"][symbol]):
            raise ValueError("Short-target source bytes differ from frozen round6")
        sources[symbol], intervals[symbol] = pair, round6.validate_intervals(pair["old"], pair["fresh"])
    return canonical({"round6_selection_sha256": fingerprint, "files": files,
                      "sources": sources, "intervals": intervals})


def output_path(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == ROOT.resolve():
        raise ValueError("Study output must be inside the repository")
    if any(output.is_relative_to((ROOT / name).resolve()) for name in ("data", "scripts", "backend", "frontend")):
        raise ValueError("Study state must not alias source, code or data directories")
    if output.is_relative_to(ROUND6.resolve()) or ROUND6.resolve().is_relative_to(output):
        raise ValueError("Study output aliases protected ancestor evidence")
    for ancestor in (ROOT / "docs").glob("spike_*"):
        if ancestor != output and (ancestor / "results.json").exists() and output.is_relative_to(ancestor.resolve()):
            raise ValueError("Study state aliases another immutable research round")
    return output


def guard(output, stage):
    assert_offline(); output = output_path(output)
    if stage not in ("declare", "develop", "evaluate"):
        raise ValueError("Unknown short-target stage")
    forbidden = ["results.json", "results.sha256", "metrics.csv", "evaluation_state.json"]
    if stage in ("declare", "develop"):
        forbidden += ["selection.json", "selection.sha256", "development_metrics.csv", "development_state.json"]
    if stage == "declare":
        forbidden += ["declaration.json", "declaration.sha256"]
        if any((output / name).exists() and any((output / name).iterdir()) for name in
               ("inputs", "matrices", "signals", "ledgers", "training_labels")):
            raise ValueError("Refusing predeclaration computed science artifact aliases")
    if any((output / name).exists() for name in forbidden):
        raise ValueError("Refusing frozen science/state/result overwrite")
    return output


def declare(output):
    output = guard(output, "declare")
    science, lineage = hashes(), metadata_lineage()
    for item in lineage["sources"].values():
        for source in item.values():
            if path_for(source["path"]).resolve().is_relative_to(output):
                raise ValueError("Output aliases an inherited source")
    document = {"stage": "short_target_premeasurement_declaration", "adaptive_round": 11,
                "run_utc": datetime.now(timezone.utc), "safety": assert_offline(),
                "config": CONFIG, "history": HISTORY, "science_code_hashes": science, "lineage": lineage,
                "symbols": SYMBOLS, "modes": MODES, "families": FAMILIES,
                "feature_names": FEATURE_NAMES, "feature_formulas": FEATURE_FORMULAS,
                "prices_decoded": False, "new_features_labels_models_or_ledgers_computed": False,
                "goal_achieved": False, "live_candidate": False,
                "user_target": {"profit_factor": 1.5, "completed_per_symbol_model": 1000,
                                "active_heldout_days": 60},
                "quote_decode_policy": "whole_saved_M1_decode_then_immediate_prefix_clip_before_features_or_targets",
                "model_scores": "continuous_uncalibrated_scores_not_probabilities"}
    if json_hash(science) != json_hash(hashes()) or json_hash(lineage) != json_hash(metadata_lineage()):
        raise ValueError("Sources/protocol changed during declaration")
    save_frozen(output / "declaration.json", document)
    print("SHORT_TARGET_DECLARED", digest(output / "declaration.json"), "metadata only", flush=True)
    return canonical(document)


def frozen_declaration(output):
    assert_offline(); output = output_path(output)
    path = output / "declaration.json"
    value = json.loads(path.read_text())
    if digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Frozen declaration changed")
    offline_document(value)
    exact = {"stage": "short_target_premeasurement_declaration", "adaptive_round": 11,
             "config": CONFIG, "history": HISTORY, "science_code_hashes": hashes(), "lineage": metadata_lineage(),
             "symbols": list(SYMBOLS), "modes": list(MODES), "families": list(FAMILIES),
             "feature_names": list(FEATURE_NAMES), "feature_formulas": FEATURE_FORMULAS,
             "prices_decoded": False, "new_features_labels_models_or_ledgers_computed": False,
             "goal_achieved": False, "live_candidate": False,
             "user_target": {"profit_factor": 1.5, "completed_per_symbol_model": 1000, "active_heldout_days": 60},
             "quote_decode_policy": "whole_saved_M1_decode_then_immediate_prefix_clip_before_features_or_targets",
             "model_scores": "continuous_uncalibrated_scores_not_probabilities"}
    if any(json_hash(value.get(key)) != json_hash(expected) for key, expected in exact.items()):
        raise ValueError("Frozen declaration semantics, history or lineage changed")
    return value


def prepared_frame(m1, symbol, end=None):
    """The optional cutoff is applied before any resampling or indicators."""
    if symbol not in SYMBOLS:
        raise ValueError("Only native600 symbols are declared")
    if end is not None:
        end = stamp(end)
        m1 = m1.loc[m1.index + pd.Timedelta(minutes=1) <= end].copy()
    if m1.empty:
        raise ValueError("No completed M1 prefix")
    m5, names = causal_multiframe_inputs(m1, "boom" if symbol.startswith("BOOM") else "crash")
    if names != list(FEATURE_NAMES) or len(names) != 44:
        raise ValueError("Exact closed44-input schema is required")
    issues = m5.index + pd.Timedelta(minutes=5)
    mask = m5.feature_valid & (issues.minute % 30 == 0) & (issues.second == 0)
    rows = m5.loc[mask].copy(); rows.index = issues[mask]; rows.index.name = "signal_time"
    if end is not None and (rows.index > end).any():
        raise ValueError("A feature row exceeds its completed prefix")
    audit = {"prefix_end": end, "last_completed_m1_close": m1.index[-1] + pd.Timedelta(minutes=1),
             "source_prefix_rows": len(m1), "causal_m5_grid_rows": len(m5), "eligible_clock_rows": len(rows),
             "frames": ["H4", "H1", "M15", "M5", "M1"], "feature_count": 44,
             "all_features_computed_after_prefix_clip": True}
    return m1, rows, audit


def prepare(value, symbol, end=None):
    m1, source_audit = load_m1(path_for(value["path"]))
    check_prepared(m1, source_audit, value)
    split = partitions(m1)
    split.update({"train40": (m1.index[0], split["wf1"][0]),
                  "train50": (m1.index[0], split["wf2"][0]),
                  "train60": (m1.index[0], split["wf3"][0])})
    prefix, rows, audit = prepared_frame(m1, symbol, end)
    return prefix, rows, {**source_audit, **audit}, split


def issued(rows, symbol, mode, model=None):
    if symbol not in SYMBOLS or mode not in MODES:
        raise ValueError("Undeclared symbol/direction")
    if not isinstance(rows.index, pd.DatetimeIndex) or rows.index.tz is None or not rows.index.is_unique or not rows.index.is_monotonic_increasing:
        raise ValueError("Opportunity timestamps must be sorted, unique and UTC-aware")
    if ((rows.index.minute % 30 != 0) | (rows.index.second != 0) | (rows.index.microsecond != 0)).any():
        raise ValueError("Only the shared closed UTC00/30 clock is permitted")
    scores, chosen = None, rows
    if model is not None:
        validate_model(model)
        if model["status"] != "TESTED":
            chosen = rows.iloc[:0]
        else:
            scores = predict_ridge(rows[list(FEATURE_NAMES)], model["estimator"], FEATURE_NAMES)
            mask = (scores >= model["estimator"]["threshold"]) & (scores > 0)
            scores, chosen = scores[mask], rows.loc[mask]
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    signals = [{"signal_time": time, "atr": float(row.atr), "side": side,
                "signal_close": float(row.close), "variant": f"{model['family'] if model else 'CLOCK'}_{mode}"}
               for time, row in chosen.iterrows()]
    if scores is not None:
        for signal, score in zip(signals, scores, strict=True):
            signal["score"] = float(score)
    return signals


def target_pair(m1, rows, symbol, mode, start, end):
    """00/30 spacing makes each individual target identical to one-open replay.

    The maximum planned target path is16minutes, strictly less than30minutes.
    Missing entries remain in the audit and are never manufactured into labels.
    """
    signals = issued(rows, symbol, mode)
    frames, audits = {}, {}
    for family in FAMILIES:
        frames[family], audits[family] = replay_timed(m1, signals, TARGET_CONFIGS[family], start, end,
                                                     purge_minutes=PURGE)
        if audits[family]["overlap_skipped"]:
            raise ValueError("Individual training targets unexpectedly overlap on declared00/30 clock")
        audits[family].update(training_targets_not_strategy_evidence=True,
                             target_horizon_minutes=TARGET_MINUTES[family], common_purge_minutes=PURGE)
    return frames, audits


def common_training(rows, targets, start, end):
    """Intersect complete targets, then require identical finite44 inputs."""
    cohorts = {}
    for family in FAMILIES:
        cohort = window(targets[family], start, end)
        if cohort.signal_time.duplicated().any():
            raise ValueError("Duplicate individual training target")
        if ((cohort.planned_end > stamp(end)) | (cohort.entry_time < stamp(start))).any():
            raise ValueError("Target crosses the training cutoff")
        known = cohort.loc[~cohort.censored & np.isfinite(cohort.net_R)].set_index("signal_time")
        cohorts[family] = known
    index = cohorts["SHORT44"].index.intersection(cohorts["LONG44"].index).sort_values()
    if not index.isin(rows.index).all():
        raise ValueError("Training label has no exact closed44 feature row")
    X = rows.loc[index, list(FEATURE_NAMES)].copy()
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError("Common training matrix contains an unknown feature")
    pair = pd.DataFrame({"short_net_R": cohorts["SHORT44"].loc[index, "net_R"],
                         "long_net_R": cohorts["LONG44"].loc[index, "net_R"],
                         "short_planned_end": cohorts["SHORT44"].loc[index, "planned_end"],
                         "long_planned_end": cohorts["LONG44"].loc[index, "planned_end"]}, index=index)
    pair.index.name = "signal_time"
    return X, pair


def native_hash(frame):
    return hashlib.sha256(np.ascontiguousarray(frame.to_numpy(dtype="<f8")).tobytes()).hexdigest()


def dump_frame(output, category, name, frame, index=False):
    path = output_path(output) / category / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        frame.to_csv(stream, index=index)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "rows": len(frame),
            "columns": list(frame.columns), "local_ignored_artifact": True}


def fit_pair(rows, targets, start, end, output, name):
    X, pair = common_training(rows, targets, start, end)
    artifacts = {"matrix_and_both_targets": dump_frame(output, "matrices", name, X.join(pair), index=True)}
    identity = {"common_training_rows": len(X), "feature_names": list(FEATURE_NAMES),
                "byte_hash_schema": {"matrix_and_targets": "C_row_major_little_endian_float64",
                                     "issuance_times": "little_endian_int64_UTC_nanoseconds"},
                "matrix_native_sha256": native_hash(X),
                "issue_timestamps_native_sha256": hashlib.sha256(np.ascontiguousarray(X.index.as_unit("ns").asi8, dtype="<i8").tobytes()).hexdigest(),
                "short_target_native_sha256": native_hash(pair[["short_net_R"]]),
                "long_target_native_sha256": native_hash(pair[["long_net_R"]]),
                "training_latest_issue": X.index.max() if len(X) else None,
                "training_latest_long_planned_end": pair.long_planned_end.max() if len(pair) else None}
    models = {}
    for family, target in (("SHORT44", "short_net_R"), ("LONG44", "long_net_R")):
        estimator = fit_ridge(X, pair[target].to_numpy(float), FEATURE_NAMES, penalty=.1, quantile=.75) if len(X) >= MIN_TRAIN else None
        model = {"family": family, "status": "TESTED" if estimator else "NOT TESTED",
                 "target_name": CONFIG["target_names"][family],
                 "target_horizon_minutes": TARGET_MINUTES[family], "execution_horizon_minutes": 1,
                 "feature_names": list(FEATURE_NAMES), "training_start": start, "training_end": end,
                 "training_completed_labels": len(X), "label_purge_minutes": PURGE,
                 "training_identity": identity, "artifacts": artifacts, "estimator": estimator,
                 "serialized_estimator_sha256": json_hash(estimator) if estimator else None,
                 "shared_scaler_sha256": json_hash({"means": estimator["means"], "std": estimator["std"]}) if estimator else None}
        validate_model(model); models[family] = model
    if models["SHORT44"]["shared_scaler_sha256"] != models["LONG44"]["shared_scaler_sha256"]:
        raise ValueError("SHORT/LONG do not share the same train-only scaler")
    return models, identity


def validate_model(model):
    family, n = model["family"], model["training_completed_labels"]
    if (family not in FAMILIES or type(model["target_horizon_minutes"]) is not int or
            model["target_name"] != CONFIG["target_names"][family] or
            model["target_horizon_minutes"] != TARGET_MINUTES[family] or
            type(model["execution_horizon_minutes"]) is not int or model["execution_horizon_minutes"] != 1 or
            model["feature_names"] != list(FEATURE_NAMES) or type(n) is not int or n < 0 or
            type(model["label_purge_minutes"]) is not int or model["label_purge_minutes"] != PURGE or
            type(model["training_identity"]["common_training_rows"]) is not int or
            model["training_identity"]["common_training_rows"] != n):
        raise ValueError("Frozen family, common-row schema, target or execution changed")
    if model["status"] == "NOT TESTED":
        if n >= MIN_TRAIN or any(model[key] is not None for key in
                                  ("estimator", "serialized_estimator_sha256", "shared_scaler_sha256")):
            raise ValueError("Insufficient fits must remain NOT TESTED")
        return
    est = model["estimator"]
    if (model["status"] != "TESTED" or n < MIN_TRAIN or type(est.get("version")) is not int or est["version"] != 1 or
            est.get("scaler_fitted_on") != "training_rows_only" or est.get("target_clipped") is not False or
            est.get("features_clipped") is not False or
            est.get("objective") != "mean_squared_error_plus_penalty_times_squared_coefficients" or
            est.get("feature_names") != list(FEATURE_NAMES) or type(est.get("fit_rows")) is not int or est["fit_rows"] != n or
            est.get("penalty") != .1 or est.get("quantile") != .75 or
            est.get("score_kind") != "continuous_uncalibrated_score"):
        raise ValueError("Frozen ridge training-only unclipped semantics changed")
    vectors = [np.asarray(est[key], float) for key in ("means", "std", "coefs")]
    if (any(value.shape != (44,) for value in vectors) or
            not np.isfinite(np.r_[*vectors, float(est["intercept"]), float(est["threshold"])]).all() or
            (vectors[1] <= 0).any() or est["threshold"] < 0 or
            json_hash(est) != model["serialized_estimator_sha256"] or
            json_hash({"means": est["means"], "std": est["std"]}) != model["shared_scaler_sha256"]):
        raise ValueError("Frozen ridge scaler/coefficient/cutoff bytes or dimensions changed")
    start, end = stamp(model["training_start"]), stamp(model["training_end"])
    identity = model["training_identity"]
    if (start >= end or stamp(identity["training_latest_issue"]) < start or
            stamp(identity["training_latest_issue"]) + pd.Timedelta(minutes=PURGE) > end or
            stamp(identity["training_latest_long_planned_end"]) > end):
        raise ValueError("Frozen training chronology crosses its purge/cutoff")


def verify_artifacts(value, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if value.get("local_ignored_artifact") is True:
            label = value["path"]
            if label not in seen:
                if digest(path_for(label)) != value["sha256"]:
                    raise ValueError("Frozen matrix/target/issuance/ledger changed")
                seen.add(label)
        if "family" in value and "estimator" in value and "status" in value:
            validate_model(value)
        for child in value.values():
            verify_artifacts(child, seen)
    elif isinstance(value, list):
        for child in value:
            verify_artifacts(child, seen)


def frozen_selection(output):
    declaration = frozen_declaration(output)
    output = output_path(output); path = output / "selection.json"
    value = json.loads(path.read_text())
    if digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Frozen selection changed")
    offline_document(value)
    if (value["stage"] != "frozen_short_target_development" or
            value["declaration_sha256"] != digest(output / "declaration.json") or
            any(json_hash(value[key]) != json_hash(declaration[key]) for key in declaration if key not in ("stage", "run_utc", "symbols"))):
        raise ValueError("Selection diverges from the frozen declaration")
    if set(value["symbols"]) != set(SYMBOLS):
        raise ValueError("Frozen symbol family changed")
    for symbol, payload in value["symbols"].items():
        if set(payload["models"]) != {f"{family}_{mode}" for family in FAMILIES for mode in MODES}:
            raise ValueError("Frozen target/direction family changed")
        expected = choose(payload["models"])
        if payload["selected_model"] != expected:
            raise ValueError("Selection is not the development-only SHORT choice")
        for key, candidate in payload["models"].items():
            if (key != f"{candidate['family']}_{candidate['mode']}" or
                    json_hash(candidate["config"]) != json_hash(asdict(CFG)) or
                    candidate["reference_only"] is not (candidate["family"] == "LONG44") or
                    candidate["development_eligible"] is not eligible(candidate)):
                raise ValueError("Frozen development eligibility/configuration changed")
        for mode in MODES:
            a = payload["models"][f"SHORT44_{mode}"]["final_model"]
            b = payload["models"][f"LONG44_{mode}"]["final_model"]
            if json_hash(a["training_identity"]) != json_hash(b["training_identity"]) or a["shared_scaler_sha256"] != b["shared_scaler_sha256"]:
                raise ValueError("Frozen SHORT/LONG training intersection/scaler diverged")
    verify_artifacts(value)
    return value, digest(path)


def known_policy(metric, audit):
    return metric["censored"] == 0 and metric["invalid_uncensored"] == 0 and audit["missing_entry"] == 0


def eligible(candidate):
    return (candidate["family"] == "SHORT44" and inherited_eligible(candidate) and
            all(known_policy(fold["metrics"], fold["audit"]) for fold in candidate["walk_forward"]))


def choose(candidates):
    pool = [(key, row) for key, row in candidates.items() if row["development_eligible"] and row["family"] == "SHORT44"]
    return sorted(pool, key=lambda pair: (-pair[1]["validation"]["selection_score"], pair[0]))[0][0] if pool else None


def save_replay(output, name, signals, trades):
    signal_frame = pd.DataFrame(signals, columns=["signal_time", "atr", "side", "signal_close", "variant", "score"])
    return {"signals": dump_frame(output, "signals", name, signal_frame),
            "ledger": dump_frame(output, "ledgers", name, trades)}


def develop(output):
    output = guard(output, "develop"); declaration = frozen_declaration(output)
    save(output / "development_state.json", {"stage": "STARTED", "safety": assert_offline(),
         "declaration_sha256": digest(output / "declaration.json"), "started_utc": datetime.now(timezone.utc)})
    selection = {**declaration, "stage": "frozen_short_target_development",
                 "declaration_sha256": digest(output / "declaration.json"), "symbols": {}}
    flat = []
    for symbol in SYMBOLS:
        source = declaration["lineage"]["sources"][symbol]["old"]
        start = pd.Timestamp(source["first_epoch"], unit="s", tz="UTC")
        end = pd.Timestamp(source["last_epoch"] + 60, unit="s", tz="UTC")
        split = {"development": (start, start + (end-start)*.7), "final_test": (start+(end-start)*.7, end)}
        for i, fraction in enumerate((.4, .5, .6), 1):
            split[f"train{int(fraction*100)}"] = (start, start+(end-start)*fraction)
            split[f"wf{i}"] = (start+(end-start)*fraction, start+(end-start)*(fraction+.1))
        # Rebuild each fit prefix separately; later validation candles cannot
        # change an earlier feature row, training target, scaler or threshold.
        contexts = {}
        for training in ("train40", "train50", "train60", "development"):
            m1, rows, audit, actual_split = prepare(source, symbol, split[training][1])
            if json_hash(actual_split) != json_hash(split):
                raise ValueError("Declared calendar70/30 cutoffs diverged")
            context = dump_frame(output, "inputs", f"{symbol}_{training}", rows, index=True)
            contexts[training] = (m1, rows, audit, context)
        validation_m1, validation_rows, final_audit, _ = contexts["development"]
        candidates = {}
        for mode in MODES:
            fitting = {}
            for training, (m1, rows, audit, context) in contexts.items():
                targets, target_audits = target_pair(m1, rows, symbol, mode, *split[training])
                artifacts = {family: dump_frame(output, "training_labels", f"{symbol}_{mode}_{training}_{family}", frame)
                             for family, frame in targets.items()}
                models, identity = fit_pair(rows, targets, *split[training], output, f"{symbol}_{mode}_{training}")
                fitting[training] = {"models": models, "identity": identity, "prefix_audit": audit,
                                     "input_artifact": context, "target_audits": target_audits,
                                     "target_artifacts": artifacts}
            for family in FAMILIES:
                folds, ledgers = [], []
                for training, test in WALK_FORWARD:
                    model = fitting[training]["models"][family]
                    rows = validation_rows.loc[(validation_rows.index >= split[test][0]) & (validation_rows.index < split[test][1])]
                    signals = issued(rows, symbol, mode, model)
                    trades, execution = replay_timed(validation_m1, signals, CFG, *split[test], purge_minutes=PURGE)
                    clock, clock_audit = replay_timed(validation_m1, issued(rows, symbol, mode), CFG, *split[test], purge_minutes=PURGE)
                    folds.append({"training": training, "test": test, "model": model,
                        "metrics": summarize(trades, *split[test], stop_atr=2.), "audit": execution,
                        "baseline": summarize(clock, *split[test], stop_atr=2.), "baseline_audit": clock_audit,
                        "baseline_artifacts": save_replay(output, f"{symbol}_{mode}_{family}_{test}_CLOCK",
                                                           issued(rows, symbol, mode), clock),
                        "artifacts": save_replay(output, f"{symbol}_{mode}_{family}_{test}", signals, trades)})
                    ledgers.append(trades)
                combined = pd.concat(ledgers, ignore_index=True)
                candidate = {"family": family, "mode": mode, "config": asdict(CFG), "walk_forward": folds,
                    "validation": summarize(combined, split["wf1"][0], split["wf3"][1], stop_atr=2.),
                    "final_model": fitting["development"]["models"][family], "reference_only": family == "LONG44",
                    "validation_artifact": dump_frame(output, "ledgers", f"{symbol}_{mode}_{family}_validation", combined)}
                candidate["development_eligible"] = eligible(candidate)
                candidates[f"{family}_{mode}"] = candidate
                flat.append({"symbol": symbol, "model": f"{family}_{mode}", **candidate["validation"],
                             "development_eligible": candidate["development_eligible"]})
                print(f"{symbol}/{family}/{mode}: development n={candidate['validation']['completed']} PF={candidate['validation']['profit_factor']} eligible={candidate['development_eligible']}", flush=True)
            candidates[f"SHORT44_{mode}"]["training_contexts"] = fitting
        selection["symbols"][symbol] = {"models": candidates, "partitions": split, "audit": final_audit,
                                         "selected_model": choose(candidates)}
    frozen_declaration(output); verify_artifacts(selection)
    dump_frame(output, ".", "development_metrics", pd.DataFrame(flat))
    selection["frozen_utc"] = datetime.now(timezone.utc)
    save_frozen(output / "selection.json", selection)
    print("SHORT_TARGET_SELECTION_FROZEN", digest(output / "selection.json"), flush=True)
    return canonical(selection)


def inference(trades, reference, start, end, metric, audit, reference_metric, reference_audit):
    day = paired_inference(trades, reference, start, end, repeats=BOOTSTRAP, seed=SEED)
    week = weekly_inference(trades, reference, start, end, BOOTSTRAP, seed=SEED)
    available = (known_policy(metric, audit) and known_policy(reference_metric, reference_audit) and
                 day["bootstrap_mean_valid_replicates"] == BOOTSTRAP and
                 day["bootstrap_difference_valid_replicates"] == BOOTSTRAP and
                 week["weekly_valid_mean_replicates"] == BOOTSTRAP and
                 week["weekly_valid_difference_replicates"] == BOOTSTRAP and day["calendar_days"] >= 7)
    return {"day": day, "weekly": week, "policy_inference_available": available,
            "p": max(day["p"], week["weekly_p"]) if available else 1.,
            "day_mean_undefined_replicates": BOOTSTRAP-day["bootstrap_mean_valid_replicates"],
            "day_difference_undefined_replicates": BOOTSTRAP-day["bootstrap_difference_valid_replicates"],
            "weekly_mean_undefined_replicates": BOOTSTRAP-week["weekly_valid_mean_replicates"],
            "weekly_difference_undefined_replicates": BOOTSTRAP-week["weekly_valid_difference_replicates"],
            "interpretation": "conditional_known_completed_subset_only" if not available else "known_historical_policy_day_blocks",
            "policy_unknowns_forbid_promotion": not (known_policy(metric, audit) and known_policy(reference_metric, reference_audit))}


def lower(interval, threshold=0):
    return interval[0] is not None and np.isfinite(interval[0]) and interval[0] > threshold


def gate(row):
    metric, reasons = row["metrics"], []
    if row["family"] != "SHORT44":
        row.update(historical_candidate=False, supports_expected_pf_1_5=False, reference_only=True,
                   rejection_reasons=["fixed_diagnostic_reference_only"])
        return
    checks = [(metric["completed"] >= 1000 and (metric["profit_factor"] or 0) >= 1.5, "user_pf1_5_n1000_not_met"),
        (metric["mean_net_R"] is not None and np.isfinite(metric["mean_net_R"]) and metric["mean_net_R"] >= .10,
         "mean_net_R_under0_10"),
        (metric["active_days"] >= 60, "under60_active_days"), (row["development_eligible"], "development_rejected"),
        (known_policy(metric, row["audit"]), "missing_or_censored_outcomes"),
        (metric["day_undefined_replicates"] == 0 and metric["weekly_undefined_replicates"] == 0, "undefined_pf_draws"),
        (lower(metric["day_profit_factor_ci95"], 1) and lower(metric["weekly_profit_factor_ci95"], 1), "pf_ci_not_above1"),
        (metric.get("holm_p", 1) < .05, "four_conjunction_holm_not_significant"),
        (all(third["metrics"]["completed"] >= 200 and (third["metrics"]["mean_net_R"] or -np.inf) > 0 for third in row["thirds"]), "chronological_thirds_unstable"),
        (not metric["equity_ruin"] and metric["closed_trade_max_drawdown"] <= .1, "drawdown_over10pct_or_ruin"),
        ((row["doubled_cost"]["mean_net_R"] or -np.inf) > 0, "doubled_cost_not_positive")]
    for reference in ("LONG44", "CLOCK"):
        test = row["comparisons"][reference]
        checks += [(test["policy_inference_available"], f"unknown_or_undefined_{reference}_inference"),
                   (lower(test["day"]["mean_net_R_ci95"]) and lower(test["weekly"]["weekly_mean_net_R_ci95"]), "mean_ci_not_positive"),
                   (lower(test["day"]["baseline_difference_ci95"]) and lower(test["weekly"]["weekly_difference_ci95"]), f"no_day_week_advantage_over_{reference}")]
    reasons = list(dict.fromkeys(name for valid, name in checks if not valid))
    # Known prices/payoffs cannot prove a new holdout or prospective cash profit.
    reasons += ["adaptive_known_history_requires_new_evidence", "broker_costs_and_fills_not_tested"]
    row.update(user_target_observed=checks[0][0], historical_candidate=False,
               conditional_economic_gates_pass=all(valid for valid, _ in checks),
               conditional_lower_pf_at_least1_5=(metric["day_undefined_replicates"] == 0 and
                   metric["weekly_undefined_replicates"] == 0 and known_policy(metric, row["audit"]) and
                   all(interval[0] is not None and np.isfinite(interval[0]) and interval[0] >= 1.5
                       for interval in (metric["day_profit_factor_ci95"], metric["weekly_profit_factor_ci95"]))),
               supports_expected_pf_1_5=False, reference_only=False, rejection_reasons=reasons,
               actual_money_profit="NOT TESTED", prospective_validation="NOT TESTED")


def evaluate(output):
    output = guard(output, "evaluate"); selection, fingerprint = frozen_selection(output)
    save(output / "evaluation_state.json", {"stage": "STARTED", "safety": assert_offline(),
         "selection_sha256": fingerprint, "started_utc": datetime.now(timezone.utc)})
    result = {"stage": "short_target_known_history_evaluation", "adaptive_round": 11,
              "run_utc": datetime.now(timezone.utc), "declaration_sha256": selection["declaration_sha256"],
              "selection_sha256": fingerprint, "science_code_hashes": selection["science_code_hashes"],
              "lineage": selection["lineage"], "config": CONFIG, "history": HISTORY, "safety": assert_offline(),
              "primary_holm_family": 4, "symbols": {}, "goal_achieved": False, "live_candidate": False,
              "historical_strategy_candidate": False, "actual_money_profit": "NOT TESTED"}
    hypotheses, flat = [], []
    for symbol in SYMBOLS:
        payload = {"selected_model": selection["symbols"][symbol]["selected_model"], "cohorts": {}}
        for cohort, kind in (("old_final30_secondary", "old"), ("later180_known_history_primary", "fresh")):
            source = selection["lineage"]["sources"][symbol][kind]
            m1, rows, audit, split = prepare(source, symbol)
            start, end = split["final_test"] if kind == "old" else (m1.index[0], m1.index[-1]+pd.Timedelta(minutes=1))
            rows = rows.loc[(rows.index >= start) & (rows.index < end)]
            cohort_artifact = dump_frame(output, "inputs", f"{symbol}_{cohort}", rows, index=True)
            models = {}
            for mode in MODES:
                policy = {}
                for family in (*FAMILIES, "CLOCK"):
                    candidate = selection["symbols"][symbol]["models"].get(f"{family}_{mode}")
                    model = candidate["final_model"] if candidate else None
                    signals = issued(rows, symbol, mode, model)
                    trades, execution = replay_timed(m1, signals, CFG, start, end, purge_minutes=PURGE)
                    metric = summarize(trades, start, end, stop_atr=2.)
                    metric.update(profit_factor_inference(trades, start, end, repeats=BOOTSTRAP, seed=SEED))
                    own_day = paired_inference(trades, trades, start, end, repeats=BOOTSTRAP, seed=SEED)
                    own_week = weekly_inference(trades, trades, start, end, BOOTSTRAP, seed=SEED)
                    metric.update(mean_net_R_ci95=own_day["mean_net_R_ci95"],
                        weekly_mean_net_R_ci95=own_week["weekly_mean_net_R_ci95"],
                        day_mean_valid_replicates=own_day["bootstrap_mean_valid_replicates"],
                        day_mean_undefined_replicates=BOOTSTRAP-own_day["bootstrap_mean_valid_replicates"],
                        weekly_mean_valid_replicates=own_week["weekly_valid_mean_replicates"],
                        weekly_mean_undefined_replicates=BOOTSTRAP-own_week["weekly_valid_mean_replicates"])
                    doubled = trades.copy(); doubled["net_R"] = doubled.gross_R - .20 / CFG.stop_atr
                    row = {"family": family, "mode": mode, "config": asdict(CFG), "metrics": metric, "audit": execution,
                        "development_eligible": candidate["development_eligible"] if candidate else False,
                        "selected_model": family == "SHORT44" and f"{family}_{mode}" == payload["selected_model"],
                        "model_sha256": json_hash(model) if model else None,
                        "artifacts": save_replay(output, f"{symbol}_{cohort}_{family}_{mode}", signals, trades),
                        "thirds": thirds(trades, start, end, 2.),
                        "tail": tail_metrics(trades, start, end, 2., .10),
                        "doubled_cost": summarize(doubled, start, end, stop_atr=2.),
                        "actual_money_profit": "NOT TESTED", "prospective_validation": "NOT TESTED"}
                    policy[family] = (row, trades); models[f"{family}_{mode}"] = row
                    print(f"{symbol}/{cohort}/{family}/{mode}: n={metric['completed']} PF={metric['profit_factor']}", flush=True)
                short, trades = policy["SHORT44"]
                short["comparisons"] = {}
                for reference in ("LONG44", "CLOCK"):
                    base, base_trades = policy[reference]
                    short["comparisons"][reference] = inference(trades, base_trades, start, end,
                        short["metrics"], short["audit"], base["metrics"], base["audit"])
                short["metrics"]["p"] = max(test["p"] for test in short["comparisons"].values())
                if kind == "fresh":
                    hypotheses.append(short)
            payload["cohorts"][cohort] = {"start": start, "end": end, "source": source, "audit": audit,
                 "common_clock_rows": len(rows), "input_artifact": cohort_artifact, "models": models}
        result["symbols"][symbol] = payload
    if len(hypotheses) != 4:
        raise ValueError("Exactly four SHORT absolute/clock/LONG conjunctions must enter Holm")
    holm(hypotheses)
    for symbol, payload in result["symbols"].items():
        for cohort, group in payload["cohorts"].items():
            for key, row in group["models"].items():
                if cohort == "later180_known_history_primary":
                    gate(row)
                else:
                    row.update(historical_candidate=False, supports_expected_pf_1_5=False,
                               rejection_reasons=["secondary_known_older_history"], reference_only=row["family"] != "SHORT44")
                flat.append({"symbol": symbol, "cohort": cohort, "model": key, **row["metrics"]})
    _, current_hash = frozen_selection(output)
    if current_hash != fingerprint:
        raise ValueError("Frozen models changed during evaluation")
    verify_artifacts(result)
    dump_frame(output, ".", "metrics", pd.DataFrame(flat))
    save_frozen(output / "results.json", result)
    print("SHORT_TARGET_EVALUATION_COMPLETE; known history, no refitting or enabled trading", flush=True)
    return canonical(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("declare", "develop", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(); assert_offline()
    {"declare": declare, "develop": develop, "evaluate": evaluate}[args.stage](args.output)


if __name__ == "__main__":
    main()
