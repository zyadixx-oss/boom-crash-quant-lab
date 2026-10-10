#!/usr/bin/env python3
"""Fixed SHORT1 tick-versus-M1 fidelity on twelve previously sampled days.

Only saved round11 issuances are consumed. No feature, prediction, fitting,
threshold search, account, order or promotion operation is performed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.payoff_ticks import TickExitConfig, replay_ticks
from app.research.payoff_timed import TimedExitConfig, replay_timed
from app.research.payoff_metrics import summarize
from app.research.spike_hunter import assert_offline, load_m1
from scripts import run_spike_short_target as round11
from scripts import run_spike_tick_tail as round8
from scripts import run_spike_tick_execution as round7
from scripts.run_spike_native300_study import check_prepared

SYMBOLS, MODES, DATES = round11.SYMBOLS, round11.MODES, round7.DATES
FAMILIES = ("SHORT44", "LONG44", "CLOCK")
ROUND11 = ROOT / "docs/spike_short_target_20261006"
OUTPUT = ROOT / "docs/spike_short_tick_20261006"
COHORT = "later180_known_history_primary"
TICK = TickExitConfig(stop_atr=2., max_hold_minutes=1, entry_delay_minutes=1,
                      round_trip_cost_atr=.10, max_gap_seconds=1, stop_latency_ticks=1)
COARSE = TimedExitConfig(stop_atr=2., max_hold_minutes=1, entry_delay_minutes=1,
                        round_trip_cost_atr=.10, fill_mode="adverse_extreme")
PURGE, BOOTSTRAP, SEED = 31, 9999, 20261005
SIGNAL_COLUMNS = ["signal_time", "atr", "side", "signal_close", "variant", "score"]
NEW_SOURCES = ("docs/SPIKE_SHORT_TICK_PROTOCOL.md", "scripts/run_spike_short_tick.py",
               "backend/tests/test_spike_short_tick.py")
SOURCES = tuple(dict.fromkeys((*round11.SOURCES, *round7.NEW_SOURCES,
                              "scripts/run_spike_tick_tail.py", *NEW_SOURCES)))
CONFIG = {"tick": asdict(TICK), "coarse": asdict(COARSE), "purge_minutes": PURGE,
          "bootstrap_repeats": BOOTSTRAP, "bootstrap_seed": SEED,
          "scheduled_observed_days": 12, "nominal_clock_opportunities": 576,
          "maximum_common_purged_opportunities_per_symbol_model": 564,
          "bootstrap_basis": "twelve_scheduled_observed_UTC_days_including_observed_empty_days",
          "unsampled_calendar_days_are_zero": False, "weekly_resampling": False,
          "new_scores_features_fits_or_thresholds": False, "unknown_outcomes_are_zero": False}
HISTORY = {"known_prices": True, "known_prior_payoffs": True, "price_oos": False,
           "new_label_holdout": False, "prospective": False,
           "all_twelve_tick_days_postdate_frozen_older70_fit": True,
           "study_kind": "fixed_known_history_short_execution_fidelity_not_profit_validation"}


def stamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("A valid timezone-aware timestamp is required")
    return value.tz_convert("UTC")


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def json_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def path_for(label):
    if not isinstance(label, str) or not label or "\\" in label:
        raise ValueError("Canonical relative POSIX paths are required")
    lexical = PurePosixPath(label)
    if lexical.is_absolute() or ".." in lexical.parts or str(lexical) != label or label == ".":
        raise ValueError("Absolute, traversal or ambiguous path aliases are refused")
    path = ROOT
    for part in lexical.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink source or state aliases are refused")
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Path escapes repository")
    return path


def output_path(output):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    try:
        label = str(output.relative_to(ROOT))
    except ValueError as exc:
        raise ValueError("Study output must remain inside the repository") from exc
    output = path_for(label)
    if not output.resolve().is_relative_to((ROOT / "docs").resolve()) or output == ROOT / "docs":
        raise ValueError("Study output must be a separate docs research directory")
    for ancestor in (ROOT / "docs").glob("spike_*"):
        if ancestor != output and ((ancestor / "results.json").exists() or ancestor == ROUND11) and output.is_relative_to(ancestor):
            raise ValueError("Output aliases frozen ancestor state")
    return output


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(canonical(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_frozen(path, value):
    save(path, value)
    with Path(path).with_suffix(".sha256").open("x") as stream:
        stream.write(digest(path) + "\n")


def safety(document):
    if any(document.get("safety", {}).get(flag) is not False for flag in assert_offline()):
        raise ValueError("All four saved safety values must be false")


def hashes():
    return {label: digest(path_for(label)) for label in SOURCES}


def pin(files, label, fingerprint=None):
    actual = digest(path_for(label))
    if fingerprint is not None and actual != fingerprint:
        raise ValueError("Pinned ancestor bytes changed: " + label)
    if label in files and files[label] != actual:
        raise ValueError("Conflicting pinned file identities")
    files[label] = actual


def verify_descriptor(descriptor):
    if (descriptor.get("local_ignored_artifact") is not True or type(descriptor.get("rows")) is not int or
            descriptor["rows"] < 0 or descriptor.get("columns") != SIGNAL_COLUMNS or
            digest(path_for(descriptor["path"])) != descriptor["sha256"]):
        raise ValueError("Exact pinned six-column saved issuance descriptor required")


def metadata_lineage():
    """Metadata and byte hashes only; never read a CSV or decode any quote."""
    assert_offline()
    selection, selection_sha = round11.frozen_selection(ROUND11)
    declaration = read_json(ROUND11 / "declaration.json")
    result = read_json(ROUND11 / "results.json")
    audit = read_json(ROUND11 / "independent_audit.json")
    for value in (selection, declaration, result, audit):
        safety(value)
    for name in ("declaration", "selection", "results", "independent_audit"):
        if digest(ROUND11 / f"{name}.json") != (ROUND11 / f"{name}.sha256").read_text().strip():
            raise ValueError("Exact frozen round11 artifact and SHA sidecar required")
    if (audit.get("status") != "PASS" or audit.get("pass") is not True or audit.get("error_count") != 0 or
            audit.get("errors") or audit.get("declaration_sha256") != digest(ROUND11 / "declaration.json") or
            audit.get("selection_sha256") != selection_sha or audit.get("results_sha256") != digest(ROUND11 / "results.json") or
            audit.get("script_sha256") != digest(path_for("scripts/verify_spike_short_target.py"))):
        raise ValueError("The exact independently passing round11 audit is required")
    if (result.get("stage") != "short_target_known_history_evaluation" or result.get("adaptive_round") != 11 or
            result.get("declaration_sha256") != digest(ROUND11 / "declaration.json") or
            result.get("selection_sha256") != selection_sha or result.get("goal_achieved") is not False or
            result.get("historical_strategy_candidate") is not False or result.get("live_candidate") is not False or
            json_hash(result.get("history")) != json_hash(round11.HISTORY) or
            any(json_hash(result.get(key)) != json_hash(selection[key]) for key in
                ("science_code_hashes", "lineage", "config"))):
        raise ValueError("Frozen round11 known-history/result semantics changed")
    files = {}
    for name in ("declaration", "selection", "results", "independent_audit"):
        for suffix in (".json", ".sha256"):
            pin(files, str((ROUND11 / (name + suffix)).relative_to(ROOT)))
    for label, sha in {**selection["science_code_hashes"], **selection["lineage"]["files"],
                       "scripts/verify_spike_short_target.py": audit["script_sha256"],
                       **audit["helper_sha256"]}.items():
        pin(files, label, sha)
    tick_lineage = round8.verified_round7_metadata()
    if tick_lineage["round6_selection_sha256"] != selection["lineage"]["round6_selection_sha256"]:
        raise ValueError("Tick and short-model sources do not share the frozen native600 ancestry")
    for label, sha in tick_lineage["inherited_files"].items():
        pin(files, label, sha)
    saved, sources = {}, {}
    identities = set()
    for symbol in SYMBOLS:
        saved[symbol] = {}
        sources[symbol] = selection["lineage"]["sources"][symbol]["fresh"]
        source = sources[symbol]
        for key, sha in (("path", "sha256"), ("manifest", "manifest_sha256"),
                         ("raw_file", "raw_sha256"), ("page_audit", "page_audit_sha256")):
            pin(files, source[key], source[sha])
        cohort = result["symbols"][symbol]["cohorts"][COHORT]
        if json_hash(cohort["source"]) != json_hash(source):
            raise ValueError("Round11 later issuance and pinned M1 bytes differ")
        for family in FAMILIES:
            for mode in MODES:
                key = f"{family}_{mode}"
                row = cohort["models"][key]
                descriptor = row["artifacts"]["signals"]
                verify_descriptor(descriptor); pin(files, descriptor["path"], descriptor["sha256"])
                if descriptor["path"] in identities:
                    raise ValueError("Saved issuance file aliases are forbidden")
                identities.add(descriptor["path"])
                candidate = selection["symbols"][symbol]["models"].get(key)
                model = candidate["final_model"] if candidate else None
                development = candidate["development_eligible"] if candidate else False
                if (row["family"] != family or row["mode"] != mode or
                        json_hash(row["config"]) != json_hash(asdict(COARSE)) or
                        row["model_sha256"] != (json_hash(model) if model else None) or
                        row["development_eligible"] is not development or
                        row["historical_candidate"] is not False or row["reference_only"] is not (family != "SHORT44")):
                    raise ValueError("Round11 saved family/config/model/eligibility identity changed")
                if model is not None and stamp(model["training_end"]) >= pd.Timestamp(DATES[0], tz="UTC"):
                    raise ValueError("Every scheduled tick day must follow the frozen older70 fit")
                saved[symbol][key] = {"signals": descriptor, "family": family, "mode": mode,
                    "model_sha256": row["model_sha256"], "development_eligible": development,
                    "reference_only": family != "SHORT44", "round11_selected_model": row["selected_model"],
                    "training_end": model["training_end"] if model else None,
                    "frozen_score_threshold": model["estimator"]["threshold"] if model and model["estimator"] else None}
    if len(identities) != 12:
        raise ValueError("Exactly twelve saved later180 issuance files are required")
    return canonical({"files": files, "round11_declaration_sha256": digest(ROUND11 / "declaration.json"),
        "round11_selection_sha256": selection_sha, "round11_results_sha256": digest(ROUND11 / "results.json"),
        "round11_audit_sha256": digest(ROUND11 / "independent_audit.json"),
        "round11_science_code_hashes": selection["science_code_hashes"],
        "sampled_signals": saved, "m1_sources": sources, "tick_lineage": tick_lineage})


def guard(output, stage):
    assert_offline(); output = output_path(output)
    if stage not in ("declare", "evaluate"):
        raise ValueError("Unknown short-tick stage")
    names = ["evaluation_state.json", "results.json", "results.sha256", "metrics.csv"]
    if stage == "declare":
        names += ["declaration.json", "declaration.sha256"]
        if any((output / name).exists() and any((output / name).iterdir()) for name in ("inputs", "signals", "ledgers")):
            raise ValueError("Refusing predeclaration computed science artifacts")
    if any((output / name).exists() for name in names):
        raise ValueError("Refusing frozen declaration/state/result overwrite")
    return output


def declare(output):
    output = guard(output, "declare")
    science, lineage = hashes(), metadata_lineage()
    if any(path_for(label).resolve().is_relative_to(output.resolve()) for label in lineage["files"]):
        raise ValueError("Output aliases pinned ancestor state")
    value = {"stage": "short_tick_premeasurement_declaration", "adaptive_round": 12,
        "run_utc": datetime.now(timezone.utc), "safety": assert_offline(), "config": CONFIG, "history": HISTORY,
        "science_code_hashes": science, "lineage": lineage, "dates": DATES, "symbols": SYMBOLS,
        "families": FAMILIES, "modes": MODES, "prices_or_signal_csv_decoded": False,
        "new_execution_ledgers_computed": False, "new_model_or_scores_computed": False,
        "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False,
        "user_target": {"profit_factor": 1.5, "completed_per_symbol_model": 1000, "active_heldout_days": 60},
        "study_can_meet_sample_gate": False, "bootstrap_is_discovery_test": False}
    if json_hash(science) != json_hash(hashes()) or json_hash(lineage) != json_hash(metadata_lineage()):
        raise ValueError("Frozen science or metadata changed during declaration")
    save_frozen(output / "declaration.json", value)
    print("SHORT_TICK_DECLARED", digest(output / "declaration.json"), "metadata and hashes only", flush=True)
    return canonical(value)


def frozen(output):
    assert_offline(); output = output_path(output)
    value = read_json(output / "declaration.json")
    if digest(output / "declaration.json") != (output / "declaration.sha256").read_text().strip():
        raise ValueError("Frozen declaration hash changed")
    safety(value)
    expected = {"stage": "short_tick_premeasurement_declaration", "adaptive_round": 12,
        "config": CONFIG, "history": HISTORY, "science_code_hashes": hashes(), "lineage": metadata_lineage(),
        "dates": DATES, "symbols": SYMBOLS, "families": FAMILIES, "modes": MODES,
        "prices_or_signal_csv_decoded": False, "new_execution_ledgers_computed": False,
        "new_model_or_scores_computed": False, "goal_achieved": False,
        "historical_strategy_candidate": False, "live_candidate": False,
        "user_target": {"profit_factor": 1.5, "completed_per_symbol_model": 1000, "active_heldout_days": 60},
        "study_can_meet_sample_gate": False, "bootstrap_is_discovery_test": False}
    if any(json_hash(value.get(key)) != json_hash(wanted) for key, wanted in expected.items()):
        raise ValueError("Frozen source/config/history/type semantics changed")
    return value


def load_signals(value, symbol):
    """Copy frozen positive issuance rows; never generate or predict a score."""
    descriptor = value["signals"]; verify_descriptor(descriptor)
    records, previous = [], None
    family, mode = value["family"], value["mode"]
    expected_side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    with path_for(descriptor["path"]).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SIGNAL_COLUMNS:
            raise ValueError("Saved issuance CSV schema changed")
        for row in reader:
            time = stamp(row["signal_time"])
            atr, close = float(row["atr"]), float(row["signal_close"])
            if (time.value % (30*60*1_000_000_000) or previous is not None and time <= previous or
                    row["side"] != str(expected_side) or row["variant"] != f"{family}_{mode}" or
                    not math.isfinite(atr) or atr <= 0 or not math.isfinite(close) or close <= 0):
                raise ValueError("Saved issuance clock/order/side/ATR/variant semantics changed")
            score = float(row["score"]) if row["score"] else None
            if family == "CLOCK":
                if score is not None:
                    raise ValueError("CLOCK must remain an unscored saved reference")
            elif (score is None or not math.isfinite(score) or score <= 0 or
                  value["frozen_score_threshold"] is None or score < value["frozen_score_threshold"] or
                  time < stamp(value["training_end"])):
                raise ValueError("Saved positive frozen score/cutoff chronology changed")
            item = {"signal_time": time, "atr": atr, "side": expected_side,
                    "signal_close": close, "variant": row["variant"]}
            if score is not None:
                item["score"] = score
            records.append(item); previous = time
    if len(records) != descriptor["rows"]:
        raise ValueError("Saved issuance row count changed")
    return records


def for_day(records, date):
    start, end = round7.day_bounds(date)
    return [record for record in records if start <= stamp(record["signal_time"]) < end]


def dump_frame(output, category, name, frame, index=False):
    path = output_path(output) / category / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        frame.to_csv(stream, index=index)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "rows": len(frame),
            "columns": list(frame.columns), "local_ignored_artifact": True}


def known_rows(trades):
    if trades.censored.isna().any() or not trades.censored.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("Known boolean censor flags are required")
    frame = trades.copy()
    frame["signal_time"] = pd.to_datetime(frame.signal_time, utc=True)
    if frame.signal_time.isna().any() or frame.signal_time.duplicated().any():
        raise ValueError("Unique valid saved path issuance times are required")
    if not frame.signal_time.dt.strftime("%Y-%m-%d").isin(DATES).all():
        raise ValueError("Unscheduled path is forbidden")
    return frame.loc[~frame.censored & np.isfinite(frame.net_R)].copy()


def daily_arrays(trades):
    known = known_rows(trades); known["day"] = known.signal_time.dt.strftime("%Y-%m-%d")
    counts, sums, gains, losses = (np.zeros(len(DATES)) for _ in range(4))
    for i, date in enumerate(DATES):
        values = known.loc[known.day == date, "net_R"].to_numpy(float)
        counts[i], sums[i] = len(values), values.sum()
        gains[i], losses[i] = np.maximum(values, 0).sum(), np.maximum(-values, 0).sum()
    return counts, sums, gains, losses


def weights():
    return np.random.default_rng(SEED).multinomial(len(DATES), np.full(len(DATES), 1/len(DATES)), size=BOOTSTRAP)


def interval(values):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]


def summary(trades, bootstrap=True, date=None):
    if date is not None and date not in DATES:
        raise ValueError("Only declared scheduled days may be summarized")
    start, end = round7.day_bounds(date) if date else (round7.day_bounds(DATES[0])[0], round7.day_bounds(DATES[-1])[1])
    known_rows(trades)
    result = summarize(trades, start, end, stop_atr=2.)
    result["selection_score"] = None
    if not bootstrap:
        return result
    counts, sums, gains, losses = daily_arrays(trades)
    result["span_calendar_days"], result["calendar_days"] = result["calendar_days"], len(DATES)
    result["calendar_basis"] = "twelve scheduled observed days; no unsampled calendar padding"
    result["cluster_se"] = (float(np.sqrt(len(DATES)/(len(DATES)-1) *
        np.sum((sums-result["mean_net_R"]*counts)**2)) / counts.sum()) if counts.sum() else None)
    w = weights(); n, total, gain, loss = (w @ x for x in (counts, sums, gains, losses))
    means = np.divide(total, n, out=np.full(BOOTSTRAP, np.nan), where=n > 0)
    pf = np.divide(gain, loss, out=np.full(BOOTSTRAP, np.nan), where=loss > 0)
    result.update(mean_net_R_ci95=interval(means), profit_factor_ci95=interval(pf),
        mean_valid_replicates=int(np.isfinite(means).sum()), mean_undefined_replicates=int((~np.isfinite(means)).sum()),
        pf_valid_replicates=int(np.isfinite(pf).sum()), pf_undefined_replicates=int((~np.isfinite(pf)).sum()),
        bootstrap_repeats=BOOTSTRAP, bootstrap_seed=SEED, no_contiguous_weekly_claim=True,
        scheduled_observed_day_clusters=len(DATES),
        inference_kind="descriptive_scheduled_day_bootstrap_not_discovery", unknowns_are_not_zero=True)
    return result


def unknown_counts(trades, audit=None):
    completed = len(known_rows(trades))
    censored = int(trades.censored.sum())
    return {"censored": censored, "invalid_uncensored": len(trades)-censored-completed,
            "missing_entry": audit["missing_entry"] if audit is not None else None}


def unknown_present(*counts):
    if any(value["censored"] or value["invalid_uncensored"] or value["missing_entry"] for value in counts):
        return True
    return None if any(value["missing_entry"] is None for value in counts) else False


def matched_known(tick, coarse, tick_audit=None, coarse_audit=None, bootstrap=True):
    a = known_rows(tick)[["signal_time", "net_R"]]
    b = known_rows(coarse)[["signal_time", "net_R"]]
    matched = a.merge(b, on="signal_time", how="inner", validate="one_to_one", suffixes=("_tick", "_coarse"))
    matched["tick_minus_coarse_R"] = matched.net_R_tick-matched.net_R_coarse
    counts, sums = np.zeros(len(DATES)), np.zeros(len(DATES))
    for i, date in enumerate(DATES):
        values = matched.loc[matched.signal_time.dt.strftime("%Y-%m-%d") == date, "tick_minus_coarse_R"]
        counts[i], sums[i] = len(values), values.sum()
    if bootstrap:
        w = weights(); n, totals = w @ counts, w @ sums
        samples = np.divide(totals, n, out=np.full(BOOTSTRAP, np.nan), where=n > 0)
    else:
        samples = np.empty(0)
    tick_unknown, coarse_unknown = unknown_counts(tick, tick_audit), unknown_counts(coarse, coarse_audit)
    return matched, {"matched_completed": len(matched), "unmatched_tick_completed": len(a)-len(matched),
        "unmatched_coarse_completed": len(b)-len(matched),
        "tick_unknown_counts": tick_unknown, "coarse_unknown_counts": coarse_unknown,
        "policy_unknowns_present": unknown_present(tick_unknown, coarse_unknown),
        "tick_minus_coarse_mean_R": float(matched.tick_minus_coarse_R.mean()) if len(matched) else None,
        "difference_ci95": interval(samples), "valid_replicates": int(np.isfinite(samples).sum()),
        "undefined_replicates": int((~np.isfinite(samples)).sum()),
        "bootstrap_repeats": BOOTSTRAP if bootstrap else 0, "bootstrap_seed": SEED if bootstrap else None,
        "inference_status": "FINITE_DRAW_DESCRIPTIVE_ONLY" if bootstrap else "NOT_REQUESTED_PER_DAY_POINT_SUMMARY",
        "descriptive_only": True,
        "interpretation": "matched known outcomes; entry stop and expiry quotes all change, not stop-only causality"}


def clock_comparison(model, clock, model_audit=None, clock_audit=None):
    n, total, _, _ = daily_arrays(model); bn, btotal, _, _ = daily_arrays(clock)
    w = weights(); mn, ms, cn, cs = w @ n, w @ total, w @ bn, w @ btotal
    means = np.divide(ms, mn, out=np.full(BOOTSTRAP, np.nan), where=mn > 0)
    base = np.divide(cs, cn, out=np.full(BOOTSTRAP, np.nan), where=cn > 0)
    delta = means-base
    model_unknown, reference_unknown = unknown_counts(model, model_audit), unknown_counts(clock, clock_audit)
    return {"mean_R_difference": float(total.sum()/n.sum()-btotal.sum()/bn.sum()) if n.sum() and bn.sum() else None,
        "model_unknown_counts": model_unknown, "reference_unknown_counts": reference_unknown,
        "policy_unknowns_present": unknown_present(model_unknown, reference_unknown),
        "difference_ci95": interval(delta), "valid_replicates": int(np.isfinite(delta).sum()),
        "undefined_replicates": int((~np.isfinite(delta)).sum()),
        "bootstrap_repeats": BOOTSTRAP, "bootstrap_seed": SEED, "descriptive_only": True,
        "comparison": "same scheduled dates, different saved issuance exposure; per-completed-path means"}


def totals(audits):
    keys = ("issued", "filled", "completed", "censored", "missing_entry", "outside_partition", "purged", "overlap_skipped", "ambiguous")
    return {key: sum(audit[key] for audit in audits) for key in keys}


def replay_day(ticks, m1, records, date):
    if date not in DATES:
        raise ValueError("Only the fixed twelve scheduled days are permitted")
    start, end = round7.day_bounds(date)
    if len(records) > 48:
        raise ValueError("A saved00/30 day cannot contain more than48 opportunities")
    primary, tick_audit = replay_ticks(ticks, records, TICK, start, end, purge_minutes=PURGE)
    coarse, coarse_audit = replay_timed(m1, records, COARSE, start, end, purge_minutes=PURGE)
    if tick_audit["purged"] != coarse_audit["purged"] or tick_audit["outside_partition"] != coarse_audit["outside_partition"]:
        raise ValueError("Common day coverage/planned purge differs between execution proxies")
    return primary, tick_audit, coarse, coarse_audit


def evaluate(output):
    output = guard(output, "evaluate"); declaration = frozen(output)
    save(output / "evaluation_state.json", {"stage": "STARTED", "safety": assert_offline(),
        "declaration_sha256": digest(output / "declaration.json"), "started_utc": datetime.now(timezone.utc)})
    result = {"stage": "short_tick_known_history_fidelity", "adaptive_round": 12,
        "run_utc": datetime.now(timezone.utc), "declaration_sha256": digest(output / "declaration.json"),
        "science_code_hashes": declaration["science_code_hashes"], "lineage": declaration["lineage"],
        "config": CONFIG, "history": HISTORY, "safety": assert_offline(), "dates": DATES, "symbols": {},
        "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False,
        "actual_money_profit": "NOT TESTED", "prospective_paper": "NOT TESTED"}
    flat = []
    for symbol in SYMBOLS:
        source = declaration["lineage"]["m1_sources"][symbol]
        m1, m1_audit = load_m1(path_for(source["path"]))
        check_prepared(m1, m1_audit, source)
        contexts, inputs = {}, {}
        for date in DATES:
            tick_source = declaration["lineage"]["tick_lineage"]["sources"][symbol][date]
            ticks = round8.load_tick_day(tick_source)
            reconciliation = round7.reconcile(ticks, m1, date)
            start, end = round7.day_bounds(date)
            minutes = m1.loc[(m1.index >= start) & (m1.index < end)].copy()
            contexts[date] = (ticks, minutes)
            ticks.index.name, minutes.index.name = "quote_time", "minute_open"
            inputs[date] = {"tick_source": tick_source, "reconciliation": reconciliation,
                "ticks": dump_frame(output, "inputs", f"{symbol}_{date}_ticks", ticks, index=True),
                "m1": dump_frame(output, "inputs", f"{symbol}_{date}_m1", minutes, index=True)}
        models, ledgers = {}, {}
        for key, value in declaration["lineage"]["sampled_signals"][symbol].items():
            records = load_signals(value, symbol)
            tick_days, coarse_days, day_reports, tick_audits, coarse_audits = [], [], [], [], []
            for date in DATES:
                day_records = for_day(records, date); ticks, minutes = contexts[date]
                tick, ta, coarse, ca = replay_day(ticks, minutes, day_records, date)
                matched, match_summary = matched_known(tick, coarse, ta, ca, bootstrap=False)
                signal_frame = pd.DataFrame(day_records, columns=SIGNAL_COLUMNS)
                day_reports.append({"date": date, "signals": dump_frame(output, "signals", f"{symbol}_{key}_{date}", signal_frame),
                    "tick": {"metrics": summary(tick, bootstrap=False, date=date), "audit": ta,
                             "ledger": dump_frame(output, "ledgers", f"{symbol}_{key}_{date}_tick", tick)},
                    "coarse": {"metrics": summary(coarse, bootstrap=False, date=date), "audit": ca,
                               "ledger": dump_frame(output, "ledgers", f"{symbol}_{key}_{date}_coarse", coarse)},
                    "matched_known": {"metrics": match_summary,
                        "ledger": dump_frame(output, "ledgers", f"{symbol}_{key}_{date}_matched", matched)}})
                tick_days.append(tick); coarse_days.append(coarse); tick_audits.append(ta); coarse_audits.append(ca)
            tick, coarse = pd.concat(tick_days, ignore_index=True), pd.concat(coarse_days, ignore_index=True)
            matched, match_summary = matched_known(tick, coarse, totals(tick_audits), totals(coarse_audits))
            row = {"family": value["family"], "mode": value["mode"], "model_sha256": value["model_sha256"],
                "development_eligible": value["development_eligible"], "reference_only": value["reference_only"],
                "round11_selected_model": value["round11_selected_model"], "selected_model": False,
                "historical_candidate": False, "supports_expected_pf_1_5": False, "user_target_observed": False,
                "conditional_economic_gates_pass": False, "days": day_reports,
                "tick": {"metrics": summary(tick), "audit_totals": totals(tick_audits),
                         "ledger": dump_frame(output, "ledgers", f"{symbol}_{key}_tick_all", tick)},
                "coarse": {"metrics": summary(coarse), "audit_totals": totals(coarse_audits),
                           "ledger": dump_frame(output, "ledgers", f"{symbol}_{key}_coarse_all", coarse)},
                "matched_known": {"metrics": match_summary,
                    "ledger": dump_frame(output, "ledgers", f"{symbol}_{key}_matched_all", matched)},
                "rejection_reasons": ["fixed_execution_diagnostic_not_candidate", "max564_paths_and12_days_below1000_and60",
                    "reused_adaptive_known_history", "measured_costs_fills_not_tested"],
                "actual_money_profit": "NOT TESTED", "prospective_validation": "NOT TESTED"}
            if not value["development_eligible"] and value["family"] == "SHORT44":
                row["rejection_reasons"].append("round11_development_rejected")
            models[key], ledgers[key] = row, (tick, coarse)
            flat.extend({"symbol": symbol, "model": key, "proxy": proxy, **row[proxy]["metrics"]}
                        for proxy in ("tick", "coarse"))
            print(f"{symbol}/{key}: tick n={row['tick']['metrics']['completed']} PF={row['tick']['metrics']['profit_factor']}; coarse n={row['coarse']['metrics']['completed']} PF={row['coarse']['metrics']['profit_factor']}", flush=True)
        for key, row in models.items():
            if row["family"] != "CLOCK":
                clock = ledgers[f"CLOCK_{row['mode']}"]
                row["clock_comparisons"] = {proxy: clock_comparison(ledgers[key][i], clock[i],
                    row[proxy]["audit_totals"], models[f"CLOCK_{row['mode']}"][proxy]["audit_totals"])
                                           for i, proxy in enumerate(("tick", "coarse"))}
            if row["family"] == "SHORT44":
                long_key = f"LONG44_{row['mode']}"
                row["long_comparisons"] = {proxy: clock_comparison(ledgers[key][i], ledgers[long_key][i],
                    row[proxy]["audit_totals"], models[long_key][proxy]["audit_totals"])
                                           for i, proxy in enumerate(("tick", "coarse"))}
        result["symbols"][symbol] = {"m1_audit": m1_audit, "inputs": inputs, "models": models}
    frozen(output)
    dump_frame(output, ".", "metrics", pd.DataFrame(flat))
    save_frozen(output / "results.json", result)
    print("SHORT_TICK_COMPLETE; diagnostic only; no new fit, score, policy selection or live trading", flush=True)
    return canonical(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("declare", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(); assert_offline()
    {"declare": declare, "evaluate": evaluate}[args.stage](args.output)


if __name__ == "__main__":
    main()
