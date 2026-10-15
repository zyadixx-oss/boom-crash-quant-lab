#!/usr/bin/env python3
"""Independent detector/chain/M5-age and unavailable-model audit.

No research engine is imported. Independent NumPy vector accumulation provides
a numerical oracle for the feature-only price chain. Its actual longdouble
precision is reported (it can equal float64 on a Mac). This checks
all M5 closes, large-bar flags/ages, complete feature availability and no-fit
decisions. It does not regenerate every other feature or certify fitted models,
broker execution or profit. A fitted model requires a separate economic audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs/representation_regions_20261008"
DECLARATION = "939cf7aa31208b89bef012d143b0ff21bd2603c1f0c91e7e8d45339dfa04c388"
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def epoch(t):
    t = pd.Timestamp(t)
    if t.tz is None or t.value % 10**9: raise ValueError("Whole aware seconds required")
    return t.value // 10**9


class Audit:
    def __init__(self): self.checks, self.errors, self.pins = 0, [], {}

    def check(self, condition, label, count=1):
        self.checks += count
        if not bool(condition) and len(self.errors) < 100: self.errors.append(label)

    def array(self, actual, expected, label, *, close=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.check(actual.shape == expected.shape, label + " shape")
        if actual.shape != expected.shape: return
        valid = np.isclose(actual, expected, rtol=2e-9, atol=1e-300, equal_nan=True) if close else (
            (actual == expected) | (pd.isna(actual) & pd.isna(expected)))
        self.check(valid.all(), label, max(1, actual.size))

    def pin(self, name, expected=None):
        p = (ROOT / name).resolve(); p.relative_to(ROOT.resolve())
        actual = sha(p)
        self.check(expected is None or actual == expected, "pin " + str(name))
        if expected is not None and actual != expected: raise ValueError("Pinned input changed: " + str(name))
        self.pins[str(p.relative_to(ROOT))] = actual
        return p


def native_increments(times, prices):
    result = np.full(len(times), np.nan)
    known = np.diff(times) == 1
    relative = (prices[1:] - prices[:-1]) / prices[:-1]
    if np.any(~np.isfinite(relative[known])) or np.any(relative[known] <= -1):
        raise ValueError("This oracle requires ordinary finite native returns")
    result[1:][known] = np.log1p(relative[known])
    return result


def training_scale(times, increments, start, end):
    selected = (times >= start + 1) & (times < end) & np.isfinite(increments)
    if not selected.any(): raise ValueError("No consecutive training increments")
    return float(np.median(np.abs(increments[selected])))


def independent_m5(times, increments, threshold, side, start, end):
    """Rebuild each exact run and all complete UTC300second candles.

    All52+ possible run resets remain real gaps. The M5 age is independent of
    H4/H1 join validity, so an always-unknown required age alone precludes44.
    """
    right = int(np.searchsorted(times, end))
    left = int(np.searchsorted(times, start))
    t, r = times[left:right], increments[left:right]
    size = math.ceil((end - start) / 300)
    close = np.full(size, np.nan)
    large, age = np.full(size, np.nan), np.full(size, np.nan)
    beginnings = np.r_[0, np.flatnonzero(np.diff(t) != 1) + 1] if len(t) else np.array([], int)
    stops = np.r_[beginnings[1:], len(t)]
    removed = 0
    smallest_threshold_margin = None
    for first, last in zip(beginnings, stops):
        retained = r[first:last].copy()
        retained[0] = 0.
        tail = side * retained > threshold
        removed += int(tail.sum())
        if len(retained) > 1:
            distance = float(np.min(np.abs(side * retained[1:] - threshold)))
            smallest_threshold_margin = distance if smallest_threshold_margin is None else min(smallest_threshold_margin, distance)
        retained[tail] = 0.
        # Different arithmetic from the production Neumaier state machine.
        q = np.exp(np.cumsum(retained.astype(np.longdouble), dtype=np.longdouble)).astype(float)
        if not np.isfinite(q).all() or np.any(q <= 0): raise ValueError("Independent chain not positive finite")
        opening = ((int(t[first]) + 299) // 300) * 300
        stop = ((int(t[last - 1]) + 1) // 300) * 300
        count = (stop - opening) // 300
        if count <= 0: continue
        offset = opening - int(t[first])
        bars = q[offset:offset + count * 300].reshape(count, 300)
        hi, lo, cl = bars.max(axis=1), bars.min(axis=1), bars[:, -1]
        ranges = hi - lo
        tr = ranges.copy(); tr[0] = np.nan
        if count > 1: tr[1:] = np.maximum(ranges[1:], np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
        atr = np.full(count, np.nan)
        for i in range(14, count): atr[i] = math.fsum(tr[i-13:i+1]) / 14
        flags, ages, last_age = np.full(count, np.nan), np.full(count, np.nan), None
        for i in range(15, count):
            flags[i] = float(ranges[i] > 3 * atr[i-1])
            if flags[i]: last_age = 0
            elif last_age is not None: last_age = min(last_age + 1, 288)
            if last_age is not None: ages[i] = math.log1p(last_age)
        where = (opening - start) // 300 + np.arange(count)
        close[where], large[where], age[where] = cl, flags, ages
    return {"close": close, "large_completed_bar": large, "log1p_large_bar_age": age}, {
        "runs": len(beginnings), "tail_increments_removed": removed,
        "known_increments": len(t) - len(beginnings),
        "minimum_distance_from_detector_threshold": smallest_threshold_margin,
        "large_transformed_m5_bars": int(np.nansum(large)), "known_age_rows": int(np.isfinite(age).sum())}


def audit_study():
    a = Audit()
    for flag in FLAGS:
        if os.environ.get(flag, "false").lower() != "false": raise ValueError("Offline-only audit")
    a.pin("scripts/verify_representation_availability.py")
    a.pin("backend/tests/test_representation_availability_audit.py")
    d = json.loads(a.pin(str((FOLDER / "declaration.json").relative_to(ROOT)), DECLARATION).read_text())
    a.check((FOLDER / "declaration.sha256").read_text().strip() == DECLARATION, "declaration sidecar")
    for name, pin in d["code_sha256"].items(): a.pin(name, pin)
    a.pin(d["sources"]["audit_path"], d["sources"]["audit_sha256"])
    r = json.loads(a.pin(str((FOLDER / "results.json").relative_to(ROOT))).read_text())
    a.check(r["declaration_sha256"] == DECLARATION, "result declaration")
    for obj in (d, r):
        a.check(obj["safety"] == dict.fromkeys(FLAGS, False), "false safety gates")
        a.check(obj["QUALIFIED"] is False, "never qualify")
    names = d["specification"]["feature_names"]
    reports, fitted = {}, 0
    for symbol in ("BOOM600", "CRASH600"):
        arrays = []
        for source in d["sources"]["symbols"][symbol]:
            data = np.loadtxt(a.pin(source["path"], source["sha256"]), delimiter=",", skiprows=1, ndmin=2)
            a.check(len(data) == source["rows"] and data.shape[1] == 2, source["path"] + " population")
            arrays.append(data)
        data = np.concatenate(arrays); del arrays
        times, prices = data[:, 0].astype(np.int64), data[:, 1]
        a.check(np.all(np.diff(times) > 0) and np.isfinite(prices).all() and (prices > 0).all(), symbol + " native source")
        increments = native_increments(times, prices)
        reports[symbol] = {}
        for fold, contract in d["specification"]["folds"].items():
            saved = r["symbols"][symbol]["folds"][fold]
            detector = json.loads(a.pin(saved["detector"]["path"], saved["detector"]["sha256"]).read_text())
            scale = training_scale(times, increments, *map(epoch, contract["train"]))
            a.check(math.isclose(scale, detector["median_abs_log_return"], rel_tol=2e-10, abs_tol=2e-16), symbol + fold + " training-only median")
            a.check(math.isclose(scale * 10, detector["threshold"], rel_tol=2e-10, abs_tol=2e-15), symbol + fold + " detector threshold")
            end = max(epoch(b[1]) for b in contract["evaluate"].values())
            oracle, details = independent_m5(times, increments, detector["threshold"], detector["native_side"],
                epoch(d["specification"]["data_start"]), end)
            features = {}
            for family in ("RAW44", "TRANSFORMED44"):
                artifact = saved["features"][family]
                features[family] = pd.read_csv(a.pin(artifact["path"], artifact["sha256"]), float_precision="round_trip")
                a.check(len(features[family]) == artifact["rows"], family + " feature rows")
            trans, raw = features["TRANSFORMED44"], features["RAW44"]
            a.array(pd.to_datetime(raw.m5_open, utc=True).astype("int64"), pd.to_datetime(trans.m5_open, utc=True).astype("int64"), "paired M5 clock")
            for k, expected in oracle.items(): a.array(trans[k].to_numpy(float), expected, symbol + fold + " independent " + k, close=True)
            finite = np.isfinite(trans[names].to_numpy(float)).all(axis=1)
            a.check(not (trans.feature_valid & ~finite).any(), "valid transformed44 require every finite input")
            a.check(details["known_age_rows"] == saved["preparation"]["transformed_large_bar_age_known_rows"], "independent known age count")
            for key in ("runs", "tail_increments_removed", "known_increments"):
                a.check(details[key] == saved["preparation"][key], "independent mechanism count " + key)
            avail_artifact = saved["features"]["availability"]
            avail = pd.read_csv(a.pin(avail_artifact["path"], avail_artifact["sha256"]), float_precision="round_trip")
            issues = pd.to_datetime(raw.m5_open, utc=True) + pd.Timedelta(minutes=5)
            clock = issues.dt.minute.isin((0, 30))
            a.array(pd.to_datetime(avail.issue_time, utc=True).astype("int64"), issues[clock].astype("int64"), "full scheduled clock")
            for family, column in (("RAW44", "raw_feature_valid"), ("TRANSFORMED44", "transformed_feature_valid")):
                a.array(avail[column], features[family].loc[clock, "feature_valid"], "clock intrinsic availability " + family)
            a.array(avail.common_feature_valid, avail.raw_feature_valid & avail.transformed_feature_valid, "common availability intersection")
            model = json.loads(a.pin(saved["fit"]["path"], saved["fit"]["sha256"]).read_text())
            labels = pd.read_csv(a.pin(saved["labels"]["path"], saved["labels"]["sha256"]))
            begin, finish = map(epoch, contract["train"])
            issue_seconds = pd.to_datetime(avail.issue_time, utc=True).astype("int64").to_numpy() // 10**9
            selected = (issue_seconds >= begin) & (issue_seconds < finish) & (issue_seconds + 1860 <= finish) & avail.common_feature_valid.to_numpy(bool)
            a.array(pd.to_datetime(labels.signal_time, utc=True).astype("int64").to_numpy() // 10**9, issue_seconds[selected], "exact training clock membership")
            completed = int(labels.completed.sum())
            a.check(completed == model["training_completed"] and len(labels) == model["training_labels"], "training counts")
            if completed < 1000:
                a.check(model["status"] == "NOT_FIT_INSUFFICIENT_COMMON_LABELS" and model["models"] is None, "no model/fallback on insufficient labels")
            else: fitted += 1
            details.update(all44_finite_rows=int(finite.sum()), raw_valid_clock_rows=int(avail.raw_feature_valid.sum()),
                transformed_valid_clock_rows=int(avail.transformed_feature_valid.sum()), common_clock_rows=int(avail.common_feature_valid.sum()),
                completed_training_labels=completed, model_status=model["status"],
                always_nonfinite_features=[k for k in names if not np.isfinite(trans[k]).any()])
            reports[symbol][fold] = details
            print(symbol + "/" + fold + ": independent chain, M5 age and availability checked", flush=True)
        del data, times, prices, increments, features, trans, raw
    expected = {(s, ep, v, p) for s in ("BOOM600", "CRASH600") for ep in ("REGION", "TIMED")
        for v in ("CLOCK", "RAW44", "TRANSFORMED44") for p in ("wf1", "wf2", "wf3", "final_test", "later180", "walk_forward_combined")}
    a.check(len(r["rows"]) == 72 and {(x["symbol"], x["endpoint"], x["variant"], x["partition"]) for x in r["rows"]} == expected, "complete72identity grid")
    no_common = all(x["common_clock_rows"] == 0 for report in reports.values() for x in report.values())
    for row in r["rows"]:
        a.check(row["QUALIFIED"] is False, "unqualified metric row")
        for artifact in row["artifacts"].values():
            path = a.pin(artifact["path"], artifact["sha256"])
            table = pd.read_csv(path)
            a.check(len(table) == artifact["rows"], "artifact actual population")
            if no_common: a.check(len(table) == 0, "no invented issuance/event/ledger rows")
        if no_common:
            # This narrowly certifies the actually empty experiment, never a
            # positive/negative payoff claim for an untrained strategy.
            a.check(row["metrics"]["completed"] == 0 and row["metrics"]["profit_factor"] is None, "empty experiment has no PF")
            a.check(row["metrics"]["trades"] == 0, "no invented trades")
            a.check(row["metrics"]["mean_net_R"] is None, "no invented mean")
            a.check(row["day_inference"]["mean_net_R_ci95"] == [None, None], "empty day interval")
            a.check(row["weekly_inference"]["weekly_mean_net_R_ci95"] == [None, None], "empty weekly interval")
            a.check(row["profit_factor_inference"]["weekly_profit_factor_ci95"] == [None, None], "empty PF interval")
            if row["variant"] != "CLOCK" and row["partition"] in ("final_test", "later180"):
                a.check(row["qualification"]["historical_criteria_passed"] is False and row["holm_p"] == 1., "empty evidence cannot pass")
    for name, pin in list(a.pins.items()): a.check(sha(ROOT / name) == pin, "unchanged post-audit pin " + name)
    return {"stage": "independent_representation_availability_audit", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not a.errors, "checks": a.checks, "errors": a.errors, "symbols": reports,
        "fitted_models_requiring_separate_economic_audit": fitted,
        "all_common_clocks_empty": no_common,
        "oracle_longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "oracle_uses_extended_precision": bool(np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant),
        "scope": "Training detector median; independent NumPy chain; all M5 close/large-bar/age; paired availability/membership; insufficient-fit decisions; empty result grid when all common clocks empty",
        "not_tested": ["other43 feature formulas independently regenerated", "fitted coefficient/path/CI audit", "broker execution/costs", "fresh OOS"],
        "declaration_sha256": DECLARATION, "verifier_sha256": sha(Path(__file__)), "input_pins": a.pins,
        "safety": dict.fromkeys(FLAGS, False), "QUALIFIED": False}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise ValueError("Do not overwrite an audit attempt")
    r = audit_study()
    with args.output.open("x") as f: json.dump(r, f, indent=2, allow_nan=False); f.write("\n")
    with args.output.with_suffix(".sha256").open("x") as f: f.write(sha(args.output) + "\n")
    print(json.dumps({k: r[k] for k in ("passed", "checks", "errors", "fitted_models_requiring_separate_economic_audit")}))
    if not r["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
