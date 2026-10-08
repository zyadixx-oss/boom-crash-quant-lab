#!/usr/bin/env python3
"""Independent original-quote audit of the twelve saved zone label definitions.

No research or diagnostic engine is imported. This verifies saved opportunity
membership, every label/first passage, point metrics, all day/week intervals,
and the two declared Holm families. It does not regenerate strategy features,
test broker fills, or promote the already exposed history to fresh OOS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs/zone_excursions_20261008"
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
VARIANTS = ("CRT_RETEST", "FIB_RETRACE", "TREND_RETEST", "CONTEXT_GEOMETRIC")
CONTROL = VARIANTS[-1]
DECLARATION = "3e423cc9d4ff413d0cb1e1aad03292dced2915d3b7bd478589876534dd4c903d"
DEFINITIONS = tuple((a, h) for a in (1.5, 2., 3.) for h in (5, 10, 15, 30))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def second(value):
    t = pd.Timestamp(value)
    if t.tz is None or t.value % 10**9:
        raise ValueError("A whole aware second is required")
    return t.value // 10**9


class Audit:
    def __init__(self):
        self.checks, self.errors, self.pins = 0, [], {}

    def check(self, value, message):
        self.checks += 1
        if not bool(value) and len(self.errors) < 100: self.errors.append(message)

    def equal(self, actual, expected, message):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), message + ": keys")
            for k, value in expected.items(): self.equal(actual.get(k), value, message + "/" + k)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message + ": length")
            if isinstance(actual, (list, tuple)):
                for i, (x, y) in enumerate(zip(actual, expected)): self.equal(x, y, f"{message}/{i}")
        elif expected is None:
            self.check(actual is None or (not isinstance(actual, (list, dict)) and pd.isna(actual)), message)
        elif isinstance(expected, (float, np.floating)):
            self.check(isinstance(actual, (int, float, np.number)) and math.isclose(float(actual), float(expected),
                abs_tol=2e-11, rel_tol=2e-11), message)
        else:
            self.check(actual == expected, message)

    def pin(self, name, expected=None):
        p = (ROOT / name).resolve()
        p.relative_to(ROOT.resolve())
        actual = sha(p)
        self.check(expected is None or actual == expected, "pin " + str(name))
        if expected is not None and actual != expected: raise ValueError("Changed pinned input: " + str(name))
        self.pins[str(p.relative_to(ROOT))] = actual
        return p


def oracle(times, quotes, row, start, end, endpoint):
    """Resolve a dense expected clock independently of the production slices."""
    issue, anchor = second(row["issue_time"]), second(row["anchor_time"])
    side, atr = int(row["side"]), float(row["atr"])
    if issue % 60 or side not in (-1, 1) or not math.isfinite(atr) or atr <= 0:
        raise ValueError("Invalid saved opportunity")
    if (endpoint == "ISSUE_DELAYED" and anchor != issue + 61) or (
            endpoint == "ENTRY_CONDITIONAL" and not issue + 62 <= anchor <= issue + 901):
        raise ValueError("Saved anchor changes frozen chronology")
    reason = "outside_partition" if not start <= issue < end else (
        "planned_purge" if issue + (1920 if endpoint == "ISSUE_DELAYED" else 2760) > end else None)
    result = {"anchor_price": None}
    for h in (5, 10, 15, 30):
        why, window = reason, None
        if why is None:
            wanted = np.arange(anchor, anchor + h * 60 + 1, dtype=np.int64)
            positions = np.searchsorted(times, wanted)
            safe = positions < len(times)
            present = np.zeros(len(wanted), dtype=bool)
            present[safe] = times[positions[safe]] == wanted[safe]
            present &= (wanted >= start) & (wanted < end)
            if not present[0]: why = "missing_anchor"
            else:
                result["anchor_price"] = float(quotes[positions[0]])
                why = "known_full_horizon" if present.all() else "incomplete_full_horizon"
                if present.all(): window = quotes[positions[1:]]
        result[f"reason_h{h:02d}"] = why
        for a in (1.5, 2., 3.):
            key = f"a{a:g}_h{h:02d}"
            hit, when = -1, None
            if window is not None:
                crossed = np.flatnonzero(side * (window - result["anchor_price"]) >= a * atr)
                hit = int(len(crossed) > 0)
                if hit: when = float(crossed[0] + 1)
            result["y_" + key], result["tts_" + key] = hit, when
    return result


def read_archive(audit, artifact):
    p = audit.pin(artifact["path"], artifact["sha256"])
    payload = gzip.decompress(p.read_bytes())
    audit.equal(len(payload), artifact["uncompressed_bytes"], str(p) + " size")
    audit.equal(hashlib.sha256(payload).hexdigest(), artifact["uncompressed_sha256"], str(p) + " uncompressed hash")
    return payload


def expected_issue(candidates, start, end):
    c = candidates[candidates.variant.eq(CONTROL)]
    c = c[c.reason.isin(["issued_fixed_region", "h4_h1_context_disagrees", "m5_trigger_absent"])]
    c = c.assign(issue_time=pd.to_datetime(c.signal_time, utc=True))
    c = c[c.issue_time.ge(start) & c.issue_time.lt(end)]
    out = c[["issue_time", "atr", "side"]].copy()
    out["anchor_time"] = out.issue_time + pd.Timedelta(seconds=61)
    for v in VARIANTS:
        allowed = candidates[candidates.variant.eq(v) & candidates.eligible].signal_time
        out["eligible_" + v] = out.issue_time.isin(pd.to_datetime(allowed, utc=True))
    return out.reset_index(drop=True)


def expected_entry(ledger):
    return ledger[["signal_time", "entry_time", "atr", "side"]].rename(columns={
        "signal_time": "issue_time", "entry_time": "anchor_time"}).reset_index(drop=True)


def check_membership(audit, frame, expected, name):
    audit.equal(len(frame), len(expected), name + " rowcount")
    if len(frame) != len(expected): raise ValueError("Saved population mismatch")
    for k in expected:
        if k.endswith("time"):
            audit.check(np.array_equal(pd.to_datetime(frame[k], utc=True).astype("int64"),
                pd.to_datetime(expected[k], utc=True).astype("int64")), name + "/" + k)
        elif k == "atr": audit.check(np.allclose(frame[k].to_numpy(dtype=float), expected[k].to_numpy(dtype=float),
            atol=1e-12, rtol=1e-12), name + "/atr")
        else: audit.check(np.array_equal(frame[k], expected[k]), name + "/" + k)
    audit.check(not frame.duplicated(["issue_time", "anchor_time"]).any(), name + " no duplicate anchors")


def sampling(start, end, repeats=9999, seed=20261008):
    first = pd.Timestamp(start).floor("D")
    n = int(((pd.Timestamp(end) - pd.Timedelta(seconds=1)).floor("D") - first).days + 1)
    day = np.random.default_rng(seed).multinomial(n, np.repeat(1 / n, n), size=repeats)
    span = min(7, n)
    starts = np.random.default_rng(seed).integers(n, size=(repeats, math.ceil(n / span)))
    # Resample the daily vectors directly, rather than using the engine's weights.
    week_ids = ((starts[..., None] + np.arange(span)) % n).reshape(repeats, -1)[:, :n]
    return first, n, day, week_ids


def fraction(x, y):
    out = np.full(np.shape(x), np.nan)
    np.divide(x, y, out=out, where=np.asarray(y) > 0)
    return out


def interval(values):
    values = values[np.isfinite(values)]
    return np.percentile(values, [2.5, 97.5]).tolist() if len(values) else [None, None]


def population(frame, a, h, calendar):
    first, n, day, week = calendar
    y = frame[f"y_a{a:g}_h{h:02d}"].to_numpy(int)
    planned = ~frame[f"reason_h{h:02d}"].isin(["planned_purge", "outside_partition"]).to_numpy()
    known, hit = planned & (y >= 0), planned & (y == 1)
    ids = ((pd.to_datetime(frame.issue_time, utc=True).dt.floor("D") - first).dt.days).to_numpy(int)
    if np.any((ids < 0) | (ids >= n)): raise ValueError("Saved anchor outside calendar")
    counts = np.bincount(ids[known], minlength=n)
    successes = np.bincount(ids[hit], minlength=n)
    issued, total, k = int(planned.sum()), int(known.sum()), int(hit.sum())
    unknown = issued - total
    p = {"offered": len(frame), "planned": issued, "purged": len(frame) - issued,
         "known": total, "unknown": unknown, "hits": k, "active_known_days": int(np.count_nonzero(counts)),
         "rate": k / total if total else None,
         "rate_bounds_with_unknown": [k / issued, (k + unknown) / issued] if issued else [None, None]}
    samples = {"day": (np.sum(day * successes, axis=1), np.sum(day * counts, axis=1)),
               "week": (successes[week].sum(axis=1), counts[week].sum(axis=1))}
    hits = frame.loc[hit]
    tts = hits[f"tts_a{a:g}_h{h:02d}"].to_numpy(float)
    delay = (pd.to_datetime(hits.anchor_time, utc=True) - pd.to_datetime(hits.issue_time, utc=True)).dt.total_seconds().to_numpy(float)
    return p, samples, (float(np.median(tts) / 60) if len(tts) else None,
                        float(np.median(tts + delay) / 60) if len(tts) else None)


def independent_metrics(model, baseline, recall):
    m, ms, medians = model
    b, bs, _ = baseline
    precision, base = m["rate"], b["rate"]
    lift = precision / base if precision is not None and base else None
    delta = precision - base if precision is not None and base is not None else None
    inference = {}
    for kind in ("day", "week"):
        mp, bp = fraction(*ms[kind]), fraction(*bs[kind])
        differences, lifts = mp - bp, fraction(mp, bp)
        r = fraction(ms[kind][0], bs[kind][0]) if recall else np.full(len(mp), np.nan)
        p = 1. if delta is None or delta <= 0 else float((1 + np.count_nonzero(
            (~np.isfinite(differences)) | (differences - delta >= delta))) / (len(mp) + 1))
        inference[kind] = {"precision_ci95": interval(mp), "base_rate_ci95": interval(bp),
            "lift_ci95": interval(lifts), "difference_ci95": interval(differences),
            "opportunity_recall_ci95": interval(r), "p": p,
            "valid_lift_replicates": int(np.count_nonzero(np.isfinite(lifts))),
            "undefined_lift_replicates": int(np.count_nonzero(~np.isfinite(lifts)))}
    ml, mu = m["rate_bounds_with_unknown"]
    bl, bu = b["rate_bounds_with_unknown"]
    return {"model": m, "baseline": b, "precision": precision, "base_rate": base, "lift": lift,
        "difference": delta, "opportunity_recall": m["hits"] / b["hits"] if recall and b["hits"] else None,
        "unique_event_recall": None,
        "recall_interpretation": "positive_subset_clock_opportunities" if recall else "not_identified_for_different_entry_anchors",
        "lift_bounds_with_unknown": [ml / bu if ml is not None and bu else None, mu / bl if mu is not None and bl else None],
        "median_time_to_excursion_minutes_from_anchor": medians[0],
        "median_time_to_excursion_minutes_from_issue": medians[1], "inference": inference,
        "p": max(inference["day"]["p"], inference["week"]["p"]), "QUALIFIED": False,
        "profit_factor": "UNCHANGED_SEPARATE_ECONOMIC_RESULT"}


def holm(values):
    order = np.argsort(values, kind="stable")
    out, previous = np.zeros(len(values)), 0.
    for rank, index in enumerate(order):
        previous = max(previous, (len(values) - rank) * values[index])
        out[index] = min(previous, 1.)
    return out


def execute():
    audit = Audit()
    audit.pin("scripts/verify_zone_excursions.py")
    audit.pin("backend/tests/test_region_excursion_audit.py")
    for flag in FLAGS:
        if os.environ.get(flag, "false").lower() != "false": raise ValueError("Offline-only audit: " + flag)
    declaration = json.loads(audit.pin(str((FOLDER / "declaration.json").relative_to(ROOT)), DECLARATION).read_text())
    audit.equal((FOLDER / "declaration.sha256").read_text().strip(), DECLARATION, "declaration sidecar")
    for key in ("code_sha256", "original_inputs"):
        for name, pin in declaration[key].items(): audit.pin(name, pin)
    index = json.loads(audit.pin(str((FOLDER / "result_index.json").relative_to(ROOT))).read_text())
    result = json.loads(read_archive(audit, index))
    audit.equal(len(result["rows"]), 2016, "full metric grid")
    audit.equal(len(result["artifacts"]), 60, "full label grid")
    for object_ in (declaration, result, index):
        audit.equal(object_["safety"], dict.fromkeys(FLAGS, False), "false safety gates")
        audit.equal(object_["QUALIFIED"], False, "never qualify")
    audit.equal(result["declaration_sha256"], DECLARATION, "result declaration")
    audit.equal(result["economic_results_unchanged"], True, "unchanged economics")
    original = json.loads((ROOT / "docs/zone_study_20261008/declaration.json").read_text())
    for name, pin in original["code_sha256"].items(): audit.pin(name, pin)
    economic = json.loads((ROOT / "docs/zone_study_20261008/results.json").read_text())
    audit.equal(result["economic_result_sha256"], sha(ROOT / "docs/zone_study_20261008/results.json"), "economic hash")
    rows_by_key = {(r["symbol"], r["variant"], r["partition"]): r for r in economic["rows"]}
    bounds = declaration["specification"]["partitions"]
    all_bounds = {**bounds, "walk_forward_combined": [bounds["wf1"][0], bounds["wf3"][1]]}
    calendars = {p: sampling(*b) for p, b in all_bounds.items()}
    rebuilt_p, identities, labels_checked = {}, set(), 0
    for symbol in ("BOOM600", "CRASH600"):
        arrays = []
        for source in original["sources"]["symbols"][symbol]:
            path = audit.pin(source["path"], source["sha256"])
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            audit.equal(len(data), source["rows"], source["path"] + " rowcount")
            audit.check(data.shape[1] == 2 and np.all(data[:, 0] == data[:, 0].astype(np.int64)) and
                np.all(np.isfinite(data)) and np.all(data[:, 1] > 0), source["path"] + " raw quote contract")
            arrays.append(data)
        data = np.concatenate(arrays)
        del arrays
        times, prices = data[:, 0].astype(np.int64), data[:, 1]
        audit.check(np.all(np.diff(times) > 0), symbol + " original monotonic native seconds")
        artifact = economic["symbols"][symbol]["candidate_artifact"]
        candidates = pd.read_csv(audit.pin(artifact["path"], artifact["sha256"]), low_memory=False)
        frames = {}
        for artifact in result["artifacts"]:
            if artifact["symbol"] != symbol: continue
            endpoint, part = artifact["endpoint"], artifact["partition"]
            start, end = bounds[part]
            frame = pd.read_csv(io.BytesIO(read_archive(audit, artifact)))
            key = (endpoint, part, artifact.get("variant", "AVAILABILITY_CLOCK"))
            audit.check(key not in frames, "distinct label artifact " + str(key))
            if endpoint == "ISSUE_DELAYED": expected = expected_issue(candidates, start, end)
            else:
                ledger = rows_by_key[(symbol, artifact["variant"], part)]["artifacts"]["ledger"]
                expected = expected_entry(pd.read_csv(audit.pin(ledger["path"], ledger["sha256"])))
            check_membership(audit, frame, expected, artifact["path"])
            for row in frame.to_dict("records"):
                label = oracle(times, prices, row, second(start), second(end), endpoint)
                for field, value in label.items(): audit.equal(row[field], value, artifact["path"] + "/" + field)
                labels_checked += len(DEFINITIONS)
            frames[key] = frame
            if endpoint == "ISSUE_DELAYED":
                for variant in VARIANTS:
                    frames[(endpoint, part, variant)] = frame[frame["eligible_" + variant]].copy()
        for endpoint in ("ISSUE_DELAYED", "ENTRY_CONDITIONAL"):
            for variant in (("AVAILABILITY_CLOCK", *VARIANTS) if endpoint == "ISSUE_DELAYED" else VARIANTS):
                frames[(endpoint, "walk_forward_combined", variant)] = pd.concat(
                    [frames[(endpoint, f, variant)] for f in ("wf1", "wf2", "wf3")], ignore_index=True)
        populations = {}
        for row in result["rows"]:
            if row["symbol"] != symbol: continue
            ep, part, variant, reference = (row[k] for k in ("endpoint", "partition", "variant", "baseline_variant"))
            a, h = row["atr_multiplier"], row["horizon_minutes"]
            identity = (symbol, ep, part, variant, reference, a, h)
            audit.check(identity not in identities, "unique metric identity " + str(identity))
            identities.add(identity)
            audit.equal([row["start"], row["end_exclusive"]], all_bounds[part], str(identity) + " calendar")
            if (a, h) not in DEFINITIONS: raise ValueError("Undeclared endpoint")
            for v in (variant, reference):
                key = (ep, part, v, a, h)
                if key not in populations: populations[key] = population(frames[(ep, part, v)], a, h, calendars[part])
            m = independent_metrics(populations[(ep, part, variant, a, h)],
                populations[(ep, part, reference, a, h)], ep == "ISSUE_DELAYED")
            audit.equal({k: row.get(k) for k in m}, m, str(identity))
            rebuilt_p[identity] = m["p"]
        print(symbol + ": every saved label, first passage and reported interval checked", flush=True)
        del times, prices, data, populations, frames, candidates
    for endpoint in ("ISSUE_DELAYED", "ENTRY_CONDITIONAL"):
        groups = {}
        for identity, p in rebuilt_p.items():
            symbol, ep, part, v, ref, a, h = identity
            if ep == endpoint and part in ("final_test", "later180") and v != CONTROL:
                groups.setdefault((symbol, part, v, a, h), []).append((identity, p))
        audit.equal(len(groups), 144, endpoint + " full multiplicity family")
        adjusted = holm([max(p for _, p in values) for values in groups.values()])
        published = {(r["symbol"], r["endpoint"], r["partition"], r["variant"], r["baseline_variant"],
            r["atr_multiplier"], r["horizon_minutes"]): r for r in result["rows"]}
        for values, p in zip(groups.values(), adjusted):
            for identity, _ in values: audit.equal(published[identity]["holm_p"], float(p), "Holm " + str(identity))
    for name, pin in list(audit.pins.items()): audit.equal(sha(ROOT / name), pin, "post-audit pin " + name)
    return {"stage": "independent_full_zone_excursion_audit", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not audit.errors, "checks": audit.checks, "errors": audit.errors,
        "definition_labels_checked": labels_checked, "metric_cells_checked": len(identities),
        "all_day_and_week_intervals_checked": True, "all_holm_families_checked": True,
        "scope": "Saved membership, all12 labels/first passage, point/bound/recall/time statistics, all paired day/week CI and Holm144x2",
        "not_tested": ["independent original feature regeneration", "fresh OOS", "measured broker fills/costs", "new profits"],
        "declaration_sha256": DECLARATION, "verifier_sha256": sha(Path(__file__)),
        "input_pins": audit.pins, "safety": dict.fromkeys(FLAGS, False), "QUALIFIED": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("Preserve all previous audit attempts")
    result = execute()
    with args.output.open("x") as f: json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    with args.output.with_suffix(".sha256").open("x") as f: f.write(sha(args.output) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "checks", "errors", "definition_labels_checked", "metric_cells_checked")}))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
