"""Synthetic scalar and contract guards; never read historical source files."""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json

import pytest

from scripts import verify_spike_short_tick as v


def text(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(sep=" ")


def signal(issue=1800, side=1):
    return {"signal_time": issue, "atr": 1., "side": side, "variant": "SHORT44_SPIKE"}


def quotes(issue=1800, price=100.):
    return {t: price for t in range(issue + 61, issue + 122)}


def replay(ticks=None, signals=None, start=1800, end=4000):
    return v.tick_replay(quotes() if ticks is None else ticks,
                         [signal()] if signals is None else signals, start, end)


def descriptor(tmp_path, monkeypatch, name, rows, columns):
    monkeypatch.setattr(v, "ROOT", tmp_path)
    path = tmp_path / name
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
            "columns": columns, "local_ignored_artifact": True}


@pytest.mark.parametrize("bad", ["", ".", "..", "/tmp/a", "a/../b", "a//b", "./a", "a\\b", "a/.", 3, None])
def test_paths_refuse_noncanonical_forms(bad):
    with pytest.raises(ValueError):
        v.relative_path(bad)


def test_path_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(v, "ROOT", root)
    with pytest.raises(ValueError, match="symlink"):
        v.relative_path("escape/file")


@pytest.mark.parametrize("actual,expected", [(True, 1), (False, 0), (1, True), (0, False), (1., 1)])
def test_canonical_metadata_rejects_type_aliases(actual, expected):
    with pytest.raises(ValueError):
        v.exact(v.Audit(), "typed", {"value": actual}, {"value": expected})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_comparison_is_failure(value):
    audit = v.Audit(); audit.equal("nonfinite", value, value)
    assert audit.failures


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("rows", True), ("columns", ["x", "x"]),
                                         ("local_ignored_artifact", 1)])
def test_artifact_descriptor_corruption_refused(tmp_path, monkeypatch, field, value):
    desc = descriptor(tmp_path, monkeypatch, "item.csv", [{"x": 1}], ["x"])
    desc[field] = value
    with pytest.raises(ValueError):
        v.csv_rows(desc, v.Audit())


def test_artifact_good(tmp_path, monkeypatch):
    desc = descriptor(tmp_path, monkeypatch, "item.csv", [{"x": 1}], ["x"])
    audit = v.Audit()
    assert v.csv_rows(desc, audit) == [{"x": "1"}]
    assert not audit.failures


@pytest.mark.parametrize("symbol,mode,side", [("BOOM600", "SPIKE", 1), ("BOOM600", "DRIFT", -1),
                                           ("CRASH600", "SPIKE", -1), ("CRASH600", "DRIFT", 1)])
def test_frozen_native_direction_identity(symbol, mode, side):
    assert v.side_for(symbol, mode) == side


def saved_signal(**updates):
    result = {"signal_time": text(1800), "atr": "1.0", "side": "1", "signal_close": "100.0",
              "variant": "SHORT44_SPIKE", "score": "0.2"}
    result.update(updates); return result


@pytest.mark.parametrize("updates", [{"signal_time": text(1801)}, {"atr": "0"}, {"side": "-1"},
                                    {"side": "1.0"}, {"side": True}, {"variant": "LONG44_SPIKE"},
                                    {"score": "0"}, {"score": "nan"}, {"signal_close": "-1"},
                                    {"signal_time": "1970-01-01T00:30:00"},
                                    {"signal_time": "1970-01-01T03:30:00+03:00"}])
def test_issuance_invalid_or_cross_family_refused(updates):
    with pytest.raises(ValueError):
        v.signals_from_rows([saved_signal(**updates)], "BOOM600", "SHORT44_SPIKE")


def test_issuance_schema_positive_score_and_sorted_membership():
    rows = v.signals_from_rows([saved_signal()], "BOOM600", "SHORT44_SPIKE")
    assert rows == [{"signal_time": 1800, "atr": 1., "side": 1, "signal_close": 100.,
                     "variant": "SHORT44_SPIKE", "score": .2}]
    with pytest.raises(ValueError):
        v.signals_from_rows([saved_signal(), saved_signal()], "BOOM600", "SHORT44_SPIKE")
    clock = saved_signal(variant="CLOCK_SPIKE", score="")
    assert "score" not in v.signals_from_rows([clock], "BOOM600", "CLOCK_SPIKE")[0]


def test_short_exact_entry_nominal_timeout_and_cost():
    ticks = quotes(); ticks[1921] = 100.8
    rows, accounting = replay(ticks)
    row = rows[0]
    assert row["nominal_entry_time"] == 1860 and row["entry_time"] == 1861
    assert row["planned_end"] == 1920 and row["exit_time"] == 1921
    assert row["holding_minutes"] == 1.
    assert row["gross_R"] == pytest.approx(.4) and row["net_R"] == pytest.approx(.35)
    assert accounting["completed"] == accounting["filled"] == 1


def test_actual_entry_plus_hold_cannot_move_nominal_expiry():
    ticks = quotes(); ticks[1922] = 9999.
    row = replay(ticks)[0][0]
    assert row["planned_end"] == 1920 and row["exit_time"] == 1921
    assert row["exit"] == 100.


@pytest.mark.parametrize("side,trigger,exit_quote,gross", [(1, 97., 96., -2.), (-1, 103., 105., -2.5)])
def test_stop_uses_strict_successor_without_barrier_cap(side, trigger, exit_quote, gross):
    ticks = quotes(); ticks[1870], ticks[1871] = trigger, exit_quote
    rows, counts = replay(ticks, [signal(side=side)])
    row = rows[0]
    assert row["reason"] == "sl" and row["trigger_time"] == 1870 and row["exit_time"] == 1871
    assert row["exit"] == exit_quote and row["gross_R"] == gross and row["net_R"] == gross - .05
    assert counts["completed"] == 1


def test_stop_at_expiry_precedes_timeout():
    ticks = quotes(); ticks[1920], ticks[1921] = 98., 97.
    row = replay(ticks)[0][0]
    assert row["reason"] == "sl" and row["trigger_time"] == 1920 and row["exit_time"] == 1921


def test_postexpiry_crossing_is_timeout():
    ticks = quotes(); ticks[1921] = 90.
    row = replay(ticks)[0][0]
    assert row["reason"] == "time" and row["trigger_time"] is None and row["exit"] == 90.


@pytest.mark.parametrize("missing", [1862, 1880, 1920, 1921])
def test_required_missing_second_censors_without_return(missing):
    ticks = quotes(); del ticks[missing]
    rows, counts = replay(ticks); row = rows[0]
    assert row["censored"] and row["missing_time"] == missing
    assert row["exit"] is row["gross_R"] is row["net_R"] is None
    assert row["planned_end"] == row["exit_time"] == 1920
    assert counts["missing_path"] == counts["censored"] == 1 and counts["completed"] == 0


def test_missing_successor_censors_stop_even_if_trigger_observed():
    ticks = quotes(); ticks[1870] = 97.; del ticks[1871]
    row = replay(ticks)[0][0]
    assert row["censored"] and row["trigger_time"] == 1870 and row["missing_time"] == 1871
    assert row["net_R"] is None


def test_gaps_after_completed_exit_do_not_censor():
    ticks = quotes(); ticks[1870] = 98.
    del ticks[1900]
    row = replay(ticks)[0][0]
    assert not row["censored"] and row["exit_time"] == 1871


def test_missing_entry_cannot_use_quote_later_than_maxgap():
    ticks = quotes(); del ticks[1861]
    rows, counts = replay(ticks)
    assert rows == [] and counts["missing_entry"] == 1 and counts["filled"] == 0


def test_missing_entry_reserves_nominal_occupancy():
    ticks = quotes(); del ticks[1861]
    ticks.update(quotes(1810))
    _, counts = replay(ticks, [signal(), signal(1810)])
    assert counts["missing_entry"] == 1 and counts["overlap_skipped"] == 1


def test_unknown_path_reserves_nominal_occupancy():
    ticks = quotes(); del ticks[1862]
    ticks.update({t: 100. for t in range(1871, 1932)})
    rows, counts = replay(ticks, [signal(), signal(1810)])
    assert rows[0]["censored"] and counts["overlap_skipped"] == 1


def test_known_exit_releases_actual_occupancy():
    ticks = quotes(); ticks[1862] = 98.
    ticks.update({t: 100. for t in range(1871, 1932)})
    rows, counts = replay(ticks, [signal(), signal(1810)])
    assert rows[0]["exit_time"] == 1863 and counts["completed"] == 2


def test_planned_purge_cannot_be_rescued_by_early_exit():
    ticks = quotes(); ticks[1862] = 98.
    rows, counts = replay(ticks, end=3659)
    assert not rows and counts["purged"] == 1
    rows, counts = replay(ticks, end=3660)
    assert rows and counts["purged"] == 0


def test_same_day_halfhour_maximum_47_after_common_purge():
    start, end = v.tick7.date_bounds(v.DATES[0])
    signals = [signal(t) for t in range(start, end, 1800)]
    ticks = {t: 100. for s in signals for t in range(s["signal_time"] + 61, s["signal_time"] + 122)}
    rows, counts = v.tick_replay(ticks, signals, start, end)
    assert counts["issued"] == 48 and counts["purged"] == 1 and len(rows) == 47
    assert 12 * len(rows) == 564 < 1000


@pytest.mark.parametrize("bad", [{"signal_time": True}, {"atr": 0}, {"atr": float("nan")}, {"side": True}, {"side": 0}])
def test_invalid_scalar_signal_refused(bad):
    item = signal(); item.update(bad)
    with pytest.raises(ValueError):
        replay(signals=[item])


def test_scalar_unsorted_or_duplicate_refused():
    with pytest.raises(ValueError):
        replay(signals=[signal(), signal()])


def metric_row(t, net, **updates):
    result = {"signal_time": t, "entry_time": t + 61, "exit_time": t + 121, "net_R": net,
              "gross_R": net + .05, "censored": False, "ambiguous": False, "holding_minutes": 1.}
    result.update(updates); return result


def test_point_metrics_empty_pf_unknown_and_scheduled_days_retained():
    point = v.point_metrics([])
    assert point["completed"] == point["active_days"] == 0
    assert point["calendar_days"] == 12 and point["span_calendar_days"] == 176
    assert point["profit_factor"] is point["mean_net_R"] is point["win_rate"] is None
    assert point["sum_net_R"] == 0


def test_point_metrics_no_losses_pf_unknown():
    t = v.tick7.date_bounds(v.DATES[0])[0]
    point = v.point_metrics([metric_row(t, 2.)])
    assert point["profit_factor"] is None and point["mean_net_R"] == 2.
    assert point["active_days"] == 1 and point["calendar_days"] == 12


def test_metrics_ignore_unknown_returns_and_preserve_counts():
    t = v.tick7.date_bounds(v.DATES[0])[0]
    rows = [metric_row(t, 2.), metric_row(t + 1800, -1.),
            metric_row(t + 3600, 0., net_R=None, gross_R=None, censored=True)]
    point = v.point_metrics(rows)
    assert point["profit_factor"] == 2. and point["mean_net_R"] == .5
    assert point["trades"] == 3 and point["completed"] == 2 and point["censored"] == 1
    assert point["closed_trade_return"] == pytest.approx((1 + .0025 * 2) * (1 - .0025) - 1)
    assert point["closed_trade_max_drawdown"] == pytest.approx(.0025)


def test_matched_known_is_innerjoin_unknowns_not_zero():
    tick = [metric_row(1800, 2.), metric_row(3600, -1.), metric_row(5400, 0., net_R=None, gross_R=None, censored=True)]
    coarse = [metric_row(1800, 1.), metric_row(5400, 4.)]
    rows, point = v.matched_known(tick, coarse)
    assert rows == [{"signal_time": 1800, "net_R_tick": 2., "net_R_coarse": 1., "delta": 1.}]
    assert point == {"matched_completed": 1, "tick_minus_coarse_mean_R": 1.,
                     "unmatched_tick_completed": 1, "unmatched_coarse_completed": 1}


def test_reference_difference_unequal_exposure_means_not_sums():
    assert v.reference_difference([metric_row(1, 2.)], [metric_row(2, 1.), metric_row(3, 1.)]) == 1.
    assert v.reference_difference([], [metric_row(1, 2.)]) is None


def ci(valid=9999):
    return {"bootstrap_repeats": 9999, "bootstrap_seed": 20261005, "valid": valid,
            "undefined": 9999 - valid, "interval": [1., 2.] if valid else [None, None]}


def check_ci(saved, **kwargs):
    audit = v.Audit()
    v.ci_contract(saved, audit, "ci", "interval", "valid", "undefined", **kwargs)
    return audit


@pytest.mark.parametrize("updates", [{"valid": True}, {"undefined": False}, {"valid": -1}, {"valid": 10000},
                                    {"undefined": 1}, {"interval": [2., 1.]}, {"interval": [None, 1.]},
                                    {"interval": [float("nan"), 1.]}, {"interval": [1.]},
                                    {"bootstrap_repeats": True}, {"bootstrap_seed": 20261005.}])
def test_ci_count_schema_refuses_invalid_contract(updates):
    saved = ci(); saved.update(updates)
    with pytest.raises(ValueError):
        check_ci(saved)


def test_ci_undefined_draws_explicit_without_regeneration_claim():
    assert not check_ci(ci(100)).failures
    assert not check_ci(ci(0), empty=True).failures
    with pytest.raises(ValueError):
        check_ci(ci(), empty=True)
    with pytest.raises(ValueError):
        check_ci(ci(), defined=False)


def test_main_refuses_output_overwrite_without_reading_study(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"; output.write_text("preserved")
    monkeypatch.setattr(v.sys, "argv", ["audit", "--study", "never-read", "--output", str(output)])
    with pytest.raises(FileExistsError):
        v.main()
    assert output.read_text() == "preserved"


def test_internal_symlink_alias_refused(tmp_path, monkeypatch):
    (tmp_path / "real").mkdir(); (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    monkeypatch.setattr(v, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        v.relative_path("alias/file")


@pytest.mark.parametrize("suffix", [".000000001", ".0000000001", ".100000000", ".000001"])
def test_subsecond_timestamps_cannot_be_truncated(suffix):
    with pytest.raises(ValueError, match="Whole"):
        v.stamp("2026-04-11 00:30:00" + suffix + "+00:00")


def test_zero_nanosecond_utc_representation_preserves_exact_time():
    assert v.stamp("2026-04-11 00:30:00.000000000+00:00") == v.stamp("2026-04-11T00:30:00Z")


@pytest.mark.parametrize("score", ["1", "0", "nan", True])
def test_clock_blank_score_only(score):
    with pytest.raises(ValueError, match="CLOCK score"):
        v.signals_from_rows([saved_signal(variant="CLOCK_SPIKE", score=score)], "BOOM600", "CLOCK_SPIKE")


def test_clock_six_column_source_and_copied_rows(tmp_path, monkeypatch):
    columns = v.SIGNAL_COLUMNS
    desc = descriptor(tmp_path, monkeypatch, "clock.csv", [saved_signal(variant="CLOCK_SPIKE", score="")], columns)
    rows = v.csv_rows(desc, v.Audit())
    assert v.signals_from_rows(rows, "BOOM600", "CLOCK_SPIKE")[0]["signal_time"] == 1800


@pytest.mark.parametrize("identity", [{"frozen_score_threshold": .3, "training_end": text(0)},
                                     {"frozen_score_threshold": .1, "training_end": text(1801)}])
def test_saved_positive_score_must_pass_frozen_cutoff_and_fit_chronology(identity):
    with pytest.raises(ValueError, match="threshold/training"):
        v.signals_from_rows([saved_signal()], "BOOM600", "SHORT44_SPIKE", identity)


@pytest.mark.parametrize("line", ["1,2,3\n", "1\n"])
def test_ragged_csv_row_refused_even_with_matching_descriptor_hash(tmp_path, monkeypatch, line):
    desc = descriptor(tmp_path, monkeypatch, "ragged.csv", [], ["a", "b"])
    path = tmp_path / "ragged.csv"; path.write_text("a,b\n" + line)
    desc.update(rows=1, sha256=v.digest(path))
    with pytest.raises(ValueError, match="rectangular"):
        v.csv_rows(desc, v.Audit())


def test_frozen_document_requires_hash_and_four_false_values(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "ROOT", tmp_path)
    document = {"safety": dict.fromkeys(v.FLAGS, False)}
    path = tmp_path / "declaration.json"; path.write_text(json.dumps(document))
    path.with_suffix(".sha256").write_text(v.digest(path))
    assert v.frozen_document(tmp_path, "declaration", v.Audit()) == document
    document["safety"][v.FLAGS[0]] = 0
    path.write_text(json.dumps(document)); path.with_suffix(".sha256").write_text(v.digest(path))
    with pytest.raises(ValueError):
        v.frozen_document(tmp_path, "declaration", v.Audit())


def test_prepared_sparse_m1_audit_counts_actual_grid_not_available_rows():
    minutes = {0: (1., 1., 1., 1.), 120: (1., 1., 1., 1.)}
    source = {"path": "nested/synthetic.csv", "sha256": "abc"}
    result = v.prepared_audit(source, minutes)
    assert result["source_rows"] == 2 and result["grid_minutes"] == 3
    assert result["missing_minutes"] == result["gap_intervals"] == 1


def synthetic_runner_context(tmp_path, monkeypatch):
    import pandas as pd
    from scripts import run_spike_short_tick as runner
    monkeypatch.setattr(v, "ROOT", tmp_path); monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "docs/synthetic_short_tick"; output.mkdir(parents=True)
    tick_days, coarse_days, tas, cas = [], [], [], []
    audit = v.Audit()
    for i, date in enumerate(v.DATES):
        start, end = v.tick7.date_bounds(date); issue = start + 1800
        scalar_signals = [{"signal_time": issue, "atr": 1., "side": 1,
                           "signal_close": 100., "variant": "SHORT44_SPIKE", "score": .2}]
        source = {t: 100. + (t-issue-60) * (.01 if i % 2 else -.01) for t in range(issue+60, issue+122)}
        if i == 1:
            source[issue+70], source[issue+71] = 97., 96.
        tick_scalars = dict(source)
        if i == 2:
            del tick_scalars[issue+100]
        if i == 3:
            del tick_scalars[issue+61]
        ticks = pd.DataFrame({"quote": list(tick_scalars.values())},
                             index=pd.to_datetime(list(tick_scalars), unit="s", utc=True))
        q = [source[t] for t in range(issue+60, issue+120)]
        minute_tuple = (q[0], max(q), min(q), q[-1])
        minutes = {issue+60: minute_tuple}
        m1 = pd.DataFrame([minute_tuple], columns=["open", "high", "low", "close"],
                          index=pd.to_datetime([issue+60], unit="s", utc=True))
        records = [{**s, "signal_time": pd.Timestamp(s["signal_time"], unit="s", tz="UTC")} for s in scalar_signals]
        tick, ta, coarse, ca = runner.replay_day(ticks, m1, records, date)
        independent_tick, ita = v.tick_replay(tick_scalars, scalar_signals, start, end)
        independent_coarse, ica = v.old.replay(minutes, scalar_signals, start, end, v.COARSE_CONFIG)
        for proxy, frame, counts, independent, independent_counts, columns in (
                ("tick", tick, ta, independent_tick, ita, v.TICK_COLUMNS),
                ("coarse", coarse, ca, independent_coarse, ica, v.COARSE_COLUMNS)):
            name = f"{date}_{proxy}"
            desc = runner.dump_frame(output, "ledgers", name, frame)
            v.audit_ledger(desc, independent, audit, name, columns)
            v.audit_counts(counts, independent_counts, audit, name, tick=proxy == "tick")
            v.audit_metrics(runner.summary(frame, bootstrap=False, date=date), independent, audit, name, date)
        matched, point = runner.matched_known(tick, coarse, ta, ca, bootstrap=False)
        desc = runner.dump_frame(output, "ledgers", f"{date}_matched", matched)
        v.audit_matched({"ledger": desc, "metrics": point}, independent_tick, independent_coarse,
                        ita, ica, audit, date)
        tick_days.append(tick); coarse_days.append(coarse); tas.append(ta); cas.append(ca)
    tick, coarse = pd.concat(tick_days, ignore_index=True), pd.concat(coarse_days, ignore_index=True)
    def decoded(frame):
        result = []
        for raw in frame.to_dict("records"):
            row = {}
            for key, value in raw.items():
                if key in v.tick7.TIMES:
                    row[key] = None if pd.isna(value) else int(value.timestamp())
                elif key in v.tick7.NUMBERS:
                    row[key] = None if pd.isna(value) else float(value)
                else:
                    row[key] = value
            result.append(row)
        return result
    tick_rows, coarse_rows = decoded(tick), decoded(coarse)
    return runner, output, tick, coarse, tick_rows, coarse_rows, runner.totals(tas), runner.totals(cas), audit


def test_actual_runner_synthetic_day_and_aggregate_schema_integration(tmp_path, monkeypatch):
    runner, output, tick, coarse, tr, cr, ta, ca, audit = synthetic_runner_context(tmp_path, monkeypatch)
    for proxy, frame, paths in (("tick", tick, tr), ("coarse", coarse, cr)):
        v.audit_metrics(runner.summary(frame), paths, audit, proxy)
    matched, point = runner.matched_known(tick, coarse, ta, ca)
    desc = runner.dump_frame(output, "ledgers", "matched_all", matched)
    v.audit_matched({"ledger": desc, "metrics": point}, tr, cr, ta, ca, audit, "matched_all", aggregate=True)
    v.audit_reference(runner.clock_comparison(tick, coarse, ta, ca), tr, cr, ta, ca, audit, "reference")
    assert ta["censored"] == 1 and ta["missing_entry"] == 1
    assert not audit.failures, audit.failures


@pytest.mark.parametrize("field", ["matched_completed", "unmatched_tick_completed", "policy_unknowns_present"])
def test_matched_summary_corruption_detected(tmp_path, monkeypatch, field):
    runner, output, tick, coarse, tr, cr, ta, ca, audit = synthetic_runner_context(tmp_path, monkeypatch)
    matched, point = runner.matched_known(tick, coarse, ta, ca)
    desc = runner.dump_frame(output, "ledgers", "matched_all", matched)
    point[field] = not point[field] if isinstance(point[field], bool) else point[field] + 1
    v.audit_matched({"ledger": desc, "metrics": point}, tr, cr, ta, ca, audit, "matched_all", aggregate=True)
    assert audit.failures


def test_run_chronology_allows_microseconds_but_quote_time_does_not():
    value = "2026-10-06T21:11:06.013585+00:00"
    assert v.run_time(value).microsecond == 13585
    with pytest.raises(ValueError):
        v.stamp(value)


def synthetic_full_study(tmp_path, monkeypatch):
    """Runner schema integration only; metadata ancestry is tested separately."""
    import pandas as pd
    from scripts import run_spike_short_tick as r
    monkeypatch.setattr(v, "ROOT", tmp_path); monkeypatch.setattr(r, "ROOT", tmp_path)
    study = tmp_path / "docs/synthetic_full"; study.mkdir(parents=True)
    declaration = {"lineage": {"m1_sources": {}, "tick_lineage": {"sources": {}}, "sampled_signals": {}}}
    result = {"symbols": {}}
    flat = []
    for symbol in v.SYMBOLS:
        source_minutes, day_contexts, inputs = {}, {}, {}
        declaration["lineage"]["tick_lineage"]["sources"][symbol] = {}
        declaration["lineage"]["sampled_signals"][symbol] = {}
        for i, date in enumerate(v.DATES):
            start, end = v.tick7.date_bounds(date); issue = start + 1800
            scalar_ticks = {t: 100.+(t-issue-60)*(.01 if i%2 else -.01) for t in range(issue+60, issue+122)}
            tick_frame = pd.DataFrame({"quote": list(scalar_ticks.values())},
                                     index=pd.to_datetime(list(scalar_ticks), unit="s", utc=True))
            tick_frame.index.name = "quote_time"
            qs = [scalar_ticks[t] for t in range(issue+60, issue+120)]
            minute = (qs[0], max(qs), min(qs), qs[-1]); source_minutes[issue+60] = minute
            minute_frame = pd.DataFrame([minute], columns=["open", "high", "low", "close"],
                                       index=pd.to_datetime([issue+60], unit="s", utc=True))
            minute_frame.index.name = "minute_open"
            clean_path = tmp_path / f"{symbol}_{date}_original_ticks.csv"
            pd.DataFrame({"epoch": list(scalar_ticks), "quote": list(scalar_ticks.values())}).to_csv(clean_path, index=False)
            tick_source = {"clean_file": clean_path.name, "rows": len(scalar_ticks), "missing_seconds": 86400-len(scalar_ticks),
                           "gap_free": False}
            declaration["lineage"]["tick_lineage"]["sources"][symbol][date] = tick_source
            inputs[date] = {"tick_source": tick_source,
                "ticks": r.dump_frame(study, "inputs", f"{symbol}_{date}_ticks", tick_frame, index=True),
                "m1": r.dump_frame(study, "inputs", f"{symbol}_{date}_m1", minute_frame.reindex(
                    pd.date_range(pd.Timestamp(start, unit="s", tz="UTC"), periods=1440, freq="min", name="minute_open")), index=True),
                "reconciliation": r.round7.reconcile(tick_frame, minute_frame, date)}
            day_contexts[date] = issue, tick_frame, minute_frame
        path = tmp_path / f"{symbol}_m1.csv"
        pd.DataFrame([{"epoch": t, **dict(zip(("open", "high", "low", "close"), q))} for t, q in source_minutes.items()]).to_csv(path, index=False)
        source = {"path": path.name, "sha256": v.digest(path), "rows": len(source_minutes),
                  "first_epoch": min(source_minutes), "last_epoch": max(source_minutes),
                  "declared_missing_minutes": (max(source_minutes)-min(source_minutes))//60+1-len(source_minutes)}
        declaration["lineage"]["m1_sources"][symbol] = source
        models, frames = {}, {}
        for family in v.FAMILIES:
            for mode in v.MODES:
                key = f"{family}_{mode}"; days, ticks, coarse, tas, cas, records = [], [], [], [], [], []
                for date, (issue, tick_frame, minute_frame) in day_contexts.items():
                    record = {"signal_time": pd.Timestamp(issue, unit="s", tz="UTC"), "atr": 1.,
                              "side": v.side_for(symbol, mode), "signal_close": 100., "variant": key}
                    if family != "CLOCK": record["score"] = .2
                    records.append(record)
                    tick, ta, co, ca = r.replay_day(tick_frame, minute_frame, [record], date)
                    matched, metric = r.matched_known(tick, co, ta, ca, bootstrap=False)
                    day = {"date": date, "signals": r.dump_frame(study, "signals", f"{symbol}_{key}_{date}",
                            pd.DataFrame([record], columns=v.SIGNAL_COLUMNS)),
                        "matched_known": {"metrics": metric, "ledger": r.dump_frame(study, "ledgers", f"{symbol}_{key}_{date}_matched", matched)}}
                    for proxy, frame, counts in (("tick", tick, ta), ("coarse", co, ca)):
                        day[proxy] = {"metrics": r.summary(frame, bootstrap=False, date=date), "audit": counts,
                            "ledger": r.dump_frame(study, "ledgers", f"{symbol}_{key}_{date}_{proxy}", frame)}
                    days.append(day); ticks.append(tick); coarse.append(co); tas.append(ta); cas.append(ca)
                signals = r.dump_frame(study, "signals", f"{symbol}_{key}_original", pd.DataFrame(records, columns=v.SIGNAL_COLUMNS))
                identity = {"signals": signals, "family": family, "mode": mode, "model_sha256": None,
                            "development_eligible": False, "reference_only": family != "SHORT44", "round11_selected_model": False,
                            "training_end": text(v.tick7.date_bounds(v.DATES[0])[0]-60) if family != "CLOCK" else None,
                            "frozen_score_threshold": .1 if family != "CLOCK" else None}
                declaration["lineage"]["sampled_signals"][symbol][key] = identity
                tick, co = pd.concat(ticks, ignore_index=True), pd.concat(coarse, ignore_index=True)
                ta, ca = r.totals(tas), r.totals(cas)
                matched, metric = r.matched_known(tick, co, ta, ca)
                row = {k: identity[k] for k in ("family", "mode", "model_sha256", "development_eligible", "reference_only", "round11_selected_model")}
                row.update(selected_model=False, historical_candidate=False, supports_expected_pf_1_5=False,
                    user_target_observed=False, conditional_economic_gates_pass=False, actual_money_profit="NOT TESTED",
                    prospective_validation="NOT TESTED", days=days, matched_known={"metrics": metric,
                        "ledger": r.dump_frame(study, "ledgers", f"{symbol}_{key}_matched_all", matched)},
                    rejection_reasons=["fixed_execution_diagnostic_not_candidate", "max564_paths_and12_days_below1000_and60",
                        "reused_adaptive_known_history", "measured_costs_fills_not_tested"])
                if family == "SHORT44": row["rejection_reasons"].append("round11_development_rejected")
                for proxy, frame, counts in (("tick", tick, ta), ("coarse", co, ca)):
                    row[proxy] = {"metrics": r.summary(frame), "audit_totals": counts,
                        "ledger": r.dump_frame(study, "ledgers", f"{symbol}_{key}_{proxy}_all", frame)}
                    flat.append({"symbol": symbol, "model": key, "proxy": proxy, **row[proxy]["metrics"]})
                models[key], frames[key] = row, (tick, co, ta, ca)
        for key, row in models.items():
            family, mode = key.rsplit("_", 1)
            for field, reference in (("clock_comparisons", "CLOCK_"+mode), ("long_comparisons", "LONG44_"+mode)):
                required = family != "CLOCK" if field == "clock_comparisons" else family == "SHORT44"
                if required:
                    a, b = frames[key], frames[reference]
                    row[field] = {proxy: r.clock_comparison(a[i], b[i], a[i+2], b[i+2]) for i, proxy in enumerate(("tick", "coarse"))}
        result["symbols"][symbol] = {"m1_audit": v.prepared_audit(source, source_minutes), "inputs": inputs, "models": models}
    r.dump_frame(study, ".", "metrics", pd.DataFrame(flat))
    (study / "declaration.json").write_text(json.dumps(declaration))
    (study / "results.json").write_text(json.dumps(result))
    monkeypatch.setattr(v, "metadata", lambda study, audit: (declaration, result))
    return study, result


def test_full_verify_runner_schema_synthetic_integration(tmp_path, monkeypatch):
    study, result = synthetic_full_study(tmp_path, monkeypatch)
    audit = v.Audit(); evidence = v.verify(study, audit)
    assert not audit.failures, audit.failures
    assert evidence["tick_policy_records"] == evidence["coarse_policy_records"] == 144
    assert evidence["matched_known_records"] == 144


def test_full_verify_refuses_diagnostic_promotion(tmp_path, monkeypatch):
    study, result = synthetic_full_study(tmp_path, monkeypatch)
    result["symbols"]["BOOM600"]["models"]["SHORT44_SPIKE"]["historical_candidate"] = True
    audit = v.Audit(); v.verify(study, audit)
    assert any("historical_candidate" in failure["check"] for failure in audit.failures)


@pytest.mark.parametrize("value", ["2026-04-11 00:30:00,000000001+00:00", "2026-04-11X00:30:00.000000001+00:00",
                                   "2026-04-11X00:30:00+00:00", "2026-04-11 00:30:00+0000"])
def test_quote_utc_lexical_aliases_refused(value):
    with pytest.raises(ValueError):
        v.stamp(value)


def synthetic_metadata(tmp_path, monkeypatch):
    """Build only JSON, hashes and header-only CSVs; no historical data involved."""
    monkeypatch.setattr(v, "ROOT", tmp_path)
    monkeypatch.setattr(v, "dependency_pins", lambda audit: None)
    safety = dict.fromkeys(v.FLAGS, False)
    def frozen(folder, name, data):
        folder.mkdir(parents=True, exist_ok=True); path = folder / (name + ".json")
        path.write_text(json.dumps({"safety": safety, **data})); path.with_suffix(".sha256").write_text(v.digest(path))
        return v.digest(path)
    science = {}
    for label in ("docs/SPIKE_SHORT_TICK_PROTOCOL.md", "scripts/run_spike_short_tick.py",
                  "backend/tests/test_spike_short_tick.py", "scripts/run_spike_short_target.py",
                  "scripts/verify_spike_short_target.py", "scripts/verify_spike_tick_execution.py"):
        path = tmp_path / label; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic source\n")
        science[label] = v.digest(path)
    parent_science = {"scripts/run_spike_short_target.py": science["scripts/run_spike_short_target.py"]}
    parent_sources = {s: {"fresh": {"path": s+"_m1.csv", "sha256": "synthetic"}} for s in v.SYMBOLS}
    parent_lineage = {"sources": parent_sources, "round6_selection_sha256": "ancestor6"}
    selected = {"science_code_hashes": parent_science, "lineage": parent_lineage, "config": {"synthetic": True}, "symbols": {}}
    parent_result = {**{k: selected[k] for k in ("science_code_hashes", "lineage", "config")},
        "history": v.old.HISTORY, "stage": "short_target_known_history_evaluation", "adaptive_round": 11,
        "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False, "symbols": {}}
    sampled = {}
    for symbol in v.SYMBOLS:
        selected["symbols"][symbol] = {"models": {}}; saved_models = {}; sampled[symbol] = {}
        for family in v.FAMILIES:
            for mode in v.MODES:
                key = f"{family}_{mode}"
                model = {"training_end": text(v.tick7.date_bounds(v.DATES[0])[0]-60),
                         "estimator": {"threshold": .1}} if family != "CLOCK" else None
                if model: selected["symbols"][symbol]["models"][key] = {"final_model": model, "development_eligible": False}
                path = tmp_path / f"{symbol}_{key}.csv"; path.write_text(",".join(v.SIGNAL_COLUMNS)+"\n")
                desc = {"path": path.name, "sha256": v.digest(path), "columns": v.SIGNAL_COLUMNS,
                        "rows": 0, "local_ignored_artifact": True}
                saved_models[key] = {"family": family, "mode": mode, "config": v.COARSE_CONFIG,
                    "model_sha256": v.canonical_hash(model) if model else None, "development_eligible": False,
                    "reference_only": family != "SHORT44", "selected_model": False,
                    "historical_candidate": False, "artifacts": {"signals": desc}}
                sampled[symbol][key] = {"signals": desc, "family": family, "mode": mode,
                    "model_sha256": saved_models[key]["model_sha256"], "development_eligible": False,
                    "reference_only": family != "SHORT44", "round11_selected_model": False,
                    "training_end": model["training_end"] if model else None,
                    "frozen_score_threshold": .1 if model else None}
        parent_result["symbols"][symbol] = {"cohorts": {"later180_known_history_primary":
            {"source": parent_sources[symbol]["fresh"], "models": saved_models}}}
    p11 = tmp_path / "docs/spike_short_target_20261006"
    dsha = frozen(p11, "declaration", {})
    ssha = frozen(p11, "selection", selected)
    parent_result.update(declaration_sha256=dsha, selection_sha256=ssha)
    rsha = frozen(p11, "results", parent_result)
    asha = frozen(p11, "independent_audit", {"status": "PASS", "pass": True, "error_count": 0, "errors": [],
        "script_sha256": science["scripts/verify_spike_short_target.py"], "declaration_sha256": dsha,
        "selection_sha256": ssha, "results_sha256": rsha})
    tick_sources = {s: {d: {"synthetic": True} for d in v.DATES} for s in v.SYMBOLS}
    p7 = tmp_path / "docs/spike_tick_execution_20261005"
    tick_lineage = {"sources": tick_sources, "inherited_files": {}, "round6_selection_sha256": "ancestor6"}
    for name, payload in (("declaration", {}), ("tick_sources", {"sources": tick_sources}), ("results", {}),
                         ("independent_audit", {"passed": True, "errors": [], "stage": "independent_primary_tick_execution_audit",
                            "sources": list(range(24)), "verifier_file": "scripts/verify_spike_tick_execution.py",
                            "verifier_sha256": science["scripts/verify_spike_tick_execution.py"]})):
        sha = frozen(p7, name, payload)
        tick_lineage["round7_"+("audit" if name == "independent_audit" else name)+"_sha256"] = sha
    lineage = {"files": {}, "round11_declaration_sha256": dsha, "round11_selection_sha256": ssha,
        "round11_results_sha256": rsha, "round11_audit_sha256": asha, "round11_science_code_hashes": parent_science,
        "sampled_signals": sampled, "m1_sources": {s: parent_sources[s]["fresh"] for s in v.SYMBOLS}, "tick_lineage": tick_lineage}
    declaration = {"stage": "short_tick_premeasurement_declaration", "adaptive_round": 12,
        "run_utc": "2026-10-06T21:11:06.013585+00:00", "science_code_hashes": science, "lineage": lineage,
        "config": v.CONFIG, "history": v.HISTORY, "dates": list(v.DATES), "symbols": list(v.SYMBOLS),
        "families": list(v.FAMILIES), "modes": list(v.MODES), "prices_or_signal_csv_decoded": False,
        "new_execution_ledgers_computed": False, "new_model_or_scores_computed": False,
        "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False,
        "study_can_meet_sample_gate": False, "bootstrap_is_discovery_test": False,
        "user_target": {"profit_factor": 1.5, "completed_per_symbol_model": 1000, "active_heldout_days": 60}}
    study = tmp_path / "docs/synthetic_meta"; sha = frozen(study, "declaration", declaration)
    result = {k: declaration[k] for k in ("adaptive_round", "science_code_hashes", "lineage", "config", "history", "dates",
              "goal_achieved", "historical_strategy_candidate", "live_candidate")}
    result.update(stage="short_tick_known_history_fidelity", run_utc="2026-10-06T21:11:07.123456+00:00",
        declaration_sha256=sha, symbols={s: {} for s in v.SYMBOLS}, actual_money_profit="NOT TESTED", prospective_paper="NOT TESTED")
    frozen(study, "results", result)
    return study, declaration, result


def test_metadata_json_lineage_synthetic_integration(tmp_path, monkeypatch):
    study, declaration, result = synthetic_metadata(tmp_path, monkeypatch)
    audit = v.Audit(); got, evaluated = v.metadata(study, audit)
    assert got["lineage"] == declaration["lineage"] and evaluated["symbols"] == result["symbols"]
    assert not audit.failures, audit.failures


@pytest.mark.parametrize("field,value", [("adaptive_round", 12.), ("prices_or_signal_csv_decoded", 0),
                                        ("study_can_meet_sample_gate", True), ("bootstrap_is_discovery_test", True)])
def test_metadata_rehashed_declaration_type_or_gate_corruption_refused(tmp_path, monkeypatch, field, value):
    study, declaration, result = synthetic_metadata(tmp_path, monkeypatch)
    path = study / "declaration.json"; document = json.loads(path.read_text()); document[field] = value
    path.write_text(json.dumps(document)); path.with_suffix(".sha256").write_text(v.digest(path))
    with pytest.raises(ValueError):
        v.metadata(study, v.Audit())


@pytest.mark.parametrize("text", ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{"x": 1, "x": 2}'])
def test_json_corruption_cannot_prevent_preserved_finite_fail_report(tmp_path, text):
    path = tmp_path / "corrupt.json"; path.write_text(text)
    with pytest.raises(ValueError):
        v.read_json(path)


def test_reconciliation_exact_match_distance_retains_float_quote_units():
    date = v.DATES[0]
    start, _ = v.tick7.date_bounds(date)
    ticks = {t: 100.0 for t in range(start, start + 60)}
    minutes = {start: (100.0, 100.0, 100.0, 100.0)}
    audit = v.Audit()
    result = v.minute_reconciliation(ticks, minutes, date, audit)
    assert result['max_abs_difference'] == 0.0
    assert type(result['max_abs_difference']) is float
    v.compare(audit, 'runner_float_distance', result, {
        'complete_minutes': 1, 'unknown_minutes': 1439,
        'max_abs_difference': 0.0, 'price_match': True,
        'coverage_is_separately_audited': True})
    assert not audit.failures
