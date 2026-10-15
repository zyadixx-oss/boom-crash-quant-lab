"""Synthetic closed bars and quotes only; no source files or strategy returns."""
from copy import deepcopy
from dataclasses import dataclass, replace

import pytest

from independent_profit_research.execution import Quote
from independent_profit_research.tape import QuoteTape
from independent_multiframe_research.policies import (
    CLOCK_POLICY_ID, MINUTE_MS as M, POLICY_IDS, clock_signals, generate_signals,
)


@dataclass(frozen=True)
class B:
    open_ms: int
    end_ms: int
    confirmed_ms: int
    open: float
    high: float
    low: float
    close: float
    run_id: int = 0


T = 8 * 60 * M


def bar(size, end, opening=103., high=105., low=101., close=104., run=0):
    return B(end-size*M, end, end, opening, high, low, close, run)


def fixture(policy="CRT_SWEEP_RECLAIM", minutes=20):
    frames = {1: [], 5: [], 15: [], 60: [], 240: []}
    frames[1] = [bar(1, end, 104., 106., 103., 105.)
                 for end in range(T-M, T+(minutes+1)*M, M)]
    frames[5] = [bar(5, end) for end in range(T-85*M, T+(minutes+1)*M, 5*M)]
    frames[60] = [bar(60, T-60*M, 102., 110., 100., 104.),
                  bar(60, T, 102., 110., 100., 104.)]
    frames[240] = [bar(240, T-240*M, 100., 112., 99., 108.),
                   bar(240, T, 100., 112., 99., 108.)]
    if policy == "CRT_SWEEP_RECLAIM":
        frames[15] = [bar(15, T, 103., 108., 99., 104.)]
    elif policy == "SQUEEZE_BREAKOUT":
        frames[15] = [bar(15, T-offset*15*M, 104., 110., 100., 105.)
                      for offset in range(30, -1, -1)]
        frames[15][-5:] = [replace(b, open=104.5, high=106., low=104., close=105.5)
                           for b in frames[15][-5:]]
    else:
        frames[15] = [bar(15, T-offset*15*M, 104., 107., 103., 105.)
                      for offset in range(20, -1, -1)]
        frames[15][-1] = replace(frames[15][-1], open=105., high=107., low=104., close=106.)
    if policy != "CRT_SWEEP_RECLAIM":
        set_bar(frames, 5, T, open=103., high=105., low=101., close=104.)
        set_bar(frames, 5, T+5*M, open=104., high=108., low=104., close=107.)
        set_bar(frames, 1, T+6*M, open=107., high=109., low=106., close=108.)
    rows = [Quote(stamp, 104., 105.) for stamp in range(T-M, T+(minutes+1)*M+1000, 1000)]
    return frames, rows


def set_bar(frames, size, end, **changes):
    index = next(i for i, b in enumerate(frames[size]) if b.end_ms == end)
    frames[size][index] = replace(frames[size][index], **changes)


def signals(frames, rows, policy="CRT_SWEEP_RECLAIM", side=1, unit=1., audit=None):
    return list(generate_signals(frames, policy_id=policy, native_direction=side,
                                 spread_unit=unit, quotes=rows, audit=audit))


def first_only(frames, rows, policy):
    result = signals(frames, rows, policy)
    assert len(result) == 1
    return result[0]


@pytest.mark.parametrize("policy", sorted(POLICY_IDS))
def test_each_fixed_setup_issues_with_frozen_atr_target_and_actual_time(policy):
    frames, rows = fixture(policy)
    result = first_only(frames, rows, policy)
    expected = T+M if policy == "CRT_SWEEP_RECLAIM" else T+6*M
    assert result.decision_ms == expected
    assert result.horizon_end_ms == expected+15*M
    assert result.direction == 1
    assert result.stop_distance == 4.
    assert result.target_distance == 8.


@pytest.mark.parametrize("policy", sorted(POLICY_IDS))
def test_future_prefixes_and_unfinished_issue_bundle_do_not_change_signals(policy):
    frames, rows = fixture(policy)
    full = signals(frames, rows, policy)
    for cutoff in (T, T+M, T+5*M, T+6*M, T+7*M, T+20*M):
        prefix = {size: [b for b in bars if b.confirmed_ms <= cutoff] for size, bars in frames.items()}
        source = [q for q in rows if q.time_msc <= cutoff]
        assert signals(prefix, source, policy) == [s for s in full if s.decision_ms <= cutoff]
    decision = full[0].decision_ms
    prefix = {size: [b for b in bars if b.confirmed_ms <= decision] for size, bars in frames.items()}
    source = [q for q in rows if q.time_msc <= decision]
    assert signals(prefix, source+[Quote(decision, 10., 900.)], policy) == full


@pytest.mark.parametrize("policy", sorted(POLICY_IDS))
def test_boom_crash_mirror_has_same_times_and_distances(policy):
    frames, rows = fixture(policy)
    reflected = {size: [replace(b, open=1000-b.open, high=1000-b.low,
                                low=1000-b.high, close=1000-b.close) for b in bars]
                 for size, bars in frames.items()}
    opposite = [Quote(q.time_msc, 1000-q.ask, 1000-q.bid) for q in rows]
    left, right = signals(frames, rows, policy), signals(reflected, opposite, policy, side=-1)
    assert len(left) == len(right) == 1
    assert [(s.decision_ms, s.stop_distance, s.target_distance) for s in left] == [
        (s.decision_ms, s.stop_distance, s.target_distance) for s in right]
    assert right[0].direction == -1


def test_crt_prior_h1_reference_and_two_atr_denominators_are_audited():
    frames, rows = fixture()
    # Only the latest M5 range changes: depth still uses the pre-setup ATR4.
    set_bar(frames, 5, T, high=115., low=101.)
    audit = {}
    result = first_only(frames, rows, "CRT_SWEEP_RECLAIM")
    signals(frames, rows, audit=audit)
    record = audit[("CRT_SWEEP_RECLAIM", result.decision_ms)]
    assert record["reference_h1_end_ms"] == T-60*M
    assert record["frozen_depth_atr"] == 4.
    assert record["frozen_atr"] == pytest.approx((13*4+14)/14)
    assert result.stop_distance == record["frozen_atr"]


def test_crt_at_hour_boundary_can_use_h1_ending_exactly_at_setup_open():
    frames, rows = fixture()
    frames[15] = [bar(15, T+15*M, 103., 108., 99., 104.)]
    audit = {}
    result = first_only(frames, rows, "CRT_SWEEP_RECLAIM")
    signals(frames, rows, audit=audit)
    assert result.decision_ms == T+16*M
    assert audit[("CRT_SWEEP_RECLAIM", result.decision_ms)]["reference_h1_end_ms"] == T


@pytest.mark.parametrize("depth,qualifies", [(0.399, False), (0.4, True), (2.6, True), (2.601, False)])
def test_crt_sweep_depth_boundaries(depth, qualifies):
    frames, rows = fixture()
    frames[15][0] = replace(frames[15][0], low=100-depth)
    assert bool(signals(frames, rows)) == qualifies


def test_crt_reclaim_boundary_and_last_m5_native_body_are_strict():
    frames, rows = fixture()
    frames[15][0] = replace(frames[15][0], close=100.)
    assert signals(frames, rows) == []
    frames, rows = fixture()
    set_bar(frames, 5, T, open=104., close=104.)
    assert signals(frames, rows) == []


def test_crt_waits_for_native_m1_but_outside_close_cancels():
    frames, rows = fixture()
    set_bar(frames, 1, T+M, open=106., close=105.)
    assert first_only(frames, rows, "CRT_SWEEP_RECLAIM").decision_ms == T+2*M
    set_bar(frames, 1, T+M, open=105., high=111., close=110.)
    assert signals(frames, rows) == []


def test_crt_uses_reference_only_once_after_issued_signal():
    frames, rows = fixture(minutes=40)
    frames[15] += [bar(15, T+15*M, 103., 108., 99., 104.),
                   bar(15, T+30*M, 103., 108., 99., 104.)]
    audit = {}
    result = signals(frames, rows, audit=audit)
    assert [s.decision_ms for s in result] == [T+M, T+16*M]
    assert len({r["reference_h1_end_ms"] for r in audit.values()}) == 2


@pytest.mark.parametrize("policy", ["SQUEEZE_BREAKOUT", "PULLBACK_CONTINUATION"])
def test_first_qualifying_m5_then_strictly_later_m1_are_required(policy):
    frames, rows = fixture(policy)
    set_bar(frames, 1, T+5*M, open=107., high=109., low=106., close=108.)
    assert first_only(frames, rows, policy).decision_ms == T+6*M
    frames[5] = [b for b in frames[5] if b.end_ms != T+5*M]
    assert signals(frames, rows, policy) == []


@pytest.mark.parametrize("policy", ["SQUEEZE_BREAKOUT", "PULLBACK_CONTINUATION"])
def test_context_latest_h1_doji_or_wrong_direction_blocks_setup(policy):
    frames, rows = fixture(policy)
    set_bar(frames, 60, T, open=104., close=104.)
    assert signals(frames, rows, policy) == []
    frames, rows = fixture(policy)
    frames[60] = frames[60][:-1]  # Never use the older native H1 instead.
    assert signals(frames, rows, policy) == []


def test_squeeze_requires_thirty_contiguous_trs_and_fixed_compression():
    frames, rows = fixture("SQUEEZE_BREAKOUT")
    frames[15] = frames[15][1:]
    assert signals(frames, rows, "SQUEEZE_BREAKOUT") == []
    frames, rows = fixture("SQUEEZE_BREAKOUT")
    frames[15] = [replace(b, open=104., high=110., low=100., close=105.) for b in frames[15]]
    assert signals(frames, rows, "SQUEEZE_BREAKOUT") == []


def test_squeeze_ratio_exactly_point_eight_is_inclusive():
    frames, rows = fixture("SQUEEZE_BREAKOUT")
    frames[15] = [replace(b, open=104., high=112., low=99., close=105.5)
                  for b in frames[15]]
    frames[15][-5:] = [replace(b, high=110., low=100.) for b in frames[15][-5:]]
    set_bar(frames, 5, T+5*M, open=104., high=112., low=104., close=111.)
    set_bar(frames, 1, T+6*M, open=111., high=113., low=110., close=112.)
    assert len(signals(frames, rows, "SQUEEZE_BREAKOUT")) == 1
    frames[15][-5:] = [replace(b, high=110.001) for b in frames[15][-5:]]
    assert signals(frames, rows, "SQUEEZE_BREAKOUT") == []


def test_squeeze_stay_outside_and_frozen_source_boundary():
    frames, rows = fixture("SQUEEZE_BREAKOUT")
    set_bar(frames, 1, T+6*M, open=105., high=107., low=104., close=106.)
    set_bar(frames, 1, T+7*M, open=107., high=109., low=106., close=108.)
    assert signals(frames, rows, "SQUEEZE_BREAKOUT") == []


def test_pullback_ema_uses_twenty_close_sma_seed_then_declared_alpha():
    frames, rows = fixture("PULLBACK_CONTINUATION")
    audit = {}
    result = signals(frames, rows, "PULLBACK_CONTINUATION", audit=audit)
    assert len(result) == 1
    assert audit[("PULLBACK_CONTINUATION", result[0].decision_ms)]["frozen_level"] == pytest.approx(105+2/21)
    frames[15] = frames[15][2:]
    assert signals(frames, rows, "PULLBACK_CONTINUATION") == []


def test_pullback_ema_warmup_restarts_after_missing_m15():
    frames, rows = fixture("PULLBACK_CONTINUATION")
    frames[15] = [b for b in frames[15] if b.end_ms != T-10*15*M]
    assert signals(frames, rows, "PULLBACK_CONTINUATION") == []


def test_pullback_must_touch_ema_and_m5_must_break_previous_bar():
    frames, rows = fixture("PULLBACK_CONTINUATION")
    frames[15][-1] = replace(frames[15][-1], open=105.6, low=105.5)
    assert signals(frames, rows, "PULLBACK_CONTINUATION") == []
    frames, rows = fixture("PULLBACK_CONTINUATION")
    set_bar(frames, 5, T+5*M, close=105.)
    assert signals(frames, rows, "PULLBACK_CONTINUATION") == []


@pytest.mark.parametrize("policy", sorted(POLICY_IDS))
def test_missing_minute_or_source_run_change_cancels_pending(policy):
    frames, rows = fixture(policy)
    if policy == "CRT_SWEEP_RECLAIM":
        set_bar(frames, 1, T+M, open=106., close=105.)
        missing = T+M
    else:
        missing = T+2*M
    frames[1] = [b for b in frames[1] if b.end_ms != missing]
    assert signals(frames, rows, policy) == []
    frames, rows = fixture(policy)
    frames[1] = [replace(b, run_id=1) if b.end_ms > T else b for b in frames[1]]
    assert signals(frames, rows, policy) == []


@pytest.mark.parametrize("policy", sorted(POLICY_IDS))
def test_first_qualifying_bad_spread_cancels_without_better_quote_search(policy):
    frames, rows = fixture(policy)
    first = T+M if policy == "CRT_SWEEP_RECLAIM" else T+6*M
    rows = [replace(q, ask=q.bid+2.01) if q.time_msc == first else q for q in rows]
    assert signals(frames, rows, policy) == []
    rows = [replace(q, ask=q.bid+2.) if q.time_msc == first else q for q in rows]
    assert len(signals(frames, rows, policy)) == 1


def test_freshness_uses_actual_first_quote_and_two_second_boundary():
    frames, rows = fixture()
    set_bar(frames, 1, T+M, confirmed_ms=T+M+2000)
    rows = [q for q in rows if not T+M <= q.time_msc < T+M+2000]
    assert first_only(frames, rows, "CRT_SWEEP_RECLAIM").decision_ms == T+M+2000
    set_bar(frames, 1, T+M, confirmed_ms=T+M+2001)
    rows = [q for q in rows if q.time_msc != T+M+2000]
    rows += [Quote(T+M+2001, 104., 105.)]
    rows.sort(key=lambda q: q.time_msc)
    assert signals(frames, rows) == []


def test_crt_deadline_uses_confirmation_not_nominal_minute_close():
    frames, rows = fixture()
    for end in range(T+M, T+5*M, M):
        set_bar(frames, 1, end, open=106., close=105.)
    assert first_only(frames, rows, "CRT_SWEEP_RECLAIM").decision_ms == T+5*M
    set_bar(frames, 1, T+5*M, confirmed_ms=T+5*M+1)
    assert signals(frames, rows) == []


def test_break_and_confirmation_deadline_prevent_late_setup_rescue():
    policy = "SQUEEZE_BREAKOUT"
    frames, rows = fixture(policy)
    set_bar(frames, 5, T+5*M, close=105.)
    set_bar(frames, 5, T+10*M, close=105.)
    set_bar(frames, 5, T+15*M, open=104., high=108., low=104., close=107.)
    for end in range(T+16*M, T+20*M, M):
        set_bar(frames, 1, end, open=108., high=109., low=106., close=107.)
    set_bar(frames, 1, T+20*M, open=107., high=109., low=106., close=108.)
    assert first_only(frames, rows, policy).decision_ms == T+20*M
    set_bar(frames, 1, T+20*M, confirmed_ms=T+20*M+1)
    assert signals(frames, rows, policy) == []


def test_new_m15_does_not_replace_pending_setup_or_its_atr():
    policy = "SQUEEZE_BREAKOUT"
    frames, rows = fixture(policy)
    set_bar(frames, 5, T+5*M, close=105.)
    set_bar(frames, 5, T+10*M, close=105.)
    set_bar(frames, 5, T+15*M, open=104., high=200., low=100., close=107.)
    frames[15].append(bar(15, T+15*M, 104., 110., 104., 109.))
    set_bar(frames, 1, T+16*M, open=107., high=109., low=106., close=108.)
    audit = {}
    result = signals(frames, rows, policy, audit=audit)
    assert len(result) == 1
    assert result[0].stop_distance == 4.
    assert audit[(policy, result[0].decision_ms)]["setup_end_ms"] == T


def test_exact_latest_h4_and_contiguous_atr_history_are_mandatory():
    frames, rows = fixture()
    frames[240] = frames[240][:-1]
    assert signals(frames, rows) == []
    frames, rows = fixture()
    frames[5] = [b for b in frames[5] if b.end_ms != T-20*M]
    assert signals(frames, rows) == []


def test_future_bar_confirmation_cannot_leak_into_setup_context():
    frames, rows = fixture()
    set_bar(frames, 240, T, confirmed_ms=T+2*M)
    assert signals(frames, rows) == []


def test_clock_has_exact_00_30_schedule_and_same_exposure_rules():
    frames, rows = fixture(minutes=40)
    audit = {}
    result = list(clock_signals(frames, QuoteTape(rows), 1., -1, T, T+40*M, audit=audit))
    assert [s.decision_ms for s in result] == [T, T+30*M]
    assert all(s.policy_id == CLOCK_POLICY_ID and s.direction == -1 and
               s.stop_distance == 4 and s.target_distance == 8 and
               s.horizon_end_ms == s.decision_ms+15*M for s in result)
    assert len(audit) == 2


def test_clock_does_not_replace_missing_m1_or_bad_first_quote():
    frames, rows = fixture()
    frames[1] = [b for b in frames[1] if b.end_ms != T]
    assert list(clock_signals(frames, QuoteTape(rows), 1., 1, T, T+10*M)) == []
    frames, rows = fixture()
    rows = [replace(q, ask=q.bid+3) if q.time_msc == T else q for q in rows]
    assert list(clock_signals(frames, QuoteTape(rows), 1., 1, T, T+10*M)) == []


@pytest.mark.parametrize("kind", ["duplicate", "reverse", "early_confirmation", "bad_ohlc", "misaligned", "nan"])
def test_invalid_bar_contracts_fail_closed(kind):
    frames, rows = fixture()
    if kind == "duplicate":
        frames[1].insert(1, frames[1][0])
    elif kind == "reverse":
        frames[1].reverse()
    elif kind == "early_confirmation":
        frames[1][0] = replace(frames[1][0], confirmed_ms=frames[1][0].end_ms-1)
    elif kind == "bad_ohlc":
        frames[1][0] = replace(frames[1][0], low=107.)
    elif kind == "misaligned":
        frames[1][0] = replace(frames[1][0], open_ms=frames[1][0].open_ms+1)
    else:
        frames[1][0] = replace(frames[1][0], close=float("nan"))
    with pytest.raises(ValueError):
        signals(frames, rows)


@pytest.mark.parametrize("changes", [dict(policy_id="unknown"), dict(native_direction=True),
                                     dict(spread_unit=0), dict(spread_unit=float("inf"))])
def test_invalid_public_arguments_rejected(changes):
    frames, rows = fixture()
    kwargs = dict(policy_id="CRT_SWEEP_RECLAIM", native_direction=1, spread_unit=1., quotes=rows)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        list(generate_signals(frames, **kwargs))


def test_requires_one_quote_source_and_does_not_mutate_inputs():
    frames, rows = fixture()
    before = deepcopy(frames)
    with pytest.raises(ValueError):
        list(generate_signals(frames, policy_id="CRT_SWEEP_RECLAIM", native_direction=1, spread_unit=1.))
    with pytest.raises(ValueError):
        list(generate_signals(frames, policy_id="CRT_SWEEP_RECLAIM", native_direction=1,
                              spread_unit=1., quotes=rows, tape=QuoteTape(rows)))
    assert signals(frames, rows)
    assert frames == before


def test_synthetic_bidask_builder_confirms_crt_signal_without_boundary_flush():
    from independent_multiframe_research.bars import build_frames

    rows = []
    for stamp in range(0, T+M+1000, 1000):
        position = stamp % (5*M)
        mid = 102 if position < M else 106 if position < 2*M else 101 if position < 3*M else 104
        if 6*60*M+10*M <= stamp < 6*60*M+11*M:
            mid = 110
        if 6*60*M+20*M <= stamp < 6*60*M+21*M:
            mid = 100
        if T-10*M <= stamp < T-9*M:
            mid = 99
        if T <= stamp < T+M:
            mid = 104 if stamp-T < 30_000 else 106
        rows.append(Quote(stamp, mid-1, mid+1))
    tape = QuoteTape(rows)
    frames = build_frames(tape, 0, T+M+1000)
    audit = {}
    result = list(generate_signals(frames, policy_id="CRT_SWEEP_RECLAIM",
                                   native_direction=1, spread_unit=1., tape=tape, audit=audit))
    assert len(result) == 1 and result[0].decision_ms == T+M
    assert audit[("CRT_SWEEP_RECLAIM", T+M)]["issue_m1_end_ms"] == T+M
    # A half-open window ending at the decision does not borrow its quote.
    prefix = build_frames(tape, 0, T+M)
    assert list(generate_signals(prefix, policy_id="CRT_SWEEP_RECLAIM",
                                 native_direction=1, spread_unit=1., tape=tape)) == []
