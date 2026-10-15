"""Hand-calculated synthetic labels; no historical payoff execution."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.research.payoff_ticks import TICK_TRADE_COLUMNS, TickExitConfig, replay_ticks
from app.research.tick_tail_labels import independent_tick_labels

BASE = pd.Timestamp("2026-01-01T00:00:00Z")
END = BASE + pd.Timedelta(seconds=6000)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"):
        monkeypatch.setenv(key, "false")


def history():
    return pd.DataFrame({"quote":100.},index=pd.date_range(BASE,periods=6001,freq="s"))


def signal(minute=0,side=1,**extra):
    return {"signal_time":BASE+pd.Timedelta(minutes=minute),"atr":1.,"side":side,"variant":"SYNTHETIC",**extra}


def set_quote(frame,second,value):
    frame.loc[BASE+pd.Timedelta(seconds=second),"quote"]=value


def run(frame,signals=None,config=None,end=END):
    return independent_tick_labels(frame,[signal()] if signals is None else signals,
                                   TickExitConfig() if config is None else config,BASE,end)


def test_label_policy_explicitly_excludes_strategy_returns_and_sample_gate():
    frame=history();before=frame.copy(deep=True)
    ledger,audit=run(frame)
    assert list(ledger.columns)==TICK_TRADE_COLUMNS
    assert audit['labels_may_overlap'] and audit['partition_bounds_preserved']
    assert not audit['one_open_portfolio_constraint'] and not audit['strategy_returns']
    assert not audit['profit_factor_evidence'] and not audit['counts_toward_profit_sample_target']
    assert not audit['training_size_floor_enforced'] and audit['purge_minutes']==31
    assert audit['no_statistical_independence_claim']
    assert audit['label_kind']=='individually_replayed_overlapping_opportunity_training_targets'
    pd.testing.assert_frame_equal(frame,before)


def test_independent_overlapping_opportunities_are_all_labeled_but_portfolio_skips():
    frame=history();signals=[signal(10),signal(0),signal(5)]
    labels,audit=run(frame,signals)
    assert labels.signal_time.tolist()==[BASE,BASE+pd.Timedelta(minutes=5),BASE+pd.Timedelta(minutes=10)]
    assert len(labels)==3 and audit['completed']==3 and audit['overlap_skipped']==0
    strategy,one_open=replay_ticks(frame,signals,TickExitConfig(),BASE,END)
    assert len(strategy)==1 and one_open['overlap_skipped']==2
    assert audit['issued']==audit['filled']==audit['independent_signal_replays']==3
    # Mixed admissions verify every aggregate counter, not only happy paths.
    mixed=frame.drop([BASE+pd.Timedelta(seconds=61),BASE+pd.Timedelta(seconds=500)])
    labels,audit=run(mixed,[signal(-1),signal(0),signal(5),signal(20),signal(55)],end=BASE+pd.Timedelta(hours=1))
    assert audit['issued']==5 and audit['outside_partition']==audit['missing_entry']==audit['purged']==1
    assert audit['filled']==2 and audit['completed']==audit['censored']==audit['missing_path']==1
    assert audit['overlap_skipped']==0 and len(labels)==2


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('successor,expected_gross',[(95.,-2.5),(101.,.5)])
def test_uncapped_jump_stop_and_recovery_are_next_quote_labels(side,successor,expected_gross):
    frame=history();set_quote(frame,70,100.+side*(96.-100.));set_quote(frame,71,100.+side*(successor-100.))
    ledger,audit=run(frame,[signal(side=side)]);row=ledger.iloc[0]
    assert row.reason=='sl' and row.trigger_time==BASE+pd.Timedelta(seconds=70)
    assert row.exit_time==BASE+pd.Timedelta(seconds=71)
    assert row.gross_R==expected_gross and row.net_R==pytest.approx(expected_gross-.05)
    assert audit['completed']==1 and not row.censored


@pytest.mark.parametrize('side',[1,-1])
def test_deadline_crossing_has_stop_priority_and_keeps_successor(side):
    frame=history();set_quote(frame,960,100.+side*(97.-100.));set_quote(frame,961,100.+side)
    ledger,_=run(frame,[signal(side=side)]);row=ledger.iloc[0]
    assert row.reason=='sl' and row.trigger_time==BASE+pd.Timedelta(seconds=960)
    assert row.exit_time==BASE+pd.Timedelta(seconds=961) and row.gross_R==.5


def test_entry_and_timeout_are_strictly_next_and_deadline_stays_nominal():
    frame=history();set_quote(frame,60,999.);set_quote(frame,960,101.);set_quote(frame,961,102.);set_quote(frame,962,1000.)
    ledger,_=run(frame);row=ledger.iloc[0]
    assert row.entry_time==BASE+pd.Timedelta(seconds=61) and row.entry==100.
    assert row.exit_time==BASE+pd.Timedelta(seconds=961) and row.exit==102. and row.reason=='time'
    assert row.planned_end==BASE+pd.Timedelta(seconds=960) and row.holding_minutes==15.


def test_missing_first_entry_quote_is_unknown_and_never_shifted():
    frame=history().drop(BASE+pd.Timedelta(seconds=61))
    ledger,audit=run(frame)
    assert ledger.empty and audit['missing_entry']==1 and audit['filled']==0


def test_required_earlier_second_gap_censors_without_return():
    frame=history().drop(BASE+pd.Timedelta(seconds=500))
    ledger,audit=run(frame);row=ledger.iloc[0]
    assert row.censored and row.missing_time==BASE+pd.Timedelta(seconds=500)
    assert np.isnan(row.net_R) and np.isnan(row.gross_R) and audit['missing_path']==1


def test_gap_after_known_exit_is_irrelevant():
    frame=history().drop(BASE+pd.Timedelta(seconds=80));set_quote(frame,70,97.)
    ledger,audit=run(frame);row=ledger.iloc[0]
    assert not row.censored and row.exit_time==BASE+pd.Timedelta(seconds=71) and audit['completed']==1


@pytest.mark.parametrize('stop',[False,True])
def test_missing_required_timeout_or_stop_successor_censors(stop):
    frame=history();missing=71 if stop else 961
    if stop:set_quote(frame,70,97.)
    frame=frame.drop(BASE+pd.Timedelta(seconds=missing))
    ledger,audit=run(frame)
    assert ledger.iloc[0].censored and ledger.iloc[0].missing_time==BASE+pd.Timedelta(seconds=missing)
    assert np.isnan(ledger.iloc[0].net_R) and audit['censored']==1


def test_effective_fold_end_purges_before_an_observed_early_stop():
    frame=history();set_quote(frame,70,97.)
    ledger,audit=run(frame,end=BASE+pd.Timedelta(seconds=1800))
    assert ledger.empty and audit['purged']==1
    ledger,audit=run(frame,end=BASE+pd.Timedelta(seconds=1860))
    assert len(ledger)==1 and audit['purged']==0


def test_empty_issuance_preserves_typed_empty_schema():
    ledger,audit=run(history(),[])
    assert ledger.empty and list(ledger.columns)==TICK_TRADE_COLUMNS and audit['issued']==0
    assert str(ledger.entry_time.dtype)=='datetime64[ns, UTC]' and ledger.censored.dtype==bool


def test_empty_observed_source_has_missing_entry_not_fabricated_return():
    ledger,audit=run(history().iloc[:0])
    assert ledger.empty and audit['issued']==audit['missing_entry']==1


@pytest.mark.parametrize('bad',['unordered','nonfinite'])
def test_whole_source_validated_before_windowing(bad):
    frame=history()
    if bad=='unordered':frame=frame.iloc[::-1]
    else:set_quote(frame,5000,np.nan)  # Outside needed candidate window, still invalid source.
    with pytest.raises(ValueError):run(frame)


@pytest.mark.parametrize('bad',[{'atr':0.},{'side':True},{'variant':5}])
def test_invalid_issuance_refused(bad):
    with pytest.raises(ValueError):run(history(),[signal(**bad)])


@pytest.mark.parametrize('config',[replace(TickExitConfig(),max_gap_seconds=2),replace(TickExitConfig(),stop_latency_ticks=0)])
def test_unsupported_gap_or_latency_conventions_refused(config):
    with pytest.raises(ValueError,match='max_gap_seconds=1'):run(history(),config=config)


def test_all_four_live_flags_block_even_empty_labels(monkeypatch):
    for flag in ('LIVE_TRADING','READY_FOR_LIVE','LIVE_ALLOWED','OPENED_TRADES'):
        monkeypatch.setenv(flag,'true')
        with pytest.raises(RuntimeError):run(history().iloc[:0],[])
        monkeypatch.setenv(flag,'false')


def test_repeated_issue_timestamp_cannot_inflate_training_floor():
    for duplicate in (signal(),signal(variant='OTHER_VARIANT'),
                      signal(signal_time=BASE.tz_convert('Asia/Riyadh'))):
        with pytest.raises(ValueError,match='Duplicate signal_time'):
            run(history(),[signal(),duplicate])
