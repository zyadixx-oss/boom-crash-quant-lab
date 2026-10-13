import asyncio

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import ShadowSignal
from app.services import shadow
from app.services.signal_engine import SignalDecision


def ticks(start, end):
    return pd.DataFrame({'epoch': range(start, end + 1),
                         'price': [100.0] * (end - start + 1)})


@pytest.mark.parametrize('timeframe,seconds', [('M1', 60), ('M5', 300), ('M15', 900)])
def test_unclosed_spike_does_not_change_closed_bar(timeframe, seconds):
    source = ticks(0, seconds + 1)
    source.loc[source['epoch'] >= seconds, 'price'] = 10000.0
    result = shadow.closed_tick_bars(source, timeframe)
    assert result['epoch'].tolist() == [0]
    assert result.iloc[-1]['high'] == 100.0


def test_partial_initial_bucket_and_missing_tick_reset_warmup():
    source = ticks(1, 240)
    source = source[source['epoch'] != 100]
    result = shadow.closed_tick_bars(source, 'M1')
    assert result['epoch'].tolist() == [120, 180]


def test_stale_completed_bar_cannot_signal_after_gap():
    source = pd.concat([ticks(0, 59), ticks(120, 120)], ignore_index=True)
    assert shadow.closed_tick_bars(source, 'M1').empty


def test_m15_context_can_actually_warm_up():
    runner = shadow.ShadowRunner('BOOM500')
    result = runner._context(ticks(0, 60 * 900), 'M15')
    assert result is not None
    assert result['epoch'] == 59 * 900


@pytest.fixture
def database(monkeypatch):
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(shadow, 'SessionLocal', sessions)
    yield sessions
    engine.dispose()


def test_signal_uses_closed_candle_and_records_actual_issuance(monkeypatch, database):
    runner = shadow.ShadowRunner('BOOM500')
    frame = ticks(0, 71 * 60 + 5)
    frame.loc[frame['epoch'] >= 71 * 60, 'price'] = 10000.0
    runner.buffer = frame.to_dict('records')
    monkeypatch.setattr(shadow, 'add_features', lambda data, profile: data)
    monkeypatch.setattr(shadow, 'add_market_structure', lambda data: data)
    observed = []

    def decision(row, profile, direction):
        observed.append((int(row['epoch']), float(row['high'])))
        return SignalDecision(profile.symbol, int(row['epoch']), direction, 100,
                              [], {}, float(row['close']), profile.test_window_bars)

    monkeypatch.setattr(shadow, 'score_row', decision)
    asyncio.run(runner._evaluate_signal())
    assert observed == [(70 * 60, 100.0)]
    with database() as db:
        signal = db.scalar(select(ShadowSignal))
        assert signal.result['issued_epoch'] == 71 * 60 + 5
        assert signal.price == 10000.0
        assert signal.result['closed_bar_reference_price'] == 100.0
        assert signal.result['cost_verified'] is False
        assert signal.result['execution_verified'] is False


def test_outcome_horizon_starts_at_issuance_not_candle_open(database):
    runner = shadow.ShadowRunner('BOOM500')
    runner.profile.test_window_bars = 2
    with database() as db:
        db.add(ShadowSignal(symbol='BOOM500', epoch=0, price=100,
                            result={'status': 'PENDING', 'issued_epoch': 60,
                                    'cost_verified': False, 'execution_verified': False}))
        db.commit()
    for epoch in range(61, 121):
        runner._evaluate_pending({'epoch': epoch, 'price': 110})
    with database() as db:
        assert db.scalar(select(ShadowSignal)).result['status'] == 'PENDING'
    for epoch in range(121, 181):
        runner._evaluate_pending({'epoch': epoch, 'price': 110})
    with database() as db:
        result = db.scalar(select(ShadowSignal)).result
        assert result['status'] == 'WIN'
        assert result['issued_epoch'] == 60
        assert result['execution_verified'] is False


@pytest.mark.parametrize('later_epoch', [63, 180])
def test_inner_tick_gap_is_unknown_even_when_horizon_quote_exists(database, later_epoch):
    runner = shadow.ShadowRunner('BOOM500')
    with database() as db:
        db.add(ShadowSignal(symbol='BOOM500', epoch=0, price=100,
                            result={'status': 'PENDING', 'issued_epoch': 60,
                                    'last_observed_epoch': 60, 'test_window_bars': 2}))
        db.commit()
    runner._evaluate_pending({'epoch': 61, 'price': 110})
    runner._evaluate_pending({'epoch': later_epoch, 'price': 10000})
    with database() as db:
        result = db.scalar(select(ShadowSignal)).result
        assert result['status'] == 'UNKNOWN'
        assert result['reason'] == 'MISSING_INTERMEDIATE_TICKS'
        assert result['last_observed_epoch'] == 61
        assert 'signed_move' not in result


def test_old_pending_signal_without_issuance_is_unknown(database):
    runner = shadow.ShadowRunner('BOOM500')
    with database() as db:
        db.add(ShadowSignal(symbol='BOOM500', epoch=0, price=100,
                            result={'status': 'PENDING'}))
        db.commit()
    runner._evaluate_pending({'epoch': 1000, 'price': 110})
    with database() as db:
        result = db.scalar(select(ShadowSignal)).result
        assert result['status'] == 'UNKNOWN'
        assert result['reason'] == 'LEGACY_ISSUANCE_TIME_NOT_RECORDED'


def test_missing_horizon_tick_cannot_turn_into_late_win(database):
    runner = shadow.ShadowRunner('BOOM500')
    with database() as db:
        db.add(ShadowSignal(symbol='BOOM500', epoch=0, price=100,
                            result={'status': 'PENDING', 'issued_epoch': 60,
                                    'test_window_bars': 2}))
        db.commit()
    runner._evaluate_pending({'epoch': 181, 'price': 10000})
    with database() as db:
        result = db.scalar(select(ShadowSignal)).result
        assert result['status'] == 'UNKNOWN'
        assert result['reason'] == 'MISSING_HORIZON_TICK'


def test_old_pending_is_not_hidden_by_newer_closed_results(database):
    runner = shadow.ShadowRunner('BOOM500')
    with database() as db:
        old = ShadowSignal(symbol='BOOM500', epoch=0, price=100,
                           result={'status': 'PENDING', 'issued_epoch': 60,
                                   'test_window_bars': 2})
        db.add(old)
        db.add_all([ShadowSignal(symbol='BOOM500', epoch=epoch, price=100,
                                 result={'status': 'WIN'}) for epoch in range(1, 102)])
        db.commit()
        old_id = old.id
    runner._evaluate_pending({'epoch': 2000, 'price': 10000})
    with database() as db:
        assert db.get(ShadowSignal, old_id).result['status'] == 'UNKNOWN'


@pytest.mark.parametrize('bad', [{'epoch': 4200, 'price': 9000.0},
                                  {'epoch': 4260, 'price': 9000.0}])
def test_old_or_conflicting_tick_cannot_be_an_issuance_quote(bad):
    runner = shadow.ShadowRunner('BOOM500')
    runner._accept_tick({'epoch': 4260, 'price': 100.0})
    with pytest.raises(ValueError):
        runner._accept_tick(bad)
    assert runner.buffer == [{'epoch': 4260, 'price': 100.0}]


def test_identical_repeated_tick_is_ignored_before_buffer():
    runner = shadow.ShadowRunner('BOOM500')
    row = {'epoch': 4260, 'price': 100.0}
    assert runner._accept_tick(row) is True
    assert runner._accept_tick(dict(row)) is False
    assert len(runner.buffer) == 1


def test_direct_unordered_buffer_is_rejected_instead_of_silently_sorted():
    source = pd.concat([ticks(0, 4260), pd.DataFrame([{'epoch': 4200, 'price': 9000.0}])])
    with pytest.raises(ValueError):
        shadow.closed_tick_bars(source, 'M1')
