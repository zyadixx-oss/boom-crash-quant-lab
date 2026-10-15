"""Synthetic TRAIN-only medians, fresh scratch files and transaction failures."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import random
import sqlite3
import statistics

import pytest

from independent_multiframe_research.spread_fit import fit_positive_spread_prefixes
from independent_multiframe_research.sharded import DAY_MS
from independent_profit_research.data import MAX_SCALED_PRICE
from independent_profit_research.execution import Quote as Q
from independent_profit_research.tape import QuoteTape


def fit(days, checkpoints, path, *, start=0, end=None):
    calls = []
    end = start + len(days) * DAY_MS if end is None else end

    def loader(day):
        assert start <= day < end and day % DAY_MS == 0
        calls.append(day)
        return QuoteTape(days[(day - start) // DAY_MS])

    result = fit_positive_spread_prefixes(start, end, checkpoints, loader, path)
    return result, calls


def tables(path):
    with sqlite3.connect(path) as connection:
        return [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]


@pytest.mark.parametrize('seed', range(20))
def test_seeded_every_prefix_exactly_matches_statistics_median(seed, tmp_path):
    rng = random.Random(29000 + seed)
    days = []
    for day in range(15):
        rows = []
        for index in range(rng.randrange(0, 25)):
            spread = rng.choice([0, 1, 2, 3, 7, 100, 10001])
            quote = Q(day * DAY_MS + index, 100000, 100000 + spread)
            rows.append(quote)
            if rng.random() < .3:
                rows.append(quote)
        if day == 0:
            rows.insert(0, Q(0, 100000, 100001))
        days.append(rows)
    checkpoints = [day * DAY_MS for day in range(1, 16)]
    path = tmp_path / 'exact.sqlite'
    results, calls = fit(days, checkpoints, path)
    assert type(results) is tuple
    assert calls == list(range(0, 15 * DAY_MS, DAY_MS))
    spreads, observed, empty = [], 0, 0
    for day, result in enumerate(results):
        spreads.extend(int(q.ask - q.bid) for q in days[day] if q.ask > q.bid)
        observed += len(days[day])
        empty += not days[day]
        assert result.U_scaled == statistics.median(spreads)
        assert type(result.U_scaled) is float
        assert result.U_quote_units == result.U_scaled / 100000000
        assert result.positive_spread_rows == len(spreads)
        assert result.observed_rows == observed
        assert result.processed_days == day + 1 and result.empty_days == empty
        assert result.train_start_ms == 0
        assert result.train_end_exclusive_ms == (day + 1) * DAY_MS
    with sqlite3.connect(path) as connection:
        histogram = dict(connection.execute('SELECT spread,n FROM spread_counts'))
        assert sum(histogram.values()) == len(spreads)
        assert len(histogram) == len(set(spreads))
        assert connection.execute('SELECT count(*) FROM prefix_fits').fetchone()[0] == 15
        assert connection.execute('SELECT completed FROM fit_scope').fetchone() == (1,)


def test_duplicate_rows_count_and_zero_spreads_do_not_change_rank(tmp_path):
    day = [Q(0, 100, 100), Q(1, 100, 101), Q(1, 100, 101), Q(1, 100, 101), Q(2, 100, 109)]
    results, _ = fit([day], [DAY_MS], tmp_path / 'duplicates.sqlite')
    result = results[0]
    assert result.observed_rows == 5 and result.positive_spread_rows == 4
    assert result.U_scaled == 1.
    assert result.median_lower_scaled == result.median_upper_scaled == 1


def test_even_half_median_near_maximum_price_is_exact_float64(tmp_path):
    day = [Q(0, 1, 2), Q(1, 1, MAX_SCALED_PRICE - 1)]
    results, _ = fit([day], [DAY_MS], tmp_path / 'large.sqlite')
    result = results[0]
    assert result.U_scaled == statistics.median([1, MAX_SCALED_PRICE - 2])
    assert result.U_scaled % 1 == .5
    assert result.median_lower_scaled == 1
    assert result.median_upper_scaled == MAX_SCALED_PRICE - 2


def test_empty_day_is_counted_as_explicit_missing_not_zero_positive_sample(tmp_path):
    results, calls = fit([[], [Q(DAY_MS, 100, 105)]], [2 * DAY_MS], tmp_path / 'empty.sqlite')
    assert results[0].U_scaled == 5
    assert results[0].observed_rows == results[0].positive_spread_rows == 1
    assert results[0].processed_days == 2 and results[0].empty_days == 1
    assert calls == [0, DAY_MS]


def test_zero_positive_checkpoint_raises_before_future_positive_day_and_rolls_back(tmp_path):
    path = tmp_path / 'no_fallback.sqlite'
    calls = []

    def loader(day):
        calls.append(day)
        return QuoteTape([Q(day, 100, 100)] if day == 0 else [Q(day, 100, 110)])

    with pytest.raises(ValueError, match='No positive training spread; no fallback'):
        fit_positive_spread_prefixes(0, 2 * DAY_MS, [DAY_MS, 2 * DAY_MS], loader, path)
    assert calls == [0]
    assert path.is_file() and tables(path) == []
    with pytest.raises(ValueError, match='new and unique'):
        fit_positive_spread_prefixes(0, DAY_MS, [DAY_MS], loader, path)
    assert calls == [0]


def test_prefix_scalar_snapshot_is_suffix_invariant_and_immutable(tmp_path):
    days = [[Q(0, 100, 102)], [Q(DAY_MS, 100, 104)],
            [Q(2 * DAY_MS, 1, 9999999)], [Q(3 * DAY_MS, 1, 3)]]
    earlier, _ = fit(days[:2], [2 * DAY_MS], tmp_path / 'earlier.sqlite')
    extended, calls = fit(days, [2 * DAY_MS, 4 * DAY_MS], tmp_path / 'extended.sqlite')
    assert earlier[0] == extended[0]
    assert extended[0].U_scaled == 3
    assert calls == [0, DAY_MS, 2 * DAY_MS, 3 * DAY_MS]
    with pytest.raises(FrozenInstanceError):
        extended[0].U_scaled = 999


def test_declared_later_suffix_is_not_loaded_without_a_requested_checkpoint(tmp_path):
    results, calls = fit([[Q(0, 100, 102)], [Q(DAY_MS, 100, 104)]],
                         [2 * DAY_MS], tmp_path / 'bounded.sqlite', end=5 * DAY_MS)
    assert calls == [0, DAY_MS]
    assert results[-1].train_end_exclusive_ms == 2 * DAY_MS


def test_nonzero_training_calendar_loads_no_predecessor_and_records_utc_bounds(tmp_path):
    start = 20 * DAY_MS
    results, calls = fit([[Q(start, 100, 103)]], [start + DAY_MS],
                         tmp_path / 'nonzero.sqlite', start=start)
    assert calls == [start]
    assert results[0].train_start_utc == '1970-01-21T00:00:00+00:00'
    assert results[0].train_end_exclusive_utc == '1970-01-22T00:00:00+00:00'


def test_later_loader_crash_rolls_back_earlier_checkpoint_and_all_histogram_rows(tmp_path):
    path = tmp_path / 'crash.sqlite'

    def loader(day):
        if day == DAY_MS:
            raise RuntimeError('Synthetic loader interruption')
        return QuoteTape([Q(day, 100, 102)])

    with pytest.raises(RuntimeError, match='Synthetic loader interruption'):
        fit_positive_spread_prefixes(0, 2 * DAY_MS, [DAY_MS, 2 * DAY_MS], loader, path)
    assert path.is_file() and tables(path) == []


@pytest.mark.parametrize('kind', ['wrong_type', 'subclass', 'outside_day', 'reversed', 'negative',
                                'crossed', 'fractional', 'too_large', 'bad_stamp'])
def test_invalid_later_source_rolls_back_without_accepting_partial_day(kind, tmp_path):
    class OtherTape(QuoteTape):
        pass

    bad = QuoteTape([Q(DAY_MS, 100, 102)])
    if kind == 'wrong_type':
        bad = [Q(DAY_MS, 100, 102)]
    elif kind == 'subclass':
        bad = OtherTape([Q(DAY_MS, 100, 102)])
    elif kind == 'outside_day':
        bad = QuoteTape([Q(DAY_MS, 100, 102), Q(2 * DAY_MS, 100, 103)])
    elif kind == 'reversed':
        object.__setattr__(bad, '_quotes', (Q(DAY_MS + 1, 100, 102), Q(DAY_MS, 100, 102)))
    else:
        field, value = {'negative': ('bid', -1), 'crossed': ('bid', 200),
                        'fractional': ('bid', 100.5), 'too_large': ('ask', MAX_SCALED_PRICE),
                        'bad_stamp': ('time_msc', True)}[kind]
        object.__setattr__(bad.quotes[0], field, value)
    path = tmp_path / (kind + '.sqlite')

    def loader(day):
        return QuoteTape([Q(0, 100, 102)]) if day == 0 else bad

    with pytest.raises(ValueError):
        fit_positive_spread_prefixes(0, 2 * DAY_MS, [DAY_MS, 2 * DAY_MS], loader, path)
    assert tables(path) == []


@pytest.mark.parametrize('start,end,checkpoints', [
    (True, DAY_MS, [DAY_MS]), (-DAY_MS, DAY_MS, [DAY_MS]), (1, DAY_MS, [DAY_MS]),
    (0, DAY_MS + 1, [DAY_MS]), (0, 0, [DAY_MS]), (0, 2**63, [DAY_MS]),
    (0, DAY_MS, []), (0, DAY_MS, [0]), (0, DAY_MS, [DAY_MS + 1]),
    (0, DAY_MS, [2 * DAY_MS]), (0, DAY_MS, [DAY_MS, DAY_MS]),
    (0, 2 * DAY_MS, [2 * DAY_MS, DAY_MS]), (0, DAY_MS, [True]),
])
def test_invalid_calendar_or_checkpoint_list_fails_before_scratch_and_loader(
        start, end, checkpoints, tmp_path):
    calls = []
    path = tmp_path / 'must_not_exist.sqlite'
    with pytest.raises(ValueError):
        fit_positive_spread_prefixes(start, end, checkpoints, lambda day: calls.append(day), path)
    assert calls == [] and not path.exists()


@pytest.mark.parametrize('kind', ['existing', 'directory', 'symlink', 'dangling_symlink',
                                'symlink_parent', 'missing_parent', 'relative', 'traversal',
                                'journal', 'wal', 'shm'])
def test_scratch_path_safety_and_no_existing_file_overwrite(kind, tmp_path):
    path = tmp_path / 'scratch.sqlite'
    original = tmp_path / 'original'
    original.write_bytes(b'preserve this synthetic fixture')
    if kind == 'existing':
        path.write_bytes(b'existing scratch must not be reused')
    elif kind == 'directory':
        path.mkdir()
    elif kind == 'symlink':
        path.symlink_to(original)
    elif kind == 'dangling_symlink':
        path.symlink_to(tmp_path / 'absent')
    elif kind == 'symlink_parent':
        link = tmp_path / 'parent_link';link.symlink_to(tmp_path, target_is_directory=True)
        path = link / 'new.sqlite'
    elif kind == 'missing_parent':
        path = tmp_path / 'absent_parent' / 'new.sqlite'
    elif kind == 'relative':
        path = Path('relative_scratch.sqlite')
    elif kind == 'traversal':
        path = tmp_path / '..' / tmp_path.name / 'new.sqlite'
    else:
        Path(str(path) + '-' + kind).write_bytes(b'existing sidecar')
    calls = []
    with pytest.raises(ValueError):
        fit_positive_spread_prefixes(0, DAY_MS, [DAY_MS], lambda d: calls.append(d), path)
    assert calls == []
    assert original.read_bytes() == b'preserve this synthetic fixture'
    if kind == 'existing':
        assert path.read_bytes() == b'existing scratch must not be reused'


def test_scratch_file_is_private_regular_file_and_success_cannot_be_reused(tmp_path):
    path = tmp_path / 'private.sqlite'
    results, _ = fit([[Q(0, 100, 103)]], [DAY_MS], path)
    assert results[0].U_scaled == 3
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match='new and unique'):
        fit([[Q(0, 100, 103)]], [DAY_MS], path)
