"""Exact positive-spread prefix medians with daily RAM and disk histograms.

This is source access and a scalar TRAIN fit, never strategy/payoff fitting.
Every duplicate contributes, zero spreads do not, and an empty positive prefix
raises without a fallback. Only explicitly requested prefixes are loaded; no
prior scratch DB, directory discovery, future scale or cached fit is reused.

The loader supplies exact QuoteTape instances for declared UTC days. Prices use
the existing integer1e-8 convention below2**45. Histograms are bounded to one
validated day in Python; the cumulative histogram and exact rank queries live
in a fresh SQLite file. Whole-process memory/coverage, historical fees/fills and
cash profit are not established. The caller must later bind actual source and
code pins in a separately frozen study protocol.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sqlite3
import stat
from typing import Callable, Iterable

from independent_profit_research.data import MAX_SCALED_PRICE
from independent_profit_research.tape import QuoteTape
from .sharded import DAY_MS, MAX_TIMESTAMP_MS


MAX_COUNT = 2**63 - 1
PRICE_SCALE = 100_000_000


@dataclass(frozen=True)
class PrefixSpreadFit:
    train_start_ms: int
    train_end_exclusive_ms: int
    train_start_utc: str
    train_end_exclusive_utc: str
    observed_rows: int
    positive_spread_rows: int
    processed_days: int
    empty_days: int
    median_lower_scaled: int
    median_upper_scaled: int
    U_scaled: float
    U_quote_units: float


def _aligned(value, name):
    if type(value) is not int or not 0 <= value <= MAX_TIMESTAMP_MS or value % DAY_MS:
        raise ValueError(name + ' must be a bounded aligned UTC day timestamp')
    try:
        datetime.fromtimestamp(value / 1000, timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError(name + ' must be a supported UTC calendar day') from error
    return value


def _new_scratch_path(raw_path):
    """Reserve a new private regular file; reject existing SQLite sidecars too."""
    try:
        path = Path(raw_path)
    except (TypeError, ValueError) as error:
        raise ValueError('scratch_db_path must be an explicit absolute path') from error
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('scratch_db_path must be absolute without parent traversal')
    for parent in reversed(path.parents):
        try:
            mode = parent.lstat().st_mode
        except OSError as error:
            raise ValueError('scratch parent directories must already exist') from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError('scratch path must have only real nonsymlink parent directories')
    for suffix in ('', '-journal', '-wal', '-shm'):
        if os.path.lexists(str(path) + suffix):
            raise ValueError('scratch database and sidecars must be new and unique')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError('scratch database must be a new regular nonsymlink file') from error
    try:
        identity = os.fstat(fd)
        if not stat.S_ISREG(identity.st_mode):
            raise ValueError('scratch database must be a regular file')
    finally:
        os.close(fd)
    return path, (identity.st_dev, identity.st_ino)


def _same_regular_file(path, identity):
    current = path.lstat()
    if (not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != identity):
        raise ValueError('reserved scratch database identity changed')


def _daily_histogram(supplied, day):
    if type(supplied) is not QuoteTape:
        raise ValueError('loader must return an exact validated QuoteTape')
    # Copy/revalidate actual rows rather than trusting a possibly stale index.
    tape = QuoteTape(supplied.quotes)
    histogram = Counter()
    for quote in tape:
        if not day <= quote.time_msc < day + DAY_MS:
            raise ValueError('source quote must belong to the requested UTC day')
        for price in (quote.bid, quote.ask):
            if (not math.isfinite(price) or not price.is_integer()
                    or not 0 < price < MAX_SCALED_PRICE):
                raise ValueError('Bid/Ask must be positive integer-scaled prices below2**45')
        spread = int(quote.ask - quote.bid)
        if spread > 0:
            histogram[spread] += 1
    return len(tape), histogram


def _median_ranks(connection, n):
    lower_rank, upper_rank = (n - 1) // 2, n // 2
    cumulative, lower, upper = 0, None, None
    for spread, count in connection.execute('SELECT spread, n FROM spread_counts ORDER BY spread'):
        if lower is None and cumulative <= lower_rank < cumulative + count:
            lower = spread
        if cumulative <= upper_rank < cumulative + count:
            upper = spread
            break
        cumulative += count
    if lower is None or upper is None:
        raise ValueError('SQLite histogram does not contain the expected exact ranks')
    # Integer prices below2**45 make the sum and its half exactly Float64.
    return lower, upper, math.fsum((lower, upper)) / 2


def fit_positive_spread_prefixes(
    train_start_ms: int, train_end_ms: int, checkpoint_end_days: Iterable[int],
    loader: Callable[[int], QuoteTape], scratch_db_path: str | Path,
) -> tuple[PrefixSpreadFit, ...]:
    """Fit exact medians for strictly increasing declared TRAIN day-end prefixes.

    Checkpoints are aligned timestamps with TRAINstart < checkpoint <= TRAINend.
    Only days before the last checkpoint are requested. Empty shards are counted
    as explicit missing source, not evidence of zero native activity/coverage.
    A zero-positive checkpoint fails, including when a later prefix would have a
    positive spread. There is one transaction for the whole call: data/loader/SQL
    errors roll back every histogram and fit row, return no earlier fits, and
    leave the failed scratch file reserved so it cannot silently be reused.

    Results are returned as immutable scalar snapshots only after commit. The
    caller chooses a new absolute scratch path in existing nonsymlink directories;
    no source directory, cache, quote file or previous DB is discovered here.
    """
    start = _aligned(train_start_ms, 'train_start_ms')
    end = _aligned(train_end_ms, 'train_end_ms')
    if end <= start:
        raise ValueError('TRAIN end must follow TRAIN start')
    if not callable(loader):
        raise ValueError('loader must be callable')
    checkpoints, previous = [], start
    try:
        for raw in checkpoint_end_days:
            checkpoint = _aligned(raw, 'checkpoint_end_day')
            if not previous < checkpoint <= end:
                raise ValueError('checkpoints must be strictly increasing unique TRAIN day ends')
            checkpoints.append(checkpoint)
            previous = checkpoint
    except TypeError as error:
        raise ValueError('checkpoint_end_days must be an iterable of declared day ends') from error
    if not checkpoints:
        raise ValueError('at least one checkpoint is required')
    path, identity = _new_scratch_path(scratch_db_path)
    connection = None
    try:
        _same_regular_file(path, identity)
        connection = sqlite3.connect(str(path), timeout=0, isolation_level=None)
        _same_regular_file(path, identity)
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute('PRAGMA temp_store=FILE')
        connection.execute('PRAGMA cache_size=-2048')
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('CREATE TABLE spread_counts ('
                           'spread INTEGER PRIMARY KEY CHECK(spread>0), '
                           "n INTEGER NOT NULL CHECK(typeof(n)='integer' AND n>0))")
        connection.execute('CREATE TABLE prefix_fits ('
                           'end_ms INTEGER PRIMARY KEY, start_ms INTEGER NOT NULL, '
                           'observed_rows INTEGER NOT NULL, positive_rows INTEGER NOT NULL, '
                           'processed_days INTEGER NOT NULL, empty_days INTEGER NOT NULL, '
                           'median_lower INTEGER NOT NULL, median_upper INTEGER NOT NULL, '
                           'U_scaled REAL NOT NULL)')
        connection.execute('CREATE TABLE fit_scope ('
                           'start_ms INTEGER, declared_end_ms INTEGER, last_checkpoint_ms INTEGER, '
                           'completed INTEGER CHECK(completed IN (0,1)))')
        connection.execute('INSERT INTO fit_scope VALUES(?,?,?,0)', (start, end, checkpoints[-1]))
        results = []
        observed = positive = processed = empty = 0
        checkpoint_index = 0
        for day in range(start, checkpoints[-1], DAY_MS):
            count, histogram = _daily_histogram(loader(day), day)
            daily_positive = sum(histogram.values())
            if observed + count > MAX_COUNT or positive + daily_positive > MAX_COUNT:
                raise ValueError('source row counts exceed exact SQLite integer capacity')
            connection.executemany(
                'INSERT INTO spread_counts(spread,n) VALUES(?,?) '
                'ON CONFLICT(spread) DO UPDATE SET n=n+excluded.n', histogram.items())
            del histogram  # Do not retain the previous day's Counter during the next load.
            observed += count
            positive += daily_positive
            processed += 1
            empty += count == 0
            if day + DAY_MS != checkpoints[checkpoint_index]:
                continue
            if positive == 0:
                raise ValueError('No positive training spread; no fallback scale')
            lower, upper, unit = _median_ranks(connection, positive)
            fit = PrefixSpreadFit(
                start, day + DAY_MS,
                datetime.fromtimestamp(start / 1000, timezone.utc).isoformat(),
                datetime.fromtimestamp((day + DAY_MS) / 1000, timezone.utc).isoformat(),
                observed, positive, processed, empty, lower, upper, unit, unit / PRICE_SCALE,
            )
            connection.execute('INSERT INTO prefix_fits VALUES(?,?,?,?,?,?,?,?,?)',
                               (fit.train_end_exclusive_ms, start, observed, positive, processed,
                                empty, lower, upper, unit))
            results.append(fit)
            checkpoint_index += 1
        connection.execute('UPDATE fit_scope SET completed=1')
        _same_regular_file(path, identity)
        connection.commit()
        return tuple(results)
    except BaseException:
        if connection is not None and connection.in_transaction:
            connection.rollback()
        raise
    finally:
        if connection is not None:
            connection.close()
