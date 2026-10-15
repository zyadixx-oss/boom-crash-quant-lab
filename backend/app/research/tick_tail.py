"""Pure exploratory tail-tick descriptions, without models or trading decisions.

The caller supplies the chronological training prefix; this module never chooses
a split or reads files. Its only fitted quantity is the median absolute log
return of consecutive one-second training quotes. A tail proxy exceeds ten
times that frozen scale in the declared direction. It is not a census of
physical spikes, and the nominal index number is not a calibrated probability.

Descriptions use only the supplied quotes and the frozen detector. Pre-event
age and prior mark exclude the current increment. A gap resets those states;
the first observed event has an unknown prior age. A continuous midnight pair
is observable and remains continuous, while disconnected days reset the state.
No missing quotes, events or ages are filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
from numbers import Integral, Real
from statistics import median

import numpy as np
import pandas as pd


SECOND_NS = 1_000_000_000
THRESHOLD_MULTIPLIER = 10.0


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _side(value: object) -> int:
    result = _integer(value, "side", -1)
    if result not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool/complex/text")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validated_ticks(ticks: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not isinstance(ticks, pd.DataFrame):
        raise TypeError("ticks must be a DataFrame with a quote column")
    if not ticks.columns.is_unique or "quote" not in ticks:
        raise ValueError("A unique quote column is required")
    index = ticks.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Tick index must contain timezone-aware UTC timestamps")
    if ticks.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Ticks must be nonempty, strictly sorted and unique")
    # Checking the UTC offset also accepts equivalent zero-offset timezone types.
    if any(stamp.utcoffset() != timedelta(0) for stamp in index):
        raise ValueError("Tick timestamps must be UTC")
    index = index.tz_convert("UTC").as_unit("ns")
    if np.any(index.asi8 % SECOND_NS):
        raise ValueError("Tick timestamps must be whole UTC seconds")
    quotes = np.fromiter((_positive_real(value, "quote") for value in ticks.quote.array),
                         dtype=float, count=len(ticks))
    return index, quotes


def _log_return(previous: float, current: float) -> float:
    # log1p preserves small changes on a quantized large price. The fallback
    # avoids a ratio overflow or rounding to -1 for extreme positive quotes.
    relative = (current - previous) / previous
    if math.isfinite(relative) and relative > -1:
        return math.log1p(relative)
    return math.log(current) - math.log(previous)


@dataclass(frozen=True, slots=True)
class FixedTailDetector:
    """Frozen training scale and direction; threshold multiplier is fixed at ten."""

    side: int
    median_abs_log_return: float

    def __post_init__(self) -> None:
        side = _side(self.side)
        scale = _positive_real(self.median_abs_log_return, "median_abs_log_return")
        if not math.isfinite(THRESHOLD_MULTIPLIER * scale):
            raise ValueError("Tail threshold must be finite")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "median_abs_log_return", scale)

    @property
    def threshold(self) -> float:
        return THRESHOLD_MULTIPLIER * self.median_abs_log_return


def fit_fixed_tail_detector(training_ticks: pd.DataFrame, side: int) -> FixedTailDetector:
    """Fit only the supplied training prefix, excluding every nonconsecutive pair.

    The caller must select the training-only prefix before calling this function
    (for the proposed pilot, earliest 40% of observed chronological seconds).
    Zero median scale is refused; there is no numerical floor, alternate window,
    nominal-frequency quantile or retrospective threshold search.
    """
    side = _side(side)
    index, quotes = _validated_ticks(training_ticks)
    times = index.asi8
    values = [abs(_log_return(float(quotes[i-1]), float(quotes[i])))
              for i in range(1, len(quotes)) if int(times[i])-int(times[i-1]) == SECOND_NS]
    if not values:
        raise ValueError("At least one observed consecutive training increment is required")
    return FixedTailDetector(side, median(values))


def describe_tick_tail(ticks: pd.DataFrame, detector: FixedTailDetector,
                       nominal_n: int) -> pd.DataFrame:
    """Describe observed increments and strictly prior event state, without fitting.

    ``nominal_n`` is only the boundary for ``lt_N``/``ge_N`` age bands. Event
    detection uses the detector's frozen scale, never a nominal event rate.
    ``tail_event`` is nullable: an unobserved increment is unknown, not False.
    Neither age nor prior mark is known before the first detected event in a
    contiguous segment. Current events update the state only for later rows.
    """
    if not isinstance(detector, FixedTailDetector):
        raise TypeError("detector must be a FixedTailDetector")
    nominal_n = _integer(nominal_n, "nominal_n", 1)
    index, quotes = _validated_ticks(ticks)
    times = index.asi8
    returns = np.full(len(quotes), np.nan)
    ages = np.full(len(quotes), np.nan)
    marks = np.full(len(quotes), np.nan)
    valid = np.zeros(len(quotes), dtype=bool)
    events: list[bool | None] = [None] * len(quotes)
    bands: list[str | None] = [None] * len(quotes)
    last_event_time: int | None = None
    last_mark: float | None = None
    for i in range(1, len(quotes)):
        stamp = int(times[i])
        if stamp - int(times[i-1]) != SECOND_NS:
            last_event_time, last_mark = None, None
            continue
        valid[i] = True
        if last_event_time is not None:
            age = (stamp-last_event_time) // SECOND_NS
            ages[i], marks[i] = age, last_mark
            bands[i] = "lt_N" if age < nominal_n else "ge_N"
        directed = detector.side * _log_return(float(quotes[i-1]), float(quotes[i]))
        returns[i] = directed
        event = directed > detector.threshold
        events[i] = event
        if event:
            last_event_time, last_mark = stamp, directed
    return pd.DataFrame({"quote": quotes, "increment_known": valid,
                         "directed_log_return": returns,
                         "tail_event": pd.array(events, dtype="boolean"),
                         "pre_event_age_seconds": ages, "prior_tail_mark": marks,
                         "age_band": pd.array(bands, dtype="string")}, index=index)
