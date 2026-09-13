"""Time helpers. Every timestamp in this project is UTC milliseconds since epoch."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

SECOND_MS = 1_000
MINUTE_MS = 60 * SECOND_MS
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS


def day_start(start_ts: int, day_index: int) -> int:
    """Midnight of the given day, counted from `start_ts`."""
    return start_ts + day_index * DAY_MS


def hour_of_day(ts: int) -> int:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).hour


def is_weekend(ts: int) -> bool:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).weekday() >= 5


def sample_time_of_day(rng: np.random.Generator, day_start_ts: int, config) -> int:
    """Draw a moment within a day, concentrated in working hours.

    The diurnal rhythm is not decoration. "Granted at three in the morning" is
    only a signal if normal activity has a shape to deviate from.
    """
    low, high = config.working_hours
    if rng.random() < config.off_hours_rate:
        off_hours = [hour for hour in range(24) if not low <= hour < high]
        hour = int(off_hours[int(rng.integers(len(off_hours)))])
    else:
        hour = int(rng.integers(low, high))
    return (
        day_start_ts
        + hour * HOUR_MS
        + int(rng.integers(60)) * MINUTE_MS
        + int(rng.integers(60)) * SECOND_MS
    )
