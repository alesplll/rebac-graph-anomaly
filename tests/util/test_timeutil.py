"""Time helpers. All timestamps in the project are UTC milliseconds."""

import numpy as np

from rga.generator.config import TimelineConfig
from rga.util.timeutil import DAY_MS, day_start, hour_of_day, is_weekend, sample_time_of_day

# 2025-01-01 was a Wednesday.
WEDNESDAY = 1_735_689_600_000

CONFIG = TimelineConfig(
    start_ts=WEDNESDAY,
    days=30,
    working_hours=(9, 19),
    off_hours_rate=0.05,
    weekend_rate=0.08,
    hires_per_day=1.0,
    departures_per_day=0.2,
    grants_per_day=5.0,
    uploads_per_day=10.0,
)


def test_day_start_advances_by_whole_days() -> None:
    assert day_start(WEDNESDAY, 0) == WEDNESDAY
    assert day_start(WEDNESDAY, 3) == WEDNESDAY + 3 * DAY_MS


def test_weekend_detection() -> None:
    assert not is_weekend(WEDNESDAY)
    assert is_weekend(day_start(WEDNESDAY, 3))  # Saturday
    assert is_weekend(day_start(WEDNESDAY, 4))  # Sunday
    assert not is_weekend(day_start(WEDNESDAY, 5))  # Monday


def test_sampled_times_stay_inside_their_day() -> None:
    rng = np.random.default_rng(0)
    start = day_start(WEDNESDAY, 2)
    for _ in range(200):
        ts = sample_time_of_day(rng, start, CONFIG)
        assert start <= ts < start + DAY_MS


def test_most_activity_lands_in_working_hours() -> None:
    rng = np.random.default_rng(0)
    hours = [hour_of_day(sample_time_of_day(rng, WEDNESDAY, CONFIG)) for _ in range(2000)]
    inside = sum(1 for hour in hours if 9 <= hour < 19)
    assert inside / len(hours) > 0.85


def test_off_hours_activity_is_present_but_rare() -> None:
    rng = np.random.default_rng(0)
    hours = [hour_of_day(sample_time_of_day(rng, WEDNESDAY, CONFIG)) for _ in range(2000)]
    outside = sum(1 for hour in hours if not 9 <= hour < 19)
    assert 0 < outside / len(hours) < 0.15
