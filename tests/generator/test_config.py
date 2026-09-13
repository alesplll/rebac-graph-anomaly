"""Configuration loading and validation."""

from pathlib import Path

import pytest

from rga.generator.config import load_dataset_config

_BAD_TEMPLATE = """\
name: bad
seed: 1
eval_window_days: {eval_window_days}
org:
  departments: 1
  teams_per_department: [{teams_low}, {teams_high}]
  users_per_team: [2, 2]
  projects_per_team: [1, 1]
  objects_per_bucket: [1, 2]
  cross_cutting_groups: 0
  cross_cutting_membership_rate: 0.0
  legitimate_exception_rate: 0.0
timeline:
  start_ts: 0
  days: 10
  working_hours: [9, 18]
  off_hours_rate: 0.05
  weekend_rate: 0.08
  hires_per_day: 1.0
  departures_per_day: 0.1
  grants_per_day: 1.0
  uploads_per_day: 1.0
anomalies:
  patterns: []
  rate: 0.01
  train_contamination: 0.0
"""


def _write(tmp_path: Path, **fields: object) -> Path:
    defaults = {"eval_window_days": 2, "teams_low": 1, "teams_high": 1}
    defaults.update(fields)
    path = tmp_path / "bad.yaml"
    path.write_text(_BAD_TEMPLATE.format(**defaults), encoding="utf-8")
    return path


def test_shipped_small_config_loads() -> None:
    config = load_dataset_config(Path("configs/generator/small.yaml"))
    assert config.seed >= 0
    assert config.org.departments >= 1
    assert config.timeline.days > config.eval_window_days
    assert 0.0 < config.anomalies.rate < 1.0


def test_shipped_default_config_loads() -> None:
    config = load_dataset_config(Path("configs/generator/default.yaml"))
    assert config.org.departments >= config.org.cross_cutting_groups
    assert config.anomalies.patterns


def test_eval_window_must_fit_inside_the_timeline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval window"):
        load_dataset_config(_write(tmp_path, eval_window_days=30))


def test_a_range_must_be_ordered(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="teams_per_department"):
        load_dataset_config(_write(tmp_path, teams_low=5, teams_high=1))
