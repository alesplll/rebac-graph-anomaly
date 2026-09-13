"""Typed configuration for dataset generation.

Plain dataclasses over a YAML document: the configuration is small and fully
enumerated here, so an extra validation dependency would buy nothing. Validation
is explicit and fails at load time, not halfway through a generation run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OrgConfig:
    """Shape of the modelled organization."""

    departments: int
    teams_per_department: tuple[int, int]
    users_per_team: tuple[int, int]
    projects_per_team: tuple[int, int]
    objects_per_bucket: tuple[int, int]
    cross_cutting_groups: int
    cross_cutting_membership_rate: float
    #: Share of grants made outside the standard group procedure. Without these
    #: the graph is a perfect hierarchy and every deviation is trivially anomalous.
    legitimate_exception_rate: float


@dataclass(frozen=True)
class TimelineConfig:
    """Intensity of the growth process."""

    start_ts: int
    days: int
    working_hours: tuple[int, int]
    off_hours_rate: float
    weekend_rate: float
    hires_per_day: float
    departures_per_day: float
    grants_per_day: float
    uploads_per_day: float


@dataclass(frozen=True)
class AnomalyConfig:
    """Which patterns to inject and how densely."""

    patterns: tuple[str, ...]
    #: Target share of edges in the evaluation window that are anomalous.
    rate: float
    #: Share of anomalous edges planted in the training span, to test robustness
    #: to a training graph that is not perfectly clean.
    train_contamination: float


@dataclass(frozen=True)
class DatasetConfig:
    """A complete, reproducible dataset recipe."""

    name: str
    seed: int
    org: OrgConfig
    timeline: TimelineConfig
    anomalies: AnomalyConfig
    eval_window_days: int


def _pair(raw: object, field_name: str) -> tuple[int, int]:
    """Read an inclusive [low, high] range and check its ordering."""
    values = tuple(int(item) for item in raw)  # type: ignore[union-attr]
    if len(values) != 2:
        raise ValueError(f"{field_name} must have exactly two values, got {values}")
    low, high = values
    if low > high:
        raise ValueError(f"{field_name} range is inverted: {low} > {high}")
    return low, high


def _rate(raw: object, field_name: str) -> float:
    value = float(raw)  # type: ignore[arg-type]
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1], got {value}")
    return value


def _org_config(document: Mapping[str, object]) -> OrgConfig:
    return OrgConfig(
        departments=int(document["departments"]),  # type: ignore[arg-type]
        teams_per_department=_pair(document["teams_per_department"], "teams_per_department"),
        users_per_team=_pair(document["users_per_team"], "users_per_team"),
        projects_per_team=_pair(document["projects_per_team"], "projects_per_team"),
        objects_per_bucket=_pair(document["objects_per_bucket"], "objects_per_bucket"),
        cross_cutting_groups=int(document["cross_cutting_groups"]),  # type: ignore[arg-type]
        cross_cutting_membership_rate=_rate(
            document["cross_cutting_membership_rate"], "cross_cutting_membership_rate"
        ),
        legitimate_exception_rate=_rate(
            document["legitimate_exception_rate"], "legitimate_exception_rate"
        ),
    )


def _timeline_config(document: Mapping[str, object]) -> TimelineConfig:
    hours = _pair(document["working_hours"], "working_hours")
    if not 0 <= hours[0] < hours[1] <= 24:
        raise ValueError(f"working_hours must lie within [0, 24], got {hours}")
    return TimelineConfig(
        start_ts=int(document["start_ts"]),  # type: ignore[arg-type]
        days=int(document["days"]),  # type: ignore[arg-type]
        working_hours=hours,
        off_hours_rate=_rate(document["off_hours_rate"], "off_hours_rate"),
        weekend_rate=_rate(document["weekend_rate"], "weekend_rate"),
        hires_per_day=float(document["hires_per_day"]),  # type: ignore[arg-type]
        departures_per_day=float(document["departures_per_day"]),  # type: ignore[arg-type]
        grants_per_day=float(document["grants_per_day"]),  # type: ignore[arg-type]
        uploads_per_day=float(document["uploads_per_day"]),  # type: ignore[arg-type]
    )


def dataset_config_from_document(document: Mapping[str, object]) -> DatasetConfig:
    """Build and validate a config from an already-parsed document."""
    timeline = _timeline_config(document["timeline"])  # type: ignore[arg-type]
    eval_window_days = int(document["eval_window_days"])  # type: ignore[arg-type]
    if eval_window_days >= timeline.days:
        raise ValueError(
            f"eval window of {eval_window_days} days does not fit in {timeline.days} days"
        )
    anomalies = document["anomalies"]
    return DatasetConfig(
        name=str(document["name"]),
        seed=int(document["seed"]),  # type: ignore[arg-type]
        org=_org_config(document["org"]),  # type: ignore[arg-type]
        timeline=timeline,
        anomalies=AnomalyConfig(
            patterns=tuple(str(name) for name in anomalies["patterns"]),  # type: ignore[index]
            rate=_rate(anomalies["rate"], "anomalies.rate"),  # type: ignore[index]
            train_contamination=_rate(
                anomalies["train_contamination"],  # type: ignore[index]
                "anomalies.train_contamination",
            ),
        ),
        eval_window_days=eval_window_days,
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset recipe from a YAML file."""
    return dataset_config_from_document(yaml.safe_load(path.read_text(encoding="utf-8")))
