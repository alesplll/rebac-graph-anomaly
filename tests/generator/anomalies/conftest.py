"""A small but realistic graph for pattern tests.

Exposed as a fixture rather than an importable helper: pytest puts this
directory on sys.path, not the repository root, so a cross-package import of a
conftest module would not resolve.
"""

import numpy as np
import pytest

import rga.generator.anomalies  # noqa: F401  registers the patterns
from rga.domain.replay import replay
from rga.generator.anomalies.base import InjectionContext
from rga.generator.config import OrgConfig, TimelineConfig
from rga.generator.org import build_organization
from rga.generator.timeline import generate_normal_journal
from rga.util.timeutil import DAY_MS

START = 1_735_689_600_000
TRAIN_DAYS = 60
WINDOW_DAYS = 14


@pytest.fixture
def context_factory():
    """Return a builder for the context patterns receive."""

    def build(seed: int, *, cross_cutting_groups: int = 2) -> InjectionContext:
        org_config = OrgConfig(
            departments=3,
            teams_per_department=(2, 4),
            users_per_team=(4, 8),
            projects_per_team=(1, 3),
            objects_per_bucket=(5, 20),
            cross_cutting_groups=cross_cutting_groups,
            cross_cutting_membership_rate=0.1,
            legitimate_exception_rate=0.04,
        )
        timeline_config = TimelineConfig(
            start_ts=START,
            days=TRAIN_DAYS,
            working_hours=(9, 19),
            off_hours_rate=0.05,
            weekend_rate=0.08,
            hires_per_day=0.6,
            departures_per_day=0.1,
            grants_per_day=5.0,
            uploads_per_day=15.0,
        )

        rng = np.random.default_rng(seed)
        org = build_organization(org_config, rng)
        journal = generate_normal_journal(
            org, timeline_config, rng, exception_rate=org_config.legitimate_exception_rate
        )
        cutoff = START + TRAIN_DAYS * DAY_MS
        return InjectionContext(
            rng=rng,
            org=org,
            graph=replay(journal, until=cutoff),
            window=(cutoff, cutoff + WINDOW_DAYS * DAY_MS),
        )

    return build
