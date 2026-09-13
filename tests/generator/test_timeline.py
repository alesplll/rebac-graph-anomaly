"""The normal growth process."""

import numpy as np

from rga.domain.events import EventOp
from rga.domain.relations import RelationType
from rga.domain.replay import journal_issues, replay
from rga.generator.config import OrgConfig, TimelineConfig
from rga.generator.org import build_organization
from rga.generator.timeline import generate_normal_journal
from rga.util.timeutil import DAY_MS, is_weekend

ORG = OrgConfig(
    departments=2,
    teams_per_department=(2, 3),
    users_per_team=(3, 5),
    projects_per_team=(1, 2),
    objects_per_bucket=(3, 10),
    cross_cutting_groups=1,
    cross_cutting_membership_rate=0.1,
    legitimate_exception_rate=0.05,
)
TIMELINE = TimelineConfig(
    start_ts=1_735_689_600_000,
    days=40,
    working_hours=(9, 19),
    off_hours_rate=0.05,
    weekend_rate=0.08,
    hires_per_day=0.5,
    departures_per_day=0.15,
    grants_per_day=3.0,
    uploads_per_day=8.0,
)


def _journal(seed: int = 3):
    rng = np.random.default_rng(seed)
    org = build_organization(ORG, rng)
    journal = generate_normal_journal(
        org, TIMELINE, rng, exception_rate=ORG.legitimate_exception_rate
    )
    return org, journal


def test_journal_is_sorted_by_time() -> None:
    _, events = _journal()
    assert [event.ts for event in events] == sorted(event.ts for event in events)


def test_journal_stays_inside_the_configured_span() -> None:
    _, events = _journal()
    end = TIMELINE.start_ts + TIMELINE.days * DAY_MS
    assert all(TIMELINE.start_ts <= event.ts < end for event in events)


def test_journal_is_internally_consistent() -> None:
    # No revoke of an edge that does not exist at that moment.
    _, events = _journal()
    assert journal_issues(events) == []


def test_every_team_group_receives_rights_on_its_own_buckets() -> None:
    org, events = _journal()
    granted = {
        (event.subject, event.object)
        for event in events
        if event.op is EventOp.GRANT and event.relation is RelationType.HAS_PERMISSION
    }
    for team in org.teams:
        for bucket in team.buckets:
            assert (team.group_id, bucket) in granted


def test_objects_are_attached_to_their_bucket() -> None:
    org, events = _journal()
    parent_edges = {
        (event.subject, event.object)
        for event in events
        if event.relation is RelationType.PARENT_OF
    }
    for bucket in org.buckets:
        attached = {obj for parent, obj in parent_edges if parent == bucket.id}
        assert attached, f"{bucket.id} has no objects attached"


def test_objects_carry_their_own_permissions() -> None:
    # The engine does not inherit rights through containment, so the normal
    # process must grant on objects explicitly. Without this, any object-level
    # right would look anomalous by itself and the bypass pattern would be
    # detectable without a model at all.
    _, events = _journal()
    with_rights = {
        event.object
        for event in events
        if event.relation is RelationType.HAS_PERMISSION and event.object.startswith("object:")
    }
    assert len(with_rights) > 10


def test_weekends_are_quieter_than_weekdays() -> None:
    _, events = _journal()
    weekend = sum(1 for event in events if is_weekend(event.ts))
    assert 0 < weekend < len(events) * 0.2


def test_generation_is_reproducible() -> None:
    assert _journal(3)[1] == _journal(3)[1]


def test_the_resulting_graph_is_not_trivial() -> None:
    _, events = _journal()
    graph = replay(events)
    assert graph.num_nodes > 50
    assert graph.num_edges > 50
