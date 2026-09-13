"""Account compromise: rights appear at a rate or after a silence that does not fit."""

from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import get_pattern, last_activity
from rga.util.timeutil import DAY_MS, MINUTE_MS


def test_grant_burst_comes_from_one_subject_in_a_short_span(context_factory) -> None:
    injection = get_pattern("grant_burst").inject(context_factory(seed=21))

    assert len(injection.events) >= 6
    assert len({event.subject for event in injection.events}) == 1
    span = max(event.ts for event in injection.events) - min(
        event.ts for event in injection.events
    )
    assert span <= 45 * MINUTE_MS


def test_grant_burst_targets_buckets_outside_the_subjects_team(context_factory) -> None:
    context = context_factory(seed=22)
    injection = get_pattern("grant_burst").inject(context)

    subject = injection.events[0].subject
    own_team = context.org.team_of(subject).id
    for event in injection.events:
        assert context.org.bucket(event.object).team != own_team


def test_grant_burst_events_are_ordered_and_inside_the_window(context_factory) -> None:
    context = context_factory(seed=23)
    events = get_pattern("grant_burst").inject(context).events
    assert [event.ts for event in events] == sorted(event.ts for event in events)
    assert all(context.window[0] <= event.ts < context.window[1] for event in events)


def test_dormant_awakening_picks_a_long_silent_account(context_factory) -> None:
    context = context_factory(seed=24)
    event = get_pattern("dormant_awakening").inject(context).events[0]

    silence = context.window[0] - last_activity(context.graph, event.subject)
    assert silence >= 21 * DAY_MS
    assert event.relation is RelationType.HAS_PERMISSION
    assert event.level >= PermissionLevel.WRITE


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=25)
    for name in ("grant_burst", "dormant_awakening"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)


def test_patterns_are_reproducible(context_factory) -> None:
    for name in ("grant_burst", "dormant_awakening"):
        assert get_pattern(name).inject(context_factory(seed=26)) == get_pattern(name).inject(
            context_factory(seed=26)
        )
