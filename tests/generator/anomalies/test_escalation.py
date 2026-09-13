"""Vertical escalation: a subject acquires rights it should not be able to give itself."""

import pytest

from rga.domain.events import EventOp
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import NoCandidateError, get_pattern, level_on


def test_self_grant_admin_makes_subject_its_own_actor(context_factory) -> None:
    context = context_factory(seed=11)
    injection = get_pattern("self_grant_admin").inject(context)

    assert len(injection.events) == 1
    event = injection.events[0]
    assert event.op is EventOp.GRANT
    assert event.relation is RelationType.HAS_PERMISSION
    assert event.level is PermissionLevel.ADMIN
    assert event.actor == event.subject


def test_self_grant_admin_targets_a_bucket_it_did_not_already_administer(context_factory) -> None:
    context = context_factory(seed=12)
    event = get_pattern("self_grant_admin").inject(context).events[0]
    assert level_on(context.graph, event.subject, event.object) < PermissionLevel.ADMIN


def test_privileged_group_join_adds_a_non_member_to_a_cross_cutting_group(
    context_factory,
) -> None:
    context = context_factory(seed=13)
    injection = get_pattern("privileged_group_join").inject(context)

    event = injection.events[0]
    assert event.relation is RelationType.MEMBER_OF
    assert event.actor == event.subject
    group_ids = {group.id for group in context.org.cross_cutting}
    assert event.object in group_ids
    joined = next(group for group in context.org.cross_cutting if group.id == event.object)
    assert event.subject not in joined.members


def test_events_land_inside_the_window(context_factory) -> None:
    context = context_factory(seed=14)
    for name in ("self_grant_admin", "privileged_group_join"):
        for event in get_pattern(name).inject(context).events:
            assert context.window[0] <= event.ts < context.window[1]


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=15)
    for name in ("self_grant_admin", "privileged_group_join"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)


def test_privileged_group_join_gives_up_without_cross_cutting_groups(context_factory) -> None:
    context = context_factory(seed=16, cross_cutting_groups=0)
    with pytest.raises(NoCandidateError):
        get_pattern("privileged_group_join").inject(context)


def test_injection_is_reproducible(context_factory) -> None:
    first = get_pattern("self_grant_admin").inject(context_factory(seed=17))
    second = get_pattern("self_grant_admin").inject(context_factory(seed=17))
    assert first == second
