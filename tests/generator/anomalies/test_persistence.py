"""Persistence: structure created to keep access after the entry point is closed."""

from itertools import pairwise

from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import available_patterns, get_pattern
from rga.util.timeutil import HOUR_MS


def test_shadow_group_has_exactly_one_member(context_factory) -> None:
    injection = get_pattern("shadow_group").inject(context_factory(seed=41))

    memberships = [
        event for event in injection.events if event.relation is RelationType.MEMBER_OF
    ]
    assert len(memberships) == 1
    assert memberships[0].object.startswith("group:svc-")


def test_shadow_group_takes_admin_on_several_buckets(context_factory) -> None:
    injection = get_pattern("shadow_group").inject(context_factory(seed=42))

    grants = [
        event for event in injection.events if event.relation is RelationType.HAS_PERMISSION
    ]
    assert len(grants) >= 4
    assert all(event.level is PermissionLevel.ADMIN for event in grants)
    assert len({event.subject for event in grants}) == 1


def test_shadow_group_is_a_new_node(context_factory) -> None:
    context = context_factory(seed=43)
    injection = get_pattern("shadow_group").inject(context)
    group_id = next(
        event.object for event in injection.events if event.relation is RelationType.MEMBER_OF
    )
    assert group_id not in context.graph.node_index


def test_delegation_cascade_chains_actor_to_previous_subject(context_factory) -> None:
    injection = get_pattern("delegation_cascade").inject(context_factory(seed=44))
    events = injection.events

    assert len(events) == 3
    assert len({event.object for event in events}) == 1
    for earlier, later in pairwise(events):
        assert later.actor == earlier.subject


def test_delegation_cascade_runs_quickly_and_in_order(context_factory) -> None:
    context = context_factory(seed=45)
    events = get_pattern("delegation_cascade").inject(context).events

    assert [event.ts for event in events] == sorted(event.ts for event in events)
    assert events[-1].ts - events[0].ts <= 2 * HOUR_MS
    assert all(context.window[0] <= event.ts < context.window[1] for event in events)


def test_delegation_cascade_uses_distinct_subjects(context_factory) -> None:
    events = get_pattern("delegation_cascade").inject(context_factory(seed=46)).events
    assert len({event.subject for event in events}) == len(events)


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=47)
    for name in ("shadow_group", "delegation_cascade"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)


def test_every_pattern_registers() -> None:
    """The last two are collective shapes, used only by the divergent dataset."""
    assert set(available_patterns()) == {
        "self_grant_admin",
        "privileged_group_join",
        "grant_burst",
        "dormant_awakening",
        "hierarchy_bypass",
        "cross_department",
        "shadow_group",
        "delegation_cascade",
        "mutual_grant_ring",
        "convergent_access",
    }
