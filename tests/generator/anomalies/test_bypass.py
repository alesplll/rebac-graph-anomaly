"""Bypassing the delegation hierarchy, horizontally and vertically."""

from rga.domain.entities import EntityType, entity_type, parent_bucket
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import get_pattern, level_on


def test_hierarchy_bypass_grants_on_an_object_not_its_bucket(context_factory) -> None:
    context = context_factory(seed=31)
    event = get_pattern("hierarchy_bypass").inject(context).events[0]

    assert entity_type(event.object) is EntityType.OBJECT
    assert event.relation is RelationType.HAS_PERMISSION
    # Object-level rights are ordinary; holding one with nothing on the
    # containing bucket is not.
    assert (
        level_on(context.graph, event.subject, parent_bucket(event.object))
        is PermissionLevel.NONE
    )


def test_hierarchy_bypass_is_not_issued_by_the_subject(context_factory) -> None:
    context = context_factory(seed=32)
    event = get_pattern("hierarchy_bypass").inject(context).events[0]
    assert event.actor is not None
    assert event.actor != event.subject


def test_cross_department_crosses_a_department_boundary(context_factory) -> None:
    context = context_factory(seed=33)
    event = get_pattern("cross_department").inject(context).events[0]

    subject_department = context.org.department_of(event.subject)
    bucket = context.org.bucket(event.object)
    assert subject_department != context.org.team(bucket.team).department


def test_cross_department_approver_belongs_to_the_target_department(context_factory) -> None:
    context = context_factory(seed=34)
    event = get_pattern("cross_department").inject(context).events[0]

    bucket = context.org.bucket(event.object)
    target_department = context.org.team(bucket.team).department
    assert context.org.department_of(event.actor) == target_department
    assert event.actor != bucket.owner


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=35)
    for name in ("hierarchy_bypass", "cross_department"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)


def test_events_land_inside_the_window(context_factory) -> None:
    context = context_factory(seed=36)
    for name in ("hierarchy_bypass", "cross_department"):
        for event in get_pattern(name).inject(context).events:
            assert context.window[0] <= event.ts < context.window[1]
