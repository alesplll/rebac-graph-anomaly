"""Patterns that exist only in a group of edges, not in any single one."""

from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import available_patterns, get_pattern
from rga.util.timeutil import DAY_MS


def test_both_collective_patterns_register() -> None:
    assert "mutual_grant_ring" in available_patterns()
    assert "convergent_access" in available_patterns()


def test_the_ring_closes(context_factory) -> None:
    """Everybody in the ring both gives once and receives once."""
    injection = get_pattern("mutual_grant_ring").inject(context_factory(seed=61))
    events = injection.events

    assert len(events) >= 3
    assert {event.subject for event in events} == {event.actor for event in events}
    assert len({event.subject for event in events}) == len(events)


def test_nobody_in_the_ring_grants_to_themselves(context_factory) -> None:
    """Otherwise it would be self_grant_admin, which the model has been trained on."""
    events = get_pattern("mutual_grant_ring").inject(context_factory(seed=62)).events

    assert all(event.subject != event.actor for event in events)


def test_every_edge_of_the_ring_looks_ordinary_on_its_own(context_factory) -> None:
    """The closure is the anomaly; no single edge of it should stand out."""
    events = get_pattern("mutual_grant_ring").inject(context_factory(seed=63)).events

    assert all(event.relation is RelationType.HAS_PERMISSION for event in events)
    assert all(event.level is not PermissionLevel.ADMIN for event in events)


def test_convergent_access_lands_on_one_object(context_factory) -> None:
    events = get_pattern("convergent_access").inject(context_factory(seed=64)).events

    assert len({event.object for event in events}) == 1
    assert len({event.subject for event in events}) == len(events)
    assert len(events) >= 5


def test_convergent_access_is_not_a_burst(context_factory) -> None:
    """The transpose of grant_burst must not be separable by the clock instead."""
    context = context_factory(seed=65)
    events = get_pattern("convergent_access").inject(context).events

    assert [event.ts for event in events] == sorted(event.ts for event in events)
    assert events[-1].ts - events[0].ts >= DAY_MS
    assert all(context.window[0] <= event.ts < context.window[1] for event in events)


def test_convergent_access_grants_a_modest_level(context_factory) -> None:
    events = get_pattern("convergent_access").inject(context_factory(seed=66)).events

    assert all(event.level is PermissionLevel.READ for event in events)


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=67)
    for name in ("mutual_grant_ring", "convergent_access"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)
