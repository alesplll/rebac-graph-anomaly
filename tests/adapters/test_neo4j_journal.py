"""A level-1 snapshot reconstructs its own change log."""

from pathlib import Path

import pytest

from rga.adapters.mapping import RelationMapping
from rga.adapters.neo4j_source import Neo4jSource
from rga.domain.events import EventOp
from rga.domain.relations import PermissionLevel, RelationType

MAPPING = RelationMapping.load(Path("configs/mapping/opens3.yaml"))


class _Reader:
    """Rows in whatever order the store returned them."""

    def __init__(self, relationships):
        self._relationships = relationships

    def nodes(self):
        seen = {}
        for row in self._relationships:
            seen.setdefault(row["subject"], row.get("created_at"))
            seen.setdefault(row["object"], row.get("created_at"))
        return iter([{"id": key, "created_at": value} for key, value in seen.items()])

    def relationships(self):
        return iter(self._relationships)


def _rows():
    return [
        {
            "subject": "group:devops",
            "relation": "HAS_PERMISSION",
            "object": "bucket:logs",
            "level": "admin",
            "created_at": 3000,
            "actor": "user:root",
        },
        {
            "subject": "user:alice",
            "relation": "MEMBER_OF",
            "object": "group:devops",
            "level": None,
            "created_at": 1000,
            "actor": None,
        },
    ]


def test_events_come_out_in_time_order() -> None:
    source = Neo4jSource(_Reader(_rows()), MAPPING)

    events = list(source.events())

    assert [event.ts for event in events] == [1000, 3000]
    assert events[0].relation is RelationType.MEMBER_OF
    assert events[1].level is PermissionLevel.ADMIN
    assert events[1].actor == "user:root"


def test_every_event_is_a_grant() -> None:
    """A snapshot cannot show a revocation: a revoked edge is simply absent."""
    source = Neo4jSource(_Reader(_rows()), MAPPING)

    assert {event.op for event in source.events()} == {EventOp.GRANT}


def test_the_window_is_honoured() -> None:
    source = Neo4jSource(_Reader(_rows()), MAPPING)

    assert [event.ts for event in source.events(since=2000)] == [3000]
    assert [event.ts for event in source.events(until=2000)] == [1000]


def test_edges_from_before_the_patch_come_first() -> None:
    rows = [
        *_rows(),
        {
            "subject": "user:bob",
            "relation": "MEMBER_OF",
            "object": "group:devops",
            "level": None,
            "created_at": None,
            "actor": None,
        },
    ]

    events = list(Neo4jSource(_Reader(rows), MAPPING).events())

    assert events[0].subject == "user:bob"
    assert events[0].ts < 1000


def test_a_source_without_timestamps_cannot_produce_a_journal() -> None:
    rows = [dict(row, created_at=None) for row in _rows()]

    with pytest.raises(ValueError, match="no timestamps"):
        list(Neo4jSource(_Reader(rows), MAPPING).events())
