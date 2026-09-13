"""Reading a live authorization graph, including one written before the patches."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from rga.adapters.base import GraphSource
from rga.adapters.mapping import RelationMapping
from rga.adapters.neo4j_source import Neo4jSource
from rga.domain.graph import UNKNOWN
from rga.domain.relations import PermissionLevel, RelationType

MAPPING = RelationMapping.load(Path("configs/mapping/opens3.yaml"))


class FakeReader:
    def __init__(self, nodes: list[dict], relationships: list[dict]) -> None:
        self._nodes = nodes
        self._relationships = relationships

    def nodes(self) -> Iterator[Mapping[str, object]]:
        return iter(self._nodes)

    def relationships(self) -> Iterator[Mapping[str, object]]:
        return iter(self._relationships)


def _modern() -> FakeReader:
    return FakeReader(
        nodes=[
            {"id": "user:alice", "created_at": 1_000},
            {"id": "group:devops", "created_at": 900},
            {"id": "bucket:photos", "created_at": 800},
        ],
        relationships=[
            {
                "subject": "user:alice", "relation": "MEMBER_OF", "object": "group:devops",
                "level": None, "created_at": 1_100, "actor": "user:root",
            },
            {
                "subject": "group:devops", "relation": "HAS_PERMISSION",
                "object": "bucket:photos", "level": "write", "created_at": 1_200,
                "actor": "user:root",
            },
        ],
    )


def _legacy() -> FakeReader:
    """A graph written before the timestamp and actor patches."""
    return FakeReader(
        nodes=[
            {"id": "user:alice", "created_at": None},
            {"id": "group:devops", "created_at": None},
        ],
        relationships=[
            {
                "subject": "user:alice", "relation": "MEMBER_OF", "object": "group:devops",
                "level": None, "created_at": None, "actor": None,
            }
        ],
    )


def test_source_satisfies_the_protocol() -> None:
    assert isinstance(Neo4jSource(_modern(), MAPPING), GraphSource)


def test_patched_deployment_reaches_level_two() -> None:
    capabilities = Neo4jSource(_modern(), MAPPING).capabilities()
    assert capabilities.level == 2
    # A snapshot cannot show revocations, so there is no change log either way.
    assert capabilities.change_log is False


def test_unpatched_deployment_is_level_zero() -> None:
    assert Neo4jSource(_legacy(), MAPPING).capabilities().level == 0


def test_snapshot_translates_relations_and_levels() -> None:
    graph = Neo4jSource(_modern(), MAPPING).snapshot()
    assert graph.num_nodes == 4  # three nodes plus the actor
    assert graph.num_edges == 2

    permission = graph.edge_rel == int(RelationType.HAS_PERMISSION)
    assert graph.edge_level[permission].tolist() == [int(PermissionLevel.WRITE)]
    assert graph.neighbors("user:alice", RelationType.MEMBER_OF).tolist() == [
        graph.index_of("group:devops")
    ]


def test_snapshot_marks_missing_timestamps_as_unknown() -> None:
    graph = Neo4jSource(_legacy(), MAPPING).snapshot()
    assert graph.edge_created.tolist() == [UNKNOWN]
    assert graph.node_created.tolist() == [UNKNOWN, UNKNOWN]
    assert graph.edge_actor.tolist() == [UNKNOWN]


def test_snapshot_cutoff_drops_later_edges() -> None:
    graph = Neo4jSource(_modern(), MAPPING).snapshot(at=1_150)
    assert graph.num_edges == 1


def test_cutoff_on_an_untimestamped_graph_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot honour a cutoff"):
        Neo4jSource(_legacy(), MAPPING).snapshot(at=1_000)


def test_events_are_not_available_from_a_snapshot() -> None:
    with pytest.raises(NotImplementedError, match="change log"):
        list(Neo4jSource(_modern(), MAPPING).events())


def test_unmapped_relation_is_reported_with_its_name() -> None:
    reader = FakeReader(
        nodes=[{"id": "user:alice", "created_at": 1}],
        relationships=[
            {
                "subject": "user:alice", "relation": "FRIENDS_WITH", "object": "user:bob",
                "level": None, "created_at": 2, "actor": None,
            }
        ],
    )
    with pytest.raises(KeyError, match="FRIENDS_WITH"):
        Neo4jSource(reader, MAPPING).snapshot()


@pytest.mark.integration
def test_reads_a_live_deployment() -> None:
    """Requires a running opens3-rebac Neo4j: docker compose up -d neo4j."""
    import os

    from rga.adapters.neo4j_source import BoltReader

    reader = BoltReader(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password123"),
    )
    try:
        graph = Neo4jSource(reader, MAPPING).snapshot()
    finally:
        reader.close()

    assert graph.num_nodes >= 0
