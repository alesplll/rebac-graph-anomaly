"""Graph assembly and adjacency."""

import numpy as np
import pytest

from rga.domain.entities import EntityType
from rga.domain.events import EventOp, GraphEvent
from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType


def _build(*events: GraphEvent):
    builder = GraphBuilder()
    for event in events:
        builder.apply(event)
    return builder.build()


def test_grant_creates_both_endpoints_and_the_edge() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops")
    )
    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    assert graph.node_type[graph.index_of("user:alice")] == EntityType.USER
    assert graph.node_type[graph.index_of("group:devops")] == EntityType.GROUP


def test_node_creation_time_is_when_it_was_first_seen() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(
            500,
            EventOp.GRANT,
            "user:alice",
            RelationType.HAS_PERMISSION,
            "bucket:photos",
            PermissionLevel.READ,
        ),
    )
    assert graph.node_created[graph.index_of("user:alice")] == 100
    assert graph.node_created[graph.index_of("bucket:photos")] == 500


def test_revoke_removes_the_edge_but_keeps_the_nodes() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(200, EventOp.REVOKE, "user:alice", RelationType.MEMBER_OF, "group:devops"),
    )
    assert graph.num_edges == 0
    assert graph.num_nodes == 2


def test_regrant_overwrites_level_and_time_without_duplicating() -> None:
    graph = _build(
        GraphEvent(
            100,
            EventOp.GRANT,
            "user:alice",
            RelationType.HAS_PERMISSION,
            "bucket:photos",
            PermissionLevel.READ,
        ),
        GraphEvent(
            200,
            EventOp.GRANT,
            "user:alice",
            RelationType.HAS_PERMISSION,
            "bucket:photos",
            PermissionLevel.ADMIN,
        ),
    )
    assert graph.num_edges == 1
    assert graph.edge_level[0] == PermissionLevel.ADMIN
    assert graph.edge_created[0] == 200


def test_unknown_actor_is_encoded_as_minus_one() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops")
    )
    assert graph.edge_actor[0] == -1


def test_known_actor_points_at_a_node() -> None:
    graph = _build(
        GraphEvent(
            100,
            EventOp.GRANT,
            "user:alice",
            RelationType.MEMBER_OF,
            "group:devops",
            actor="user:root",
        )
    )
    assert graph.edge_actor[0] == graph.index_of("user:root")


def test_out_of_order_events_are_rejected() -> None:
    builder = GraphBuilder()
    builder.apply(GraphEvent(200, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:g"))
    with pytest.raises(ValueError, match="non-decreasing"):
        builder.apply(GraphEvent(100, EventOp.GRANT, "user:b", RelationType.MEMBER_OF, "group:g"))


def test_neighbors_respects_relation_and_direction() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(101, EventOp.GRANT, "user:bob", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(
            102,
            EventOp.GRANT,
            "user:alice",
            RelationType.HAS_PERMISSION,
            "bucket:photos",
            PermissionLevel.READ,
        ),
    )
    outgoing = graph.neighbors("user:alice", RelationType.MEMBER_OF)
    assert outgoing.tolist() == [graph.index_of("group:devops")]

    incoming = graph.neighbors("group:devops", RelationType.MEMBER_OF, incoming=True)
    assert sorted(incoming.tolist()) == sorted(
        [graph.index_of("user:alice"), graph.index_of("user:bob")]
    )

    assert graph.neighbors("user:bob", RelationType.HAS_PERMISSION).size == 0


def test_neighbors_of_an_empty_graph_is_empty() -> None:
    graph = GraphBuilder().build()
    assert graph.num_nodes == 0
    assert graph.num_edges == 0
    assert isinstance(graph.edge_src, np.ndarray)


def test_index_of_reports_missing_nodes_clearly() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops")
    )
    with pytest.raises(KeyError, match="node not in graph"):
        graph.index_of("user:ghost")
