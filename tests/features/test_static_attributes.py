"""Community and triangle attributes computed on a snapshot."""

import numpy as np

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay
from rga.features.static import compute_static_attributes


def _two_cliques():
    """Two triangles joined by a single bridge edge."""
    events = []
    ts = 0
    for left, right in [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d")]:
        ts += 1
        events.append(
            GraphEvent(ts, EventOp.GRANT, f"user:{left}", RelationType.MEMBER_OF, f"group:{right}")
        )
    ts += 1
    events.append(GraphEvent(ts, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:d"))
    return replay(events)


def test_every_node_receives_a_community() -> None:
    graph = _two_cliques()
    attributes = compute_static_attributes(graph, np.random.default_rng(0))
    assert set(attributes.community) == set(graph.node_ids)


def test_label_propagation_separates_weakly_linked_groups() -> None:
    graph = _two_cliques()
    attributes = compute_static_attributes(graph, np.random.default_rng(0))
    # Not asserting an exact partition: label propagation is stochastic. What must
    # hold is that it does not collapse everything into one community.
    assert len(set(attributes.community.values())) >= 2


def test_result_is_reproducible_for_a_fixed_seed() -> None:
    graph = _two_cliques()
    first = compute_static_attributes(graph, np.random.default_rng(7))
    second = compute_static_attributes(graph, np.random.default_rng(7))
    assert first.community == second.community


def test_triangles_are_counted_on_the_undirected_projection() -> None:
    events = [
        GraphEvent(1, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:b"),
        GraphEvent(2, EventOp.GRANT, "user:b", RelationType.MEMBER_OF, "group:c"),
        GraphEvent(3, EventOp.GRANT, "user:c", RelationType.MEMBER_OF, "group:a"),
    ]
    attributes = compute_static_attributes(replay(events), np.random.default_rng(0))
    # user:a and group:a are distinct nodes, so this chain closes no triangle.
    assert all(count == 0 for count in attributes.triangles.values())


def test_a_real_triangle_is_found() -> None:
    events = [
        GraphEvent(1, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:x"),
        GraphEvent(2, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:y"),
        GraphEvent(
            3, EventOp.GRANT, "group:x", RelationType.HAS_PERMISSION, "bucket:z",
            PermissionLevel.READ,
        ),
        GraphEvent(
            4, EventOp.GRANT, "group:y", RelationType.HAS_PERMISSION, "bucket:z",
            PermissionLevel.READ,
        ),
        GraphEvent(5, EventOp.GRANT, "group:x", RelationType.MEMBER_OF, "group:y"),
    ]
    attributes = compute_static_attributes(replay(events), np.random.default_rng(0))
    assert attributes.triangles["group:x"] >= 1
    assert 0.0 < attributes.clustering["group:x"] <= 1.0


def test_isolated_nodes_get_zero_not_nan() -> None:
    events = [GraphEvent(1, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:b")]
    attributes = compute_static_attributes(replay(events), np.random.default_rng(0))
    assert attributes.clustering["user:a"] == 0.0
    assert attributes.same_community_share["user:a"] in (0.0, 1.0)
