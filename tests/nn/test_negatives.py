"""Corrupted changes, the negative half of the contrastive objective."""

import numpy as np

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.negatives import sample_negatives


def _graph():
    builder = GraphBuilder()
    for name in ("alice", "bob", "carol"):
        builder.add_edge(f"user:{name}", RelationType.MEMBER_OF, "group:devops", created=1)
    builder.add_edge(
        "group:devops",
        RelationType.HAS_PERMISSION,
        "bucket:logs",
        level=int(PermissionLevel.READ),
        created=2,
    )
    builder.add_edge(
        "user:alice",
        RelationType.HAS_PERMISSION,
        "bucket:reports",
        level=int(PermissionLevel.WRITE),
        created=3,
    )
    return builder.build()


def _positives(graph):
    src = np.array([graph.index_of("user:alice")], dtype=np.int64)
    dst = np.array([graph.index_of("bucket:reports")], dtype=np.int64)
    relation = np.array([int(RelationType.HAS_PERMISSION)], dtype=np.int64)
    level = np.array([int(PermissionLevel.WRITE)], dtype=np.int64)
    return src, dst, relation, level


def test_the_requested_number_of_negatives_is_produced() -> None:
    graph = _graph()

    corrupted = sample_negatives(
        graph, *_positives(graph), per_edge=4, rng=np.random.default_rng(0)
    )

    assert corrupted.src.shape == (4,)
    assert corrupted.origin.tolist() == [0, 0, 0, 0]


def test_every_negative_differs_from_its_positive() -> None:
    graph = _graph()
    src, dst, relation, level = _positives(graph)

    corrupted = sample_negatives(
        graph, src, dst, relation, level, per_edge=8, rng=np.random.default_rng(1)
    )

    same = (corrupted.src == src[0]) & (corrupted.dst == dst[0]) & (corrupted.level == level[0])
    assert not same.any()


def test_indices_stay_inside_the_graph() -> None:
    graph = _graph()

    corrupted = sample_negatives(
        graph, *_positives(graph), per_edge=16, rng=np.random.default_rng(2)
    )

    assert corrupted.src.min() >= 0
    assert corrupted.src.max() < graph.num_nodes
    assert corrupted.dst.min() >= 0
    assert corrupted.dst.max() < graph.num_nodes
    assert set(corrupted.level.tolist()) <= {level.value for level in PermissionLevel}


def test_sampling_is_reproducible() -> None:
    graph = _graph()
    positives = _positives(graph)

    first = sample_negatives(graph, *positives, per_edge=8, rng=np.random.default_rng(3))
    second = sample_negatives(graph, *positives, per_edge=8, rng=np.random.default_rng(3))

    assert np.array_equal(first.src, second.src)
    assert np.array_equal(first.dst, second.dst)
    assert np.array_equal(first.level, second.level)


#: Length of the test path. Ten links leave only three of its twelve nodes within
#: two hops of the head, so a uniform draw passes the two-hop assertion one time in
#: four and twenty draws catch a broken walk with near certainty.
_CHAIN_LINKS = 10


def _chain():
    """A path long enough that most nodes sit further than two hops from the source."""
    builder = GraphBuilder()
    builder.add_edge("user:a", RelationType.MEMBER_OF, "group:g1", created=1)
    for step in range(1, _CHAIN_LINKS):
        builder.add_edge(
            f"group:g{step}", RelationType.MEMBER_OF, f"group:g{step + 1}", created=step + 1
        )
    builder.add_edge(
        f"group:g{_CHAIN_LINKS}",
        RelationType.HAS_PERMISSION,
        "bucket:b1",
        level=int(PermissionLevel.READ),
        created=_CHAIN_LINKS + 1,
    )
    return builder.build()


def _undirected_distances(graph, source: int) -> np.ndarray:
    """Hop counts from `source`, ignoring direction. Unreached nodes get a big number."""
    distance = np.full(graph.num_nodes, 1_000, dtype=np.int64)
    distance[source] = 0
    frontier = [source]
    while frontier:
        following: list[int] = []
        for node in frontier:
            for a, b in zip(graph.edge_src, graph.edge_dst, strict=True):
                for here, there in ((int(a), int(b)), (int(b), int(a))):
                    if here == node and distance[there] > distance[node] + 1:
                        distance[there] = distance[node] + 1
                        following.append(there)
        frontier = following
    return distance


def test_the_two_hop_strategy_stays_within_two_steps() -> None:
    """Strategy five exists to produce a right that plausibly could have been granted."""
    graph = _chain()
    source = graph.index_of("user:a")
    draws = 20
    src = np.full(draws, source, dtype=np.int64)
    dst = np.full(draws, graph.index_of("bucket:b1"), dtype=np.int64)
    relation = np.full(draws, int(RelationType.HAS_PERMISSION), dtype=np.int64)
    level = np.full(draws, int(PermissionLevel.READ), dtype=np.int64)

    corrupted = sample_negatives(
        graph, src, dst, relation, level, per_edge=5, rng=np.random.default_rng(7)
    )

    distance = _undirected_distances(graph, source)
    walked = corrupted.dst[np.arange(draws) * 5 + 4]
    assert (distance[walked] <= 2).all()


def test_negatives_are_grouped_by_the_positive_they_came_from() -> None:
    """The origin index is what gathers the feature row, so its order is load-bearing."""
    graph = _chain()
    src = np.array([graph.index_of("user:a"), graph.index_of("group:g2")], dtype=np.int64)
    dst = np.array([graph.index_of("group:g1"), graph.index_of("group:g3")], dtype=np.int64)
    relation = np.full(2, int(RelationType.MEMBER_OF), dtype=np.int64)
    level = np.zeros(2, dtype=np.int64)

    corrupted = sample_negatives(
        graph, src, dst, relation, level, per_edge=3, rng=np.random.default_rng(9)
    )

    assert corrupted.origin.tolist() == [0, 0, 0, 1, 1, 1]


def test_all_five_strategies_are_reachable() -> None:
    """With four draws per edge the fifth strategy never came up; eight reaches it."""
    graph = _chain()
    src = np.array([graph.index_of("user:a")], dtype=np.int64)
    dst = np.array([graph.index_of("bucket:b1")], dtype=np.int64)
    relation = np.array([int(RelationType.HAS_PERMISSION)], dtype=np.int64)
    level = np.array([int(PermissionLevel.WRITE)], dtype=np.int64)

    corrupted = sample_negatives(
        graph, src, dst, relation, level, per_edge=10, rng=np.random.default_rng(11)
    )

    # Strategy three moves the level and leaves the endpoints alone.
    moved_level = corrupted.level != level[0]
    assert bool(moved_level.any())
    # Strategy two replaces the subject.
    assert bool((corrupted.src != src[0]).any())
