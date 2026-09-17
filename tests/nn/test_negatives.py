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
