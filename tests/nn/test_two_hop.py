"""The diagnostic behind module 3, section 4.3."""

import numpy as np

from rga.domain.graph import GraphBuilder
from rga.domain.relations import RelationType
from rga.nn.two_hop import BEYOND, hop_distribution


def _path(links: int):
    """user:a joined to group:g1 joined to group:g2 and so on."""
    builder = GraphBuilder()
    builder.add_edge("user:a", RelationType.MEMBER_OF, "group:g1", created=1)
    for step in range(1, links):
        builder.add_edge(
            f"group:g{step}", RelationType.MEMBER_OF, f"group:g{step + 1}", created=step + 1
        )
    return builder.build()


def test_the_four_distances_are_told_apart() -> None:
    graph = _path(4)
    source = graph.index_of("user:a")
    src = np.full(4, source, dtype=np.int64)
    dst = np.array(
        [source, graph.index_of("group:g1"), graph.index_of("group:g2"), graph.index_of("group:g4")],
        dtype=np.int64,
    )

    assert hop_distribution(graph, src, dst).tolist() == [0, 1, 2, BEYOND]


def test_an_unknown_endpoint_counts_as_beyond() -> None:
    """A candidate whose subject the graph has never seen has no distance at all."""
    graph = _path(2)
    src = np.array([-1, graph.index_of("user:a")], dtype=np.int64)
    dst = np.array([graph.index_of("group:g1"), -1], dtype=np.int64)

    assert hop_distribution(graph, src, dst).tolist() == [BEYOND, BEYOND]


def test_direction_is_ignored() -> None:
    """The walk that draws hard negatives ignores direction, so this must too."""
    graph = _path(2)
    src = np.array([graph.index_of("group:g1")], dtype=np.int64)
    dst = np.array([graph.index_of("user:a")], dtype=np.int64)

    assert hop_distribution(graph, src, dst).tolist() == [1]
