"""Node attributes computed once on a snapshot.

Community membership and triangle counts cannot be maintained incrementally at a
sensible cost, and they do not need to be: an organization's community structure
does not turn over inside a two-week evaluation window. Both are computed on the
training snapshot and read as stable context.

Communities come from label propagation on the graph itself. Taking them from the
generator's departments instead would hand the model the answer to "does this
cross an organizational boundary", which is precisely what it is supposed to
infer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rga.domain.graph import AccessGraph


@dataclass(frozen=True)
class StaticAttributes:
    """Per-node attributes of one snapshot, keyed by entity id."""

    community: dict[str, int]
    triangles: dict[str, int]
    clustering: dict[str, float]
    same_community_share: dict[str, float]


def _undirected_adjacency(graph: AccessGraph) -> list[set[int]]:
    """Neighbour sets of the undirected projection, ignoring relation type."""
    adjacency: list[set[int]] = [set() for _ in range(graph.num_nodes)]
    for source, destination in zip(graph.edge_src, graph.edge_dst, strict=True):
        left, right = int(source), int(destination)
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _label_propagation(
    adjacency: list[set[int]], rng: np.random.Generator, rounds: int
) -> np.ndarray:
    """Assign each node the label most common among its neighbours.

    Visiting order is shuffled every round, which is what lets labels spread and
    is why the result depends on the seed.
    """
    labels = np.arange(len(adjacency), dtype=np.int64)
    order = np.arange(len(adjacency))

    for _ in range(rounds):
        rng.shuffle(order)
        changed = False
        for node in order:
            neighbours = adjacency[int(node)]
            if not neighbours:
                continue
            counts: dict[int, int] = {}
            for neighbour in neighbours:
                label = int(labels[neighbour])
                counts[label] = counts.get(label, 0) + 1
            # Ties broken by the smaller label, so the pass is deterministic
            # given the visiting order.
            best = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            if best != labels[int(node)]:
                labels[int(node)] = best
                changed = True
        if not changed:
            break

    return labels


def compute_static_attributes(
    graph: AccessGraph, rng: np.random.Generator, *, rounds: int = 20
) -> StaticAttributes:
    """Compute community, triangle and clustering attributes for every node."""
    adjacency = _undirected_adjacency(graph)
    labels = _label_propagation(adjacency, rng, rounds)

    community: dict[str, int] = {}
    triangles: dict[str, int] = {}
    clustering: dict[str, float] = {}
    same_share: dict[str, float] = {}

    for index, node_id in enumerate(graph.node_ids):
        neighbours = adjacency[index]
        degree = len(neighbours)

        closed = 0
        for neighbour in neighbours:
            closed += len(adjacency[neighbour] & neighbours)
        closed //= 2

        community[node_id] = int(labels[index])
        triangles[node_id] = closed
        clustering[node_id] = (2.0 * closed) / (degree * (degree - 1)) if degree > 1 else 0.0
        same_share[node_id] = (
            sum(1 for neighbour in neighbours if labels[neighbour] == labels[index]) / degree
            if degree
            else 0.0
        )

    return StaticAttributes(
        community=community,
        triangles=triangles,
        clustering=clustering,
        same_community_share=same_share,
    )
