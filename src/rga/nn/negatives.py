"""Changes that did not happen.

Five strategies, rotated across the draws so every batch contains all of them:

1. the object replaced uniformly — the easy case, teaches the gross shape;
2. the object replaced in proportion to how many rights already point at it —
   popular targets are plausible targets, so these are harder;
3. the subject replaced uniformly — the same right granted to somebody else;
4. the level moved one step along the ordinal scale — the hardest kind of near
   miss, and the reason the level is modelled as an order rather than a category;
5. a node two hops away in the undirected projection — a right that does not exist
   but plausibly could, which is exactly the boundary the model has to learn.

The feature row of the positive travels with its negatives unchanged: only the
structure is corrupted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rga.domain.graph import AccessGraph
from rga.domain.relations import LEVEL_CARRYING, PermissionLevel, RelationType

_MIN_LEVEL = int(PermissionLevel.READ)
_MAX_LEVEL = int(PermissionLevel.ADMIN)


@dataclass(frozen=True)
class CorruptedEdges:
    """Negatives and the positive each one was made from."""

    src: np.ndarray
    dst: np.ndarray
    relation: np.ndarray
    level: np.ndarray
    #: Index of the positive a negative belongs to, for gathering its feature row.
    origin: np.ndarray


def _undirected_neighbours(graph: AccessGraph) -> list[np.ndarray]:
    """Neighbour indices per node, ignoring direction and relation."""
    buckets: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for source, target in zip(graph.edge_src, graph.edge_dst, strict=True):
        buckets[int(source)].append(int(target))
        buckets[int(target)].append(int(source))
    return [np.array(sorted(set(items)), dtype=np.int64) for items in buckets]


def _two_hop(neighbours: list[np.ndarray], node: int, rng: np.random.Generator) -> int:
    """A node two hops away, or -1 when the neighbourhood is too small."""
    first = neighbours[node]
    if first.size == 0:
        return -1
    middle = int(rng.choice(first))
    second = neighbours[middle]
    if second.size == 0:
        return -1
    return int(rng.choice(second))


def sample_negatives(
    graph: AccessGraph,
    src: np.ndarray,
    dst: np.ndarray,
    relation: np.ndarray,
    level: np.ndarray,
    *,
    per_edge: int,
    rng: np.random.Generator,
) -> CorruptedEdges:
    """Draw `per_edge` corrupted variants of every positive."""
    neighbours = _undirected_neighbours(graph)
    in_degree = np.bincount(graph.edge_dst, minlength=graph.num_nodes).astype(np.float64)
    popularity = in_degree + 1.0
    popularity /= popularity.sum()

    out_src: list[int] = []
    out_dst: list[int] = []
    out_relation: list[int] = []
    out_level: list[int] = []
    out_origin: list[int] = []

    for position in range(len(src)):
        base = (
            int(src[position]),
            int(dst[position]),
            int(relation[position]),
            int(level[position]),
        )
        for draw in range(per_edge):
            new_src, new_dst, new_relation, new_level = base
            strategy = draw % 5

            if strategy == 0:
                new_dst = int(rng.integers(graph.num_nodes))
            elif strategy == 1:
                new_dst = int(rng.choice(graph.num_nodes, p=popularity))
            elif strategy == 2:
                new_src = int(rng.integers(graph.num_nodes))
            elif strategy == 3 and RelationType(new_relation) in LEVEL_CARRYING:
                if new_level >= _MAX_LEVEL:
                    step = -1
                elif new_level <= _MIN_LEVEL:
                    step = 1
                else:
                    step = int(rng.choice([-1, 1]))
                new_level = int(np.clip(new_level + step, _MIN_LEVEL, _MAX_LEVEL))
            else:
                candidate = _two_hop(neighbours, new_src, rng)
                new_dst = candidate if candidate >= 0 else int(rng.integers(graph.num_nodes))

            if (new_src, new_dst, new_relation, new_level) == base:
                # A corruption that changed nothing is not a negative. Fall back to
                # the uniform object swap, retrying until it lands elsewhere.
                while new_dst == base[1]:
                    new_dst = int(rng.integers(graph.num_nodes))

            out_src.append(new_src)
            out_dst.append(new_dst)
            out_relation.append(new_relation)
            out_level.append(new_level)
            out_origin.append(position)

    return CorruptedEdges(
        src=np.array(out_src, dtype=np.int64),
        dst=np.array(out_dst, dtype=np.int64),
        relation=np.array(out_relation, dtype=np.int64),
        level=np.array(out_level, dtype=np.int64),
        origin=np.array(out_origin, dtype=np.int64),
    )
