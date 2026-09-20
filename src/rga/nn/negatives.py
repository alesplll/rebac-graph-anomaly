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
from rga.domain.relations import LEVEL_CARRYING, PermissionLevel

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


def undirected_csr(graph: AccessGraph) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour lists of the undirected projection, as (indptr, indices).

    Parallel edges are kept rather than deduplicated: building a set per node was a
    second Python loop over the graph, and keeping duplicates only reweights the
    draw towards heavily connected pairs. The set of reachable nodes is unchanged.
    """
    tail = np.concatenate([graph.edge_src, graph.edge_dst]).astype(np.int64)
    head = np.concatenate([graph.edge_dst, graph.edge_src]).astype(np.int64)
    order = np.argsort(tail, kind="stable")
    indices = head[order]
    indptr = np.zeros(graph.num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(tail, minlength=graph.num_nodes), out=indptr[1:])
    return indptr, indices


def _random_neighbour(
    indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One neighbour of every given node, or -1 where the node has none."""
    if indices.size == 0:
        return np.full(nodes.shape, -1, dtype=np.int64)
    degree = indptr[nodes + 1] - indptr[nodes]
    offset = np.floor(rng.random(nodes.size) * np.maximum(degree, 1)).astype(np.int64)
    picked = indices[np.clip(indptr[nodes] + offset, 0, indices.size - 1)]
    return np.where(degree > 0, picked, -1)


def _two_hop(
    indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A node two undirected steps away, or -1 where the walk died out."""
    middle = _random_neighbour(indptr, indices, nodes, rng)
    second = _random_neighbour(indptr, indices, np.where(middle >= 0, middle, 0), rng)
    return np.where(middle >= 0, second, -1)


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
    """Draw `per_edge` corrupted variants of every positive.

    Every draw is made for the whole batch at once. The strategy of a draw is still
    its index modulo five, so a batch carries all five kinds in the same proportion
    as the per-edge loop this replaced; what changed is the order in which random
    numbers are consumed, and therefore which edges come out for a given seed.
    """
    count = len(src)
    origin = np.repeat(np.arange(count, dtype=np.int64), per_edge)
    strategy = np.tile(np.arange(per_edge, dtype=np.int64) % 5, count)

    base_src = np.asarray(src, dtype=np.int64)[origin]
    base_dst = np.asarray(dst, dtype=np.int64)[origin]
    base_relation = np.asarray(relation, dtype=np.int64)[origin]
    base_level = np.asarray(level, dtype=np.int64)[origin]

    out_src, out_dst = base_src.copy(), base_dst.copy()
    out_relation, out_level = base_relation.copy(), base_level.copy()

    in_degree = np.bincount(graph.edge_dst, minlength=graph.num_nodes).astype(np.float64)
    popularity = in_degree + 1.0
    popularity /= popularity.sum()

    uniform = strategy == 0
    out_dst[uniform] = rng.integers(graph.num_nodes, size=int(uniform.sum()))

    popular = strategy == 1
    out_dst[popular] = rng.choice(graph.num_nodes, size=int(popular.sum()), p=popularity)

    swapped = strategy == 2
    out_src[swapped] = rng.integers(graph.num_nodes, size=int(swapped.sum()))

    carries = np.isin(base_relation, [int(kind) for kind in LEVEL_CARRYING])
    shifted = (strategy == 3) & carries
    if shifted.any():
        here = out_level[shifted]
        step = np.where(rng.random(int(shifted.sum())) < 0.5, -1, 1)
        step = np.where(here >= _MAX_LEVEL, -1, step)
        step = np.where(here <= _MIN_LEVEL, 1, step)
        out_level[shifted] = np.clip(here + step, _MIN_LEVEL, _MAX_LEVEL)

    # A relation that carries no level has nothing for strategy four to move, so it
    # falls through to the walk, exactly as the per-edge loop did.
    walked = (strategy == 4) | ((strategy == 3) & ~carries)
    if walked.any():
        indptr, indices = undirected_csr(graph)
        reached = _two_hop(indptr, indices, out_src[walked], rng)
        fallback = rng.integers(graph.num_nodes, size=int(walked.sum()))
        out_dst[walked] = np.where(reached >= 0, reached, fallback)

    unchanged = (
        (out_src == base_src)
        & (out_dst == base_dst)
        & (out_relation == base_relation)
        & (out_level == base_level)
    )
    if unchanged.any() and graph.num_nodes > 1:
        # A corruption that changed nothing is not a negative. Draw among the nodes
        # other than the original object: the index is taken over `num_nodes - 1`
        # and stepped over the one that must not come out, which is the loop-free
        # form of the rejection the per-edge version used.
        avoid = base_dst[unchanged]
        drawn = rng.integers(graph.num_nodes - 1, size=int(unchanged.sum()))
        out_dst[unchanged] = drawn + (drawn >= avoid)

    return CorruptedEdges(
        src=out_src, dst=out_dst, relation=out_relation, level=out_level, origin=origin
    )
