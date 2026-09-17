"""Attribution over the graph around a change.

Each edge within two hops is removed in turn and the change re-scored; the drop is
that edge's importance. Two hops because that is the radius the encoder actually sees
through its layers, and because it covers the chain the authorization engine itself
walks: a user, the group it belongs to, the bucket that group can reach.

The work is bounded by `cap`: each masked edge costs a full re-encode, and an analyst
waiting on a card will not thank us for a hundred of them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rga.domain.graph import AccessGraph
from rga.domain.relations import RelationType
from rga.features.spec import CandidateSet


@dataclass(frozen=True)
class EdgeImportance:
    """One relationship's share of the score."""

    subject: str
    relation: str
    object: str
    importance: float


def neighbourhood(
    graph: AccessGraph, subject: str, target: str, *, hops: int = 2, cap: int = 60
) -> np.ndarray:
    """Positions of the edges within `hops` of either endpoint, at most `cap`."""
    reached = {graph.node_index[name] for name in (subject, target) if name in graph.node_index}
    if not reached:
        return np.zeros(0, dtype=np.int64)

    frontier = set(reached)
    for _ in range(hops):
        touching = np.flatnonzero(
            np.isin(graph.edge_src, list(frontier)) | np.isin(graph.edge_dst, list(frontier))
        )
        frontier = set(graph.edge_src[touching].tolist()) | set(graph.edge_dst[touching].tolist())
        reached |= frontier

    positions = np.flatnonzero(
        np.isin(graph.edge_src, list(reached)) & np.isin(graph.edge_dst, list(reached))
    )
    return positions[:cap]


def _without(graph: AccessGraph, position: int) -> AccessGraph:
    """The same graph minus one edge. Node indices are preserved."""
    keep = np.ones(graph.num_edges, dtype=bool)
    keep[position] = False
    return replace(
        graph,
        edge_src=graph.edge_src[keep],
        edge_dst=graph.edge_dst[keep],
        edge_rel=graph.edge_rel[keep],
        edge_level=graph.edge_level[keep],
        edge_created=graph.edge_created[keep],
        edge_actor=graph.edge_actor[keep],
    )


def edge_importance(
    scorer, candidates: CandidateSet, position: int, *, hops: int = 2, cap: int = 60
) -> tuple[EdgeImportance, ...]:
    """How much each nearby relationship holds this candidate's score up."""
    graph = candidates.graph
    if graph is None:
        raise ValueError("the candidate set carries no graph; structure cannot be explained")

    single = candidates.row(position)
    # Margins rather than scores where the scorer offers them: a saturated sigmoid
    # moves by nothing when an edge is removed, while the logit behind it moves.
    measure = getattr(scorer, "margins", scorer.score)
    baseline = float(measure(single)[0])
    subject, _, target = candidates.keys[position]

    found: list[EdgeImportance] = []
    for edge in neighbourhood(graph, subject, target, hops=hops, cap=cap):
        masked = replace(single, graph=_without(graph, int(edge)))
        without = float(measure(masked)[0])
        found.append(
            EdgeImportance(
                subject=graph.node_ids[int(graph.edge_src[edge])],
                relation=RelationType(int(graph.edge_rel[edge])).name,
                object=graph.node_ids[int(graph.edge_dst[edge])],
                importance=baseline - without,
            )
        )

    found.sort(key=lambda item: abs(item.importance), reverse=True)
    return tuple(found)
