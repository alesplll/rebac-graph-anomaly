"""One change, assembled into what an analyst is shown.

The identifier is derived from the change itself rather than from its position in the
queue: a refresh reorders the queue, and a link an analyst kept must still point at
the same change afterwards.

`explain=False` builds the cheap form used for list rows. The expensive part is edge
masking, which re-encodes the graph once per neighbouring edge — fine for the one card
on screen, not for fifty rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from rga.domain.entities import entity_type
from rga.domain.relations import PermissionLevel, RelationType
from rga.explain.features import FeatureContribution, feature_contributions
from rga.explain.structure import EdgeImportance, edge_importance, neighbourhood
from rga.explain.text import describe
from rga.features.spec import CandidateSet


def incident_id(key: tuple[str, int, str], ts: int) -> str:
    """A stable name for one change, independent of its rank."""
    subject, relation, target = key
    raw = f"{subject}|{relation}|{target}|{ts}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class Incident:
    """A ranked change with its grounds."""

    id: str
    score: float
    rank: int
    ts: int
    subject: str
    relation: str
    object: str
    level: str
    summary: tuple[str, ...]
    features: tuple[FeatureContribution, ...] = ()
    edges: tuple[EdgeImportance, ...] = ()
    subgraph: dict[str, list] = field(default_factory=lambda: {"nodes": [], "edges": []})

    def as_dict(self) -> dict[str, object]:
        """The shape the API returns."""
        return {
            "id": self.id,
            "score": self.score,
            "rank": self.rank,
            "ts": self.ts,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "level": self.level,
            "summary": list(self.summary),
            "features": [
                {
                    "name": item.name,
                    "group": str(item.group),
                    "value": item.value,
                    "contribution": item.contribution,
                    "observed": item.observed,
                }
                for item in self.features
            ],
            "edges": [
                {
                    "subject": item.subject,
                    "relation": item.relation,
                    "object": item.object,
                    "importance": item.importance,
                }
                for item in self.edges
            ],
            "subgraph": self.subgraph,
        }


def _distances(graph, seeds: tuple[str, ...], *, depth: int) -> dict[str, int]:
    """Hop distance from `seeds` to every node within `depth`, over the whole graph."""
    hops = {name: 0 for name in seeds if name in graph.node_index}
    frontier = {graph.node_index[name] for name in hops}

    for step in range(1, depth + 1):
        if not frontier:
            break
        touching = np.flatnonzero(
            np.isin(graph.edge_src, list(frontier)) | np.isin(graph.edge_dst, list(frontier))
        )
        reached = set(graph.edge_src[touching].tolist()) | set(graph.edge_dst[touching].tolist())
        frontier = set()
        for index in reached:
            name = graph.node_ids[int(index)]
            if name not in hops:
                hops[name] = step
                frontier.add(int(index))
    return hops


def _subgraph(candidates: CandidateSet, position: int, edges, *, cap: int) -> dict[str, list]:
    """Nodes and edges around the change, each node with its distance in hops."""
    graph = candidates.graph
    assert graph is not None
    subject, _, target = candidates.keys[position]

    importance = {(item.subject, item.relation, item.object): item.importance for item in edges}
    positions = neighbourhood(graph, subject, target, cap=cap)

    hops = {name: 0 for name in (subject, target) if name in graph.node_index}
    drawn = []
    for edge in positions:
        source = graph.node_ids[int(graph.edge_src[edge])]
        sink = graph.node_ids[int(graph.edge_dst[edge])]
        relation = RelationType(int(graph.edge_rel[edge])).name
        drawn.append(
            {
                "subject": source,
                "relation": relation,
                "object": sink,
                "level": PermissionLevel(int(graph.edge_level[edge])).name.lower(),
                "importance": float(importance.get((source, relation, sink), 0.0)),
            }
        )

    # Distances come from the whole graph, not from what ended up drawn. The cap on
    # edges can cut a short connection, and a walk over the remainder would report a
    # node as five hops away when the graph itself puts it at two.
    distances = _distances(graph, tuple(hops), depth=2)

    # Only the nodes that an edge actually touches are drawn. The two-hop ball around
    # a busy user runs to hundreds of nodes; the picture shows the connections the
    # score leaned on, not everything within reach of them.
    shown = {edge["subject"] for edge in drawn} | {edge["object"] for edge in drawn}
    shown |= set(hops)

    nodes = [
        {
            "id": name,
            "type": entity_type(name).name.lower(),
            "hops": distances.get(name, 2),
        }
        for name in sorted(shown, key=lambda name: (distances.get(name, 2), name))
    ]
    return {"nodes": nodes, "edges": drawn}


def build_incident(
    scorer,
    candidates: CandidateSet,
    position: int,
    *,
    score: float,
    rank: int,
    explain: bool = True,
    cap: int = 60,
) -> Incident:
    """Assemble one change, with or without the expensive attributions."""
    subject, relation, target = candidates.keys[position]
    ordinal = round(float(candidates.matrix.column("level_ordinal")[position]))
    edges = edge_importance(scorer, candidates, position, cap=cap) if explain else ()
    return Incident(
        id=incident_id(candidates.keys[position], int(candidates.ts[position])),
        score=score,
        rank=rank,
        ts=int(candidates.ts[position]),
        subject=subject,
        relation=RelationType(relation).name,
        object=target,
        level=PermissionLevel(max(ordinal, 0)).name.lower(),
        summary=describe(candidates, position),
        features=feature_contributions(scorer, candidates, position) if explain else (),
        edges=edges,
        subgraph=(
            _subgraph(candidates, position, edges, cap=cap)
            if explain
            else {"nodes": [], "edges": []}
        ),
    )
