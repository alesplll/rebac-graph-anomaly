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
import math
from dataclasses import dataclass, field, replace

import numpy as np

from rga.domain.entities import entity_type
from rga.domain.relations import PermissionLevel, RelationType
from rga.explain.features import FeatureContribution, feature_contributions
from rga.explain.picture import render_subgraph
from rga.explain.structure import EdgeImportance, edge_importance, neighbourhood
from rga.explain.text import Observation, describe
from rga.features.edges import decode_level
from rga.features.spec import CandidateSet
from rga.util.timeutil import hour_of_day


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
    summary: tuple[Observation, ...]
    #: Who made the change, where the source records it.
    actor: str | None = None
    #: The facts the sentences are drawn from, so the page can lay them out itself.
    #: A value of None means the source does not record it — not that it is zero.
    context: dict[str, object] = field(default_factory=dict)
    features: tuple[FeatureContribution, ...] = ()
    edges: tuple[EdgeImportance, ...] = ()
    subgraph: dict[str, list] = field(default_factory=lambda: {"nodes": [], "edges": []})
    #: The neighbourhood already drawn. Rendering here rather than in the page keeps
    #: it under test; a layout that throws in a browser leaves an empty box and says
    #: nothing about why.
    picture: str = ""

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
            "actor": self.actor,
            "context": self.context,
            "summary": [{"text": item.text, "why": item.why} for item in self.summary],
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
            "picture": self.picture,
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


#: How many neighbouring relationships the drawing shows.
#:
#: The two-hop ball around a change runs to dozens of edges and draws as a thicket;
#: untangling it helps, but the real trouble is the count, not the layout. The score
#: does not lean on dozens of relationships — it leans on a handful, and the picture
#: exists to show which. Everything else stays in the contributions table, where a
#: number reads better than a line anyway.
DRAWN_EDGES = 6


def _subgraph(candidates: CandidateSet, position: int, edges, *, cap: int) -> dict[str, list]:
    """Nodes and edges around the change, each node with its distance in hops."""
    graph = candidates.graph
    assert graph is not None
    subject, _, target = candidates.keys[position]

    importance = {(item.subject, item.relation, item.object): item.importance for item in edges}
    positions = neighbourhood(graph, subject, target, cap=cap)

    def weight(edge: int) -> float:
        source = graph.node_ids[int(graph.edge_src[edge])]
        sink = graph.node_ids[int(graph.edge_dst[edge])]
        if (source, sink) == (subject, target):
            # The change itself is always drawn: it is what the card is about.
            return float("inf")
        relation = RelationType(int(graph.edge_rel[edge])).name
        return abs(float(importance.get((source, relation, sink), 0.0)))

    # The change and the contributors are picked separately. Taking the top seven of
    # everything would silently drop a contributor whenever the change is already in
    # the graph, and admit a seventh whenever it is not.
    ordered = sorted(positions.tolist(), key=weight, reverse=True)
    itself = [edge for edge in ordered if weight(edge) == float("inf")]
    chosen = itself + [edge for edge in ordered if edge not in itself][:DRAWN_EDGES]

    hops = {name: 0 for name in (subject, target) if name in graph.node_index}
    drawn = []
    for edge in chosen:
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


def _fact(candidates: CandidateSet, position: int, name: str) -> float | None:
    """A feature's value, or None when the source could not supply it."""
    if not bool(candidates.matrix.observed(name)[position]):
        return None
    return float(candidates.matrix.column(name)[position])


def _context(candidates: CandidateSet, position: int) -> dict[str, object]:
    """The facts behind the sentences, decoded back into readable quantities."""
    jump = _fact(candidates, position, "level_jump")
    common = _fact(candidates, position, "common_neighbours")
    hops = _fact(candidates, position, "path_hops")
    unreachable = _fact(candidates, position, "path_unreachable")

    def flag(name: str) -> bool | None:
        value = _fact(candidates, position, name)
        return None if value is None else bool(value >= 0.5)

    return {
        "actor_is_subject": flag("actor_is_subject"),
        "off_hours": flag("is_off_hours"),
        "weekend": flag("is_weekend"),
        "bypasses_bucket": flag("bypasses_bucket"),
        "level_jump": None if jump is None else round(jump),
        # Both are stored through log1p, so they come back through expm1.
        "common_neighbours": None if common is None else round(math.expm1(common)),
        "path_hops": None if hops is None or unreachable == 1.0 else round(hops),
        "hour": hour_of_day(int(candidates.ts[position])),
    }


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
    ordinal = decode_level(candidates.matrix.column("level_ordinal")[position])
    edges = edge_importance(scorer, candidates, position, cap=cap) if explain else ()
    subgraph = (
        _subgraph(candidates, position, edges, cap=cap)
        if explain
        else {"nodes": [], "edges": []}
    )
    drawn = Incident(
        id=incident_id(candidates.keys[position], int(candidates.ts[position])),
        score=score,
        rank=rank,
        ts=int(candidates.ts[position]),
        subject=subject,
        relation=RelationType(relation).name,
        object=target,
        level=PermissionLevel(max(ordinal, 0)).name.lower(),
        actor=candidates.actors[position] if candidates.actors else None,
        context=_context(candidates, position),
        summary=describe(candidates, position),
        features=feature_contributions(scorer, candidates, position) if explain else (),
        edges=edges,
        subgraph=subgraph,
    )
    return replace(drawn, picture=render_subgraph(drawn.as_dict()) if explain else "")
