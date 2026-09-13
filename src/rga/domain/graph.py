"""The access graph in array form.

Stored as parallel arrays rather than objects because every consumer — feature
extraction, the network, the baselines — works on the whole graph at once.
Adjacency is materialised lazily in compressed form, once per relation and
direction, because feature extraction queries neighbourhoods per node and a mask
scan per query would be quadratic.

Unknown values are -1 everywhere: unknown is not zero and not absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from rga.domain.entities import entity_type
from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import RelationType

UNKNOWN = -1


@dataclass(frozen=True, eq=False)
class AccessGraph:
    """An immutable snapshot of the access graph."""

    node_ids: tuple[str, ...]
    node_index: dict[str, int]
    node_type: np.ndarray
    node_created: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_rel: np.ndarray
    edge_level: np.ndarray
    edge_created: np.ndarray
    edge_actor: np.ndarray

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.shape[0])

    def index_of(self, entity_id: str) -> int:
        """Row index of a node, raising if it is not in the graph."""
        try:
            return self.node_index[entity_id]
        except KeyError:
            raise KeyError(f"node not in graph: {entity_id!r}") from None

    def neighbors(
        self, entity_id: str, relation: RelationType, *, incoming: bool = False
    ) -> np.ndarray:
        """Node indices reachable from `entity_id` over one edge of `relation`."""
        node = self.index_of(entity_id)
        indptr, indices = self._adjacency[(int(relation), incoming)]
        return indices[indptr[node] : indptr[node + 1]]

    @cached_property
    def _adjacency(self) -> dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]]:
        """(relation, incoming) -> (indptr of length N+1, neighbour indices)."""
        table: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]] = {}
        for relation in RelationType:
            mask = self.edge_rel == int(relation)
            src = self.edge_src[mask]
            dst = self.edge_dst[mask]
            for incoming in (False, True):
                anchor, other = (dst, src) if incoming else (src, dst)
                order = np.argsort(anchor, kind="stable")
                counts = np.bincount(anchor, minlength=self.num_nodes)
                indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
                np.cumsum(counts, out=indptr[1:])
                table[(int(relation), incoming)] = (
                    indptr,
                    other[order].astype(np.int32, copy=False),
                )
        return table


@dataclass
class _EdgeRecord:
    level: int
    created: int
    actor: int


@dataclass
class GraphBuilder:
    """Applies events in time order and freezes the result into an AccessGraph."""

    _node_index: dict[str, int] = field(default_factory=dict)
    _node_ids: list[str] = field(default_factory=list)
    _node_type: list[int] = field(default_factory=list)
    _node_created: list[int] = field(default_factory=list)
    _edges: dict[tuple[int, int, int], _EdgeRecord] = field(default_factory=dict)
    _last_ts: int | None = None

    def apply(self, event: GraphEvent) -> None:
        """Apply one event. Events must arrive in non-decreasing time order."""
        if self._last_ts is not None and event.ts < self._last_ts:
            raise ValueError(
                f"events must arrive in non-decreasing ts order: {event.ts} after {self._last_ts}"
            )
        self._last_ts = event.ts

        if event.op is EventOp.GRANT:
            self.add_edge(
                event.subject,
                event.relation,
                event.object,
                level=int(event.level),
                created=event.ts,
                actor=event.actor,
            )
        else:
            self.register_node(event.subject, event.ts)
            self.register_node(event.object, event.ts)
            self.remove_edge(event.subject, event.relation, event.object)

    def register_node(self, entity_id: str, created: int = UNKNOWN) -> int:
        """Index of a node, registering it on first sight with its creation time."""
        index = self._node_index.get(entity_id)
        if index is None:
            index = len(self._node_ids)
            self._node_index[entity_id] = index
            self._node_ids.append(entity_id)
            self._node_type.append(int(entity_type(entity_id)))
            self._node_created.append(created)
        return index

    def add_edge(
        self,
        subject: str,
        relation: RelationType,
        target: str,
        *,
        level: int = 0,
        created: int = UNKNOWN,
        actor: str | None = None,
    ) -> None:
        """Add or replace an edge directly, without the event ordering rules.

        Used when the data arrives as a snapshot rather than a journal, where
        some rows carry no timestamp at all.
        """
        source = self.register_node(subject, created)
        destination = self.register_node(target, created)
        actor_index = UNKNOWN if actor is None else self.register_node(actor, created)
        self._edges[(source, int(relation), destination)] = _EdgeRecord(level, created, actor_index)

    def remove_edge(self, subject: str, relation: RelationType, target: str) -> None:
        """Remove an edge if it is present."""
        source = self._node_index.get(subject)
        destination = self._node_index.get(target)
        if source is None or destination is None:
            return
        self._edges.pop((source, int(relation), destination), None)

    def build(self) -> AccessGraph:
        """Freeze the accumulated state into arrays."""
        count = len(self._edges)
        edge_src = np.empty(count, dtype=np.int32)
        edge_dst = np.empty(count, dtype=np.int32)
        edge_rel = np.empty(count, dtype=np.int8)
        edge_level = np.empty(count, dtype=np.int8)
        edge_created = np.empty(count, dtype=np.int64)
        edge_actor = np.empty(count, dtype=np.int32)

        for position, ((src, relation, dst), record) in enumerate(self._edges.items()):
            edge_src[position] = src
            edge_dst[position] = dst
            edge_rel[position] = relation
            edge_level[position] = record.level
            edge_created[position] = record.created
            edge_actor[position] = record.actor

        return AccessGraph(
            node_ids=tuple(self._node_ids),
            node_index=dict(self._node_index),
            node_type=np.array(self._node_type, dtype=np.int8),
            node_created=np.array(self._node_created, dtype=np.int64),
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_rel=edge_rel,
            edge_level=edge_level,
            edge_created=edge_created,
            edge_actor=edge_actor,
        )
