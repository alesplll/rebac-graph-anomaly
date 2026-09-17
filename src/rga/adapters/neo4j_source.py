"""A live opens3-rebac deployment as a data source.

Written against a narrow reader interface rather than the driver, so the
translation logic is testable without a database. The driver-backed reader is a
dozen lines at the bottom of this file and is exercised by a marked integration
test.

Capability level depends on the data, not on the product: a deployment whose
edges predate the timestamp patch reports level 0 and its temporal features are
masked, exactly as any third-party engine without timestamps would be.

A deployment that does record timestamps also gets a change log, reconstructed by
ordering the edges by creation time. It holds grants only — a snapshot cannot show
what was taken away — which is the limitation section 12.2 of the design document
accepts in exchange for needing nothing but a snapshot.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol

from rga.adapters.base import Capabilities
from rga.adapters.mapping import RelationMapping
from rga.domain.events import EventOp, GraphEvent
from rga.domain.graph import UNKNOWN, AccessGraph, GraphBuilder
from rga.domain.relations import PermissionLevel

#: Rows inspected when deciding what the deployment can provide.
_PROBE_SIZE = 512


class GraphRecordReader(Protocol):
    """Rows of the authorization graph, in whatever shape the store returns them."""

    def nodes(self) -> Iterator[Mapping[str, object]]:
        """Yield mappings with `id` and `created_at`."""
        ...

    def relationships(self) -> Iterator[Mapping[str, object]]:
        """Yield mappings with subject, relation, object, level, created_at, actor."""
        ...


def _as_int(value: object) -> int:
    """Coerce an optional timestamp, mapping absence to the unknown sentinel."""
    return UNKNOWN if value is None else int(value)  # type: ignore[arg-type]


class Neo4jSource:
    """Reads the access graph from a Neo4j-backed authorization engine."""

    def __init__(self, reader: GraphRecordReader, mapping: RelationMapping) -> None:
        self._reader = reader
        self._mapping = mapping

    def capabilities(self) -> Capabilities:
        """Probe the data to decide the level.

        `change_log` is always False: a snapshot cannot show a revocation, since
        a revoked edge is simply absent. Observing revocations needs the Kafka
        adapter, which arrives with the streaming mode in a later module.
        """
        timestamps = False
        provenance = False
        for index, row in enumerate(self._reader.relationships()):
            if index >= _PROBE_SIZE:
                break
            timestamps = timestamps or row.get("created_at") is not None
            provenance = provenance or row.get("actor") is not None
        return Capabilities(timestamps=timestamps, provenance=provenance, change_log=False)

    def snapshot(self, at: int | None = None) -> AccessGraph:
        """Build the graph, optionally as it stood at `at`."""
        if at is not None and not self.capabilities().timestamps:
            raise ValueError("cannot honour a cutoff: this deployment records no timestamps")

        builder = GraphBuilder()
        for row in self._reader.nodes():
            builder.register_node(str(row["id"]), _as_int(row.get("created_at")))

        for row in self._reader.relationships():
            created = _as_int(row.get("created_at"))
            if at is not None and created > at:
                continue
            level_property = row.get("level")
            relation, level = self._mapping.translate(
                str(row["relation"]),
                None if level_property is None else str(level_property),
            )
            actor = row.get("actor")
            builder.add_edge(
                str(row["subject"]),
                relation,
                str(row["object"]),
                level=int(level),
                created=created,
                actor=None if actor is None else str(actor),
            )

        return builder.build()

    def events(self, since: int = 0, until: int | None = None) -> Iterator[GraphEvent]:
        """The change log implied by the edge timestamps.

        Two limits are inherent and not worked around. A snapshot shows no
        revocations — a revoked edge is simply absent — so every event is a grant,
        and a right granted and taken back between two snapshots never existed as
        far as this source is concerned. And an edge written before the timestamp
        patch carries no time; it is real context, so it is emitted just ahead of
        the earliest known change rather than dropped.
        """
        rows = list(self._reader.relationships())
        stamped = [row for row in rows if row.get("created_at") is not None]
        if not stamped:
            raise ValueError("this deployment records no timestamps; it has no journal")

        earliest = min(int(row["created_at"]) for row in stamped)  # type: ignore[arg-type]
        ordered: list[tuple[int, Mapping[str, object]]] = [
            (earliest - 1, row) for row in rows if row.get("created_at") is None
        ]
        ordered += [(int(row["created_at"]), row) for row in stamped]  # type: ignore[arg-type]
        ordered.sort(key=lambda pair: pair[0])

        for ts, row in ordered:
            if ts < since or (until is not None and ts > until):
                continue
            level_property = row.get("level")
            relation, level = self._mapping.translate(
                str(row["relation"]),
                None if level_property is None else str(level_property),
            )
            actor = row.get("actor")
            yield GraphEvent(
                ts=ts,
                op=EventOp.GRANT,
                subject=str(row["subject"]),
                relation=relation,
                object=str(row["object"]),
                level=PermissionLevel(int(level)),
                actor=None if actor is None else str(actor),
            )


class BoltReader:
    """Reads graph rows over the official Neo4j driver."""

    _NODES = "MATCH (n) RETURN n.id AS id, n.created_at AS created_at"
    _RELATIONSHIPS = """
    MATCH (s)-[r]->(o)
    RETURN s.id AS subject, type(r) AS relation, o.id AS object,
           r.level AS level, r.created_at AS created_at, r.actor AS actor
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def nodes(self) -> Iterator[Mapping[str, object]]:
        with self._driver.session() as session:
            for record in session.run(self._NODES):
                yield dict(record)

    def relationships(self) -> Iterator[Mapping[str, object]]:
        with self._driver.session() as session:
            for record in session.run(self._RELATIONSHIPS):
                yield dict(record)

    def close(self) -> None:
        self._driver.close()
