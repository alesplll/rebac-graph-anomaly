"""Replay a generated journal into a live Neo4j, for the demonstration.

Writes the property names the authorization engine itself writes — `created_at`,
`updated_at`, `actor`, `level` — so the detector's adapter reads the result exactly as
it reads a graph the engine produced. This is the mapping of
`configs/mapping/opens3.yaml` turned around to write instead of read.

It does not go through the engine's gRPC API, and that is deliberate: doing so would
drag grpcio and another project's generated stubs into this repository, which section
12 of the design document forbids. The demonstration's key moment — granting yourself
admin — is performed with the engine's own client, not with this script.

    uv run python scripts/fill_live_graph.py --config configs/generator/small-history.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.util.timeutil import DAY_MS

_LABELS = {"user": "User", "group": "Group", "bucket": "Bucket", "object": "Object"}

_WRITE = """
MERGE (subject:`%s` {id: $subject})
  ON CREATE SET subject.created_at = $ts
MERGE (object:`%s` {id: $object})
  ON CREATE SET object.created_at = $ts
MERGE (subject)-[rel:`%s`]->(object)
  ON CREATE SET rel.created_at = $ts
SET rel.updated_at = $ts, rel.actor = $actor%s
RETURN rel.updated_at AS written
"""


def _label(entity_id: str) -> str:
    return _LABELS[entity_id.split(":", 1)[0]]


def write_journal(driver, events: Iterable[GraphEvent]) -> int:
    """Apply a journal to the graph in time order. Returns the number of grants."""
    written = 0
    with driver.session() as session:
        for event in events:
            if event.op is not EventOp.GRANT:
                session.run(
                    f"MATCH (s {{id: $subject}})-[r:`{event.relation.name}`]->"
                    "(o {id: $object}) DELETE r",
                    subject=event.subject,
                    object=event.object,
                )
                continue

            carries_level = event.level is not PermissionLevel.NONE
            query = _WRITE % (
                _label(event.subject),
                _label(event.object),
                event.relation.name,
                ", rel.level = $level" if carries_level else "",
            )
            parameters: dict[str, object] = {
                "subject": event.subject,
                "object": event.object,
                "ts": event.ts,
                "actor": event.actor,
            }
            if carries_level:
                parameters["level"] = event.level.name.lower()
            session.run(query, **parameters)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generator/small-history.yaml")
    )
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password123")
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="keep only the last N days of the journal; 0 keeps all of it",
    )
    arguments = parser.parse_args()

    from neo4j import GraphDatabase

    dataset = build_dataset(load_dataset_config(arguments.config))
    events = dataset.events
    if arguments.days:
        cutoff = dataset.window_end - arguments.days * DAY_MS
        events = tuple(event for event in events if event.ts >= cutoff)

    driver = GraphDatabase.driver(arguments.uri, auth=(arguments.user, arguments.password))
    try:
        written = write_journal(driver, events)
    finally:
        driver.close()
    print(f"wrote {written} relationships into {arguments.uri}")


if __name__ == "__main__":
    main()
