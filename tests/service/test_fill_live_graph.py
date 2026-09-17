"""Filling a live Neo4j with a generated organization."""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _credentials() -> tuple[str, str, str]:
    return (
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password123"),
    )


def test_a_filled_graph_reads_back_through_the_adapter() -> None:
    from neo4j import GraphDatabase
    from scripts.fill_live_graph import write_journal

    from rga.adapters.mapping import RelationMapping
    from rga.adapters.neo4j_source import BoltReader, Neo4jSource
    from rga.generator.config import load_dataset_config
    from rga.generator.dataset import build_dataset

    uri, user, password = _credentials()
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        written = write_journal(driver, dataset.events[:500])
    finally:
        driver.close()

    reader = BoltReader(uri, user, password)
    try:
        source = Neo4jSource(reader, RelationMapping.load(Path("configs/mapping/opens3.yaml")))
        capabilities = source.capabilities()
        events = list(source.events())
        graph = source.snapshot()
    finally:
        reader.close()

    assert written > 0
    assert capabilities.level == 2
    assert len(events) == graph.num_edges
    assert [event.ts for event in events] == sorted(event.ts for event in events)
