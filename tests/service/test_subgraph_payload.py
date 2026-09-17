"""The card gives the drawing everything it needs and nothing it cannot use."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer
from rga.service.app import create_app
from rga.service.config import load_service_config

FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def card():
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(build_candidates(dataset, Span.TRAIN))
    config = load_service_config(Path("configs/service/synthetic.yaml"))
    client = TestClient(create_app(config, scorer=scorer))
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]
    return client.get(f"/api/incidents/{listed['id']}").json()


def test_every_edge_endpoint_is_a_declared_node(card) -> None:
    names = {node["id"] for node in card["subgraph"]["nodes"]}

    for edge in card["subgraph"]["edges"]:
        assert edge["subject"] in names
        assert edge["object"] in names


def test_the_change_itself_sits_at_the_centre(card) -> None:
    centre = [node["id"] for node in card["subgraph"]["nodes"] if node["hops"] == 0]

    assert card["subject"] in centre or card["object"] in centre


def test_layers_are_small_enough_to_draw(card) -> None:
    assert len(card["subgraph"]["nodes"]) <= 80
    assert max(node["hops"] for node in card["subgraph"]["nodes"]) <= 3
