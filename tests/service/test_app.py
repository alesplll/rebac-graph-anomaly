"""The routes the page talks to."""

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
def client():
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(build_candidates(dataset, Span.TRAIN))
    config = load_service_config(Path("configs/service/synthetic.yaml"))
    return TestClient(create_app(config, scorer=scorer))


def test_status_describes_what_is_running(client) -> None:
    body = client.get("/api/status").json()

    assert body["source"] == "synthetic"
    assert body["capability_level"] >= 1
    assert body["scorer"] == "gnn_supervised"
    assert body["candidates"] > 0


def test_the_queue_comes_back_ranked(client) -> None:
    body = client.get("/api/incidents", params={"limit": 10}).json()

    assert len(body["incidents"]) == 10
    scores = [item["score"] for item in body["incidents"]]
    assert scores == sorted(scores, reverse=True)
    assert body["incidents"][0]["rank"] == 1


def test_the_queue_can_be_filtered_by_subject(client) -> None:
    everything = client.get("/api/incidents", params={"limit": 50}).json()["incidents"]
    subject = everything[0]["subject"]

    filtered = client.get("/api/incidents", params={"limit": 50, "subject": subject}).json()

    assert filtered["incidents"]
    assert {item["subject"] for item in filtered["incidents"]} == {subject}


def test_a_card_carries_its_grounds(client) -> None:
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]

    card = client.get(f"/api/incidents/{listed['id']}").json()

    assert card["id"] == listed["id"]
    assert card["summary"]
    assert card["subgraph"]["nodes"]


def test_an_unknown_card_is_a_clean_404(client) -> None:
    assert client.get("/api/incidents/" + "0" * 16).status_code == 404


def test_a_node_neighbourhood_comes_back(client) -> None:
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]

    body = client.get("/api/nodes", params={"id": listed["subject"]}).json()

    assert body["id"] == listed["subject"]
    assert body["subgraph"]["nodes"]


def test_an_unknown_node_is_a_clean_404(client) -> None:
    assert client.get("/api/nodes", params={"id": "user:nobody"}).status_code == 404


def test_refresh_reruns_the_analysis(client) -> None:
    before = client.get("/api/status").json()["refreshed_at"]

    assert client.post("/api/refresh").status_code == 200
    assert client.get("/api/status").json()["refreshed_at"] >= before
