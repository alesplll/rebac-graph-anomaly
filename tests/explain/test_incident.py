"""The card the service hands the interface."""

from pathlib import Path

import numpy as np
import pytest

from rga.explain.incident import build_incident, incident_id
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def fitted():
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(train)
    return scorer, evaluation


def test_the_identifier_is_stable_and_specific() -> None:
    key = ("user:alex", 2, "bucket:logs")

    assert incident_id(key, 1000) == incident_id(key, 1000)
    assert incident_id(key, 1000) != incident_id(key, 1001)
    assert len(incident_id(key, 1000)) == 16


def test_the_card_carries_everything_the_page_needs(fitted) -> None:
    scorer, evaluation = fitted

    card = build_incident(scorer, evaluation, 0, score=0.9, rank=1, cap=10)
    payload = card.as_dict()

    assert payload["id"] == incident_id(evaluation.keys[0], int(evaluation.ts[0]))
    assert payload["score"] == 0.9
    assert payload["rank"] == 1
    assert payload["summary"]
    assert "features" in payload and "edges" in payload and "subgraph" in payload
    assert {"nodes", "edges"} <= set(payload["subgraph"])


def test_the_cheap_form_skips_the_expensive_parts(fitted) -> None:
    scorer, evaluation = fitted

    listed = build_incident(scorer, evaluation, 0, score=0.9, rank=1, explain=False)

    assert listed.as_dict()["summary"]
    assert listed.as_dict()["edges"] == []


def test_every_subgraph_node_has_a_hop_distance(fitted) -> None:
    scorer, evaluation = fitted

    subgraph = build_incident(scorer, evaluation, 0, score=0.5, rank=2, cap=10).as_dict()[
        "subgraph"
    ]

    assert all(isinstance(node["hops"], int) for node in subgraph["nodes"])
    assert min(node["hops"] for node in subgraph["nodes"]) == 0
    assert np.isfinite([edge["importance"] for edge in subgraph["edges"]]).all()
