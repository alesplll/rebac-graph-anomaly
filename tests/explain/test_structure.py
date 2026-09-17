"""Which relationships around the change drove its score."""

from pathlib import Path

import pytest

from rga.explain.structure import edge_importance, neighbourhood
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


def test_the_neighbourhood_is_bounded(fitted) -> None:
    _, evaluation = fitted
    subject, _, target = evaluation.keys[0]

    found = neighbourhood(evaluation.graph, subject, target, hops=2, cap=25)

    assert 0 < len(found) <= 25
    assert len(set(found.tolist())) == len(found)


def test_every_edge_gets_an_importance(fitted) -> None:
    scorer, evaluation = fitted

    found = edge_importance(scorer, evaluation, 0, cap=15)

    assert 0 < len(found) <= 15
    assert all(isinstance(item.importance, float) for item in found)
    magnitudes = [abs(item.importance) for item in found]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_graph_is_left_untouched(fitted) -> None:
    scorer, evaluation = fitted
    before = evaluation.graph.num_edges

    edge_importance(scorer, evaluation, 0, cap=10)

    assert evaluation.graph.num_edges == before
