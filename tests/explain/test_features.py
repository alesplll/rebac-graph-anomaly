"""Which of the candidate's own features moved the score."""

from pathlib import Path

import pytest

from rga.explain.features import feature_contributions
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def fitted():
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(train)
    return scorer, train, evaluation


def test_contributions_are_named_grouped_and_ordered(fitted) -> None:
    scorer, _, evaluation = fitted

    found = feature_contributions(scorer, evaluation, 0, top=5)

    assert len(found) == 5
    assert all(item.name in evaluation.matrix.block.names for item in found)
    assert all(isinstance(item.group, FeatureGroup) for item in found)
    magnitudes = [abs(item.contribution) for item in found]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_an_unobserved_feature_is_marked_not_zeroed(fitted) -> None:
    scorer, _, evaluation = fitted

    found = feature_contributions(scorer, evaluation, 0, top=len(evaluation.matrix.block))
    unobserved = [item for item in found if not item.observed]

    assert all(item.contribution == 0.0 for item in unobserved)


def test_a_structure_only_scorer_reports_no_feature_contributions(fitted) -> None:
    _, train, evaluation = fitted
    structural = GnnScorer(seed=0, config=FAST)
    structural.fit(train)

    assert feature_contributions(structural, evaluation, 0) == ()
