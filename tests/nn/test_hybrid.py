"""The classical detector with the network's structural view added as a column."""

from pathlib import Path

import numpy as np
import pytest

from rga.baselines.base import Scorer
from rga.baselines.outliers import IsolationForestScorer
from rga.eval.experiment import build_scorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.hybrid import GnnAugmentedForestScorer

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def spans():
    dataset = build_dataset(CONFIG)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


def test_the_scorer_satisfies_the_protocol() -> None:
    assert isinstance(GnnAugmentedForestScorer(seed=0, config=FAST), Scorer)


def test_the_factory_knows_the_name() -> None:
    assert build_scorer("gnn_forest", seed=0).name == "gnn_forest"


def test_one_finite_score_per_candidate(spans) -> None:
    train, evaluation = spans
    scorer = GnnAugmentedForestScorer(seed=0, config=FAST)

    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.shape == (evaluation.n_candidates,)
    assert np.isfinite(scores).all()


def test_the_network_column_changes_the_ranking(spans) -> None:
    """Without this, the hybrid would just be the forest under another name."""
    train, evaluation = spans

    hybrid = GnnAugmentedForestScorer(seed=0, config=FAST)
    hybrid.fit(train)
    plain = IsolationForestScorer(seed=0)
    plain.fit(train)

    mixed = np.argsort(-hybrid.score(evaluation))[:50]
    alone = np.argsort(-plain.score(evaluation))[:50]

    assert set(mixed) != set(alone)


def test_scoring_before_fitting_is_refused(spans) -> None:
    _, evaluation = spans

    with pytest.raises(RuntimeError, match="must be fit"):
        GnnAugmentedForestScorer(seed=0, config=FAST).score(evaluation)
