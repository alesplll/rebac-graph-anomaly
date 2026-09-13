"""Classical outlier detectors on the same feature space."""

from pathlib import Path

import numpy as np
import pytest

from rga.baselines.base import Scorer
from rga.baselines.outliers import IsolationForestScorer, LocalOutlierFactorScorer
from rga.eval.metrics import evaluate_ranking
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def _spans():
    dataset = build_dataset(CONFIG)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_scorers_satisfy_the_protocol(factory) -> None:
    assert isinstance(factory(), Scorer)


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_scores_are_finite_and_one_per_candidate(factory) -> None:
    train, evaluation = _spans()
    scorer = factory()
    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.shape == (evaluation.n_candidates,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_scoring_before_fitting_is_refused(factory) -> None:
    _, evaluation = _spans()
    with pytest.raises(RuntimeError, match="fit"):
        factory().score(evaluation)


def test_isolation_forest_is_reproducible() -> None:
    train, evaluation = _spans()
    first = IsolationForestScorer(seed=5)
    first.fit(train)
    second = IsolationForestScorer(seed=5)
    second.fit(train)
    assert np.allclose(first.score(evaluation), second.score(evaluation))


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_outlier_detectors_carry_some_signal(factory) -> None:
    train, evaluation = _spans()
    scorer = factory()
    scorer.fit(train)
    metrics = evaluate_ranking(evaluation.y_true(), scorer.score(evaluation), ks=(50,))
    assert metrics.roc_auc > 0.5
