"""The network as one more scorer in the stand."""

from pathlib import Path

import numpy as np
import pytest

from rga.baselines.base import Scorer
from rga.eval.experiment import build_scorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def spans():
    dataset = build_dataset(CONFIG)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


def test_the_scorer_satisfies_the_protocol() -> None:
    assert isinstance(GnnScorer(seed=0, config=FAST), Scorer)


def test_the_factory_knows_the_name() -> None:
    assert build_scorer("gnn", seed=0).name == "gnn"


def test_one_finite_score_per_candidate(spans) -> None:
    train, evaluation = spans
    scorer = GnnScorer(seed=0, config=FAST)

    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.shape == (evaluation.n_candidates,)
    assert np.isfinite(scores).all()


def test_scores_stay_inside_the_unit_interval(spans) -> None:
    train, evaluation = spans
    scorer = GnnScorer(seed=0, config=FAST)

    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_the_same_seed_gives_the_same_queue(spans) -> None:
    """The analyst's queue is what has to be reproducible, to float tolerance."""
    train, evaluation = spans

    first = GnnScorer(seed=4, config=FAST)
    first.fit(train)
    second = GnnScorer(seed=4, config=FAST)
    second.fit(train)

    left, right = first.score(evaluation), second.score(evaluation)

    assert np.allclose(left, right, atol=1e-6)
    assert set(np.argsort(-left)[:50]) == set(np.argsort(-right)[:50])


def test_scoring_before_fitting_is_refused(spans) -> None:
    _, evaluation = spans

    with pytest.raises(RuntimeError, match="must be fit"):
        GnnScorer(seed=0, config=FAST).score(evaluation)
