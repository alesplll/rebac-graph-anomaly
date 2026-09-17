"""The supervised contrast baseline."""

from pathlib import Path

import numpy as np
import pytest

from rga.baselines.base import Scorer
from rga.eval.experiment import build_scorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer

HISTORY = load_dataset_config(Path("configs/generator/small-history.yaml"))
PLAIN = load_dataset_config(Path("configs/generator/small.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def spans():
    dataset = build_dataset(HISTORY)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


def test_the_scorer_satisfies_the_protocol() -> None:
    assert isinstance(SupervisedGnnScorer(seed=0, config=FAST), Scorer)


def test_the_factory_knows_the_name() -> None:
    assert build_scorer("gnn_supervised", seed=0).name == "gnn_supervised"


def test_one_finite_score_per_candidate(spans) -> None:
    train, evaluation = spans
    scorer = SupervisedGnnScorer(seed=0, config=FAST)

    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.shape == (evaluation.n_candidates,)
    assert np.isfinite(scores).all()


def test_fitting_without_labels_is_refused() -> None:
    train = build_candidates(build_dataset(PLAIN), Span.TRAIN)

    with pytest.raises(ValueError, match="labelled"):
        SupervisedGnnScorer(seed=0, config=FAST).fit(train)
