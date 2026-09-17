"""The network as one more scorer in the stand."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from rga.baselines.base import Scorer
from rga.eval.experiment import build_scorer
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
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
    """The analyst's queue is what has to be reproducible.

    Not the scores themselves: the rank transform is a step function over the
    training distribution, so the 1e-8 wobble that threaded CPU accumulation leaves
    in a logit pushes a candidate a whole step of 1/N. Measured over three trials,
    25 to 59 of 637 candidates move, by up to 0.01 — while the top of the queue and
    the order as a whole stay put. Those are the properties asserted here.
    """
    train, evaluation = spans

    first = GnnScorer(seed=4, config=FAST)
    first.fit(train)
    second = GnnScorer(seed=4, config=FAST)
    second.fit(train)

    left, right = first.score(evaluation), second.score(evaluation)

    assert set(np.argsort(-left)[:50]) == set(np.argsort(-right)[:50])
    assert spearmanr(left, right).statistic > 0.999


def test_scoring_before_fitting_is_refused(spans) -> None:
    _, evaluation = spans

    with pytest.raises(RuntimeError, match="must be fit"):
        GnnScorer(seed=0, config=FAST).score(evaluation)


def test_the_ranking_ignores_the_candidate_feature_row(spans) -> None:
    """The self-supervised scorer reads structure, not the context row.

    A positive and its corrupted negatives share a feature row, so nothing during
    training constrains the head's weights on those inputs; at scoring time the row
    varies and would feed the logit noise. Measured over five seeds, keeping the row
    cost 0.073 +/- 0.043 PR-AUC against 0.368 +/- 0.235 without it. So the row is
    kept out, and perturbing it must not move the queue.
    """
    train, evaluation = spans
    scorer = GnnScorer(seed=7, config=FAST)
    scorer.fit(train)

    before = scorer.score(evaluation)
    moved = evaluation.matrix.values.copy()
    for group in (FeatureGroup.TEMPORAL, FeatureGroup.PROVENANCE):
        moved[:, evaluation.matrix.group_indices(group)] += np.float32(5.0)
    after = scorer.score(replace(evaluation, matrix=replace(evaluation.matrix, values=moved)))

    assert np.array_equal(before, after)
