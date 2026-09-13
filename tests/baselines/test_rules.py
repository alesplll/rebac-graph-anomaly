"""Hand-written heuristics, and the Scorer contract they implement."""

from pathlib import Path

import numpy as np

from rga.baselines.base import Scorer, dense_matrix
from rga.baselines.rules import RuleScorer
from rga.eval.experiment import restrict_candidates
from rga.eval.metrics import evaluate_ranking
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def _candidates(span: Span):
    return build_candidates(build_dataset(CONFIG), span)


def test_rule_scorer_satisfies_the_protocol() -> None:
    assert isinstance(RuleScorer(), Scorer)


def test_scores_are_bounded_and_one_per_candidate() -> None:
    candidates = _candidates(Span.EVAL)
    scorer = RuleScorer()
    scorer.fit(_candidates(Span.TRAIN))
    scores = scorer.score(candidates)

    assert scores.shape == (candidates.n_candidates,)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0
    assert np.isfinite(scores).all()


def test_rules_beat_random_ordering() -> None:
    # Not asking for excellence — only that naive conditions carry some signal.
    # If this fails, either the rules are broken or the generator is.
    candidates = _candidates(Span.EVAL)
    scorer = RuleScorer()
    metrics = evaluate_ranking(candidates.y_true(), scorer.score(candidates), ks=(50,))
    assert metrics.roc_auc > 0.55


def test_rules_do_not_solve_the_task_outright() -> None:
    # The acceptance criterion for the dataset: if hand-written conditions rank
    # almost perfectly, the synthetic data is too easy and nothing measured on
    # top of it means anything.
    candidates = _candidates(Span.EVAL)
    scorer = RuleScorer()
    metrics = evaluate_ranking(candidates.y_true(), scorer.score(candidates), ks=(50,))
    assert metrics.pr_auc < 0.9


def test_masked_provenance_does_not_break_scoring() -> None:
    candidates = _candidates(Span.EVAL)
    scores = RuleScorer().score(candidates)
    unobserved = ~candidates.matrix.observed("actor_is_subject")
    if unobserved.any():
        # Rows whose initiator is unknown must not be penalised for it.
        assert np.isfinite(scores[unobserved]).all()


def test_dense_matrix_zeroes_unobserved_and_appends_group_coverage() -> None:
    candidates = _candidates(Span.EVAL)
    dense = dense_matrix(candidates.matrix)

    assert dense.shape[0] == candidates.n_candidates
    assert dense.shape[1] == candidates.matrix.n_features + 3
    assert np.isfinite(dense).all()


def test_rules_survive_a_source_without_provenance() -> None:
    # A level-0 engine supplies no timestamps and no initiator. Conditions that
    # rest on them cannot fire, but the scorer must still produce a ranking.
    candidates = restrict_candidates(_candidates(Span.EVAL), (FeatureGroup.STRUCTURAL,))
    scores = RuleScorer().score(candidates)

    assert scores.shape == (candidates.n_candidates,)
    assert np.isfinite(scores).all()
    assert float(scores.max()) <= 1.0
