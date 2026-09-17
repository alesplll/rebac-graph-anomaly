"""A fitted scorer must work on a graph it has never seen."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer
from rga.nn.supervised import SupervisedGnnScorer

BASE = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def two_worlds():
    """Two datasets from different seeds: different graphs, different node counts."""
    here = build_dataset(replace(BASE, seed=11))
    elsewhere = build_dataset(replace(BASE, seed=12))
    return (
        build_candidates(here, Span.TRAIN),
        build_candidates(elsewhere, Span.EVAL),
    )


def test_the_graphs_really_differ(two_worlds) -> None:
    """Guards the two tests below: on identical graphs they would prove nothing."""
    home, foreign = two_worlds

    assert home.graph.num_nodes != foreign.graph.num_nodes


def test_the_self_supervised_scorer_scores_a_foreign_graph(two_worlds) -> None:
    home, foreign = two_worlds
    scorer = GnnScorer(seed=0, config=FAST)
    scorer.fit(home)

    scores = scorer.score(foreign)

    assert scores.shape == (foreign.n_candidates,)
    assert np.isfinite(scores).all()


def test_the_supervised_scorer_scores_a_foreign_graph(two_worlds) -> None:
    home, foreign = two_worlds
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(home)

    scores = scorer.score(foreign)

    assert scores.shape == (foreign.n_candidates,)
    assert np.isfinite(scores).all()
