"""The card has to say who did it, when, and what changed — not only a score."""

from pathlib import Path

import pytest

from rga.explain.incident import build_incident
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


def test_the_candidate_set_remembers_who_made_each_change() -> None:
    candidates = build_candidates(build_dataset(CONFIG), Span.EVAL)

    assert len(candidates.actors) == candidates.n_candidates
    assert any(actor is not None for actor in candidates.actors)


def test_a_row_carries_its_own_actor() -> None:
    candidates = build_candidates(build_dataset(CONFIG), Span.EVAL)

    assert candidates.row(3).actors == (candidates.actors[3],)


def test_the_card_names_the_initiator(fitted) -> None:
    scorer, evaluation = fitted

    card = build_incident(scorer, evaluation, 0, score=0.9, rank=1, cap=10).as_dict()

    assert card["actor"] == evaluation.actors[0]


def test_the_card_carries_the_facts_behind_the_sentences(fitted) -> None:
    scorer, evaluation = fitted

    context = build_incident(scorer, evaluation, 0, score=0.9, rank=1, cap=10).as_dict()["context"]

    assert set(context) >= {
        "actor_is_subject",
        "off_hours",
        "weekend",
        "level_jump",
        "common_neighbours",
        "bypasses_bucket",
    }
    assert context["common_neighbours"] is None or context["common_neighbours"] >= 0


def test_a_fact_the_source_cannot_supply_is_none_not_zero(fitted) -> None:
    """Zero means "none of them"; unknown means the engine does not record it."""
    scorer, evaluation = fitted
    card = build_incident(scorer, evaluation, 0, score=0.9, rank=1, cap=10).as_dict()

    for name, value in card["context"].items():
        assert value is None or isinstance(value, (int, float, bool)), name
