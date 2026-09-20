"""Source to candidates to a ranked queue."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rga.explain.incident import incident_id
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer
from rga.service.analysis import analyse
from rga.service.config import load_service_config

FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)
SYNTHETIC = Path("configs/service/synthetic.yaml")


@pytest.fixture(scope="module")
def scorer():
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    fitted = SupervisedGnnScorer(seed=0, config=FAST)
    fitted.fit(build_candidates(dataset, Span.TRAIN))
    return fitted


def test_the_synthetic_source_produces_a_ranked_queue(scorer) -> None:
    analysis = analyse(load_service_config(SYNTHETIC), scorer)

    assert analysis.candidates.n_candidates > 0
    assert analysis.scores.shape == (analysis.candidates.n_candidates,)
    ranked = analysis.scores[analysis.order]
    assert np.all(np.diff(ranked) <= 0)
    assert analysis.level >= 1


def test_an_incident_can_be_found_by_its_identifier(scorer) -> None:
    analysis = analyse(load_service_config(SYNTHETIC), scorer)
    position = int(analysis.order[0])
    wanted = incident_id(
        analysis.candidates.keys[position], int(analysis.candidates.ts[position])
    )

    assert analysis.find(wanted) == position
    assert analysis.find("0" * 16) is None


def test_the_window_width_limits_what_is_scored(scorer) -> None:
    wide = load_service_config(SYNTHETIC)
    narrow = replace(wide, window_days=1)

    assert (
        analyse(narrow, scorer).candidates.n_candidates
        < analyse(wide, scorer).candidates.n_candidates
    )


def test_an_unknown_source_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        replace(load_service_config(SYNTHETIC), source="carrier pigeon")
