"""Sentences an analyst reads."""

from pathlib import Path

import numpy as np
import pytest

from rga.explain.text import FORBIDDEN_WORDS, describe
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))


@pytest.fixture(scope="module")
def candidates():
    return build_candidates(build_dataset(CONFIG), Span.EVAL)


def test_every_candidate_gets_at_least_one_sentence(candidates) -> None:
    for position in range(min(candidates.n_candidates, 50)):
        assert describe(candidates, position)


def test_a_self_grant_is_named_as_an_observation(candidates) -> None:
    column = candidates.matrix.column("actor_is_subject")
    observed = candidates.matrix.observed("actor_is_subject")
    position = int(np.flatnonzero((column > 0.5) & observed)[0])

    said = " ".join(describe(candidates, position))

    assert "сам" in said


def test_no_verdict_is_ever_pronounced(candidates) -> None:
    """The system supports a decision. It does not announce a compromise."""
    for position in range(min(candidates.n_candidates, 200)):
        said = " ".join(describe(candidates, position)).lower()
        for word in FORBIDDEN_WORDS:
            assert word not in said, f"candidate {position} says {word!r}"
