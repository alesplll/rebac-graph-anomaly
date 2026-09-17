"""Every observation says why it matters, for a reader who does not know the system."""

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


def test_each_observation_carries_its_reason(candidates) -> None:
    for position in range(min(candidates.n_candidates, 40)):
        for observation in describe(candidates, position):
            assert observation.text
            assert observation.why
            assert observation.why != observation.text


def test_the_reason_explains_the_norm_not_the_verdict(candidates) -> None:
    """A reason says what the usual process looks like. It accuses nobody."""
    for position in range(min(candidates.n_candidates, 120)):
        for observation in describe(candidates, position):
            said = observation.why.lower()
            for word in FORBIDDEN_WORDS:
                assert word not in said


def test_a_self_grant_explains_who_normally_grants(candidates) -> None:
    column = candidates.matrix.column("actor_is_subject")
    observed = candidates.matrix.observed("actor_is_subject")
    position = int(np.flatnonzero((column > 0.5) & observed)[0])

    reasons = " ".join(item.why for item in describe(candidates, position))

    assert "владелец" in reasons or "администратор" in reasons
