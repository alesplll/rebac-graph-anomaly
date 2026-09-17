"""The reference page is built from the same texts the cards use."""

from pathlib import Path

import pytest

from rga.explain.reference import reference
from rga.explain.text import CATALOGUE, describe
from rga.features.build import CANDIDATE_BLOCK, Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))


@pytest.fixture(scope="module")
def candidates():
    return build_candidates(build_dataset(CONFIG), Span.EVAL)


def test_every_observation_a_card_can_show_is_in_the_catalogue(candidates) -> None:
    kinds = {entry.kind for entry in CATALOGUE}

    for position in range(min(candidates.n_candidates, 200)):
        for observation in describe(candidates, position):
            assert observation.kind in kinds


def test_the_catalogue_says_what_to_look_at() -> None:
    for entry in CATALOGUE:
        assert entry.title and entry.why and entry.look_at


def test_the_reference_carries_everything_the_page_renders() -> None:
    body = reference()

    assert {"observations", "groups", "features", "tables"} <= set(body)
    assert len(body["observations"]) == len(CATALOGUE)
    assert len(body["features"]) == len(CANDIDATE_BLOCK.names)
    assert {"edges", "features"} <= set(body["tables"])
