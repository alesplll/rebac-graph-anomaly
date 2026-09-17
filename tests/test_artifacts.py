"""A fitted scorer survives a trip through the filesystem."""

from pathlib import Path

import numpy as np
import pytest

from rga.artifacts import block_fingerprint, load_scorer, read_manifest, save_scorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def spans():
    dataset = build_dataset(CONFIG)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


@pytest.mark.parametrize(
    "factory", [SupervisedGnnScorer, GnnScorer], ids=["supervised", "self-supervised"]
)
def test_a_reloaded_scorer_ranks_identically(spans, tmp_path, factory) -> None:
    train, evaluation = spans
    scorer = factory(seed=3, config=FAST)
    scorer.fit(train)
    before = scorer.score(evaluation)

    save_scorer(tmp_path / "artifact", scorer, dataset="small-history")
    after = load_scorer(tmp_path / "artifact").score(evaluation)

    assert np.allclose(before, after, atol=1e-6)


def test_the_manifest_records_what_was_trained(spans, tmp_path) -> None:
    train, _ = spans
    scorer = SupervisedGnnScorer(seed=3, config=FAST)
    scorer.fit(train)

    save_scorer(tmp_path / "artifact", scorer, dataset="small-history")
    manifest = read_manifest(tmp_path / "artifact")

    assert manifest["scorer"] == "gnn_supervised"
    assert manifest["dataset"] == "small-history"
    assert manifest["features"] == block_fingerprint()
    assert manifest["seed"] == 3


def test_an_artefact_from_another_feature_space_is_refused(spans, tmp_path) -> None:
    train, _ = spans
    scorer = SupervisedGnnScorer(seed=3, config=FAST)
    scorer.fit(train)
    save_scorer(tmp_path / "artifact", scorer, dataset="small-history")

    manifest = tmp_path / "artifact" / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(block_fingerprint(), "0" * 16),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature"):
        load_scorer(tmp_path / "artifact")


def test_saving_before_fitting_is_refused(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        save_scorer(tmp_path / "artifact", GnnScorer(seed=0, config=FAST), dataset="x")
