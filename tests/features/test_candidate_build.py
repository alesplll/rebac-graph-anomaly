"""The streaming pass that turns a dataset into scored rows."""

from pathlib import Path

import numpy as np

from rga.features.build import CANDIDATE_BLOCK, Span, build_candidates
from rga.features.edges import EDGE_BLOCK
from rga.features.nodes import NODE_BLOCK
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def _dataset():
    return build_dataset(CONFIG)


def test_block_is_the_edge_block_plus_both_endpoints() -> None:
    assert len(CANDIDATE_BLOCK) == len(EDGE_BLOCK) + 2 * len(NODE_BLOCK)
    assert CANDIDATE_BLOCK.names[len(EDGE_BLOCK)].startswith("subj_")
    assert CANDIDATE_BLOCK.names[len(EDGE_BLOCK) + len(NODE_BLOCK)].startswith("obj_")


def test_eval_candidates_come_from_the_window_only() -> None:
    dataset = _dataset()
    candidates = build_candidates(dataset, Span.EVAL)
    assert candidates.n_candidates > 0
    assert bool((candidates.ts >= dataset.split_ts).all())
    assert bool((candidates.ts < dataset.window_end).all())


def test_train_candidates_stop_at_the_split_and_skip_the_warm_up() -> None:
    dataset = _dataset()
    candidates = build_candidates(dataset, Span.TRAIN)
    assert candidates.n_candidates > 0
    assert bool((candidates.ts < dataset.split_ts).all())
    assert bool((candidates.ts > dataset.config.timeline.start_ts).all())


def test_matrix_shape_agrees_with_the_block() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    assert candidates.matrix.n_features == len(CANDIDATE_BLOCK)
    assert candidates.matrix.n_rows == candidates.n_candidates
    assert candidates.matrix.values.dtype == np.float32


def test_every_injected_anomaly_is_labelled_among_the_candidates() -> None:
    dataset = _dataset()
    candidates = build_candidates(dataset, Span.EVAL)
    labelled = {
        key for key, flag in zip(candidates.keys, candidates.labels, strict=True) if flag
    }
    assert labelled == dataset.anomaly_keys()


def test_training_candidates_carry_no_labels() -> None:
    # Contamination, when configured, is deliberately unlabelled.
    candidates = build_candidates(_dataset(), Span.TRAIN)
    assert not candidates.labels.any()


def test_patterns_accompany_the_labels() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    named = {
        pattern
        for pattern, flag in zip(candidates.patterns, candidates.labels, strict=True)
        if flag
    }
    assert named
    assert "" not in named


def test_no_feature_is_nan_or_infinite() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    assert np.isfinite(candidates.matrix.values).all()


def test_provenance_is_observed_because_the_generator_records_actors() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    columns = candidates.matrix.group_indices(FeatureGroup.PROVENANCE)
    assert columns.size > 0
    assert candidates.matrix.mask[:, columns].any()


def test_build_is_reproducible() -> None:
    dataset = _dataset()
    first = build_candidates(dataset, Span.EVAL)
    second = build_candidates(dataset, Span.EVAL)
    assert np.array_equal(first.matrix.values, second.matrix.values)
    assert first.keys == second.keys
