"""Labelled incidents in the training span, for the supervised baseline."""

from pathlib import Path

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

HISTORY = load_dataset_config(Path("configs/generator/small-history.yaml"))
PLAIN = load_dataset_config(Path("configs/generator/small.yaml"))

#: The five patterns the supervised baseline is allowed to learn.
KNOWN = {
    "self_grant_admin",
    "privileged_group_join",
    "grant_burst",
    "hierarchy_bypass",
    "cross_department",
}


def test_plain_config_keeps_an_unlabelled_training_span() -> None:
    dataset = build_dataset(PLAIN)

    assert dataset.train_labels == ()
    assert int(build_candidates(dataset, Span.TRAIN).labels.sum()) == 0


def test_history_config_labels_incidents_before_the_split() -> None:
    dataset = build_dataset(HISTORY)

    assert dataset.train_labels != ()
    assert all(label.ts < dataset.split_ts for label in dataset.train_labels)


def test_only_the_configured_patterns_appear_before_the_split() -> None:
    dataset = build_dataset(HISTORY)

    planted = {label.pattern for label in dataset.train_labels}

    assert planted <= KNOWN


def test_training_candidates_carry_those_labels() -> None:
    dataset = build_dataset(HISTORY)

    train = build_candidates(dataset, Span.TRAIN)

    assert int(train.labels.sum()) > 0
    assert {pattern for pattern in train.patterns if pattern} <= KNOWN


def test_the_evaluation_window_still_carries_every_pattern() -> None:
    dataset = build_dataset(HISTORY)

    window = {label.pattern for label in dataset.labels}

    assert "shadow_group" in window
    assert "delegation_cascade" in window
