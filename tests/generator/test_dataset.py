"""Dataset assembly: split, injection, consistency."""

from pathlib import Path

from rga.domain.events import EventOp
from rga.domain.replay import journal_issues, replay
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def test_dataset_journal_is_consistent() -> None:
    dataset = build_dataset(CONFIG)
    assert journal_issues(dataset.events) == []


def test_split_leaves_a_populated_training_graph() -> None:
    dataset = build_dataset(CONFIG)
    graph = replay(dataset.events, until=dataset.split_ts)
    assert graph.num_nodes > 50
    assert graph.num_edges > 50


def test_all_labels_fall_inside_the_evaluation_window() -> None:
    dataset = build_dataset(CONFIG)
    assert dataset.labels
    assert all(dataset.split_ts <= label.ts < dataset.window_end for label in dataset.labels)


def test_every_label_corresponds_to_a_real_event() -> None:
    dataset = build_dataset(CONFIG)
    granted = {event.edge_key() for event in dataset.events if event.op is EventOp.GRANT}
    assert {label.edge_key() for label in dataset.labels} <= granted


def test_anomaly_rate_stays_in_a_sane_band() -> None:
    # The configured rate is a target, not a guarantee: coverage of every pattern
    # comes first, and one burst incident creates a dozen edges. What must hold is
    # that anomalies are present and remain a minority of the window.
    dataset = build_dataset(CONFIG)
    window_grants = [
        event for event in dataset.window_events() if event.op is EventOp.GRANT
    ]
    observed = len(dataset.labels) / len(window_grants)
    assert 0.0 < observed < 0.25


def test_labelled_edges_are_not_also_produced_normally() -> None:
    # Ambiguous ground truth would silently corrupt every metric downstream.
    dataset = build_dataset(CONFIG)
    normal_keys = {
        event.edge_key()
        for event in dataset.events
        if event.op is EventOp.GRANT and event.ts < dataset.split_ts
    }
    assert not (dataset.anomaly_keys() & normal_keys)


def test_nearly_every_configured_pattern_is_represented() -> None:
    # A pattern may exhaust its candidates on a small graph, but most must land,
    # otherwise per-pattern evaluation in Module 2 has nothing to measure.
    dataset = build_dataset(CONFIG)
    present = {label.pattern for label in dataset.labels}
    assert len(present) >= len(CONFIG.anomalies.patterns) - 2


def test_generation_is_reproducible() -> None:
    assert build_dataset(CONFIG).events == build_dataset(CONFIG).events
