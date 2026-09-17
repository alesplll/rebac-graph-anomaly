"""The candidate set carries the graph the network propagates over."""

from pathlib import Path

from rga.eval.experiment import restrict_candidates
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def test_both_spans_carry_a_graph_of_the_same_shape() -> None:
    dataset = build_dataset(CONFIG)

    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)

    assert train.graph is not None
    assert evaluation.graph is not None
    assert train.graph.num_edges == evaluation.graph.num_edges
    assert train.graph.num_nodes == evaluation.graph.num_nodes


def test_the_graph_holds_nothing_from_the_evaluation_window() -> None:
    dataset = build_dataset(CONFIG)

    graph = build_candidates(dataset, Span.EVAL).graph

    assert graph is not None
    assert int(graph.edge_created.max()) <= dataset.split_ts


def test_restricting_feature_groups_keeps_the_graph() -> None:
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)

    narrowed = restrict_candidates(train, (FeatureGroup.STRUCTURAL,))

    assert narrowed.graph is train.graph
