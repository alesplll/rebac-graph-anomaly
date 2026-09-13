"""Dataset statistics — the acceptance check for a generated dataset."""

from pathlib import Path

from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.generator.stats import dataset_stats, format_stats


def _stats():
    return dataset_stats(build_dataset(load_dataset_config(Path("configs/generator/small.yaml"))))


def test_reports_graph_size_at_the_split() -> None:
    stats = _stats()
    assert stats["train_nodes"] > 50
    assert stats["train_edges"] > 50


def test_reports_counts_per_node_type_and_relation() -> None:
    stats = _stats()
    assert set(stats["nodes_by_type"]) <= {"USER", "GROUP", "BUCKET", "OBJECT"}
    assert "HAS_PERMISSION" in stats["edges_by_relation"]


def test_reports_anomalies_per_pattern() -> None:
    stats = _stats()
    per_pattern = stats["anomalies_by_pattern"]
    assert isinstance(per_pattern, dict)
    assert sum(per_pattern.values()) == stats["anomalies"]


def test_the_graph_has_hubs() -> None:
    # A flat degree distribution would mean an unrealistic graph, and the whole
    # premise of using a GNN is that neighbourhood structure varies.
    stats = _stats()
    assert stats["out_degree_max"] > 5 * stats["out_degree_mean"]


def test_formats_without_raising() -> None:
    assert "train_nodes" in format_stats(_stats())
