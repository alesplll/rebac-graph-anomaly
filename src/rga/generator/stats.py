"""Dataset statistics.

These numbers are the acceptance check for a generated dataset. A graph whose
degree distribution is flat, or whose anomalies all come from one pattern, is not
a usable experiment input however cleanly the code ran.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from rga.domain.entities import EntityType
from rga.domain.events import EventOp
from rga.domain.relations import RelationType
from rga.domain.replay import replay
from rga.generator.dataset import Dataset


def dataset_stats(dataset: Dataset) -> dict[str, object]:
    """Summarise a dataset."""
    graph = replay(dataset.events, until=dataset.split_ts)
    degrees = np.bincount(graph.edge_src, minlength=graph.num_nodes)
    window_grants = [event for event in dataset.window_events() if event.op is EventOp.GRANT]

    return {
        "name": dataset.config.name,
        "seed": dataset.config.seed,
        "events": len(dataset.events),
        "train_nodes": graph.num_nodes,
        "train_edges": graph.num_edges,
        "nodes_by_type": {
            entity.name: int((graph.node_type == int(entity)).sum()) for entity in EntityType
        },
        "edges_by_relation": {
            relation.name: int((graph.edge_rel == int(relation)).sum())
            for relation in RelationType
        },
        "out_degree_mean": float(degrees.mean()) if graph.num_nodes else 0.0,
        "out_degree_max": int(degrees.max()) if graph.num_nodes else 0,
        "window_grants": len(window_grants),
        "anomalies": len(dataset.labels),
        "anomaly_rate": (len(dataset.labels) / len(window_grants)) if window_grants else 0.0,
        "anomalies_by_pattern": dict(Counter(label.pattern for label in dataset.labels)),
    }


def format_stats(stats: dict[str, object]) -> str:
    """Render statistics as aligned lines."""
    width = max(len(key) for key in stats)
    lines = []
    for key, value in stats.items():
        if isinstance(value, dict):
            lines.append(f"{key:<{width}} :")
            inner = max((len(str(name)) for name in value), default=0)
            lines.extend(f"  {name!s:<{inner}} : {count}" for name, count in value.items())
        elif isinstance(value, float):
            lines.append(f"{key:<{width}} : {value:.4f}")
        else:
            lines.append(f"{key:<{width}} : {value}")
    return "\n".join(lines)
