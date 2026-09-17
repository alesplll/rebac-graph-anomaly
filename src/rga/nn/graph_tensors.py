"""The access graph as the index tensors the layer consumes.

Each (relation, direction) pair is a slot with its own weight matrix in the layer,
because membership in a group and a right on a bucket are not the same kind of
evidence and must not be summed into one neighbourhood.

Only structure is read here: endpoints, relation type and ordinal level. Creation
times and initiators are deliberately left behind — they reach the model through the
candidate feature matrix, where the ablation study can mask them. A layer that read
them from the graph would make the `structural` row of that study a lie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rga.domain.graph import AccessGraph
from rga.domain.relations import RelationType

#: (relation, incoming). Messages on an incoming slot travel against the edge.
SLOTS: tuple[tuple[RelationType, bool], ...] = tuple(
    (relation, incoming) for relation in RelationType for incoming in (False, True)
)
NUM_SLOTS = len(SLOTS)


@dataclass(frozen=True)
class GraphTensors:
    """Index tensors for one graph, one entry per slot."""

    num_nodes: int
    node_type: torch.Tensor
    #: Where each message comes from, per slot.
    src: tuple[torch.Tensor, ...]
    #: Where each message goes, per slot.
    dst: tuple[torch.Tensor, ...]
    #: Ordinal permission level of the edge carrying the message, per slot.
    level: tuple[torch.Tensor, ...]
    #: 1/sqrt(degree) for every receiving node, per slot.
    norm: tuple[torch.Tensor, ...]

    @property
    def device(self) -> torch.device:
        return self.node_type.device


def graph_tensors(
    graph: AccessGraph, *, device: torch.device, keep: np.ndarray | None = None
) -> GraphTensors:
    """Slice a graph into per-slot index tensors.

    `keep` is a boolean mask over edges; the edges it excludes take no part in
    propagation. Training uses it to hold out the edges it validates on.
    """
    selected = (
        np.ones(graph.num_edges, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    )
    if selected.shape != (graph.num_edges,):
        raise ValueError(f"keep must have {graph.num_edges} entries, got {selected.shape}")

    src: list[torch.Tensor] = []
    dst: list[torch.Tensor] = []
    level: list[torch.Tensor] = []
    norm: list[torch.Tensor] = []

    for relation, incoming in SLOTS:
        mask = selected & (graph.edge_rel == int(relation))
        tail = graph.edge_dst[mask] if incoming else graph.edge_src[mask]
        head = graph.edge_src[mask] if incoming else graph.edge_dst[mask]

        degree = np.bincount(head, minlength=graph.num_nodes).astype(np.float32)
        src.append(torch.as_tensor(tail.astype(np.int64), device=device))
        dst.append(torch.as_tensor(head.astype(np.int64), device=device))
        level.append(torch.as_tensor(graph.edge_level[mask].astype(np.int64), device=device))
        norm.append(torch.as_tensor(1.0 / np.sqrt(np.maximum(degree, 1.0)), device=device))

    return GraphTensors(
        num_nodes=graph.num_nodes,
        node_type=torch.as_tensor(graph.node_type.astype(np.int64), device=device),
        src=tuple(src),
        dst=tuple(dst),
        level=tuple(level),
        norm=tuple(norm),
    )
