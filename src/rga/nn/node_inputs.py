"""The starting vector of every node, read off the graph structure.

Four columns for the node kind, one per slot for how connected it is, and two for
the strongest right it holds and the strongest right held over it. Degrees go
through log1p because they are heavy-tailed: the difference between two and three
group memberships means far more than the difference between two hundred and two
hundred and one.

Nothing here reads a timestamp or an initiator. This vector is also the target of
the reconstruction head, so "the profile of this node is unusual" is measured
against exactly these quantities.
"""

from __future__ import annotations

import torch

from rga.domain.entities import EntityType
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import NUM_SLOTS, SLOTS, GraphTensors

_NUM_TYPES = len(EntityType)
_MAX_LEVEL = int(PermissionLevel.ADMIN)
#: type one-hot + degree per slot + level held + level held over
NODE_INPUT_DIM = _NUM_TYPES + NUM_SLOTS + 2


def node_input_features(tensors: GraphTensors) -> torch.Tensor:
    """Structural attributes of every node, as one float32 matrix."""
    device = tensors.device
    count = tensors.num_nodes

    kind = torch.zeros(count, _NUM_TYPES, device=device)
    # EntityType values start at 1; column 0 is USER.
    kind.scatter_(1, (tensors.node_type - 1).clamp(min=0).unsqueeze(1), 1.0)

    degrees = torch.zeros(count, NUM_SLOTS, device=device)
    for slot in range(NUM_SLOTS):
        counts = torch.zeros(count, device=device)
        arriving = torch.ones_like(tensors.dst[slot], dtype=torch.float32)
        counts.index_add_(0, tensors.dst[slot], arriving)
        degrees[:, slot] = torch.log1p(counts)

    held = torch.zeros(count, device=device)
    held_over = torch.zeros(count, device=device)
    for slot, (relation, incoming) in enumerate(SLOTS):
        if relation is not RelationType.HAS_PERMISSION or incoming:
            continue
        levels = tensors.level[slot]
        # Ascending order makes the last write the maximum. index_reduce_ would say
        # this in one call but is still a beta API, and the two machines run
        # different torch versions.
        for value in range(1, _MAX_LEVEL + 1):
            at_value = levels == value
            if not bool(at_value.any()):
                continue
            held[tensors.src[slot][at_value]] = float(value)
            held_over[tensors.dst[slot][at_value]] = float(value)

    return torch.cat([kind, degrees, held.unsqueeze(1), held_over.unsqueeze(1)], dim=1)


def neighbourhood_targets(tensors: GraphTensors, inputs: torch.Tensor) -> torch.Tensor:
    """The mean attribute vector of every node's undirected neighbourhood.

    Reconstructing a node's own profile turned out to measure how central it is
    rather than how strange: the hardest rows to rebuild are the busy ones, and busy
    nodes produce most of the ordinary changes. Averaging the surroundings removes
    the node's own degree from its target.

    A node with no neighbours keeps its own vector: there is nothing to average, and
    a row of zeros would make every isolate look equally odd.
    """
    total = torch.zeros_like(inputs)
    seen = torch.zeros(tensors.num_nodes, device=tensors.device)
    for slot in range(NUM_SLOTS):
        source, target = tensors.src[slot], tensors.dst[slot]
        total.index_add_(0, target, inputs[source])
        seen.index_add_(0, target, torch.ones_like(target, dtype=torch.float32))
    averaged = total / seen.clamp(min=1.0).unsqueeze(1)
    return torch.where((seen == 0).unsqueeze(1), inputs, averaged)


def reconstruction_target(
    tensors: GraphTensors, inputs: torch.Tensor, config: ModelConfig
) -> torch.Tensor:
    """What the reconstruction head is asked to rebuild."""
    if config.reconstruction_target == "profile":
        return inputs
    if config.reconstruction_target == "neighbourhood":
        return neighbourhood_targets(tensors, inputs)
    raise ValueError(f"unknown reconstruction target: {config.reconstruction_target!r}")
