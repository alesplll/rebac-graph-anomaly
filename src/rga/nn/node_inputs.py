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
