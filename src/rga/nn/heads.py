"""What sits on top of the encoder.

The likelihood head scores one change: both endpoint representations, their
elementwise product, the relation and level embeddings, and the candidate's own
feature row. The feature row is what makes the network sensitive to the capability
level of the source — without it the model would see nothing but structure and the
feature-group study would have nothing to say about it.

The reconstruction head rebuilds a node's structural profile from its
representation. The squared error is how much that node departs from what the graph
around it would lead one to expect.
"""

from __future__ import annotations

import torch
from torch import nn

from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig


class EdgeLikelihoodHead(nn.Module):
    """A logit per change: high means ordinary."""

    def __init__(self, node_dim: int, edge_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.relation = nn.Embedding(len(RelationType) + 1, config.embedding_dim)
        self.level = nn.Embedding(len(PermissionLevel), config.embedding_dim)
        width = 3 * node_dim + 2 * config.embedding_dim + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(width, config.edge_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_hidden, 1),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        relation: torch.Tensor,
        level: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Logits for a batch of changes."""
        parts = [
            h_src,
            h_dst,
            h_src * h_dst,
            self.relation(relation),
            self.level(level),
            edge_features,
        ]
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


class NodeReconstructionHead(nn.Module):
    """Rebuilds the structural profile of a node from its representation."""

    def __init__(self, node_dim: int, output_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, config.edge_hidden),
            nn.GELU(),
            nn.Linear(config.edge_hidden, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """The reconstructed profile of every node."""
        return self.mlp(h)

    def deviation(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared reconstruction error per node."""
        return (self.mlp(h) - target).pow(2).mean(dim=1)
