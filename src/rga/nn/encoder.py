"""Stacked relational layers: node inputs in, node representations out."""

from __future__ import annotations

import torch
from torch import nn

from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import GraphTensors
from rga.nn.layers import RelationalLayer


class GraphEncoder(nn.Module):
    """Projects structural inputs and propagates them over the graph."""

    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            RelationalLayer(
                config.hidden_dim,
                config.hidden_dim,
                num_bases=config.num_bases,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )

    def forward(self, x: torch.Tensor, tensors: GraphTensors) -> torch.Tensor:
        """Representations of every node after the full stack."""
        h = self.input(x)
        for layer in self.layers:
            h = layer(h, tensors)
        return h
