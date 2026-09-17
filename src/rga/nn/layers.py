"""One round of relational message passing, written by hand.

    h_v' = LayerNorm(h_v + Dropout(gelu(W_self h_v + Sum_r c_{v,r}^-1 Sum_u m_{u->v,r})))
    m_{u->v,r} = W_r h_u + W_lvl emb(level_uv)

Hand-written for two reasons. HAS_PERMISSION carries an ordinal level that belongs in
the message, and no stock relational layer models that. And the libraries that would
supply the aggregation need torch-scatter and torch-sparse, which build from source
and would have to do so on both machines this project runs on; `index_add_` is one
line and behaves identically on CPU and GPU.

The eight per-slot matrices are composed from a shared basis: at this graph size,
eight independent matrices of hidden by hidden would have more parameters than the
graph has edges.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from rga.domain.relations import PermissionLevel
from rga.nn.graph_tensors import NUM_SLOTS, GraphTensors


class RelationalLayer(nn.Module):
    """Propagates node representations one hop across every slot."""

    def __init__(
        self, in_dim: int, out_dim: int, *, num_bases: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.basis = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coefficients = nn.Parameter(torch.empty(NUM_SLOTS, num_bases))
        self.level = nn.Embedding(len(PermissionLevel), out_dim)
        self.residual = (
            nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier on the basis, small noise on the level embedding."""
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.coefficients)
        nn.init.normal_(self.level.weight, std=0.02)

    def forward(self, h: torch.Tensor, tensors: GraphTensors) -> torch.Tensor:
        """One hop of propagation over every slot of `tensors`."""
        weights = torch.einsum("sb,bio->sio", self.coefficients, self.basis)
        total = self.self_loop(h)

        for slot in range(NUM_SLOTS):
            source = tensors.src[slot]
            if source.numel() == 0:
                continue
            messages = h[source] @ weights[slot] + self.level(tensors.level[slot])
            gathered = torch.zeros_like(total)
            gathered.index_add_(0, tensors.dst[slot], messages)
            total = total + gathered * tensors.norm[slot].unsqueeze(1)

        return self.norm(self.residual(h) + self.dropout(F.gelu(total)))
