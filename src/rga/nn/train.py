"""Self-supervised training: real changes against corrupted ones.

Model selection never sees an anomaly label. A tenth of the training candidates are
held out, their edges are taken out of the graph the encoder propagates over, and
early stopping watches the likelihood the model gives those held-out changes. Any
label-aware criterion here would leak the ground truth into the model through the
choice of epoch, and every number measured afterwards would be worth less.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from rga.domain.graph import AccessGraph
from rga.nn.candidates import CandidateArrays, edge_positions
from rga.nn.config import ModelConfig
from rga.nn.encoder import GraphEncoder
from rga.nn.graph_tensors import graph_tensors
from rga.nn.heads import EdgeLikelihoodHead, NodeReconstructionHead
from rga.nn.negatives import sample_negatives
from rga.nn.node_inputs import NODE_INPUT_DIM, node_input_features
from rga.nn.runtime import seed_torch


class GnnModel(nn.Module):
    """The encoder and both heads, trained together."""

    def __init__(self, edge_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = GraphEncoder(NODE_INPUT_DIM, config)
        self.likelihood = EdgeLikelihoodHead(config.hidden_dim, edge_dim, config)
        self.reconstruction = NodeReconstructionHead(config.hidden_dim, NODE_INPUT_DIM, config)


def train_model(
    graph: AccessGraph,
    arrays: CandidateArrays,
    config: ModelConfig,
    *,
    seed: int,
    device: torch.device,
) -> GnnModel:
    """Fit the model on one training span and return it at its best epoch."""
    rng = np.random.default_rng(seed)
    seed_torch(seed)

    usable = np.flatnonzero((arrays.src >= 0) & (arrays.dst >= 0))
    if usable.size < 2:
        raise ValueError("not enough candidates with known endpoints to train on")
    rng.shuffle(usable)
    cut = max(1, int(usable.size * config.validation_share))
    validation, fit = usable[:cut], usable[cut:]

    positions = edge_positions(graph, arrays.src, arrays.dst, arrays.relation)
    keep = np.ones(graph.num_edges, dtype=bool)
    held = positions[validation]
    keep[held[held >= 0]] = False

    tensors = graph_tensors(graph, device=device, keep=keep)
    inputs = node_input_features(tensors)

    model = GnnModel(edge_dim=arrays.features.shape[1], config=config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(config.epochs, 1))

    features = torch.as_tensor(arrays.features, device=device)
    relation = torch.as_tensor(arrays.relation, device=device)
    level = torch.as_tensor(arrays.level, device=device)
    src = torch.as_tensor(arrays.src, device=device)
    dst = torch.as_tensor(arrays.dst, device=device)
    fit_index = torch.as_tensor(fit, device=device)
    validation_index = torch.as_tensor(validation, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_likelihood = -float("inf")
    waited = 0

    for _ in range(config.epochs):
        model.train()
        corrupted = sample_negatives(
            graph,
            arrays.src[fit],
            arrays.dst[fit],
            arrays.relation[fit],
            arrays.level[fit],
            per_edge=config.negatives_per_edge,
            rng=rng,
        )
        origin = torch.as_tensor(fit[corrupted.origin], device=device)

        h = model.encoder(inputs, tensors)
        positive = model.likelihood(
            h[src[fit_index]],
            h[dst[fit_index]],
            relation[fit_index],
            level[fit_index],
            features[fit_index],
        )
        negative = model.likelihood(
            h[torch.as_tensor(corrupted.src, device=device)],
            h[torch.as_tensor(corrupted.dst, device=device)],
            torch.as_tensor(corrupted.relation, device=device),
            torch.as_tensor(corrupted.level, device=device),
            features[origin],
        )

        loss = F.binary_cross_entropy_with_logits(
            positive, torch.ones_like(positive)
        ) + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
        loss = loss + config.reconstruction_weight * F.mse_loss(model.reconstruction(h), inputs)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()

        model.eval()
        with torch.no_grad():
            held_out = model.encoder(inputs, tensors)
            logits = model.likelihood(
                held_out[src[validation_index]],
                held_out[dst[validation_index]],
                relation[validation_index],
                level[validation_index],
                features[validation_index],
            )
            likelihood = float(F.logsigmoid(logits).mean())

        if likelihood > best_likelihood:
            best_likelihood, waited = likelihood, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config.patience:
                break

    model.load_state_dict(best_state)
    return model.eval()
