"""The network wearing the Scorer protocol.

Fitting trains the model on the training span and records the distributions the rank
transform needs. Scoring propagates over the whole graph at the split — the edges
held out during training were held out to stop honestly, not to be thrown away — and
combines the three terms of spec section 3.
"""

from __future__ import annotations

import numpy as np
import torch

from rga.features.spec import CandidateSet
from rga.nn.candidates import CandidateArrays, candidate_arrays
from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import node_input_features
from rga.nn.runtime import select_device
from rga.nn.scoring import NEUTRAL, RankTransform, combine
from rga.nn.train import GnnModel, train_model


class GnnScorer:
    """Ranks changes by how unusual the graph makes them look."""

    name = "gnn"

    def __init__(
        self,
        seed: int,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._seed = seed
        self._config = config or ModelConfig()
        self._device = device or select_device()
        self._model: GnnModel | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._state: torch.Tensor | None = None
        self._deviation: np.ndarray | None = None
        self._likelihood_rank: RankTransform | None = None
        self._deviation_rank: RankTransform | None = None

    def fit(self, train: CandidateSet) -> None:
        """Train on the span and record what the rank transform needs."""
        if train.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        arrays, mean, std = candidate_arrays(train)
        self._mean, self._std = mean, std
        self._model = train_model(
            train.graph, arrays, self._config, seed=self._seed, device=self._device
        )

        tensors = graph_tensors(train.graph, device=self._device)
        inputs = node_input_features(tensors)
        with torch.no_grad():
            self._state = self._model.encoder(inputs, tensors)
            self._deviation = (
                self._model.reconstruction.deviation(self._state, inputs).cpu().numpy()
            )

        self._likelihood_rank = RankTransform.fit(1.0 - self._likelihood(arrays))
        self._deviation_rank = RankTransform.fit(self._deviation)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """One score per candidate in [0, 1], higher meaning more unusual."""
        if self._model is None or self._likelihood_rank is None:
            raise RuntimeError("the gnn scorer must be fit before scoring")

        arrays, _, _ = candidate_arrays(candidates, mean=self._mean, std=self._std)
        unlikeliness = self._likelihood_rank.apply(1.0 - self._likelihood(arrays))
        return combine(unlikeliness, self._node_rank(arrays.src), self._node_rank(arrays.dst))

    def _likelihood(self, arrays: CandidateArrays) -> np.ndarray:
        """Probability the model assigns to each change being ordinary."""
        assert self._model is not None and self._state is not None
        padded = torch.cat(
            [self._state, torch.zeros(1, self._state.shape[1], device=self._device)]
        )
        unknown = padded.shape[0] - 1

        def endpoints(index: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.where(index < 0, unknown, index), device=self._device)

        with torch.no_grad():
            logits = self._model.likelihood(
                padded[endpoints(arrays.src)],
                padded[endpoints(arrays.dst)],
                torch.as_tensor(arrays.relation, device=self._device),
                torch.as_tensor(arrays.level, device=self._device),
                torch.as_tensor(arrays.features, device=self._device),
            )
        return torch.sigmoid(logits).cpu().numpy()

    def _node_rank(self, index: np.ndarray) -> np.ndarray:
        """Ranked profile deviation of an endpoint, neutral where it is unknown."""
        assert self._deviation is not None and self._deviation_rank is not None
        known = index >= 0
        ranks = np.full(index.shape, NEUTRAL, dtype=np.float64)
        if known.any():
            ranks[known] = self._deviation_rank.apply(self._deviation[index[known]])
        return ranks
