"""The network wearing the Scorer protocol.

Fitting trains the model on the training span and records the distributions the rank
transform needs. Scoring propagates over the whole graph at the split — the edges
held out during training were held out to stop honestly, not to be thrown away — and
combines the three terms of spec section 3.

The candidate context row is deliberately withheld from this scorer; see
`rga.nn.candidates.without_features` for the measurement that settled it. What
remains is a structural detector that needs nothing but a snapshot of the relation
tuples, which is what every ReBAC engine can supply.
"""

from __future__ import annotations

import numpy as np
import torch

from rga.domain.graph import AccessGraph
from rga.features.spec import CandidateSet
from rga.nn.candidates import CandidateArrays, candidate_arrays, without_features
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
        self._likelihood_rank: RankTransform | None = None
        self._deviation_rank: RankTransform | None = None

    def fit(self, train: CandidateSet) -> None:
        """Train on the span and record what the rank transform needs."""
        if train.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        arrays, mean, std = candidate_arrays(train)
        arrays = without_features(arrays)
        self._mean, self._std = mean, std
        self._model = train_model(
            train.graph, arrays, self._config, seed=self._seed, device=self._device
        )

        state, deviation = self._encode(train.graph)
        self._likelihood_rank = RankTransform.fit(1.0 - self._likelihood(state, arrays))
        self._deviation_rank = RankTransform.fit(deviation)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """One score per candidate in [0, 1], higher meaning more unusual."""
        if self._model is None or self._likelihood_rank is None:
            raise RuntimeError("the gnn scorer must be fit before scoring")

        if candidates.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        state, deviation = self._encode(candidates.graph)
        arrays, _, _ = candidate_arrays(candidates, mean=self._mean, std=self._std)
        unlikeliness = self._likelihood_rank.apply(
            1.0 - self._likelihood(state, without_features(arrays))
        )
        return combine(
            unlikeliness,
            self._node_rank(deviation, arrays.src),
            self._node_rank(deviation, arrays.dst),
        )

    def _encode(self, graph: AccessGraph) -> tuple[torch.Tensor, np.ndarray]:
        """Representations and profile deviations for every node of `graph`.

        Recomputed per call rather than cached: the graph a service scores is not the
        graph the model was fitted on, and a node index means something only inside
        one graph. Caching them cost nothing inside the experiment runner, where both
        spans share a graph, and would have produced nonsense anywhere else.
        """
        assert self._model is not None
        tensors = graph_tensors(graph, device=self._device)
        inputs = node_input_features(tensors)
        with torch.no_grad():
            state = self._model.encoder(inputs, tensors)
            deviation = self._model.reconstruction.deviation(state, inputs).cpu().numpy()
        return state, deviation

    def _likelihood(self, state: torch.Tensor, arrays: CandidateArrays) -> np.ndarray:
        """Probability the model assigns to each change being ordinary."""
        assert self._model is not None
        padded = torch.cat([state, torch.zeros(1, state.shape[1], device=self._device)])
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

    def _node_rank(self, deviation: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Ranked profile deviation of an endpoint, neutral where it is unknown."""
        assert self._deviation_rank is not None
        known = index >= 0
        ranks = np.full(index.shape, NEUTRAL, dtype=np.float64)
        if known.any():
            ranks[known] = self._deviation_rank.apply(deviation[index[known]])
        return ranks
