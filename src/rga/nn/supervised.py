"""The same encoder, trained as a classifier on labelled incidents.

It exists to be beaten in one specific way. On the patterns it was trained on it
should do very well, and on the three it has never seen it is expected to fall over.
The self-supervised model never sees a label, so nothing is familiar to it and
nothing is unfamiliar either — which is the comparison spec section 8 is after.

Incidents are a percent or so of the training span, so the positive class is weighted
up rather than resampled; resampling would change the graph the encoder propagates
over, and the two models must see the same graph for the comparison to mean anything.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from rga.domain.graph import AccessGraph
from rga.features.spec import CandidateSet
from rga.nn.candidates import candidate_arrays
from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import node_input_features
from rga.nn.runtime import seed_torch, select_device
from rga.nn.train import GnnModel


class SupervisedGnnScorer:
    """Ranks changes by a classifier trained on labelled incidents."""

    name = "gnn_supervised"

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


    @property
    def seed(self) -> int:
        return self._seed

    @property
    def config(self) -> ModelConfig:
        return self._config

    def state_for_artifact(self) -> dict[str, object]:
        """Everything a reloaded copy needs, as plain arrays and numbers."""
        if self._model is None:
            raise RuntimeError("the supervised gnn scorer must be fit before saving")
        assert self._mean is not None and self._std is not None
        return {
            "weights": self._model.state_dict(),
            "edge_dim": int(self._model.likelihood.mlp[0].in_features)
            - 3 * self._config.hidden_dim
            - 2 * self._config.embedding_dim,
            "mean": self._mean,
            "std": self._std,
        }

    def restore_from_artifact(self, state: dict[str, object]) -> None:
        """Rebuild a fitted scorer from `state_for_artifact`."""
        model = GnnModel(edge_dim=int(state["edge_dim"]), config=self._config)  # type: ignore[arg-type]
        model.load_state_dict(state["weights"])  # type: ignore[arg-type]
        self._model = model.to(self._device).eval()
        self._mean = np.asarray(state["mean"])
        self._std = np.asarray(state["std"])

    def fit(self, train: CandidateSet) -> None:
        """Train the classifier on the labelled training span."""
        if train.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")
        positives = int(train.labels.sum())
        if positives == 0:
            raise ValueError(
                "the supervised baseline needs labelled incidents in the training span; "
                "use a dataset recipe with train_patterns, such as small-history.yaml"
            )

        seed_torch(self._seed)
        arrays, mean, std = candidate_arrays(train)
        self._mean, self._std = mean, std

        tensors = graph_tensors(train.graph, device=self._device)
        inputs = node_input_features(tensors)
        model = GnnModel(edge_dim=arrays.features.shape[1], config=self._config).to(self._device)
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        usable = np.flatnonzero((arrays.src >= 0) & (arrays.dst >= 0))
        index = torch.as_tensor(usable, device=self._device)
        target = torch.as_tensor(train.labels[usable].astype(np.float32), device=self._device)
        weight = torch.tensor(
            [max(len(usable) - positives, 1) / max(positives, 1)], device=self._device
        )

        features = torch.as_tensor(arrays.features, device=self._device)
        relation = torch.as_tensor(arrays.relation, device=self._device)
        level = torch.as_tensor(arrays.level, device=self._device)
        src = torch.as_tensor(arrays.src, device=self._device)
        dst = torch.as_tensor(arrays.dst, device=self._device)

        best_state = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        waited = 0

        for _ in range(self._config.epochs):
            model.train()
            h = model.encoder(inputs, tensors)
            logits = model.likelihood(
                h[src[index]], h[dst[index]], relation[index], level[index], features[index]
            )
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            current = float(loss.detach())
            if current < best_loss - 1e-4:
                best_loss, waited = current, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= self._config.patience:
                    break

        model.load_state_dict(best_state)
        self._model = model.eval()

    def _encode(self, graph: AccessGraph) -> torch.Tensor:
        """Representations for every node of `graph`, computed per call.

        A node index means something only inside one graph, so the representations of
        the training graph cannot be reused for a graph the service was pointed at.
        """
        assert self._model is not None
        tensors = graph_tensors(graph, device=self._device)
        with torch.no_grad():
            return self._model.encoder(node_input_features(tensors), tensors)

    def feature_gradients(self, candidates: CandidateSet, position: int) -> np.ndarray:
        """d(score)/d(feature) times the feature, over the dense matrix columns."""
        if self._model is None:
            raise RuntimeError("the supervised gnn scorer must be fit before explaining")
        if candidates.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        single = candidates.row(position)
        arrays, _, _ = candidate_arrays(single, mean=self._mean, std=self._std)
        state = self._encode(single.graph)

        padded = torch.cat([state, torch.zeros(1, state.shape[1], device=self._device)])
        unknown = padded.shape[0] - 1

        def endpoints(index: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.where(index < 0, unknown, index), device=self._device)

        features = torch.as_tensor(arrays.features, device=self._device).requires_grad_(True)
        logit = self._model.likelihood(
            padded[endpoints(arrays.src)],
            padded[endpoints(arrays.dst)],
            torch.as_tensor(arrays.relation, device=self._device),
            torch.as_tensor(arrays.level, device=self._device),
            features,
        )
        # The logit, not the probability: the sigmoid saturates on exactly the
        # candidates an analyst opens first, and a saturated sigmoid has no
        # gradient to attribute with.
        logit.sum().backward()
        assert features.grad is not None
        return (features.grad * features.detach()).squeeze(0).cpu().numpy()

    def margins(self, candidates: CandidateSet) -> np.ndarray:
        """The raw logits behind the scores.

        Ranking is unaffected — the sigmoid is monotone — but explanation is not:
        differences between saturated probabilities vanish into float error, while
        the logits behind them stay apart.
        """
        return self._logits(candidates).cpu().numpy()

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """Probability the classifier assigns to a change being an incident."""
        return torch.sigmoid(self._logits(candidates)).cpu().numpy()

    def _logits(self, candidates: CandidateSet) -> torch.Tensor:
        """The head's output for every candidate, before the sigmoid."""
        if self._model is None:
            raise RuntimeError("the supervised gnn scorer must be fit before scoring")
        if candidates.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        arrays, _, _ = candidate_arrays(candidates, mean=self._mean, std=self._std)
        state = self._encode(candidates.graph)
        padded = torch.cat([state, torch.zeros(1, state.shape[1], device=self._device)])
        unknown = padded.shape[0] - 1

        def endpoints(index: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.where(index < 0, unknown, index), device=self._device)

        with torch.no_grad():
            return self._model.likelihood(
                padded[endpoints(arrays.src)],
                padded[endpoints(arrays.dst)],
                torch.as_tensor(arrays.relation, device=self._device),
                torch.as_tensor(arrays.level, device=self._device),
                torch.as_tensor(arrays.features, device=self._device),
            )
