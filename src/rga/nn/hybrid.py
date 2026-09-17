"""The classical detector, given the network's structural view as one more column.

Measured separately, the two see different things. `IsolationForest` reads the
candidate's own context — the hour it happened, whether the initiator granted it to
himself, how the level jumped — and ranks well on the patterns that leave a trace
there. The self-supervised network reads nothing but the shape of the graph, and
finds what hand-written conditions and the forest both struggle with: a membership
that quietly joins a privileged group.

So the question worth measuring is not which of them wins but whether the structural
view adds anything on top. This scorer answers it: the same 70 features the forest
already had, plus a single column holding the network's score, and nothing else
changed. No labels are consulted at any point.
"""

from __future__ import annotations

import numpy as np
import torch

from rga.baselines.base import dense_matrix
from rga.features.spec import CandidateSet
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer

#: Matches the forest baseline, so the comparison isolates the added column.
_TREES = 200


class GnnAugmentedForestScorer:
    """Isolation forest over the candidate features plus the network's score."""

    name = "gnn_forest"

    def __init__(
        self,
        seed: int,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._seed = seed
        self._network = GnnScorer(seed=seed, config=config, device=device)
        self._scaler = None
        self._forest = None

    def fit(self, train: CandidateSet) -> None:
        """Train the network, then the forest over the widened feature space."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        self._network.fit(train)
        self._scaler = StandardScaler()
        widened = self._scaler.fit_transform(self._widen(train))
        self._forest = IsolationForest(n_estimators=_TREES, random_state=self._seed, n_jobs=-1)
        self._forest.fit(widened)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """One score per candidate, higher meaning more unusual."""
        if self._forest is None or self._scaler is None:
            raise RuntimeError("the hybrid scorer must be fit before scoring")

        widened = self._scaler.transform(self._widen(candidates))
        # The forest returns higher values for more normal points; the project's
        # convention is the opposite, so the sign is flipped.
        return -np.asarray(self._forest.score_samples(widened), dtype=np.float64)

    def _widen(self, candidates: CandidateSet) -> np.ndarray:
        """The usual feature matrix with the network's score appended."""
        structural = self._network.score(candidates).reshape(-1, 1)
        return np.hstack([dense_matrix(candidates.matrix), structural])
