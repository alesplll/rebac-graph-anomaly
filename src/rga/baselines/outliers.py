"""Classical outlier detection on the candidate features.

Both estimators see exactly what the network will see in Module 3, minus any
propagation across the graph. That is the point of the comparison: whatever gap
opens up is attributable to graph aggregation, not to a richer feature set.

Both are fitted on the training span and applied to the evaluation window, never
fitted on the window itself — that would be the same leak the temporal split
exists to prevent.

scikit-learn is imported lazily so the package stays importable without the `ml`
extra installed.
"""

from __future__ import annotations

import numpy as np

from rga.baselines.base import dense_matrix
from rga.features.spec import CandidateSet


class _ScaledEstimator:
    """Shared plumbing: standardise on the training span, then score."""

    name = "unset"

    def __init__(self) -> None:
        self._scaler = None
        self._estimator = None

    def _build(self):
        raise NotImplementedError

    def fit(self, train: CandidateSet) -> None:
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        features = self._scaler.fit_transform(dense_matrix(train.matrix))
        self._estimator = self._build()
        self._estimator.fit(features)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        if self._estimator is None or self._scaler is None:
            raise RuntimeError(f"{self.name} must be fit before scoring")
        features = self._scaler.transform(dense_matrix(candidates.matrix))
        # Both estimators return higher values for more normal points; the
        # project's convention is the opposite, so the sign is flipped.
        return -np.asarray(self._estimator.score_samples(features), dtype=np.float64)


class IsolationForestScorer(_ScaledEstimator):
    """Isolates points by random splits: the fewer splits needed, the odder."""

    name = "isolation_forest"

    def __init__(self, seed: int = 0, n_estimators: int = 200) -> None:
        super().__init__()
        self._seed = seed
        self._n_estimators = n_estimators

    def _build(self):
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=self._n_estimators, random_state=self._seed, n_jobs=-1
        )


class LocalOutlierFactorScorer(_ScaledEstimator):
    """Compares a point's local density with its neighbours'.

    `novelty=True` is required to score points the estimator was not fitted on,
    which is exactly our protocol: fit on the training span, score the window.
    """

    name = "lof"

    def __init__(self, n_neighbors: int = 20) -> None:
        super().__init__()
        self._n_neighbors = n_neighbors

    def _build(self):
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(n_neighbors=self._n_neighbors, novelty=True, n_jobs=-1)
