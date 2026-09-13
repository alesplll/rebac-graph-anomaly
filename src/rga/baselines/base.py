"""What every scorer implements, and how a masked matrix becomes a dense one."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rga.features.spec import CandidateSet, FeatureGroup, FeatureMatrix


@runtime_checkable
class Scorer(Protocol):
    """Assigns an anomaly score to every candidate.

    Higher means more suspicious. Scorers that need no fitting implement `fit` as
    a no-op so the experiment runner can treat them all alike.
    """

    name: str

    def fit(self, train: CandidateSet) -> None:
        """Learn whatever the scorer needs from the training span."""
        ...

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """One score per candidate, higher meaning more anomalous."""
        ...


def dense_matrix(matrix: FeatureMatrix) -> np.ndarray:
    """Flatten values and mask into something a classical estimator accepts.

    Unobserved entries become zero, which on its own would be a lie — zero is a
    legitimate value for most of these features. So one column per group is
    appended holding the share of that group's features actually observed in that
    row, which lets the estimator tell a genuine zero from a gap.
    """
    values = np.where(matrix.mask, matrix.values, 0.0).astype(np.float64)

    coverage = []
    for group in FeatureGroup:
        columns = matrix.group_indices(group)
        if columns.size:
            coverage.append(matrix.mask[:, columns].mean(axis=1))
        else:
            coverage.append(np.zeros(matrix.n_rows))

    return np.hstack([values, np.column_stack(coverage)])
