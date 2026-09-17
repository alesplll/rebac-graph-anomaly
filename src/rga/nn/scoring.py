"""Putting three incomparable quantities on one scale.

The likelihood term and the two profile deviations differ by orders of magnitude and
the deviations are heavily skewed, so a z-score would be dominated by whichever tail
happens to be longest. Each term is replaced by its quantile within the distribution
seen on the training span, which is scale-free and robust, and the quantiles are
averaged with equal weight — the only weighting that requires no labels to justify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The rank given where a quantity cannot be computed — an endpoint the graph has
#: never seen. Neither evidence for nor against; a new hire is not a suspect.
NEUTRAL = 0.5


@dataclass(frozen=True)
class RankTransform:
    """Maps a value to its quantile in a reference distribution."""

    reference: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> RankTransform:
        """Take the training-span distribution as the reference."""
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("cannot fit a rank transform on an empty reference")
        return cls(reference=np.sort(finite))

    def apply(self, values: np.ndarray) -> np.ndarray:
        """The share of the reference at or below each value, in [0, 1]."""
        probe = np.asarray(values, dtype=np.float64)
        positions = np.searchsorted(self.reference, probe, side="right")
        return positions / float(self.reference.size)


def combine(
    likelihood_rank: np.ndarray, subject_rank: np.ndarray, object_rank: np.ndarray
) -> np.ndarray:
    """The anomaly score: equal weights over the three ranked terms."""
    stacked = np.vstack([likelihood_rank, subject_rank, object_rank])
    return stacked.mean(axis=0)
