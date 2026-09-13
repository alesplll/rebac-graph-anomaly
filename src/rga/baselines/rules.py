"""Hand-written heuristics.

This is the "could you not just grep the audit log" baseline. The conditions are
the obvious ones an engineer would write in an afternoon, and the weights are
picked by hand rather than fitted — fitting them would make this a model, and the
point of the comparison is that it is not one.

It doubles as the acceptance test for the generator: if these conditions rank
nearly perfectly, the synthetic data is too easy and every number measured above
it is worthless.
"""

from __future__ import annotations

import numpy as np

from rga.domain.relations import PermissionLevel
from rga.features.spec import CandidateSet, FeatureMatrix

#: Condition name to weight. Weights are deliberately round numbers.
_WEIGHTS = {
    "self_grant_elevated": 3.0,
    "bypasses_bucket": 2.0,
    "big_level_jump": 2.0,
    "structurally_isolated": 2.0,
    "off_hours": 1.0,
    "weekend": 1.0,
    "new_subject": 1.0,
}

#: A grant at or above this level counts as elevated.
_ELEVATED = float(PermissionLevel.WRITE) / float(PermissionLevel.ADMIN)
#: A level increase of this much or more counts as a jump.
_BIG_JUMP = 2.0 / float(PermissionLevel.ADMIN)


def _values(matrix: FeatureMatrix, name: str) -> np.ndarray:
    """A feature's values, or zeros when the source does not supply it.

    Under a restricted feature set — a level-0 engine that records no timestamps,
    say — a condition resting on an absent feature cannot fire. Silence is the
    honest answer; raising would make the ablation study impossible to run.
    """
    if name not in matrix.block.names:
        return np.zeros(matrix.n_rows, dtype=np.float64)
    return matrix.column(name).astype(np.float64)


def _is_observed(matrix: FeatureMatrix, name: str) -> np.ndarray:
    """Whether a feature was observed, or all-false when it is absent entirely."""
    if name not in matrix.block.names:
        return np.zeros(matrix.n_rows, dtype=bool)
    return matrix.observed(name)


class RuleScorer:
    """Weighted sum of hand-written conditions, rescaled to [0, 1]."""

    name = "rules"

    def fit(self, train: CandidateSet) -> None:
        """Nothing is learned. Present so the runner can treat scorers alike."""

    def score(self, candidates: CandidateSet) -> np.ndarray:
        matrix = candidates.matrix

        # A condition resting on an unobserved feature must not fire. Silence is
        # the honest answer when the source cannot tell us.
        self_grant = (
            _values(matrix, "actor_is_subject")
            * _is_observed(matrix, "actor_is_subject")
            * (_values(matrix, "level_ordinal") >= _ELEVATED)
        )
        bypass = _values(matrix, "bypasses_bucket") * _is_observed(matrix, "bypasses_bucket")

        conditions = {
            "self_grant_elevated": self_grant,
            "bypasses_bucket": bypass,
            "big_level_jump": (_values(matrix, "level_jump") >= _BIG_JUMP).astype(np.float64),
            "structurally_isolated": _values(matrix, "path_unreachable"),
            "off_hours": _values(matrix, "is_off_hours"),
            "weekend": _values(matrix, "is_weekend"),
            "new_subject": _values(matrix, "subj_is_new"),
        }

        total = np.zeros(matrix.n_rows, dtype=np.float64)
        for name, fired in conditions.items():
            total += _WEIGHTS[name] * np.asarray(fired, dtype=np.float64)

        return total / sum(_WEIGHTS.values())
