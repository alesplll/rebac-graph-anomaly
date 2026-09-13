"""Ranking metrics.

Area under the precision-recall curve is the headline number. Anomalies are a few
percent of the window, and at that imbalance the area under the ROC curve looks
flattering: the negative class is so large that even many false positives barely
move the false-positive rate. It is reported anyway, because it is what most of
the literature quotes and leaving it out invites the question.

Precision and recall at k are the operational numbers — k is the size of the queue
a person will actually work through. Lift restates precision at k against the base
rate, which is the form a reader can judge without arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Queue sizes a security analyst plausibly works through in one sitting.
DEFAULT_KS = (20, 50, 100)


@dataclass(frozen=True)
class RankingMetrics:
    """One scorer's performance on one candidate set."""

    pr_auc: float
    roc_auc: float
    precision_at: dict[int, float]
    recall_at: dict[int, float]
    lift_at: dict[int, float]
    n_candidates: int
    n_anomalies: int

    def as_row(self) -> dict[str, float]:
        """Flatten into one record for a results table."""
        row: dict[str, float] = {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "n_candidates": float(self.n_candidates),
            "n_anomalies": float(self.n_anomalies),
        }
        for k, value in self.precision_at.items():
            row[f"precision_at_{k}"] = value
        for k, value in self.recall_at.items():
            row[f"recall_at_{k}"] = value
        for k, value in self.lift_at.items():
            row[f"lift_at_{k}"] = value
        return row


def evaluate_ranking(
    y_true: np.ndarray, scores: np.ndarray, *, ks: Sequence[int] = DEFAULT_KS
) -> RankingMetrics:
    """Score a ranking of candidates against ground truth.

    With no anomalies present every rate is undefined; NaN is returned rather than
    a zero that would average into a results table as if it were measured.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    truth = np.asarray(y_true).astype(np.int8)
    total = len(truth)
    positives = int(truth.sum())

    if positives == 0:
        nan_by_k = dict.fromkeys((int(k) for k in ks), float("nan"))
        return RankingMetrics(
            pr_auc=float("nan"),
            roc_auc=float("nan"),
            precision_at=dict(nan_by_k),
            recall_at=dict(nan_by_k),
            lift_at=dict(nan_by_k),
            n_candidates=total,
            n_anomalies=0,
        )

    from sklearn.metrics import average_precision_score, roc_auc_score

    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranked = truth[order]
    base_rate = positives / total

    precision_at: dict[int, float] = {}
    recall_at: dict[int, float] = {}
    lift_at: dict[int, float] = {}
    for k in ks:
        cut = min(int(k), total)
        hits = int(ranked[:cut].sum())
        precision = hits / cut
        precision_at[int(k)] = precision
        recall_at[int(k)] = hits / positives
        lift_at[int(k)] = precision / base_rate

    return RankingMetrics(
        pr_auc=float(average_precision_score(truth, scores)),
        roc_auc=float(roc_auc_score(truth, scores)),
        precision_at=precision_at,
        recall_at=recall_at,
        lift_at=lift_at,
        n_candidates=total,
        n_anomalies=positives,
    )


def recall_by_pattern(
    y_true: np.ndarray, scores: np.ndarray, patterns: Sequence[str], *, k: int
) -> dict[str, float]:
    """Share of each pattern's edges that reach the top k.

    This is the breakdown section 9 asks for: which threats the system catches and
    which it misses. An aggregate number hides a detector that is excellent at one
    loud pattern and blind to the rest.
    """
    truth = np.asarray(y_true).astype(bool)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    top = set(order[: min(int(k), len(order))].tolist())

    totals: dict[str, int] = {}
    found: dict[str, int] = {}
    for index, (is_anomaly, pattern) in enumerate(zip(truth, patterns, strict=True)):
        if not is_anomaly or not pattern:
            continue
        totals[pattern] = totals.get(pattern, 0) + 1
        if index in top:
            found[pattern] = found.get(pattern, 0) + 1

    return {pattern: found.get(pattern, 0) / count for pattern, count in totals.items()}
