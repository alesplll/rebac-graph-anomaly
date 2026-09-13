"""Ranking metrics and the per-pattern breakdown."""

import numpy as np
import pytest

from rga.eval.metrics import evaluate_ranking, recall_by_pattern


def test_perfect_ranking_scores_one() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.05, 0.01])
    metrics = evaluate_ranking(y_true, scores, ks=(2,))
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.recall_at[2] == pytest.approx(1.0)
    assert metrics.precision_at[2] == pytest.approx(1.0)


def test_reversed_ranking_scores_poorly() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.01, 0.05, 0.8, 0.9, 0.7, 0.6])
    metrics = evaluate_ranking(y_true, scores, ks=(2,))
    assert metrics.roc_auc < 0.5
    assert metrics.recall_at[2] == pytest.approx(0.0)


def test_lift_compares_against_the_base_rate() -> None:
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.arange(10, 0, -1, dtype=float)
    metrics = evaluate_ranking(y_true, scores, ks=(2,))
    # Base rate 0.1, precision@2 is 0.5, so the queue is five times denser.
    assert metrics.lift_at[2] == pytest.approx(5.0)


def test_k_larger_than_the_candidate_pool_is_clipped() -> None:
    y_true = np.array([1, 0, 0])
    scores = np.array([0.9, 0.5, 0.1])
    metrics = evaluate_ranking(y_true, scores, ks=(100,))
    assert metrics.recall_at[100] == pytest.approx(1.0)
    assert metrics.precision_at[100] == pytest.approx(1.0 / 3.0)


def test_counts_are_reported() -> None:
    y_true = np.array([1, 1, 0, 0])
    metrics = evaluate_ranking(y_true, np.array([0.4, 0.3, 0.2, 0.1]), ks=(2,))
    assert metrics.n_candidates == 4
    assert metrics.n_anomalies == 2


def test_a_window_without_anomalies_yields_nan_rather_than_a_lie() -> None:
    metrics = evaluate_ranking(np.zeros(5, dtype=int), np.arange(5, dtype=float), ks=(2,))
    assert np.isnan(metrics.pr_auc)
    assert np.isnan(metrics.roc_auc)
    assert np.isnan(metrics.recall_at[2])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_ranking(np.array([1, 0]), np.array([0.5]))


def test_as_row_flattens_for_a_results_table() -> None:
    metrics = evaluate_ranking(np.array([1, 0, 0, 0]), np.array([0.9, 0.3, 0.2, 0.1]), ks=(2,))
    row = metrics.as_row()
    assert row["pr_auc"] == pytest.approx(1.0)
    assert "precision_at_2" in row
    assert "lift_at_2" in row


def test_recall_by_pattern_splits_the_queue() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    patterns = ("shadow_group", "grant_burst", "", "")
    found = recall_by_pattern(y_true, scores, patterns, k=2)
    assert found["shadow_group"] == pytest.approx(1.0)
    assert found["grant_burst"] == pytest.approx(0.0)


def test_recall_by_pattern_ignores_normal_rows() -> None:
    y_true = np.array([1, 0])
    scores = np.array([0.9, 0.8])
    found = recall_by_pattern(y_true, scores, ("self_grant_admin", ""), k=1)
    assert set(found) == {"self_grant_admin"}
