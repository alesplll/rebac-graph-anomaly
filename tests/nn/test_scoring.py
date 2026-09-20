"""Rank transform and the anomaly score built out of ranked terms."""

import numpy as np
import pytest

from rga.nn.scoring import NEUTRAL, RankTransform, combine


def test_the_transform_maps_the_reference_onto_the_unit_interval() -> None:
    transform = RankTransform.fit(np.arange(100.0))

    ranks = transform.apply(np.array([0.0, 50.0, 99.0]))

    assert ranks.min() >= 0.0
    assert ranks.max() <= 1.0
    assert ranks[0] < ranks[1] < ranks[2]


def test_values_beyond_the_reference_saturate() -> None:
    transform = RankTransform.fit(np.arange(100.0))

    assert transform.apply(np.array([1e9]))[0] == 1.0
    assert transform.apply(np.array([-1e9]))[0] == 0.0


def test_the_transform_is_monotone() -> None:
    rng = np.random.default_rng(0)
    reference = rng.lognormal(size=500)
    transform = RankTransform.fit(reference)

    probe = np.sort(rng.lognormal(size=50))
    ranks = transform.apply(probe)

    assert np.all(np.diff(ranks) >= 0.0)


def test_the_score_averages_the_three_terms() -> None:
    score = combine(np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.4, 0.0]))

    assert np.allclose(score, np.array([0.8, 0.0]))


def test_a_neutral_term_is_the_middle_of_the_range() -> None:
    assert NEUTRAL == 0.5


def test_combine_accepts_a_fourth_term() -> None:
    """A model with a correspondence head contributes one more ranked quantity."""
    ranks = [np.array([0.0, 1.0]) for _ in range(4)]

    assert combine(*ranks).tolist() == [0.0, 1.0]


def test_combine_refuses_to_average_nothing() -> None:
    with pytest.raises(ValueError, match="at least one"):
        combine()
