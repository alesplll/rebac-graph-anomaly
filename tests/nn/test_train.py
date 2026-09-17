"""Candidate arrays and the self-supervised training loop."""

from pathlib import Path

import numpy as np
import pytest
import torch

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.candidates import candidate_arrays, edge_positions
from rga.nn.config import ModelConfig
from rga.nn.train import train_model

CPU = torch.device("cpu")
CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def train_set():
    return build_candidates(build_dataset(CONFIG), Span.TRAIN)


def test_arrays_have_one_row_per_candidate(train_set) -> None:
    arrays, mean, std = candidate_arrays(train_set)

    assert arrays.src.shape == (train_set.n_candidates,)
    assert arrays.features.shape[0] == train_set.n_candidates
    assert mean.shape == (arrays.features.shape[1],)
    assert std.shape == (arrays.features.shape[1],)


def test_standardisation_can_be_reused_on_another_span() -> None:
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)

    _, mean, std = candidate_arrays(train)
    arrays, reused_mean, reused_std = candidate_arrays(evaluation, mean=mean, std=std)

    assert np.array_equal(reused_mean, mean)
    assert np.array_equal(reused_std, std)
    assert np.isfinite(arrays.features).all()


def test_levels_stay_inside_the_embedding_range(train_set) -> None:
    arrays, _, _ = candidate_arrays(train_set)

    assert arrays.level.min() >= 0
    assert arrays.level.max() <= 5


def test_edge_positions_find_the_edges_the_candidates_created(train_set) -> None:
    arrays, _, _ = candidate_arrays(train_set)

    positions = edge_positions(train_set.graph, arrays.src, arrays.dst, arrays.relation)

    assert positions.shape == (train_set.n_candidates,)
    assert (positions >= 0).sum() > 0


def test_training_runs_and_returns_a_usable_model(train_set) -> None:
    arrays, _, _ = candidate_arrays(train_set)

    model = train_model(train_set.graph, arrays, FAST, seed=1, device=CPU)

    assert isinstance(model, torch.nn.Module)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_training_is_reproducible(train_set) -> None:
    """Same seed, same model — to float32 tolerance.

    Not bit-exact: `index_add_` accumulates in whatever order the CPU threads
    finish in, which moves the last couple of bits. A leaked RNG would show up as
    a wholly different model, not as a 1e-8 wobble.
    """
    arrays, _, _ = candidate_arrays(train_set)

    first = train_model(train_set.graph, arrays, FAST, seed=1, device=CPU)
    second = train_model(train_set.graph, arrays, FAST, seed=1, device=CPU)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.allclose(left, right, atol=1e-6)


def test_a_different_seed_gives_a_different_model(train_set) -> None:
    """The guard on the test above: seeding has to matter."""
    arrays, _, _ = candidate_arrays(train_set)

    first = train_model(train_set.graph, arrays, FAST, seed=1, device=CPU)
    other = train_model(train_set.graph, arrays, FAST, seed=2, device=CPU)

    deviations = [
        float((left - right).detach().abs().max())
        for left, right in zip(first.parameters(), other.parameters(), strict=True)
    ]
    assert max(deviations) > 1e-4
