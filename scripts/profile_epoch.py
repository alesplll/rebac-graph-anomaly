"""Where the time of one training epoch goes.

Prints the cost of drawing negative samples against the cost of a forward and
backward pass. That ratio is what decides whether moving the run to a graphics
card is worth anything: a Python-bound sampler leaves the device idle no matter
how fast it is.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.candidates import candidate_arrays
from rga.nn.config import ModelConfig
from rga.nn.encoder import GraphEncoder
from rga.nn.graph_tensors import graph_tensors
from rga.nn.negatives import sample_negatives
from rga.nn.node_inputs import NODE_INPUT_DIM, node_input_features

#: The dataset module 3 measured on, so the numbers stay comparable.
_DATASET = Path("configs/generator/default-history.yaml")
_SEED = 20260912


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DATASET, help="dataset recipe")
    arguments = parser.parse_args()
    config = ModelConfig()

    print(f"dataset: {arguments.config}")
    started = time.perf_counter()
    dataset = build_dataset(load_dataset_config(arguments.config))
    print(f"build_dataset: {time.perf_counter() - started:.1f}s, events {len(dataset.events)}")

    started = time.perf_counter()
    train = build_candidates(dataset, Span.TRAIN)
    graph = train.graph
    assert graph is not None
    print(
        f"build_candidates(TRAIN): {time.perf_counter() - started:.1f}s, "
        f"candidates {len(train.keys)}, graph {graph.num_nodes} nodes / {graph.num_edges} edges"
    )

    arrays, _, _ = candidate_arrays(train)
    rng = np.random.default_rng(_SEED)

    started = time.perf_counter()
    corrupted = sample_negatives(
        graph,
        arrays.src,
        arrays.dst,
        arrays.relation,
        arrays.level,
        per_edge=config.negatives_per_edge,
        rng=rng,
    )
    sampling = time.perf_counter() - started
    print(f"sample_negatives once: {sampling:.2f}s, drew {corrupted.src.size} negatives")

    device = torch.device("cpu")
    tensors = graph_tensors(graph, device=device)
    inputs = node_input_features(tensors)
    encoder = GraphEncoder(NODE_INPUT_DIM, config).to(device)

    started = time.perf_counter()
    encoder(inputs, tensors).sum().backward()
    forward = time.perf_counter() - started
    print(f"one encoder forward+backward: {forward:.2f}s")

    epoch = sampling + forward
    print(f"estimate: {epoch:.2f}s per epoch -> {epoch * config.epochs / 60:.1f} min per model")


if __name__ == "__main__":
    main()
