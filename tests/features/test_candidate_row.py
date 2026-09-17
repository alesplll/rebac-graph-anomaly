"""A candidate set can hand out one of its rows."""

from pathlib import Path

import numpy as np

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def test_a_row_is_a_candidate_set_of_one() -> None:
    candidates = build_candidates(build_dataset(CONFIG), Span.EVAL)

    single = candidates.row(4)

    assert single.n_candidates == 1
    assert single.keys == (candidates.keys[4],)
    assert single.ts[0] == candidates.ts[4]
    assert np.array_equal(single.matrix.values[0], candidates.matrix.values[4])
    assert single.matrix.block is candidates.matrix.block
    assert single.graph is candidates.graph
