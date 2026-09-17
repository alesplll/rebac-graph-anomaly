"""The ordinal permission level survives the round trip through the feature row.

`level_ordinal` is stored normalised, so admin is 1.0 rather than 5. Anything that
reads it back has to undo that: the level embedding of the likelihood head, the level
in an incident card, and the sentence an analyst reads. Treating the normalised value
as an ordinal silently collapses five levels into two.
"""

from pathlib import Path

import numpy as np

from rga.domain.relations import PermissionLevel
from rga.features.build import Span, build_candidates
from rga.features.edges import decode_level
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.candidates import candidate_arrays

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))


def test_every_level_decodes_to_itself() -> None:
    for level in PermissionLevel:
        normalised = float(level) / float(PermissionLevel.ADMIN)
        assert decode_level(normalised) == int(level)


def test_the_model_sees_every_level_not_just_two() -> None:
    candidates = build_candidates(build_dataset(CONFIG), Span.EVAL)

    arrays, _, _ = candidate_arrays(candidates)

    assert len(set(arrays.level.tolist())) > 2
    assert arrays.level.max() == int(PermissionLevel.ADMIN)


def test_an_admin_grant_is_described_as_admin() -> None:
    from rga.explain.text import describe

    candidates = build_candidates(build_dataset(CONFIG), Span.EVAL)
    column = candidates.matrix.column("level_ordinal")
    position = int(np.flatnonzero(column >= 1.0)[0])

    assert "admin" in " ".join(describe(candidates, position))
