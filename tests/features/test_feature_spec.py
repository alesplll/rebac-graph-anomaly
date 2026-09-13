"""Feature naming, grouping and masking."""

import numpy as np
import pytest

from rga.features.spec import CandidateSet, FeatureBlock, FeatureGroup, FeatureMatrix


def _block() -> FeatureBlock:
    return FeatureBlock(
        names=("degree", "age", "actor_is_subject"),
        groups=(FeatureGroup.STRUCTURAL, FeatureGroup.TEMPORAL, FeatureGroup.PROVENANCE),
    )


def test_block_length_matches_names() -> None:
    assert len(_block()) == 3


def test_block_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        FeatureBlock(names=("a", "b"), groups=(FeatureGroup.STRUCTURAL,))


def test_block_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate feature name"):
        FeatureBlock(names=("a", "a"), groups=(FeatureGroup.STRUCTURAL, FeatureGroup.STRUCTURAL))


def test_prefixing_renames_without_touching_groups() -> None:
    prefixed = _block().prefixed("subj_")
    assert prefixed.names == ("subj_degree", "subj_age", "subj_actor_is_subject")
    assert prefixed.groups == _block().groups


def test_concat_joins_blocks_in_order() -> None:
    joined = FeatureBlock.concat(_block().prefixed("a_"), _block().prefixed("b_"))
    assert len(joined) == 6
    assert joined.names[0] == "a_degree"
    assert joined.names[3] == "b_degree"


def test_matrix_reports_its_shape() -> None:
    matrix = FeatureMatrix(
        values=np.zeros((4, 3), dtype=np.float32),
        mask=np.ones((4, 3), dtype=bool),
        block=_block(),
    )
    assert matrix.n_rows == 4
    assert matrix.n_features == 3


def test_matrix_rejects_shape_disagreement() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        FeatureMatrix(
            values=np.zeros((4, 3), dtype=np.float32),
            mask=np.ones((4, 2), dtype=bool),
            block=_block(),
        )


def test_group_indices_select_the_right_columns() -> None:
    matrix = FeatureMatrix(
        values=np.zeros((2, 3), dtype=np.float32),
        mask=np.ones((2, 3), dtype=bool),
        block=_block(),
    )
    assert matrix.group_indices(FeatureGroup.TEMPORAL).tolist() == [1]


def test_with_groups_keeps_only_the_named_groups() -> None:
    matrix = FeatureMatrix(
        values=np.arange(6, dtype=np.float32).reshape(2, 3),
        mask=np.ones((2, 3), dtype=bool),
        block=_block(),
    )
    reduced = matrix.with_groups((FeatureGroup.STRUCTURAL, FeatureGroup.PROVENANCE))
    assert reduced.block.names == ("degree", "actor_is_subject")
    assert reduced.values.tolist() == [[0.0, 2.0], [3.0, 5.0]]


def test_candidate_set_exposes_binary_truth() -> None:
    candidates = CandidateSet(
        keys=(("user:a", 2, "bucket:x"), ("user:b", 2, "bucket:y")),
        ts=np.array([10, 20], dtype=np.int64),
        matrix=FeatureMatrix(
            values=np.zeros((2, 3), dtype=np.float32),
            mask=np.ones((2, 3), dtype=bool),
            block=_block(),
        ),
        labels=np.array([False, True]),
        patterns=("", "shadow_group"),
    )
    assert candidates.n_candidates == 2
    assert candidates.y_true().tolist() == [0, 1]
