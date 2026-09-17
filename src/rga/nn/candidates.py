"""A candidate set as the arrays the model indexes with.

Endpoints are resolved against the graph at the split. An endpoint the graph has
never seen — a user hired inside the evaluation window, say — comes back as -1, and
the scorer gives it a zero representation rather than pretending it knows the node.

The permission level is read off the `level_ordinal` feature, which belongs to the
structural group and is therefore present at every capability level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rga.baselines.base import dense_matrix
from rga.domain.graph import AccessGraph
from rga.domain.relations import PermissionLevel
from rga.features.edges import decode_level
from rga.features.spec import CandidateSet

_MAX_LEVEL = int(PermissionLevel.ADMIN)


@dataclass(frozen=True)
class CandidateArrays:
    """Index arrays and the standardised feature matrix of one span."""

    src: np.ndarray
    dst: np.ndarray
    relation: np.ndarray
    level: np.ndarray
    features: np.ndarray


def candidate_arrays(
    candidates: CandidateSet,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[CandidateArrays, np.ndarray, np.ndarray]:
    """Arrays for one span, standardised by the training span's statistics."""
    if candidates.graph is None:
        raise ValueError("the candidate set carries no graph; the network needs one")

    index = candidates.graph.node_index
    src = np.array([index.get(key[0], -1) for key in candidates.keys], dtype=np.int64)
    dst = np.array([index.get(key[2], -1) for key in candidates.keys], dtype=np.int64)
    relation = np.array([key[1] for key in candidates.keys], dtype=np.int64)

    ordinal = candidates.matrix.column("level_ordinal")
    level = np.clip(
        [decode_level(value) for value in ordinal], 0, _MAX_LEVEL
    ).astype(np.int64)

    dense = dense_matrix(candidates.matrix)
    if mean is None or std is None:
        mean = dense.mean(axis=0)
        std = dense.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    features = ((dense - mean) / std).astype(np.float32)

    return (
        CandidateArrays(src=src, dst=dst, relation=relation, level=level, features=features),
        mean,
        std,
    )


def without_features(arrays: CandidateArrays) -> CandidateArrays:
    """The same candidates with an empty context row.

    The self-supervised objective gives a positive and its corrupted negatives the
    same feature row, so no gradient ever distinguishes those inputs and the head's
    weights on them stay where initialisation left them. At scoring time the row
    varies from candidate to candidate and those untrained weights feed the logit
    noise. Measured over five seeds on small-history: keeping the row scored
    0.073 +/- 0.043 PR-AUC, dropping it 0.368 +/- 0.235.

    The consequence is worth stating plainly: the self-supervised network reads
    structure alone, which is capability level 0 — the minimum any ReBAC engine
    offers. The context row still reaches the supervised variant, where positives
    and negatives differ in it and its weights do get trained.
    """
    return CandidateArrays(
        src=arrays.src,
        dst=arrays.dst,
        relation=arrays.relation,
        level=arrays.level,
        features=np.zeros((len(arrays.src), 0), dtype=np.float32),
    )


def edge_positions(
    graph: AccessGraph, src: np.ndarray, dst: np.ndarray, relation: np.ndarray
) -> np.ndarray:
    """Row of each candidate's edge in the graph, or -1 when it is not there."""
    lookup = {
        (int(a), int(r), int(b)): position
        for position, (a, r, b) in enumerate(
            zip(graph.edge_src, graph.edge_rel, graph.edge_dst, strict=True)
        )
    }
    return np.array(
        [
            lookup.get((int(a), int(r), int(b)), -1)
            for a, r, b in zip(src, relation, dst, strict=True)
        ],
        dtype=np.int64,
    )
