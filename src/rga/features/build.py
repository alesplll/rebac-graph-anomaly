"""Turning a dataset into scored rows.

A single pass over the journal, in time order. Each grant's features are read off
the context before that grant is applied, so a candidate describes the graph as it
stood immediately before the edge appeared.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

import numpy as np

from rga.domain.events import EventOp, GraphEvent
from rga.domain.graph import AccessGraph
from rga.domain.replay import replay
from rga.features.context import FeatureContext
from rga.features.edges import EDGE_BLOCK, edge_features
from rga.features.nodes import NODE_BLOCK, node_features
from rga.features.spec import CandidateSet, FeatureBlock, FeatureMatrix
from rga.features.static import StaticAttributes, compute_static_attributes
from rga.generator.dataset import Dataset
from rga.util.timeutil import DAY_MS


class Span(StrEnum):
    """Which side of the temporal split candidates are drawn from."""

    TRAIN = "train"
    EVAL = "eval"


#: Principals whose changes are platform automation rather than decisions. The
#: permission an engine writes on an object as part of an upload is not something
#: a person chose and is not worth an analyst's attention; scoring it is like
#: scoring a log line. It also swamps the candidate set — half the window would be
#: "a right on an object created an instant ago", and the age of the target would
#: separate normal from anomalous on its own.
#:
#: An integrator names their own automation principals here; the default matches
#: what the generator emits.
AUTOMATION_ACTORS = frozenset({"user:system"})

#: Days at the start of the journal excluded from training candidates. Day zero
#: founds the organization in one burst against an empty graph, and those rows
#: describe no structure at all.
WARMUP_DAYS = 7

CANDIDATE_BLOCK = FeatureBlock.concat(
    EDGE_BLOCK, NODE_BLOCK.prefixed("subj_"), NODE_BLOCK.prefixed("obj_")
)


def _span_bounds(dataset: Dataset, span: Span, warmup_days: int) -> tuple[int, int]:
    """Half-open [start, end) of the requested span."""
    if span is Span.EVAL:
        return dataset.split_ts, dataset.window_end
    return dataset.config.timeline.start_ts + warmup_days * DAY_MS, dataset.split_ts


def _candidate_row(
    context: FeatureContext, static: StaticAttributes, event: GraphEvent
) -> tuple[np.ndarray, np.ndarray]:
    """The full feature row: the change itself, then both endpoints."""
    edge_values, edge_mask = edge_features(context, static, event)
    subject_values, subject_mask = node_features(context, static, event.subject, event.ts)
    object_values, object_mask = node_features(context, static, event.object, event.ts)
    return (
        np.concatenate([edge_values, subject_values, object_values]),
        np.concatenate([edge_mask, subject_mask, object_mask]),
    )


def candidates_from_events(
    events: Iterable[GraphEvent],
    *,
    start: int,
    end: int,
    graph: AccessGraph,
    static_seed: int = 0,
    labels: Mapping[tuple[tuple[str, int, str], int], str] | None = None,
    automation_actors: frozenset[str] = AUTOMATION_ACTORS,
) -> CandidateSet:
    """Extract features for every grant inside [start, end).

    `graph` is the state the network propagates over — normally the snapshot at the
    temporal split. `labels` is ground truth where it exists; a live source has none
    and passes nothing, which makes every candidate unlabelled rather than normal.
    """
    static = compute_static_attributes(graph, np.random.default_rng(static_seed))
    # Labels are matched on identity and time together: a normal re-grant of the
    # same edge later in the window is a different candidate, not an anomaly.
    label_of = dict(labels or {})

    context = FeatureContext()

    keys: list[tuple[str, int, str]] = []
    stamps: list[int] = []
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    flags: list[bool] = []
    patterns: list[str] = []
    actors: list[str | None] = []

    for event in events:
        is_candidate = (
            event.op is EventOp.GRANT
            and start <= event.ts < end
            and event.actor not in automation_actors
        )
        if is_candidate:
            values, mask = _candidate_row(context, static, event)
            keys.append(event.edge_key())
            stamps.append(event.ts)
            rows.append(values)
            masks.append(mask)
            pattern = label_of.get((event.edge_key(), event.ts), "")
            flags.append(bool(pattern))
            patterns.append(pattern)
            actors.append(event.actor)
        context.apply(event)

    matrix = FeatureMatrix(
        values=(
            np.vstack(rows).astype(np.float32)
            if rows
            else np.zeros((0, len(CANDIDATE_BLOCK)), dtype=np.float32)
        ),
        mask=(
            np.vstack(masks) if masks else np.zeros((0, len(CANDIDATE_BLOCK)), dtype=bool)
        ),
        block=CANDIDATE_BLOCK,
    )
    return CandidateSet(
        keys=tuple(keys),
        ts=np.array(stamps, dtype=np.int64),
        matrix=matrix,
        labels=np.array(flags, dtype=bool),
        patterns=tuple(patterns),
        actors=tuple(actors),
        graph=graph,
    )


def build_candidates(
    dataset: Dataset,
    span: Span,
    *,
    static_seed: int = 0,
    warmup_days: int = WARMUP_DAYS,
    automation_actors: frozenset[str] = AUTOMATION_ACTORS,
) -> CandidateSet:
    """Extract features for every grant inside the requested span of a dataset."""
    start, end = _span_bounds(dataset, span, warmup_days)
    return candidates_from_events(
        dataset.events,
        start=start,
        end=end,
        graph=replay(dataset.events, until=dataset.split_ts),
        static_seed=static_seed,
        labels={
            (label.edge_key(), label.ts): label.pattern
            for label in (*dataset.labels, *dataset.train_labels)
        },
        automation_actors=automation_actors,
    )
