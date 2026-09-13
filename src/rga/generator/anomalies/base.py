"""The anomaly pattern contract.

A pattern receives the graph as it stood at the start of the evaluation window
and plants a group of related events inside that window, returning exact labels
for the edges it created. Patterns are stateless and independent; the dataset
assembler decides how many of each to inject.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph
from rga.domain.relations import PermissionLevel, RelationType, parse_relation
from rga.generator.org import Organization
from rga.util.timeutil import DAY_MS, HOUR_MS, MINUTE_MS

#: How many times a pattern retries candidate selection before giving up.
ATTEMPTS = 64


class NoCandidateError(RuntimeError):
    """A pattern found no suitable place to inject itself.

    Expected on small graphs; the assembler skips the pattern and tries another.
    """


@dataclass(frozen=True)
class AnomalyLabel:
    """Ground truth: one edge that a pattern created."""

    ts: int
    subject: str
    relation: RelationType
    object: str
    pattern: str

    def edge_key(self) -> tuple[str, int, str]:
        return (self.subject, int(self.relation), self.object)

    def to_dict(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "subject": self.subject,
            "relation": self.relation.name,
            "object": self.object,
            "pattern": self.pattern,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AnomalyLabel:
        return cls(
            ts=int(payload["ts"]),  # type: ignore[arg-type]
            subject=str(payload["subject"]),
            relation=parse_relation(str(payload["relation"])),
            object=str(payload["object"]),
            pattern=str(payload["pattern"]),
        )


@dataclass(frozen=True)
class Injection:
    """What one pattern produced."""

    events: tuple[GraphEvent, ...]
    labels: tuple[AnomalyLabel, ...]


@dataclass(frozen=True)
class InjectionContext:
    """Everything a pattern may look at."""

    rng: np.random.Generator
    org: Organization
    #: The graph as it stood at the start of the window.
    graph: AccessGraph
    #: Half-open [start, end) of the evaluation window, in milliseconds.
    window: tuple[int, int]


class AnomalyPattern(Protocol):
    """One threat scenario."""

    name: str

    def inject(self, context: InjectionContext) -> Injection:
        """Plant the pattern, or raise NoCandidateError if the graph has no room."""
        ...


REGISTRY: dict[str, AnomalyPattern] = {}


def register(cls: type) -> type:
    """Class decorator registering a stateless pattern under its name."""
    instance = cls()
    if instance.name in REGISTRY:
        raise ValueError(f"duplicate anomaly pattern name: {instance.name!r}")
    REGISTRY[instance.name] = instance
    return cls


def get_pattern(name: str) -> AnomalyPattern:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown anomaly pattern: {name!r}") from None


def available_patterns() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def pick[T](rng: np.random.Generator, sequence: Sequence[T]) -> T:
    """Draw one element.

    Written by hand rather than with `rng.choice` because numpy coerces a
    sequence of dataclasses or strings into an object array and silently changes
    their type.
    """
    if not sequence:
        raise NoCandidateError("cannot pick from an empty sequence")
    return sequence[int(rng.integers(len(sequence)))]


def sample_ts(rng: np.random.Generator, window: tuple[int, int]) -> int:
    """A moment uniformly inside the window."""
    start, end = window
    return int(rng.integers(start, end))


def sample_night_ts(rng: np.random.Generator, window: tuple[int, int]) -> int:
    """A moment inside the window, forced into the small hours."""
    start, end = window
    day_count = max(1, (end - start) // DAY_MS)
    midnight = start - (start % DAY_MS) + int(rng.integers(day_count)) * DAY_MS
    ts = midnight + int(rng.integers(5)) * HOUR_MS + int(rng.integers(60)) * MINUTE_MS
    return int(min(max(ts, start), end - 1))


def level_on(graph: AccessGraph, subject_id: str, object_id: str) -> PermissionLevel:
    """Highest permission the subject holds directly on the object."""
    if subject_id not in graph.node_index or object_id not in graph.node_index:
        return PermissionLevel.NONE
    mask = (
        (graph.edge_src == graph.index_of(subject_id))
        & (graph.edge_dst == graph.index_of(object_id))
        & (graph.edge_rel == int(RelationType.HAS_PERMISSION))
    )
    levels = graph.edge_level[mask]
    return PermissionLevel(int(levels.max())) if levels.size else PermissionLevel.NONE


def last_activity(graph: AccessGraph, subject_id: str) -> int:
    """Creation time of the subject's newest outgoing edge, or -1 if it has none."""
    if subject_id not in graph.node_index:
        return -1
    times = graph.edge_created[graph.edge_src == graph.index_of(subject_id)]
    return int(times.max()) if times.size else -1


def label_for(event: GraphEvent, pattern: str) -> AnomalyLabel:
    """Ground-truth label for an event a pattern created."""
    return AnomalyLabel(
        ts=event.ts,
        subject=event.subject,
        relation=event.relation,
        object=event.object,
        pattern=pattern,
    )
