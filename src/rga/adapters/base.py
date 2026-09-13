"""The contract every data source implements.

Capability levels, from section 12 of the design document:

    0  a snapshot of relation tuples — any authorization engine has this
    1  plus creation timestamps — enables the temporal feature group
    2  plus the initiator of each change — enables the provenance group

A source states what it has; feature extraction masks what it lacks. Nothing
above this layer knows which product is behind the source.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph


@dataclass(frozen=True)
class Capabilities:
    """What a source can provide.

    `change_log` is orthogonal to the level: it says whether `events()` is
    supported, which also decides whether revocations are observable at all — a
    revoked edge is simply absent from a snapshot.
    """

    timestamps: bool
    provenance: bool
    change_log: bool

    @property
    def level(self) -> int:
        """Capability level 0, 1 or 2."""
        if not self.timestamps:
            return 0
        return 2 if self.provenance else 1


@runtime_checkable
class GraphSource(Protocol):
    """A source of access-graph data."""

    def capabilities(self) -> Capabilities:
        """Describe what this source can provide."""
        ...

    def snapshot(self, at: int | None = None) -> AccessGraph:
        """The graph as it stood at `at`, or the latest state when `at` is None.

        A source without timestamps ignores `at` and returns the current state.
        """
        ...

    def events(self, since: int = 0, until: int | None = None) -> Iterator[GraphEvent]:
        """Changes in the interval, in non-decreasing time order.

        Raises NotImplementedError when `capabilities().change_log` is False.
        """
        ...
