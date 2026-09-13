"""A source backed by a saved journal file.

Used by the generator, by saved datasets, and by any integration that exports a
change log offline rather than being queried live.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from rga.adapters.base import Capabilities
from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph
from rga.domain.replay import replay
from rga.io.jsonl import read_events

#: How many leading events to inspect when deciding whether actors are recorded.
_PROBE_SIZE = 256


class FileSource:
    """Reads a newline-delimited journal from disk."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def capabilities(self) -> Capabilities:
        """A journal always has timestamps; provenance depends on the data."""
        probe = list(islice(read_events(self._path), _PROBE_SIZE))
        has_actor = any(event.actor is not None for event in probe)
        return Capabilities(timestamps=True, provenance=has_actor, change_log=True)

    def snapshot(self, at: int | None = None) -> AccessGraph:
        """Replay the journal up to `at`."""
        return replay(read_events(self._path), until=at)

    def events(self, since: int = 0, until: int | None = None) -> Iterator[GraphEvent]:
        """Yield journal events within the interval, stopping early past `until`."""
        for event in read_events(self._path):
            if until is not None and event.ts > until:
                return
            if event.ts >= since:
                yield event
