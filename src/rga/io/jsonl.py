"""Journal storage as newline-delimited JSON.

Chosen over a binary format because a change journal is something a person reads
while debugging a generator or a live integration. Line endings are forced to
"\\n" so that a dataset produced on Windows is byte-identical to one produced on
Linux.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from rga.domain.events import GraphEvent


def write_events(path: Path, events: Iterable[GraphEvent]) -> int:
    """Write events to a journal file, returning how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_events(path: Path) -> Iterator[GraphEvent]:
    """Stream events from a journal file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield GraphEvent.from_dict(json.loads(stripped))
