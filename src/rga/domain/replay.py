"""Journal to graph."""

from __future__ import annotations

from collections.abc import Iterable

from rga.domain.events import EventOp, GraphEvent
from rga.domain.graph import AccessGraph, GraphBuilder


def replay(events: Iterable[GraphEvent], until: int | None = None) -> AccessGraph:
    """Build the graph as it stood at `until`, inclusive.

    Events must be in non-decreasing time order; iteration stops at the first
    event past the cutoff, so a sorted journal is never read in full.
    """
    builder = GraphBuilder()
    for event in events:
        if until is not None and event.ts > until:
            break
        builder.apply(event)
    return builder.build()


def journal_issues(events: Iterable[GraphEvent]) -> list[str]:
    """Describe every internal inconsistency in a journal.

    Catches the mistakes a generator makes: revoking an edge that was never
    granted, and events arriving out of order. An empty list means the journal
    replays cleanly.
    """
    issues: list[str] = []
    live: set[tuple[str, int, str]] = set()
    last_ts: int | None = None

    for position, event in enumerate(events):
        if last_ts is not None and event.ts < last_ts:
            issues.append(f"event {position} at ts={event.ts} precedes ts={last_ts}")
        last_ts = event.ts

        key = event.edge_key()
        if event.op is EventOp.GRANT:
            live.add(key)
        elif key not in live:
            issues.append(f"event {position} revokes an edge that is not live: {key}")
        else:
            live.discard(key)

    return issues
