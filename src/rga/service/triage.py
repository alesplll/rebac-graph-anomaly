"""What the analyst decided about a change.

Three states, and no more: a change is open until somebody looks at it, and once
they have, it is either dismissed — nothing needs doing — or the rights were taken
back. A fourth shade of "we are still thinking" is a queue that never empties.

Kept in memory, on purpose. Decisions do not outlive the process, and no file is
written anywhere; when they need to, `record`, `current` and `history` are the
whole surface a durable store would have to offer.

The journal only ever grows: changing your mind adds an entry, so the history says
who decided what and when they decided otherwise. Nothing here ever reaches the
model — these are human judgements about the very changes it ranks, and feeding them
back would turn every measured number into self-confirmation.
`tests/test_layering.py` checks that mechanically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: `reopened` is how a decided change goes back to being open.
OUTCOMES: tuple[str, ...] = ("dismissed", "revoked", "reopened")
#: The outcome that leaves a change in the queue rather than taking it out.
OPEN_OUTCOME = "reopened"
#: What each state is called on screen.
TITLES: dict[str, str] = {
    "open": "Открыто",
    "dismissed": "Пропущено",
    "revoked": "Права отозваны",
    "reopened": "Открыто",
}


@dataclass(frozen=True)
class Decision:
    """One recorded judgement about one change."""

    seq: int
    incident: str
    outcome: str
    note: str
    analyst: str
    decided_at: str
    subject: str
    relation: str
    object: str
    score: float

    def as_dict(self) -> dict[str, object]:
        """The shape the page receives."""
        return {
            "seq": self.seq,
            "incident": self.incident,
            "outcome": self.outcome,
            "title": TITLES.get(self.outcome, self.outcome),
            "note": self.note,
            "analyst": self.analyst,
            "decided_at": self.decided_at,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "score": self.score,
        }


class MemoryStore:
    """The decision journal, for as long as the service is running."""

    def __init__(self) -> None:
        self._entries: list[Decision] = []

    def record(self, entries: Sequence[Decision]) -> None:
        """Append one entry per decision.

        Every outcome is checked before anything lands: a batch applied halfway
        would leave the analyst unable to tell what they had decided.
        """
        for entry in entries:
            if entry.outcome not in OUTCOMES:
                raise ValueError(f"unknown outcome: {entry.outcome!r}")

        for entry in entries:
            self._entries.append(
                Decision(**{**entry.__dict__, "seq": len(self._entries) + 1})
            )

    def current(self) -> dict[str, Decision]:
        """The latest decision for every change that has one."""
        latest: dict[str, Decision] = {}
        for entry in self._entries:
            latest[entry.incident] = entry
        return latest

    def history(self, *, incident: str | None = None, limit: int = 200) -> tuple[Decision, ...]:
        """Recorded decisions, newest first."""
        chosen = [
            entry
            for entry in reversed(self._entries)
            if incident is None or entry.incident == incident
        ]
        return tuple(chosen[:limit])

    def close(self) -> None:
        """Nothing to release; kept so a durable store can slot in unchanged."""
