"""What the analyst decided, and when.

An append-only journal. A decision is never edited and never deleted: a change of
mind adds a row, and the current state of an incident is its latest row. That is
what makes the history worth keeping — the alternative, a mutable flag, answers
"is this closed" and nothing else, while a security review asks "who decided what,
and did anyone change their mind".

Each row carries a copy of the change it was about. The evaluation window moves and
the source can be swapped, so an incident decided on yesterday may not exist in
today's analysis; the history has to stay readable regardless.

Nothing here ever reaches the model. These are human judgements about the very
changes the model ranks, and feeding them back would turn every measured number
into self-confirmation. `tests/test_layering.py` checks that mechanically.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Outcomes a decision may carry. `reopened` puts the incident back in the queue.
OUTCOMES: tuple[str, ...] = ("confirmed", "false_positive", "accepted_risk", "reopened")
#: The outcome that leaves an incident in the queue rather than taking it out.
OPEN_OUTCOME = "reopened"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    incident   TEXT    NOT NULL,
    outcome    TEXT    NOT NULL,
    note       TEXT    NOT NULL DEFAULT '',
    analyst    TEXT    NOT NULL DEFAULT 'analyst',
    decided_at TEXT    NOT NULL,
    subject    TEXT    NOT NULL,
    relation   TEXT    NOT NULL,
    object     TEXT    NOT NULL,
    score      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_incident ON decisions (incident);
CREATE INDEX IF NOT EXISTS decisions_seq ON decisions (seq DESC);
"""

_COLUMNS = "seq, incident, outcome, note, analyst, decided_at, subject, relation, object, score"


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
            "note": self.note,
            "analyst": self.analyst,
            "decided_at": self.decided_at,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "score": self.score,
        }


class TriageStore:
    """The decision journal, kept in one SQLite file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The service is one process, but the framework answers on a thread pool, so
        # the connection crosses threads and leans on SQLite's own locking.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def record(self, entries: Sequence[Decision]) -> None:
        """Append one row per decision.

        Every outcome is checked before anything is written: a batch that is half
        applied would leave the analyst unable to tell what they had decided.
        """
        for entry in entries:
            if entry.outcome not in OUTCOMES:
                raise ValueError(f"unknown outcome: {entry.outcome!r}")

        self._db.executemany(
            "INSERT INTO decisions"
            " (incident, outcome, note, analyst, decided_at,"
            "  subject, relation, object, score)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    entry.incident,
                    entry.outcome,
                    entry.note,
                    entry.analyst,
                    entry.decided_at,
                    entry.subject,
                    entry.relation,
                    entry.object,
                    entry.score,
                )
                for entry in entries
            ],
        )
        self._db.commit()

    def current(self) -> dict[str, Decision]:
        """The latest decision for every incident that has one."""
        rows = self._db.execute(
            f"SELECT {_COLUMNS} FROM decisions"
            " WHERE seq IN (SELECT MAX(seq) FROM decisions GROUP BY incident)"
        ).fetchall()
        return {str(row["incident"]): _decision(row) for row in rows}

    def history(self, *, incident: str | None = None, limit: int = 200) -> tuple[Decision, ...]:
        """Recorded decisions, newest first."""
        if incident is None:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM decisions ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM decisions WHERE incident = ? ORDER BY seq DESC LIMIT ?",
                (incident, limit),
            ).fetchall()
        return tuple(_decision(row) for row in rows)

    def close(self) -> None:
        """Release the file."""
        self._db.close()


def _decision(row: sqlite3.Row) -> Decision:
    return Decision(
        seq=int(row["seq"]),
        incident=str(row["incident"]),
        outcome=str(row["outcome"]),
        note=str(row["note"]),
        analyst=str(row["analyst"]),
        decided_at=str(row["decided_at"]),
        subject=str(row["subject"]),
        relation=str(row["relation"]),
        object=str(row["object"]),
        score=float(row["score"]),
    )
