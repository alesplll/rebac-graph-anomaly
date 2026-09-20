"""The journal of what the analyst decided."""

from pathlib import Path

import pytest

from rga.service.triage import Decision, TriageStore


def _decision(incident: str, outcome: str, **overrides) -> Decision:
    fields = {
        "seq": 0,
        "incident": incident,
        "outcome": outcome,
        "note": "",
        "analyst": "analyst",
        "decided_at": "2026-09-20T12:00:00+00:00",
        "subject": "user:alice",
        "relation": "HAS_PERMISSION",
        "object": "bucket:logs",
        "score": 0.9,
    }
    fields.update(overrides)
    return Decision(**fields)


def test_a_recorded_decision_comes_back(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")
    store.record([_decision("abc", "confirmed", note="escalation")])

    current = store.current()

    assert set(current) == {"abc"}
    assert current["abc"].outcome == "confirmed"
    assert current["abc"].note == "escalation"


def test_the_latest_decision_wins(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")
    store.record([_decision("abc", "confirmed")])
    store.record([_decision("abc", "false_positive")])

    assert store.current()["abc"].outcome == "false_positive"


def test_history_keeps_every_decision_newest_first(tmp_path: Path) -> None:
    """The journal is append-only: nothing is overwritten and nothing is lost."""
    store = TriageStore(tmp_path / "t.db")
    store.record([_decision("abc", "confirmed")])
    store.record([_decision("abc", "reopened")])
    store.record([_decision("abc", "accepted_risk")])

    history = store.history(incident="abc")

    assert [entry.outcome for entry in history] == [
        "accepted_risk",
        "reopened",
        "confirmed",
    ]


def test_a_decision_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    first = TriageStore(path)
    first.record([_decision("abc", "confirmed")])
    first.close()

    assert TriageStore(path).current()["abc"].outcome == "confirmed"


def test_a_batch_records_one_row_per_incident(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")

    store.record([_decision("a", "accepted_risk"), _decision("b", "accepted_risk")])

    assert set(store.current()) == {"a", "b"}
    assert len(store.history()) == 2


def test_an_unknown_outcome_is_refused(tmp_path: Path) -> None:
    store = TriageStore(tmp_path / "t.db")
    with pytest.raises(ValueError, match="outcome"):
        store.record([_decision("abc", "probably-fine")])


def test_nothing_is_written_when_one_outcome_is_wrong(tmp_path: Path) -> None:
    """A batch is checked before any of it lands, so a typo cannot half-apply."""
    store = TriageStore(tmp_path / "t.db")
    with pytest.raises(ValueError, match="outcome"):
        store.record([_decision("a", "confirmed"), _decision("b", "nonsense")])

    assert store.current() == {}


def test_the_decision_carries_what_it_was_about(tmp_path: Path) -> None:
    """The window moves; the history has to stay readable without the incident."""
    store = TriageStore(tmp_path / "t.db")
    store.record(
        [_decision("abc", "confirmed", subject="user:bob", object="bucket:pay", score=0.77)]
    )

    entry = store.history()[0]

    assert (entry.subject, entry.object, entry.score) == ("user:bob", "bucket:pay", 0.77)
