"""The three states a change can be in, and where they are kept."""

import pytest

from rga.service.triage import OUTCOMES, Decision, MemoryStore


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


def test_there_are_exactly_three_states() -> None:
    """Open, and two ways of being done with it."""
    assert set(OUTCOMES) == {"dismissed", "revoked", "reopened"}


def test_a_recorded_decision_comes_back() -> None:
    store = MemoryStore()
    store.record([_decision("abc", "revoked", note="права сняты")])

    current = store.current()

    assert current["abc"].outcome == "revoked"
    assert current["abc"].note == "права сняты"


def test_the_latest_decision_wins() -> None:
    store = MemoryStore()
    store.record([_decision("abc", "revoked")])
    store.record([_decision("abc", "dismissed")])

    assert store.current()["abc"].outcome == "dismissed"


def test_history_keeps_every_decision_newest_first() -> None:
    store = MemoryStore()
    store.record([_decision("abc", "revoked")])
    store.record([_decision("abc", "reopened")])
    store.record([_decision("abc", "dismissed")])

    assert [entry.outcome for entry in store.history(incident="abc")] == [
        "dismissed",
        "reopened",
        "revoked",
    ]


def test_a_batch_records_one_row_per_incident() -> None:
    store = MemoryStore()

    store.record([_decision("a", "dismissed"), _decision("b", "dismissed")])

    assert set(store.current()) == {"a", "b"}
    assert len(store.history()) == 2


def test_nothing_outlives_the_process() -> None:
    """Decisions live in memory on purpose: no file appears anywhere."""
    first = MemoryStore()
    first.record([_decision("abc", "revoked")])

    assert MemoryStore().current() == {}


def test_an_unknown_outcome_is_refused() -> None:
    store = MemoryStore()
    with pytest.raises(ValueError, match="outcome"):
        store.record([_decision("abc", "probably-fine")])


def test_nothing_is_written_when_one_outcome_is_wrong() -> None:
    """A batch is checked before any of it lands, so a typo cannot half-apply."""
    store = MemoryStore()
    with pytest.raises(ValueError, match="outcome"):
        store.record([_decision("a", "revoked"), _decision("b", "nonsense")])

    assert store.current() == {}


def test_the_decision_carries_what_it_was_about() -> None:
    store = MemoryStore()
    store.record([_decision("abc", "revoked", subject="user:bob", object="bucket:pay", score=0.77)])

    entry = store.history()[0]

    assert (entry.subject, entry.object, entry.score) == ("user:bob", "bucket:pay", 0.77)
