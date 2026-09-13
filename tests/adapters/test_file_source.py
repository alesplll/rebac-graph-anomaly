"""A journal file behaves as a full-capability source."""

from pathlib import Path

import pytest

from rga.adapters.base import GraphSource
from rga.adapters.file_source import FileSource
from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.io.jsonl import write_events


def _journal(tmp_path: Path, *, actors: bool) -> Path:
    path = tmp_path / "events.jsonl"
    actor = "user:root" if actors else None
    write_events(
        path,
        [
            GraphEvent(
                100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops",
                actor=actor,
            ),
            GraphEvent(
                200, EventOp.GRANT, "group:devops", RelationType.HAS_PERMISSION,
                "bucket:photos", PermissionLevel.WRITE, actor=actor,
            ),
            GraphEvent(
                300, EventOp.REVOKE, "user:alice", RelationType.MEMBER_OF, "group:devops",
                actor=actor,
            ),
        ],
    )
    return path


def test_file_source_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(FileSource(_journal(tmp_path, actors=True)), GraphSource)


def test_a_journal_with_actors_reaches_level_two(tmp_path: Path) -> None:
    capabilities = FileSource(_journal(tmp_path, actors=True)).capabilities()
    assert capabilities.level == 2
    assert capabilities.change_log is True


def test_a_journal_without_actors_stays_at_level_one(tmp_path: Path) -> None:
    assert FileSource(_journal(tmp_path, actors=False)).capabilities().level == 1


def test_snapshot_honours_the_cutoff(tmp_path: Path) -> None:
    source = FileSource(_journal(tmp_path, actors=True))
    assert source.snapshot(at=200).num_edges == 2
    assert source.snapshot().num_edges == 1


def test_events_are_bounded_at_both_ends(tmp_path: Path) -> None:
    source = FileSource(_journal(tmp_path, actors=True))
    assert [event.ts for event in source.events(since=150, until=250)] == [200]


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.jsonl").capabilities()
