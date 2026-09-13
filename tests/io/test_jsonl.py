"""Journal files must round-trip identically on both platforms."""

from pathlib import Path

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.io.jsonl import read_events, write_events


def _events() -> list[GraphEvent]:
    return [
        GraphEvent(1_000, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(
            2_000,
            EventOp.GRANT,
            "group:devops",
            RelationType.HAS_PERMISSION,
            "bucket:photos",
            PermissionLevel.WRITE,
            actor="user:root",
        ),
        GraphEvent(3_000, EventOp.REVOKE, "user:alice", RelationType.MEMBER_OF, "group:devops"),
    ]


def test_write_then_read_preserves_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    written = write_events(path, _events())
    assert written == 3
    assert list(read_events(path)) == _events()


def test_file_uses_unix_line_endings(tmp_path: Path) -> None:
    # Windows would otherwise write \r\n and make the same dataset differ by platform.
    path = tmp_path / "events.jsonl"
    write_events(path, _events())
    assert b"\r\n" not in path.read_bytes()


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_events(path, _events())
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(list(read_events(path))) == 3
