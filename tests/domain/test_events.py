"""Graph events: the single unit of input for the whole system."""

import pytest

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType


def _grant(**overrides: object) -> GraphEvent:
    base: dict[str, object] = {
        "ts": 1_700_000_000_000,
        "op": EventOp.GRANT,
        "subject": "user:alice",
        "relation": RelationType.HAS_PERMISSION,
        "object": "bucket:photos",
        "level": PermissionLevel.READ,
    }
    base.update(overrides)
    return GraphEvent(**base)  # type: ignore[arg-type]


def test_has_permission_grant_requires_a_level() -> None:
    with pytest.raises(ValueError, match="requires a level"):
        _grant(level=PermissionLevel.NONE)


def test_other_relations_must_not_carry_a_level() -> None:
    with pytest.raises(ValueError, match="must not carry a level"):
        _grant(relation=RelationType.MEMBER_OF, object="group:devops")


def test_revoke_does_not_need_a_level() -> None:
    event = _grant(op=EventOp.REVOKE, level=PermissionLevel.NONE)
    assert event.level is PermissionLevel.NONE


def test_edge_key_ignores_level_and_actor() -> None:
    read = _grant(level=PermissionLevel.READ, actor="user:root")
    admin = _grant(level=PermissionLevel.ADMIN, actor="user:alice")
    assert read.edge_key() == admin.edge_key()
    assert read.edge_key() == ("user:alice", int(RelationType.HAS_PERMISSION), "bucket:photos")


def test_round_trips_through_a_dict() -> None:
    event = _grant(level=PermissionLevel.ADMIN, actor="user:root")
    assert GraphEvent.from_dict(event.to_dict()) == event


def test_dict_form_omits_absent_optional_fields() -> None:
    event = _grant(
        relation=RelationType.MEMBER_OF, object="group:devops", level=PermissionLevel.NONE
    )
    payload = event.to_dict()
    assert "level" not in payload
    assert "actor" not in payload


def test_dict_form_is_human_readable() -> None:
    payload = _grant().to_dict()
    assert payload["op"] == "grant"
    assert payload["relation"] == "HAS_PERMISSION"
    assert payload["level"] == "read"
