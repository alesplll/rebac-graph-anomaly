"""The graph event: one change of access rights.

Every source — the synthetic generator, a live authorization engine, a saved
file — is reduced to a stream of these. Nothing above the adapter layer knows
where they came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from rga.domain.relations import (
    LEVEL_CARRYING,
    PermissionLevel,
    RelationType,
    parse_level,
    parse_relation,
)


class EventOp(IntEnum):
    """Whether the event adds or removes an edge."""

    GRANT = 1
    REVOKE = 2


@dataclass(frozen=True, slots=True)
class GraphEvent:
    """A single change of the access graph.

    `actor` is whoever performed the change, and is the difference between a
    right granted by an administrator and a right a subject granted to itself.
    Most authorization engines do not record it; `None` means unknown, which is
    a distinct state from "nobody".
    """

    ts: int
    op: EventOp
    subject: str
    relation: RelationType
    object: str
    level: PermissionLevel = PermissionLevel.NONE
    actor: str | None = None

    def __post_init__(self) -> None:
        if self.relation in LEVEL_CARRYING:
            if self.op is EventOp.GRANT and self.level is PermissionLevel.NONE:
                raise ValueError(f"{self.relation.name} grant requires a level")
        elif self.level is not PermissionLevel.NONE:
            raise ValueError(f"{self.relation.name} must not carry a level")

    def edge_key(self) -> tuple[str, int, str]:
        """Identity of the edge this event affects, independent of level and actor."""
        return (self.subject, int(self.relation), self.object)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a readable mapping; absent optional fields are omitted."""
        payload: dict[str, object] = {
            "ts": self.ts,
            "op": self.op.name.lower(),
            "subject": self.subject,
            "relation": self.relation.name,
            "object": self.object,
        }
        if self.level is not PermissionLevel.NONE:
            payload["level"] = self.level.name.lower()
        if self.actor is not None:
            payload["actor"] = self.actor
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GraphEvent:
        """Rebuild an event from its mapping form."""
        level = payload.get("level")
        actor = payload.get("actor")
        return cls(
            ts=int(payload["ts"]),  # type: ignore[arg-type]
            op=EventOp[str(payload["op"]).upper()],
            subject=str(payload["subject"]),
            relation=parse_relation(str(payload["relation"])),
            object=str(payload["object"]),
            level=parse_level(None if level is None else str(level)),
            actor=None if actor is None else str(actor),
        )
