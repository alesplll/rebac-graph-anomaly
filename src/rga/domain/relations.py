"""Canonical relation types and permission levels.

Permission levels are deliberately an ordered scale rather than a categorical
set: the engine defines admin > delete > create > write > read, and the distance
between two levels carries meaning the model should be able to use.
"""

from __future__ import annotations

from enum import IntEnum


class RelationType(IntEnum):
    """Canonical edge types. Values are stable and index model parameters."""

    MEMBER_OF = 1
    HAS_PERMISSION = 2
    PARENT_OF = 3
    OWNER_OF = 4


class PermissionLevel(IntEnum):
    """Ordered permission levels. NONE marks a relation that carries no level."""

    NONE = 0
    READ = 1
    WRITE = 2
    CREATE = 3
    DELETE = 4
    ADMIN = 5


#: Only HAS_PERMISSION carries a level; every other relation must use NONE.
LEVEL_CARRYING = frozenset({RelationType.HAS_PERMISSION})

_LEVEL_BY_NAME = {
    level.name.lower(): level for level in PermissionLevel if level is not PermissionLevel.NONE
}
_RELATION_BY_NAME = {relation.name.lower(): relation for relation in RelationType}


def parse_level(name: str | None) -> PermissionLevel:
    """Convert the engine's level spelling to the ordered enum."""
    if not name:
        return PermissionLevel.NONE
    try:
        return _LEVEL_BY_NAME[name.lower()]
    except KeyError:
        raise ValueError(f"unknown permission level: {name!r}") from None


def parse_relation(name: str) -> RelationType:
    """Convert the engine's relation spelling to the canonical enum."""
    try:
        return _RELATION_BY_NAME[name.lower()]
    except KeyError:
        raise ValueError(f"unknown relation type: {name!r}") from None
