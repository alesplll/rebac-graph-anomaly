"""Relation and permission vocabulary."""

import pytest

from rga.domain.relations import (
    LEVEL_CARRYING,
    PermissionLevel,
    RelationType,
    parse_level,
    parse_relation,
)


def test_permission_levels_are_ordered() -> None:
    assert PermissionLevel.READ < PermissionLevel.WRITE < PermissionLevel.ADMIN
    assert PermissionLevel.NONE < PermissionLevel.READ


def test_parse_level_accepts_engine_spelling() -> None:
    assert parse_level("admin") is PermissionLevel.ADMIN
    assert parse_level("READ") is PermissionLevel.READ


def test_parse_level_maps_absence_to_none() -> None:
    assert parse_level(None) is PermissionLevel.NONE
    assert parse_level("") is PermissionLevel.NONE


def test_parse_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown permission level"):
        parse_level("superuser")


def test_parse_relation_accepts_engine_spelling() -> None:
    assert parse_relation("HAS_PERMISSION") is RelationType.HAS_PERMISSION
    assert parse_relation("member_of") is RelationType.MEMBER_OF


def test_only_has_permission_carries_a_level() -> None:
    assert RelationType.HAS_PERMISSION in LEVEL_CARRYING
    assert len(LEVEL_CARRYING) == 1


def test_enum_values_are_stable() -> None:
    # These integers are written into saved datasets and index model parameters.
    assert (RelationType.MEMBER_OF, RelationType.HAS_PERMISSION) == (1, 2)
    assert (RelationType.PARENT_OF, RelationType.OWNER_OF) == (3, 4)
