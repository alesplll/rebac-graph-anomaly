"""Translation from a source's own vocabulary to the canonical one."""

from pathlib import Path

import pytest

from rga.adapters.mapping import RelationMapping
from rga.domain.relations import PermissionLevel, RelationType


def test_level_carrying_relation_reads_the_level_from_a_property() -> None:
    mapping = RelationMapping.from_config(
        {"HAS_PERMISSION": {"relation": "has_permission", "level": "from_property"}}
    )
    assert mapping.translate("HAS_PERMISSION", "admin") == (
        RelationType.HAS_PERMISSION,
        PermissionLevel.ADMIN,
    )


def test_a_system_without_levels_encodes_them_in_relation_names() -> None:
    # SpiceDB and OpenFGA style: each right is its own relation, no ordinal property.
    mapping = RelationMapping.from_config(
        {
            "viewer": {"relation": "has_permission", "level": "read"},
            "editor": {"relation": "has_permission", "level": "write"},
            "owner": {"relation": "owner_of"},
        }
    )
    assert mapping.translate("viewer") == (RelationType.HAS_PERMISSION, PermissionLevel.READ)
    assert mapping.translate("editor") == (RelationType.HAS_PERMISSION, PermissionLevel.WRITE)
    assert mapping.translate("owner") == (RelationType.OWNER_OF, PermissionLevel.NONE)


def test_missing_property_on_a_level_carrying_relation_yields_none() -> None:
    # An unknown level is masked downstream rather than guessed.
    mapping = RelationMapping.from_config(
        {"HAS_PERMISSION": {"relation": "has_permission", "level": "from_property"}}
    )
    assert mapping.translate("HAS_PERMISSION", None) == (
        RelationType.HAS_PERMISSION,
        PermissionLevel.NONE,
    )


def test_unknown_source_relation_is_rejected() -> None:
    mapping = RelationMapping.from_config(
        {"viewer": {"relation": "has_permission", "level": "read"}}
    )
    with pytest.raises(KeyError, match="unmapped source relation"):
        mapping.translate("banana")


def test_a_fixed_level_on_a_relation_that_cannot_carry_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot carry a level"):
        RelationMapping.from_config({"boss": {"relation": "member_of", "level": "admin"}})


def test_reading_a_level_from_a_relation_that_cannot_carry_one_is_rejected() -> None:
    # A source claiming member_of has a level property is misconfigured, not
    # something to silently accept and mask later.
    with pytest.raises(ValueError, match="cannot carry a level"):
        RelationMapping.from_config({"boss": {"relation": "member_of", "level": "from_property"}})


def test_the_shipped_opens3_mapping_loads() -> None:
    mapping = RelationMapping.load(Path("configs/mapping/opens3.yaml"))
    assert mapping.translate("MEMBER_OF") == (RelationType.MEMBER_OF, PermissionLevel.NONE)
    assert mapping.translate("HAS_PERMISSION", "write") == (
        RelationType.HAS_PERMISSION,
        PermissionLevel.WRITE,
    )
    assert mapping.translate("OWNER_OF") == (RelationType.OWNER_OF, PermissionLevel.NONE)
