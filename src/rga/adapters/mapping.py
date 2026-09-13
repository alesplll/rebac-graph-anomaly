"""Declarative translation of a source's relation vocabulary to the canonical one.

Vocabularies differ in kind, not just in spelling. This engine expresses a right
as one relation with an ordered level property. Most Zanzibar-like systems have
no ordinal level at all and model each right as a separate relation, with the
hierarchy between them living in schema rules. Both shapes map here:

    HAS_PERMISSION: {relation: has_permission, level: from_property}
    viewer:         {relation: has_permission, level: read}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from rga.domain.relations import (
    LEVEL_CARRYING,
    PermissionLevel,
    RelationType,
    parse_level,
    parse_relation,
)

#: Sentinel meaning "the level is a property of the edge, read it from the source".
FROM_PROPERTY = "from_property"


@dataclass(frozen=True)
class RelationRule:
    """How one source relation becomes a canonical one.

    `level` of None means the level is carried by the edge itself.
    """

    relation: RelationType
    level: PermissionLevel | None


@dataclass(frozen=True)
class RelationMapping:
    """The full vocabulary translation for one source."""

    rules: Mapping[str, RelationRule]

    @classmethod
    def from_config(cls, config: Mapping[str, Mapping[str, str]]) -> RelationMapping:
        """Build from the parsed `relation_mapping` section of a config file."""
        rules: dict[str, RelationRule] = {}
        for source_name, spec in config.items():
            relation = parse_relation(spec["relation"])
            raw_level = spec.get("level")
            if raw_level is None:
                level: PermissionLevel | None = PermissionLevel.NONE
            elif raw_level == FROM_PROPERTY:
                level = None
            else:
                level = parse_level(raw_level)

            if level is not PermissionLevel.NONE and relation not in LEVEL_CARRYING:
                raise ValueError(f"{relation.name} cannot carry a level (source {source_name!r})")
            rules[source_name] = RelationRule(relation=relation, level=level)
        return cls(rules=rules)

    @classmethod
    def load(cls, path: Path) -> RelationMapping:
        """Load from a YAML file holding a `relation_mapping` key."""
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_config(document["relation_mapping"])

    def translate(
        self, source_relation: str, level_property: str | None = None
    ) -> tuple[RelationType, PermissionLevel]:
        """Translate one source relation, with its level property when it has one."""
        try:
            rule = self.rules[source_relation]
        except KeyError:
            raise KeyError(f"unmapped source relation: {source_relation!r}") from None
        if rule.level is not None:
            return rule.relation, rule.level
        return rule.relation, parse_level(level_property)
