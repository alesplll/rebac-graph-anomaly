"""The per-change feature block.

The age of the edge itself is deliberately absent. For a candidate drawn from the
evaluation window it is zero by construction and carries nothing; what carries the
temporal signal is when in the day and week the change happened, and how recently
and how often its subject acted — and those live on the endpoint block.

Provenance is masked, not zeroed, when the initiator is unknown. Most
authorization engines do not record it, and a model given zeros would learn that
"nobody granted this" rather than "we do not know who did".
"""

from __future__ import annotations

import numpy as np

from rga.domain.entities import EntityType, entity_type, parent_bucket
from rga.domain.events import GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.features.context import FeatureContext
from rga.features.spec import FeatureBlock, FeatureGroup
from rga.features.static import StaticAttributes
from rga.util.timeutil import hour_of_day, is_weekend

#: Hop limit for the shortest-path feature. Three hops spans the whole
#: user -> group -> bucket -> object chain the engine itself traverses.
PATH_CAP = 3

#: Hours outside which activity counts as off-hours, matching the generator.
_WORKING_HOURS = (9, 19)

_STRUCTURAL_NAMES = (
    "rel_member_of",
    "rel_has_permission",
    "rel_parent_of",
    "rel_owner_of",
    "level_ordinal",
    "is_permission",
    "common_neighbours",
    "adamic_adar",
    "jaccard",
    "path_hops",
    "path_unreachable",
    "bypasses_bucket",
    "level_jump",
    "same_community",
)
_TEMPORAL_NAMES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_off_hours", "is_weekend")
_PROVENANCE_NAMES = ("actor_is_subject", "actor_level_on_object")

EDGE_BLOCK = FeatureBlock(
    names=_STRUCTURAL_NAMES + _TEMPORAL_NAMES + _PROVENANCE_NAMES,
    groups=(FeatureGroup.STRUCTURAL,) * len(_STRUCTURAL_NAMES)
    + (FeatureGroup.TEMPORAL,) * len(_TEMPORAL_NAMES)
    + (FeatureGroup.PROVENANCE,) * len(_PROVENANCE_NAMES),
)

_INDEX = {name: position for position, name in enumerate(EDGE_BLOCK.names)}


def _prior_level(context: FeatureContext, event: GraphEvent) -> PermissionLevel:
    """Strongest permission the subject already held over this target or its bucket."""
    best = context.level_on(event.subject, event.object)
    if entity_type(event.object) is EntityType.OBJECT:
        best = max(best, context.level_on(event.subject, parent_bucket(event.object)))
    return PermissionLevel(int(best))


def edge_features(
    context: FeatureContext, static: StaticAttributes, event: GraphEvent
) -> tuple[np.ndarray, np.ndarray]:
    """Feature values and observability for one change."""
    values = np.zeros(len(EDGE_BLOCK), dtype=np.float32)
    mask = np.ones(len(EDGE_BLOCK), dtype=bool)

    values[_INDEX["rel_member_of"]] = float(event.relation is RelationType.MEMBER_OF)
    values[_INDEX["rel_has_permission"]] = float(event.relation is RelationType.HAS_PERMISSION)
    values[_INDEX["rel_parent_of"]] = float(event.relation is RelationType.PARENT_OF)
    values[_INDEX["rel_owner_of"]] = float(event.relation is RelationType.OWNER_OF)
    values[_INDEX["level_ordinal"]] = float(event.level) / float(PermissionLevel.ADMIN)
    values[_INDEX["is_permission"]] = float(event.relation is RelationType.HAS_PERMISSION)

    values[_INDEX["common_neighbours"]] = np.log1p(
        context.common_neighbours(event.subject, event.object)
    )
    values[_INDEX["adamic_adar"]] = np.log1p(context.adamic_adar(event.subject, event.object))
    values[_INDEX["jaccard"]] = context.jaccard(event.subject, event.object)

    hops = context.path_length(event.subject, event.object, cap=PATH_CAP)
    if hops is None:
        values[_INDEX["path_hops"]] = 1.0
        values[_INDEX["path_unreachable"]] = 1.0
    else:
        values[_INDEX["path_hops"]] = hops / PATH_CAP
        values[_INDEX["path_unreachable"]] = 0.0

    if entity_type(event.object) is EntityType.OBJECT:
        holds_bucket = context.level_on(event.subject, parent_bucket(event.object))
        values[_INDEX["bypasses_bucket"]] = float(holds_bucket is PermissionLevel.NONE)
    else:
        # Nothing contains a bucket, so the question does not arise.
        mask[_INDEX["bypasses_bucket"]] = False

    jump = int(event.level) - int(_prior_level(context, event))
    values[_INDEX["level_jump"]] = jump / float(PermissionLevel.ADMIN)

    subject_community = static.community.get(event.subject)
    object_community = static.community.get(event.object)
    if subject_community is None or object_community is None:
        mask[_INDEX["same_community"]] = False
    else:
        values[_INDEX["same_community"]] = float(subject_community == object_community)

    hour = hour_of_day(event.ts)
    weekday = (event.ts // 86_400_000 + 3) % 7  # 1970-01-01 was a Thursday.
    values[_INDEX["hour_sin"]] = np.sin(2.0 * np.pi * hour / 24.0)
    values[_INDEX["hour_cos"]] = np.cos(2.0 * np.pi * hour / 24.0)
    values[_INDEX["dow_sin"]] = np.sin(2.0 * np.pi * weekday / 7.0)
    values[_INDEX["dow_cos"]] = np.cos(2.0 * np.pi * weekday / 7.0)
    values[_INDEX["is_off_hours"]] = float(not _WORKING_HOURS[0] <= hour < _WORKING_HOURS[1])
    values[_INDEX["is_weekend"]] = float(is_weekend(event.ts))

    if event.actor is None:
        mask[_INDEX["actor_is_subject"]] = False
        mask[_INDEX["actor_level_on_object"]] = False
    else:
        values[_INDEX["actor_is_subject"]] = float(event.actor == event.subject)
        values[_INDEX["actor_level_on_object"]] = float(
            context.level_on(event.actor, event.object)
        ) / float(PermissionLevel.ADMIN)

    return values, mask
