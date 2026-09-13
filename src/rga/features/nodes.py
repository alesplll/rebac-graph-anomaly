"""The per-endpoint feature block.

Degrees are kept separate per relation and direction on purpose. Membership in a
group and a permission on a resource are different kinds of link, and summing
them into one degree throws away the distinction the whole model rests on.

Counts are put through log1p because degree distributions here are heavy-tailed:
a hub bucket with four hundred objects would otherwise dominate every linear
model in the baseline suite.
"""

from __future__ import annotations

import numpy as np

from rga.domain.entities import EntityType, entity_type
from rga.domain.relations import PermissionLevel, RelationType
from rga.features.context import FeatureContext
from rga.features.spec import FeatureBlock, FeatureGroup
from rga.features.static import StaticAttributes
from rga.util.timeutil import DAY_MS, HOUR_MS

_STRUCTURAL_NAMES = (
    "type_user",
    "type_group",
    "type_bucket",
    "type_object",
    "deg_member_of_out",
    "deg_member_of_in",
    "deg_has_permission_out",
    "deg_has_permission_in",
    "deg_parent_of_out",
    "deg_parent_of_in",
    "deg_owner_of_out",
    "deg_owner_of_in",
    "max_level",
    "group_count",
    "depth",
    "triangles",
    "clustering",
    "same_community_share",
    "is_new",
)
_TEMPORAL_NAMES = ("age", "gap", "activity_1h", "activity_24h", "activity_7d")

NODE_BLOCK = FeatureBlock(
    names=_STRUCTURAL_NAMES + _TEMPORAL_NAMES,
    groups=(FeatureGroup.STRUCTURAL,) * len(_STRUCTURAL_NAMES)
    + (FeatureGroup.TEMPORAL,) * len(_TEMPORAL_NAMES),
)

_INDEX = {name: position for position, name in enumerate(NODE_BLOCK.names)}

#: The one feature that stays observed for a node the context has never seen.
_IS_NEW = _INDEX["is_new"]
#: Attributes that only exist for nodes present in the snapshot.
_STATIC_FEATURES = ("triangles", "clustering", "same_community_share")

_DEPTH_BY_TYPE = {
    EntityType.USER: 0.0,
    EntityType.GROUP: 0.0,
    EntityType.BUCKET: 0.0,
    EntityType.OBJECT: 1.0,
}


def node_features(
    context: FeatureContext, static: StaticAttributes, node_id: str, now: int
) -> tuple[np.ndarray, np.ndarray]:
    """Feature values and observability for one endpoint."""
    values = np.zeros(len(NODE_BLOCK), dtype=np.float32)
    mask = np.zeros(len(NODE_BLOCK), dtype=bool)

    if not context.knows(node_id):
        # A node nobody has linked yet. Everything except the flag is unknown,
        # and unknown must not be confused with zero.
        values[_IS_NEW] = 1.0
        mask[_IS_NEW] = True
        return values, mask

    mask[:] = True
    kind = entity_type(node_id)

    values[_INDEX["type_user"]] = float(kind is EntityType.USER)
    values[_INDEX["type_group"]] = float(kind is EntityType.GROUP)
    values[_INDEX["type_bucket"]] = float(kind is EntityType.BUCKET)
    values[_INDEX["type_object"]] = float(kind is EntityType.OBJECT)

    for relation in RelationType:
        stem = f"deg_{relation.name.lower()}"
        values[_INDEX[f"{stem}_out"]] = np.log1p(context.degree(node_id, relation))
        values[_INDEX[f"{stem}_in"]] = np.log1p(context.degree(node_id, relation, incoming=True))

    values[_INDEX["max_level"]] = float(context.max_level_of(node_id)) / float(
        PermissionLevel.ADMIN
    )
    values[_INDEX["group_count"]] = np.log1p(context.group_count(node_id))
    values[_INDEX["depth"]] = _DEPTH_BY_TYPE[kind]

    if node_id in static.community:
        values[_INDEX["triangles"]] = np.log1p(static.triangles[node_id])
        values[_INDEX["clustering"]] = static.clustering[node_id]
        values[_INDEX["same_community_share"]] = static.same_community_share[node_id]
    else:
        # Appeared after the snapshot was taken; its structure there is unknown.
        for name in _STATIC_FEATURES:
            mask[_INDEX[name]] = False

    values[_IS_NEW] = 0.0

    created = context.created(node_id)
    values[_INDEX["age"]] = np.log1p(max(now - created, 0)) if created is not None else 0.0
    last = context.last_change(node_id)
    values[_INDEX["gap"]] = np.log1p(max(now - last, 0)) if last is not None else 0.0
    values[_INDEX["activity_1h"]] = np.log1p(context.activity(node_id, HOUR_MS, now))
    values[_INDEX["activity_24h"]] = np.log1p(context.activity(node_id, DAY_MS, now))
    values[_INDEX["activity_7d"]] = np.log1p(context.activity(node_id, 7 * DAY_MS, now))

    return values, mask
