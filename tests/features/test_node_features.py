"""The per-endpoint feature block."""

import numpy as np
import pytest

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay
from rga.features.context import FeatureContext
from rga.features.nodes import NODE_BLOCK, node_features
from rga.features.spec import FeatureGroup
from rga.features.static import compute_static_attributes
from rga.util.timeutil import DAY_MS, HOUR_MS


def _events():
    return [
        GraphEvent(HOUR_MS, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:g"),
        GraphEvent(2 * HOUR_MS, EventOp.GRANT, "user:b", RelationType.MEMBER_OF, "group:g"),
        GraphEvent(
            3 * HOUR_MS, EventOp.GRANT, "group:g", RelationType.HAS_PERMISSION,
            "bucket:x", PermissionLevel.WRITE,
        ),
        GraphEvent(
            4 * HOUR_MS, EventOp.GRANT, "bucket:x", RelationType.PARENT_OF, "object:x/file.dat"
        ),
    ]


def _prepared():
    context = FeatureContext()
    for event in _events():
        context.apply(event)
    static = compute_static_attributes(replay(_events()), np.random.default_rng(0))
    return context, static


def _named(values, mask):
    return {name: (float(values[i]), bool(mask[i])) for i, name in enumerate(NODE_BLOCK.names)}


def test_block_length_matches_the_vectors() -> None:
    context, static = _prepared()
    values, mask = node_features(context, static, "user:a", now=5 * HOUR_MS)
    assert len(values) == len(NODE_BLOCK) == len(mask)
    assert values.dtype == np.float32


def test_block_declares_both_groups() -> None:
    assert FeatureGroup.STRUCTURAL in NODE_BLOCK.groups
    assert FeatureGroup.TEMPORAL in NODE_BLOCK.groups
    assert FeatureGroup.PROVENANCE not in NODE_BLOCK.groups


def test_entity_type_is_one_hot() -> None:
    context, static = _prepared()
    named = _named(*node_features(context, static, "group:g", now=5 * HOUR_MS))
    assert named["type_group"][0] == 1.0
    assert named["type_user"][0] == 0.0
    assert named["type_bucket"][0] == 0.0
    assert named["type_object"][0] == 0.0


def test_degrees_are_split_by_relation_and_direction() -> None:
    context, static = _prepared()
    named = _named(*node_features(context, static, "group:g", now=5 * HOUR_MS))
    assert named["deg_member_of_in"][0] > 0.0
    assert named["deg_has_permission_out"][0] > 0.0
    assert named["deg_member_of_out"][0] == 0.0


def test_max_level_is_scaled_into_the_unit_interval() -> None:
    # approx, not equality: values are stored as float32, so 0.4 comes back as
    # 0.40000000596 and an exact comparison against float64 arithmetic fails.
    context, static = _prepared()
    named = _named(*node_features(context, static, "group:g", now=5 * HOUR_MS))
    expected = float(PermissionLevel.WRITE) / float(PermissionLevel.ADMIN)
    assert named["max_level"][0] == pytest.approx(expected)


def test_depth_separates_buckets_from_objects() -> None:
    context, static = _prepared()
    bucket = _named(*node_features(context, static, "bucket:x", now=5 * HOUR_MS))
    obj = _named(*node_features(context, static, "object:x/file.dat", now=5 * HOUR_MS))
    assert bucket["depth"][0] == 0.0
    assert obj["depth"][0] == 1.0


def test_age_grows_with_time() -> None:
    context, static = _prepared()
    early = _named(*node_features(context, static, "user:a", now=5 * HOUR_MS))
    late = _named(*node_features(context, static, "user:a", now=5 * DAY_MS))
    assert late["age"][0] > early["age"][0]


def test_unknown_node_is_flagged_and_everything_else_masked() -> None:
    context, static = _prepared()
    values, mask = node_features(context, static, "user:ghost", now=5 * HOUR_MS)
    named = _named(values, mask)

    assert named["is_new"] == (1.0, True)
    observed = [name for name in NODE_BLOCK.names if named[name][1]]
    assert observed == ["is_new"]
    assert float(values.sum()) == 1.0


def test_nodes_absent_from_the_snapshot_mask_only_the_static_features() -> None:
    context, static = _prepared()
    context.apply(
        GraphEvent(6 * HOUR_MS, EventOp.GRANT, "user:late", RelationType.MEMBER_OF, "group:g")
    )
    named = _named(*node_features(context, static, "user:late", now=7 * HOUR_MS))

    assert named["is_new"] == (0.0, True)
    assert named["triangles"][1] is False
    assert named["clustering"][1] is False
    assert named["deg_member_of_out"][1] is True
