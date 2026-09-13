"""The per-change feature block."""

import numpy as np

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay
from rga.features.context import FeatureContext
from rga.features.edges import EDGE_BLOCK, edge_features
from rga.features.spec import FeatureGroup
from rga.features.static import compute_static_attributes
from rga.util.timeutil import DAY_MS, HOUR_MS

# 2025-01-01T00:00:00Z was a Wednesday.
WEDNESDAY = 1_735_689_600_000


def _history():
    return [
        GraphEvent(WEDNESDAY, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:g"),
        GraphEvent(WEDNESDAY + 1, EventOp.GRANT, "user:b", RelationType.MEMBER_OF, "group:g"),
        GraphEvent(
            WEDNESDAY + 2, EventOp.GRANT, "group:g", RelationType.HAS_PERMISSION,
            "bucket:x", PermissionLevel.READ,
        ),
        GraphEvent(
            WEDNESDAY + 3, EventOp.GRANT, "bucket:x", RelationType.PARENT_OF, "object:x/file.dat"
        ),
    ]


def _prepared():
    context = FeatureContext()
    for event in _history():
        context.apply(event)
    static = compute_static_attributes(replay(_history()), np.random.default_rng(0))
    return context, static


def _named(values, mask):
    return {name: (float(values[i]), bool(mask[i])) for i, name in enumerate(EDGE_BLOCK.names)}


def _event(subject, target, level=PermissionLevel.READ, actor="user:root", ts=WEDNESDAY + 10):
    return GraphEvent(
        ts, EventOp.GRANT, subject, RelationType.HAS_PERMISSION, target, level, actor
    )


def test_block_length_matches_the_vectors() -> None:
    context, static = _prepared()
    values, mask = edge_features(
        context, static, _event("user:a", "bucket:x", PermissionLevel.ADMIN, "user:a")
    )
    assert len(values) == len(EDGE_BLOCK) == len(mask)
    assert values.dtype == np.float32


def test_block_declares_all_three_groups() -> None:
    assert set(EDGE_BLOCK.groups) == {
        FeatureGroup.STRUCTURAL,
        FeatureGroup.TEMPORAL,
        FeatureGroup.PROVENANCE,
    }


def test_relation_is_one_hot_and_level_is_scaled() -> None:
    context, static = _prepared()
    named = _named(
        *edge_features(context, static, _event("user:a", "bucket:x", PermissionLevel.ADMIN))
    )
    assert named["rel_has_permission"][0] == 1.0
    assert named["rel_member_of"][0] == 0.0
    assert named["level_ordinal"][0] == 1.0
    assert named["is_permission"][0] == 1.0


def test_similarity_reflects_shared_structure() -> None:
    context, static = _prepared()
    near = _named(*edge_features(context, static, _event("user:a", "bucket:x")))
    far = _named(*edge_features(context, static, _event("user:a", "bucket:unrelated")))
    assert near["common_neighbours"][0] > far["common_neighbours"][0]


def test_path_hops_and_unreachable_flag() -> None:
    context, static = _prepared()
    named = _named(*edge_features(context, static, _event("user:a", "bucket:x")))
    assert named["path_unreachable"][0] == 0.0
    assert 0.0 < named["path_hops"][0] <= 1.0

    named = _named(*edge_features(context, static, _event("user:a", "bucket:elsewhere")))
    assert named["path_unreachable"][0] == 1.0


def test_bypass_flag_is_only_meaningful_for_objects() -> None:
    context, static = _prepared()
    named = _named(
        *edge_features(
            context, static, _event("user:a", "object:x/file.dat", PermissionLevel.WRITE)
        )
    )
    # user:a holds nothing on bucket:x directly, so this grant skips the bucket.
    assert named["bypasses_bucket"] == (1.0, True)

    named = _named(
        *edge_features(context, static, _event("user:a", "bucket:x", PermissionLevel.WRITE))
    )
    assert named["bypasses_bucket"][1] is False


def test_level_jump_measures_the_step_up() -> None:
    context, static = _prepared()
    context.apply(_event("user:a", "bucket:x", PermissionLevel.READ, ts=WEDNESDAY + 5))
    named = _named(
        *edge_features(context, static, _event("user:a", "bucket:x", PermissionLevel.ADMIN))
    )
    assert named["level_jump"][0] > 0.0


def test_cyclical_time_encoding_and_calendar_flags() -> None:
    context, static = _prepared()
    # Wednesday 03:00 UTC — off hours, a weekday.
    named = _named(
        *edge_features(
            context, static, _event("user:a", "bucket:x", ts=WEDNESDAY + 3 * HOUR_MS)
        )
    )
    assert -1.0 <= named["hour_sin"][0] <= 1.0
    assert -1.0 <= named["dow_cos"][0] <= 1.0
    assert named["is_off_hours"][0] == 1.0
    assert named["is_weekend"][0] == 0.0

    named = _named(
        *edge_features(
            context, static,
            _event("user:a", "bucket:x", ts=WEDNESDAY + 3 * DAY_MS + 12 * HOUR_MS),
        )
    )
    assert named["is_weekend"][0] == 1.0
    assert named["is_off_hours"][0] == 0.0


def test_provenance_is_masked_when_the_actor_is_unknown() -> None:
    context, static = _prepared()
    named = _named(
        *edge_features(context, static, _event("user:a", "bucket:x", actor=None))
    )
    assert named["actor_is_subject"] == (0.0, False)
    assert named["actor_level_on_object"] == (0.0, False)


def test_self_grant_is_visible_when_the_actor_is_known() -> None:
    context, static = _prepared()
    named = _named(
        *edge_features(
            context, static, _event("user:a", "bucket:x", PermissionLevel.ADMIN, "user:a")
        )
    )
    assert named["actor_is_subject"] == (1.0, True)
