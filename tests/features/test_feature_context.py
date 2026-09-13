"""The incremental graph state features are read from."""

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.features.context import FeatureContext
from rga.util.timeutil import DAY_MS, HOUR_MS


def _grant(ts, subject, relation, target, level=PermissionLevel.NONE, actor=None):
    return GraphEvent(ts, EventOp.GRANT, subject, relation, target, level, actor)


def _revoke(ts, subject, relation, target):
    return GraphEvent(ts, EventOp.REVOKE, subject, relation, target)


def _context(*events) -> FeatureContext:
    context = FeatureContext()
    for event in events:
        context.apply(event)
    return context


def test_unknown_node_is_reported_as_unknown() -> None:
    assert FeatureContext().knows("user:ghost") is False


def test_grant_registers_both_endpoints() -> None:
    context = _context(_grant(1, "user:a", RelationType.MEMBER_OF, "group:g"))
    assert context.knows("user:a")
    assert context.knows("group:g")


def test_degree_is_counted_per_relation_and_direction() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.MEMBER_OF, "group:g"),
        _grant(2, "user:b", RelationType.MEMBER_OF, "group:g"),
        _grant(3, "user:a", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
    )
    assert context.degree("user:a", RelationType.MEMBER_OF) == 1
    assert context.degree("group:g", RelationType.MEMBER_OF, incoming=True) == 2
    assert context.degree("user:a", RelationType.HAS_PERMISSION) == 1
    assert context.degree("user:a", RelationType.PARENT_OF) == 0


def test_regrant_does_not_inflate_degree() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
        _grant(2, "user:a", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.ADMIN),
    )
    assert context.degree("user:a", RelationType.HAS_PERMISSION) == 1
    assert context.level_on("user:a", "bucket:x") is PermissionLevel.ADMIN


def test_revoke_removes_the_edge_and_its_level() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
        _revoke(2, "user:a", RelationType.HAS_PERMISSION, "bucket:x"),
    )
    assert context.degree("user:a", RelationType.HAS_PERMISSION) == 0
    assert context.level_on("user:a", "bucket:x") is PermissionLevel.NONE


def test_undirected_link_survives_while_another_relation_holds_it() -> None:
    # Two relations connect the same pair; removing one must not sever the pair.
    context = _context(
        _grant(1, "user:a", RelationType.MEMBER_OF, "group:g"),
        _grant(2, "user:a", RelationType.HAS_PERMISSION, "group:g", PermissionLevel.READ),
        _revoke(3, "user:a", RelationType.MEMBER_OF, "group:g"),
    )
    assert "group:g" in context.neighbours("user:a")

    context.apply(_revoke(4, "user:a", RelationType.HAS_PERMISSION, "group:g"))
    assert "group:g" not in context.neighbours("user:a")


def test_common_neighbours_and_similarity_indices() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.MEMBER_OF, "group:shared"),
        _grant(2, "user:b", RelationType.MEMBER_OF, "group:shared"),
        _grant(3, "user:a", RelationType.MEMBER_OF, "group:only-a"),
    )
    assert context.common_neighbours("user:a", "user:b") == 1
    assert context.jaccard("user:a", "user:b") == 0.5
    assert context.adamic_adar("user:a", "user:b") > 0.0


def test_similarity_of_unknown_nodes_is_zero() -> None:
    context = _context(_grant(1, "user:a", RelationType.MEMBER_OF, "group:g"))
    assert context.common_neighbours("user:a", "user:ghost") == 0
    assert context.jaccard("user:a", "user:ghost") == 0.0
    assert context.adamic_adar("user:a", "user:ghost") == 0.0


def test_path_length_walks_the_undirected_projection() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.MEMBER_OF, "group:g"),
        _grant(2, "group:g", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
    )
    assert context.path_length("user:a", "group:g") == 1
    assert context.path_length("user:a", "bucket:x") == 2


def test_path_length_returns_none_beyond_the_cap_or_when_disconnected() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.MEMBER_OF, "group:g"),
        _grant(2, "group:g", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
        _grant(3, "user:z", RelationType.MEMBER_OF, "group:other"),
    )
    assert context.path_length("user:a", "bucket:x", cap=1) is None
    assert context.path_length("user:a", "user:z") is None
    assert context.path_length("user:a", "user:ghost") is None


def test_max_level_reflects_the_strongest_outgoing_permission() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
        _grant(2, "user:a", RelationType.HAS_PERMISSION, "bucket:y", PermissionLevel.DELETE),
    )
    assert context.max_level_of("user:a") is PermissionLevel.DELETE
    assert context.max_level_of("user:ghost") is PermissionLevel.NONE


def test_group_count_counts_memberships_only() -> None:
    context = _context(
        _grant(1, "user:a", RelationType.MEMBER_OF, "group:g"),
        _grant(2, "user:a", RelationType.MEMBER_OF, "group:h"),
        _grant(3, "user:a", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
    )
    assert context.group_count("user:a") == 2


def test_creation_time_is_first_sight() -> None:
    context = _context(
        _grant(100, "user:a", RelationType.MEMBER_OF, "group:g"),
        _grant(500, "user:a", RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ),
    )
    assert context.created("user:a") == 100
    assert context.created("bucket:x") == 500
    assert context.created("user:ghost") is None


def test_last_change_tracks_the_subject_and_the_actor() -> None:
    context = _context(
        _grant(100, "user:a", RelationType.MEMBER_OF, "group:g", actor="user:root"),
        _grant(300, "user:b", RelationType.MEMBER_OF, "group:g", actor="user:root"),
    )
    assert context.last_change("user:a") == 100
    assert context.last_change("user:root") == 300
    assert context.last_change("user:ghost") is None


def test_activity_counts_inside_a_window() -> None:
    context = _context(
        _grant(1 * HOUR_MS, "user:a", RelationType.MEMBER_OF, "group:g"),
        _grant(2 * HOUR_MS, "user:a", RelationType.MEMBER_OF, "group:h"),
        _grant(30 * HOUR_MS, "user:a", RelationType.MEMBER_OF, "group:i"),
    )
    now = 30 * HOUR_MS
    assert context.activity("user:a", HOUR_MS, now) == 1
    assert context.activity("user:a", DAY_MS, now) == 1
    assert context.activity("user:a", 7 * DAY_MS, now) == 3
    assert context.activity("user:ghost", DAY_MS, now) == 0
