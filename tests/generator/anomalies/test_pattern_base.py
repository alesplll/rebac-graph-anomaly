"""Pattern registry and the helpers every pattern uses."""

import numpy as np
import pytest

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay
from rga.generator.anomalies.base import (
    AnomalyLabel,
    get_pattern,
    label_for,
    last_activity,
    level_on,
    pick,
    sample_night_ts,
    sample_ts,
)
from rga.util.timeutil import DAY_MS, hour_of_day

WINDOW = (1_735_689_600_000, 1_735_689_600_000 + 14 * DAY_MS)


def _graph():
    return replay(
        [
            GraphEvent(
                1_000, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
                "bucket:x", PermissionLevel.READ,
            ),
            GraphEvent(
                2_000, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
                "bucket:y", PermissionLevel.ADMIN,
            ),
        ]
    )


def test_level_on_finds_a_direct_permission() -> None:
    assert level_on(_graph(), "user:a", "bucket:x") is PermissionLevel.READ


def test_level_on_returns_none_for_absent_nodes_or_edges() -> None:
    graph = _graph()
    assert level_on(graph, "user:a", "bucket:z") is PermissionLevel.NONE
    assert level_on(graph, "user:ghost", "bucket:x") is PermissionLevel.NONE


def test_last_activity_is_the_newest_outgoing_edge() -> None:
    assert last_activity(_graph(), "user:a") == 2_000


def test_last_activity_of_an_unknown_node_is_minus_one() -> None:
    assert last_activity(_graph(), "user:ghost") == -1


def test_pick_returns_a_member_of_the_sequence() -> None:
    rng = np.random.default_rng(0)
    options = ["a", "b", "c"]
    assert all(pick(rng, options) in options for _ in range(20))


def test_sample_ts_stays_inside_the_window() -> None:
    rng = np.random.default_rng(0)
    assert all(WINDOW[0] <= sample_ts(rng, WINDOW) < WINDOW[1] for _ in range(200))


def test_sample_night_ts_lands_in_the_small_hours() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        ts = sample_night_ts(rng, WINDOW)
        assert WINDOW[0] <= ts < WINDOW[1]
        assert hour_of_day(ts) < 5


def test_label_for_mirrors_its_event() -> None:
    event = GraphEvent(
        5_000, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.ADMIN, actor="user:a",
    )
    label = label_for(event, "demo")
    assert label == AnomalyLabel(5_000, "user:a", RelationType.HAS_PERMISSION, "bucket:x", "demo")
    assert label.edge_key() == event.edge_key()


def test_label_round_trips_through_a_dict() -> None:
    label = AnomalyLabel(5_000, "user:a", RelationType.HAS_PERMISSION, "bucket:x", "demo")
    assert AnomalyLabel.from_dict(label.to_dict()) == label


def test_unknown_pattern_is_reported_by_name() -> None:
    with pytest.raises(KeyError, match="unknown anomaly pattern"):
        get_pattern("does_not_exist")
