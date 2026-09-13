"""Replay: a graph is always a journal evaluated up to a point in time."""

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay


def _journal() -> list[GraphEvent]:
    return [
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(
            200,
            EventOp.GRANT,
            "group:devops",
            RelationType.HAS_PERMISSION,
            "bucket:photos",
            PermissionLevel.WRITE,
        ),
        GraphEvent(300, EventOp.REVOKE, "user:alice", RelationType.MEMBER_OF, "group:devops"),
    ]


def test_replay_of_the_whole_journal() -> None:
    graph = replay(_journal())
    assert graph.num_edges == 1


def test_replay_stops_at_the_cutoff_inclusive() -> None:
    graph = replay(_journal(), until=200)
    assert graph.num_edges == 2


def test_cutoff_before_everything_gives_an_empty_graph() -> None:
    graph = replay(_journal(), until=50)
    assert graph.num_nodes == 0
    assert graph.num_edges == 0


def test_replay_does_not_consume_events_past_the_cutoff() -> None:
    consumed: list[int] = []

    def counting_journal():
        for event in _journal():
            consumed.append(event.ts)
            yield event

    replay(counting_journal(), until=100)
    # The 200 event is inspected to discover it is past the cutoff, 300 never is.
    assert consumed == [100, 200]
