"""The candidate core runs on any journal, labelled or not."""

from pathlib import Path

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay
from rga.features.build import Span, build_candidates, candidates_from_events
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.util.timeutil import DAY_MS

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def test_it_reproduces_what_the_dataset_path_produces() -> None:
    dataset = build_dataset(CONFIG)
    expected = build_candidates(dataset, Span.EVAL)

    graph = replay(dataset.events, until=dataset.split_ts)
    produced = candidates_from_events(
        dataset.events,
        start=dataset.split_ts,
        end=dataset.window_end,
        graph=graph,
        labels={(label.edge_key(), label.ts): label.pattern for label in dataset.labels},
    )

    assert produced.keys == expected.keys
    assert produced.matrix.values.shape == expected.matrix.values.shape
    assert (produced.matrix.values == expected.matrix.values).all()
    assert (produced.labels == expected.labels).all()


def test_an_unlabelled_journal_yields_candidates_with_no_positives() -> None:
    dataset = build_dataset(CONFIG)
    graph = replay(dataset.events, until=dataset.split_ts)

    produced = candidates_from_events(
        dataset.events,
        start=dataset.split_ts,
        end=dataset.window_end,
        graph=graph,
    )

    assert produced.n_candidates > 0
    assert int(produced.labels.sum()) == 0
    assert set(produced.patterns) == {""}


def test_a_narrower_window_yields_fewer_candidates() -> None:
    dataset = build_dataset(CONFIG)
    graph = replay(dataset.events, until=dataset.split_ts)

    wide = candidates_from_events(
        dataset.events, start=dataset.split_ts, end=dataset.window_end, graph=graph
    )
    narrow = candidates_from_events(
        dataset.events,
        start=dataset.window_end - DAY_MS,
        end=dataset.window_end,
        graph=graph,
    )

    assert narrow.n_candidates < wide.n_candidates


def test_changes_made_by_platform_automation_are_not_candidates() -> None:
    """The platform writing a permission after an upload is not anyone's decision."""
    grant = PermissionLevel.WRITE
    events = [
        GraphEvent(
            1, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION, "bucket:x", grant, "user:root"
        ),
        GraphEvent(
            2,
            EventOp.GRANT,
            "user:a",
            RelationType.HAS_PERMISSION,
            "bucket:y",
            grant,
            "user:system",
        ),
        GraphEvent(
            3,
            EventOp.GRANT,
            "user:b",
            RelationType.HAS_PERMISSION,
            "bucket:x",
            grant,
            "user:ops-bot",
        ),
    ]

    default = candidates_from_events(events, start=0, end=10, graph=replay([]))
    custom = candidates_from_events(
        events,
        start=0,
        end=10,
        graph=replay([]),
        automation_actors=frozenset({"user:system", "user:ops-bot"}),
    )

    assert [key[2] for key in default.keys] == ["bucket:x", "bucket:x"]
    assert default.actors == ("user:root", "user:ops-bot")
    assert custom.actors == ("user:root",)
