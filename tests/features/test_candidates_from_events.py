"""The candidate core runs on any journal, labelled or not."""

from pathlib import Path

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
