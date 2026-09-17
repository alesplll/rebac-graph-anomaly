"""Turning whatever the source offers into a ranked queue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from rga.adapters.base import Capabilities
from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph
from rga.domain.replay import replay
from rga.explain.incident import incident_id
from rga.features.build import candidates_from_events
from rga.features.spec import CandidateSet
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.service.config import ServiceConfig
from rga.util.timeutil import DAY_MS

_Loaded = tuple[tuple[GraphEvent, ...], AccessGraph, int, int, Capabilities]


@dataclass(frozen=True)
class Analysis:
    """One pass over a window: what was scored and in what order."""

    candidates: CandidateSet
    scores: np.ndarray
    order: np.ndarray
    source: str
    level: int
    window: tuple[int, int]
    refreshed_at: str

    def find(self, wanted: str) -> int | None:
        """Position of an incident by its identifier, or None."""
        for position in range(self.candidates.n_candidates):
            key = self.candidates.keys[position]
            if incident_id(key, int(self.candidates.ts[position])) == wanted:
                return position
        return None


def _synthetic(config: ServiceConfig) -> _Loaded:
    assert config.dataset is not None
    dataset = build_dataset(load_dataset_config(config.dataset))
    end = dataset.window_end
    start = end - config.window_days * DAY_MS
    return (
        dataset.events,
        replay(dataset.events, until=start),
        start,
        end,
        Capabilities(timestamps=True, provenance=True, change_log=True),
    )


def _live(config: ServiceConfig) -> _Loaded:
    from rga.adapters.mapping import RelationMapping
    from rga.adapters.neo4j_source import BoltReader, Neo4jSource

    reader = BoltReader(config.uri, config.user, config.password)
    source = Neo4jSource(reader, RelationMapping.load(config.mapping))
    events = tuple(source.events())
    if not events:
        raise ValueError("the live graph is empty; fill it before pointing the service at it")

    end = int(events[-1].ts) + 1
    start = end - config.window_days * DAY_MS
    return events, source.snapshot(at=start), start, end, source.capabilities()


def analyse(config: ServiceConfig, scorer) -> Analysis:
    """Read the source, build the window's candidates and rank them."""
    events, graph, start, end, capabilities = (
        _synthetic(config) if config.source == "synthetic" else _live(config)
    )
    candidates = candidates_from_events(events, start=start, end=end, graph=graph)
    scores = (
        scorer.score(candidates) if candidates.n_candidates else np.zeros(0, dtype=np.float64)
    )
    return Analysis(
        candidates=candidates,
        scores=scores,
        order=np.argsort(-scores, kind="stable"),
        source=config.source,
        level=capabilities.level,
        window=(start, end),
        refreshed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
