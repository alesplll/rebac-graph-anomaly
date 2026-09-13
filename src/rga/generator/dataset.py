"""Dataset assembly.

A dataset is a normal journal, a cut in time, and a set of anomalies planted in
the span after the cut. The training graph is everything before the cut; the
edges to be scored are the grants after it.

One subtlety decides whether every metric downstream is meaningful: an injected
edge must not also be produced by the normal process. If the same edge key
appears both as ground truth and as ordinary traffic, the label is ambiguous and
no measurement made on it means anything. Such injections are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import rga.generator.anomalies  # noqa: F401  registers every pattern
from rga.domain.events import EventOp, GraphEvent
from rga.domain.replay import replay
from rga.generator.anomalies.base import (
    AnomalyLabel,
    InjectionContext,
    NoCandidateError,
    get_pattern,
)
from rga.generator.config import DatasetConfig
from rga.generator.org import Organization, build_organization
from rga.generator.timeline import generate_normal_journal
from rga.util.timeutil import DAY_MS

#: How many injection attempts per wanted anomaly before giving up on a pattern.
_INJECTION_ATTEMPTS = 40


@dataclass(frozen=True)
class Dataset:
    """A reproducible experiment input."""

    config: DatasetConfig
    events: tuple[GraphEvent, ...]
    labels: tuple[AnomalyLabel, ...]
    #: End of the training span and start of the evaluation window.
    split_ts: int
    window_end: int

    def window_events(self) -> tuple[GraphEvent, ...]:
        """Events inside the evaluation window — the edges to be scored."""
        return tuple(
            event for event in self.events if self.split_ts <= event.ts < self.window_end
        )

    def anomaly_keys(self) -> set[tuple[str, int, str]]:
        return {label.edge_key() for label in self.labels}


def build_dataset(config: DatasetConfig) -> Dataset:
    """Generate a complete dataset from its recipe."""
    rng = np.random.default_rng(config.seed)
    org = build_organization(config.org, rng)
    normal = generate_normal_journal(
        org, config.timeline, rng, exception_rate=config.org.legitimate_exception_rate
    )

    start = config.timeline.start_ts
    window_end = start + config.timeline.days * DAY_MS
    split_ts = window_end - config.eval_window_days * DAY_MS

    normal_keys = {event.edge_key() for event in normal}
    window_grants = sum(
        1 for event in normal if event.op is EventOp.GRANT and split_ts <= event.ts < window_end
    )
    # A target, not a guarantee: every configured pattern is injected at least
    # once first, and a single incident of a burst pattern creates a dozen edges.
    # On a small dataset that alone can exceed the target share; dataset_stats
    # reports what was actually achieved.
    wanted = max(1, round(config.anomalies.rate * max(window_grants, 1)))

    injected, labels = _inject(
        config=config,
        org=org,
        rng=rng,
        journal=normal,
        window=(split_ts, window_end),
        cutoff=split_ts,
        wanted=wanted,
        forbidden=normal_keys,
    )

    if config.anomalies.train_contamination > 0.0:
        train_grants = sum(
            1 for event in normal if event.op is EventOp.GRANT and event.ts < split_ts
        )
        contaminated, _ = _inject(
            config=config,
            org=org,
            rng=rng,
            journal=normal,
            window=(start + DAY_MS, split_ts),
            cutoff=start + DAY_MS,
            wanted=max(1, round(config.anomalies.train_contamination * train_grants)),
            forbidden=normal_keys | {label.edge_key() for label in labels},
        )
        # Contamination is deliberately unlabelled: it exists to dirty the
        # training graph, not to be scored.
        injected.extend(contaminated)

    events = sorted([*normal, *injected], key=lambda event: event.ts)
    return Dataset(
        config=config,
        events=tuple(events),
        labels=tuple(sorted(labels, key=lambda label: label.ts)),
        split_ts=split_ts,
        window_end=window_end,
    )


def _inject(
    *,
    config: DatasetConfig,
    org: Organization,
    rng: np.random.Generator,
    journal: list[GraphEvent],
    window: tuple[int, int],
    cutoff: int,
    wanted: int,
    forbidden: set[tuple[str, int, str]],
) -> tuple[list[GraphEvent], list[AnomalyLabel]]:
    """Plant anomalies until `wanted` labelled edges exist or attempts run out."""
    if not config.anomalies.patterns:
        return [], []

    graph = replay(journal, until=cutoff)
    events: list[GraphEvent] = []
    labels: list[AnomalyLabel] = []
    taken = set(forbidden)
    exhausted: set[str] = set()
    names = list(config.anomalies.patterns)

    def attempt(name: str) -> bool:
        """Try one injection of a pattern. True when it landed."""
        context = InjectionContext(rng=rng, org=org, graph=graph, window=window)
        try:
            injection = get_pattern(name).inject(context)
        except NoCandidateError:
            exhausted.add(name)
            return False

        keys = {event.edge_key() for event in injection.events}
        if keys & taken:
            return False

        taken.update(keys)
        events.extend(injection.events)
        labels.extend(injection.labels)
        return True

    # First pass: one incident of every configured pattern. Coverage of the
    # threat catalogue comes before the target share, because a dataset missing a
    # pattern cannot measure detection of that pattern at all — and patterns
    # differ enormously in how many edges one incident creates, so chasing the
    # share first can satisfy it with two patterns out of eight.
    for name in names:
        for _ in range(_INJECTION_ATTEMPTS):
            if name in exhausted or attempt(name):
                break

    # Second pass: top up towards the target share, if the first pass fell short.
    for _ in range(wanted * _INJECTION_ATTEMPTS):
        if len(labels) >= wanted:
            break
        available = [name for name in names if name not in exhausted]
        if not available:
            break
        attempt(available[int(rng.integers(len(available)))])

    return events, labels
