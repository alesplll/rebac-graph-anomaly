# Module 4: Service, Explanation and Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the measuring stand into a working system: a trained model saved as an
artefact, a service that ranks suspicious permission changes and explains each one,
and a single-page interface an analyst reads — running on synthetic data and on a
live `opens3-rebac` graph without a line of code changing between them.

**Architecture:** A source yields a journal — the synthetic generator directly, a
Neo4j snapshot by reconstructing one from edge timestamps. The candidate builder is
refactored so it consumes that journal rather than a synthetic `Dataset`. A saved
artefact holds the trained scorer. The service scores a window, and for any one
change explains the score two ways: gradient times input over the features, and edge
masking over the two-hop neighbourhood. The page draws the result, subgraph included,
with no build pipeline and no third-party JavaScript.

**Tech Stack:** Python 3.12 under `uv`, FastAPI and uvicorn (the `service` extra),
PyTorch for the attribution gradients, the Neo4j driver (the `neo4j` extra), plain
HTML/CSS/JavaScript with hand-written SVG.

**Spec:** `docs/superpowers/specs/2026-09-17-module-4-service-and-interface.md`, which
develops sections 10, 11 and 12 of
`docs/superpowers/specs/2026-09-12-rebac-graph-anomaly-design.md`.

## Global Constraints

Every task's requirements implicitly include this section.

- Python pinned to `3.12` under `uv`. The service needs `--extra service`, the live
  source `--extra neo4j`, the network `--extra cpu` (laptop) or `--extra gpu` (PC).
- **The system supports a decision, it never pronounces one.** Every user-facing
  string — API field, template sentence, button label — says "suspicious", "unusual",
  "worth a look". Words like "compromise", "attack", "intruder" are forbidden in
  output, and a test enforces it.
- Nothing above the adapter layer knows the data came from Neo4j. No `grpcio`, no
  generated protobuf, no `opens3-rebac` import, ever.
- Every random draw through an explicitly seeded `numpy.random.Generator` or
  `torch.Generator`.
- Paths through `pathlib`; `\n` newlines written explicitly; every entry point under
  `if __name__ == "__main__":`.
- Comments, docstrings and commit messages in English. **No AI attribution lines.**
  After every commit, `git push origin main`.
- `uv run ruff check .` clean, line length 100.
- `uv run pytest -m "not integration and not gpu"` green after every task. Tests that
  need a live Neo4j carry `@pytest.mark.integration`.
- No new experiment series. Anything that needs a measurement run belongs to module 5.

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/rga/explain/__init__.py` | Package marker |
| `src/rga/explain/features.py` | Gradient-times-input attribution over head inputs |
| `src/rga/explain/structure.py` | Edge importance by masking the neighbourhood |
| `src/rga/explain/text.py` | Template sentences describing one change |
| `src/rga/explain/incident.py` | Stable incident id and the assembled payload |
| `src/rga/service/__init__.py` | Package marker |
| `src/rga/service/config.py` | Service configuration loaded from YAML |
| `src/rga/service/analysis.py` | Source to candidates to scores, held for the API |
| `src/rga/service/app.py` | FastAPI application and the five routes |
| `src/rga/artifacts.py` | Saving and loading a fitted scorer |
| `configs/service/synthetic.yaml` | Service against a generated dataset |
| `configs/service/opens3.yaml` | Service against a live Neo4j |
| `configs/train/gnn-supervised.yaml` | Training recipe for the shipped artefact |
| `web/index.html`, `web/app.js`, `web/style.css` | The single page |
| `scripts/fill_live_graph.py` | Replay a synthetic journal into Neo4j for the demo |
| `deploy/docker-compose.yml` | Service next to a running opens3-rebac |
| `tests/explain/*`, `tests/service/*` | One test module per source module |

**Modified:**

| File | Change |
|---|---|
| `src/rga/adapters/neo4j_source.py` | `events()` reconstructs a journal from timestamps |
| `src/rga/features/build.py` | Core accepting an event stream; `build_candidates` wraps it |
| `src/rga/nn/scorer.py`, `src/rga/nn/supervised.py` | `save` and `load` |
| `src/rga/cli/main.py` | `train` and `serve` commands |
| `pyproject.toml` | `service` extra gains nothing; `demo` extra for the filler script |

---

### Task 1: A journal reconstructed from edge timestamps

`Neo4jSource.events()` refuses to work today, and the service cannot build candidates
without a journal. Section 12.2 of the system spec says a level-1 source reconstructs
one from timestamps, and those timestamps are exactly what PR #70 added.

Two honest limits are written into the code rather than glossed over. A snapshot shows
no revocations, so the journal holds grants only. And edges predating the timestamp
patch carry no time at all; they are real context but cannot be placed in the
sequence, so they are emitted first, just before the earliest known change.

**Files:**
- Modify: `src/rga/adapters/neo4j_source.py`
- Test: `tests/adapters/test_neo4j_journal.py`

**Interfaces:**
- Consumes: `GraphRecordReader`, `RelationMapping`, `Capabilities` — all existing.
- Produces: `Neo4jSource.events(since: int = 0, until: int | None = None) -> Iterator[GraphEvent]`,
  ordered by time, grants only, raising `ValueError` when the source has no timestamps.

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/test_neo4j_journal.py
"""A level-1 snapshot reconstructs its own change log."""

from pathlib import Path

import pytest

from rga.adapters.mapping import RelationMapping
from rga.adapters.neo4j_source import Neo4jSource
from rga.domain.events import EventOp
from rga.domain.relations import PermissionLevel, RelationType

MAPPING = RelationMapping.load(Path("configs/mapping/opens3.yaml"))


class _Reader:
    """Rows in whatever order the store returned them."""

    def __init__(self, relationships):
        self._relationships = relationships

    def nodes(self):
        seen = {}
        for row in self._relationships:
            seen.setdefault(row["subject"], row.get("created_at"))
            seen.setdefault(row["object"], row.get("created_at"))
        return iter([{"id": key, "created_at": value} for key, value in seen.items()])

    def relationships(self):
        return iter(self._relationships)


def _rows():
    return [
        {
            "subject": "group:devops",
            "relation": "HAS_PERMISSION",
            "object": "bucket:logs",
            "level": "admin",
            "created_at": 3000,
            "actor": "user:root",
        },
        {
            "subject": "user:alice",
            "relation": "MEMBER_OF",
            "object": "group:devops",
            "level": None,
            "created_at": 1000,
            "actor": None,
        },
    ]


def test_events_come_out_in_time_order() -> None:
    source = Neo4jSource(_Reader(_rows()), MAPPING)

    events = list(source.events())

    assert [event.ts for event in events] == [1000, 3000]
    assert events[0].relation is RelationType.MEMBER_OF
    assert events[1].level is PermissionLevel.ADMIN
    assert events[1].actor == "user:root"


def test_every_event_is_a_grant() -> None:
    """A snapshot cannot show a revocation: a revoked edge is simply absent."""
    source = Neo4jSource(_Reader(_rows()), MAPPING)

    assert {event.op for event in source.events()} == {EventOp.GRANT}


def test_the_window_is_honoured() -> None:
    source = Neo4jSource(_Reader(_rows()), MAPPING)

    assert [event.ts for event in source.events(since=2000)] == [3000]
    assert [event.ts for event in source.events(until=2000)] == [1000]


def test_edges_from_before_the_patch_come_first() -> None:
    rows = [*_rows(), {
        "subject": "user:bob",
        "relation": "MEMBER_OF",
        "object": "group:devops",
        "level": None,
        "created_at": None,
        "actor": None,
    }]

    events = list(Neo4jSource(_Reader(rows), MAPPING).events())

    assert events[0].subject == "user:bob"
    assert events[0].ts < 1000


def test_a_source_without_timestamps_cannot_produce_a_journal() -> None:
    rows = [dict(row, created_at=None) for row in _rows()]

    with pytest.raises(ValueError, match="no timestamps"):
        list(Neo4jSource(_Reader(rows), MAPPING).events())
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/adapters/test_neo4j_journal.py -v`

Expected: FAIL with `NotImplementedError: a Neo4j snapshot has no change log`.

- [ ] **Step 3: Implement the reconstruction**

Replace `Neo4jSource.events` in `src/rga/adapters/neo4j_source.py`:

```python
    def events(self, since: int = 0, until: int | None = None) -> Iterator[GraphEvent]:
        """The change log implied by the edge timestamps.

        Two limits are inherent and not worked around. A snapshot shows no
        revocations — a revoked edge is simply absent — so every event is a grant,
        and a right granted and taken back between two snapshots never existed as
        far as this source is concerned. And an edge written before the timestamp
        patch carries no time; it is real context, so it is emitted just ahead of
        the earliest known change rather than dropped.
        """
        if not self.capabilities().timestamps:
            raise ValueError("this deployment records no timestamps; it has no journal")

        rows = list(self._reader.relationships())
        stamped = [row for row in rows if row.get("created_at") is not None]
        if not stamped:
            raise ValueError("this deployment records no timestamps; it has no journal")

        earliest = min(int(row["created_at"]) for row in stamped)  # type: ignore[arg-type]
        undated = [row for row in rows if row.get("created_at") is None]

        ordered = [(earliest - 1, row) for row in undated]
        ordered += [(int(row["created_at"]), row) for row in stamped]  # type: ignore[arg-type]
        ordered.sort(key=lambda pair: pair[0])

        for ts, row in ordered:
            if ts < since or (until is not None and ts > until):
                continue
            level_property = row.get("level")
            relation, level = self._mapping.translate(
                str(row["relation"]),
                None if level_property is None else str(level_property),
            )
            actor = row.get("actor")
            yield GraphEvent(
                ts=ts,
                op=EventOp.GRANT,
                subject=str(row["subject"]),
                relation=relation,
                object=str(row["object"]),
                level=PermissionLevel(int(level)),
                actor=None if actor is None else str(actor),
            )
```

Add the imports the body needs at the top of the file:

```python
from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel
```

and update the module docstring's last paragraph to say the source now reconstructs a
journal when the deployment records timestamps.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/adapters -q`

Expected: green, including the existing `test_neo4j_source.py`.

- [ ] **Step 5: Commit**

```bash
git add src/rga/adapters/neo4j_source.py tests/adapters/test_neo4j_journal.py
git commit -m "feat: reconstruct a change log from neo4j edge timestamps"
git push origin main
```

---

### Task 2: Candidates from any journal

`build_candidates` takes a synthetic `Dataset`: a journal plus ground truth plus a
generator config. A live source has none of those. The scoring core is the same
either way, so it moves into a function that takes what it actually needs.

**Files:**
- Modify: `src/rga/features/build.py`
- Test: `tests/features/test_candidates_from_events.py`

**Interfaces:**
- Produces: `candidates_from_events(events, *, start, end, graph, static_seed=0, labels=None, automation_actors=AUTOMATION_ACTORS) -> CandidateSet`
  where `labels` maps `(edge_key, ts)` to a pattern name and defaults to none.
- `build_candidates(dataset, span, ...)` keeps its signature and calls the new function.

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_candidates_from_events.py
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/features/test_candidates_from_events.py -v`

Expected: FAIL with `ImportError: cannot import name 'candidates_from_events'`.

- [ ] **Step 3: Extract the core**

In `src/rga/features/build.py`, add the new function and reduce `build_candidates` to
a wrapper. The body below is the current loop with the `Dataset` references replaced
by arguments:

```python
def candidates_from_events(
    events: Iterable[GraphEvent],
    *,
    start: int,
    end: int,
    graph: AccessGraph,
    static_seed: int = 0,
    labels: Mapping[tuple[tuple[str, int, str], int], str] | None = None,
    automation_actors: frozenset[str] = AUTOMATION_ACTORS,
) -> CandidateSet:
    """Extract features for every grant inside [start, end).

    `graph` is the state the network propagates over — normally the snapshot at the
    temporal split. `labels` is ground truth where it exists; a live source has none
    and passes nothing, which makes every candidate unlabelled rather than normal.
    """
    static = compute_static_attributes(graph, np.random.default_rng(static_seed))
    label_of = dict(labels or {})
    context = FeatureContext()

    keys: list[tuple[str, int, str]] = []
    stamps: list[int] = []
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    flags: list[bool] = []
    patterns: list[str] = []

    for event in events:
        is_candidate = (
            event.op is EventOp.GRANT
            and start <= event.ts < end
            and event.actor not in automation_actors
        )
        if is_candidate:
            values, mask = _candidate_row(context, static, event)
            keys.append(event.edge_key())
            stamps.append(event.ts)
            rows.append(values)
            masks.append(mask)
            pattern = label_of.get((event.edge_key(), event.ts), "")
            flags.append(bool(pattern))
            patterns.append(pattern)
        context.apply(event)

    matrix = FeatureMatrix(
        values=(
            np.vstack(rows).astype(np.float32)
            if rows
            else np.zeros((0, len(CANDIDATE_BLOCK)), dtype=np.float32)
        ),
        mask=(np.vstack(masks) if masks else np.zeros((0, len(CANDIDATE_BLOCK)), dtype=bool)),
        block=CANDIDATE_BLOCK,
    )
    return CandidateSet(
        keys=tuple(keys),
        ts=np.array(stamps, dtype=np.int64),
        matrix=matrix,
        labels=np.array(flags, dtype=bool),
        patterns=tuple(patterns),
        graph=graph,
    )


def build_candidates(
    dataset: Dataset,
    span: Span,
    *,
    static_seed: int = 0,
    warmup_days: int = WARMUP_DAYS,
    automation_actors: frozenset[str] = AUTOMATION_ACTORS,
) -> CandidateSet:
    """Extract features for every grant inside the requested span of a dataset."""
    graph = replay(dataset.events, until=dataset.split_ts)
    start, end = _span_bounds(dataset, span, warmup_days)
    return candidates_from_events(
        dataset.events,
        start=start,
        end=end,
        graph=graph,
        static_seed=static_seed,
        labels={
            (label.edge_key(), label.ts): label.pattern
            for label in (*dataset.labels, *dataset.train_labels)
        },
        automation_actors=automation_actors,
    )
```

Add to the imports at the top of the file:

```python
from collections.abc import Iterable, Mapping

from rga.domain.graph import AccessGraph
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/features tests/eval -q && uv run pytest -m "not integration and not gpu" -q`

Expected: green throughout. The first test of the new module is the regression guard:
the refactor must not move a single feature value.

- [ ] **Step 5: Commit**

```bash
git add src/rga/features/build.py tests/features/test_candidates_from_events.py
git commit -m "feat: build candidates from any journal, not only a dataset"
git push origin main
```

---

### Task 3: The scorer encodes the graph it is asked to score

A latent defect that only bites once the scorer leaves the experiment runner. `fit`
stores the node representations computed from the training graph, and `score` reuses
them — indexing them with node indices that come from *the graph of the candidates
being scored*. In the experiment both spans carry the same graph object, so nothing
shows. A service scoring a live graph would index one graph's representations with
another graph's indices: at best an `IndexError`, at worst silent nonsense.

**Files:**
- Modify: `src/rga/nn/scorer.py`, `src/rga/nn/supervised.py`
- Test: `tests/nn/test_scorer_across_graphs.py`

**Interfaces:**
- Unchanged from outside: `fit(train)`, `score(candidates)`.
- Internally both scorers stop caching node state and encode `candidates.graph` when
  scoring. The rank-transform references stay fitted on the training span, which is
  what makes them a reference.

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_scorer_across_graphs.py
"""A fitted scorer must work on a graph it has never seen."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer
from rga.nn.supervised import SupervisedGnnScorer

BASE = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def two_worlds():
    """Two datasets from different seeds: different graphs, different node counts."""
    here = build_dataset(replace(BASE, seed=11))
    elsewhere = build_dataset(replace(BASE, seed=12))
    return (
        build_candidates(here, Span.TRAIN),
        build_candidates(elsewhere, Span.EVAL),
    )


def test_the_self_supervised_scorer_scores_a_foreign_graph(two_worlds) -> None:
    home, foreign = two_worlds
    scorer = GnnScorer(seed=0, config=FAST)
    scorer.fit(home)

    scores = scorer.score(foreign)

    assert scores.shape == (foreign.n_candidates,)
    assert np.isfinite(scores).all()


def test_the_supervised_scorer_scores_a_foreign_graph(two_worlds) -> None:
    home, foreign = two_worlds
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(home)

    scores = scorer.score(foreign)

    assert scores.shape == (foreign.n_candidates,)
    assert np.isfinite(scores).all()


def test_the_graphs_really_differ(two_worlds) -> None:
    """Guards the two tests above: on identical graphs they would prove nothing."""
    home, foreign = two_worlds

    assert home.graph.num_nodes != foreign.graph.num_nodes
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_scorer_across_graphs.py -v`

Expected: FAIL with `IndexError: index ... is out of bounds` from `_node_rank`, or from
the padded representation lookup. If the foreign graph happens to be smaller the
failure may instead be silently wrong numbers — check
`test_the_graphs_really_differ` passes first, and if the two graphs come out the same
size, change seed 12 to 13.

- [ ] **Step 3: Encode at scoring time**

In `src/rga/nn/scorer.py`, drop the cached `_state` and `_deviation` and add a helper
that encodes whichever graph it is handed. Replace the body of `fit` after training,
and `score`, with:

```python
    def fit(self, train: CandidateSet) -> None:
        """Train on the span and record what the rank transform needs."""
        if train.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        arrays, mean, std = candidate_arrays(train)
        arrays = without_features(arrays)
        self._mean, self._std = mean, std
        self._model = train_model(
            train.graph, arrays, self._config, seed=self._seed, device=self._device
        )

        state, deviation = self._encode(train.graph)
        self._likelihood_rank = RankTransform.fit(1.0 - self._likelihood(state, arrays))
        self._deviation_rank = RankTransform.fit(deviation)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """One score per candidate in [0, 1], higher meaning more unusual."""
        if self._model is None or self._likelihood_rank is None:
            raise RuntimeError("the gnn scorer must be fit before scoring")
        if candidates.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        state, deviation = self._encode(candidates.graph)
        arrays, _, _ = candidate_arrays(candidates, mean=self._mean, std=self._std)
        unlikeliness = self._likelihood_rank.apply(
            1.0 - self._likelihood(state, without_features(arrays))
        )
        return combine(
            unlikeliness,
            self._node_rank(deviation, arrays.src),
            self._node_rank(deviation, arrays.dst),
        )

    def _encode(self, graph: AccessGraph) -> tuple[torch.Tensor, np.ndarray]:
        """Representations and profile deviations for every node of `graph`.

        Recomputed per call rather than cached: the graph a service scores is not
        the graph the model was fitted on, and node indices are meaningful only
        within one graph.
        """
        assert self._model is not None
        tensors = graph_tensors(graph, device=self._device)
        inputs = node_input_features(tensors)
        with torch.no_grad():
            state = self._model.encoder(inputs, tensors)
            deviation = self._model.reconstruction.deviation(state, inputs).cpu().numpy()
        return state, deviation
```

and change the two helpers to take what they need instead of reading fields:

```python
    def _likelihood(self, state: torch.Tensor, arrays: CandidateArrays) -> np.ndarray:
        """Probability the model assigns to each change being ordinary."""
        assert self._model is not None
        padded = torch.cat([state, torch.zeros(1, state.shape[1], device=self._device)])
        unknown = padded.shape[0] - 1

        def endpoints(index: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.where(index < 0, unknown, index), device=self._device)

        with torch.no_grad():
            logits = self._model.likelihood(
                padded[endpoints(arrays.src)],
                padded[endpoints(arrays.dst)],
                torch.as_tensor(arrays.relation, device=self._device),
                torch.as_tensor(arrays.level, device=self._device),
                torch.as_tensor(arrays.features, device=self._device),
            )
        return torch.sigmoid(logits).cpu().numpy()

    def _node_rank(self, deviation: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Ranked profile deviation of an endpoint, neutral where it is unknown."""
        assert self._deviation_rank is not None
        known = index >= 0
        ranks = np.full(index.shape, NEUTRAL, dtype=np.float64)
        if known.any():
            ranks[known] = self._deviation_rank.apply(deviation[index[known]])
        return ranks
```

Delete the `self._state` and `self._deviation` attributes from `__init__`, and add
`from rga.domain.graph import AccessGraph` and `from rga.nn.node_inputs import node_input_features`
to the imports if they are not already there.

Apply the same change to `src/rga/nn/supervised.py`: drop `self._state`, and in
`score` encode `candidates.graph` before looking up endpoints.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/nn -q`

Expected: green, including `test_scorer.py`, whose numbers must not move — both spans
of one dataset carry the same graph, so encoding it per call gives what caching gave.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/scorer.py src/rga/nn/supervised.py tests/nn/test_scorer_across_graphs.py
git commit -m "fix: encode the graph being scored rather than the fitted one"
git push origin main
```

---

### Task 4: The model artefact

Training takes minutes and needs labels the live graph does not have. Neither can
happen at the defense, so a fitted scorer is written to disk and the service loads it.

The manifest exists to make a mismatch loud. An artefact fitted before the feature
block changed will produce confident nonsense on new code unless something refuses to
load it.

**Files:**
- Create: `src/rga/artifacts.py`
- Modify: `src/rga/nn/scorer.py`, `src/rga/nn/supervised.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Produces:
  - `block_fingerprint() -> str` — sixteen hex characters over the candidate feature names
  - `save_scorer(path: Path, scorer, *, dataset: str) -> None`
  - `load_scorer(path: Path) -> Scorer`
  - `read_manifest(path: Path) -> dict[str, object]`
  - `GnnScorer.state_for_artifact()` / `restore_from_artifact(state)` and the same on
    `SupervisedGnnScorer`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_artifacts.py
"""A fitted scorer survives a trip through the filesystem."""

from pathlib import Path

import numpy as np
import pytest

from rga.artifacts import block_fingerprint, load_scorer, read_manifest, save_scorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def spans():
    dataset = build_dataset(CONFIG)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


def test_a_reloaded_scorer_ranks_identically(spans, tmp_path) -> None:
    train, evaluation = spans
    scorer = SupervisedGnnScorer(seed=3, config=FAST)
    scorer.fit(train)
    before = scorer.score(evaluation)

    save_scorer(tmp_path / "artifact", scorer, dataset="small-history")
    after = load_scorer(tmp_path / "artifact").score(evaluation)

    assert np.allclose(before, after, atol=1e-6)


def test_the_manifest_records_what_was_trained(spans, tmp_path) -> None:
    train, _ = spans
    scorer = SupervisedGnnScorer(seed=3, config=FAST)
    scorer.fit(train)

    save_scorer(tmp_path / "artifact", scorer, dataset="small-history")
    manifest = read_manifest(tmp_path / "artifact")

    assert manifest["scorer"] == "gnn_supervised"
    assert manifest["dataset"] == "small-history"
    assert manifest["features"] == block_fingerprint()
    assert manifest["seed"] == 3


def test_an_artefact_from_another_feature_space_is_refused(spans, tmp_path) -> None:
    train, _ = spans
    scorer = SupervisedGnnScorer(seed=3, config=FAST)
    scorer.fit(train)
    save_scorer(tmp_path / "artifact", scorer, dataset="small-history")

    manifest = tmp_path / "artifact" / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(block_fingerprint(), "0" * 16),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature"):
        load_scorer(tmp_path / "artifact")
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/test_artifacts.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.artifacts'`.

- [ ] **Step 3: Give the scorers a serialisable state**

Add to `GnnScorer` in `src/rga/nn/scorer.py`:

```python
    def state_for_artifact(self) -> dict[str, object]:
        """Everything a reloaded copy needs, as plain arrays and numbers."""
        if self._model is None or self._likelihood_rank is None:
            raise RuntimeError("the gnn scorer must be fit before saving")
        assert self._deviation_rank is not None and self._mean is not None
        assert self._std is not None
        return {
            "weights": self._model.state_dict(),
            "edge_dim": 0,
            "mean": self._mean,
            "std": self._std,
            "likelihood_reference": self._likelihood_rank.reference,
            "deviation_reference": self._deviation_rank.reference,
        }

    def restore_from_artifact(self, state: dict[str, object]) -> None:
        """Rebuild a fitted scorer from `state_for_artifact`."""
        model = GnnModel(edge_dim=int(state["edge_dim"]), config=self._config)
        model.load_state_dict(state["weights"])  # type: ignore[arg-type]
        self._model = model.to(self._device).eval()
        self._mean = np.asarray(state["mean"])
        self._std = np.asarray(state["std"])
        self._likelihood_rank = RankTransform(np.asarray(state["likelihood_reference"]))
        self._deviation_rank = RankTransform(np.asarray(state["deviation_reference"]))
```

Add the mirror image to `SupervisedGnnScorer`, whose state is the weights, the edge
dimension (the real feature width, not zero) and the standardisation — it has no rank
transforms.

- [ ] **Step 4: Write the artefact module**

```python
# src/rga/artifacts.py
"""Writing a fitted scorer to disk and reading it back.

Three files, so that each part is inspectable on its own: `model.pt` holds the
network weights, `state.npz` the fitted numpy state — standardisation and the rank
transform references — and `manifest.json` says what was trained and against which
feature space.

The manifest is not bookkeeping. An artefact fitted before the feature block changed
will score confidently and wrongly on new code, and nothing else would notice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from rga.baselines.base import Scorer
from rga.features.build import CANDIDATE_BLOCK
from rga.nn.config import ModelConfig


def block_fingerprint() -> str:
    """Identity of the feature space, as sixteen hex characters."""
    joined = "\n".join(CANDIDATE_BLOCK.names).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def save_scorer(path: Path, scorer, *, dataset: str) -> None:
    """Write a fitted scorer to `path`, creating the directory."""
    state = scorer.state_for_artifact()
    path.mkdir(parents=True, exist_ok=True)

    torch.save(state["weights"], path / "model.pt")
    arrays = {
        key: np.asarray(value)
        for key, value in state.items()
        if key not in {"weights", "edge_dim"}
    }
    np.savez(path / "state.npz", **arrays)

    manifest = {
        "scorer": scorer.name,
        "dataset": dataset,
        "seed": scorer.seed,
        "edge_dim": int(state["edge_dim"]),
        "features": block_fingerprint(),
        "config": asdict(scorer.config),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_manifest(path: Path) -> dict[str, object]:
    """The manifest of an artefact directory."""
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def load_scorer(path: Path) -> Scorer:
    """Rebuild the scorer an artefact holds, refusing a stale feature space."""
    manifest = read_manifest(path)
    if manifest["features"] != block_fingerprint():
        raise ValueError(
            f"artefact at {path} was fitted against a different feature space "
            f"({manifest['features']} against {block_fingerprint()}); retrain it"
        )

    config = ModelConfig(**manifest["config"])  # type: ignore[arg-type]
    name = str(manifest["scorer"])
    seed = int(manifest["seed"])  # type: ignore[arg-type]

    if name == "gnn":
        from rga.nn.scorer import GnnScorer

        scorer = GnnScorer(seed=seed, config=config)
    elif name == "gnn_supervised":
        from rga.nn.supervised import SupervisedGnnScorer

        scorer = SupervisedGnnScorer(seed=seed, config=config)
    else:
        raise ValueError(f"no artefact format for scorer {name!r}")

    with np.load(path / "state.npz") as stored:
        state: dict[str, object] = {key: stored[key] for key in stored.files}
    state["weights"] = torch.load(path / "model.pt", weights_only=True)
    state["edge_dim"] = int(manifest["edge_dim"])  # type: ignore[arg-type]
    scorer.restore_from_artifact(state)
    return scorer
```

The manifest reads `scorer.seed` and `scorer.config`, so add those two read-only
properties to both scorers next to the state methods:

```python
    @property
    def seed(self) -> int:
        return self._seed

    @property
    def config(self) -> ModelConfig:
        return self._config
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_artifacts.py tests/nn -q`

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/rga/artifacts.py src/rga/nn/scorer.py src/rga/nn/supervised.py tests/test_artifacts.py
git commit -m "feat: save and load a fitted scorer as an artefact"
git push origin main
```

---

### Task 5: The train command

**Files:**
- Create: `configs/train/gnn-supervised.yaml`
- Modify: `src/rga/cli/main.py`
- Test: `tests/cli/test_train_command.py`

**Interfaces:**
- Produces: `rga train --config <recipe> --out <directory>`, writing an artefact
  directory Task 4 can load. The recipe names a dataset config, a scorer and a seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_train_command.py
"""Training a shipped artefact from the command line."""

from pathlib import Path

from rga.artifacts import read_manifest
from rga.cli.main import main


def _recipe(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "dataset: configs/generator/small-history.yaml\n"
        "scorer: gnn_supervised\n"
        "seed: 5\n"
        "model:\n"
        "  hidden_dim: 16\n"
        "  num_layers: 2\n"
        "  epochs: 3\n"
        "  patience: 3\n",
        encoding="utf-8",
        newline="\n",
    )
    return recipe


def test_training_writes_a_loadable_artefact(tmp_path) -> None:
    out = tmp_path / "artifact"

    code = main(["train", "--config", str(_recipe(tmp_path)), "--out", str(out)])

    assert code == 0
    assert (out / "model.pt").exists()
    assert (out / "state.npz").exists()
    manifest = read_manifest(out)
    assert manifest["scorer"] == "gnn_supervised"
    assert manifest["seed"] == 5


def test_an_unknown_scorer_is_refused(tmp_path) -> None:
    recipe = tmp_path / "bad.yaml"
    recipe.write_text(
        "dataset: configs/generator/small-history.yaml\nscorer: nonsense\nseed: 1\n",
        encoding="utf-8",
        newline="\n",
    )

    assert main(["train", "--config", str(recipe), "--out", str(tmp_path / "x")]) == 2
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/cli/test_train_command.py -v`

Expected: FAIL — argparse rejects the unknown command `train` and `main` returns 2 for
the first test too.

- [ ] **Step 3: Add the command**

In `src/rga/cli/main.py`, register the parser next to the others:

```python
    train = commands.add_parser("train", help="fit a scorer and save it as an artefact")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
```

and handle it in `main`, keeping the heavy imports local so the CLI stays importable
without torch:

```python
    if arguments.command == "train":
        import yaml

        from rga.artifacts import save_scorer
        from rga.eval.experiment import build_scorer
        from rga.features.build import Span, build_candidates
        from rga.nn.config import ModelConfig

        recipe = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
        name = str(recipe["scorer"])
        if name not in {"gnn", "gnn_supervised"}:
            print(f"no artefact format for scorer {name!r}")
            return 2

        dataset_config = load_dataset_config(Path(recipe["dataset"]))
        dataset = build_dataset(dataset_config)
        train_set = build_candidates(dataset, Span.TRAIN)

        scorer = build_scorer(name, seed=int(recipe.get("seed", 0)))
        overrides = recipe.get("model") or {}
        if overrides:
            scorer = type(scorer)(
                seed=int(recipe.get("seed", 0)), config=ModelConfig(**overrides)
            )
        scorer.fit(train_set)
        save_scorer(arguments.out, scorer, dataset=dataset_config.name)
        print(f"wrote {arguments.out}")
        return 0
```

- [ ] **Step 4: Write the shipped recipe**

```yaml
# configs/train/gnn-supervised.yaml
# The artefact the service ships with. The supervised variant is used because it
# scored best in module 3 (PR-AUC 0.739 against 0.719 for the best classical
# baseline), and because its head reads the candidate features, which is what makes
# feature attribution possible at all.
dataset: configs/generator/small-history.yaml
scorer: gnn_supervised
seed: 1
model:
  hidden_dim: 64
  num_layers: 3
  epochs: 200
  patience: 20
```

- [ ] **Step 5: Run the tests and train the shipped artefact**

Run: `uv run pytest tests/cli -q`

Expected: green. Then produce the artefact the service will use:

```bash
uv run rga train --config configs/train/gnn-supervised.yaml --out artifacts/gnn-supervised
```

Expected: a few minutes, then `wrote artifacts/gnn-supervised`. Add `artifacts/` to
`.gitignore` — it is reproducible from the recipe, like `experiments/runs/`.

- [ ] **Step 6: Commit**

```bash
git add src/rga/cli/main.py configs/train/gnn-supervised.yaml tests/cli/test_train_command.py .gitignore
git commit -m "feat: add the train command and the shipped model recipe"
git push origin main
```

---

### Task 6: One candidate on its own

Both explanation mechanisms score a single change repeatedly. A candidate set that can
hand out one of its rows as a candidate set of its own keeps that from being written
three times.

**Files:**
- Modify: `src/rga/features/spec.py`
- Test: `tests/features/test_candidate_row.py`

**Interfaces:**
- Produces: `CandidateSet.row(position: int) -> CandidateSet` — one candidate, the same
  feature block, the same graph object.

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_candidate_row.py
"""A candidate set can hand out one of its rows."""

from pathlib import Path

import numpy as np

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def test_a_row_is_a_candidate_set_of_one() -> None:
    candidates = build_candidates(build_dataset(CONFIG), Span.EVAL)

    single = candidates.row(4)

    assert single.n_candidates == 1
    assert single.keys == (candidates.keys[4],)
    assert single.ts[0] == candidates.ts[4]
    assert np.array_equal(single.matrix.values[0], candidates.matrix.values[4])
    assert single.matrix.block is candidates.matrix.block
    assert single.graph is candidates.graph
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/features/test_candidate_row.py -v`

Expected: FAIL with `AttributeError: 'CandidateSet' object has no attribute 'row'`.

- [ ] **Step 3: Add the method**

In `src/rga/features/spec.py`, inside `CandidateSet`:

```python
    def row(self, position: int) -> CandidateSet:
        """One candidate as a set of its own, sharing the block and the graph."""
        window = slice(position, position + 1)
        return CandidateSet(
            keys=(self.keys[position],),
            ts=self.ts[window],
            matrix=FeatureMatrix(
                values=self.matrix.values[window],
                mask=self.matrix.mask[window],
                block=self.matrix.block,
            ),
            labels=self.labels[window],
            patterns=(self.patterns[position],),
            graph=self.graph,
        )
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/features -q`

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/rga/features/spec.py tests/features/test_candidate_row.py
git commit -m "feat: let a candidate set hand out a single row"
git push origin main
```

---

### Task 7: Which features drove the score

Gradient of the score with respect to each input, times the input itself. The torch
part lives on the scorer, where the model is; the explain package turns bare numbers
into named, grouped, observability-aware contributions and never touches a tensor.

A masked feature is reported as unobserved rather than as a zero contribution: zero
means "this had no influence", unobserved means "the source could not tell us", and
an analyst must be able to tell those apart.

**Files:**
- Create: `src/rga/explain/__init__.py`, `src/rga/explain/features.py`
- Modify: `src/rga/nn/supervised.py`, `src/rga/nn/scorer.py`
- Test: `tests/explain/test_features.py`

**Interfaces:**
- Produces:
  - `SupervisedGnnScorer.feature_gradients(candidates, position) -> np.ndarray` over the
    columns of `dense_matrix`; `GnnScorer.feature_gradients(...) -> None` because it
    reads no context row
  - `FeatureContribution` frozen dataclass: `name`, `group`, `value`, `contribution`, `observed`
  - `feature_contributions(scorer, candidates, position, *, top=10) -> tuple[FeatureContribution, ...]`,
    empty when the scorer reads no features

- [ ] **Step 1: Write the failing test**

```python
# tests/explain/test_features.py
"""Which of the candidate's own features moved the score."""

from pathlib import Path

import pytest

from rga.explain.features import feature_contributions
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def fitted():
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(train)
    return scorer, train, evaluation


def test_contributions_are_named_grouped_and_ordered(fitted) -> None:
    scorer, _, evaluation = fitted

    found = feature_contributions(scorer, evaluation, 0, top=5)

    assert len(found) == 5
    assert all(item.name in evaluation.matrix.block.names for item in found)
    assert all(isinstance(item.group, FeatureGroup) for item in found)
    magnitudes = [abs(item.contribution) for item in found]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_an_unobserved_feature_is_marked_not_zeroed(fitted) -> None:
    scorer, _, evaluation = fitted

    found = feature_contributions(scorer, evaluation, 0, top=len(evaluation.matrix.block))
    unobserved = [item for item in found if not item.observed]

    assert all(item.contribution == 0.0 for item in unobserved)


def test_a_structure_only_scorer_reports_no_feature_contributions(fitted) -> None:
    _, train, evaluation = fitted
    structural = GnnScorer(seed=0, config=FAST)
    structural.fit(train)

    assert feature_contributions(structural, evaluation, 0) == ()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/explain/test_features.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.explain'`.

- [ ] **Step 3: Expose the gradients on the scorers**

Add to `SupervisedGnnScorer` in `src/rga/nn/supervised.py`:

```python
    def feature_gradients(self, candidates: CandidateSet, position: int) -> np.ndarray:
        """d(score)/d(feature) times the feature, over the dense matrix columns."""
        if self._model is None:
            raise RuntimeError("the supervised gnn scorer must be fit before explaining")

        single = candidates.row(position)
        arrays, _, _ = candidate_arrays(single, mean=self._mean, std=self._std)
        state, _ = self._encode(single.graph)

        padded = torch.cat([state, torch.zeros(1, state.shape[1], device=self._device)])
        unknown = padded.shape[0] - 1
        endpoints = lambda index: torch.as_tensor(  # noqa: E731
            np.where(index < 0, unknown, index), device=self._device
        )

        features = torch.as_tensor(arrays.features, device=self._device).requires_grad_(True)
        logit = self._model.likelihood(
            padded[endpoints(arrays.src)],
            padded[endpoints(arrays.dst)],
            torch.as_tensor(arrays.relation, device=self._device),
            torch.as_tensor(arrays.level, device=self._device),
            features,
        )
        torch.sigmoid(logit).sum().backward()
        assert features.grad is not None
        return (features.grad * features.detach()).squeeze(0).cpu().numpy()
```

and to `GnnScorer` in `src/rga/nn/scorer.py`:

```python
    def feature_gradients(self, candidates: CandidateSet, position: int) -> np.ndarray | None:
        """None: this scorer reads structure only, so no feature moved the score."""
        return None
```

`SupervisedGnnScorer` needs `_encode` too — the same helper Task 3 added to
`GnnScorer`, returning representations and deviations for a graph.

- [ ] **Step 4: Write the explain module**

```python
# src/rga/explain/__init__.py
"""Why a change scored the way it did: features, structure and words."""
```

```python
# src/rga/explain/features.py
"""Attribution over the candidate's own features.

Gradient times input: how much the score would move per unit of a feature, times how
much of that feature this candidate actually has. It answers "which of the things we
know about this change pushed it up the list".

Unobserved features are reported as unobserved, never as a zero contribution. Zero
means the value had no influence; unobserved means the authorization engine could not
tell us, and an analyst has to be able to tell those apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from rga.features.spec import CandidateSet, FeatureGroup


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's share of the score."""

    name: str
    group: FeatureGroup
    value: float
    contribution: float
    observed: bool


def feature_contributions(
    scorer, candidates: CandidateSet, position: int, *, top: int = 10
) -> tuple[FeatureContribution, ...]:
    """The features that moved this candidate's score, strongest first.

    Empty when the scorer reads no candidate features — which is itself worth showing
    an analyst, since it says the ranking came from graph structure alone.
    """
    gradients = getattr(scorer, "feature_gradients", None)
    if gradients is None:
        return ()
    attribution = gradients(candidates, position)
    if attribution is None:
        return ()

    block = candidates.matrix.block
    values = candidates.matrix.values[position]
    observed = candidates.matrix.mask[position]

    found = [
        FeatureContribution(
            name=name,
            group=block.groups[column],
            value=float(values[column]),
            contribution=float(attribution[column]) if bool(observed[column]) else 0.0,
            observed=bool(observed[column]),
        )
        # dense_matrix appends one coverage column per group after the block, and
        # those are bookkeeping rather than features an analyst would recognise.
        for column, name in enumerate(block.names)
    ]
    found.sort(key=lambda item: abs(item.contribution), reverse=True)
    return tuple(found[:top])
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/explain -q`

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/rga/explain tests/explain/test_features.py src/rga/nn/supervised.py src/rga/nn/scorer.py
git commit -m "feat: attribute a score to the candidate features"
git push origin main
```

---

### Task 8: Which edges of the neighbourhood mattered

Masking. Each edge within two hops of the change is removed in turn, the graph is
re-encoded, and the drop in the score is that edge's importance. This is what turns
"the model finds this unusual" into "the model finds this unusual *because of these
three relationships*", and it is what the subgraph picture highlights.

**Files:**
- Create: `src/rga/explain/structure.py`
- Test: `tests/explain/test_structure.py`

**Interfaces:**
- Produces:
  - `EdgeImportance` frozen dataclass: `subject`, `relation`, `object`, `importance`
  - `neighbourhood(graph, subject, object, *, hops=2, cap=60) -> np.ndarray` of edge positions
  - `edge_importance(scorer, candidates, position, *, hops=2, cap=60) -> tuple[EdgeImportance, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/explain/test_structure.py
"""Which relationships around the change drove its score."""

from pathlib import Path

import pytest

from rga.explain.structure import edge_importance, neighbourhood
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def fitted():
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(train)
    return scorer, evaluation


def test_the_neighbourhood_is_bounded(fitted) -> None:
    _, evaluation = fitted
    subject, _, target = evaluation.keys[0]

    found = neighbourhood(evaluation.graph, subject, target, hops=2, cap=25)

    assert 0 < len(found) <= 25
    assert len(set(found.tolist())) == len(found)


def test_every_edge_gets_an_importance(fitted) -> None:
    scorer, evaluation = fitted

    found = edge_importance(scorer, evaluation, 0, cap=15)

    assert 0 < len(found) <= 15
    assert all(isinstance(item.importance, float) for item in found)
    magnitudes = [abs(item.importance) for item in found]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_graph_is_left_untouched(fitted) -> None:
    scorer, evaluation = fitted
    before = evaluation.graph.num_edges

    edge_importance(scorer, evaluation, 0, cap=10)

    assert evaluation.graph.num_edges == before
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/explain/test_structure.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.explain.structure'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/explain/structure.py
"""Attribution over the graph around a change.

Each edge within two hops is removed in turn and the change re-scored; the drop is
that edge's importance. Two hops because that is the radius the encoder actually
sees through its layers, and because it covers the chain the authorization engine
itself walks: a user, the group it belongs to, the bucket that group can reach.

The work is bounded by `cap`: each masked edge costs a full re-encode, and an analyst
waiting on a card will not thank us for a hundred of them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rga.domain.graph import AccessGraph
from rga.domain.relations import RelationType
from rga.features.spec import CandidateSet


@dataclass(frozen=True)
class EdgeImportance:
    """One relationship's share of the score."""

    subject: str
    relation: str
    object: str
    importance: float


def neighbourhood(
    graph: AccessGraph, subject: str, target: str, *, hops: int = 2, cap: int = 60
) -> np.ndarray:
    """Positions of the edges within `hops` of either endpoint, at most `cap`."""
    reached = {graph.node_index[name] for name in (subject, target) if name in graph.node_index}
    if not reached:
        return np.zeros(0, dtype=np.int64)

    frontier = set(reached)
    for _ in range(hops):
        touching = np.flatnonzero(
            np.isin(graph.edge_src, list(frontier)) | np.isin(graph.edge_dst, list(frontier))
        )
        frontier = set(graph.edge_src[touching].tolist()) | set(graph.edge_dst[touching].tolist())
        reached |= frontier

    positions = np.flatnonzero(
        np.isin(graph.edge_src, list(reached)) & np.isin(graph.edge_dst, list(reached))
    )
    return positions[:cap]


def _without(graph: AccessGraph, position: int) -> AccessGraph:
    """The same graph minus one edge. Node indices are preserved."""
    keep = np.ones(graph.num_edges, dtype=bool)
    keep[position] = False
    return replace(
        graph,
        edge_src=graph.edge_src[keep],
        edge_dst=graph.edge_dst[keep],
        edge_rel=graph.edge_rel[keep],
        edge_level=graph.edge_level[keep],
        edge_created=graph.edge_created[keep],
        edge_actor=graph.edge_actor[keep],
    )


def edge_importance(
    scorer, candidates: CandidateSet, position: int, *, hops: int = 2, cap: int = 60
) -> tuple[EdgeImportance, ...]:
    """How much each nearby relationship holds this candidate's score up."""
    graph = candidates.graph
    if graph is None:
        raise ValueError("the candidate set carries no graph; structure cannot be explained")

    single = candidates.row(position)
    baseline = float(scorer.score(single)[0])
    subject, _, target = candidates.keys[position]

    found: list[EdgeImportance] = []
    for edge in neighbourhood(graph, subject, target, hops=hops, cap=cap):
        masked = replace(single, graph=_without(graph, int(edge)))
        without = float(scorer.score(masked)[0])
        found.append(
            EdgeImportance(
                subject=graph.node_ids[int(graph.edge_src[edge])],
                relation=RelationType(int(graph.edge_rel[edge])).name,
                object=graph.node_ids[int(graph.edge_dst[edge])],
                importance=baseline - without,
            )
        )

    found.sort(key=lambda item: abs(item.importance), reverse=True)
    return tuple(found)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/explain -q`

Expected: green. If `test_every_edge_gets_an_importance` runs longer than about
fifteen seconds, lower the cap in the test rather than in the module — the module's
default serves the service, where one card at a time is the workload.

- [ ] **Step 5: Commit**

```bash
git add src/rga/explain/structure.py tests/explain/test_structure.py
git commit -m "feat: attribute a score to the surrounding relationships"
git push origin main
```

---

### Task 9: Saying it in words

Numbers rank; sentences explain. Each template states an observation and stops there.
The system supports a decision, so it says "the right was granted by the subject to
itself", never "privilege escalation detected" — and a test enforces exactly that,
because this is the line the whole framing of the work rests on.

**Files:**
- Create: `src/rga/explain/text.py`
- Test: `tests/explain/test_text.py`

**Interfaces:**
- Produces: `describe(candidates: CandidateSet, position: int) -> tuple[str, ...]`, Russian
  sentences, most telling first, and `FORBIDDEN_WORDS` naming what must never appear.

- [ ] **Step 1: Write the failing test**

```python
# tests/explain/test_text.py
"""Sentences an analyst reads."""

from pathlib import Path

import numpy as np
import pytest

from rga.explain.text import FORBIDDEN_WORDS, describe
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))


@pytest.fixture(scope="module")
def candidates():
    return build_candidates(build_dataset(CONFIG), Span.EVAL)


def test_every_candidate_gets_at_least_one_sentence(candidates) -> None:
    for position in range(min(candidates.n_candidates, 50)):
        assert describe(candidates, position)


def test_a_self_grant_is_named_as_an_observation(candidates) -> None:
    column = candidates.matrix.column("actor_is_subject")
    observed = candidates.matrix.observed("actor_is_subject")
    position = int(np.flatnonzero((column > 0.5) & observed)[0])

    said = " ".join(describe(candidates, position))

    assert "сам" in said


def test_no_verdict_is_ever_pronounced(candidates) -> None:
    """The system supports a decision. It does not announce a compromise."""
    for position in range(min(candidates.n_candidates, 200)):
        said = " ".join(describe(candidates, position)).lower()
        for word in FORBIDDEN_WORDS:
            assert word not in said, f"candidate {position} says {word!r}"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/explain/test_text.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.explain.text'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/explain/text.py
"""Template sentences describing one change.

Every sentence states something observed and stops. The system hands an analyst a
ranked list of hypotheses with grounds; it does not announce a compromise, and the
wording has to hold that line even when the signal looks obvious.

Sentences are only produced for features the source actually supplied. A right whose
initiator is unknown gets no sentence about its initiator rather than a sentence
saying nobody granted it.
"""

from __future__ import annotations

from rga.domain.relations import PermissionLevel
from rga.features.spec import CandidateSet
from rga.util.timeutil import hour_of_day

#: Words that pronounce a verdict. None of them belongs in what an analyst is shown.
FORBIDDEN_WORDS = (
    "компрометац",
    "атака",
    "злоумышленник",
    "взлом",
    "инцидент подтверж",
    "нарушитель",
)


def _said(candidates: CandidateSet, position: int, name: str) -> float | None:
    """A feature's value, or None when the source could not supply it."""
    if not bool(candidates.matrix.observed(name)[position]):
        return None
    return float(candidates.matrix.column(name)[position])


def describe(candidates: CandidateSet, position: int) -> tuple[str, ...]:
    """What is worth saying about this change, most telling first."""
    subject, _, target = candidates.keys[position]
    lines: list[str] = []

    level = _said(candidates, position, "level_ordinal")
    if level is not None and level > 0:
        name = PermissionLevel(int(round(level))).name.lower()
        lines.append(f"Выдан уровень «{name}» на ресурс {target}.")
    else:
        lines.append(f"Связь {subject} с {target} создана.")

    if _said(candidates, position, "actor_is_subject") == 1.0:
        lines.append("Право выдал сам субъект, а не кто-то другой.")

    jump = _said(candidates, position, "level_jump")
    if jump is not None and jump >= 2:
        lines.append(
            f"Уровень поднят сразу на {int(jump)} ступени относительно того, "
            "что у субъекта было в этом поддереве ресурсов."
        )

    if _said(candidates, position, "bypasses_bucket") == 1.0:
        lines.append("Право выдано на объект, хотя прав на содержащий его бакет нет.")

    common = _said(candidates, position, "common_neighbours")
    if common is not None and common == 0:
        lines.append("У субъекта и ресурса нет ни одного общего соседа в графе.")

    if _said(candidates, position, "path_unreachable") == 1.0:
        lines.append("До этого ресурса от субъекта не было пути по графу прав.")

    if _said(candidates, position, "is_off_hours") == 1.0:
        hour = hour_of_day(int(candidates.ts[position]))
        lines.append(f"Изменение сделано в {hour}:00 — вне рабочих часов.")

    if _said(candidates, position, "is_weekend") == 1.0:
        lines.append("Изменение сделано в выходной день.")

    return tuple(lines)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/explain -q`

Expected: green. If `test_a_self_grant_is_named_as_an_observation` cannot find a
candidate, the evaluation window of `small-history.yaml` has no observed self-grant —
check `actor_is_subject` is in the provenance group and the dataset supplies it.

- [ ] **Step 5: Commit**

```bash
git add src/rga/explain/text.py tests/explain/test_text.py
git commit -m "feat: describe a change in sentences an analyst reads"
git push origin main
```

---

### Task 10: The incident payload

What the API returns for one change, assembled once so the routes stay thin.

**Files:**
- Create: `src/rga/explain/incident.py`
- Test: `tests/explain/test_incident.py`

**Interfaces:**
- Produces:
  - `incident_id(key: tuple[str, int, str], ts: int) -> str` — sixteen hex characters
  - `Incident` frozen dataclass with `as_dict()`
  - `build_incident(scorer, candidates, position, score, rank, *, explain=True) -> Incident`

- [ ] **Step 1: Write the failing test**

```python
# tests/explain/test_incident.py
"""The card the service hands the interface."""

from pathlib import Path

import numpy as np
import pytest

from rga.explain.incident import build_incident, incident_id
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small-history.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def fitted():
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(train)
    return scorer, evaluation


def test_the_identifier_is_stable_and_specific() -> None:
    key = ("user:alex", 2, "bucket:logs")

    assert incident_id(key, 1000) == incident_id(key, 1000)
    assert incident_id(key, 1000) != incident_id(key, 1001)
    assert len(incident_id(key, 1000)) == 16


def test_the_card_carries_everything_the_page_needs(fitted) -> None:
    scorer, evaluation = fitted

    card = build_incident(scorer, evaluation, 0, score=0.9, rank=1, cap=10)
    payload = card.as_dict()

    assert payload["id"] == incident_id(evaluation.keys[0], int(evaluation.ts[0]))
    assert payload["score"] == 0.9
    assert payload["rank"] == 1
    assert payload["summary"]
    assert "features" in payload and "edges" in payload and "subgraph" in payload
    assert {"nodes", "edges"} <= set(payload["subgraph"])


def test_the_cheap_form_skips_the_expensive_parts(fitted) -> None:
    scorer, evaluation = fitted

    listed = build_incident(scorer, evaluation, 0, score=0.9, rank=1, explain=False)

    assert listed.as_dict()["summary"]
    assert listed.as_dict()["edges"] == []


def test_every_subgraph_node_has_a_hop_distance(fitted) -> None:
    scorer, evaluation = fitted

    subgraph = build_incident(scorer, evaluation, 0, score=0.5, rank=2, cap=10).as_dict()[
        "subgraph"
    ]

    assert all(isinstance(node["hops"], int) for node in subgraph["nodes"])
    assert min(node["hops"] for node in subgraph["nodes"]) == 0
    assert np.isfinite([edge["importance"] for edge in subgraph["edges"]]).all()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/explain/test_incident.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.explain.incident'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/explain/incident.py
"""One change, assembled into what an analyst is shown.

The identifier is derived from the change itself rather than from its position in the
queue: a refresh reorders the queue, and a link an analyst saved must still point at
the same change afterwards.

`explain=False` builds the cheap form used for list rows. The expensive part is edge
masking, which re-encodes the graph once per neighbouring edge — fine for the one card
on screen, not for fifty rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from rga.domain.entities import entity_type
from rga.domain.relations import PermissionLevel, RelationType
from rga.explain.features import FeatureContribution, feature_contributions
from rga.explain.structure import EdgeImportance, edge_importance, neighbourhood
from rga.explain.text import describe
from rga.features.spec import CandidateSet


def incident_id(key: tuple[str, int, str], ts: int) -> str:
    """A stable name for one change, independent of its rank."""
    subject, relation, target = key
    raw = f"{subject}|{relation}|{target}|{ts}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class Incident:
    """A ranked change with its grounds."""

    id: str
    score: float
    rank: int
    ts: int
    subject: str
    relation: str
    object: str
    level: str
    summary: tuple[str, ...]
    features: tuple[FeatureContribution, ...] = ()
    edges: tuple[EdgeImportance, ...] = ()
    subgraph: dict[str, list] = field(default_factory=lambda: {"nodes": [], "edges": []})

    def as_dict(self) -> dict[str, object]:
        """The shape the API returns."""
        return {
            "id": self.id,
            "score": self.score,
            "rank": self.rank,
            "ts": self.ts,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "level": self.level,
            "summary": list(self.summary),
            "features": [
                {
                    "name": item.name,
                    "group": str(item.group),
                    "value": item.value,
                    "contribution": item.contribution,
                    "observed": item.observed,
                }
                for item in self.features
            ],
            "edges": [
                {
                    "subject": item.subject,
                    "relation": item.relation,
                    "object": item.object,
                    "importance": item.importance,
                }
                for item in self.edges
            ],
            "subgraph": self.subgraph,
        }


def _subgraph(candidates: CandidateSet, position: int, edges, *, cap: int) -> dict[str, list]:
    """Nodes and edges around the change, each node with its distance in hops."""
    graph = candidates.graph
    assert graph is not None
    subject, _, target = candidates.keys[position]

    importance = {
        (item.subject, item.relation, item.object): item.importance for item in edges
    }
    positions = neighbourhood(graph, subject, target, cap=cap)

    hops = {name: 0 for name in (subject, target) if name in graph.node_index}
    drawn_edges = []
    for edge in positions:
        source = graph.node_ids[int(graph.edge_src[edge])]
        sink = graph.node_ids[int(graph.edge_dst[edge])]
        relation = RelationType(int(graph.edge_rel[edge])).name
        drawn_edges.append(
            {
                "subject": source,
                "relation": relation,
                "object": sink,
                "level": PermissionLevel(int(graph.edge_level[edge])).name.lower(),
                "importance": float(importance.get((source, relation, sink), 0.0)),
            }
        )

    # Breadth first from the endpoints, over the edges we are going to draw.
    for _ in range(2):
        for edge in drawn_edges:
            for near, far in ((edge["subject"], edge["object"]), (edge["object"], edge["subject"])):
                if near in hops:
                    hops.setdefault(far, hops[near] + 1)

    nodes = [
        {"id": name, "type": entity_type(name).name.lower(), "hops": distance}
        for name, distance in sorted(hops.items(), key=lambda pair: (pair[1], pair[0]))
    ]
    drawn_edges = [
        edge for edge in drawn_edges if edge["subject"] in hops and edge["object"] in hops
    ]
    return {"nodes": nodes, "edges": drawn_edges}


def build_incident(
    scorer,
    candidates: CandidateSet,
    position: int,
    *,
    score: float,
    rank: int,
    explain: bool = True,
    cap: int = 60,
) -> Incident:
    """Assemble one change, with or without the expensive attributions."""
    subject, relation, target = candidates.keys[position]
    ordinal = int(round(float(candidates.matrix.column("level_ordinal")[position])))
    edges = (
        edge_importance(scorer, candidates, position, cap=cap) if explain else ()
    )
    return Incident(
        id=incident_id(candidates.keys[position], int(candidates.ts[position])),
        score=score,
        rank=rank,
        ts=int(candidates.ts[position]),
        subject=subject,
        relation=RelationType(relation).name,
        object=target,
        level=PermissionLevel(ordinal).name.lower(),
        summary=describe(candidates, position),
        features=feature_contributions(scorer, candidates, position) if explain else (),
        edges=edges,
        subgraph=_subgraph(candidates, position, edges, cap=cap) if explain else {
            "nodes": [],
            "edges": [],
        },
    )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/explain -q`

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/rga/explain/incident.py tests/explain/test_incident.py
git commit -m "feat: assemble a ranked change with its grounds"
git push origin main
```

---

### Task 11: Configuration and the analysis it drives

The one place that knows a source can be a generated dataset or a live Neo4j. Above
it, nothing does — which is the requirement of section 12 and the reason the same
service demonstrates both without a code change.

**Files:**
- Create: `src/rga/service/__init__.py`, `src/rga/service/config.py`, `src/rga/service/analysis.py`
- Create: `configs/service/synthetic.yaml`, `configs/service/opens3.yaml`
- Test: `tests/service/test_analysis.py`

**Interfaces:**
- Produces:
  - `ServiceConfig` frozen dataclass with `load_service_config(path) -> ServiceConfig`
  - `Analysis` frozen dataclass: `candidates`, `scores`, `order`, `source`, `level`,
    `refreshed_at`, plus `find(incident_id) -> int | None`
  - `analyse(config, scorer) -> Analysis`

- [ ] **Step 1: Write the failing test**

```python
# tests/service/test_analysis.py
"""Source to candidates to a ranked queue."""

from pathlib import Path

import numpy as np
import pytest

from rga.explain.incident import incident_id
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer
from rga.service.analysis import analyse
from rga.service.config import load_service_config

FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def scorer():
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    fitted = SupervisedGnnScorer(seed=0, config=FAST)
    fitted.fit(build_candidates(dataset, Span.TRAIN))
    return fitted


def test_the_synthetic_source_produces_a_ranked_queue(scorer) -> None:
    config = load_service_config(Path("configs/service/synthetic.yaml"))

    analysis = analyse(config, scorer)

    assert analysis.candidates.n_candidates > 0
    assert analysis.scores.shape == (analysis.candidates.n_candidates,)
    ranked = analysis.scores[analysis.order]
    assert np.all(np.diff(ranked) <= 0)
    assert analysis.level >= 1


def test_an_incident_can_be_found_by_its_identifier(scorer) -> None:
    analysis = analyse(load_service_config(Path("configs/service/synthetic.yaml")), scorer)
    position = int(analysis.order[0])
    wanted = incident_id(analysis.candidates.keys[position], int(analysis.candidates.ts[position]))

    assert analysis.find(wanted) == position
    assert analysis.find("0" * 16) is None


def test_the_window_width_limits_what_is_scored(scorer) -> None:
    wide = load_service_config(Path("configs/service/synthetic.yaml"))
    narrow = type(wide)(**{**wide.__dict__, "window_days": 1})

    assert analyse(narrow, scorer).candidates.n_candidates < analyse(wide, scorer).candidates.n_candidates
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/service/test_analysis.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.service'`.

- [ ] **Step 3: Write the configuration**

```python
# src/rga/service/__init__.py
"""The scoring service and the page it serves."""
```

```python
# src/rga/service/config.py
"""What the service is pointed at.

The source is the only place in the running system that knows whether the graph came
from the generator or from a live authorization engine. Switching between them is a
configuration change and nothing else, which is what section 12 of the design
document demands and what the closing minute of the demonstration shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the service needs to start."""

    source: str
    model: Path
    window_days: int
    queue: int
    #: Synthetic source: the generator recipe to replay.
    dataset: Path | None = None
    #: Live source: where Neo4j is and how relations are named there.
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"
    mapping: Path = Path("configs/mapping/opens3.yaml")

    def __post_init__(self) -> None:
        if self.source not in {"synthetic", "neo4j"}:
            raise ValueError(f"unknown source kind: {self.source!r}")
        if self.source == "synthetic" and self.dataset is None:
            raise ValueError("a synthetic source needs a dataset recipe")


def load_service_config(path: Path) -> ServiceConfig:
    """Read a service configuration from YAML."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = document["source"]
    return ServiceConfig(
        source=str(source["kind"]),
        model=Path(document["model"]),
        window_days=int(document.get("window_days", 7)),
        queue=int(document.get("queue", 50)),
        dataset=Path(source["dataset"]) if "dataset" in source else None,
        uri=str(source.get("uri", "bolt://localhost:7687")),
        user=str(source.get("user", "neo4j")),
        password=str(source.get("password", "password123")),
        mapping=Path(source.get("mapping", "configs/mapping/opens3.yaml")),
    )
```

```yaml
# configs/service/synthetic.yaml
# The demonstration's main scenario: a generated organization, its evaluation window,
# and the queue an analyst would work. Depends on nothing but this repository.
source:
  kind: synthetic
  dataset: configs/generator/small-history.yaml
model: artifacts/gnn-supervised
window_days: 7
queue: 50
```

```yaml
# configs/service/opens3.yaml
# The same service against a live opens3-rebac, running from the feat/graph-timestamps
# branch. Nothing above the adapter changes; only this file does.
source:
  kind: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password: password123
  mapping: configs/mapping/opens3.yaml
model: artifacts/gnn-supervised
window_days: 7
queue: 50
```

- [ ] **Step 4: Write the analysis**

```python
# src/rga/service/analysis.py
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
            if incident_id(self.candidates.keys[position], int(self.candidates.ts[position])) == wanted:
                return position
        return None


def _synthetic(config: ServiceConfig) -> tuple[tuple[GraphEvent, ...], AccessGraph, int, int, Capabilities]:
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


def _live(config: ServiceConfig) -> tuple[tuple[GraphEvent, ...], AccessGraph, int, int, Capabilities]:
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
        scorer.score(candidates)
        if candidates.n_candidates
        else np.zeros(0, dtype=np.float64)
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
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/service -q`

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/rga/service configs/service tests/service/test_analysis.py
git commit -m "feat: turn any source into a ranked queue"
git push origin main
```

---

### Task 12: The service

Five routes over the analysis, and the page served beside them.

**Files:**
- Create: `src/rga/service/app.py`
- Modify: `pyproject.toml` (dev group gains `httpx`, which `TestClient` needs)
- Test: `tests/service/test_app.py`

**Interfaces:**
- Produces: `create_app(config: ServiceConfig) -> FastAPI` with
  `GET /api/status`, `POST /api/refresh`, `GET /api/incidents`,
  `GET /api/incidents/{incident}`, `GET /api/nodes`.

- [ ] **Step 1: Write the failing test**

```python
# tests/service/test_app.py
"""The routes the page talks to."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer
from rga.service.app import create_app
from rga.service.config import load_service_config

FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def client():
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(build_candidates(dataset, Span.TRAIN))
    config = load_service_config(Path("configs/service/synthetic.yaml"))
    return TestClient(create_app(config, scorer=scorer))


def test_status_describes_what_is_running(client) -> None:
    body = client.get("/api/status").json()

    assert body["source"] == "synthetic"
    assert body["capability_level"] >= 1
    assert body["scorer"] == "gnn_supervised"
    assert body["candidates"] > 0


def test_the_queue_comes_back_ranked(client) -> None:
    body = client.get("/api/incidents", params={"limit": 10}).json()

    assert len(body["incidents"]) == 10
    scores = [item["score"] for item in body["incidents"]]
    assert scores == sorted(scores, reverse=True)
    assert body["incidents"][0]["rank"] == 1


def test_the_queue_can_be_filtered_by_subject(client) -> None:
    everything = client.get("/api/incidents", params={"limit": 50}).json()["incidents"]
    subject = everything[0]["subject"]

    filtered = client.get(
        "/api/incidents", params={"limit": 50, "subject": subject}
    ).json()["incidents"]

    assert filtered
    assert {item["subject"] for item in filtered} == {subject}


def test_a_card_carries_its_grounds(client) -> None:
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]

    card = client.get(f"/api/incidents/{listed['id']}").json()

    assert card["id"] == listed["id"]
    assert card["summary"]
    assert card["subgraph"]["nodes"]


def test_an_unknown_card_is_a_clean_404(client) -> None:
    assert client.get("/api/incidents/" + "0" * 16).status_code == 404


def test_a_node_neighbourhood_comes_back(client) -> None:
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]

    body = client.get("/api/nodes", params={"id": listed["subject"]}).json()

    assert body["id"] == listed["subject"]
    assert body["subgraph"]["nodes"]


def test_an_unknown_node_is_a_clean_404(client) -> None:
    assert client.get("/api/nodes", params={"id": "user:nobody"}).status_code == 404


def test_refresh_reruns_the_analysis(client) -> None:
    before = client.get("/api/status").json()["refreshed_at"]

    assert client.post("/api/refresh").status_code == 200
    assert client.get("/api/status").json()["refreshed_at"] >= before
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/service/test_app.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.service.app'`. If it
fails earlier on `fastapi.testclient`, add `httpx>=0.27` to the `dev` dependency group
in `pyproject.toml` and run `uv sync --extra cpu` first.

- [ ] **Step 3: Write the application**

```python
# src/rga/service/app.py
"""The HTTP surface: a queue, a card, a neighbourhood.

The service reads and never writes. It holds one analysis in memory and replaces it
on refresh, which is what the closing minute of the demonstration leans on: grant
yourself a right through the engine, press refresh, watch it arrive.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rga.explain.incident import build_incident, incident_id
from rga.explain.structure import neighbourhood
from rga.service.analysis import Analysis, analyse
from rga.service.config import ServiceConfig

WEB = Path("web")


def create_app(config: ServiceConfig, *, scorer=None) -> FastAPI:
    """Build the application. `scorer` is injected by tests; otherwise it is loaded."""
    if scorer is None:
        from rga.artifacts import load_scorer

        scorer = load_scorer(config.model)

    app = FastAPI(title="ReBAC access graph review", docs_url="/api/docs")
    state: dict[str, Analysis] = {"analysis": analyse(config, scorer)}

    def current() -> Analysis:
        return state["analysis"]

    @app.get("/api/status")
    def status() -> dict[str, object]:
        analysis = current()
        return {
            "source": analysis.source,
            "capability_level": analysis.level,
            "scorer": scorer.name,
            "window": {"start": analysis.window[0], "end": analysis.window[1]},
            "candidates": analysis.candidates.n_candidates,
            "refreshed_at": analysis.refreshed_at,
        }

    @app.post("/api/refresh")
    def refresh() -> dict[str, object]:
        state["analysis"] = analyse(config, scorer)
        return status()

    @app.get("/api/incidents")
    def incidents(
        limit: int = Query(default=config.queue, ge=1, le=500),
        since: int | None = None,
        relation: str | None = None,
        subject: str | None = None,
    ) -> dict[str, object]:
        analysis = current()
        rows = []
        for rank, position in enumerate(analysis.order.tolist(), start=1):
            key = analysis.candidates.keys[position]
            ts = int(analysis.candidates.ts[position])
            if since is not None and ts < since:
                continue
            if subject is not None and key[0] != subject:
                continue
            card = build_incident(
                scorer,
                analysis.candidates,
                position,
                score=float(analysis.scores[position]),
                rank=rank,
                explain=False,
            )
            if relation is not None and card.relation != relation:
                continue
            rows.append(card.as_dict())
            if len(rows) >= limit:
                break
        return {"incidents": rows, "total": analysis.candidates.n_candidates}

    @app.get("/api/incidents/{incident}")
    def incident(incident: str) -> dict[str, object]:
        analysis = current()
        position = analysis.find(incident)
        if position is None:
            raise HTTPException(status_code=404, detail="no such change in the current window")
        rank = int(analysis.order.tolist().index(position)) + 1
        return build_incident(
            scorer,
            analysis.candidates,
            position,
            score=float(analysis.scores[position]),
            rank=rank,
        ).as_dict()

    @app.get("/api/nodes")
    def node(id: str = Query(...)) -> dict[str, object]:  # noqa: A002
        analysis = current()
        graph = analysis.candidates.graph
        if graph is None or id not in graph.node_index:
            raise HTTPException(status_code=404, detail="no such node in the current graph")

        from rga.domain.relations import PermissionLevel, RelationType

        positions = neighbourhood(graph, id, id, hops=1, cap=60)
        nodes = {id: 0}
        edges = []
        for edge in positions:
            source = graph.node_ids[int(graph.edge_src[edge])]
            sink = graph.node_ids[int(graph.edge_dst[edge])]
            nodes.setdefault(source, 1)
            nodes.setdefault(sink, 1)
            edges.append(
                {
                    "subject": source,
                    "relation": RelationType(int(graph.edge_rel[edge])).name,
                    "object": sink,
                    "level": PermissionLevel(int(graph.edge_level[edge])).name.lower(),
                    "importance": 0.0,
                }
            )

        from rga.domain.entities import entity_type

        return {
            "id": id,
            "subgraph": {
                "nodes": [
                    {"id": name, "type": entity_type(name).name.lower(), "hops": hops}
                    for name, hops in nodes.items()
                ],
                "edges": edges,
            },
        }

    if WEB.is_dir():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/")
        def page() -> FileResponse:
            return FileResponse(WEB / "index.html")

    return app
```

Note that `incident_id` is imported for the identifier contract even though the routes
reach it through `Analysis.find`; keep the import only if the module uses it, and drop
it otherwise so ruff stays quiet.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/service -q && uv run ruff check src/rga/service`

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/rga/service/app.py tests/service/test_app.py pyproject.toml
git commit -m "feat: serve the queue, the card and the neighbourhood"
git push origin main
```

---

### Task 13: The page

One HTML file, one stylesheet, one script. No build pipeline, no framework, no
third-party code — the page has to open on a machine that may have no network.

**Files:**
- Create: `web/index.html`, `web/style.css`, `web/app.js`
- Test: `tests/service/test_page.py`

**Interfaces:**
- Consumes: the five routes of Task 12.
- Produces: nothing importable; the page is the deliverable.

- [ ] **Step 1: Write the failing test**

```python
# tests/service/test_page.py
"""The page is served, self-contained, and says nothing it should not."""

from pathlib import Path

import pytest

from rga.explain.text import FORBIDDEN_WORDS

WEB = Path("web")


def test_the_page_exists() -> None:
    assert (WEB / "index.html").is_file()
    assert (WEB / "style.css").is_file()
    assert (WEB / "app.js").is_file()


def test_nothing_is_loaded_from_the_network() -> None:
    """A defence room may have no internet, and a CDN is a point of failure."""
    markup = (WEB / "index.html").read_text(encoding="utf-8")

    assert "http://" not in markup
    assert "https://" not in markup


@pytest.mark.parametrize("name", ["index.html", "app.js"])
def test_the_page_pronounces_no_verdict(name) -> None:
    text = (WEB / name).read_text(encoding="utf-8").lower()

    for word in FORBIDDEN_WORDS:
        assert word not in text
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/service/test_page.py -v`

Expected: FAIL — `web/index.html` does not exist.

- [ ] **Step 3: Write the page**

`web/index.html` — a header with the source, the capability level and a refresh
button; a filter row; a two-column body with the queue on the left and the card on
the right:

```html
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Обзор изменений прав доступа</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>Обзор изменений прав доступа</h1>
    <p id="status">Загрузка…</p>
    <button id="refresh" type="button">Обновить</button>
  </header>
  <section id="filters">
    <label>Субъект <input id="filter-subject" type="text" placeholder="user:…"></label>
    <label>Отношение
      <select id="filter-relation">
        <option value="">любое</option>
        <option>HAS_PERMISSION</option>
        <option>MEMBER_OF</option>
        <option>PARENT_OF</option>
        <option>OWNER_OF</option>
      </select>
    </label>
    <button id="apply" type="button">Применить</button>
  </section>
  <main>
    <ol id="queue"></ol>
    <article id="card"><p class="hint">Выберите изменение из списка слева.</p></article>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

`web/style.css` — a two-column grid, a readable monospace for identifiers, a score
bar, and a muted style for unobserved features. Keep it under a hundred lines and use
system fonts only.

`web/app.js` — fetch the queue, render rows, fetch a card on click, render the
summary, the feature table and the edge table. Feature rows that are not observed are
rendered greyed with «источник не сообщает» instead of a number. Leave the subgraph
container empty; Task 14 fills it.

- [ ] **Step 4: Run the tests and look at the page**

Run: `uv run pytest tests/service -q`

Then start it by hand and open `http://127.0.0.1:8000`:

```bash
uv run python -c "
import uvicorn
from pathlib import Path
from rga.artifacts import load_scorer
from rga.service.app import create_app
from rga.service.config import load_service_config
config = load_service_config(Path('configs/service/synthetic.yaml'))
uvicorn.run(create_app(config, scorer=load_scorer(config.model)), port=8000)
"
```

Expected: the queue renders, a click fills the card, the numbers match what
`/api/incidents` returns. Fix what looks wrong before moving on — this is the first
moment the work becomes visible.

- [ ] **Step 5: Commit**

```bash
git add web tests/service/test_page.py
git commit -m "feat: add the analyst page"
git push origin main
```

---

### Task 14: Drawing the neighbourhood

The picture that makes the argument on the defence day. Laid out in layers by distance
from the change rather than by a force simulation: at twenty nodes a deterministic
layout reads better, never jitters, and adds no dependency to a repository that must
work offline.

**Files:**
- Modify: `web/app.js`, `web/style.css`
- Test: `tests/service/test_subgraph_payload.py`

**Interfaces:**
- Consumes: `subgraph` from the card route — nodes with `hops`, edges with `importance`.
- Produces: an inline SVG inside the card.

- [ ] **Step 1: Write the failing test**

The drawing is JavaScript and is checked by eye; what a test can hold is the contract
the drawing depends on.

```python
# tests/service/test_subgraph_payload.py
"""The card gives the drawing everything it needs and nothing it cannot use."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer
from rga.service.app import create_app
from rga.service.config import load_service_config

FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def card():
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    scorer = SupervisedGnnScorer(seed=0, config=FAST)
    scorer.fit(build_candidates(dataset, Span.TRAIN))
    config = load_service_config(Path("configs/service/synthetic.yaml"))
    client = TestClient(create_app(config, scorer=scorer))
    listed = client.get("/api/incidents", params={"limit": 1}).json()["incidents"][0]
    return client.get(f"/api/incidents/{listed['id']}").json()


def test_every_edge_endpoint_is_a_declared_node(card) -> None:
    names = {node["id"] for node in card["subgraph"]["nodes"]}

    for edge in card["subgraph"]["edges"]:
        assert edge["subject"] in names
        assert edge["object"] in names


def test_the_change_itself_sits_at_the_centre(card) -> None:
    centre = [node["id"] for node in card["subgraph"]["nodes"] if node["hops"] == 0]

    assert card["subject"] in centre or card["object"] in centre


def test_layers_are_small_enough_to_draw(card) -> None:
    assert len(card["subgraph"]["nodes"]) <= 80
    assert max(node["hops"] for node in card["subgraph"]["nodes"]) <= 3
```

- [ ] **Step 2: Run the test and confirm it fails or passes**

Run: `uv run pytest tests/service/test_subgraph_payload.py -v`

Expected: PASS if Task 10 got the payload right. If `test_the_change_itself_sits_at_the_centre`
fails, `_subgraph` is not seeding the hop map from both endpoints — fix that before
drawing anything.

- [ ] **Step 3: Draw it**

In `web/app.js`, add a function that takes the `subgraph` object and returns an SVG
element:

- group nodes by `hops`; each layer is a column, `x = 120 + hops * 220`;
- inside a layer spread nodes evenly down the height, `y = (index + 1) * H / (count + 1)`;
- draw edges first as lines, stroke width `1 + 4 * importance / max_importance` so the
  relationships that held the score up are visibly thicker, and grey when importance is
  zero;
- draw nodes as rounded rectangles coloured by `type`, labelled with the identifier
  trimmed to its last segment, with the full identifier in a `<title>`;
- draw the scored change itself as a thicker, differently coloured line between its two
  endpoints, and label it with the level;
- size the `viewBox` to the content so it scales without measuring anything.

Keep it under about 120 lines, with no external library and no animation.

- [ ] **Step 4: Look at it**

Start the service as in Task 13 and open two or three cards. Expected: the change
stands out, thick edges correspond to the top rows of the edge table, nothing overlaps
badly at twenty nodes. Adjust spacing until it reads.

- [ ] **Step 5: Commit**

```bash
git add web/app.js web/style.css tests/service/test_subgraph_payload.py
git commit -m "feat: draw the neighbourhood of a change"
git push origin main
```

---

### Task 15: Filling a live graph for the demonstration

A freshly started `opens3-rebac` has an empty graph, and on an empty graph every
change looks unusual — there is no settled structure to be unusual against. The demo
needs a populated one, so the generator's journal is replayed into Neo4j.

It writes through the driver rather than through gRPC on purpose. Going through the
engine's API would mean pulling `grpcio` and another project's generated stubs into
this repository, and section 12 of the design document requires the opposite: above
the adapter layer nothing knows `opens3-rebac` exists. The property names this script
writes are the ones already declared in `configs/mapping/opens3.yaml`.

The moment that matters on the day still goes through the engine: the author grants
themselves `admin` with the engine's own gRPC call, and the detector sees it by
re-reading the snapshot.

**Files:**
- Create: `scripts/fill_live_graph.py`
- Test: `tests/service/test_fill_live_graph.py` (marked `integration`)

**Interfaces:**
- Produces: `write_journal(driver, events) -> int` and a `main()` entry point taking
  `--config`, `--uri`, `--user`, `--password`, `--days`.

- [ ] **Step 1: Write the failing test**

```python
# tests/service/test_fill_live_graph.py
"""Filling a live Neo4j with a generated organization."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "password123")),
    )


def test_a_filled_graph_reads_back_through_the_adapter() -> None:
    from scripts.fill_live_graph import write_journal

    from rga.adapters.mapping import RelationMapping
    from rga.adapters.neo4j_source import BoltReader, Neo4jSource
    from rga.generator.config import load_dataset_config
    from rga.generator.dataset import build_dataset

    dataset = build_dataset(load_dataset_config(Path("configs/generator/small-history.yaml")))
    driver = _driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    written = write_journal(driver, dataset.events[:500])
    driver.close()

    reader = BoltReader(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j",
        os.environ.get("NEO4J_PASSWORD", "password123"),
    )
    try:
        source = Neo4jSource(reader, RelationMapping.load(Path("configs/mapping/opens3.yaml")))
        capabilities = source.capabilities()
        events = list(source.events())
        graph = source.snapshot()
    finally:
        reader.close()

    assert written > 0
    assert capabilities.level == 2
    assert len(events) == graph.num_edges
    assert [event.ts for event in events] == sorted(event.ts for event in events)
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
cd /home/wexel/Data/Code/Projects/opens3-rebac && docker compose up -d --wait neo4j
cd /home/wexel/Data/Code/Projects/rebac-graph-anomaly
uv run pytest tests/service/test_fill_live_graph.py -m integration -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.fill_live_graph'`.

- [ ] **Step 3: Write the script**

```python
# scripts/fill_live_graph.py
"""Replay a generated journal into a live Neo4j, for the demonstration.

Writes the property names the authorization engine itself writes — `created_at`,
`updated_at`, `actor`, `level` — so the detector's adapter reads the result exactly as
it reads a graph the engine produced. This is the mapping of
configs/mapping/opens3.yaml turned around to write instead of read.

It does not go through the engine's gRPC API, and that is deliberate: doing so would
drag grpcio and another project's generated stubs into this repository, which section
12 of the design document forbids. The demonstration's key moment — granting yourself
admin — is performed with the engine's own client, not with this script.

    uv run python scripts/fill_live_graph.py --config configs/generator/small-history.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.util.timeutil import DAY_MS

_LABELS = {"user": "User", "group": "Group", "bucket": "Bucket", "object": "Object"}

_WRITE = """
MERGE (subject:`%s` {id: $subject})
  ON CREATE SET subject.created_at = $ts
MERGE (object:`%s` {id: $object})
  ON CREATE SET object.created_at = $ts
MERGE (subject)-[rel:`%s`]->(object)
  ON CREATE SET rel.created_at = $ts
SET rel.updated_at = $ts, rel.actor = $actor%s
RETURN rel.updated_at AS written
"""


def _label(entity_id: str) -> str:
    return _LABELS[entity_id.split(":", 1)[0]]


def write_journal(driver, events: Iterable[GraphEvent]) -> int:
    """Apply a journal to the graph in time order. Returns the number of writes."""
    written = 0
    with driver.session() as session:
        for event in events:
            if event.op is not EventOp.GRANT:
                session.run(
                    "MATCH (s {id: $subject})-[r:`%s`]->(o {id: $object}) DELETE r"
                    % event.relation.name,
                    subject=event.subject,
                    object=event.object,
                )
                continue
            carries_level = event.level is not PermissionLevel.NONE
            query = _WRITE % (
                _label(event.subject),
                _label(event.object),
                event.relation.name,
                ", rel.level = $level" if carries_level else "",
            )
            parameters = {
                "subject": event.subject,
                "object": event.object,
                "ts": event.ts,
                "actor": event.actor,
            }
            if carries_level:
                parameters["level"] = event.level.name.lower()
            session.run(query, **parameters)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/generator/small-history.yaml"))
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password123")
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="keep only the last N days of the journal; 0 keeps all of it",
    )
    arguments = parser.parse_args()

    from neo4j import GraphDatabase

    dataset = build_dataset(load_dataset_config(arguments.config))
    events = dataset.events
    if arguments.days:
        cutoff = dataset.window_end - arguments.days * DAY_MS
        events = tuple(event for event in events if event.ts >= cutoff)

    driver = GraphDatabase.driver(arguments.uri, auth=(arguments.user, arguments.password))
    try:
        written = write_journal(driver, events)
    finally:
        driver.close()
    print(f"wrote {written} relationships into {arguments.uri}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/service/test_fill_live_graph.py -m integration -v`

Expected: green. Then stop the container so nothing is left running:
`cd /home/wexel/Data/Code/Projects/opens3-rebac && docker compose stop neo4j`.

- [ ] **Step 5: Commit**

```bash
git add scripts/fill_live_graph.py tests/service/test_fill_live_graph.py
git commit -m "feat: replay a generated journal into a live neo4j"
git push origin main
```

---

### Task 16: Starting it, and writing down how

**Files:**
- Modify: `src/rga/cli/main.py`, `README.md`
- Create: `deploy/docker-compose.yml`, `docs/demo.md`
- Test: `tests/cli/test_serve_command.py`

**Interfaces:**
- Produces: `rga serve --config <configuration> [--host] [--port]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_serve_command.py
"""The serve command builds an application without starting a server."""

from pathlib import Path

from rga.cli.main import _parser


def test_serve_is_a_known_command() -> None:
    arguments = _parser().parse_args(
        ["serve", "--config", "configs/service/synthetic.yaml", "--port", "9001"]
    )

    assert arguments.command == "serve"
    assert arguments.config == Path("configs/service/synthetic.yaml")
    assert arguments.port == 9001


def test_serve_defaults_to_a_local_port() -> None:
    arguments = _parser().parse_args(["serve", "--config", "configs/service/synthetic.yaml"])

    assert arguments.host == "127.0.0.1"
    assert arguments.port == 8000
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/cli/test_serve_command.py -v`

Expected: FAIL — argparse exits because `serve` is not a command.

- [ ] **Step 3: Add the command**

In `src/rga/cli/main.py`:

```python
    serve = commands.add_parser("serve", help="run the scoring service and the page")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
```

and in `main`:

```python
    if arguments.command == "serve":
        import uvicorn

        from rga.service.app import create_app
        from rga.service.config import load_service_config

        config = load_service_config(arguments.config)
        uvicorn.run(create_app(config), host=arguments.host, port=arguments.port)
        return 0
```

- [ ] **Step 4: Write the deployment and the demonstration script**

`deploy/docker-compose.yml` runs the service beside a Neo4j, with the artefact and the
configuration mounted, and `NEO4J_URI` pointing at the neighbouring container.

`docs/demo.md` is the running order for the defence, written so it can be followed
under pressure:

1. beforehand: `uv sync --extra cpu --extra service --extra neo4j`, train the artefact,
   check the page opens on synthetic data;
2. the main scenario: the queue, two or three cards, what to say about each —
   the summary sentences, the feature table, the thick edges in the picture;
3. the switch: start `opens3-rebac` from `feat/graph-timestamps`, fill the graph with
   `scripts/fill_live_graph.py`, restart the service with `configs/service/opens3.yaml`;
4. the finale: grant yourself `admin` with the engine's gRPC client, press refresh,
   open the new top row;
5. what to say if the live part fails: the synthetic scenario has already shown
   everything the work claims, and the adapter is covered by an integration test.

Add a section to `README.md` describing `rga train` and `rga serve` in three sentences,
linking to `docs/demo.md`.

- [ ] **Step 5: Run everything**

```bash
uv run pytest -m "not integration and not gpu"
uv run ruff check .
uv run rga serve --config configs/service/synthetic.yaml
```

Expected: green, clean, and a page at `http://127.0.0.1:8000` that works end to end.

- [ ] **Step 6: Commit**

```bash
git add src/rga/cli/main.py deploy/docker-compose.yml docs/demo.md README.md tests/cli/test_serve_command.py
git commit -m "feat: add the serve command, deployment and the demo script"
git push origin main
```

---

### Task 17: The rehearsal

Not a code task. The module is finished when the whole thing has been run once, in
order, the way it will be run on the day — because every demonstration that fails,
fails at a seam nobody rehearsed.

- [ ] **Step 1: The synthetic scenario, cold**

From a clean checkout: `uv sync --extra cpu --extra service`, train the artefact, start
the service, open the page. Expected: queue, card, subgraph, no console errors.
Time it — if training takes more than five minutes, say so in `docs/demo.md`.

- [ ] **Step 2: The live scenario**

Start `opens3-rebac` from `feat/graph-timestamps`, fill the graph, restart the service
against `configs/service/opens3.yaml`. Expected: the status line reports source `neo4j`
and capability level 2, and the queue is not empty.

- [ ] **Step 3: The finale**

Grant yourself `admin` through the engine's gRPC, press refresh, find the change.
Expected: it is in the window, the card names the self-grant in words, and the
subgraph shows it.

Record what actually happened — where it appeared in the queue, what the card said.
If the change lands far down the list, that is a result to report honestly in the
thesis, not something to hide by re-running until it looks better.

- [ ] **Step 4: Write down what the rehearsal showed**

Append a short section to `docs/demo.md`: how long each step took, what broke, what
the fallback is. Commit.

```bash
git add docs/demo.md
git commit -m "docs: record what the rehearsal showed"
git push origin main
```

---

## Self-Review

**Spec coverage.** Section 2 of the spec is Tasks 1 and 2; section 3 is Tasks 4 and 5;
section 4 is Tasks 7, 8 and 9; section 5 is Tasks 10, 11 and 12; section 6 is Tasks 13
and 14; section 7 is Tasks 15, 16 and 17. Section 8's risks are addressed where they
arise: the empty-graph risk in Task 15, the stale-artefact risk in Task 4, the time
risk by the ordering — the queue, the card and the explanation come before the
neighbourhood drawing, and the node navigation route is the cheapest thing to drop.

**One defect the spec did not foresee**, found while writing Task 3: the scorer cached
the node representations of the graph it was fitted on and indexed them with the
indices of whatever graph it was asked to score. Inside the experiment runner both are
the same object and nothing shows; a service scoring a live graph would have produced
nonsense. It is fixed as its own task, before anything depends on it.

**Type consistency.** `CandidateSet.row` (Task 6) is used by Tasks 7, 8 and 10.
`feature_gradients` returns `np.ndarray | None` and every caller handles `None`.
`Analysis.find` returns `int | None` and the route turns `None` into a 404.
`build_incident` takes `score` and `rank` as keyword arguments in every call site.
`neighbourhood` is called with `cap` by both Task 8 and Task 10.

**Left for module 5**, so that this module ends: the Kafka adapter and the streaming
mode, vectorised negative sampling, the full-size run, and the second attempt at
self-supervised training.
