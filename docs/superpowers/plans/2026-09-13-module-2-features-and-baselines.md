# Module 2: Features and Baselines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a dataset into a scored, measured pipeline — a feature representation of every permission change, an honest temporal evaluation protocol, and three levels of baseline to compare against.

**Architecture:** Features are computed in one streaming pass over the journal. A mutable `FeatureContext` walks the events in time order, and each grant's features are read off the context as it stood immediately before that grant appeared. This is linear in the number of events and cannot leak the future into the past. Every feature belongs to a named group — structural, temporal, provenance — and carries an observability mask, which is what makes both the ablation study and the portability contract work.

**Tech Stack:** NumPy, scikit-learn (metrics and outlier baselines only), existing `rga.domain` and `rga.generator`.

**Spec:** `docs/superpowers/specs/2026-09-12-rebac-graph-anomaly-design.md`, sections 6, 8 and 9.

## Global Constraints

- Python 3.12 exactly, via `uv`. Never invoke `python3` directly, always `uv run`.
- Torch is not used in this module. Nothing here may import it.
- No PyTorch Geometric, no DGL, ever.
- Cross-platform: `pathlib` only, no symlinks, entry points guarded by `if __name__ == "__main__":`.
- Code, comments, docstrings and commit messages in English. Short messages, no emoji, no AI attribution.
- Every commit is pushed: `git push origin main`. No feature branches in this repository.
- Every random draw goes through an explicitly seeded `numpy.random.Generator`.
- Verify with real exit codes, never `pytest | tail` inside a `&&` chain — the pipe reports `tail`'s status and hides failures. Use:
  `uv run ruff check .; R=$?; uv run pytest -q -m "not integration and not gpu" >/dev/null 2>&1; P=$?; echo "ruff=$R pytest=$P"`
- Test file basenames must be unique across the whole suite: pytest imports them as top-level modules and two files named alike collide.
- scikit-learn is an optional extra (`ml`). Modules that need it import it lazily inside the function, so the package stays importable without it.

---

## What exists already

From Module 1, all importable and tested:

| Symbol | Where |
|---|---|
| `EntityType`, `entity_type(id)`, `parent_bucket(object_id)` | `rga.domain.entities` |
| `RelationType`, `PermissionLevel`, `LEVEL_CARRYING`, `parse_level`, `parse_relation` | `rga.domain.relations` |
| `EventOp`, `GraphEvent(ts, op, subject, relation, object, level, actor)`, `.edge_key()` | `rga.domain.events` |
| `AccessGraph` (array fields, `neighbors`, `index_of`), `GraphBuilder`, `UNKNOWN = -1` | `rga.domain.graph` |
| `replay(events, until)`, `journal_issues(events)` | `rga.domain.replay` |
| `Dataset(config, events, labels, split_ts, window_end)`, `.window_events()`, `.anomaly_keys()` | `rga.generator.dataset` |
| `AnomalyLabel(ts, subject, relation, object, pattern)`, `.edge_key()` | `rga.generator.anomalies.base` |
| `load_dataset_config`, `dataset_config_from_document` | `rga.generator.config` |
| `save_dataset`, `load_dataset` | `rga.io.dataset_io` |
| `DAY_MS`, `HOUR_MS`, `MINUTE_MS`, `SECOND_MS`, `hour_of_day`, `is_weekend` | `rga.util.timeutil` |

A generated dataset lives in `data/small` and `data/default`; regenerate with
`uv run rga generate --config configs/generator/small.yaml --out data/small`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/rga/features/spec.py` | Feature groups, `FeatureMatrix`, `CandidateSet` |
| `src/rga/features/static.py` | Snapshot-level attributes: communities, triangles |
| `src/rga/features/context.py` | Mutable incremental graph state and its queries |
| `src/rga/features/nodes.py` | The node feature block |
| `src/rga/features/edges.py` | The edge feature block |
| `src/rga/features/build.py` | The streaming pass producing candidates |
| `src/rga/eval/metrics.py` | Ranking metrics and per-pattern breakdown |
| `src/rga/eval/experiment.py` | Runner over seeds and configurations, result tables |
| `src/rga/baselines/rules.py` | Hand-written heuristics |
| `src/rga/baselines/outliers.py` | IsolationForest and LOF |
| `configs/experiments/baselines.yaml` | Which datasets, seeds and scorers to run |

---

## Task 1: Feature vocabulary and containers

Every feature has a name and a group. The group is not bookkeeping: the ablation
study in section 9 of the spec is run by masking groups, and section 12 reuses
exactly that to tell an integrator what a weaker authorization engine costs them.

**Files:**
- Create: `src/rga/features/__init__.py`, `src/rga/features/spec.py`
- Test: `tests/features/test_feature_spec.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `FeatureGroup` StrEnum: `STRUCTURAL = "structural"`, `TEMPORAL = "temporal"`, `PROVENANCE = "provenance"`
  - `FeatureBlock(names: tuple[str, ...], groups: tuple[FeatureGroup, ...])` with `__len__`, `prefixed(prefix) -> FeatureBlock`, `concat(*blocks) -> FeatureBlock`
  - `FeatureMatrix(values: np.ndarray, mask: np.ndarray, block: FeatureBlock)` with `.n_rows`, `.n_features`, `.group_indices(group) -> np.ndarray`, `.with_groups(groups) -> FeatureMatrix`
  - `CandidateSet(keys, ts, matrix, labels, patterns)` with `.n_candidates`, `.y_true() -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_feature_spec.py
"""Feature naming, grouping and masking."""

import numpy as np
import pytest

from rga.features.spec import CandidateSet, FeatureBlock, FeatureGroup, FeatureMatrix


def _block() -> FeatureBlock:
    return FeatureBlock(
        names=("degree", "age", "actor_is_subject"),
        groups=(FeatureGroup.STRUCTURAL, FeatureGroup.TEMPORAL, FeatureGroup.PROVENANCE),
    )


def test_block_length_matches_names() -> None:
    assert len(_block()) == 3


def test_block_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        FeatureBlock(names=("a", "b"), groups=(FeatureGroup.STRUCTURAL,))


def test_block_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate feature name"):
        FeatureBlock(
            names=("a", "a"), groups=(FeatureGroup.STRUCTURAL, FeatureGroup.STRUCTURAL)
        )


def test_prefixing_renames_without_touching_groups() -> None:
    prefixed = _block().prefixed("subj_")
    assert prefixed.names == ("subj_degree", "subj_age", "subj_actor_is_subject")
    assert prefixed.groups == _block().groups


def test_concat_joins_blocks_in_order() -> None:
    joined = FeatureBlock.concat(_block().prefixed("a_"), _block().prefixed("b_"))
    assert len(joined) == 6
    assert joined.names[0] == "a_degree"
    assert joined.names[3] == "b_degree"


def test_matrix_reports_its_shape() -> None:
    matrix = FeatureMatrix(
        values=np.zeros((4, 3), dtype=np.float32),
        mask=np.ones((4, 3), dtype=bool),
        block=_block(),
    )
    assert matrix.n_rows == 4
    assert matrix.n_features == 3


def test_matrix_rejects_shape_disagreement() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        FeatureMatrix(
            values=np.zeros((4, 3), dtype=np.float32),
            mask=np.ones((4, 2), dtype=bool),
            block=_block(),
        )


def test_group_indices_select_the_right_columns() -> None:
    matrix = FeatureMatrix(
        values=np.zeros((2, 3), dtype=np.float32),
        mask=np.ones((2, 3), dtype=bool),
        block=_block(),
    )
    assert matrix.group_indices(FeatureGroup.TEMPORAL).tolist() == [1]


def test_with_groups_keeps_only_the_named_groups() -> None:
    matrix = FeatureMatrix(
        values=np.arange(6, dtype=np.float32).reshape(2, 3),
        mask=np.ones((2, 3), dtype=bool),
        block=_block(),
    )
    reduced = matrix.with_groups((FeatureGroup.STRUCTURAL, FeatureGroup.PROVENANCE))
    assert reduced.block.names == ("degree", "actor_is_subject")
    assert reduced.values.tolist() == [[0.0, 2.0], [3.0, 5.0]]


def test_candidate_set_exposes_binary_truth() -> None:
    candidates = CandidateSet(
        keys=(("user:a", 2, "bucket:x"), ("user:b", 2, "bucket:y")),
        ts=np.array([10, 20], dtype=np.int64),
        matrix=FeatureMatrix(
            values=np.zeros((2, 3), dtype=np.float32),
            mask=np.ones((2, 3), dtype=bool),
            block=_block(),
        ),
        labels=np.array([False, True]),
        patterns=("", "shadow_group"),
    )
    assert candidates.n_candidates == 2
    assert candidates.y_true().tolist() == [0, 1]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/features/test_feature_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.features'`

- [ ] **Step 3: Implement `spec.py`**

```python
# src/rga/features/spec.py
"""Feature naming, grouping and the containers that carry them.

Every feature belongs to a group. That is not bookkeeping: masking a group is
how the ablation study is run, and the same mechanism tells an integrator what a
weaker authorization engine costs them. A feature the source cannot supply is
marked unobserved rather than silently set to zero, because zero is a legitimate
value for most of these and a model cannot tell the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class FeatureGroup(StrEnum):
    """Which capability level a feature needs from the source."""

    #: Available from a bare snapshot of relation tuples.
    STRUCTURAL = "structural"
    #: Needs creation timestamps.
    TEMPORAL = "temporal"
    #: Needs the initiator of each change.
    PROVENANCE = "provenance"


@dataclass(frozen=True)
class FeatureBlock:
    """The names and groups of a contiguous run of features."""

    names: tuple[str, ...]
    groups: tuple[FeatureGroup, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.groups):
            raise ValueError("names and groups must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate feature name in block")

    def __len__(self) -> int:
        return len(self.names)

    def prefixed(self, prefix: str) -> FeatureBlock:
        """The same block with every name prefixed, for reuse on both endpoints."""
        return FeatureBlock(
            names=tuple(f"{prefix}{name}" for name in self.names), groups=self.groups
        )

    @classmethod
    def concat(cls, *blocks: FeatureBlock) -> FeatureBlock:
        """Join blocks end to end."""
        names: tuple[str, ...] = ()
        groups: tuple[FeatureGroup, ...] = ()
        for block in blocks:
            names += block.names
            groups += block.groups
        return cls(names=names, groups=groups)


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature values and their observability."""

    values: np.ndarray
    mask: np.ndarray
    block: FeatureBlock

    def __post_init__(self) -> None:
        if self.values.shape[1] != len(self.block):
            raise ValueError(
                f"values have {self.values.shape[1]} columns "
                f"but the block names {len(self.block)} features"
            )
        if self.mask.shape != self.values.shape:
            raise ValueError(f"mask shape {self.mask.shape} differs from {self.values.shape}")

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def group_indices(self, group: FeatureGroup) -> np.ndarray:
        """Column indices belonging to one group."""
        return np.array(
            [index for index, own in enumerate(self.block.groups) if own is group], dtype=np.int64
        )

    def with_groups(self, groups: tuple[FeatureGroup, ...]) -> FeatureMatrix:
        """A matrix restricted to the named groups, for the ablation study."""
        wanted = set(groups)
        columns = np.array(
            [index for index, own in enumerate(self.block.groups) if own in wanted],
            dtype=np.int64,
        )
        return FeatureMatrix(
            values=self.values[:, columns],
            mask=self.mask[:, columns],
            block=FeatureBlock(
                names=tuple(self.block.names[index] for index in columns),
                groups=tuple(self.block.groups[index] for index in columns),
            ),
        )


@dataclass(frozen=True)
class CandidateSet:
    """The edges to be scored, their features and their ground truth."""

    #: Edge identity, matching GraphEvent.edge_key().
    keys: tuple[tuple[str, int, str], ...]
    ts: np.ndarray
    matrix: FeatureMatrix
    labels: np.ndarray
    #: Pattern name per candidate, empty string when the candidate is normal.
    patterns: tuple[str, ...]

    @property
    def n_candidates(self) -> int:
        return len(self.keys)

    def y_true(self) -> np.ndarray:
        """Ground truth as integers, the shape every metric expects."""
        return self.labels.astype(np.int8)
```

- [ ] **Step 4: Create the package init**

```python
# src/rga/features/__init__.py
"""Feature extraction from the access graph."""
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/features/test_feature_spec.py -q`
Expected: PASS

- [ ] **Step 6: Commit and push**

```bash
git add src/rga/features tests/features
git commit -m "feat: add feature vocabulary and matrix containers"
git push origin main
```

---

## Task 2: Snapshot attributes

Two node attributes cannot be maintained incrementally at a sensible cost:
community membership and triangle counts. Both are computed once on the training
snapshot and treated as stable context, which is also the honest reading — an
organization's community structure does not turn over inside a two-week window.

Communities come from label propagation on the graph itself, never from the
generator's ground truth. Using the generator's departments would be a leak: the
model would be handed the answer to "does this cross an organizational boundary".

**Files:**
- Create: `src/rga/features/static.py`
- Test: `tests/features/test_static_attributes.py`

**Interfaces:**
- Consumes: `AccessGraph` (Module 1)
- Produces:
  - `StaticAttributes(community: dict[str, int], triangles: dict[str, int], clustering: dict[str, float], same_community_share: dict[str, float])`
  - `compute_static_attributes(graph: AccessGraph, rng: np.random.Generator, *, rounds: int = 20) -> StaticAttributes`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_static_attributes.py
"""Community and triangle attributes computed on a snapshot."""

import numpy as np

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay
from rga.features.static import compute_static_attributes


def _two_cliques():
    """Two triangles joined by a single bridge edge."""
    events = []
    ts = 0
    for left, right in [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d")]:
        ts += 1
        events.append(
            GraphEvent(ts, EventOp.GRANT, f"user:{left}", RelationType.MEMBER_OF, f"group:{right}")
        )
    ts += 1
    events.append(
        GraphEvent(ts, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:d")
    )
    return replay(events)


def test_every_node_receives_a_community() -> None:
    graph = _two_cliques()
    attributes = compute_static_attributes(graph, np.random.default_rng(0))
    assert set(attributes.community) == set(graph.node_ids)


def test_label_propagation_separates_weakly_linked_groups() -> None:
    graph = _two_cliques()
    attributes = compute_static_attributes(graph, np.random.default_rng(0))
    # Not asserting an exact partition: label propagation is stochastic. What must
    # hold is that it does not collapse everything into one community.
    assert len(set(attributes.community.values())) >= 2


def test_result_is_reproducible_for_a_fixed_seed() -> None:
    graph = _two_cliques()
    first = compute_static_attributes(graph, np.random.default_rng(7))
    second = compute_static_attributes(graph, np.random.default_rng(7))
    assert first.community == second.community


def test_triangles_are_counted_on_the_undirected_projection() -> None:
    events = [
        GraphEvent(1, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:b"),
        GraphEvent(2, EventOp.GRANT, "user:b", RelationType.MEMBER_OF, "group:c"),
        GraphEvent(3, EventOp.GRANT, "user:c", RelationType.MEMBER_OF, "group:a"),
    ]
    attributes = compute_static_attributes(replay(events), np.random.default_rng(0))
    # user:a and group:a are distinct nodes, so this chain closes no triangle.
    assert all(count == 0 for count in attributes.triangles.values())


def test_a_real_triangle_is_found() -> None:
    events = [
        GraphEvent(1, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:x"),
        GraphEvent(2, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:y"),
        GraphEvent(
            3, EventOp.GRANT, "group:x", RelationType.HAS_PERMISSION, "bucket:z",
            PermissionLevel.READ,
        ),
        GraphEvent(
            4, EventOp.GRANT, "group:y", RelationType.HAS_PERMISSION, "bucket:z",
            PermissionLevel.READ,
        ),
        GraphEvent(5, EventOp.GRANT, "group:x", RelationType.MEMBER_OF, "group:y"),
    ]
    attributes = compute_static_attributes(replay(events), np.random.default_rng(0))
    assert attributes.triangles["group:x"] >= 1
    assert 0.0 < attributes.clustering["group:x"] <= 1.0


def test_isolated_nodes_get_zero_not_nan() -> None:
    events = [GraphEvent(1, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:b")]
    attributes = compute_static_attributes(replay(events), np.random.default_rng(0))
    assert attributes.clustering["user:a"] == 0.0
    assert attributes.same_community_share["user:a"] in (0.0, 1.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/features/test_static_attributes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.features.static'`

- [ ] **Step 3: Implement `static.py`**

```python
# src/rga/features/static.py
"""Node attributes computed once on a snapshot.

Community membership and triangle counts cannot be maintained incrementally at a
sensible cost, and they do not need to be: an organization's community structure
does not turn over inside a two-week evaluation window. Both are computed on the
training snapshot and read as stable context.

Communities come from label propagation on the graph itself. Taking them from the
generator's departments instead would hand the model the answer to "does this
cross an organizational boundary", which is precisely what it is supposed to
infer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rga.domain.graph import AccessGraph


@dataclass(frozen=True)
class StaticAttributes:
    """Per-node attributes of one snapshot, keyed by entity id."""

    community: dict[str, int]
    triangles: dict[str, int]
    clustering: dict[str, float]
    same_community_share: dict[str, float]


def _undirected_adjacency(graph: AccessGraph) -> list[set[int]]:
    """Neighbour sets of the undirected projection, ignoring relation type."""
    adjacency: list[set[int]] = [set() for _ in range(graph.num_nodes)]
    for source, destination in zip(graph.edge_src, graph.edge_dst, strict=True):
        left, right = int(source), int(destination)
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _label_propagation(
    adjacency: list[set[int]], rng: np.random.Generator, rounds: int
) -> np.ndarray:
    """Assign each node the label most common among its neighbours.

    Visiting order is shuffled every round, which is what lets labels spread and
    is why the result depends on the seed.
    """
    labels = np.arange(len(adjacency), dtype=np.int64)
    order = np.arange(len(adjacency))

    for _ in range(rounds):
        rng.shuffle(order)
        changed = False
        for node in order:
            neighbours = adjacency[int(node)]
            if not neighbours:
                continue
            counts: dict[int, int] = {}
            for neighbour in neighbours:
                label = int(labels[neighbour])
                counts[label] = counts.get(label, 0) + 1
            # Ties broken by the smaller label, so the pass is deterministic
            # given the visiting order.
            best = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            if best != labels[int(node)]:
                labels[int(node)] = best
                changed = True
        if not changed:
            break

    return labels


def compute_static_attributes(
    graph: AccessGraph, rng: np.random.Generator, *, rounds: int = 20
) -> StaticAttributes:
    """Compute community, triangle and clustering attributes for every node."""
    adjacency = _undirected_adjacency(graph)
    labels = _label_propagation(adjacency, rng, rounds)

    community: dict[str, int] = {}
    triangles: dict[str, int] = {}
    clustering: dict[str, float] = {}
    same_share: dict[str, float] = {}

    for index, node_id in enumerate(graph.node_ids):
        neighbours = adjacency[index]
        degree = len(neighbours)

        closed = 0
        for neighbour in neighbours:
            closed += len(adjacency[neighbour] & neighbours)
        closed //= 2

        community[node_id] = int(labels[index])
        triangles[node_id] = closed
        clustering[node_id] = (
            (2.0 * closed) / (degree * (degree - 1)) if degree > 1 else 0.0
        )
        same_share[node_id] = (
            sum(1 for neighbour in neighbours if labels[neighbour] == labels[index]) / degree
            if degree
            else 0.0
        )

    return StaticAttributes(
        community=community,
        triangles=triangles,
        clustering=clustering,
        same_community_share=same_share,
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/features/test_static_attributes.py -q`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/features/static.py tests/features/test_static_attributes.py
git commit -m "feat: add snapshot community and triangle attributes"
git push origin main
```

---

## Task 3: Incremental feature context

The centre of the module. A mutable structure that walks the journal in time
order; every query answers about the graph as it stood at that moment. Features
for a grant are read off it *before* the grant is applied, which is what makes
the temporal split honest without rebuilding the graph per candidate.

**Files:**
- Create: `src/rga/features/context.py`
- Test: `tests/features/test_feature_context.py`

**Interfaces:**
- Consumes: `GraphEvent`, `EventOp`, `RelationType`, `PermissionLevel`, `entity_type`, `EntityType`
- Produces: `FeatureContext` with
  - `apply(event: GraphEvent) -> None`
  - `knows(node: str) -> bool`
  - `degree(node: str, relation: RelationType, *, incoming: bool = False) -> int`
  - `neighbours(node: str) -> set[str]`
  - `common_neighbours(a: str, b: str) -> int`
  - `adamic_adar(a: str, b: str) -> float`
  - `jaccard(a: str, b: str) -> float`
  - `path_length(a: str, b: str, *, cap: int = 3) -> int | None`
  - `level_on(subject: str, target: str) -> PermissionLevel`
  - `max_level_of(node: str) -> PermissionLevel`
  - `group_count(node: str) -> int`
  - `created(node: str) -> int | None`
  - `last_change(node: str) -> int | None`
  - `activity(node: str, window_ms: int, now: int) -> int`
  - constant `ACTIVITY_HORIZON_MS`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_feature_context.py
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/features/test_feature_context.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.features.context'`

- [ ] **Step 3: Implement `context.py`**

```python
# src/rga/features/context.py
"""Incremental graph state, queried as features are extracted.

Feature extraction walks the journal once. Before each grant is applied, the
features of that grant are read off this structure, so they describe the graph as
it stood immediately before the edge appeared. That is what makes the temporal
split honest: nothing a candidate can see happened after it.

The alternative — replaying the graph up to each candidate's timestamp — is
quadratic, and computing every candidate against one fixed snapshot leaks the
future into the past for anything inside the training span.
"""

from __future__ import annotations

from collections import deque
from math import log

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.util.timeutil import DAY_MS

#: Activity older than this is dropped, bounding memory. The widest activity
#: window a feature asks for must not exceed it.
ACTIVITY_HORIZON_MS = 7 * DAY_MS


class FeatureContext:
    """Mutable view of the graph at one point in time."""

    def __init__(self) -> None:
        self._out: dict[int, dict[str, set[str]]] = {int(r): {} for r in RelationType}
        self._in: dict[int, dict[str, set[str]]] = {int(r): {} for r in RelationType}
        self._undirected: dict[str, set[str]] = {}
        #: How many relations currently connect an unordered pair.
        self._pair_count: dict[tuple[str, str], int] = {}
        #: Edge existence and level, keyed the same way as GraphEvent.edge_key().
        self._edges: dict[tuple[str, int, str], int] = {}
        self._created: dict[str, int] = {}
        self._last_change: dict[str, int] = {}
        self._activity: dict[str, deque[int]] = {}

    # ── mutation ────────────────────────────────────────────────────────────

    def apply(self, event: GraphEvent) -> None:
        """Advance the context by one journal event."""
        self._touch(event.subject, event.ts)
        self._touch(event.object, event.ts)
        if event.actor is not None:
            self._touch(event.actor, event.ts)
            self._record_activity(event.actor, event.ts)

        self._last_change[event.subject] = event.ts
        self._record_activity(event.subject, event.ts)

        key = event.edge_key()
        if event.op is EventOp.GRANT:
            if key not in self._edges:
                self._link(event.subject, event.relation, event.object)
            self._edges[key] = int(event.level)
        elif key in self._edges:
            del self._edges[key]
            self._unlink(event.subject, event.relation, event.object)

    def _touch(self, node: str, ts: int) -> None:
        self._created.setdefault(node, ts)
        self._undirected.setdefault(node, set())

    def _record_activity(self, node: str, ts: int) -> None:
        history = self._activity.setdefault(node, deque())
        history.append(ts)
        horizon = ts - ACTIVITY_HORIZON_MS
        while history and history[0] < horizon:
            history.popleft()

    def _link(self, subject: str, relation: RelationType, target: str) -> None:
        self._out[int(relation)].setdefault(subject, set()).add(target)
        self._in[int(relation)].setdefault(target, set()).add(subject)
        if subject == target:
            return
        pair = (subject, target) if subject < target else (target, subject)
        count = self._pair_count.get(pair, 0)
        self._pair_count[pair] = count + 1
        if count == 0:
            self._undirected[subject].add(target)
            self._undirected[target].add(subject)

    def _unlink(self, subject: str, relation: RelationType, target: str) -> None:
        self._out[int(relation)].get(subject, set()).discard(target)
        self._in[int(relation)].get(target, set()).discard(subject)
        if subject == target:
            return
        pair = (subject, target) if subject < target else (target, subject)
        count = self._pair_count.get(pair, 0) - 1
        if count <= 0:
            self._pair_count.pop(pair, None)
            self._undirected[subject].discard(target)
            self._undirected[target].discard(subject)
        else:
            self._pair_count[pair] = count

    # ── queries ─────────────────────────────────────────────────────────────

    def knows(self, node: str) -> bool:
        return node in self._created

    def degree(self, node: str, relation: RelationType, *, incoming: bool = False) -> int:
        table = self._in if incoming else self._out
        return len(table[int(relation)].get(node, ()))

    def neighbours(self, node: str) -> set[str]:
        return self._undirected.get(node, set())

    def common_neighbours(self, a: str, b: str) -> int:
        return len(self.neighbours(a) & self.neighbours(b))

    def jaccard(self, a: str, b: str) -> float:
        left, right = self.neighbours(a), self.neighbours(b)
        union = len(left | right)
        return len(left & right) / union if union else 0.0

    def adamic_adar(self, a: str, b: str) -> float:
        """Shared neighbours weighted down by how many links they already have.

        A hub everyone touches says little; a neighbour shared by only these two
        says a lot.
        """
        total = 0.0
        for shared in self.neighbours(a) & self.neighbours(b):
            degree = len(self.neighbours(shared))
            if degree > 1:
                total += 1.0 / log(degree)
        return total

    def path_length(self, a: str, b: str, *, cap: int = 3) -> int | None:
        """Undirected hop distance, or None if farther than `cap` or disconnected."""
        if not self.knows(a) or not self.knows(b):
            return None
        if a == b:
            return 0

        frontier = {a}
        seen = {a}
        for distance in range(1, cap + 1):
            nxt: set[str] = set()
            for node in frontier:
                for neighbour in self._undirected.get(node, ()):
                    if neighbour == b:
                        return distance
                    if neighbour not in seen:
                        seen.add(neighbour)
                        nxt.add(neighbour)
            if not nxt:
                return None
            frontier = nxt
        return None

    def level_on(self, subject: str, target: str) -> PermissionLevel:
        """Permission the subject holds directly on the target."""
        key = (subject, int(RelationType.HAS_PERMISSION), target)
        return PermissionLevel(self._edges.get(key, int(PermissionLevel.NONE)))

    def max_level_of(self, node: str) -> PermissionLevel:
        """Strongest permission the node holds anywhere."""
        best = int(PermissionLevel.NONE)
        for target in self._out[int(RelationType.HAS_PERMISSION)].get(node, ()):
            key = (node, int(RelationType.HAS_PERMISSION), target)
            best = max(best, self._edges.get(key, 0))
        return PermissionLevel(best)

    def group_count(self, node: str) -> int:
        return self.degree(node, RelationType.MEMBER_OF)

    def created(self, node: str) -> int | None:
        return self._created.get(node)

    def last_change(self, node: str) -> int | None:
        return self._last_change.get(node)

    def activity(self, node: str, window_ms: int, now: int) -> int:
        """How many changes this node initiated within the trailing window."""
        history = self._activity.get(node)
        if not history:
            return 0
        threshold = now - window_ms
        return sum(1 for ts in history if ts >= threshold)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/features/test_feature_context.py -q`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/features/context.py tests/features/test_feature_context.py
git commit -m "feat: add incremental feature context"
git push origin main
```

---

## Task 4: Node feature block

Computed for both endpoints of every candidate and prefixed `subj_` / `obj_`.

A node the context has never seen gets `is_new = 1` with every other feature
marked unobserved. Zero is a legitimate degree, so a model handed zeros cannot
tell "no connections" from "no information"; the mask can.

**Files:**
- Create: `src/rga/features/nodes.py`
- Test: `tests/features/test_node_features.py`

**Interfaces:**
- Consumes: `FeatureContext` (Task 3), `StaticAttributes` (Task 2), `FeatureBlock`, `FeatureGroup` (Task 1)
- Produces:
  - `NODE_BLOCK: FeatureBlock` — 23 features in this exact order:
    structural — `type_user`, `type_group`, `type_bucket`, `type_object`, `deg_member_of_out`, `deg_member_of_in`, `deg_has_permission_out`, `deg_has_permission_in`, `deg_parent_of_out`, `deg_parent_of_in`, `deg_owner_of_out`, `deg_owner_of_in`, `max_level`, `group_count`, `depth`, `triangles`, `clustering`, `same_community_share`, `is_new`;
    temporal — `age`, `gap`, `activity_1h`, `activity_24h`, `activity_7d`
  - `node_features(context, static, node_id, now) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_node_features.py
"""The per-endpoint feature block."""

import numpy as np

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
            4 * HOUR_MS, EventOp.GRANT, "bucket:x", RelationType.PARENT_OF,
            "object:x/file.dat",
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
    # Two members point at the group, and the group holds one permission.
    assert named["deg_member_of_in"][0] > 0.0
    assert named["deg_has_permission_out"][0] > 0.0
    assert named["deg_member_of_out"][0] == 0.0


def test_max_level_is_scaled_into_the_unit_interval() -> None:
    context, static = _prepared()
    named = _named(*node_features(context, static, "group:g", now=5 * HOUR_MS))
    assert named["max_level"][0] == float(PermissionLevel.WRITE) / float(PermissionLevel.ADMIN)


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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/features/test_node_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.features.nodes'`

- [ ] **Step 3: Implement `nodes.py`**

```python
# src/rga/features/nodes.py
"""The per-endpoint feature block.

Degrees are kept separate per relation and direction on purpose. Membership in a
group and a permission on a resource are different kinds of link, and summing
them into one degree throws away the distinction the whole model rests on.

Counts are put through log1p because degree distributions here are heavy-tailed:
a hub bucket with four hundred objects would otherwise dominate every linear
model in the baseline suite.
"""

from __future__ import annotations

import numpy as np

from rga.domain.entities import EntityType, entity_type
from rga.domain.relations import PermissionLevel, RelationType
from rga.features.context import FeatureContext
from rga.features.spec import FeatureBlock, FeatureGroup
from rga.features.static import StaticAttributes
from rga.util.timeutil import DAY_MS, HOUR_MS

_STRUCTURAL_NAMES = (
    "type_user",
    "type_group",
    "type_bucket",
    "type_object",
    "deg_member_of_out",
    "deg_member_of_in",
    "deg_has_permission_out",
    "deg_has_permission_in",
    "deg_parent_of_out",
    "deg_parent_of_in",
    "deg_owner_of_out",
    "deg_owner_of_in",
    "max_level",
    "group_count",
    "depth",
    "triangles",
    "clustering",
    "same_community_share",
    "is_new",
)
_TEMPORAL_NAMES = ("age", "gap", "activity_1h", "activity_24h", "activity_7d")

NODE_BLOCK = FeatureBlock(
    names=_STRUCTURAL_NAMES + _TEMPORAL_NAMES,
    groups=(FeatureGroup.STRUCTURAL,) * len(_STRUCTURAL_NAMES)
    + (FeatureGroup.TEMPORAL,) * len(_TEMPORAL_NAMES),
)

_INDEX = {name: position for position, name in enumerate(NODE_BLOCK.names)}

#: The one feature that stays observed for a node the context has never seen.
_IS_NEW = _INDEX["is_new"]
#: Attributes that only exist for nodes present in the snapshot.
_STATIC_FEATURES = ("triangles", "clustering", "same_community_share")

_DEPTH_BY_TYPE = {
    EntityType.USER: 0.0,
    EntityType.GROUP: 0.0,
    EntityType.BUCKET: 0.0,
    EntityType.OBJECT: 1.0,
}


def node_features(
    context: FeatureContext, static: StaticAttributes, node_id: str, now: int
) -> tuple[np.ndarray, np.ndarray]:
    """Feature values and observability for one endpoint."""
    values = np.zeros(len(NODE_BLOCK), dtype=np.float32)
    mask = np.zeros(len(NODE_BLOCK), dtype=bool)

    if not context.knows(node_id):
        # A node nobody has linked yet. Everything except the flag is unknown,
        # and unknown must not be confused with zero.
        values[_IS_NEW] = 1.0
        mask[_IS_NEW] = True
        return values, mask

    mask[:] = True
    kind = entity_type(node_id)

    values[_INDEX["type_user"]] = float(kind is EntityType.USER)
    values[_INDEX["type_group"]] = float(kind is EntityType.GROUP)
    values[_INDEX["type_bucket"]] = float(kind is EntityType.BUCKET)
    values[_INDEX["type_object"]] = float(kind is EntityType.OBJECT)

    for relation in RelationType:
        stem = f"deg_{relation.name.lower()}"
        values[_INDEX[f"{stem}_out"]] = np.log1p(context.degree(node_id, relation))
        values[_INDEX[f"{stem}_in"]] = np.log1p(
            context.degree(node_id, relation, incoming=True)
        )

    values[_INDEX["max_level"]] = float(context.max_level_of(node_id)) / float(
        PermissionLevel.ADMIN
    )
    values[_INDEX["group_count"]] = np.log1p(context.group_count(node_id))
    values[_INDEX["depth"]] = _DEPTH_BY_TYPE[kind]

    if node_id in static.community:
        values[_INDEX["triangles"]] = np.log1p(static.triangles[node_id])
        values[_INDEX["clustering"]] = static.clustering[node_id]
        values[_INDEX["same_community_share"]] = static.same_community_share[node_id]
    else:
        # Appeared after the snapshot was taken; its structure there is unknown.
        for name in _STATIC_FEATURES:
            mask[_INDEX[name]] = False

    values[_IS_NEW] = 0.0

    created = context.created(node_id)
    values[_INDEX["age"]] = np.log1p(max(now - created, 0)) if created is not None else 0.0
    last = context.last_change(node_id)
    values[_INDEX["gap"]] = np.log1p(max(now - last, 0)) if last is not None else 0.0
    values[_INDEX["activity_1h"]] = np.log1p(context.activity(node_id, HOUR_MS, now))
    values[_INDEX["activity_24h"]] = np.log1p(context.activity(node_id, DAY_MS, now))
    values[_INDEX["activity_7d"]] = np.log1p(context.activity(node_id, 7 * DAY_MS, now))

    return values, mask
```

Every position is looked up by name. Writing literal indices for a block this wide
invites exactly one bug — an off-by-one that silently overwrites a neighbouring
feature and shows up only as an unexplained dip in a metric weeks later.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/features/test_node_features.py -q`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/features/nodes.py tests/features/test_node_features.py
git commit -m "feat: add node feature block"
git push origin main
```

---

## Task 5: Edge feature block

Everything specific to the change itself, as opposed to its endpoints.

**Files:**
- Create: `src/rga/features/edges.py`
- Test: `tests/features/test_edge_features.py`

**Interfaces:**
- Consumes: `FeatureContext`, `StaticAttributes`, `FeatureBlock`, `FeatureGroup`, `GraphEvent`
- Produces:
  - `PATH_CAP = 3`
  - `EDGE_BLOCK: FeatureBlock` — 22 features:
    structural — `rel_member_of`, `rel_has_permission`, `rel_parent_of`, `rel_owner_of`, `level_ordinal`, `is_permission`, `common_neighbours`, `adamic_adar`, `jaccard`, `path_hops`, `path_unreachable`, `bypasses_bucket`, `level_jump`, `same_community`;
    temporal — `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_off_hours`, `is_weekend`;
    provenance — `actor_is_subject`, `actor_level_on_object`
  - `edge_features(context, static, event) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_edge_features.py
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
            WEDNESDAY + 3, EventOp.GRANT, "bucket:x", RelationType.PARENT_OF,
            "object:x/file.dat",
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


def test_block_length_matches_the_vectors() -> None:
    context, static = _prepared()
    event = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.ADMIN, actor="user:a",
    )
    values, mask = edge_features(context, static, event)
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
    event = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.ADMIN, actor="user:root",
    )
    named = _named(*edge_features(context, static, event))
    assert named["rel_has_permission"][0] == 1.0
    assert named["rel_member_of"][0] == 0.0
    assert named["level_ordinal"][0] == 1.0
    assert named["is_permission"][0] == 1.0


def test_similarity_reflects_shared_structure() -> None:
    context, static = _prepared()
    close = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.READ, actor="user:root",
    )
    far = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:unrelated", PermissionLevel.READ, actor="user:root",
    )
    near_named = _named(*edge_features(context, static, close))
    far_named = _named(*edge_features(context, static, far))
    assert near_named["common_neighbours"][0] > far_named["common_neighbours"][0]


def test_path_hops_and_unreachable_flag() -> None:
    context, static = _prepared()
    reachable = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.READ, actor="user:root",
    )
    named = _named(*edge_features(context, static, reachable))
    assert named["path_unreachable"][0] == 0.0
    assert 0.0 < named["path_hops"][0] <= 1.0

    disconnected = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:elsewhere", PermissionLevel.READ, actor="user:root",
    )
    named = _named(*edge_features(context, static, disconnected))
    assert named["path_unreachable"][0] == 1.0


def test_bypass_flag_is_only_meaningful_for_objects() -> None:
    context, static = _prepared()
    on_object = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "object:x/file.dat", PermissionLevel.WRITE, actor="user:root",
    )
    named = _named(*edge_features(context, static, on_object))
    # user:a holds nothing on bucket:x directly, so this grant skips the bucket.
    assert named["bypasses_bucket"] == (1.0, True)

    on_bucket = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.WRITE, actor="user:root",
    )
    named = _named(*edge_features(context, static, on_bucket))
    assert named["bypasses_bucket"][1] is False


def test_level_jump_measures_the_step_up() -> None:
    context, static = _prepared()
    context.apply(
        GraphEvent(
            WEDNESDAY + 5, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
            "bucket:x", PermissionLevel.READ, actor="user:root",
        )
    )
    event = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.ADMIN, actor="user:root",
    )
    named = _named(*edge_features(context, static, event))
    assert named["level_jump"][0] > 0.0


def test_cyclical_time_encoding_and_calendar_flags() -> None:
    context, static = _prepared()
    # Wednesday 03:00 UTC — off hours, a weekday.
    event = GraphEvent(
        WEDNESDAY + 3 * HOUR_MS, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.READ, actor="user:root",
    )
    named = _named(*edge_features(context, static, event))
    assert -1.0 <= named["hour_sin"][0] <= 1.0
    assert -1.0 <= named["dow_cos"][0] <= 1.0
    assert named["is_off_hours"][0] == 1.0
    assert named["is_weekend"][0] == 0.0

    saturday = GraphEvent(
        WEDNESDAY + 3 * DAY_MS + 12 * HOUR_MS, EventOp.GRANT, "user:a",
        RelationType.HAS_PERMISSION, "bucket:x", PermissionLevel.READ, actor="user:root",
    )
    named = _named(*edge_features(context, static, saturday))
    assert named["is_weekend"][0] == 1.0
    assert named["is_off_hours"][0] == 0.0


def test_provenance_is_masked_when_the_actor_is_unknown() -> None:
    context, static = _prepared()
    event = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.READ,
    )
    named = _named(*edge_features(context, static, event))
    assert named["actor_is_subject"] == (0.0, False)
    assert named["actor_level_on_object"] == (0.0, False)


def test_self_grant_is_visible_when_the_actor_is_known() -> None:
    context, static = _prepared()
    event = GraphEvent(
        WEDNESDAY + 10, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
        "bucket:x", PermissionLevel.ADMIN, actor="user:a",
    )
    named = _named(*edge_features(context, static, event))
    assert named["actor_is_subject"] == (1.0, True)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/features/test_edge_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.features.edges'`

- [ ] **Step 3: Implement `edges.py`**

```python
# src/rga/features/edges.py
"""The per-change feature block.

The age of the edge itself is deliberately absent. For a candidate drawn from the
evaluation window it is zero by construction and carries nothing; what carries
the temporal signal is when in the day and week the change happened, and how
recently and how often its subject acted — and those live on the endpoint block.

Provenance is masked, not zeroed, when the initiator is unknown. Most
authorization engines do not record it, and a model given zeros would learn that
"nobody granted this" rather than "we do not know who did".
"""

from __future__ import annotations

import numpy as np

from rga.domain.entities import EntityType, entity_type, parent_bucket
from rga.domain.events import GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.features.context import FeatureContext
from rga.features.spec import FeatureBlock, FeatureGroup
from rga.features.static import StaticAttributes
from rga.util.timeutil import hour_of_day, is_weekend

#: Hop limit for the shortest-path feature. Three hops spans the whole
#: user -> group -> bucket -> object chain the engine itself traverses.
PATH_CAP = 3

#: Hours outside which activity counts as off-hours, matching the generator.
_WORKING_HOURS = (9, 19)

_STRUCTURAL_NAMES = (
    "rel_member_of",
    "rel_has_permission",
    "rel_parent_of",
    "rel_owner_of",
    "level_ordinal",
    "is_permission",
    "common_neighbours",
    "adamic_adar",
    "jaccard",
    "path_hops",
    "path_unreachable",
    "bypasses_bucket",
    "level_jump",
    "same_community",
)
_TEMPORAL_NAMES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_off_hours", "is_weekend")
_PROVENANCE_NAMES = ("actor_is_subject", "actor_level_on_object")

EDGE_BLOCK = FeatureBlock(
    names=_STRUCTURAL_NAMES + _TEMPORAL_NAMES + _PROVENANCE_NAMES,
    groups=(FeatureGroup.STRUCTURAL,) * len(_STRUCTURAL_NAMES)
    + (FeatureGroup.TEMPORAL,) * len(_TEMPORAL_NAMES)
    + (FeatureGroup.PROVENANCE,) * len(_PROVENANCE_NAMES),
)

_INDEX = {name: position for position, name in enumerate(EDGE_BLOCK.names)}


def _prior_level(context: FeatureContext, event: GraphEvent) -> PermissionLevel:
    """Strongest permission the subject already held over this target or its bucket."""
    best = context.level_on(event.subject, event.object)
    if entity_type(event.object) is EntityType.OBJECT:
        on_bucket = context.level_on(event.subject, parent_bucket(event.object))
        best = max(best, on_bucket)
    return PermissionLevel(int(best))


def edge_features(
    context: FeatureContext, static: StaticAttributes, event: GraphEvent
) -> tuple[np.ndarray, np.ndarray]:
    """Feature values and observability for one change."""
    values = np.zeros(len(EDGE_BLOCK), dtype=np.float32)
    mask = np.ones(len(EDGE_BLOCK), dtype=bool)

    values[_INDEX["rel_member_of"]] = float(event.relation is RelationType.MEMBER_OF)
    values[_INDEX["rel_has_permission"]] = float(event.relation is RelationType.HAS_PERMISSION)
    values[_INDEX["rel_parent_of"]] = float(event.relation is RelationType.PARENT_OF)
    values[_INDEX["rel_owner_of"]] = float(event.relation is RelationType.OWNER_OF)
    values[_INDEX["level_ordinal"]] = float(event.level) / float(PermissionLevel.ADMIN)
    values[_INDEX["is_permission"]] = float(event.relation is RelationType.HAS_PERMISSION)

    values[_INDEX["common_neighbours"]] = np.log1p(
        context.common_neighbours(event.subject, event.object)
    )
    values[_INDEX["adamic_adar"]] = np.log1p(context.adamic_adar(event.subject, event.object))
    values[_INDEX["jaccard"]] = context.jaccard(event.subject, event.object)

    hops = context.path_length(event.subject, event.object, cap=PATH_CAP)
    if hops is None:
        values[_INDEX["path_hops"]] = 1.0
        values[_INDEX["path_unreachable"]] = 1.0
    else:
        values[_INDEX["path_hops"]] = hops / PATH_CAP
        values[_INDEX["path_unreachable"]] = 0.0

    if entity_type(event.object) is EntityType.OBJECT:
        holds_bucket = context.level_on(event.subject, parent_bucket(event.object))
        values[_INDEX["bypasses_bucket"]] = float(holds_bucket is PermissionLevel.NONE)
    else:
        # Nothing contains a bucket, so the question does not arise.
        mask[_INDEX["bypasses_bucket"]] = False

    jump = int(event.level) - int(_prior_level(context, event))
    values[_INDEX["level_jump"]] = jump / float(PermissionLevel.ADMIN)

    subject_community = static.community.get(event.subject)
    object_community = static.community.get(event.object)
    if subject_community is None or object_community is None:
        mask[_INDEX["same_community"]] = False
    else:
        values[_INDEX["same_community"]] = float(subject_community == object_community)

    hour = hour_of_day(event.ts)
    weekday = (event.ts // 86_400_000 + 3) % 7  # 1970-01-01 was a Thursday.
    values[_INDEX["hour_sin"]] = np.sin(2.0 * np.pi * hour / 24.0)
    values[_INDEX["hour_cos"]] = np.cos(2.0 * np.pi * hour / 24.0)
    values[_INDEX["dow_sin"]] = np.sin(2.0 * np.pi * weekday / 7.0)
    values[_INDEX["dow_cos"]] = np.cos(2.0 * np.pi * weekday / 7.0)
    values[_INDEX["is_off_hours"]] = float(not _WORKING_HOURS[0] <= hour < _WORKING_HOURS[1])
    values[_INDEX["is_weekend"]] = float(is_weekend(event.ts))

    if event.actor is None:
        mask[_INDEX["actor_is_subject"]] = False
        mask[_INDEX["actor_level_on_object"]] = False
    else:
        values[_INDEX["actor_is_subject"]] = float(event.actor == event.subject)
        values[_INDEX["actor_level_on_object"]] = float(
            context.level_on(event.actor, event.object)
        ) / float(PermissionLevel.ADMIN)

    return values, mask
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/features/test_edge_features.py -q`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/features/edges.py tests/features/test_edge_features.py
git commit -m "feat: add edge feature block"
git push origin main
```

---

## Task 6: Candidate builder

One pass over the journal, producing the rows every scorer consumes.

Two decisions are worth stating, because both are places where a plausible
shortcut would quietly invalidate the results.

*Why a streaming pass rather than a snapshot.* Computing every candidate against
one fixed graph would mean a grant from day five of the training span sees the
graph of day fifty-three. That is the future leaking into the past, and it would
make the training-span features — the ones the outlier baselines are fitted on —
describe a world that did not exist yet.

*Why a warm-up.* The first day of the journal founds the organization in one
burst against an empty graph. Those rows are degenerate: no neighbours, no
history, no structure to describe. Including them would teach an outlier detector
that "an empty neighbourhood" is normal, which it is not once the company exists.

**Files:**
- Create: `src/rga/features/build.py`
- Test: `tests/features/test_candidate_build.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5; `Dataset` (Module 1)
- Produces:
  - `Span` StrEnum: `TRAIN = "train"`, `EVAL = "eval"`
  - `WARMUP_DAYS = 7`
  - `CANDIDATE_BLOCK: FeatureBlock` = `EDGE_BLOCK` + `NODE_BLOCK.prefixed("subj_")` + `NODE_BLOCK.prefixed("obj_")`
  - `build_candidates(dataset, span, *, static_seed=0, warmup_days=WARMUP_DAYS) -> CandidateSet`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_candidate_build.py
"""The streaming pass that turns a dataset into scored rows."""

from pathlib import Path

import numpy as np

from rga.features.build import CANDIDATE_BLOCK, Span, build_candidates
from rga.features.edges import EDGE_BLOCK
from rga.features.nodes import NODE_BLOCK
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def _dataset():
    return build_dataset(CONFIG)


def test_block_is_the_edge_block_plus_both_endpoints() -> None:
    assert len(CANDIDATE_BLOCK) == len(EDGE_BLOCK) + 2 * len(NODE_BLOCK)
    assert CANDIDATE_BLOCK.names[len(EDGE_BLOCK)].startswith("subj_")
    assert CANDIDATE_BLOCK.names[len(EDGE_BLOCK) + len(NODE_BLOCK)].startswith("obj_")


def test_eval_candidates_come_from_the_window_only() -> None:
    dataset = _dataset()
    candidates = build_candidates(dataset, Span.EVAL)
    assert candidates.n_candidates > 0
    assert bool((candidates.ts >= dataset.split_ts).all())
    assert bool((candidates.ts < dataset.window_end).all())


def test_train_candidates_stop_at_the_split_and_skip_the_warm_up() -> None:
    dataset = _dataset()
    candidates = build_candidates(dataset, Span.TRAIN)
    assert candidates.n_candidates > 0
    assert bool((candidates.ts < dataset.split_ts).all())
    assert bool((candidates.ts > dataset.config.timeline.start_ts).all())


def test_matrix_shape_agrees_with_the_block() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    assert candidates.matrix.n_features == len(CANDIDATE_BLOCK)
    assert candidates.matrix.n_rows == candidates.n_candidates
    assert candidates.matrix.values.dtype == np.float32


def test_every_injected_anomaly_is_labelled_among_the_candidates() -> None:
    dataset = _dataset()
    candidates = build_candidates(dataset, Span.EVAL)
    labelled = {
        key for key, flag in zip(candidates.keys, candidates.labels, strict=True) if flag
    }
    assert labelled == dataset.anomaly_keys()


def test_training_candidates_carry_no_labels() -> None:
    # Contamination, when configured, is deliberately unlabelled.
    candidates = build_candidates(_dataset(), Span.TRAIN)
    assert not candidates.labels.any()


def test_patterns_accompany_the_labels() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    named = {
        pattern
        for pattern, flag in zip(candidates.patterns, candidates.labels, strict=True)
        if flag
    }
    assert named
    assert "" not in named


def test_no_feature_is_nan_or_infinite() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    assert np.isfinite(candidates.matrix.values).all()


def test_provenance_is_observed_because_the_generator_records_actors() -> None:
    candidates = build_candidates(_dataset(), Span.EVAL)
    columns = candidates.matrix.group_indices(FeatureGroup.PROVENANCE)
    assert columns.size > 0
    assert candidates.matrix.mask[:, columns].any()


def test_build_is_reproducible() -> None:
    dataset = _dataset()
    first = build_candidates(dataset, Span.EVAL)
    second = build_candidates(dataset, Span.EVAL)
    assert np.array_equal(first.matrix.values, second.matrix.values)
    assert first.keys == second.keys
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/features/test_candidate_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.features.build'`

- [ ] **Step 3: Implement `build.py`**

```python
# src/rga/features/build.py
"""Turning a dataset into scored rows.

A single pass over the journal, in time order. Each grant's features are read off
the context before that grant is applied, so a candidate describes the graph as it
stood immediately before the edge appeared.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from rga.domain.events import EventOp, GraphEvent
from rga.domain.replay import replay
from rga.features.context import FeatureContext
from rga.features.edges import EDGE_BLOCK, edge_features
from rga.features.nodes import NODE_BLOCK, node_features
from rga.features.spec import CandidateSet, FeatureBlock, FeatureMatrix
from rga.features.static import compute_static_attributes
from rga.generator.dataset import Dataset
from rga.util.timeutil import DAY_MS


class Span(StrEnum):
    """Which side of the temporal split candidates are drawn from."""

    TRAIN = "train"
    EVAL = "eval"


#: Days at the start of the journal excluded from training candidates. Day zero
#: founds the organization in one burst against an empty graph, and those rows
#: describe no structure at all.
WARMUP_DAYS = 7

CANDIDATE_BLOCK = FeatureBlock.concat(
    EDGE_BLOCK, NODE_BLOCK.prefixed("subj_"), NODE_BLOCK.prefixed("obj_")
)


def _span_bounds(dataset: Dataset, span: Span, warmup_days: int) -> tuple[int, int]:
    """Half-open [start, end) of the requested span."""
    if span is Span.EVAL:
        return dataset.split_ts, dataset.window_end
    return dataset.config.timeline.start_ts + warmup_days * DAY_MS, dataset.split_ts


def build_candidates(
    dataset: Dataset,
    span: Span,
    *,
    static_seed: int = 0,
    warmup_days: int = WARMUP_DAYS,
) -> CandidateSet:
    """Extract features for every grant inside the requested span."""
    static = compute_static_attributes(
        replay(dataset.events, until=dataset.split_ts), np.random.default_rng(static_seed)
    )
    # Labels are matched on identity and time together: a normal re-grant of the
    # same edge later in the window is a different candidate, not an anomaly.
    label_of = {(label.edge_key(), label.ts): label.pattern for label in dataset.labels}

    start, end = _span_bounds(dataset, span, warmup_days)
    context = FeatureContext()

    keys: list[tuple[str, int, str]] = []
    stamps: list[int] = []
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[bool] = []
    patterns: list[str] = []

    for event in dataset.events:
        if event.op is EventOp.GRANT and start <= event.ts < end:
            values, mask = _candidate_row(context, static, event)
            keys.append(event.edge_key())
            stamps.append(event.ts)
            rows.append(values)
            masks.append(mask)
            pattern = label_of.get((event.edge_key(), event.ts), "")
            labels.append(bool(pattern))
            patterns.append(pattern)
        context.apply(event)

    matrix = FeatureMatrix(
        values=np.vstack(rows).astype(np.float32)
        if rows
        else np.zeros((0, len(CANDIDATE_BLOCK)), dtype=np.float32),
        mask=np.vstack(masks)
        if masks
        else np.zeros((0, len(CANDIDATE_BLOCK)), dtype=bool),
        block=CANDIDATE_BLOCK,
    )
    return CandidateSet(
        keys=tuple(keys),
        ts=np.array(stamps, dtype=np.int64),
        matrix=matrix,
        labels=np.array(labels, dtype=bool),
        patterns=tuple(patterns),
    )


def _candidate_row(
    context: FeatureContext, static, event: GraphEvent
) -> tuple[np.ndarray, np.ndarray]:
    """The full feature row: the change itself, then both endpoints."""
    edge_values, edge_mask = edge_features(context, static, event)
    subject_values, subject_mask = node_features(context, static, event.subject, event.ts)
    object_values, object_mask = node_features(context, static, event.object, event.ts)
    return (
        np.concatenate([edge_values, subject_values, object_values]),
        np.concatenate([edge_mask, subject_mask, object_mask]),
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/features/test_candidate_build.py -q`
Expected: PASS

If `test_train_candidates_stop_at_the_split_and_skip_the_warm_up` finds zero
candidates, the small config's warm-up swallows the whole training span — check
that `timeline.days` minus `eval_window_days` comfortably exceeds `WARMUP_DAYS`.

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/features/build.py tests/features/test_candidate_build.py
git commit -m "feat: add streaming candidate builder"
git push origin main
```

---

## Task 7: Metrics harness

Section 9 of the spec fixes what is reported and why. Area under the
precision-recall curve is the headline: anomalies are a few percent of the window,
and area under the ROC curve looks flattering at that imbalance because the
enormous negative class swamps the false-positive rate.

Precision and recall at `k` are the operational numbers — `k` is the size of the
queue a human will actually work through. Lift states the same thing in the form
a reader can judge instantly: how many times better than reading the window in
random order.

**Files:**
- Create: `src/rga/eval/__init__.py`, `src/rga/eval/metrics.py`
- Modify: `pyproject.toml` (nothing to add — `ml` extra already declares scikit-learn)
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: nothing from this module
- Produces:
  - `DEFAULT_KS = (20, 50, 100)`
  - `RankingMetrics(pr_auc, roc_auc, precision_at, recall_at, lift_at, n_candidates, n_anomalies)` with `.as_row() -> dict[str, float]`
  - `evaluate_ranking(y_true, scores, *, ks=DEFAULT_KS) -> RankingMetrics`
  - `recall_by_pattern(y_true, scores, patterns, *, k) -> dict[str, float]`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
"""Ranking metrics and the per-pattern breakdown."""

import numpy as np
import pytest

from rga.eval.metrics import evaluate_ranking, recall_by_pattern


def test_perfect_ranking_scores_one() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.05, 0.01])
    metrics = evaluate_ranking(y_true, scores, ks=(2,))
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.recall_at[2] == pytest.approx(1.0)
    assert metrics.precision_at[2] == pytest.approx(1.0)


def test_reversed_ranking_scores_poorly() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.01, 0.05, 0.8, 0.9, 0.7, 0.6])
    metrics = evaluate_ranking(y_true, scores, ks=(2,))
    assert metrics.roc_auc < 0.5
    assert metrics.recall_at[2] == pytest.approx(0.0)


def test_lift_compares_against_the_base_rate() -> None:
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.arange(10, 0, -1, dtype=float)
    metrics = evaluate_ranking(y_true, scores, ks=(2,))
    # Base rate 0.1, precision@2 is 0.5, so the queue is five times denser.
    assert metrics.lift_at[2] == pytest.approx(5.0)


def test_k_larger_than_the_candidate_pool_is_clipped() -> None:
    y_true = np.array([1, 0, 0])
    scores = np.array([0.9, 0.5, 0.1])
    metrics = evaluate_ranking(y_true, scores, ks=(100,))
    assert metrics.recall_at[100] == pytest.approx(1.0)
    assert metrics.precision_at[100] == pytest.approx(1.0 / 3.0)


def test_counts_are_reported() -> None:
    y_true = np.array([1, 1, 0, 0])
    metrics = evaluate_ranking(y_true, np.array([0.4, 0.3, 0.2, 0.1]), ks=(2,))
    assert metrics.n_candidates == 4
    assert metrics.n_anomalies == 2


def test_a_window_without_anomalies_yields_nan_rather_than_a_lie() -> None:
    metrics = evaluate_ranking(np.zeros(5, dtype=int), np.arange(5, dtype=float), ks=(2,))
    assert np.isnan(metrics.pr_auc)
    assert np.isnan(metrics.roc_auc)
    assert np.isnan(metrics.recall_at[2])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_ranking(np.array([1, 0]), np.array([0.5]))


def test_as_row_flattens_for_a_results_table() -> None:
    metrics = evaluate_ranking(np.array([1, 0, 0, 0]), np.array([0.9, 0.3, 0.2, 0.1]), ks=(2,))
    row = metrics.as_row()
    assert row["pr_auc"] == pytest.approx(1.0)
    assert "precision_at_2" in row
    assert "lift_at_2" in row


def test_recall_by_pattern_splits_the_queue() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    patterns = ("shadow_group", "grant_burst", "", "")
    found = recall_by_pattern(y_true, scores, patterns, k=2)
    assert found["shadow_group"] == pytest.approx(1.0)
    assert found["grant_burst"] == pytest.approx(0.0)


def test_recall_by_pattern_ignores_normal_rows() -> None:
    y_true = np.array([1, 0])
    scores = np.array([0.9, 0.8])
    found = recall_by_pattern(y_true, scores, ("self_grant_admin", ""), k=1)
    assert set(found) == {"self_grant_admin"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/eval/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.eval'`

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/rga/eval/metrics.py
"""Ranking metrics.

Area under the precision-recall curve is the headline number. Anomalies are a few
percent of the window, and at that imbalance the area under the ROC curve looks
flattering: the negative class is so large that even many false positives barely
move the false-positive rate. It is reported anyway, because it is what most of
the literature quotes and leaving it out invites the question.

Precision and recall at k are the operational numbers — k is the size of the
queue a person will actually work through. Lift restates precision at k against
the base rate, which is the form a reader can judge without arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Queue sizes a security analyst plausibly works through in one sitting.
DEFAULT_KS = (20, 50, 100)


@dataclass(frozen=True)
class RankingMetrics:
    """One scorer's performance on one candidate set."""

    pr_auc: float
    roc_auc: float
    precision_at: dict[int, float]
    recall_at: dict[int, float]
    lift_at: dict[int, float]
    n_candidates: int
    n_anomalies: int

    def as_row(self) -> dict[str, float]:
        """Flatten into one record for a results table."""
        row: dict[str, float] = {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "n_candidates": float(self.n_candidates),
            "n_anomalies": float(self.n_anomalies),
        }
        for k, value in self.precision_at.items():
            row[f"precision_at_{k}"] = value
        for k, value in self.recall_at.items():
            row[f"recall_at_{k}"] = value
        for k, value in self.lift_at.items():
            row[f"lift_at_{k}"] = value
        return row


def evaluate_ranking(
    y_true: np.ndarray, scores: np.ndarray, *, ks: Sequence[int] = DEFAULT_KS
) -> RankingMetrics:
    """Score a ranking of candidates against ground truth.

    With no anomalies present every rate is undefined; NaN is returned rather
    than a zero that would average into a results table as if it were measured.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    truth = np.asarray(y_true).astype(np.int8)
    total = len(truth)
    positives = int(truth.sum())

    if positives == 0:
        nan_by_k = dict.fromkeys(ks, float("nan"))
        return RankingMetrics(
            pr_auc=float("nan"),
            roc_auc=float("nan"),
            precision_at=dict(nan_by_k),
            recall_at=dict(nan_by_k),
            lift_at=dict(nan_by_k),
            n_candidates=total,
            n_anomalies=0,
        )

    from sklearn.metrics import average_precision_score, roc_auc_score

    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranked = truth[order]
    base_rate = positives / total

    precision_at: dict[int, float] = {}
    recall_at: dict[int, float] = {}
    lift_at: dict[int, float] = {}
    for k in ks:
        cut = min(int(k), total)
        hits = int(ranked[:cut].sum())
        precision = hits / cut
        precision_at[int(k)] = precision
        recall_at[int(k)] = hits / positives
        lift_at[int(k)] = precision / base_rate

    return RankingMetrics(
        pr_auc=float(average_precision_score(truth, scores)),
        roc_auc=float(roc_auc_score(truth, scores)),
        precision_at=precision_at,
        recall_at=recall_at,
        lift_at=lift_at,
        n_candidates=total,
        n_anomalies=positives,
    )


def recall_by_pattern(
    y_true: np.ndarray, scores: np.ndarray, patterns: Sequence[str], *, k: int
) -> dict[str, float]:
    """Share of each pattern's edges that reach the top k.

    This is the breakdown section 9 asks for: which threats the system catches and
    which it misses. An aggregate number hides a detector that is excellent at one
    loud pattern and blind to the rest.
    """
    truth = np.asarray(y_true).astype(bool)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    top = set(order[: min(int(k), len(order))].tolist())

    totals: dict[str, int] = {}
    found: dict[str, int] = {}
    for index, (is_anomaly, pattern) in enumerate(zip(truth, patterns, strict=True)):
        if not is_anomaly or not pattern:
            continue
        totals[pattern] = totals.get(pattern, 0) + 1
        if index in top:
            found[pattern] = found.get(pattern, 0) + 1

    return {pattern: found.get(pattern, 0) / count for pattern, count in totals.items()}
```

```python
# src/rga/eval/__init__.py
"""Evaluation protocol, metrics and experiment running."""
```

- [ ] **Step 4: Install the extra and run the tests**

Run:
```bash
uv sync --extra cpu --extra ml
uv run pytest tests/eval/test_metrics.py -q
```
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/eval tests/eval uv.lock
git commit -m "feat: add ranking metrics harness"
git push origin main
```

---

## Task 8: Rule baseline

The lowest rung of comparison, and the one that decides whether the dataset is
worth anything. If hand-written conditions already rank nearly as well as
everything above them, the synthetic data is too easy and the generator needs
recalibrating — not the model. That makes this task a measuring instrument as
much as a baseline.

**Files:**
- Create: `src/rga/baselines/__init__.py`, `src/rga/baselines/base.py`, `src/rga/baselines/rules.py`
- Modify: `src/rga/features/spec.py` — add `FeatureMatrix.column(name) -> np.ndarray`
- Test: `tests/baselines/test_rules.py`, add one case to `tests/features/test_feature_spec.py`

**Interfaces:**
- Consumes: `CandidateSet`, `FeatureMatrix`, `FeatureGroup`
- Produces:
  - `FeatureMatrix.column(name: str) -> np.ndarray` and `FeatureMatrix.observed(name: str) -> np.ndarray`
  - `Scorer` Protocol: attribute `name: str`, `fit(train: CandidateSet) -> None`, `score(candidates: CandidateSet) -> np.ndarray`
  - `dense_matrix(matrix: FeatureMatrix) -> np.ndarray` — unobserved entries zeroed, with one observed-fraction column appended per group
  - `RuleScorer()` with `name = "rules"`

- [ ] **Step 1: Add the column accessor to `spec.py`**

Append to `FeatureMatrix`:

```python
    def column(self, name: str) -> np.ndarray:
        """One feature's values across all rows, by name."""
        try:
            index = self.block.names.index(name)
        except ValueError:
            raise KeyError(f"no such feature: {name!r}") from None
        return self.values[:, index]

    def observed(self, name: str) -> np.ndarray:
        """Whether one feature was observed, across all rows, by name."""
        try:
            index = self.block.names.index(name)
        except ValueError:
            raise KeyError(f"no such feature: {name!r}") from None
        return self.mask[:, index]
```

Append to `tests/features/test_feature_spec.py`:

```python
def test_column_and_observed_look_up_by_name() -> None:
    matrix = FeatureMatrix(
        values=np.arange(6, dtype=np.float32).reshape(2, 3),
        mask=np.array([[True, False, True], [True, True, True]]),
        block=_block(),
    )
    assert matrix.column("age").tolist() == [1.0, 4.0]
    assert matrix.observed("age").tolist() == [False, True]
    with pytest.raises(KeyError, match="no such feature"):
        matrix.column("nope")
```

- [ ] **Step 2: Write the failing rule test**

```python
# tests/baselines/test_rules.py
"""Hand-written heuristics, and the Scorer contract they implement."""

from pathlib import Path

import numpy as np

from rga.baselines.base import Scorer, dense_matrix
from rga.baselines.rules import RuleScorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def _candidates(span: Span):
    return build_candidates(build_dataset(CONFIG), span)


def test_rule_scorer_satisfies_the_protocol() -> None:
    assert isinstance(RuleScorer(), Scorer)


def test_scores_are_bounded_and_one_per_candidate() -> None:
    candidates = _candidates(Span.EVAL)
    scorer = RuleScorer()
    scorer.fit(_candidates(Span.TRAIN))
    scores = scorer.score(candidates)

    assert scores.shape == (candidates.n_candidates,)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0
    assert np.isfinite(scores).all()


def test_rules_beat_random_ordering() -> None:
    # Not asking for excellence — only that naive conditions carry some signal.
    # If this fails, either the rules are broken or the generator is.
    candidates = _candidates(Span.EVAL)
    scorer = RuleScorer()
    scorer.fit(_candidates(Span.TRAIN))

    from rga.eval.metrics import evaluate_ranking

    metrics = evaluate_ranking(candidates.y_true(), scorer.score(candidates), ks=(50,))
    assert metrics.roc_auc > 0.55


def test_rules_do_not_solve_the_task_outright() -> None:
    # The acceptance criterion for the dataset, from section 16 of the spec: if
    # hand-written conditions rank almost perfectly, the synthetic data is too
    # easy and nothing measured on top of it means anything.
    candidates = _candidates(Span.EVAL)
    scorer = RuleScorer()
    scorer.fit(_candidates(Span.TRAIN))

    from rga.eval.metrics import evaluate_ranking

    metrics = evaluate_ranking(candidates.y_true(), scorer.score(candidates), ks=(50,))
    assert metrics.pr_auc < 0.9


def test_masked_provenance_does_not_fire_the_self_grant_rule() -> None:
    scorer = RuleScorer()
    candidates = _candidates(Span.EVAL)
    scores = scorer.score(candidates)

    unobserved = ~candidates.matrix.observed("actor_is_subject")
    if unobserved.any():
        # Rows whose initiator is unknown must not be penalised for it.
        assert np.isfinite(scores[unobserved]).all()


def test_dense_matrix_zeroes_unobserved_and_appends_group_coverage() -> None:
    candidates = _candidates(Span.EVAL)
    dense = dense_matrix(candidates.matrix)

    assert dense.shape[0] == candidates.n_candidates
    assert dense.shape[1] == candidates.matrix.n_features + 3
    assert np.isfinite(dense).all()
```

- [ ] **Step 3: Run both to verify they fail**

Run: `uv run pytest tests/baselines/test_rules.py tests/features/test_feature_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.baselines'`

- [ ] **Step 4: Implement `base.py`**

```python
# src/rga/baselines/__init__.py
"""Baselines the neural model is compared against."""
```

```python
# src/rga/baselines/base.py
"""What every scorer implements, and how a masked matrix becomes a dense one."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rga.features.spec import CandidateSet, FeatureGroup, FeatureMatrix


@runtime_checkable
class Scorer(Protocol):
    """Assigns an anomaly score to every candidate.

    Higher means more suspicious. Scorers that need no fitting implement `fit`
    as a no-op so the experiment runner can treat them all alike.
    """

    name: str

    def fit(self, train: CandidateSet) -> None:
        """Learn whatever the scorer needs from the training span."""
        ...

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """One score per candidate, higher meaning more anomalous."""
        ...


def dense_matrix(matrix: FeatureMatrix) -> np.ndarray:
    """Flatten values and mask into something a classical estimator accepts.

    Unobserved entries become zero, which on its own would be a lie — zero is a
    legitimate value for most of these features. So one column per group is
    appended holding the share of that group's features actually observed in
    that row, which lets the estimator tell a genuine zero from a gap.
    """
    values = np.where(matrix.mask, matrix.values, 0.0).astype(np.float64)

    coverage = []
    for group in FeatureGroup:
        columns = matrix.group_indices(group)
        if columns.size:
            coverage.append(matrix.mask[:, columns].mean(axis=1))
        else:
            coverage.append(np.zeros(matrix.n_rows))

    return np.hstack([values, np.column_stack(coverage)])
```

- [ ] **Step 5: Implement `rules.py`**

```python
# src/rga/baselines/rules.py
"""Hand-written heuristics.

This is the "could you not just grep the audit log" baseline. The conditions are
the obvious ones an engineer would write in an afternoon, and the weights are
picked by hand rather than fitted — fitting them would make this a model, and the
point of the comparison is that it is not one.

It doubles as the acceptance test for the generator: if these conditions rank
nearly perfectly, the synthetic data is too easy and every number measured above
it is worthless.
"""

from __future__ import annotations

import numpy as np

from rga.domain.relations import PermissionLevel
from rga.features.spec import CandidateSet

#: Condition name to weight. Weights are deliberately round numbers.
_WEIGHTS = {
    "self_grant_elevated": 3.0,
    "bypasses_bucket": 2.0,
    "big_level_jump": 2.0,
    "structurally_isolated": 2.0,
    "off_hours": 1.0,
    "weekend": 1.0,
    "new_subject": 1.0,
}

#: A grant at or above this level counts as elevated.
_ELEVATED = float(PermissionLevel.WRITE) / float(PermissionLevel.ADMIN)
#: A level increase of this much or more counts as a jump.
_BIG_JUMP = 2.0 / float(PermissionLevel.ADMIN)


class RuleScorer:
    """Weighted sum of hand-written conditions, rescaled to [0, 1]."""

    name = "rules"

    def fit(self, train: CandidateSet) -> None:
        """Nothing is learned. Present so the runner can treat scorers alike."""

    def score(self, candidates: CandidateSet) -> np.ndarray:
        matrix = candidates.matrix

        # A condition resting on an unobserved feature must not fire. Silence is
        # the honest answer when the source cannot tell us.
        self_grant = (
            matrix.column("actor_is_subject")
            * matrix.observed("actor_is_subject")
            * (matrix.column("level_ordinal") >= _ELEVATED)
        )
        bypass = matrix.column("bypasses_bucket") * matrix.observed("bypasses_bucket")

        conditions = {
            "self_grant_elevated": self_grant,
            "bypasses_bucket": bypass,
            "big_level_jump": (matrix.column("level_jump") >= _BIG_JUMP).astype(np.float64),
            "structurally_isolated": matrix.column("path_unreachable"),
            "off_hours": matrix.column("is_off_hours"),
            "weekend": matrix.column("is_weekend"),
            "new_subject": matrix.column("subj_is_new"),
        }

        total = np.zeros(matrix.n_rows, dtype=np.float64)
        for name, fired in conditions.items():
            total += _WEIGHTS[name] * np.asarray(fired, dtype=np.float64)

        return total / sum(_WEIGHTS.values())
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/baselines tests/features/test_feature_spec.py -q`
Expected: PASS

If `test_rules_do_not_solve_the_task_outright` fails, that is a finding about the
dataset, not a bug in the rules. Record it, raise `legitimate_exception_rate` and
soften the pattern parameters in `configs/generator/small.yaml`, and regenerate.

- [ ] **Step 7: Commit and push**

```bash
git add src/rga/baselines src/rga/features/spec.py tests/baselines tests/features/test_feature_spec.py
git commit -m "feat: add rule baseline and scorer contract"
git push origin main
```

---

## Task 9: Outlier baselines

`IsolationForest` and `LOF` on exactly the same features, fitted on the training
span and applied to the window. Same inputs, no graph propagation — so the gap
between these and the network in Module 3 isolates the contribution of graph
aggregation rather than of feature engineering.

**Files:**
- Create: `src/rga/baselines/outliers.py`
- Test: `tests/baselines/test_outliers.py`

**Interfaces:**
- Consumes: `Scorer`, `dense_matrix` (Task 8); `CandidateSet`
- Produces:
  - `IsolationForestScorer(seed: int = 0, n_estimators: int = 200)` with `name = "isolation_forest"`
  - `LocalOutlierFactorScorer(n_neighbors: int = 20)` with `name = "lof"`

- [ ] **Step 1: Write the failing test**

```python
# tests/baselines/test_outliers.py
"""Classical outlier detectors on the same feature space."""

from pathlib import Path

import numpy as np
import pytest

from rga.baselines.base import Scorer
from rga.baselines.outliers import IsolationForestScorer, LocalOutlierFactorScorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def _spans():
    dataset = build_dataset(CONFIG)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_scorers_satisfy_the_protocol(factory) -> None:
    assert isinstance(factory(), Scorer)


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_scores_are_finite_and_one_per_candidate(factory) -> None:
    train, evaluation = _spans()
    scorer = factory()
    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.shape == (evaluation.n_candidates,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_scoring_before_fitting_is_refused(factory) -> None:
    _, evaluation = _spans()
    with pytest.raises(RuntimeError, match="fit"):
        factory().score(evaluation)


def test_isolation_forest_is_reproducible() -> None:
    train, evaluation = _spans()
    first = IsolationForestScorer(seed=5)
    first.fit(train)
    second = IsolationForestScorer(seed=5)
    second.fit(train)
    assert np.allclose(first.score(evaluation), second.score(evaluation))


@pytest.mark.parametrize(
    "factory", [IsolationForestScorer, LocalOutlierFactorScorer], ids=["forest", "lof"]
)
def test_outlier_detectors_carry_some_signal(factory) -> None:
    train, evaluation = _spans()
    scorer = factory()
    scorer.fit(train)

    from rga.eval.metrics import evaluate_ranking

    metrics = evaluate_ranking(evaluation.y_true(), scorer.score(evaluation), ks=(50,))
    assert metrics.roc_auc > 0.5
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/baselines/test_outliers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.baselines.outliers'`

- [ ] **Step 3: Implement `outliers.py`**

```python
# src/rga/baselines/outliers.py
"""Classical outlier detection on the candidate features.

Both estimators see exactly what the network will see in Module 3, minus any
propagation across the graph. That is the point of the comparison: whatever gap
opens up is attributable to graph aggregation, not to a richer feature set.

Both are fitted on the training span and applied to the evaluation window, never
fitted on the window itself — that would be the same leak the temporal split
exists to prevent.

scikit-learn is imported lazily so the package stays importable without the `ml`
extra installed.
"""

from __future__ import annotations

import numpy as np

from rga.baselines.base import dense_matrix
from rga.features.spec import CandidateSet


class _ScaledEstimator:
    """Shared plumbing: standardise on the training span, then score."""

    name = "unset"

    def __init__(self) -> None:
        self._scaler = None
        self._estimator = None

    def _build(self):
        raise NotImplementedError

    def fit(self, train: CandidateSet) -> None:
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        features = self._scaler.fit_transform(dense_matrix(train.matrix))
        self._estimator = self._build()
        self._estimator.fit(features)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        if self._estimator is None or self._scaler is None:
            raise RuntimeError(f"{self.name} must be fit before scoring")
        features = self._scaler.transform(dense_matrix(candidates.matrix))
        # Both estimators return higher values for more normal points; the
        # project's convention is the opposite, so the sign is flipped.
        return -np.asarray(self._estimator.score_samples(features), dtype=np.float64)


class IsolationForestScorer(_ScaledEstimator):
    """Isolates points by random splits: the fewer splits needed, the odder."""

    name = "isolation_forest"

    def __init__(self, seed: int = 0, n_estimators: int = 200) -> None:
        super().__init__()
        self._seed = seed
        self._n_estimators = n_estimators

    def _build(self):
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=self._n_estimators, random_state=self._seed, n_jobs=-1
        )


class LocalOutlierFactorScorer(_ScaledEstimator):
    """Compares a point's local density with its neighbours'.

    `novelty=True` is required to score points the estimator was not fitted on,
    which is exactly our protocol: fit on the training span, score the window.
    """

    name = "lof"

    def __init__(self, n_neighbors: int = 20) -> None:
        super().__init__()
        self._n_neighbors = n_neighbors

    def _build(self):
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(n_neighbors=self._n_neighbors, novelty=True, n_jobs=-1)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/baselines -q`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/baselines/outliers.py tests/baselines/test_outliers.py
git commit -m "feat: add isolation forest and lof baselines"
git push origin main
```

---

## Task 10: Experiment runner

Everything above, run over several seeds and written down in a form that goes
straight into the thesis. Section 9 requires five fixed seeds per configuration
with mean and standard deviation reported — a single run of a stochastic
estimator on synthetic data is an anecdote, not a result.

**Files:**
- Create: `src/rga/eval/experiment.py`, `configs/experiments/baselines.yaml`
- Modify: `src/rga/cli/main.py` — add the `evaluate` subcommand
- Test: `tests/eval/test_experiment.py`, add one case to `tests/cli/test_main.py`

**Interfaces:**
- Consumes: `build_candidates`, `Span`, `evaluate_ranking`, `recall_by_pattern`, all scorers
- Produces:
  - `ExperimentConfig(name, dataset, seeds, scorers, ks, feature_groups)`
  - `load_experiment_config(path: Path) -> ExperimentConfig`
  - `build_scorer(name: str, seed: int) -> Scorer`
  - `ExperimentResult(config, rows, per_pattern)` with `.aggregate() -> dict[str, dict[str, tuple[float, float]]]`
  - `run_experiment(config: ExperimentConfig) -> ExperimentResult`
  - `format_results_table(result: ExperimentResult) -> str`
  - `save_results(path: Path, result: ExperimentResult) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_experiment.py
"""Running baselines over seeds and writing the result down."""

import json
from pathlib import Path

import pytest

from rga.eval.experiment import (
    ExperimentConfig,
    build_scorer,
    format_results_table,
    load_experiment_config,
    run_experiment,
    save_results,
)


def _small(seeds=(1, 2)) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        dataset=Path("configs/generator/small.yaml"),
        seeds=seeds,
        scorers=("rules", "isolation_forest"),
        ks=(20, 50),
        feature_groups=None,
    )


def test_shipped_config_loads() -> None:
    config = load_experiment_config(Path("configs/experiments/baselines.yaml"))
    assert config.seeds
    assert config.scorers
    assert config.dataset.exists()


def test_unknown_scorer_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="unknown scorer"):
        build_scorer("magic", seed=0)


def test_every_declared_scorer_can_be_built() -> None:
    for name in ("rules", "isolation_forest", "lof"):
        assert build_scorer(name, seed=0).name == name


def test_run_produces_one_row_per_scorer_and_seed() -> None:
    result = run_experiment(_small())
    assert len(result.rows) == 2 * 2
    assert {row["scorer"] for row in result.rows} == {"rules", "isolation_forest"}
    assert {row["seed"] for row in result.rows} == {1, 2}


def test_rows_carry_the_metrics() -> None:
    result = run_experiment(_small(seeds=(1,)))
    row = result.rows[0]
    assert "pr_auc" in row
    assert "precision_at_20" in row
    assert "lift_at_50" in row


def test_aggregate_reports_mean_and_deviation() -> None:
    aggregate = run_experiment(_small()).aggregate()
    mean, deviation = aggregate["rules"]["pr_auc"]
    assert 0.0 <= mean <= 1.0
    assert deviation >= 0.0


def test_per_pattern_recall_is_collected() -> None:
    result = run_experiment(_small(seeds=(1,)))
    assert result.per_pattern
    for scorer_name, by_pattern in result.per_pattern.items():
        assert scorer_name in {"rules", "isolation_forest"}
        assert all(0.0 <= value <= 1.0 for value in by_pattern.values())


def test_table_mentions_every_scorer() -> None:
    table = format_results_table(run_experiment(_small(seeds=(1,))))
    assert "rules" in table
    assert "isolation_forest" in table
    assert "pr_auc" in table


def test_results_are_saved_as_json_and_markdown(tmp_path: Path) -> None:
    result = run_experiment(_small(seeds=(1,)))
    save_results(tmp_path / "run", result)

    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [
        "results.json",
        "results.md",
    ]
    payload = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    assert payload["config"]["name"] == "test"
    assert payload["rows"]
```

Append to `tests/cli/test_main.py`:

```python
def test_evaluate_runs_and_writes_results(tmp_path: Path, capsys) -> None:
    # The smoke config, not the real one: a five-seed three-scorer comparison
    # does not belong in a unit test run on every commit.
    out = tmp_path / "run"
    code = main(
        ["evaluate", "--config", "configs/experiments/smoke.yaml", "--out", str(out)]
    )
    assert code == 0
    assert (out / "results.md").exists()
    assert "pr_auc" in capsys.readouterr().out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/eval/test_experiment.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.eval.experiment'`

- [ ] **Step 3: Write the shipped experiment config**

```yaml
# configs/experiments/baselines.yaml
# The comparison that has to exist before any neural result is believable.
name: baselines
dataset: configs/generator/small.yaml
seeds: [1, 2, 3, 4, 5]
scorers:
  - rules
  - isolation_forest
  - lof
ks: [20, 50, 100]
```

Also write the light configuration the command-line test uses:

```yaml
# configs/experiments/smoke.yaml
# One seed, one scorer. Exists so the command line can be tested on every commit
# without running the real comparison.
name: smoke
dataset: configs/generator/small.yaml
seeds: [1]
scorers:
  - rules
ks: [20]
```

- [ ] **Step 4: Implement `experiment.py`**

```python
# src/rga/eval/experiment.py
"""Running scorers over several seeds and writing the result down.

Section 9 of the spec asks for five fixed seeds per configuration with mean and
standard deviation. A single run of a stochastic estimator on synthetic data is
an anecdote: the seed moves both the dataset and the estimator, and the spread
between seeds is itself a result worth reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from rga.baselines.base import Scorer
from rga.baselines.outliers import IsolationForestScorer, LocalOutlierFactorScorer
from rga.baselines.rules import RuleScorer
from rga.eval.metrics import evaluate_ranking, recall_by_pattern
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

#: Queue size the per-pattern breakdown is measured at.
_PATTERN_K = 50


@dataclass(frozen=True)
class ExperimentConfig:
    """One comparison to run."""

    name: str
    dataset: Path
    seeds: tuple[int, ...]
    scorers: tuple[str, ...]
    ks: tuple[int, ...]
    #: Feature-group subsets for the ablation study; None means the full set.
    feature_groups: tuple[tuple[FeatureGroup, ...], ...] | None


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment recipe from YAML."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_groups = document.get("feature_groups")
    groups = (
        tuple(tuple(FeatureGroup(name) for name in variant) for variant in raw_groups)
        if raw_groups
        else None
    )
    return ExperimentConfig(
        name=str(document["name"]),
        dataset=Path(document["dataset"]),
        seeds=tuple(int(seed) for seed in document["seeds"]),
        scorers=tuple(str(name) for name in document["scorers"]),
        ks=tuple(int(k) for k in document["ks"]),
        feature_groups=groups,
    )


def build_scorer(name: str, seed: int) -> Scorer:
    """Construct a scorer by the name used in configuration files."""
    if name == "rules":
        return RuleScorer()
    if name == "isolation_forest":
        return IsolationForestScorer(seed=seed)
    if name == "lof":
        return LocalOutlierFactorScorer()
    raise KeyError(f"unknown scorer: {name!r}")


@dataclass(frozen=True)
class ExperimentResult:
    """Every measurement from one experiment."""

    config: ExperimentConfig
    rows: tuple[dict[str, object], ...]
    #: Scorer name to pattern name to recall at the reporting queue size.
    per_pattern: dict[str, dict[str, float]]

    def aggregate(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Mean and standard deviation of every metric, per scorer."""
        summary: dict[str, dict[str, tuple[float, float]]] = {}
        for scorer in {str(row["scorer"]) for row in self.rows}:
            own = [row for row in self.rows if row["scorer"] == scorer]
            metrics: dict[str, tuple[float, float]] = {}
            for key, value in own[0].items():
                if not isinstance(value, float):
                    continue
                series = np.array([float(row[key]) for row in own], dtype=np.float64)
                metrics[key] = (float(np.nanmean(series)), float(np.nanstd(series)))
            summary[scorer] = metrics
        return summary


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run every scorer on every seed and collect the numbers."""
    base = load_dataset_config(config.dataset)
    rows: list[dict[str, object]] = []
    pattern_totals: dict[str, dict[str, list[float]]] = {}

    for seed in config.seeds:
        dataset = build_dataset(replace(base, seed=seed))
        train = build_candidates(dataset, Span.TRAIN)
        evaluation = build_candidates(dataset, Span.EVAL)

        for name in config.scorers:
            scorer = build_scorer(name, seed=seed)
            scorer.fit(train)
            scores = scorer.score(evaluation)

            metrics = evaluate_ranking(evaluation.y_true(), scores, ks=config.ks)
            rows.append({"scorer": name, "seed": seed, **metrics.as_row()})

            found = recall_by_pattern(
                evaluation.y_true(), scores, evaluation.patterns, k=_PATTERN_K
            )
            bucket = pattern_totals.setdefault(name, {})
            for pattern, recall in found.items():
                bucket.setdefault(pattern, []).append(recall)

    per_pattern = {
        name: {pattern: float(np.mean(values)) for pattern, values in patterns.items()}
        for name, patterns in pattern_totals.items()
    }
    return ExperimentResult(config=config, rows=tuple(rows), per_pattern=per_pattern)


def format_results_table(result: ExperimentResult) -> str:
    """Render the aggregate as a Markdown table, ready to paste into the thesis."""
    aggregate = result.aggregate()
    headline = ("pr_auc", "roc_auc") + tuple(f"precision_at_{k}" for k in result.config.ks)

    lines = [f"### {result.config.name} — {len(result.config.seeds)} seeds", ""]
    lines.append("| scorer | " + " | ".join(headline) + " |")
    lines.append("|---" * (len(headline) + 1) + "|")
    for scorer in sorted(aggregate):
        cells = []
        for metric in headline:
            mean, deviation = aggregate[scorer].get(metric, (float("nan"), float("nan")))
            cells.append(f"{mean:.3f} ± {deviation:.3f}")
        lines.append(f"| {scorer} | " + " | ".join(cells) + " |")

    if result.per_pattern:
        patterns = sorted({p for byname in result.per_pattern.values() for p in byname})
        lines += ["", f"### Recall at {_PATTERN_K}, by pattern", ""]
        lines.append("| scorer | " + " | ".join(patterns) + " |")
        lines.append("|---" * (len(patterns) + 1) + "|")
        for scorer in sorted(result.per_pattern):
            cells = [f"{result.per_pattern[scorer].get(p, 0.0):.2f}" for p in patterns]
            lines.append(f"| {scorer} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def save_results(path: Path, result: ExperimentResult) -> None:
    """Write the raw rows and the rendered table side by side."""
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "name": result.config.name,
            "dataset": str(result.config.dataset),
            "seeds": list(result.config.seeds),
            "scorers": list(result.config.scorers),
            "ks": list(result.config.ks),
        },
        "rows": [dict(row) for row in result.rows],
        "per_pattern": result.per_pattern,
        "aggregate": {
            scorer: {metric: list(pair) for metric, pair in metrics.items()}
            for scorer, metrics in result.aggregate().items()
        },
    }
    (path / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (path / "results.md").write_text(
        format_results_table(result) + "\n", encoding="utf-8"
    )
```

- [ ] **Step 5: Add the `evaluate` subcommand**

In `src/rga/cli/main.py`, inside `_parser()` after the `stats` parser:

```python
    evaluate = commands.add_parser("evaluate", help="run baselines and write results")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
```

and inside `main()` before the final `return 2`:

```python
    if arguments.command == "evaluate":
        from rga.eval.experiment import (
            format_results_table,
            load_experiment_config,
            run_experiment,
            save_results,
        )

        result = run_experiment(load_experiment_config(arguments.config))
        save_results(arguments.out, result)
        print(format_results_table(result))
        return 0
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/eval tests/cli -q`
Expected: PASS

- [ ] **Step 7: Run the real comparison and read it**

Run: `uv run rga evaluate --config configs/experiments/baselines.yaml --out experiments/runs/baselines`

Read the table before continuing. What must hold for the dataset to be usable:
the rules carry signal but do not dominate, and the outlier detectors are somewhere
above them without being near perfect. If any scorer reaches `pr_auc` above 0.9,
the dataset is too easy — record it as a finding, soften the generator, regenerate.
If every scorer sits near the base rate, it is too hard in the opposite direction.

- [ ] **Step 8: Commit and push**

```bash
git add src/rga/eval/experiment.py src/rga/cli/main.py configs/experiments tests/eval tests/cli
git commit -m "feat: add baseline experiment runner"
git push origin main
```

---

## Task 11: Feature group ablation

The same machinery, run on restricted feature sets. This answers two questions
with one experiment, which is why it is worth doing properly: which feature groups
carry the signal, and — from section 12 of the spec — what an integrator whose
authorization engine records no timestamps or no initiator actually gives up.

**Files:**
- Create: `configs/experiments/ablation.yaml`, `docs/thesis/README.md`
- Modify: `src/rga/eval/experiment.py` — honour `feature_groups`
- Test: `tests/eval/test_ablation.py`

**Interfaces:**
- Consumes: everything from Task 10
- Produces:
  - `restrict_candidates(candidates: CandidateSet, groups: tuple[FeatureGroup, ...]) -> CandidateSet`
  - rows gain a `"groups"` field naming the variant
  - `format_ablation_table(result: ExperimentResult) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_ablation.py
"""Restricting the feature set, for the ablation study and the integration guide."""

from pathlib import Path

import pytest

from rga.eval.experiment import (
    ExperimentConfig,
    format_ablation_table,
    restrict_candidates,
    run_experiment,
)
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))

STRUCTURAL_ONLY = (FeatureGroup.STRUCTURAL,)
WITH_TIME = (FeatureGroup.STRUCTURAL, FeatureGroup.TEMPORAL)
EVERYTHING = (FeatureGroup.STRUCTURAL, FeatureGroup.TEMPORAL, FeatureGroup.PROVENANCE)


def _candidates():
    return build_candidates(build_dataset(CONFIG), Span.EVAL)


def test_restriction_narrows_the_matrix_but_keeps_the_rows() -> None:
    candidates = _candidates()
    reduced = restrict_candidates(candidates, STRUCTURAL_ONLY)

    assert reduced.n_candidates == candidates.n_candidates
    assert reduced.matrix.n_features < candidates.matrix.n_features
    assert set(reduced.matrix.block.groups) == {FeatureGroup.STRUCTURAL}


def test_restriction_preserves_labels_and_keys() -> None:
    candidates = _candidates()
    reduced = restrict_candidates(candidates, WITH_TIME)

    assert reduced.keys == candidates.keys
    assert reduced.patterns == candidates.patterns
    assert (reduced.labels == candidates.labels).all()


def test_capability_levels_are_nested() -> None:
    # Level 0 sees structural only, level 1 adds time, level 2 adds provenance.
    candidates = _candidates()
    widths = [
        restrict_candidates(candidates, groups).matrix.n_features
        for groups in (STRUCTURAL_ONLY, WITH_TIME, EVERYTHING)
    ]
    assert widths[0] < widths[1] < widths[2]


def test_experiment_runs_every_variant() -> None:
    config = ExperimentConfig(
        name="ablation-test",
        dataset=Path("configs/generator/small.yaml"),
        seeds=(1,),
        scorers=("isolation_forest",),
        ks=(50,),
        feature_groups=(STRUCTURAL_ONLY, EVERYTHING),
    )
    result = run_experiment(config)

    assert len(result.rows) == 2
    assert {row["groups"] for row in result.rows} == {"structural", "structural+temporal+provenance"}


def test_ablation_table_names_the_variants() -> None:
    config = ExperimentConfig(
        name="ablation-test",
        dataset=Path("configs/generator/small.yaml"),
        seeds=(1,),
        scorers=("isolation_forest",),
        ks=(50,),
        feature_groups=(STRUCTURAL_ONLY, EVERYTHING),
    )
    table = format_ablation_table(run_experiment(config))
    assert "structural" in table
    assert "pr_auc" in table


def test_full_run_without_variants_labels_rows_as_full() -> None:
    config = ExperimentConfig(
        name="plain",
        dataset=Path("configs/generator/small.yaml"),
        seeds=(1,),
        scorers=("rules",),
        ks=(50,),
        feature_groups=None,
    )
    assert {row["groups"] for row in run_experiment(config).rows} == {"all"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/eval/test_ablation.py -q`
Expected: FAIL — `ImportError: cannot import name 'restrict_candidates'`

- [ ] **Step 3: Extend `experiment.py`**

Widen the existing `rga.features.spec` import — a second import line from the same
module would be rejected by the linter:

```python
from rga.features.spec import CandidateSet, FeatureGroup
```

Then add near the top, after the imports:

```python
def restrict_candidates(
    candidates: CandidateSet, groups: tuple[FeatureGroup, ...]
) -> CandidateSet:
    """The same candidates seen through a narrower feature set.

    This is how a weaker authorization engine is simulated: level 0 keeps the
    structural group only, level 1 adds temporal, level 2 adds provenance.
    """
    return CandidateSet(
        keys=candidates.keys,
        ts=candidates.ts,
        matrix=candidates.matrix.with_groups(groups),
        labels=candidates.labels,
        patterns=candidates.patterns,
    )


def _variant_name(groups: tuple[FeatureGroup, ...] | None) -> str:
    return "all" if groups is None else "+".join(str(group) for group in groups)
```

Replace the scorer loop inside `run_experiment` so each variant is run in turn:

```python
        variants = config.feature_groups or (None,)
        for groups in variants:
            train_view = train if groups is None else restrict_candidates(train, groups)
            eval_view = (
                evaluation if groups is None else restrict_candidates(evaluation, groups)
            )
            for name in config.scorers:
                scorer = build_scorer(name, seed=seed)
                scorer.fit(train_view)
                scores = scorer.score(eval_view)

                metrics = evaluate_ranking(eval_view.y_true(), scores, ks=config.ks)
                rows.append(
                    {
                        "scorer": name,
                        "seed": seed,
                        "groups": _variant_name(groups),
                        **metrics.as_row(),
                    }
                )

                found = recall_by_pattern(
                    eval_view.y_true(), scores, eval_view.patterns, k=_PATTERN_K
                )
                bucket = pattern_totals.setdefault(name, {})
                for pattern, recall in found.items():
                    bucket.setdefault(pattern, []).append(recall)
```

Add the ablation table renderer at the end of the module:

```python
def format_ablation_table(result: ExperimentResult) -> str:
    """Quality per feature-group variant.

    Doubles as the integration guide from section 12: each row is what an engine
    at that capability level can deliver.
    """
    variants = sorted({str(row["groups"]) for row in result.rows})
    scorers = sorted({str(row["scorer"]) for row in result.rows})

    lines = [f"### {result.config.name} — feature groups", ""]
    lines.append("| feature groups | " + " | ".join(scorers) + " |")
    lines.append("|---" * (len(scorers) + 1) + "|")
    for variant in variants:
        cells = []
        for scorer in scorers:
            series = [
                float(row["pr_auc"])
                for row in result.rows
                if row["groups"] == variant and row["scorer"] == scorer
            ]
            cells.append(
                f"{np.nanmean(series):.3f} ± {np.nanstd(series):.3f}" if series else "—"
            )
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")

    lines += ["", "pr_auc, mean ± standard deviation over seeds."]
    return "\n".join(lines)
```

Include the ablation table in `save_results` by appending it to the Markdown when
any row carries a variant other than `all`:

```python
    table = format_results_table(result)
    if {str(row["groups"]) for row in result.rows} != {"all"}:
        table += "\n\n" + format_ablation_table(result)
    (path / "results.md").write_text(table + "\n", encoding="utf-8")
```

- [ ] **Step 4: Write the ablation config**

```yaml
# configs/experiments/ablation.yaml
# Which feature groups carry the signal — and, read the other way, what an
# authorization engine at each capability level can deliver. See section 12 of
# the design document.
name: ablation
dataset: configs/generator/small.yaml
seeds: [1, 2, 3, 4, 5]
scorers:
  - rules
  - isolation_forest
  - lof
ks: [20, 50, 100]
feature_groups:
  - [structural]                             # level 0: a bare snapshot of tuples
  - [structural, temporal]                    # level 1: plus creation timestamps
  - [structural, temporal, provenance]        # level 2: plus the initiator
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/eval -q`
Expected: PASS

- [ ] **Step 6: Run both experiments and collect the artefacts**

Run:
```bash
uv run rga evaluate --config configs/experiments/baselines.yaml --out experiments/runs/baselines
uv run rga evaluate --config configs/experiments/ablation.yaml --out experiments/runs/ablation
mkdir -p docs/thesis
cp experiments/runs/baselines/results.md docs/thesis/baselines.md
cp experiments/runs/ablation/results.md docs/thesis/ablation.md
```

Write `docs/thesis/README.md` describing what each table is and how to regenerate
it, so the numbers in the thesis can always be traced back to a command:

```markdown
# Материалы для ВКР

Таблицы генерируются из кода и воспроизводятся одной командой. Ничего здесь не
правится руками: если число в работе расходится с числом здесь, верным считается
то, что выдала команда.

| Файл | Что это | Как получить |
|---|---|---|
| `baselines.md` | Сравнение бейзлайнов, пять сидов, среднее и отклонение | `uv run rga evaluate --config configs/experiments/baselines.yaml --out experiments/runs/baselines` |
| `ablation.md` | Вклад групп признаков; он же руководство по интеграции из раздела 12 | `uv run rga evaluate --config configs/experiments/ablation.yaml --out experiments/runs/ablation` |

Таблица вклада групп читается в две стороны. Слева направо — какие признаки
несут сигнал. Сверху вниз — что получит интегратор, чья система авторизации
отдаёт только снимок кортежей, снимок с метками времени, или ещё и инициатора
изменения.
```

- [ ] **Step 7: Commit and push**

```bash
git add src/rga/eval/experiment.py configs/experiments tests/eval/test_ablation.py docs/thesis
git commit -m "feat: add feature group ablation study"
git push origin main
```

---

## Deliberately out of scope

Section 9 of the spec lists experiment series this module does not run: quality
against anomaly density, against graph size, on the held-out patterns, and under a
contaminated training span. All four compare a model that learns against one that
does not, and there is nothing here that learns. They belong to Module 3 and reuse
this module's runner unchanged — a new scorer name is the whole of the change.

The supervised baseline from section 8 is likewise Module 3's: it is the same
encoder as the neural model, trained differently, and it cannot exist before the
encoder does.

## Module Completion

Module 2 is done when all of the following hold:

- `uv run ruff check .` is clean and `uv run pytest -m "not integration and not gpu"` is green.
- `uv run rga evaluate --config configs/experiments/baselines.yaml --out experiments/runs/baselines` produces a table where the rule baseline carries signal without dominating, and no scorer reaches `pr_auc` above 0.9.
- The ablation table shows quality rising from structural-only through to the full feature set. If it does not — if timestamps and provenance add nothing — that is a finding to record and investigate, not a result to paste into the thesis unexamined.
- `docs/thesis/baselines.md` and `docs/thesis/ablation.md` exist and are reproducible from the documented commands.

At that point there is a defensible result with measured quality and no neural
network in sight. Module 3 replaces the scorer and reuses everything else: the
candidate builder, the metrics harness and the experiment runner are the contract
the network plugs into.
