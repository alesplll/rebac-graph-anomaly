# Module 3: Neural Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the self-supervised relational graph network that scores permission
changes, wire it into the existing experiment stand as one more scorer, and produce
the tables that compare it with the baselines — including the contrast on the three
anomaly patterns hidden from supervised training.

**Architecture:** The encoder is a stack of hand-written relational message-passing
layers over the access graph at the temporal split. Two heads sit on top: one scores
the likelihood of an edge, one reconstructs a node's structural profile. Training is
self-supervised — real edges against corrupted ones — and model selection uses
held-out normal edges only, never anomaly labels. The finished model enters the stand
through `build_scorer("gnn", seed)`; the candidate builder, the metrics and the
experiment runner keep their current shapes.

**Tech Stack:** Python 3.12 under `uv`, PyTorch >= 2.7 as a tensor library with
autograd, NumPy, pytest. No PyTorch Geometric, no DGL, no `torch-scatter`.

**Spec:** `docs/superpowers/specs/2026-09-12-rebac-graph-anomaly-design.md` —
sections 3 (learning task), 5.3 (pattern catalogue), 6 (features), 7 and 7.4 (model
and the decisions taken before this module), 8 (baselines), 9 (evaluation protocol).

## Global Constraints

Every task's requirements implicitly include this section.

- Python is pinned to `3.12`; the laptop installs `uv sync --extra cpu`, the training
  PC installs `uv sync --extra gpu`. The two extras conflict by declaration and are
  never installed together.
- `torch>=2.7` from the `cu128` index on the PC. The card is Blackwell, `sm_120`:
  builds for CUDA 12.1 and older do not start on it.
- **PyTorch Geometric and DGL are never added.** All graph layers, aggregations and
  scoring are written here. Sparse aggregation goes through `Tensor.index_add_`.
- Every random draw goes through an explicitly seeded `numpy.random.Generator` or
  `torch.Generator`. No calls to the `random` module, no unseeded defaults, no
  reliance on the global torch RNG outside `seed_torch`.
- The message-passing layer reads `edge_src`, `edge_dst`, `edge_rel` and `edge_level`
  from the graph and nothing else. `edge_created` and `edge_actor` stay out of it, so
  that the `structural` row of the feature-group table keeps meaning what it says.
- Paths through `pathlib`. No symlinks. Newlines in data files forced to `\n`. Every
  entry point guarded by `if __name__ == "__main__":` — Windows spawns processes.
- Comments, docstrings and commit messages in English. Commit messages short and
  conventional. **No AI attribution lines in commits.** After every commit,
  `git push origin main`; this repository has no branches.
- `uv run ruff check .` must pass: line length 100, rules `E,F,I,N,UP,B,SIM,RUF`.
- `uv run pytest -m "not integration and not gpu"` must stay green after every task.
- Tests that need a CUDA device carry `@pytest.mark.gpu` and are expected to be
  skipped on the laptop.
- The system supports a decision, it does not pronounce a verdict. Names, docstrings
  and any user-facing wording say "suspicious", "unusual", "worth review" — never
  "compromised", "attack" or "malicious".

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/rga/nn/__init__.py` | Package marker, no logic |
| `src/rga/nn/runtime.py` | Device selection and explicit seeding |
| `src/rga/nn/graph_tensors.py` | `AccessGraph` → per-slot index tensors, structure only |
| `src/rga/nn/node_inputs.py` | Structural input vector per node |
| `src/rga/nn/layers.py` | One relational message-passing layer |
| `src/rga/nn/encoder.py` | Stack of layers producing node representations |
| `src/rga/nn/heads.py` | Edge likelihood head, node reconstruction head |
| `src/rga/nn/negatives.py` | Corrupted edges for the contrastive objective |
| `src/rga/nn/train.py` | Training loop, label-free early stopping |
| `src/rga/nn/scoring.py` | Rank transform and the three-term anomaly score |
| `src/rga/nn/scorer.py` | `GnnScorer`, the `Scorer` implementation |
| `src/rga/nn/supervised.py` | The supervised contrast baseline |
| `src/rga/nn/config.py` | Hyperparameters as a frozen dataclass |
| `scripts/gradient_check.py` | Finite-difference check as a reportable artefact |
| `configs/generator/small-history.yaml` | Small dataset whose training span carries labelled incidents |
| `configs/generator/default-history.yaml` | Full-size version of the same |
| `configs/experiments/gnn.yaml` | Fast comparison on the small dataset, CPU |
| `configs/experiments/gnn-full.yaml` | Final comparison, five seeds, GPU |
| `tests/nn/*` | One test module per source module above |

**Modified:**

| File | Change |
|---|---|
| `src/rga/features/spec.py` | `CandidateSet` gains the `graph` field |
| `src/rga/features/build.py` | Keeps the graph it already replays and hands it over |
| `src/rga/eval/experiment.py` | `restrict_candidates` carries the graph; `build_scorer` learns two names |
| `src/rga/generator/config.py` | `AnomalyConfig` gains `train_patterns` and `train_rate` |
| `src/rga/generator/dataset.py` | Labelled injection into the training span, `Dataset.train_labels` |
| `scripts/thesis_tables.py` | Hidden-pattern contrast table |

---

### Task 1: The candidate set carries the graph

The network needs the access graph; the `Scorer` protocol hands it only a
`CandidateSet`. The graph costs nothing to supply: `build_candidates` already
replays the journal to the split to compute static node attributes, and then throws
the result away. Keep it instead.

**Files:**
- Modify: `src/rga/features/spec.py` (the `CandidateSet` dataclass)
- Modify: `src/rga/features/build.py:82-84` and the `CandidateSet(...)` call at the end
- Modify: `src/rga/eval/experiment.py` (`restrict_candidates`)
- Test: `tests/features/test_candidate_graph.py`

**Interfaces:**
- Consumes: `replay`, `AccessGraph`, `build_candidates`, `restrict_candidates` — all existing.
- Produces: `CandidateSet.graph: AccessGraph | None` — the graph as it stood at
  `dataset.split_ts`, identical object for both spans of one dataset. Every later
  task reads the graph through this field.

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_candidate_graph.py
"""The candidate set carries the graph the network propagates over."""

from pathlib import Path

from rga.eval.experiment import restrict_candidates
from rga.features.build import Span, build_candidates
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def test_both_spans_carry_a_graph_of_the_same_shape() -> None:
    dataset = build_dataset(CONFIG)

    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)

    assert train.graph is not None
    assert evaluation.graph is not None
    assert train.graph.num_edges == evaluation.graph.num_edges
    assert train.graph.num_nodes == evaluation.graph.num_nodes


def test_the_graph_holds_nothing_from_the_evaluation_window() -> None:
    dataset = build_dataset(CONFIG)

    graph = build_candidates(dataset, Span.EVAL).graph

    assert graph is not None
    assert int(graph.edge_created.max()) <= dataset.split_ts


def test_restricting_feature_groups_keeps_the_graph() -> None:
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)

    narrowed = restrict_candidates(train, (FeatureGroup.STRUCTURAL,))

    assert narrowed.graph is train.graph
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/features/test_candidate_graph.py -v`

Expected: FAIL with `AttributeError: 'CandidateSet' object has no attribute 'graph'`.
If it fails for any other reason, fix that first.

- [ ] **Step 3: Add the field**

In `src/rga/features/spec.py`, add the import and the field. The field carries a
default so that hand-built fixtures in existing tests keep working.

```python
from rga.domain.graph import AccessGraph
```

```python
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
    #: The graph as it stood at the temporal split, shared by both spans. The
    #: network propagates over it; the classical scorers never read it. None only
    #: in hand-built fixtures that have no graph to speak of.
    graph: AccessGraph | None = None
```

- [ ] **Step 4: Keep the graph the builder already makes**

In `src/rga/features/build.py`, replace the two lines that build the static
attributes:

```python
    graph = replay(dataset.events, until=dataset.split_ts)
    static = compute_static_attributes(graph, np.random.default_rng(static_seed))
```

and pass it through at the end of the same function:

```python
    return CandidateSet(
        keys=tuple(keys),
        ts=np.array(stamps, dtype=np.int64),
        matrix=matrix,
        labels=np.array(labels, dtype=bool),
        patterns=tuple(patterns),
        graph=graph,
    )
```

In `src/rga/eval/experiment.py`, `restrict_candidates` must carry the field:

```python
    return CandidateSet(
        keys=candidates.keys,
        ts=candidates.ts,
        matrix=candidates.matrix.with_groups(groups),
        labels=candidates.labels,
        patterns=candidates.patterns,
        graph=candidates.graph,
    )
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/features -q && uv run pytest -m "not integration and not gpu" -q`

Expected: everything green. The second command guards against a fixture elsewhere
that builds a `CandidateSet` positionally.

- [ ] **Step 6: Commit**

```bash
git add src/rga/features/spec.py src/rga/features/build.py src/rga/eval/experiment.py tests/features/test_candidate_graph.py
git commit -m "feat: carry the split graph on the candidate set"
git push origin main
```

---

### Task 2: Labelled incidents in the training span

The supervised contrast baseline of spec section 8 needs labelled examples of
patterns 1-5 to learn from. The generator currently plants **all** labelled
anomalies in the evaluation window: a training span built from
`configs/generator/small.yaml` contains 4136 candidates and zero positives. The
existing `train_contamination` knob does plant anomalies in the training span but
deliberately drops their labels, because it exists to dirty the training graph for
the robustness experiment, not to teach anything.

So add a second knob. `train_patterns` names the patterns planted, labelled, before
the split, and `train_rate` says how densely. The patterns left out of that list are
exactly the ones the supervised model has never seen — that omission is what the
hidden-pattern contrast measures, so it is configuration, not an accident.

The labels live in a new `Dataset.train_labels` field, kept apart from `labels` so
that no existing consumer starts counting training incidents as window ground truth.

`configs/generator/small.yaml` and `default.yaml` are **not** touched: the published
baseline numbers in `docs/thesis/` must stay reproducible. The new datasets are new
files.

**Files:**
- Modify: `src/rga/generator/config.py` (`AnomalyConfig`, `dataset_config_from_document`)
- Modify: `src/rga/generator/dataset.py` (`Dataset`, `build_dataset`, `_inject`)
- Modify: `src/rga/features/build.py` (label lookup)
- Create: `configs/generator/small-history.yaml`
- Create: `configs/generator/default-history.yaml`
- Test: `tests/generator/test_train_labels.py`

**Interfaces:**
- Consumes: `AnomalyConfig`, `Dataset`, `_inject`, `build_candidates` — all existing.
- Produces:
  - `AnomalyConfig.train_patterns: tuple[str, ...]`, `AnomalyConfig.train_rate: float`
  - `Dataset.train_labels: tuple[AnomalyLabel, ...]`
  - `build_candidates(dataset, Span.TRAIN)` returns positives in `labels` and pattern
    names in `patterns` for those incidents.

- [ ] **Step 1: Write the failing test**

```python
# tests/generator/test_train_labels.py
"""Labelled incidents in the training span, for the supervised baseline."""

from pathlib import Path

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

HISTORY = load_dataset_config(Path("configs/generator/small-history.yaml"))
PLAIN = load_dataset_config(Path("configs/generator/small.yaml"))

#: The five patterns the supervised baseline is allowed to learn.
KNOWN = {
    "self_grant_admin",
    "privileged_group_join",
    "grant_burst",
    "hierarchy_bypass",
    "cross_department",
}


def test_plain_config_keeps_an_unlabelled_training_span() -> None:
    dataset = build_dataset(PLAIN)

    assert dataset.train_labels == ()
    assert int(build_candidates(dataset, Span.TRAIN).labels.sum()) == 0


def test_history_config_labels_incidents_before_the_split() -> None:
    dataset = build_dataset(HISTORY)

    assert dataset.train_labels != ()
    assert all(label.ts < dataset.split_ts for label in dataset.train_labels)


def test_only_the_configured_patterns_appear_before_the_split() -> None:
    dataset = build_dataset(HISTORY)

    planted = {label.pattern for label in dataset.train_labels}

    assert planted <= KNOWN


def test_training_candidates_carry_those_labels() -> None:
    dataset = build_dataset(HISTORY)

    train = build_candidates(dataset, Span.TRAIN)

    assert int(train.labels.sum()) > 0
    assert {pattern for pattern in train.patterns if pattern} <= KNOWN


def test_the_evaluation_window_still_carries_every_pattern() -> None:
    dataset = build_dataset(HISTORY)

    window = {label.pattern for label in dataset.labels}

    assert "shadow_group" in window
    assert "delegation_cascade" in window
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/generator/test_train_labels.py -v`

Expected: FAIL at import/collection time with
`FileNotFoundError: configs/generator/small-history.yaml`. That is the right first
failure: the config does not exist yet.

- [ ] **Step 3: Add the configuration knobs**

In `src/rga/generator/config.py`, extend `AnomalyConfig`:

```python
@dataclass(frozen=True)
class AnomalyConfig:
    """Which patterns to inject and how densely."""

    patterns: tuple[str, ...]
    #: Target share of edges in the evaluation window that are anomalous.
    rate: float
    #: Share of anomalous edges planted in the training span, to test robustness
    #: to a training graph that is not perfectly clean.
    train_contamination: float
    #: Patterns planted, labelled, before the split so the supervised baseline has
    #: something to learn from. Patterns absent from this list are the ones it has
    #: never seen, which is what the hidden-pattern contrast measures.
    train_patterns: tuple[str, ...] = ()
    #: Target share of training-span grants carrying such a labelled incident.
    train_rate: float = 0.0
```

and in `dataset_config_from_document`, parse them with defaults so that every
existing config file keeps loading unchanged:

```python
        anomalies=AnomalyConfig(
            patterns=tuple(str(name) for name in anomalies["patterns"]),  # type: ignore[index]
            rate=_rate(anomalies["rate"], "anomalies.rate"),  # type: ignore[index]
            train_contamination=_rate(
                anomalies["train_contamination"],  # type: ignore[index]
                "anomalies.train_contamination",
            ),
            train_patterns=tuple(
                str(name)
                for name in anomalies.get("train_patterns", ())  # type: ignore[union-attr]
            ),
            train_rate=_rate(
                anomalies.get("train_rate", 0.0),  # type: ignore[union-attr]
                "anomalies.train_rate",
            ),
        ),
```

- [ ] **Step 4: Plant and keep the labels**

In `src/rga/generator/dataset.py`, give `_inject` an explicit pattern list instead of
reading the config field, so the same helper serves both spans. Change its signature
and the two lines that use the field:

```python
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
    patterns: Sequence[str],
) -> tuple[list[GraphEvent], list[AnomalyLabel]]:
    """Plant anomalies until `wanted` labelled edges exist or attempts run out."""
    if not patterns:
        return [], []

    graph = replay(journal, until=cutoff)
    events: list[GraphEvent] = []
    labels: list[AnomalyLabel] = []
    taken = set(forbidden)
    exhausted: set[str] = set()
    names = list(patterns)
```

Add `from collections.abc import Sequence` to the imports. Pass
`patterns=config.anomalies.patterns` at both existing call sites.

Then extend `Dataset`:

```python
    #: End of the training span and start of the evaluation window.
    split_ts: int
    window_end: int
    #: Ground truth for incidents planted before the split. Kept apart from
    #: `labels`, which is the evaluation window's ground truth and the only thing
    #: the metrics ever see.
    train_labels: tuple[AnomalyLabel, ...] = ()
```

and in `build_dataset`, after the contamination block and before `events = sorted(...)`:

```python
    train_labels: list[AnomalyLabel] = []
    if config.anomalies.train_rate > 0.0 and config.anomalies.train_patterns:
        history_grants = sum(
            1 for event in normal if event.op is EventOp.GRANT and event.ts < split_ts
        )
        planted, train_labels = _inject(
            config=config,
            org=org,
            rng=rng,
            journal=normal,
            window=(start + WARMUP_MS, split_ts),
            cutoff=start + WARMUP_MS,
            wanted=max(1, round(config.anomalies.train_rate * history_grants)),
            forbidden=normal_keys | {label.edge_key() for label in labels},
            patterns=config.anomalies.train_patterns,
        )
        injected.extend(planted)

    events = sorted([*normal, *injected], key=lambda event: event.ts)
    return Dataset(
        config=config,
        events=tuple(events),
        labels=tuple(sorted(labels, key=lambda label: label.ts)),
        split_ts=split_ts,
        window_end=window_end,
        train_labels=tuple(sorted(train_labels, key=lambda label: label.ts)),
    )
```

Define the warm-up constant next to the other module constants in the same file:

```python
#: Incidents are not planted into the founding burst of day zero: there is no
#: settled structure there for them to stand out against. The same seven days the
#: candidate builder skips.
WARMUP_MS = 7 * DAY_MS
```

- [ ] **Step 5: Label the training candidates**

In `src/rga/features/build.py`, widen the lookup. Training labels sit before the
split and window labels after it, so the two cannot collide:

```python
    # Labels are matched on identity and time together: a normal re-grant of the
    # same edge later in the window is a different candidate, not an anomaly.
    label_of = {
        (label.edge_key(), label.ts): label.pattern
        for label in (*dataset.labels, *dataset.train_labels)
    }
```

- [ ] **Step 6: Write the two dataset recipes**

```yaml
# configs/generator/small-history.yaml
# The small dataset plus labelled incidents before the split, so the supervised
# baseline of spec section 8 has something to learn from. Patterns 6-8
# (dormant_awakening, shadow_group, delegation_cascade) are deliberately absent
# from train_patterns: they appear only in the evaluation window, and how a model
# behaves on them is the result the module is after.
name: small-history
seed: 20260912
eval_window_days: 7

org:
  departments: 3
  teams_per_department: [2, 3]
  users_per_team: [3, 6]
  projects_per_team: [1, 2]
  objects_per_bucket: [200, 400]
  cross_cutting_groups: 1
  cross_cutting_membership_rate: 0.08
  legitimate_exception_rate: 0.04

timeline:
  start_ts: 1735689600000   # 2025-01-01T00:00:00Z
  days: 60
  working_hours: [9, 19]
  off_hours_rate: 0.05
  weekend_rate: 0.08
  hires_per_day: 0.5
  departures_per_day: 0.15
  grants_per_day: 120.0
  uploads_per_day: 50.0

anomalies:
  patterns:
    - self_grant_admin
    - privileged_group_join
    - grant_burst
    - hierarchy_bypass
    - cross_department
    - dormant_awakening
    - shadow_group
    - delegation_cascade
  rate: 0.02
  train_contamination: 0.0
  train_patterns:
    - self_grant_admin
    - privileged_group_join
    - grant_burst
    - hierarchy_bypass
    - cross_department
  train_rate: 0.004
```

Write `configs/generator/default-history.yaml` the same way: copy
`configs/generator/default.yaml` verbatim, change `name` to `default-history`, and
append the same `train_patterns` and `train_rate: 0.004` under `anomalies`.

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `uv run pytest tests/generator tests/features -q`

Expected: green. Then check the shape of what was planted:

```bash
uv run python -c "
from pathlib import Path
from collections import Counter
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
d = build_dataset(load_dataset_config(Path('configs/generator/small-history.yaml')))
t = build_candidates(d, Span.TRAIN)
print('train positives:', int(t.labels.sum()), 'of', t.n_candidates)
print(Counter(p for p in t.patterns if p))
"
```

Expected: at least one incident of every one of the five configured patterns, and a
positive share on the order of a percent. If a pattern is missing, raise `train_rate`
to `0.006` and re-run — the injector plants one of each pattern first, so a missing
one means the generator ran out of candidates, not that the rate was too low.

- [ ] **Step 8: Commit**

```bash
git add src/rga/generator/config.py src/rga/generator/dataset.py src/rga/features/build.py configs/generator/small-history.yaml configs/generator/default-history.yaml tests/generator/test_train_labels.py
git commit -m "feat: plant labelled incidents in the training span"
git push origin main
```

---

### Task 3: Torch runtime — device and seeding

One place decides where tensors live and where randomness comes from. Development is
on a CPU laptop and training on a Blackwell card under Windows, so nothing below may
assume an accelerator exists or hard-code one.

**Files:**
- Create: `src/rga/nn/__init__.py`, `src/rga/nn/runtime.py`
- Test: `tests/nn/test_runtime.py`

**Interfaces:**
- Produces: `select_device(*, prefer_cuda: bool = True) -> torch.device`,
  `seed_torch(seed: int) -> None`, `torch_generator(seed: int, device: torch.device) -> torch.Generator`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_runtime.py
"""Device choice and seeding are explicit and reproducible."""

import torch

from rga.nn.runtime import seed_torch, select_device, torch_generator

CPU = torch.device("cpu")


def test_cpu_is_chosen_when_cuda_is_not_wanted() -> None:
    assert select_device(prefer_cuda=False) == CPU


def test_the_same_seed_initialises_the_same_parameters() -> None:
    seed_torch(7)
    first = torch.nn.Linear(4, 4).weight.detach().clone()
    seed_torch(7)
    second = torch.nn.Linear(4, 4).weight.detach().clone()

    assert torch.equal(first, second)


def test_a_seeded_generator_repeats_its_draws() -> None:
    first = torch.randint(0, 100, (16,), generator=torch_generator(3, CPU))
    second = torch.randint(0, 100, (16,), generator=torch_generator(3, CPU))

    assert torch.equal(first, second)


def test_different_seeds_diverge() -> None:
    first = torch.randint(0, 100, (16,), generator=torch_generator(3, CPU))
    second = torch.randint(0, 100, (16,), generator=torch_generator(4, CPU))

    assert not torch.equal(first, second)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_runtime.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/__init__.py
"""The graph network: layers, heads, training and the scorer that wraps them."""
```

```python
# src/rga/nn/runtime.py
"""Where tensors live and where randomness comes from.

Both are decided once, here, and passed down. Development runs on a CPU laptop and
training on a Blackwell card under Windows, so no module below may assume an
accelerator exists, and reproducibility rules out the implicit global RNG.
"""

from __future__ import annotations

import torch


def select_device(*, prefer_cuda: bool = True) -> torch.device:
    """The device to run on: CUDA when present and wanted, CPU otherwise."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_torch(seed: int) -> None:
    """Seed the global torch RNG, which parameter initialisation reads."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_generator(seed: int, device: torch.device) -> torch.Generator:
    """An explicitly seeded generator for every draw made during training."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_runtime.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/__init__.py src/rga/nn/runtime.py tests/nn/test_runtime.py
git commit -m "feat: add torch device selection and seeding"
git push origin main
```

---

### Task 4: Graph tensors — structure and nothing else

The layer needs the graph as index tensors: for every (relation, direction) slot,
which node each message leaves and which node it arrives at, the ordinal level it
carries, and the normalisation per receiving node. Four relations in two directions
give eight slots; the self-loop is a separate weight inside the layer.

This is also where the rule from spec section 7.4 is enforced in code: only
`edge_src`, `edge_dst`, `edge_rel` and `edge_level` are read. `edge_created` and
`edge_actor` are not, and a test proves it by feeding a graph whose times and actors
have been wiped and asserting the tensors come out identical.

The `keep` argument exists for Task 11: training holds out a slice of edges, and the
encoder must not propagate over the very edges it is being asked to predict.

**Files:**
- Create: `src/rga/nn/graph_tensors.py`
- Test: `tests/nn/test_graph_tensors.py`

**Interfaces:**
- Consumes: `AccessGraph` (`rga.domain.graph`), `RelationType`, `PermissionLevel`
  (`rga.domain.relations`).
- Produces:
  - `SLOTS: tuple[tuple[RelationType, bool], ...]` and `NUM_SLOTS: int` (8)
  - `GraphTensors` frozen dataclass with `num_nodes: int`, `node_type: Tensor`,
    and per-slot tuples `src`, `dst`, `level`, `norm`
  - `graph_tensors(graph, *, device, keep=None) -> GraphTensors`

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_graph_tensors.py
"""The graph as index tensors — structure only."""

from dataclasses import replace

import numpy as np
import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import NUM_SLOTS, SLOTS, graph_tensors

CPU = torch.device("cpu")


def _graph():
    """alice -> devops (member), devops -> logs (admin), logs -> logs/app (parent)."""
    builder = GraphBuilder()
    builder.add_edge("user:alice", RelationType.MEMBER_OF, "group:devops", created=10)
    builder.add_edge(
        "group:devops",
        RelationType.HAS_PERMISSION,
        "bucket:logs",
        level=int(PermissionLevel.ADMIN),
        created=20,
        actor="user:root",
    )
    builder.add_edge(
        "bucket:logs", RelationType.PARENT_OF, "object:logs/app.log", created=30
    )
    return builder.build()


def test_there_are_eight_slots() -> None:
    assert NUM_SLOTS == 8
    assert len(set(SLOTS)) == 8


def test_messages_run_from_source_to_target_on_the_outgoing_slot() -> None:
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    slot = SLOTS.index((RelationType.MEMBER_OF, False))

    assert tensors.src[slot].tolist() == [graph.index_of("user:alice")]
    assert tensors.dst[slot].tolist() == [graph.index_of("group:devops")]


def test_the_incoming_slot_reverses_the_same_edge() -> None:
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    slot = SLOTS.index((RelationType.MEMBER_OF, True))

    assert tensors.src[slot].tolist() == [graph.index_of("group:devops")]
    assert tensors.dst[slot].tolist() == [graph.index_of("user:alice")]


def test_only_permission_edges_carry_a_level() -> None:
    tensors = graph_tensors(_graph(), device=CPU)

    for slot, (relation, _) in enumerate(SLOTS):
        levels = set(tensors.level[slot].tolist())
        if relation is RelationType.HAS_PERMISSION:
            assert levels == {int(PermissionLevel.ADMIN)}
        else:
            assert levels <= {int(PermissionLevel.NONE)}


def test_the_normalisation_is_one_over_root_degree() -> None:
    tensors = graph_tensors(_graph(), device=CPU)
    slot = SLOTS.index((RelationType.MEMBER_OF, False))
    devops = _graph().index_of("group:devops")

    assert tensors.norm[slot][devops].item() == 1.0


def test_times_and_actors_do_not_reach_the_tensors() -> None:
    graph = _graph()
    wiped = replace(
        graph,
        edge_created=np.full_like(graph.edge_created, -1),
        edge_actor=np.full_like(graph.edge_actor, -1),
        node_created=np.full_like(graph.node_created, -1),
    )

    original = graph_tensors(graph, device=CPU)
    blind = graph_tensors(wiped, device=CPU)

    for slot in range(NUM_SLOTS):
        assert torch.equal(original.src[slot], blind.src[slot])
        assert torch.equal(original.dst[slot], blind.dst[slot])
        assert torch.equal(original.level[slot], blind.level[slot])
        assert torch.equal(original.norm[slot], blind.norm[slot])


def test_keeping_a_subset_drops_the_rest() -> None:
    graph = _graph()
    keep = np.array([True, False, True])

    tensors = graph_tensors(graph, device=CPU, keep=keep)

    total = sum(int(tensors.src[slot].numel()) for slot in range(NUM_SLOTS))
    assert total == 4  # two edges, two directions each
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_graph_tensors.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.graph_tensors'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/graph_tensors.py
"""The access graph as the index tensors the layer consumes.

Each (relation, direction) pair is a slot with its own weight matrix in the layer,
because membership in a group and a right on a bucket are not the same kind of
evidence and must not be summed into one neighbourhood.

Only structure is read here: endpoints, relation type and ordinal level. Creation
times and initiators are deliberately left behind — they reach the model through the
candidate feature matrix, where the ablation study can mask them. A layer that read
them from the graph would make the `structural` row of that study a lie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rga.domain.graph import AccessGraph
from rga.domain.relations import RelationType

#: (relation, incoming). Messages on an incoming slot travel against the edge.
SLOTS: tuple[tuple[RelationType, bool], ...] = tuple(
    (relation, incoming) for relation in RelationType for incoming in (False, True)
)
NUM_SLOTS = len(SLOTS)


@dataclass(frozen=True)
class GraphTensors:
    """Index tensors for one graph, one entry per slot."""

    num_nodes: int
    node_type: torch.Tensor
    #: Where each message comes from, per slot.
    src: tuple[torch.Tensor, ...]
    #: Where each message goes, per slot.
    dst: tuple[torch.Tensor, ...]
    #: Ordinal permission level of the edge carrying the message, per slot.
    level: tuple[torch.Tensor, ...]
    #: 1/sqrt(degree) for every receiving node, per slot.
    norm: tuple[torch.Tensor, ...]

    @property
    def device(self) -> torch.device:
        return self.node_type.device


def graph_tensors(
    graph: AccessGraph, *, device: torch.device, keep: np.ndarray | None = None
) -> GraphTensors:
    """Slice a graph into per-slot index tensors.

    `keep` is a boolean mask over edges; the edges it excludes take no part in
    propagation. Training uses it to hold out the edges it validates on.
    """
    selected = (
        np.ones(graph.num_edges, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    )
    if selected.shape != (graph.num_edges,):
        raise ValueError(f"keep must have {graph.num_edges} entries, got {selected.shape}")

    src: list[torch.Tensor] = []
    dst: list[torch.Tensor] = []
    level: list[torch.Tensor] = []
    norm: list[torch.Tensor] = []

    for relation, incoming in SLOTS:
        mask = selected & (graph.edge_rel == int(relation))
        tail = graph.edge_dst[mask] if incoming else graph.edge_src[mask]
        head = graph.edge_src[mask] if incoming else graph.edge_dst[mask]

        degree = np.bincount(head, minlength=graph.num_nodes).astype(np.float32)
        src.append(torch.as_tensor(tail.astype(np.int64), device=device))
        dst.append(torch.as_tensor(head.astype(np.int64), device=device))
        level.append(torch.as_tensor(graph.edge_level[mask].astype(np.int64), device=device))
        norm.append(
            torch.as_tensor(1.0 / np.sqrt(np.maximum(degree, 1.0)), device=device)
        )

    return GraphTensors(
        num_nodes=graph.num_nodes,
        node_type=torch.as_tensor(graph.node_type.astype(np.int64), device=device),
        src=tuple(src),
        dst=tuple(dst),
        level=tuple(level),
        norm=tuple(norm),
    )
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_graph_tensors.py -v`

Expected: 7 passed. If `test_the_normalisation_is_one_over_root_degree` fails,
check that `head` is the receiving end: the normalisation is indexed by the node a
message arrives at, not the one it leaves.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/graph_tensors.py tests/nn/test_graph_tensors.py
git commit -m "feat: turn the access graph into per-slot index tensors"
git push origin main
```

---

### Task 5: Structural node inputs

The encoder needs a starting vector per node. It cannot be the candidate feature row:
that describes one change, and there are far more nodes than candidates. It is built
from the graph instead — node kind, how connected the node is in each slot, and the
strongest right it holds and grants.

Two properties matter and are tested. The vector is derived from the same structure
the layer sees, so no time or provenance sneaks in through this door. And it is what
the reconstruction head of Task 9 has to rebuild, which is what makes "this node's
profile is unusual" a measurable quantity.

**Files:**
- Create: `src/rga/nn/node_inputs.py`
- Test: `tests/nn/test_node_inputs.py`

**Interfaces:**
- Consumes: `GraphTensors`, `NUM_SLOTS` from Task 4.
- Produces: `NODE_INPUT_DIM: int` (14) and
  `node_input_features(tensors: GraphTensors) -> torch.Tensor` of shape
  `[num_nodes, NODE_INPUT_DIM]`, dtype float32, on the tensors' device.

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_node_inputs.py
"""Every node's starting vector, derived from structure alone."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import NODE_INPUT_DIM, node_input_features

CPU = torch.device("cpu")


def _graph():
    builder = GraphBuilder()
    builder.add_edge("user:alice", RelationType.MEMBER_OF, "group:devops", created=10)
    builder.add_edge(
        "group:devops",
        RelationType.HAS_PERMISSION,
        "bucket:logs",
        level=int(PermissionLevel.ADMIN),
        created=20,
    )
    builder.register_node("user:lonely", 30)
    return builder.build()


def test_one_row_per_node_of_the_declared_width() -> None:
    graph = _graph()

    features = node_input_features(graph_tensors(graph, device=CPU))

    assert features.shape == (graph.num_nodes, NODE_INPUT_DIM)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_an_isolated_node_carries_only_its_kind() -> None:
    graph = _graph()

    features = node_input_features(graph_tensors(graph, device=CPU))
    row = features[graph.index_of("user:lonely")]

    assert row.sum().item() == 1.0


def test_a_node_holding_admin_records_the_strongest_level_it_holds() -> None:
    graph = _graph()

    features = node_input_features(graph_tensors(graph, device=CPU))
    devops = features[graph.index_of("group:devops")]
    logs = features[graph.index_of("bucket:logs")]

    assert devops[-2].item() == float(PermissionLevel.ADMIN)
    assert logs[-1].item() == float(PermissionLevel.ADMIN)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_node_inputs.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.node_inputs'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/node_inputs.py
"""The starting vector of every node, read off the graph structure.

Four columns for the node kind, one per slot for how connected it is, and two for
the strongest right it holds and the strongest right held over it. Degrees go
through log1p because they are heavy-tailed: the difference between two and three
group memberships means far more than the difference between two hundred and two
hundred and one.

Nothing here reads a timestamp or an initiator. This vector is also the target of
the reconstruction head, so "the profile of this node is unusual" is measured
against exactly these quantities.
"""

from __future__ import annotations

import torch

from rga.domain.entities import EntityType
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import NUM_SLOTS, SLOTS, GraphTensors

_NUM_TYPES = len(EntityType)
_MAX_LEVEL = int(PermissionLevel.ADMIN)
#: type one-hot + degree per slot + level held + level held over
NODE_INPUT_DIM = _NUM_TYPES + NUM_SLOTS + 2


def node_input_features(tensors: GraphTensors) -> torch.Tensor:
    """Structural attributes of every node, as one float32 matrix."""
    device = tensors.device
    count = tensors.num_nodes

    kind = torch.zeros(count, _NUM_TYPES, device=device)
    # EntityType values start at 1; column 0 is USER.
    kind.scatter_(1, (tensors.node_type - 1).clamp(min=0).unsqueeze(1), 1.0)

    degrees = torch.zeros(count, NUM_SLOTS, device=device)
    for slot in range(NUM_SLOTS):
        counts = torch.zeros(count, device=device)
        arriving = torch.ones_like(tensors.dst[slot], dtype=torch.float32)
        counts.index_add_(0, tensors.dst[slot], arriving)
        degrees[:, slot] = torch.log1p(counts)

    held = torch.zeros(count, device=device)
    held_over = torch.zeros(count, device=device)
    for slot, (relation, incoming) in enumerate(SLOTS):
        if relation is not RelationType.HAS_PERMISSION or incoming:
            continue
        levels = tensors.level[slot]
        # Ascending order makes the last write the maximum. index_reduce_ would say
        # this in one call but is still a beta API, and the two machines run
        # different torch versions.
        for value in range(1, _MAX_LEVEL + 1):
            at_value = levels == value
            if not bool(at_value.any()):
                continue
            held[tensors.src[slot][at_value]] = float(value)
            held_over[tensors.dst[slot][at_value]] = float(value)

    return torch.cat([kind, degrees, held.unsqueeze(1), held_over.unsqueeze(1)], dim=1)
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_node_inputs.py -v`

Expected: 3 passed, with no warnings. The maximum is taken by writing ascending
level values in turn rather than with `index_reduce_`, which is still a beta API and
warns on every call.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/node_inputs.py tests/nn/test_node_inputs.py
git commit -m "feat: derive structural input features for every node"
git push origin main
```

---

### Task 6: The relational message-passing layer

The core of the module, and the reason the spec forbids PyTorch Geometric: the graph
is heterogeneous, multi-relational, and `HAS_PERMISSION` carries an ordinal level
that modulates the message. No stock layer models that, and the alternative would be
inheriting from a library base class and writing the aggregation anyway — at the cost
of `torch-scatter` and `torch-sparse`, which build from source and complicate two
machines. `index_add_` does the same job.

Per spec 7.1:

```
h_v' = LayerNorm(h_v + Dropout(gelu(W_self h_v + Σ_r c_{v,r}^-1 Σ_{u ∈ N_r(v)} m_{u→v,r})))
m_{u→v,r} = W_r h_u + W_lvl emb(level_uv)
```

The eight slot matrices share a basis, as the spec asks, so slots with few edges
borrow strength from the rest instead of overfitting.

**Files:**
- Create: `src/rga/nn/layers.py`
- Test: `tests/nn/test_layers.py`

**Interfaces:**
- Consumes: `GraphTensors`, `NUM_SLOTS` (Task 4), `PermissionLevel`.
- Produces: `RelationalLayer(in_dim, out_dim, *, num_bases=4, dropout=0.1)` with
  `forward(h: Tensor[N, in_dim], tensors: GraphTensors) -> Tensor[N, out_dim]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_layers.py
"""One round of relational message passing."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.layers import RelationalLayer

CPU = torch.device("cpu")


def _graph(level: PermissionLevel = PermissionLevel.READ):
    builder = GraphBuilder()
    builder.add_edge("user:alice", RelationType.MEMBER_OF, "group:devops", created=10)
    builder.add_edge(
        "group:devops", RelationType.HAS_PERMISSION, "bucket:logs", level=int(level), created=20
    )
    builder.register_node("user:lonely", 30)
    return builder.build()


def _layer(dim: int = 5) -> RelationalLayer:
    torch.manual_seed(11)
    layer = RelationalLayer(dim, dim, num_bases=2, dropout=0.0)
    return layer.eval()


def test_output_has_one_row_per_node() -> None:
    graph = _graph()
    layer = _layer()
    h = torch.randn(graph.num_nodes, 5)

    out = layer(h, graph_tensors(graph, device=CPU))

    assert out.shape == (graph.num_nodes, 5)
    assert torch.isfinite(out).all()


def test_an_isolated_node_sees_only_its_own_self_loop() -> None:
    graph = _graph()
    layer = _layer()
    h = torch.randn(graph.num_nodes, 5)
    lonely = graph.index_of("user:lonely")

    out = layer(h, graph_tensors(graph, device=CPU))
    expected = layer.norm(h[lonely] + torch.nn.functional.gelu(layer.self_loop(h[lonely])))

    assert torch.allclose(out[lonely], expected, atol=1e-6)


def test_the_permission_level_changes_the_message() -> None:
    layer = _layer()
    h = torch.randn(_graph().num_nodes, 5)
    target = _graph().index_of("bucket:logs")

    low = layer(h, graph_tensors(_graph(PermissionLevel.READ), device=CPU))
    high = layer(h, graph_tensors(_graph(PermissionLevel.ADMIN), device=CPU))

    assert not torch.allclose(low[target], high[target], atol=1e-6)


def test_the_same_seed_gives_the_same_layer() -> None:
    graph = _graph()
    h = torch.randn(graph.num_nodes, 5)
    tensors = graph_tensors(graph, device=CPU)

    first = _layer()(h, tensors)
    second = _layer()(h, tensors)

    assert torch.equal(first, second)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_layers.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.layers'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/layers.py
"""One round of relational message passing, written by hand.

    h_v' = LayerNorm(h_v + Dropout(gelu(W_self h_v + Σ_r c_{v,r}^-1 Σ_u m_{u→v,r})))
    m_{u→v,r} = W_r h_u + W_lvl emb(level_uv)

Hand-written for two reasons. HAS_PERMISSION carries an ordinal level that belongs in
the message, and no stock relational layer models that. And the libraries that would
supply the aggregation need torch-scatter and torch-sparse, which build from source
and would have to do so on both machines this project runs on; `index_add_` is one
line and behaves identically on CPU and GPU.

The eight per-slot matrices are composed from a shared basis: at this graph size,
eight independent matrices of hidden by hidden would have more parameters than the
graph has edges.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from rga.domain.relations import PermissionLevel
from rga.nn.graph_tensors import NUM_SLOTS, GraphTensors


class RelationalLayer(nn.Module):
    """Propagates node representations one hop across every slot."""

    def __init__(
        self, in_dim: int, out_dim: int, *, num_bases: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.basis = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coefficients = nn.Parameter(torch.empty(NUM_SLOTS, num_bases))
        self.level = nn.Embedding(len(PermissionLevel), out_dim)
        self.residual = (
            nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier on the basis, small noise on the level embedding."""
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.coefficients)
        nn.init.normal_(self.level.weight, std=0.02)

    def forward(self, h: torch.Tensor, tensors: GraphTensors) -> torch.Tensor:
        """One hop of propagation over every slot of `tensors`."""
        weights = torch.einsum("sb,bio->sio", self.coefficients, self.basis)
        total = self.self_loop(h)

        for slot in range(NUM_SLOTS):
            source = tensors.src[slot]
            if source.numel() == 0:
                continue
            messages = h[source] @ weights[slot] + self.level(tensors.level[slot])
            gathered = torch.zeros_like(total)
            gathered.index_add_(0, tensors.dst[slot], messages)
            total = total + gathered * tensors.norm[slot].unsqueeze(1)

        return self.norm(self.residual(h) + self.dropout(F.gelu(total)))
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_layers.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/layers.py tests/nn/test_layers.py
git commit -m "feat: add the relational message passing layer"
git push origin main
```

---

### Task 7: Gradient check against finite differences

Spec 7.3 calls this a separate verifiable result of the work, not merely an internal
test: the layer is hand-written, so the claim that its gradients are correct has to be
demonstrated rather than assumed. Two artefacts come out of this task — a test that
guards every future edit, and a table for the thesis.

**Files:**
- Create: `tests/nn/test_gradients.py`
- Create: `scripts/gradient_check.py`
- Create (generated): `docs/thesis/gradcheck.md`

**Interfaces:**
- Consumes: `RelationalLayer` (Task 6), `graph_tensors` (Task 4).
- Produces: `scripts/gradient_check.py` writing `docs/thesis/gradcheck.md`; no
  importable API.

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_gradients.py
"""Analytic gradients of the hand-written layer against finite differences."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.layers import RelationalLayer

CPU = torch.device("cpu")


def _small_graph():
    """Small enough for finite differences, wide enough to use every slot."""
    builder = GraphBuilder()
    builder.add_edge("user:a", RelationType.MEMBER_OF, "group:g", created=1)
    builder.add_edge("user:b", RelationType.MEMBER_OF, "group:g", created=2)
    builder.add_edge(
        "group:g",
        RelationType.HAS_PERMISSION,
        "bucket:x",
        level=int(PermissionLevel.ADMIN),
        created=3,
    )
    builder.add_edge("bucket:x", RelationType.PARENT_OF, "object:x/k", created=4)
    builder.add_edge("user:a", RelationType.OWNER_OF, "bucket:x", created=5)
    return builder.build()


def test_layer_gradients_match_finite_differences() -> None:
    torch.manual_seed(0)
    graph = _small_graph()
    tensors = graph_tensors(graph, device=CPU)
    layer = RelationalLayer(3, 3, num_bases=2, dropout=0.0).double().eval()

    names = [name for name, _ in layer.named_parameters()]
    values = tuple(
        parameter.detach().clone().requires_grad_(True) for _, parameter in layer.named_parameters()
    )
    h = torch.randn(graph.num_nodes, 3, dtype=torch.double, requires_grad=True)

    def run(node_state, *parameters):
        bound = dict(zip(names, parameters, strict=True))
        return torch.func.functional_call(layer, bound, (node_state, tensors))

    assert torch.autograd.gradcheck(run, (h, *values), eps=1e-6, atol=1e-4, rtol=1e-3)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_gradients.py -v`

Expected: PASS, in fact — the layer of Task 6 is already correct, so this test
documents rather than drives. Confirm it is really exercising the layer by breaking
it on purpose: change `total = total + gathered * tensors.norm[slot].unsqueeze(1)` to
use `gathered.detach()`, re-run, and watch `GradcheckError` appear. Then restore the
line. Do not commit the broken version.

- [ ] **Step 3: Write the reporting script**

```python
# scripts/gradient_check.py
"""Finite-difference check of the hand-written layer, as a table for the report.

Run: uv run python scripts/gradient_check.py
Writes: docs/thesis/gradcheck.md
"""

from __future__ import annotations

from pathlib import Path

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.graph_tensors import graph_tensors
from rga.nn.layers import RelationalLayer

OUTPUT = Path("docs/thesis/gradcheck.md")
EPS = 1e-6


def _graph():
    builder = GraphBuilder()
    builder.add_edge("user:a", RelationType.MEMBER_OF, "group:g", created=1)
    builder.add_edge("user:b", RelationType.MEMBER_OF, "group:g", created=2)
    builder.add_edge(
        "group:g",
        RelationType.HAS_PERMISSION,
        "bucket:x",
        level=int(PermissionLevel.ADMIN),
        created=3,
    )
    builder.add_edge("bucket:x", RelationType.PARENT_OF, "object:x/k", created=4)
    builder.add_edge("user:a", RelationType.OWNER_OF, "bucket:x", created=5)
    return builder.build()


def _deviation(layer, parameter, loss_of) -> float:
    """Largest absolute difference between the two gradients of one tensor."""
    analytic = parameter.grad.detach().clone()
    numeric = torch.zeros_like(parameter)
    flat = parameter.data.view(-1)

    for index in range(flat.numel()):
        original = flat[index].item()
        flat[index] = original + EPS
        plus = loss_of()
        flat[index] = original - EPS
        minus = loss_of()
        flat[index] = original
        numeric.view(-1)[index] = (plus - minus) / (2 * EPS)

    return float((analytic - numeric).abs().max())


def main() -> None:
    torch.manual_seed(0)
    graph = _graph()
    tensors = graph_tensors(graph, device=torch.device("cpu"))
    layer = RelationalLayer(3, 3, num_bases=2, dropout=0.0).double().eval()
    state = torch.randn(graph.num_nodes, 3, dtype=torch.double)

    def loss_of() -> float:
        with torch.no_grad():
            return float(layer(state, tensors).pow(2).sum())

    layer(state, tensors).pow(2).sum().backward()

    rows = [
        (name, tuple(parameter.shape), _deviation(layer, parameter, loss_of))
        for name, parameter in layer.named_parameters()
    ]

    lines = [
        "### Проверка градиентов конечными разностями",
        "",
        "Аналитический градиент против центральной разности с шагом 1e-6, двойная",
        f"точность, граф из {graph.num_nodes} узлов и {graph.num_edges} рёбер.",
        "",
        "| параметр | форма | max |аналитический − численный| |",
        "|---|---|---|",
    ]
    for name, shape, deviation in rows:
        lines.append(f"| `{name}` | {tuple(shape)} | {deviation:.2e} |")
    lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    for name, _, deviation in rows:
        print(f"  {name}: {deviation:.2e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run both and confirm**

Run: `uv run pytest tests/nn/test_gradients.py -v && uv run python scripts/gradient_check.py`

Expected: the test passes, and every printed deviation is below `1e-6`. A deviation
above `1e-4` means a real bug in the layer — stop and find it before going further.

- [ ] **Step 5: Commit**

```bash
git add tests/nn/test_gradients.py scripts/gradient_check.py docs/thesis/gradcheck.md
git commit -m "test: check layer gradients against finite differences"
git push origin main
```

---

### Task 8: The encoder

Three layers give a radius that covers `user → group → bucket → object` — exactly the
paths the authorization engine itself walks.

**Files:**
- Create: `src/rga/nn/config.py`, `src/rga/nn/encoder.py`
- Test: `tests/nn/test_encoder.py`

**Interfaces:**
- Consumes: `RelationalLayer` (Task 6), `NODE_INPUT_DIM` (Task 5).
- Produces:
  - `ModelConfig` frozen dataclass: `hidden_dim: int = 64`, `num_layers: int = 3`,
    `num_bases: int = 4`, `dropout: float = 0.1`, `edge_hidden: int = 64`,
    `embedding_dim: int = 16`, `epochs: int = 200`, `patience: int = 20`,
    `learning_rate: float = 3e-3`, `weight_decay: float = 1e-4`,
    `negatives_per_edge: int = 4`, `validation_share: float = 0.1`,
    `reconstruction_weight: float = 0.5`
  - `GraphEncoder(input_dim: int, config: ModelConfig)` with
    `forward(x: Tensor[N, input_dim], tensors: GraphTensors) -> Tensor[N, hidden_dim]`

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_encoder.py
"""The stack of layers that turns node inputs into representations."""

import torch

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig
from rga.nn.encoder import GraphEncoder
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import NODE_INPUT_DIM, node_input_features

CPU = torch.device("cpu")


def _graph():
    builder = GraphBuilder()
    builder.add_edge("user:alice", RelationType.MEMBER_OF, "group:devops", created=10)
    builder.add_edge(
        "group:devops",
        RelationType.HAS_PERMISSION,
        "bucket:logs",
        level=int(PermissionLevel.ADMIN),
        created=20,
    )
    builder.add_edge("bucket:logs", RelationType.PARENT_OF, "object:logs/a", created=30)
    return builder.build()


def test_the_encoder_returns_one_representation_per_node() -> None:
    torch.manual_seed(5)
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    config = ModelConfig(hidden_dim=16, num_layers=2, dropout=0.0)
    encoder = GraphEncoder(NODE_INPUT_DIM, config).eval()

    out = encoder(node_input_features(tensors), tensors)

    assert out.shape == (graph.num_nodes, 16)
    assert torch.isfinite(out).all()


def test_three_layers_carry_information_the_length_of_the_authorization_path() -> None:
    torch.manual_seed(5)
    graph = _graph()
    tensors = graph_tensors(graph, device=CPU)
    encoder = GraphEncoder(
        NODE_INPUT_DIM, ModelConfig(hidden_dim=16, num_layers=3, dropout=0.0)
    ).eval()
    inputs = node_input_features(tensors)

    baseline = encoder(inputs, tensors)
    moved = inputs.clone()
    moved[graph.index_of("user:alice")] += 1.0
    changed = encoder(moved, tensors)

    far = graph.index_of("object:logs/a")
    assert not torch.allclose(baseline[far], changed[far], atol=1e-6)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_encoder.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.config'`.

- [ ] **Step 3: Write both modules**

```python
# src/rga/nn/config.py
"""Hyperparameters of the network, in one frozen place.

Defaults are the spec's: hidden width 64, three layers — the radius that covers
user → group → bucket → object, which is the path the authorization engine itself
walks. Experiment configs override what they need; nothing reads a loose float.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Everything the model and its training loop need to know."""

    hidden_dim: int = 64
    num_layers: int = 3
    num_bases: int = 4
    dropout: float = 0.1
    edge_hidden: int = 64
    embedding_dim: int = 16
    epochs: int = 200
    patience: int = 20
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    #: Corrupted edges drawn per real edge, spread across the sampling strategies.
    negatives_per_edge: int = 4
    #: Share of the training graph's edges held out to stop on.
    validation_share: float = 0.1
    #: Weight of the profile reconstruction term against the edge likelihood term.
    reconstruction_weight: float = 0.5
```

```python
# src/rga/nn/encoder.py
"""Stacked relational layers: node inputs in, node representations out."""

from __future__ import annotations

import torch
from torch import nn

from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import GraphTensors
from rga.nn.layers import RelationalLayer


class GraphEncoder(nn.Module):
    """Projects structural inputs and propagates them over the graph."""

    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            RelationalLayer(
                config.hidden_dim,
                config.hidden_dim,
                num_bases=config.num_bases,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )

    def forward(self, x: torch.Tensor, tensors: GraphTensors) -> torch.Tensor:
        """Representations of every node after the full stack."""
        h = self.input(x)
        for layer in self.layers:
            h = layer(h, tensors)
        return h
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_encoder.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/config.py src/rga/nn/encoder.py tests/nn/test_encoder.py
git commit -m "feat: add the graph encoder and its hyperparameters"
git push origin main
```

---

### Task 9: The two heads

Spec 7.2. The likelihood head answers "how ordinary is this change", the
reconstruction head answers "how ordinary is this node".

One decision the spec leaves open and this task settles: **the likelihood head takes
the candidate feature row as input.** Without it the network would see only
structure, would be identical at every capability level, and the feature-group
ablation would say nothing about it. With it, the positives and negatives of Task 10
share a feature row and differ only in their endpoints — so the head has to judge
whether *this* structure is plausible *given* that context, which is the question the
whole system asks.

**Files:**
- Create: `src/rga/nn/heads.py`
- Test: `tests/nn/test_heads.py`

**Interfaces:**
- Consumes: `ModelConfig` (Task 8), `RelationType`, `PermissionLevel`.
- Produces:
  - `EdgeLikelihoodHead(node_dim, edge_dim, config)` with
    `forward(h_src, h_dst, relation, level, edge_features) -> Tensor[B]` of logits
  - `NodeReconstructionHead(node_dim, output_dim, config)` with
    `forward(h) -> Tensor[N, output_dim]` and
    `deviation(h, target) -> Tensor[N]`, the mean squared error per node

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_heads.py
"""The likelihood head and the profile reconstruction head."""

import torch

from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig
from rga.nn.heads import EdgeLikelihoodHead, NodeReconstructionHead

CONFIG = ModelConfig(hidden_dim=8, embedding_dim=4, edge_hidden=8, dropout=0.0)


def test_the_likelihood_head_returns_one_logit_per_edge() -> None:
    torch.manual_seed(1)
    head = EdgeLikelihoodHead(node_dim=8, edge_dim=5, config=CONFIG).eval()

    logits = head(
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.tensor([int(RelationType.HAS_PERMISSION)] * 3),
        torch.tensor([int(PermissionLevel.ADMIN)] * 3),
        torch.randn(3, 5),
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_the_likelihood_head_reacts_to_the_permission_level() -> None:
    torch.manual_seed(1)
    head = EdgeLikelihoodHead(node_dim=8, edge_dim=5, config=CONFIG).eval()
    source, target = torch.randn(1, 8), torch.randn(1, 8)
    relation = torch.tensor([int(RelationType.HAS_PERMISSION)])
    features = torch.randn(1, 5)

    read = head(source, target, relation, torch.tensor([int(PermissionLevel.READ)]), features)
    admin = head(source, target, relation, torch.tensor([int(PermissionLevel.ADMIN)]), features)

    assert not torch.allclose(read, admin, atol=1e-6)


def test_the_reconstruction_head_rebuilds_the_profile_shape() -> None:
    torch.manual_seed(1)
    head = NodeReconstructionHead(node_dim=8, output_dim=14, config=CONFIG).eval()

    rebuilt = head(torch.randn(6, 8))

    assert rebuilt.shape == (6, 14)


def test_deviation_is_zero_for_a_perfect_reconstruction() -> None:
    torch.manual_seed(1)
    head = NodeReconstructionHead(node_dim=8, output_dim=14, config=CONFIG).eval()
    state = torch.randn(6, 8)

    perfect = head.deviation(state, head(state))

    assert torch.allclose(perfect, torch.zeros(6), atol=1e-6)


def test_deviation_grows_with_the_error() -> None:
    torch.manual_seed(1)
    head = NodeReconstructionHead(node_dim=8, output_dim=14, config=CONFIG).eval()
    state = torch.randn(6, 8)
    target = head(state)

    near = head.deviation(state, target + 0.1)
    far = head.deviation(state, target + 1.0)

    assert (far > near).all()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_heads.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.heads'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/heads.py
"""What sits on top of the encoder.

The likelihood head scores one change: both endpoint representations, their
elementwise product, the relation and level embeddings, and the candidate's own
feature row. The feature row is what makes the network sensitive to the capability
level of the source — without it the model would see nothing but structure and the
feature-group study would have nothing to say about it.

The reconstruction head rebuilds a node's structural profile from its
representation. The squared error is how much that node departs from what the graph
around it would lead one to expect.
"""

from __future__ import annotations

import torch
from torch import nn

from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.config import ModelConfig


class EdgeLikelihoodHead(nn.Module):
    """A logit per change: high means ordinary."""

    def __init__(self, node_dim: int, edge_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.relation = nn.Embedding(len(RelationType) + 1, config.embedding_dim)
        self.level = nn.Embedding(len(PermissionLevel), config.embedding_dim)
        width = 3 * node_dim + 2 * config.embedding_dim + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(width, config.edge_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_hidden, 1),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        relation: torch.Tensor,
        level: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Logits for a batch of changes."""
        parts = [
            h_src,
            h_dst,
            h_src * h_dst,
            self.relation(relation),
            self.level(level),
            edge_features,
        ]
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


class NodeReconstructionHead(nn.Module):
    """Rebuilds the structural profile of a node from its representation."""

    def __init__(self, node_dim: int, output_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, config.edge_hidden),
            nn.GELU(),
            nn.Linear(config.edge_hidden, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """The reconstructed profile of every node."""
        return self.mlp(h)

    def deviation(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared reconstruction error per node."""
        return (self.mlp(h) - target).pow(2).mean(dim=1)
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_heads.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/heads.py tests/nn/test_heads.py
git commit -m "feat: add the edge likelihood and node reconstruction heads"
git push origin main
```

---

### Task 10: Negative sampling

A negative is a change that did not happen. Spec 7.3 names five ways to build one, and
says the two-hop hard negatives matter most: they force the model to separate "a right
that does not exist but plausibly could" from "a right that does not exist and has no
business existing".

Corruption changes the endpoints or the level of a real candidate and **keeps its
feature row**. The negative therefore describes the same actor, at the same hour, with
the same burst history, granting something else. That is the comparison worth
learning; a negative with a blanked feature row would teach the model only to detect
blank rows.

Sampling is done in NumPy with one seeded `Generator`, then handed to torch.

**Files:**
- Create: `src/rga/nn/negatives.py`
- Test: `tests/nn/test_negatives.py`

**Interfaces:**
- Consumes: `AccessGraph`, `RelationType`, `PermissionLevel`.
- Produces:
  - `CorruptedEdges` frozen dataclass: `src`, `dst`, `relation`, `level`, `origin`
    (all `np.ndarray`, `origin` naming the positive each came from)
  - `sample_negatives(graph, src, dst, relation, level, *, per_edge, rng) -> CorruptedEdges`

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_negatives.py
"""Corrupted changes, the negative half of the contrastive objective."""

import numpy as np

from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType
from rga.nn.negatives import sample_negatives


def _graph():
    builder = GraphBuilder()
    for name in ("alice", "bob", "carol"):
        builder.add_edge(f"user:{name}", RelationType.MEMBER_OF, "group:devops", created=1)
    builder.add_edge(
        "group:devops",
        RelationType.HAS_PERMISSION,
        "bucket:logs",
        level=int(PermissionLevel.READ),
        created=2,
    )
    builder.add_edge(
        "user:alice",
        RelationType.HAS_PERMISSION,
        "bucket:reports",
        level=int(PermissionLevel.WRITE),
        created=3,
    )
    return builder.build()


def _positives(graph):
    src = np.array([graph.index_of("user:alice")], dtype=np.int64)
    dst = np.array([graph.index_of("bucket:reports")], dtype=np.int64)
    relation = np.array([int(RelationType.HAS_PERMISSION)], dtype=np.int64)
    level = np.array([int(PermissionLevel.WRITE)], dtype=np.int64)
    return src, dst, relation, level


def test_the_requested_number_of_negatives_is_produced() -> None:
    graph = _graph()

    corrupted = sample_negatives(
        graph, *_positives(graph), per_edge=4, rng=np.random.default_rng(0)
    )

    assert corrupted.src.shape == (4,)
    assert corrupted.origin.tolist() == [0, 0, 0, 0]


def test_every_negative_differs_from_its_positive() -> None:
    graph = _graph()
    src, dst, relation, level = _positives(graph)

    corrupted = sample_negatives(
        graph, src, dst, relation, level, per_edge=8, rng=np.random.default_rng(1)
    )

    same = (
        (corrupted.src == src[0])
        & (corrupted.dst == dst[0])
        & (corrupted.level == level[0])
    )
    assert not same.any()


def test_indices_stay_inside_the_graph() -> None:
    graph = _graph()

    corrupted = sample_negatives(
        graph, *_positives(graph), per_edge=16, rng=np.random.default_rng(2)
    )

    assert corrupted.src.min() >= 0
    assert corrupted.src.max() < graph.num_nodes
    assert corrupted.dst.min() >= 0
    assert corrupted.dst.max() < graph.num_nodes
    assert set(corrupted.level.tolist()) <= {level.value for level in PermissionLevel}


def test_sampling_is_reproducible() -> None:
    graph = _graph()
    positives = _positives(graph)

    first = sample_negatives(graph, *positives, per_edge=8, rng=np.random.default_rng(3))
    second = sample_negatives(graph, *positives, per_edge=8, rng=np.random.default_rng(3))

    assert np.array_equal(first.src, second.src)
    assert np.array_equal(first.dst, second.dst)
    assert np.array_equal(first.level, second.level)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_negatives.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.negatives'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/negatives.py
"""Changes that did not happen.

Five strategies, rotated across the draws so every batch contains all of them:

1. the object replaced uniformly — the easy case, teaches the gross shape;
2. the object replaced in proportion to how many rights already point at it —
   popular targets are plausible targets, so these are harder;
3. the subject replaced uniformly — the same right granted to somebody else;
4. the level moved one step along the ordinal scale — the hardest kind of near
   miss, and the reason the level is modelled as an order rather than a category;
5. a node two hops away in the undirected projection — a right that does not exist
   but plausibly could, which is exactly the boundary the model has to learn.

The feature row of the positive travels with its negatives unchanged: only the
structure is corrupted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rga.domain.graph import AccessGraph
from rga.domain.relations import LEVEL_CARRYING, PermissionLevel, RelationType

_MIN_LEVEL = int(PermissionLevel.READ)
_MAX_LEVEL = int(PermissionLevel.ADMIN)


@dataclass(frozen=True)
class CorruptedEdges:
    """Negatives and the positive each one was made from."""

    src: np.ndarray
    dst: np.ndarray
    relation: np.ndarray
    level: np.ndarray
    #: Index of the positive a negative belongs to, for gathering its feature row.
    origin: np.ndarray


def _undirected_neighbours(graph: AccessGraph) -> list[np.ndarray]:
    """Neighbour indices per node, ignoring direction and relation."""
    buckets: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for source, target in zip(graph.edge_src, graph.edge_dst, strict=True):
        buckets[int(source)].append(int(target))
        buckets[int(target)].append(int(source))
    return [np.array(sorted(set(items)), dtype=np.int64) for items in buckets]


def _two_hop(neighbours: list[np.ndarray], node: int, rng: np.random.Generator) -> int:
    """A node two hops away, or -1 when the neighbourhood is too small."""
    first = neighbours[node]
    if first.size == 0:
        return -1
    middle = int(rng.choice(first))
    second = neighbours[middle]
    if second.size == 0:
        return -1
    return int(rng.choice(second))


def sample_negatives(
    graph: AccessGraph,
    src: np.ndarray,
    dst: np.ndarray,
    relation: np.ndarray,
    level: np.ndarray,
    *,
    per_edge: int,
    rng: np.random.Generator,
) -> CorruptedEdges:
    """Draw `per_edge` corrupted variants of every positive."""
    neighbours = _undirected_neighbours(graph)
    in_degree = np.bincount(graph.edge_dst, minlength=graph.num_nodes).astype(np.float64)
    popularity = in_degree + 1.0
    popularity /= popularity.sum()

    out_src: list[int] = []
    out_dst: list[int] = []
    out_relation: list[int] = []
    out_level: list[int] = []
    out_origin: list[int] = []

    for position in range(len(src)):
        base = (
            int(src[position]),
            int(dst[position]),
            int(relation[position]),
            int(level[position]),
        )
        for draw in range(per_edge):
            new_src, new_dst, new_relation, new_level = base
            strategy = draw % 5

            if strategy == 0:
                new_dst = int(rng.integers(graph.num_nodes))
            elif strategy == 1:
                new_dst = int(rng.choice(graph.num_nodes, p=popularity))
            elif strategy == 2:
                new_src = int(rng.integers(graph.num_nodes))
            elif strategy == 3 and RelationType(new_relation) in LEVEL_CARRYING:
                if new_level >= _MAX_LEVEL:
                    step = -1
                elif new_level <= _MIN_LEVEL:
                    step = 1
                else:
                    step = int(rng.choice([-1, 1]))
                new_level = int(np.clip(new_level + step, _MIN_LEVEL, _MAX_LEVEL))
            else:
                candidate = _two_hop(neighbours, new_src, rng)
                new_dst = candidate if candidate >= 0 else int(rng.integers(graph.num_nodes))

            if (new_src, new_dst, new_relation, new_level) == base:
                # A corruption that changed nothing is not a negative. Fall back to
                # the uniform object swap, retrying until it lands elsewhere.
                while new_dst == base[1]:
                    new_dst = int(rng.integers(graph.num_nodes))

            out_src.append(new_src)
            out_dst.append(new_dst)
            out_relation.append(new_relation)
            out_level.append(new_level)
            out_origin.append(position)

    return CorruptedEdges(
        src=np.array(out_src, dtype=np.int64),
        dst=np.array(out_dst, dtype=np.int64),
        relation=np.array(out_relation, dtype=np.int64),
        level=np.array(out_level, dtype=np.int64),
        origin=np.array(out_origin, dtype=np.int64),
    )
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_negatives.py -v`

Expected: 4 passed. A graph with two nodes would make
`test_every_negative_differs_from_its_positive` loop forever in the fallback; the
fixture has six nodes, and the real graphs have thousands.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/negatives.py tests/nn/test_negatives.py
git commit -m "feat: add negative sampling for the contrastive objective"
git push origin main
```

---

### Task 11: Candidate arrays and the training loop

Two modules, one deliverable: turning a `CandidateSet` into the index arrays the model
consumes, and training the model on them.

The methodological point of spec 7.3 is enforced here. A tenth of the training
candidates are held out, **their edges are removed from the graph the encoder
propagates over**, and early stopping watches the likelihood the model assigns to
those held-out edges. No anomaly label is read anywhere in this file. Stopping on a
labelled metric would leak the ground truth into the model through the choice of
epoch, and the result would stop being honest.

**Files:**
- Create: `src/rga/nn/candidates.py`, `src/rga/nn/train.py`
- Test: `tests/nn/test_train.py`

**Interfaces:**
- Consumes: everything from Tasks 3-10, plus `dense_matrix` from
  `rga.baselines.base` and `CandidateSet` from `rga.features.spec`.
- Produces:
  - `CandidateArrays` frozen dataclass: `src`, `dst`, `relation`, `level`, `features`
    (`src`/`dst` hold `-1` for an endpoint the graph has never seen)
  - `candidate_arrays(candidates: CandidateSet, *, mean=None, std=None) -> tuple[CandidateArrays, np.ndarray, np.ndarray]`
    returning the arrays and the standardisation it used
  - `edge_positions(graph, src, dst, relation) -> np.ndarray`
  - `GnnModel(edge_dim: int, config: ModelConfig)` holding `encoder`, `likelihood`,
    `reconstruction`
  - `train_model(graph, arrays, config, *, seed, device) -> GnnModel`

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_train.py
"""Candidate arrays and the self-supervised training loop."""

from pathlib import Path

import numpy as np
import torch

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.candidates import candidate_arrays, edge_positions
from rga.nn.config import ModelConfig
from rga.nn.train import train_model

CPU = torch.device("cpu")
CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


def _train_set():
    return build_candidates(build_dataset(CONFIG), Span.TRAIN)


def test_arrays_have_one_row_per_candidate() -> None:
    train = _train_set()

    arrays, mean, std = candidate_arrays(train)

    assert arrays.src.shape == (train.n_candidates,)
    assert arrays.features.shape[0] == train.n_candidates
    assert mean.shape == (arrays.features.shape[1],)
    assert std.shape == (arrays.features.shape[1],)


def test_standardisation_can_be_reused_on_another_span() -> None:
    dataset = build_dataset(CONFIG)
    train = build_candidates(dataset, Span.TRAIN)
    evaluation = build_candidates(dataset, Span.EVAL)

    _, mean, std = candidate_arrays(train)
    arrays, reused_mean, reused_std = candidate_arrays(evaluation, mean=mean, std=std)

    assert np.array_equal(reused_mean, mean)
    assert np.array_equal(reused_std, std)
    assert np.isfinite(arrays.features).all()


def test_levels_stay_inside_the_embedding_range() -> None:
    arrays, _, _ = candidate_arrays(_train_set())

    assert arrays.level.min() >= 0
    assert arrays.level.max() <= 5


def test_edge_positions_find_the_edges_the_candidates_created() -> None:
    train = _train_set()
    arrays, _, _ = candidate_arrays(train)

    positions = edge_positions(train.graph, arrays.src, arrays.dst, arrays.relation)

    assert positions.shape == (train.n_candidates,)
    assert (positions >= 0).sum() > 0


def test_training_runs_and_returns_a_usable_model() -> None:
    train = _train_set()
    arrays, _, _ = candidate_arrays(train)

    model = train_model(train.graph, arrays, FAST, seed=1, device=CPU)

    assert isinstance(model, torch.nn.Module)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_training_is_reproducible() -> None:
    train = _train_set()
    arrays, _, _ = candidate_arrays(train)

    first = train_model(train.graph, arrays, FAST, seed=1, device=CPU)
    second = train_model(train.graph, arrays, FAST, seed=1, device=CPU)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_train.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.candidates'`.

- [ ] **Step 3: Write the candidate arrays**

```python
# src/rga/nn/candidates.py
"""A candidate set as the arrays the model indexes with.

Endpoints are resolved against the graph at the split. An endpoint the graph has
never seen — a user hired inside the evaluation window, say — comes back as -1, and
the scorer gives it a zero representation rather than pretending it knows the node.

The permission level is read off the `level_ordinal` feature, which belongs to the
structural group and is therefore present at every capability level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rga.baselines.base import dense_matrix
from rga.domain.graph import AccessGraph
from rga.domain.relations import PermissionLevel
from rga.features.spec import CandidateSet

_MAX_LEVEL = int(PermissionLevel.ADMIN)


@dataclass(frozen=True)
class CandidateArrays:
    """Index arrays and the standardised feature matrix of one span."""

    src: np.ndarray
    dst: np.ndarray
    relation: np.ndarray
    level: np.ndarray
    features: np.ndarray


def candidate_arrays(
    candidates: CandidateSet,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[CandidateArrays, np.ndarray, np.ndarray]:
    """Arrays for one span, standardised by the training span's statistics."""
    if candidates.graph is None:
        raise ValueError("the candidate set carries no graph; the network needs one")

    index = candidates.graph.node_index
    src = np.array([index.get(key[0], -1) for key in candidates.keys], dtype=np.int64)
    dst = np.array([index.get(key[2], -1) for key in candidates.keys], dtype=np.int64)
    relation = np.array([key[1] for key in candidates.keys], dtype=np.int64)

    ordinal = candidates.matrix.column("level_ordinal")
    level = np.clip(np.rint(ordinal), 0, _MAX_LEVEL).astype(np.int64)

    dense = dense_matrix(candidates.matrix)
    if mean is None or std is None:
        mean = dense.mean(axis=0)
        std = dense.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    features = ((dense - mean) / std).astype(np.float32)

    return (
        CandidateArrays(src=src, dst=dst, relation=relation, level=level, features=features),
        mean,
        std,
    )


def edge_positions(
    graph: AccessGraph, src: np.ndarray, dst: np.ndarray, relation: np.ndarray
) -> np.ndarray:
    """Row of each candidate's edge in the graph, or -1 when it is not there."""
    lookup = {
        (int(a), int(r), int(b)): position
        for position, (a, r, b) in enumerate(
            zip(graph.edge_src, graph.edge_rel, graph.edge_dst, strict=True)
        )
    }
    return np.array(
        [
            lookup.get((int(a), int(r), int(b)), -1)
            for a, r, b in zip(src, relation, dst, strict=True)
        ],
        dtype=np.int64,
    )
```

- [ ] **Step 4: Write the training loop**

```python
# src/rga/nn/train.py
"""Self-supervised training: real changes against corrupted ones.

Model selection never sees an anomaly label. A tenth of the training candidates are
held out, their edges are taken out of the graph the encoder propagates over, and
early stopping watches the likelihood the model gives those held-out changes. Any
label-aware criterion here would leak the ground truth into the model through the
choice of epoch, and every number measured afterwards would be worth less.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from rga.domain.graph import AccessGraph
from rga.nn.candidates import CandidateArrays, edge_positions
from rga.nn.config import ModelConfig
from rga.nn.encoder import GraphEncoder
from rga.nn.graph_tensors import graph_tensors
from rga.nn.heads import EdgeLikelihoodHead, NodeReconstructionHead
from rga.nn.negatives import sample_negatives
from rga.nn.node_inputs import NODE_INPUT_DIM, node_input_features
from rga.nn.runtime import seed_torch


class GnnModel(nn.Module):
    """The encoder and both heads, trained together."""

    def __init__(self, edge_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = GraphEncoder(NODE_INPUT_DIM, config)
        self.likelihood = EdgeLikelihoodHead(config.hidden_dim, edge_dim, config)
        self.reconstruction = NodeReconstructionHead(config.hidden_dim, NODE_INPUT_DIM, config)


def train_model(
    graph: AccessGraph,
    arrays: CandidateArrays,
    config: ModelConfig,
    *,
    seed: int,
    device: torch.device,
) -> GnnModel:
    """Fit the model on one training span and return it at its best epoch."""
    rng = np.random.default_rng(seed)
    seed_torch(seed)

    usable = np.flatnonzero((arrays.src >= 0) & (arrays.dst >= 0))
    if usable.size < 2:
        raise ValueError("not enough candidates with known endpoints to train on")
    rng.shuffle(usable)
    cut = max(1, int(usable.size * config.validation_share))
    validation, fit = usable[:cut], usable[cut:]

    positions = edge_positions(graph, arrays.src, arrays.dst, arrays.relation)
    keep = np.ones(graph.num_edges, dtype=bool)
    held = positions[validation]
    keep[held[held >= 0]] = False

    tensors = graph_tensors(graph, device=device, keep=keep)
    inputs = node_input_features(tensors)

    model = GnnModel(edge_dim=arrays.features.shape[1], config=config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(config.epochs, 1))

    features = torch.as_tensor(arrays.features, device=device)
    relation = torch.as_tensor(arrays.relation, device=device)
    level = torch.as_tensor(arrays.level, device=device)
    src = torch.as_tensor(arrays.src, device=device)
    dst = torch.as_tensor(arrays.dst, device=device)
    fit_index = torch.as_tensor(fit, device=device)
    validation_index = torch.as_tensor(validation, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_likelihood = -float("inf")
    waited = 0

    for _ in range(config.epochs):
        model.train()
        corrupted = sample_negatives(
            graph,
            arrays.src[fit],
            arrays.dst[fit],
            arrays.relation[fit],
            arrays.level[fit],
            per_edge=config.negatives_per_edge,
            rng=rng,
        )
        origin = torch.as_tensor(fit[corrupted.origin], device=device)

        h = model.encoder(inputs, tensors)
        positive = model.likelihood(
            h[src[fit_index]],
            h[dst[fit_index]],
            relation[fit_index],
            level[fit_index],
            features[fit_index],
        )
        negative = model.likelihood(
            h[torch.as_tensor(corrupted.src, device=device)],
            h[torch.as_tensor(corrupted.dst, device=device)],
            torch.as_tensor(corrupted.relation, device=device),
            torch.as_tensor(corrupted.level, device=device),
            features[origin],
        )

        loss = F.binary_cross_entropy_with_logits(
            positive, torch.ones_like(positive)
        ) + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
        loss = loss + config.reconstruction_weight * F.mse_loss(
            model.reconstruction(h), inputs
        )

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()

        model.eval()
        with torch.no_grad():
            held_out = model.encoder(inputs, tensors)
            logits = model.likelihood(
                held_out[src[validation_index]],
                held_out[dst[validation_index]],
                relation[validation_index],
                level[validation_index],
                features[validation_index],
            )
            likelihood = float(F.logsigmoid(logits).mean())

        if likelihood > best_likelihood:
            best_likelihood, waited = likelihood, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config.patience:
                break

    model.load_state_dict(best_state)
    return model.eval()
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/nn/test_train.py -v`

Expected: 6 passed. The fast config keeps this under a minute on the laptop; if
`test_training_is_reproducible` fails, something is drawing from an unseeded RNG —
find it rather than loosening the assertion.

- [ ] **Step 6: Commit**

```bash
git add src/rga/nn/candidates.py src/rga/nn/train.py tests/nn/test_train.py
git commit -m "feat: add the self-supervised training loop"
git push origin main
```

---

### Task 12: The anomaly score

Spec section 3:

```
A(e) = α · (1 − σ(s(e))) + β · d(u) + γ · d(v)
```

The three terms live on different scales and the last two are badly skewed, so each
is turned into its quantile within the distribution observed on the training span,
and the quantiles are averaged with equal weight. Equal weights are the only choice
that needs no labels.

An endpoint the graph has never seen gets a neutral 0.5 rather than a maximum: a
newly hired employee has no history, and treating "no history" as "maximally
suspicious" would fill the analyst's queue with every new hire.

**Files:**
- Create: `src/rga/nn/scoring.py`
- Test: `tests/nn/test_scoring.py`

**Interfaces:**
- Produces:
  - `RankTransform.fit(values: np.ndarray) -> RankTransform` and
    `RankTransform.apply(values: np.ndarray) -> np.ndarray` mapping to `[0, 1]`
  - `combine(likelihood_rank, subject_rank, object_rank) -> np.ndarray`
  - `NEUTRAL: float = 0.5`

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_scoring.py
"""Rank transform and the three-term anomaly score."""

import numpy as np

from rga.nn.scoring import NEUTRAL, RankTransform, combine


def test_the_transform_maps_the_reference_onto_the_unit_interval() -> None:
    transform = RankTransform.fit(np.arange(100.0))

    ranks = transform.apply(np.array([0.0, 50.0, 99.0]))

    assert ranks.min() >= 0.0
    assert ranks.max() <= 1.0
    assert ranks[0] < ranks[1] < ranks[2]


def test_values_beyond_the_reference_saturate() -> None:
    transform = RankTransform.fit(np.arange(100.0))

    assert transform.apply(np.array([1e9]))[0] == 1.0
    assert transform.apply(np.array([-1e9]))[0] == 0.0


def test_the_transform_is_monotone() -> None:
    rng = np.random.default_rng(0)
    reference = rng.lognormal(size=500)
    transform = RankTransform.fit(reference)

    probe = np.sort(rng.lognormal(size=50))
    ranks = transform.apply(probe)

    assert np.all(np.diff(ranks) >= 0.0)


def test_the_score_averages_the_three_terms() -> None:
    score = combine(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.4, 0.0])
    )

    assert np.allclose(score, np.array([0.8, 0.0]))


def test_a_neutral_term_is_the_middle_of_the_range() -> None:
    assert NEUTRAL == 0.5
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_scoring.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.scoring'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/scoring.py
"""Putting three incomparable quantities on one scale.

The likelihood term and the two profile deviations differ by orders of magnitude and
the deviations are heavily skewed, so a z-score would be dominated by whichever tail
happens to be longest. Each term is replaced by its quantile within the distribution
seen on the training span, which is scale-free and robust, and the quantiles are
averaged with equal weight — the only weighting that requires no labels to justify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The rank given where a quantity cannot be computed — an endpoint the graph has
#: never seen. Neither evidence for nor against; a new hire is not a suspect.
NEUTRAL = 0.5


@dataclass(frozen=True)
class RankTransform:
    """Maps a value to its quantile in a reference distribution."""

    reference: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> RankTransform:
        """Take the training-span distribution as the reference."""
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("cannot fit a rank transform on an empty reference")
        return cls(reference=np.sort(finite))

    def apply(self, values: np.ndarray) -> np.ndarray:
        """The share of the reference at or below each value, in [0, 1]."""
        probe = np.asarray(values, dtype=np.float64)
        positions = np.searchsorted(self.reference, probe, side="right")
        return positions / float(self.reference.size)


def combine(
    likelihood_rank: np.ndarray, subject_rank: np.ndarray, object_rank: np.ndarray
) -> np.ndarray:
    """The anomaly score: equal weights over the three ranked terms."""
    stacked = np.vstack([likelihood_rank, subject_rank, object_rank])
    return stacked.mean(axis=0)
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/nn/test_scoring.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rga/nn/scoring.py tests/nn/test_scoring.py
git commit -m "feat: add the rank transform and the combined anomaly score"
git push origin main
```

---

### Task 13: The scorer the experiment stand sees

Everything so far becomes one more implementation of the existing `Scorer` protocol.
Nothing in the runner, the metrics or the candidate builder changes.

At scoring time the encoder propagates over the **whole** graph at the split — the
held-out edges of training were held out to stop honestly, not to be forgotten.

**Files:**
- Create: `src/rga/nn/scorer.py`
- Modify: `src/rga/eval/experiment.py` (`build_scorer`)
- Test: `tests/nn/test_scorer.py`

**Interfaces:**
- Consumes: Tasks 3-12.
- Produces: `GnnScorer(seed: int, config: ModelConfig | None = None, device: torch.device | None = None)`
  with `name = "gnn"`, `fit(train: CandidateSet) -> None`,
  `score(candidates: CandidateSet) -> np.ndarray`; `build_scorer("gnn", seed)` returns it.

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_scorer.py
"""The network as one more scorer in the stand."""

from pathlib import Path

import numpy as np
import pytest

from rga.baselines.base import Scorer
from rga.eval.experiment import build_scorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3, negatives_per_edge=2)


@pytest.fixture(scope="module")
def spans():
    dataset = build_dataset(CONFIG)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


def test_the_scorer_satisfies_the_protocol() -> None:
    assert isinstance(GnnScorer(seed=0, config=FAST), Scorer)


def test_the_factory_knows_the_name() -> None:
    assert build_scorer("gnn", seed=0).name == "gnn"


def test_one_finite_score_per_candidate(spans) -> None:
    train, evaluation = spans
    scorer = GnnScorer(seed=0, config=FAST)

    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.shape == (evaluation.n_candidates,)
    assert np.isfinite(scores).all()


def test_scores_stay_inside_the_unit_interval(spans) -> None:
    train, evaluation = spans
    scorer = GnnScorer(seed=0, config=FAST)

    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_the_same_seed_gives_the_same_ranking(spans) -> None:
    train, evaluation = spans

    first = GnnScorer(seed=4, config=FAST)
    first.fit(train)
    second = GnnScorer(seed=4, config=FAST)
    second.fit(train)

    assert np.array_equal(first.score(evaluation), second.score(evaluation))


def test_scoring_before_fitting_is_refused(spans) -> None:
    _, evaluation = spans

    with pytest.raises(RuntimeError, match="must be fit"):
        GnnScorer(seed=0, config=FAST).score(evaluation)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_scorer.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.scorer'`.

- [ ] **Step 3: Write the scorer**

```python
# src/rga/nn/scorer.py
"""The network wearing the Scorer protocol.

Fitting trains the model on the training span and records the distributions the rank
transform needs. Scoring propagates over the whole graph at the split — the edges
held out during training were held out to stop honestly, not to be thrown away — and
combines the three terms of spec section 3.
"""

from __future__ import annotations

import numpy as np
import torch

from rga.features.spec import CandidateSet
from rga.nn.candidates import CandidateArrays, candidate_arrays
from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import node_input_features
from rga.nn.runtime import select_device
from rga.nn.scoring import NEUTRAL, RankTransform, combine
from rga.nn.train import GnnModel, train_model


class GnnScorer:
    """Ranks changes by how unusual the graph makes them look."""

    name = "gnn"

    def __init__(
        self,
        seed: int,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._seed = seed
        self._config = config or ModelConfig()
        self._device = device or select_device()
        self._model: GnnModel | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._state: torch.Tensor | None = None
        self._deviation: np.ndarray | None = None
        self._likelihood_rank: RankTransform | None = None
        self._deviation_rank: RankTransform | None = None

    def fit(self, train: CandidateSet) -> None:
        """Train on the span and record what the rank transform needs."""
        if train.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")

        arrays, mean, std = candidate_arrays(train)
        self._mean, self._std = mean, std
        self._model = train_model(
            train.graph, arrays, self._config, seed=self._seed, device=self._device
        )

        tensors = graph_tensors(train.graph, device=self._device)
        inputs = node_input_features(tensors)
        with torch.no_grad():
            self._state = self._model.encoder(inputs, tensors)
            self._deviation = (
                self._model.reconstruction.deviation(self._state, inputs).cpu().numpy()
            )

        unlikeliness = 1.0 - self._likelihood(arrays)
        self._likelihood_rank = RankTransform.fit(unlikeliness)
        self._deviation_rank = RankTransform.fit(self._deviation)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """One score per candidate in [0, 1], higher meaning more unusual."""
        if self._model is None or self._likelihood_rank is None:
            raise RuntimeError("the gnn scorer must be fit before scoring")

        arrays, _, _ = candidate_arrays(candidates, mean=self._mean, std=self._std)
        unlikeliness = self._likelihood_rank.apply(1.0 - self._likelihood(arrays))
        return combine(
            unlikeliness,
            self._node_rank(arrays.src),
            self._node_rank(arrays.dst),
        )

    def _likelihood(self, arrays: CandidateArrays) -> np.ndarray:
        """Probability the model assigns to each change being ordinary."""
        assert self._model is not None and self._state is not None
        padded = torch.cat(
            [self._state, torch.zeros(1, self._state.shape[1], device=self._device)]
        )
        unknown = padded.shape[0] - 1

        def endpoints(index: np.ndarray) -> torch.Tensor:
            resolved = np.where(index < 0, unknown, index)
            return torch.as_tensor(resolved, device=self._device)

        with torch.no_grad():
            logits = self._model.likelihood(
                padded[endpoints(arrays.src)],
                padded[endpoints(arrays.dst)],
                torch.as_tensor(arrays.relation, device=self._device),
                torch.as_tensor(arrays.level, device=self._device),
                torch.as_tensor(arrays.features, device=self._device),
            )
        return torch.sigmoid(logits).cpu().numpy()

    def _node_rank(self, index: np.ndarray) -> np.ndarray:
        """Ranked profile deviation of an endpoint, neutral where it is unknown."""
        assert self._deviation is not None and self._deviation_rank is not None
        known = index >= 0
        ranks = np.full(index.shape, NEUTRAL, dtype=np.float64)
        if known.any():
            ranks[known] = self._deviation_rank.apply(self._deviation[index[known]])
        return ranks
```

- [ ] **Step 4: Teach the factory the new name**

In `src/rga/eval/experiment.py`, extend `build_scorer`. The import is local so that
the package stays importable without torch installed, exactly as the baselines keep
scikit-learn optional:

```python
def build_scorer(name: str, seed: int) -> Scorer:
    """Construct a scorer by the name used in configuration files."""
    if name == "rules":
        return RuleScorer()
    if name == "isolation_forest":
        return IsolationForestScorer(seed=seed)
    if name == "lof":
        return LocalOutlierFactorScorer()
    if name == "gnn":
        from rga.nn.scorer import GnnScorer

        return GnnScorer(seed=seed)
    raise KeyError(f"unknown scorer: {name!r}")
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/nn tests/eval -q`

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/rga/nn/scorer.py src/rga/eval/experiment.py tests/nn/test_scorer.py
git commit -m "feat: wire the graph network into the experiment stand"
git push origin main
```

---

### Task 14: The supervised contrast

Spec section 8 asks for the same encoder trained as a classifier on labelled patterns
1-5. It gives the upper bound on familiar threats, and — this is the point — it is
expected to fall over on patterns 6-8, which it has never seen. The self-supervised
model has seen no labels at all, so all eight patterns are equally unfamiliar to it.
That contrast is the module's main result.

This scorer needs a dataset whose training span carries labelled incidents, which is
what Task 2 built. On a dataset without them it refuses to fit rather than silently
training on nothing.

**Files:**
- Create: `src/rga/nn/supervised.py`
- Modify: `src/rga/eval/experiment.py` (`build_scorer`)
- Test: `tests/nn/test_supervised.py`

**Interfaces:**
- Consumes: Tasks 3-13.
- Produces: `SupervisedGnnScorer(seed, config=None, device=None)` with
  `name = "gnn_supervised"`; `build_scorer("gnn_supervised", seed)` returns it.

- [ ] **Step 1: Write the failing test**

```python
# tests/nn/test_supervised.py
"""The supervised contrast baseline."""

from pathlib import Path

import numpy as np
import pytest

from rga.baselines.base import Scorer
from rga.eval.experiment import build_scorer
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.supervised import SupervisedGnnScorer

HISTORY = load_dataset_config(Path("configs/generator/small-history.yaml"))
PLAIN = load_dataset_config(Path("configs/generator/small.yaml"))
FAST = ModelConfig(hidden_dim=16, num_layers=2, epochs=3, patience=3)


@pytest.fixture(scope="module")
def spans():
    dataset = build_dataset(HISTORY)
    return build_candidates(dataset, Span.TRAIN), build_candidates(dataset, Span.EVAL)


def test_the_scorer_satisfies_the_protocol() -> None:
    assert isinstance(SupervisedGnnScorer(seed=0, config=FAST), Scorer)


def test_the_factory_knows_the_name() -> None:
    assert build_scorer("gnn_supervised", seed=0).name == "gnn_supervised"


def test_one_finite_score_per_candidate(spans) -> None:
    train, evaluation = spans
    scorer = SupervisedGnnScorer(seed=0, config=FAST)

    scorer.fit(train)
    scores = scorer.score(evaluation)

    assert scores.shape == (evaluation.n_candidates,)
    assert np.isfinite(scores).all()


def test_fitting_without_labels_is_refused() -> None:
    train = build_candidates(build_dataset(PLAIN), Span.TRAIN)

    with pytest.raises(ValueError, match="labelled"):
        SupervisedGnnScorer(seed=0, config=FAST).fit(train)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/nn/test_supervised.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rga.nn.supervised'`.

- [ ] **Step 3: Write the module**

```python
# src/rga/nn/supervised.py
"""The same encoder, trained as a classifier on labelled incidents.

It exists to be beaten in one specific way. On the patterns it was trained on it
should do very well, and on the three it has never seen it is expected to fall over.
The self-supervised model never sees a label, so nothing is familiar to it and
nothing is unfamiliar either — which is the comparison spec section 8 is after.

Anomalies are a few percent of the training span, so the positive class is weighted
up rather than resampled; resampling would change the graph the encoder propagates
over, and the two models must see the same graph for the comparison to mean anything.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from rga.features.spec import CandidateSet
from rga.nn.candidates import candidate_arrays
from rga.nn.config import ModelConfig
from rga.nn.graph_tensors import graph_tensors
from rga.nn.node_inputs import node_input_features
from rga.nn.runtime import seed_torch, select_device
from rga.nn.train import GnnModel


class SupervisedGnnScorer:
    """Ranks changes by a classifier trained on labelled incidents."""

    name = "gnn_supervised"

    def __init__(
        self,
        seed: int,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._seed = seed
        self._config = config or ModelConfig()
        self._device = device or select_device()
        self._model: GnnModel | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._state: torch.Tensor | None = None

    def fit(self, train: CandidateSet) -> None:
        """Train the classifier on the labelled training span."""
        if train.graph is None:
            raise ValueError("the candidate set carries no graph; the network needs one")
        positives = int(train.labels.sum())
        if positives == 0:
            raise ValueError(
                "the supervised baseline needs labelled incidents in the training span; "
                "use a dataset recipe with train_patterns, such as small-history.yaml"
            )

        seed_torch(self._seed)
        arrays, mean, std = candidate_arrays(train)
        self._mean, self._std = mean, std

        tensors = graph_tensors(train.graph, device=self._device)
        inputs = node_input_features(tensors)
        model = GnnModel(edge_dim=arrays.features.shape[1], config=self._config).to(self._device)
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        usable = np.flatnonzero((arrays.src >= 0) & (arrays.dst >= 0))
        index = torch.as_tensor(usable, device=self._device)
        target = torch.as_tensor(
            train.labels[usable].astype(np.float32), device=self._device
        )
        weight = torch.tensor(
            [max(len(usable) - positives, 1) / max(positives, 1)], device=self._device
        )

        features = torch.as_tensor(arrays.features, device=self._device)
        relation = torch.as_tensor(arrays.relation, device=self._device)
        level = torch.as_tensor(arrays.level, device=self._device)
        src = torch.as_tensor(arrays.src, device=self._device)
        dst = torch.as_tensor(arrays.dst, device=self._device)

        best_state = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        waited = 0

        for _ in range(self._config.epochs):
            model.train()
            h = model.encoder(inputs, tensors)
            logits = model.likelihood(
                h[src[index]], h[dst[index]], relation[index], level[index], features[index]
            )
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            current = float(loss)
            if current < best_loss - 1e-4:
                best_loss, waited = current, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= self._config.patience:
                    break

        model.load_state_dict(best_state)
        self._model = model.eval()
        with torch.no_grad():
            self._state = self._model.encoder(inputs, tensors)

    def score(self, candidates: CandidateSet) -> np.ndarray:
        """Probability the classifier assigns to a change being an incident."""
        if self._model is None or self._state is None:
            raise RuntimeError("the supervised gnn scorer must be fit before scoring")

        arrays, _, _ = candidate_arrays(candidates, mean=self._mean, std=self._std)
        padded = torch.cat(
            [self._state, torch.zeros(1, self._state.shape[1], device=self._device)]
        )
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
```

- [ ] **Step 4: Teach the factory the second name**

In `src/rga/eval/experiment.py`, inside `build_scorer`, after the `gnn` branch:

```python
    if name == "gnn_supervised":
        from rga.nn.supervised import SupervisedGnnScorer

        return SupervisedGnnScorer(seed=seed)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/nn -q`

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/rga/nn/supervised.py src/rga/eval/experiment.py tests/nn/test_supervised.py
git commit -m "feat: add the supervised contrast baseline"
git push origin main
```

---

### Task 15: The comparison run, and the acceptance gate

The first run where the network is measured against the baselines on the same data,
the same seeds and the same metrics. This is the task that decides whether the module
succeeded, so its acceptance criteria are written down before it starts:

1. Mean PR-AUC of `gnn` over five seeds is **above `isolation_forest`** on the same
   run. The published baseline figure is 0.642 ± 0.116 on `small.yaml`; the number to
   beat is whatever `isolation_forest` scores on `small-history.yaml` in this very
   run, not the published one.
2. `gnn` recall at 50 on `privileged_group_join` is **above the rules' 0.10** — the
   pattern hand-written conditions are blind to.
3. `gnn_supervised` scores higher than `gnn` on the five patterns it was trained on.
   If it does not, it is not a credible upper bound and something is wrong with it.

If (1) fails, the diagnosis order is: check that training actually converged (the
validation likelihood should improve over epochs), then raise `epochs`, then widen
`hidden_dim` to 128. **Stop after three such attempts** and report the numbers as
they are — a module that eats the schedule leaves nothing for the interface, and an
honest negative result is a defensible thesis chapter.

**Files:**
- Create: `configs/experiments/gnn.yaml`, `configs/experiments/gnn-full.yaml`
- Create (generated): `docs/thesis/gnn.md`
- Modify: `docs/thesis/README.md`

**Interfaces:**
- Consumes: `build_scorer` names `gnn` and `gnn_supervised` (Tasks 13, 14).
- Produces: `experiments/runs/gnn/results.{json,md}` and `docs/thesis/gnn.md`.

- [ ] **Step 1: Write the experiment recipes**

```yaml
# configs/experiments/gnn.yaml
# The comparison the module exists for. Every scorer sees the same dataset, the
# same seeds and the same candidates; the only thing that differs is what it does
# with them. Runs on the laptop CPU in minutes.
name: gnn
dataset: configs/generator/small-history.yaml
seeds: [1, 2, 3, 4, 5]
scorers:
  - rules
  - isolation_forest
  - lof
  - gnn
  - gnn_supervised
ks: [20, 50, 100]
```

```yaml
# configs/experiments/gnn-full.yaml
# The same comparison at full size, for the training PC. Not runnable in
# reasonable time on a laptop.
name: gnn-full
dataset: configs/generator/default-history.yaml
seeds: [1, 2, 3, 4, 5]
scorers:
  - rules
  - isolation_forest
  - lof
  - gnn
  - gnn_supervised
ks: [20, 50, 100]
```

- [ ] **Step 2: Run the comparison**

Run: `uv run rga evaluate --config configs/experiments/gnn.yaml --out experiments/runs/gnn`

Expected: a table with five scorers. Note how long it takes; if a single seed runs
longer than about ten minutes on the laptop, drop `epochs` to 100 in
`src/rga/nn/config.py` before iterating further, and say so in the commit message.

- [ ] **Step 3: Check the run against the acceptance criteria**

Read the printed table. Write down, in the commit message of Step 5, the three
numbers the criteria above name: `gnn` PR-AUC against `isolation_forest` PR-AUC,
`gnn` recall at 50 on `privileged_group_join`, and `gnn_supervised` against `gnn` on
the five trained patterns. If a criterion fails, follow the diagnosis order in the
task preamble; do not move on with a failing criterion unquestioned and do not adjust
a criterion to fit the result.

- [ ] **Step 4: Publish the table**

```bash
uv run python -c "
from pathlib import Path
source = Path('experiments/runs/gnn/results.md').read_text(encoding='utf-8')
Path('docs/thesis/gnn.md').write_text(source, encoding='utf-8', newline='\n')
print('wrote docs/thesis/gnn.md')
"
```

Add two rows to the table in `docs/thesis/README.md`, matching the existing style:

```markdown
| `gnn.md` | Сеть против бейзлайнов, пять сидов, один и тот же датасет | `uv run rga evaluate --config configs/experiments/gnn.yaml --out experiments/runs/gnn` |
| `gradcheck.md` | Сверка аналитических градиентов с конечными разностями | `uv run python scripts/gradient_check.py` |
```

- [ ] **Step 5: Commit**

`experiments/runs/` is in `.gitignore`: a run is reproducible from its recipe, so the
published table is committed and the raw output is not.

```bash
git add configs/experiments/gnn.yaml configs/experiments/gnn-full.yaml docs/thesis/gnn.md docs/thesis/README.md
git commit -m "feat: compare the graph network against the baselines"
git push origin main
```

---

### Task 16: The hidden-pattern contrast table

The main result of the module deserves its own table rather than being read out of a
sixteen-column per-pattern grid. One row per scorer, one column for the five patterns
the supervised baseline trained on, one for the three it never saw, and the gap
between them.

**Files:**
- Modify: `src/rga/eval/experiment.py` (add `format_hidden_pattern_table`, call it from `save_results`)
- Test: `tests/eval/test_hidden_patterns.py`

**Interfaces:**
- Consumes: `ExperimentResult.per_pattern` (existing).
- Produces: `KNOWN_PATTERNS`, `HIDDEN_PATTERNS`, `format_hidden_pattern_table(result) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_hidden_patterns.py
"""The table that carries the module's main result."""

from pathlib import Path

from rga.eval.experiment import (
    ExperimentConfig,
    ExperimentResult,
    format_hidden_pattern_table,
)


def _result() -> ExperimentResult:
    config = ExperimentConfig(
        name="demo",
        dataset=Path("configs/generator/small-history.yaml"),
        seeds=(1,),
        scorers=("gnn", "gnn_supervised"),
        ks=(50,),
        feature_groups=None,
    )
    return ExperimentResult(
        config=config,
        rows=(),
        per_pattern={
            "gnn": {
                "self_grant_admin": 1.0,
                "privileged_group_join": 0.8,
                "grant_burst": 0.9,
                "hierarchy_bypass": 1.0,
                "cross_department": 0.8,
                "dormant_awakening": 0.9,
                "shadow_group": 0.9,
                "delegation_cascade": 0.9,
            },
            "gnn_supervised": {
                "self_grant_admin": 1.0,
                "privileged_group_join": 1.0,
                "grant_burst": 1.0,
                "hierarchy_bypass": 1.0,
                "cross_department": 1.0,
                "dormant_awakening": 0.1,
                "shadow_group": 0.2,
                "delegation_cascade": 0.0,
            },
        },
    )


def test_the_table_names_both_groups() -> None:
    table = format_hidden_pattern_table(_result())

    assert "known" in table
    assert "hidden" in table


def test_a_model_that_generalises_shows_a_small_gap() -> None:
    table = format_hidden_pattern_table(_result())
    row = next(line for line in table.splitlines() if line.startswith("| gnn |"))

    assert "0.90" in row  # both halves average to 0.90
    assert "+0.00" in row


def test_a_model_that_memorised_shows_a_large_gap() -> None:
    table = format_hidden_pattern_table(_result())
    row = next(line for line in table.splitlines() if line.startswith("| gnn_supervised |"))

    assert "1.00" in row
    assert "0.10" in row
    assert "+0.90" in row
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/eval/test_hidden_patterns.py -v`

Expected: FAIL with `ImportError: cannot import name 'format_hidden_pattern_table'`.

- [ ] **Step 3: Add the table**

In `src/rga/eval/experiment.py`, next to the other formatters:

```python
#: The five patterns the supervised baseline is trained on.
KNOWN_PATTERNS = (
    "self_grant_admin",
    "privileged_group_join",
    "grant_burst",
    "hierarchy_bypass",
    "cross_department",
)
#: The three it never sees. How a model does here is what section 9 is after.
HIDDEN_PATTERNS = ("dormant_awakening", "shadow_group", "delegation_cascade")


def format_hidden_pattern_table(result: ExperimentResult) -> str:
    """Recall on trained patterns against recall on unseen ones.

    A model that learned the nature of an anomaly keeps its recall on patterns it
    has never seen. A model that learned the generator loses it, and the gap column
    measures exactly that.
    """
    lines = [
        f"### Recall at {_PATTERN_K}: known patterns against hidden ones",
        "",
        "| scorer | known | hidden | gap |",
        "|---|---|---|---|",
    ]
    for scorer in sorted(result.per_pattern):
        found = result.per_pattern[scorer]
        known = np.nanmean([found.get(name, 0.0) for name in KNOWN_PATTERNS])
        hidden = np.nanmean([found.get(name, 0.0) for name in HIDDEN_PATTERNS])
        lines.append(
            f"| {scorer} | {known:.2f} | {hidden:.2f} | {known - hidden:+.2f} |"
        )

    lines += [
        "",
        "Известными считаются паттерны 1-5 из раздела 5.3, на которых обучается",
        "супервизорный вариант; скрытыми — паттерны 6-8, которых он не видел.",
        "Самообучаемая модель не видит меток вообще, поэтому для неё все восемь",
        "одинаково незнакомы, и разрыв у неё должен быть близок к нулю.",
    ]
    return "\n".join(lines)
```

and append it in `save_results`, right after the ablation block:

```python
    table = format_results_table(result)
    if {str(row["groups"]) for row in result.rows} != {"all"}:
        table += "\n\n" + format_ablation_table(result)
    if result.per_pattern:
        table += "\n\n" + format_hidden_pattern_table(result)
    (path / "results.md").write_text(table + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/eval -q`

Expected: green.

- [ ] **Step 5: Regenerate the table with the new section**

Run: `uv run rga evaluate --config configs/experiments/gnn.yaml --out experiments/runs/gnn`

then republish `docs/thesis/gnn.md` exactly as in Task 15 Step 4.

- [ ] **Step 6: Commit**

```bash
git add src/rga/eval/experiment.py tests/eval/test_hidden_patterns.py docs/thesis/gnn.md
git commit -m "feat: report recall on known patterns against hidden ones"
git push origin main
```

---

### Task 17: The run on the training PC

**This task runs on the Windows machine with the RTX 5060 Ti, not on the laptop.**
Everything before it is CPU work and must already be green. Ask the author to run
these commands and paste the output back; do not try to run them here.

The card is Blackwell, `sm_120`. A torch build for CUDA 12.1 or older will import
fine and then fail on the first kernel, so the probe comes first and nothing else
runs until it passes.

**Files:**
- Create (generated): `docs/thesis/gnn-full.md`

- [ ] **Step 1: Install the GPU dependency group**

```powershell
uv sync --extra gpu
```

Expected: torch resolved from the `pytorch-cu128` index. If uv reports a conflict
with the `cpu` extra, the environment has both — remove `.venv` and re-run.

- [ ] **Step 2: Probe the card**

```powershell
uv run python scripts/gpu_smoke.py
```

Expected: the device reports as an RTX 5060 Ti with capability 12.0 and a matrix
multiply completes. This is the check the design document's risk section calls for and
the only item still open from Module 1. **If it fails, stop and report the exact
error** — the whole module's training plan depends on the answer, and a wrong torch
build is the most likely cause.

- [ ] **Step 3: Repeat the CPU comparison to confirm nothing moved**

```powershell
uv run pytest -m "not integration and not gpu"
uv run rga evaluate --config configs/experiments/gnn.yaml --out experiments/runs/gnn-pc
```

Expected: the same numbers as Task 15, within the noise of a different platform. A
large difference means something depends on the platform that should not — report it
rather than picking the nicer table.

- [ ] **Step 4: Run the full-size comparison**

```powershell
uv run rga evaluate --config configs/experiments/gnn-full.yaml --out experiments/runs/gnn-full
```

Expected: the same five scorers on the full-size dataset. This is the table the
thesis reports.

- [ ] **Step 5: Publish and commit from the PC**

```powershell
uv run python -c "from pathlib import Path; Path('docs/thesis/gnn-full.md').write_text(Path('experiments/runs/gnn-full/results.md').read_text(encoding='utf-8'), encoding='utf-8', newline='\n')"
git add docs/thesis/gnn-full.md
git commit -m "feat: add the full-size comparison from the training PC"
git push origin main
```

---

## Self-Review

Checked against the spec before handing over.

**Spec coverage.** Section 7.1 is Task 6, 7.2 is Task 9, 7.3 is Tasks 10, 11 and 7,
7.4 is Tasks 1, 4 and 17. Section 3's score combination is Task 12. Section 8's
supervised variant is Task 14. Section 9's per-pattern breakdown is Task 16 and the
five-seed protocol is Task 15.

**Deliberately left out of this module.** Section 9 also lists series that vary the
anomaly share, the graph size and the training-set contamination, and an ablation over
the number of layers and the negative-sampling strategies. They are cheap once Task 15
runs — each is another experiment recipe — but they are not what makes the module
defensible, and Module 4 has a service, an interface and a live integration to build
in the remaining time. Add them after Module 4 is standing, or when Task 15's numbers
raise a question one of them would answer.

**One thing this plan changes that the spec assumed otherwise.** The spec says the
network scores an edge from the graph; it does not say the likelihood head also reads
the candidate feature row. Task 9 argues why it must: without it the network would be
insensitive to the source's capability level and the feature-group study would have
nothing to say about it. This is recorded here rather than silently done.
