# Module 1: Data and Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data foundation — a canonical access-graph domain model, a portable `GraphSource` contract, a synthetic organization generator with eight labelled anomaly patterns, and a Neo4j adapter for the live `opens3-rebac` engine.

**Architecture:** Everything upstream is reduced to one stream of `GraphEvent`. A graph is the replay of that stream up to a timestamp. Sources declare capability levels (snapshot / timestamps / provenance) and feature groups they cannot fill are masked rather than faked. Nothing above the adapter layer knows whether data came from the generator or from a live authorization engine.

**Tech Stack:** Python 3.12 (pinned via uv), NumPy, SciPy, NetworkX, PyYAML, pytest, ruff. Torch enters in Module 3 but its install path is validated here.

**Spec:** `docs/superpowers/specs/2026-09-12-rebac-graph-anomaly-design.md`

## Global Constraints

- Python 3.12 exactly, pinned in `.python-version` and `pyproject.toml`. The laptop's system Python is 3.14 and has no torch wheels; never invoke `python3` directly, always `uv run`.
- Torch, when installed, must be `>=2.7` from the `cu128` index for the `gpu` extra. The training GPU is RTX 5060 Ti (Blackwell, sm_120) and will not run cu121 or older builds.
- No PyTorch Geometric, no DGL, ever. Graph layers are our own.
- Cross-platform: development on Arch Linux, training on Windows 11. Paths via `pathlib` only, no symlinks, every entry point guarded by `if __name__ == "__main__":` because Windows spawns rather than forks.
- Code, comments, docstrings and commit messages in English. Commit messages short, no emoji, no AI attribution lines.
- Every commit is pushed: `git push origin main`. No feature branches in this repo.
- Changes to `/home/wexel/Data/Code/Projects/opens3-rebac` are the sole exception: that is a shared team repository, so they go on a branch and through a pull request, never directly to its `main`.
- Every random draw goes through an explicitly seeded `numpy.random.Generator`. No module-level `random` calls, no unseeded defaults.
- Entity id format is fixed by the live system and must not be "improved": `user:<uuid>`, `group:<name>`, `bucket:<name>`, `object:<bucket>/<key>`.

---

## File Structure

**Created in this module:**

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, cpu/gpu extras, tool config |
| `src/rga/util/device.py` | Torch device selection and description |
| `src/rga/util/timeutil.py` | Millisecond time helpers, working-hours sampling |
| `src/rga/domain/entities.py` | Entity ids, types, bucket extraction |
| `src/rga/domain/relations.py` | Canonical relation types and ordinal permission levels |
| `src/rga/domain/events.py` | `GraphEvent`, `EventOp`, validation, dict conversion |
| `src/rga/domain/graph.py` | `AccessGraph` arrays, `GraphBuilder`, CSR adjacency |
| `src/rga/domain/replay.py` | Journal → graph at a point in time |
| `src/rga/io/jsonl.py` | Event journal read/write |
| `src/rga/io/dataset_io.py` | Dataset directory layout, save/load |
| `src/rga/adapters/base.py` | `Capabilities`, `GraphSource` protocol |
| `src/rga/adapters/mapping.py` | Declarative source-relation → canonical mapping |
| `src/rga/adapters/file_source.py` | Source backed by a journal file |
| `src/rga/adapters/neo4j_source.py` | Source backed by a live opens3-rebac graph |
| `src/rga/generator/config.py` | Typed configuration loaded from YAML |
| `src/rga/generator/org.py` | Static organization structure |
| `src/rga/generator/timeline.py` | Growth process producing the normal journal |
| `src/rga/generator/anomalies/base.py` | Pattern protocol, labels, registry, shared helpers |
| `src/rga/generator/anomalies/escalation.py` | Patterns 1–2 |
| `src/rga/generator/anomalies/compromise.py` | Patterns 3, 6 |
| `src/rga/generator/anomalies/bypass.py` | Patterns 4–5 |
| `src/rga/generator/anomalies/persistence.py` | Patterns 7–8 |
| `src/rga/generator/dataset.py` | Assembly, temporal split, anomaly injection |
| `src/rga/generator/stats.py` | Dataset statistics |
| `src/rga/cli/main.py` | `generate`, `stats`, `inspect` commands |
| `scripts/gpu_smoke.py` | Standalone GPU compatibility probe |
| `configs/generator/*.yaml` | Named dataset configurations |

**Modified in the `opens3-rebac` repository (branch + PR):**

| Path | Change |
|---|---|
| `shared/api/authz/v1/authz.proto` | `actor` field on write and delete requests |
| `services/authz/internal/types.py` | `actor` on `Tuple` |
| `services/authz/internal/repositories/neo4j/store.py` | Timestamps and actor on nodes and edges |
| `services/authz/internal/repositories/kafka/producer.py` | Actor in audit events |
| `services/authz/entrypoints/server/servicer.py` | Pass actor through |

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/rga/__init__.py`, `tests/test_smoke.py`, `.github/workflows/ci.yml`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing
- Produces: package `rga` importable as `from rga import __version__`; test command `uv run pytest`; lint command `uv run ruff check .`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
"""The package imports and the test harness runs."""


def test_package_exposes_version() -> None:
    from rga import __version__

    assert isinstance(__version__, str)
    assert __version__.count(".") >= 1
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga'`

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "rga"
version = "0.1.0"
description = "Anomaly detection in ReBAC access graphs"
requires-python = "==3.12.*"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "networkx>=3.2",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
cpu = ["torch>=2.7"]
gpu = ["torch>=2.7"]
neo4j = ["neo4j>=5.0"]
kafka = ["confluent-kafka>=2.5"]
ml = ["scikit-learn>=1.4"]
service = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]

[project.scripts]
rga = "rga.cli.main:main"

[dependency-groups]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6"]

# Torch ships per-accelerator wheels on its own indexes. `explicit = true` means
# nothing is pulled from them unless a source below points at them by name.
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu128", extra = "gpu" },
]

# The laptop installs `cpu`, the training PC installs `gpu`; never both.
[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "gpu" }]]

[tool.hatch.build.targets.wheel]
packages = ["src/rga"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = [
    "integration: requires external services (run with -m integration)",
    "gpu: requires a CUDA device",
]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]
```

- [ ] **Step 4: Write `.python-version` and `.gitignore`**

`.python-version` contains exactly one line:

```
3.12
```

`.gitignore` — the existing file already ignores `issues/`, keep that line and append:

```
issues/

__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/

data/
experiments/runs/
*.npz
*.parquet
```

- [ ] **Step 5: Write the package init**

```python
# src/rga/__init__.py
"""Anomaly detection in ReBAC access graphs."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Create the environment and run the test**

Run:
```bash
uv sync --extra cpu
uv run pytest tests/test_smoke.py -v
```
Expected: PASS

- [ ] **Step 7: Add CI**

```yaml
# .github/workflows/ci.yml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --extra cpu
      - run: uv run ruff check .
      - run: uv run pytest -m "not integration and not gpu"
```

- [ ] **Step 8: Verify lint passes**

Run: `uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 9: Commit and push**

```bash
git add pyproject.toml uv.lock .python-version .gitignore src tests .github
git commit -m "chore: scaffold project with uv, pytest and ruff"
git push origin main
```

---

## Task 2: Device selection and GPU probe

This task exists to retire the single largest technical risk in the project — that
Blackwell sm_120 is unsupported by the installed torch build — before anything
depends on it. Torch is imported lazily so the package stays importable without it.

**Files:**
- Create: `src/rga/util/__init__.py`, `src/rga/util/device.py`, `scripts/gpu_smoke.py`
- Test: `tests/util/test_device.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `select_device(prefer: str = "auto") -> str` returning `"cuda"` or `"cpu"`
  - `describe_device() -> dict[str, object]` with keys `torch`, `cuda_available`, `device`, `capability`, `name`, `supported`

- [ ] **Step 1: Write the failing test**

```python
# tests/util/test_device.py
"""Device description must work on a machine with no CUDA at all."""

import pytest

from rga.util.device import describe_device, select_device

torch = pytest.importorskip("torch")


def test_describe_device_reports_expected_keys() -> None:
    info = describe_device()
    assert set(info) == {"torch", "cuda_available", "device", "capability", "name", "supported"}
    assert isinstance(info["torch"], str)
    assert isinstance(info["cuda_available"], bool)


def test_select_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == "cpu"
    assert select_device("cuda") == "cpu"


def test_select_device_rejects_unknown_preference() -> None:
    with pytest.raises(ValueError, match="unknown device preference"):
        select_device("tpu")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/util/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.util'`

- [ ] **Step 3: Implement the module**

```python
# src/rga/util/device.py
"""Torch device selection.

Torch is imported lazily: the data and generator layers of this project run
without it, and the laptop used for development installs the CPU build only.
"""

from __future__ import annotations

# Minimum compute capability of the training GPU (RTX 5060 Ti, Blackwell).
# Wheels built against CUDA 12.1 and earlier do not contain sm_120 kernels and
# fail at the first kernel launch rather than at import time.
_TARGET_CAPABILITY = (12, 0)


def select_device(prefer: str = "auto") -> str:
    """Return the torch device string to use.

    `auto` picks CUDA when it is usable, `cuda` degrades to CPU rather than
    raising so the same command line works on both machines, and `cpu` forces
    the CPU path.
    """
    if prefer not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"unknown device preference: {prefer!r}")
    if prefer == "cpu":
        return "cpu"

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device() -> dict[str, object]:
    """Collect everything needed to diagnose a broken CUDA install."""
    import torch

    available = bool(torch.cuda.is_available())
    capability: tuple[int, int] | None = None
    name: str | None = None
    if available:
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)

    return {
        "torch": torch.__version__,
        "cuda_available": available,
        "device": select_device("auto"),
        "capability": capability,
        "name": name,
        "supported": capability is not None and capability >= _TARGET_CAPABILITY,
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/util/test_device.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the standalone probe script**

```python
# scripts/gpu_smoke.py
"""Verify that the installed torch build actually runs on this machine's GPU.

Run on the training PC after `uv sync --extra gpu`:

    uv run python scripts/gpu_smoke.py

A build without sm_120 kernels imports cleanly and reports CUDA as available,
then fails on the first real kernel launch. The matmul below is that launch.
"""

from __future__ import annotations

import sys

from rga.util.device import describe_device, select_device


def main() -> int:
    info = describe_device()
    for key, value in info.items():
        print(f"{key:16}: {value}")

    device = select_device("auto")
    if device == "cpu":
        print("\nNo CUDA device. This is expected on the development laptop.")
        return 0

    import torch

    print("\nRunning a matmul on the device...")
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    c = a @ b
    torch.cuda.synchronize()
    print(f"OK: result sum = {float(c.sum()):.4f}")

    if not info["supported"]:
        print("\nWARNING: compute capability is below the expected sm_120.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the probe on the laptop**

Run: `uv run python scripts/gpu_smoke.py`
Expected: prints the torch version, `cuda_available: False`, `device: cpu`, and exits 0 with the "expected on the development laptop" note.

- [ ] **Step 7: Run the probe on the Windows training PC**

Run, in the repository checkout on the PC:
```
uv sync --extra gpu
uv run python scripts/gpu_smoke.py
```
Expected: `cuda_available: True`, `capability: (12, 0)`, `supported: True`, and the matmul prints a finite sum.

If instead it raises `CUDA error: no kernel image is available for execution on the device`, the installed wheel predates sm_120 support: check that `uv` resolved torch from the `pytorch-cu128` index and that the installed version is at least 2.7. **Do not continue past this step with a failing probe** — every later module depends on it.

- [ ] **Step 8: Commit and push**

```bash
git add src/rga/util scripts/gpu_smoke.py tests/util
git commit -m "feat: add device selection and GPU compatibility probe"
git push origin main
```

---

## Task 3: Entity and relation vocabulary

**Files:**
- Create: `src/rga/domain/__init__.py`, `src/rga/domain/entities.py`, `src/rga/domain/relations.py`
- Test: `tests/domain/test_entities.py`, `tests/domain/test_relations.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `EntityType` IntEnum: `USER=1, GROUP=2, BUCKET=3, OBJECT=4`
  - `entity_type(entity_id: str) -> EntityType`
  - `parent_bucket(object_id: str) -> str`
  - `RelationType` IntEnum: `MEMBER_OF=1, HAS_PERMISSION=2, PARENT_OF=3, OWNER_OF=4`
  - `PermissionLevel` IntEnum: `NONE=0, READ=1, WRITE=2, CREATE=3, DELETE=4, ADMIN=5`
  - `parse_level(name: str | None) -> PermissionLevel`
  - `parse_relation(name: str) -> RelationType`
  - `LEVEL_CARRYING: frozenset[RelationType]`

Enum values are stable identifiers: they index model parameter tensors in Module 3
and are written into saved datasets. Never renumber them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/domain/test_entities.py
"""Entity id parsing. Formats are fixed by the live opens3-rebac engine."""

import pytest

from rga.domain.entities import EntityType, entity_type, parent_bucket


@pytest.mark.parametrize(
    ("entity_id", "expected"),
    [
        ("user:550e8400-e29b-41d4-a716-446655440000", EntityType.USER),
        ("group:devops", EntityType.GROUP),
        ("bucket:my-photos", EntityType.BUCKET),
        ("object:my-photos/2024/cat.jpg", EntityType.OBJECT),
    ],
)
def test_entity_type_recognises_every_prefix(entity_id: str, expected: EntityType) -> None:
    assert entity_type(entity_id) is expected


@pytest.mark.parametrize("bad", ["", "user", "user:", ":alice", "resource:r1"])
def test_entity_type_rejects_malformed_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        entity_type(bad)


def test_parent_bucket_extracts_the_containing_bucket() -> None:
    assert parent_bucket("object:my-photos/2024/cat.jpg") == "bucket:my-photos"


def test_parent_bucket_handles_keys_without_slashes_in_the_name() -> None:
    assert parent_bucket("object:logs/app.log") == "bucket:logs"


def test_parent_bucket_rejects_non_objects() -> None:
    with pytest.raises(ValueError, match="not an object id"):
        parent_bucket("bucket:my-photos")
```

```python
# tests/domain/test_relations.py
"""Relation and permission vocabulary."""

import pytest

from rga.domain.relations import (
    LEVEL_CARRYING,
    PermissionLevel,
    RelationType,
    parse_level,
    parse_relation,
)


def test_permission_levels_are_ordered() -> None:
    assert PermissionLevel.READ < PermissionLevel.WRITE < PermissionLevel.ADMIN
    assert PermissionLevel.NONE < PermissionLevel.READ


def test_parse_level_accepts_engine_spelling() -> None:
    assert parse_level("admin") is PermissionLevel.ADMIN
    assert parse_level("READ") is PermissionLevel.READ


def test_parse_level_maps_absence_to_none() -> None:
    assert parse_level(None) is PermissionLevel.NONE
    assert parse_level("") is PermissionLevel.NONE


def test_parse_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown permission level"):
        parse_level("superuser")


def test_parse_relation_accepts_engine_spelling() -> None:
    assert parse_relation("HAS_PERMISSION") is RelationType.HAS_PERMISSION
    assert parse_relation("member_of") is RelationType.MEMBER_OF


def test_only_has_permission_carries_a_level() -> None:
    assert LEVEL_CARRYING == frozenset({RelationType.HAS_PERMISSION})


def test_enum_values_are_stable() -> None:
    # These integers are written into saved datasets and index model parameters.
    assert (RelationType.MEMBER_OF, RelationType.HAS_PERMISSION) == (1, 2)
    assert (RelationType.PARENT_OF, RelationType.OWNER_OF) == (3, 4)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest tests/domain -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.domain'`

- [ ] **Step 3: Implement `entities.py`**

```python
# src/rga/domain/entities.py
"""Entity identifiers.

The textual format is dictated by the live opens3-rebac engine and is reproduced
verbatim so that a graph read from it needs no translation:

    user:<uuid>              group:<name>
    bucket:<name>            object:<bucket>/<key>
"""

from __future__ import annotations

from enum import IntEnum


class EntityType(IntEnum):
    """Node kinds. Values are stable and index model parameters."""

    USER = 1
    GROUP = 2
    BUCKET = 3
    OBJECT = 4


_PREFIX_TO_TYPE = {
    "user": EntityType.USER,
    "group": EntityType.GROUP,
    "bucket": EntityType.BUCKET,
    "object": EntityType.OBJECT,
}


def entity_type(entity_id: str) -> EntityType:
    """Infer the node kind from an entity id prefix."""
    prefix, separator, rest = entity_id.partition(":")
    if not separator or not rest:
        raise ValueError(f"malformed entity id: {entity_id!r}")
    try:
        return _PREFIX_TO_TYPE[prefix]
    except KeyError:
        raise ValueError(f"unknown entity prefix: {prefix!r}") from None


def parent_bucket(object_id: str) -> str:
    """Return the bucket containing an object.

    The key may itself contain slashes; only the first segment is the bucket.
    """
    if entity_type(object_id) is not EntityType.OBJECT:
        raise ValueError(f"not an object id: {object_id!r}")
    path = object_id.split(":", 1)[1]
    bucket, separator, _ = path.partition("/")
    if not separator or not bucket:
        raise ValueError(f"object id has no bucket separator: {object_id!r}")
    return f"bucket:{bucket}"
```

- [ ] **Step 4: Implement `relations.py`**

```python
# src/rga/domain/relations.py
"""Canonical relation types and permission levels.

Permission levels are deliberately an ordered scale rather than a categorical
set: the engine defines admin > delete > create > write > read, and the distance
between two levels carries meaning the model should be able to use.
"""

from __future__ import annotations

from enum import IntEnum


class RelationType(IntEnum):
    """Canonical edge types. Values are stable and index model parameters."""

    MEMBER_OF = 1
    HAS_PERMISSION = 2
    PARENT_OF = 3
    OWNER_OF = 4


class PermissionLevel(IntEnum):
    """Ordered permission levels. NONE marks a relation that carries no level."""

    NONE = 0
    READ = 1
    WRITE = 2
    CREATE = 3
    DELETE = 4
    ADMIN = 5


#: Only HAS_PERMISSION carries a level; every other relation must use NONE.
LEVEL_CARRYING = frozenset({RelationType.HAS_PERMISSION})

_LEVEL_BY_NAME = {
    level.name.lower(): level for level in PermissionLevel if level is not PermissionLevel.NONE
}
_RELATION_BY_NAME = {relation.name.lower(): relation for relation in RelationType}


def parse_level(name: str | None) -> PermissionLevel:
    """Convert the engine's level spelling to the ordered enum."""
    if not name:
        return PermissionLevel.NONE
    try:
        return _LEVEL_BY_NAME[name.lower()]
    except KeyError:
        raise ValueError(f"unknown permission level: {name!r}") from None


def parse_relation(name: str) -> RelationType:
    """Convert the engine's relation spelling to the canonical enum."""
    try:
        return _RELATION_BY_NAME[name.lower()]
    except KeyError:
        raise ValueError(f"unknown relation type: {name!r}") from None
```

- [ ] **Step 5: Create the package init**

```python
# src/rga/domain/__init__.py
"""Domain model: entities, relations, events, graph."""
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/domain -v`
Expected: PASS (all parametrised cases green)

- [ ] **Step 7: Commit and push**

```bash
git add src/rga/domain tests/domain
git commit -m "feat: add entity and relation vocabulary"
git push origin main
```

---

## Task 4: Graph events and the journal file

**Files:**
- Create: `src/rga/domain/events.py`, `src/rga/io/__init__.py`, `src/rga/io/jsonl.py`
- Test: `tests/domain/test_events.py`, `tests/io/test_jsonl.py`

**Interfaces:**
- Consumes: `RelationType`, `PermissionLevel`, `parse_level`, `parse_relation` from Task 3
- Produces:
  - `EventOp` IntEnum: `GRANT=1, REVOKE=2`
  - `GraphEvent` frozen dataclass with fields `ts: int`, `op: EventOp`, `subject: str`, `relation: RelationType`, `object: str`, `level: PermissionLevel = PermissionLevel.NONE`, `actor: str | None = None`
  - `GraphEvent.edge_key() -> tuple[str, int, str]`
  - `GraphEvent.to_dict() -> dict[str, object]` / `GraphEvent.from_dict(payload) -> GraphEvent`
  - `write_events(path: Path, events: Iterable[GraphEvent]) -> int`
  - `read_events(path: Path) -> Iterator[GraphEvent]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/domain/test_events.py
"""Graph events: the single unit of input for the whole system."""

import pytest

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType


def _grant(**overrides: object) -> GraphEvent:
    base: dict[str, object] = {
        "ts": 1_700_000_000_000,
        "op": EventOp.GRANT,
        "subject": "user:alice",
        "relation": RelationType.HAS_PERMISSION,
        "object": "bucket:photos",
        "level": PermissionLevel.READ,
    }
    base.update(overrides)
    return GraphEvent(**base)  # type: ignore[arg-type]


def test_has_permission_grant_requires_a_level() -> None:
    with pytest.raises(ValueError, match="requires a level"):
        _grant(level=PermissionLevel.NONE)


def test_other_relations_must_not_carry_a_level() -> None:
    with pytest.raises(ValueError, match="must not carry a level"):
        _grant(relation=RelationType.MEMBER_OF, object="group:devops")


def test_revoke_does_not_need_a_level() -> None:
    event = _grant(op=EventOp.REVOKE, level=PermissionLevel.NONE)
    assert event.level is PermissionLevel.NONE


def test_edge_key_ignores_level_and_actor() -> None:
    read = _grant(level=PermissionLevel.READ, actor="user:root")
    admin = _grant(level=PermissionLevel.ADMIN, actor="user:alice")
    assert read.edge_key() == admin.edge_key()
    assert read.edge_key() == ("user:alice", int(RelationType.HAS_PERMISSION), "bucket:photos")


def test_round_trips_through_a_dict() -> None:
    event = _grant(level=PermissionLevel.ADMIN, actor="user:root")
    assert GraphEvent.from_dict(event.to_dict()) == event


def test_dict_form_omits_absent_optional_fields() -> None:
    event = _grant(relation=RelationType.MEMBER_OF, object="group:devops",
                   level=PermissionLevel.NONE)
    payload = event.to_dict()
    assert "level" not in payload
    assert "actor" not in payload


def test_dict_form_is_human_readable() -> None:
    payload = _grant().to_dict()
    assert payload["op"] == "grant"
    assert payload["relation"] == "HAS_PERMISSION"
    assert payload["level"] == "read"
```

```python
# tests/io/test_jsonl.py
"""Journal files must round-trip identically on both platforms."""

from pathlib import Path

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.io.jsonl import read_events, write_events


def _events() -> list[GraphEvent]:
    return [
        GraphEvent(1_000, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(2_000, EventOp.GRANT, "group:devops", RelationType.HAS_PERMISSION,
                   "bucket:photos", PermissionLevel.WRITE, actor="user:root"),
        GraphEvent(3_000, EventOp.REVOKE, "user:alice", RelationType.MEMBER_OF, "group:devops"),
    ]


def test_write_then_read_preserves_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    written = write_events(path, _events())
    assert written == 3
    assert list(read_events(path)) == _events()


def test_file_uses_unix_line_endings(tmp_path: Path) -> None:
    # Windows would otherwise write \r\n and make the same dataset differ by platform.
    path = tmp_path / "events.jsonl"
    write_events(path, _events())
    assert b"\r\n" not in path.read_bytes()


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_events(path, _events())
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(list(read_events(path))) == 3
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest tests/domain/test_events.py tests/io -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.domain.events'`

- [ ] **Step 3: Implement `events.py`**

```python
# src/rga/domain/events.py
"""The graph event: one change of access rights.

Every source — the synthetic generator, a live authorization engine, a saved
file — is reduced to a stream of these. Nothing above the adapter layer knows
where they came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from rga.domain.relations import (
    LEVEL_CARRYING,
    PermissionLevel,
    RelationType,
    parse_level,
    parse_relation,
)


class EventOp(IntEnum):
    """Whether the event adds or removes an edge."""

    GRANT = 1
    REVOKE = 2


@dataclass(frozen=True, slots=True)
class GraphEvent:
    """A single change of the access graph.

    `actor` is whoever performed the change, and is the difference between a
    right granted by an administrator and a right a subject granted to itself.
    Most authorization engines do not record it; `None` means unknown, which is
    a distinct state from "nobody".
    """

    ts: int
    op: EventOp
    subject: str
    relation: RelationType
    object: str
    level: PermissionLevel = PermissionLevel.NONE
    actor: str | None = None

    def __post_init__(self) -> None:
        if self.relation in LEVEL_CARRYING:
            if self.op is EventOp.GRANT and self.level is PermissionLevel.NONE:
                raise ValueError(f"{self.relation.name} grant requires a level")
        elif self.level is not PermissionLevel.NONE:
            raise ValueError(f"{self.relation.name} must not carry a level")

    def edge_key(self) -> tuple[str, int, str]:
        """Identity of the edge this event affects, independent of level and actor."""
        return (self.subject, int(self.relation), self.object)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a readable mapping; absent optional fields are omitted."""
        payload: dict[str, object] = {
            "ts": self.ts,
            "op": self.op.name.lower(),
            "subject": self.subject,
            "relation": self.relation.name,
            "object": self.object,
        }
        if self.level is not PermissionLevel.NONE:
            payload["level"] = self.level.name.lower()
        if self.actor is not None:
            payload["actor"] = self.actor
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GraphEvent:
        """Rebuild an event from its mapping form."""
        return cls(
            ts=int(payload["ts"]),  # type: ignore[arg-type]
            op=EventOp[str(payload["op"]).upper()],
            subject=str(payload["subject"]),
            relation=parse_relation(str(payload["relation"])),
            object=str(payload["object"]),
            level=parse_level(payload.get("level")),  # type: ignore[arg-type]
            actor=None if payload.get("actor") is None else str(payload["actor"]),
        )
```

- [ ] **Step 4: Implement `jsonl.py`**

```python
# src/rga/io/jsonl.py
"""Journal storage as newline-delimited JSON.

Chosen over a binary format because a change journal is something a person
reads while debugging a generator or a live integration. Line endings are
forced to "\\n" so that a dataset produced on Windows is byte-identical to one
produced on Linux.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from rga.domain.events import GraphEvent


def write_events(path: Path, events: Iterable[GraphEvent]) -> int:
    """Write events to a journal file, returning how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_events(path: Path) -> Iterator[GraphEvent]:
    """Stream events from a journal file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield GraphEvent.from_dict(json.loads(stripped))
```

- [ ] **Step 5: Create the io package init**

```python
# src/rga/io/__init__.py
"""Reading and writing journals and datasets."""
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/domain/test_events.py tests/io -v`
Expected: PASS

- [ ] **Step 7: Commit and push**

```bash
git add src/rga/domain/events.py src/rga/io tests/domain/test_events.py tests/io
git commit -m "feat: add graph event model and journal file format"
git push origin main
```

---

## Task 5: Access graph and journal replay

The centre of the module. A graph is never constructed directly — it is always the
replay of a journal up to a timestamp, which is what makes a temporal evaluation
split honest.

**Files:**
- Create: `src/rga/domain/graph.py`, `src/rga/domain/replay.py`
- Test: `tests/domain/test_graph.py`, `tests/domain/test_replay.py`

**Interfaces:**
- Consumes: `EntityType`, `entity_type` (Task 3); `GraphEvent`, `EventOp` (Task 4)
- Produces:
  - `AccessGraph` frozen dataclass, fields `node_ids: tuple[str, ...]`, `node_index: dict[str, int]`, and `np.ndarray` fields `node_type` (int8), `node_created` (int64), `edge_src` (int32), `edge_dst` (int32), `edge_rel` (int8), `edge_level` (int8), `edge_created` (int64), `edge_actor` (int32)
  - `AccessGraph.num_nodes -> int`, `AccessGraph.num_edges -> int`
  - `AccessGraph.index_of(entity_id: str) -> int`
  - `AccessGraph.neighbors(entity_id: str, relation: RelationType, *, incoming: bool = False) -> np.ndarray`
  - `GraphBuilder` with `apply(event)`, `register_node(entity_id, created=UNKNOWN)`, `add_edge(subject, relation, target, *, level=0, created=UNKNOWN, actor=None)`, `remove_edge(subject, relation, target)` and `build() -> AccessGraph`
  - `UNKNOWN = -1` sentinel
  - `replay(events: Iterable[GraphEvent], until: int | None = None) -> AccessGraph`

Unknown values are encoded as `-1` throughout: an unknown timestamp and an unknown
actor are distinct from zero and from absence, and Module 2 turns them into masks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/domain/test_graph.py
"""Graph assembly and adjacency."""

import numpy as np
import pytest

from rga.domain.entities import EntityType
from rga.domain.events import EventOp, GraphEvent
from rga.domain.graph import GraphBuilder
from rga.domain.relations import PermissionLevel, RelationType


def _build(*events: GraphEvent):
    builder = GraphBuilder()
    for event in events:
        builder.apply(event)
    return builder.build()


def test_grant_creates_both_endpoints_and_the_edge() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops")
    )
    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    assert graph.node_type[graph.index_of("user:alice")] == EntityType.USER
    assert graph.node_type[graph.index_of("group:devops")] == EntityType.GROUP


def test_node_creation_time_is_when_it_was_first_seen() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(500, EventOp.GRANT, "user:alice", RelationType.HAS_PERMISSION,
                   "bucket:photos", PermissionLevel.READ),
    )
    assert graph.node_created[graph.index_of("user:alice")] == 100
    assert graph.node_created[graph.index_of("bucket:photos")] == 500


def test_revoke_removes_the_edge_but_keeps_the_nodes() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(200, EventOp.REVOKE, "user:alice", RelationType.MEMBER_OF, "group:devops"),
    )
    assert graph.num_edges == 0
    assert graph.num_nodes == 2


def test_regrant_overwrites_level_and_time_without_duplicating() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.HAS_PERMISSION,
                   "bucket:photos", PermissionLevel.READ),
        GraphEvent(200, EventOp.GRANT, "user:alice", RelationType.HAS_PERMISSION,
                   "bucket:photos", PermissionLevel.ADMIN),
    )
    assert graph.num_edges == 1
    assert graph.edge_level[0] == PermissionLevel.ADMIN
    assert graph.edge_created[0] == 200


def test_unknown_actor_is_encoded_as_minus_one() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops")
    )
    assert graph.edge_actor[0] == -1


def test_known_actor_points_at_a_node() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops",
                   actor="user:root")
    )
    assert graph.edge_actor[0] == graph.index_of("user:root")


def test_out_of_order_events_are_rejected() -> None:
    builder = GraphBuilder()
    builder.apply(GraphEvent(200, EventOp.GRANT, "user:a", RelationType.MEMBER_OF, "group:g"))
    with pytest.raises(ValueError, match="non-decreasing"):
        builder.apply(GraphEvent(100, EventOp.GRANT, "user:b", RelationType.MEMBER_OF, "group:g"))


def test_neighbors_respects_relation_and_direction() -> None:
    graph = _build(
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(101, EventOp.GRANT, "user:bob", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(102, EventOp.GRANT, "user:alice", RelationType.HAS_PERMISSION,
                   "bucket:photos", PermissionLevel.READ),
    )
    outgoing = graph.neighbors("user:alice", RelationType.MEMBER_OF)
    assert outgoing.tolist() == [graph.index_of("group:devops")]

    incoming = graph.neighbors("group:devops", RelationType.MEMBER_OF, incoming=True)
    assert sorted(incoming.tolist()) == sorted(
        [graph.index_of("user:alice"), graph.index_of("user:bob")]
    )

    assert graph.neighbors("user:bob", RelationType.HAS_PERMISSION).size == 0


def test_neighbors_of_an_empty_graph_is_empty() -> None:
    graph = GraphBuilder().build()
    assert graph.num_nodes == 0
    assert graph.num_edges == 0
    assert isinstance(graph.edge_src, np.ndarray)
```

```python
# tests/domain/test_replay.py
"""Replay: a graph is always a journal evaluated up to a point in time."""

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay


def _journal() -> list[GraphEvent]:
    return [
        GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF, "group:devops"),
        GraphEvent(200, EventOp.GRANT, "group:devops", RelationType.HAS_PERMISSION,
                   "bucket:photos", PermissionLevel.WRITE),
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
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest tests/domain/test_graph.py tests/domain/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.domain.graph'`

- [ ] **Step 3: Implement `graph.py`**

```python
# src/rga/domain/graph.py
"""The access graph in array form.

Stored as parallel arrays rather than objects because every consumer — feature
extraction, the network, the baselines — works on the whole graph at once.
Adjacency is materialised lazily in compressed form, once per relation and
direction, because feature extraction queries neighbourhoods per node and a
mask scan per query would be quadratic.

Unknown values are -1 everywhere: unknown is not zero and not absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from rga.domain.entities import entity_type
from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import RelationType

UNKNOWN = -1


@dataclass(frozen=True, eq=False)
class AccessGraph:
    """An immutable snapshot of the access graph."""

    node_ids: tuple[str, ...]
    node_index: dict[str, int]
    node_type: np.ndarray
    node_created: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_rel: np.ndarray
    edge_level: np.ndarray
    edge_created: np.ndarray
    edge_actor: np.ndarray

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.shape[0])

    def index_of(self, entity_id: str) -> int:
        """Row index of a node, raising if it is not in the graph."""
        try:
            return self.node_index[entity_id]
        except KeyError:
            raise KeyError(f"node not in graph: {entity_id!r}") from None

    def neighbors(
        self, entity_id: str, relation: RelationType, *, incoming: bool = False
    ) -> np.ndarray:
        """Node indices reachable from `entity_id` over one edge of `relation`."""
        node = self.index_of(entity_id)
        indptr, indices = self._adjacency[(int(relation), incoming)]
        return indices[indptr[node] : indptr[node + 1]]

    @cached_property
    def _adjacency(self) -> dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]]:
        """(relation, incoming) -> (indptr of length N+1, neighbour indices)."""
        table: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]] = {}
        for relation in RelationType:
            mask = self.edge_rel == int(relation)
            src = self.edge_src[mask]
            dst = self.edge_dst[mask]
            for incoming in (False, True):
                anchor, other = (dst, src) if incoming else (src, dst)
                order = np.argsort(anchor, kind="stable")
                counts = np.bincount(anchor, minlength=self.num_nodes)
                indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
                np.cumsum(counts, out=indptr[1:])
                table[(int(relation), incoming)] = (
                    indptr,
                    other[order].astype(np.int32, copy=False),
                )
        return table


@dataclass
class _EdgeRecord:
    level: int
    created: int
    actor: int


@dataclass
class GraphBuilder:
    """Applies events in time order and freezes the result into an AccessGraph."""

    _node_index: dict[str, int] = field(default_factory=dict)
    _node_ids: list[str] = field(default_factory=list)
    _node_type: list[int] = field(default_factory=list)
    _node_created: list[int] = field(default_factory=list)
    _edges: dict[tuple[int, int, int], _EdgeRecord] = field(default_factory=dict)
    _last_ts: int | None = None

    def apply(self, event: GraphEvent) -> None:
        """Apply one event. Events must arrive in non-decreasing time order."""
        if self._last_ts is not None and event.ts < self._last_ts:
            raise ValueError(
                f"events must arrive in non-decreasing ts order: {event.ts} after {self._last_ts}"
            )
        self._last_ts = event.ts

        if event.op is EventOp.GRANT:
            self.add_edge(
                event.subject,
                event.relation,
                event.object,
                level=int(event.level),
                created=event.ts,
                actor=event.actor,
            )
        else:
            self.register_node(event.subject, event.ts)
            self.register_node(event.object, event.ts)
            self.remove_edge(event.subject, event.relation, event.object)

    def add_edge(
        self,
        subject: str,
        relation: RelationType,
        target: str,
        *,
        level: int = 0,
        created: int = UNKNOWN,
        actor: str | None = None,
    ) -> None:
        """Add or replace an edge directly, without the event ordering rules.

        Used when the data arrives as a snapshot rather than a journal, where
        some rows carry no timestamp at all.
        """
        source = self.register_node(subject, created)
        destination = self.register_node(target, created)
        actor_index = UNKNOWN if actor is None else self.register_node(actor, created)
        self._edges[(source, int(relation), destination)] = _EdgeRecord(
            level, created, actor_index
        )

    def remove_edge(self, subject: str, relation: RelationType, target: str) -> None:
        """Remove an edge if it is present."""
        source = self._node_index.get(subject)
        destination = self._node_index.get(target)
        if source is None or destination is None:
            return
        self._edges.pop((source, int(relation), destination), None)

    def build(self) -> AccessGraph:
        """Freeze the accumulated state into arrays."""
        count = len(self._edges)
        edge_src = np.empty(count, dtype=np.int32)
        edge_dst = np.empty(count, dtype=np.int32)
        edge_rel = np.empty(count, dtype=np.int8)
        edge_level = np.empty(count, dtype=np.int8)
        edge_created = np.empty(count, dtype=np.int64)
        edge_actor = np.empty(count, dtype=np.int32)

        for position, ((src, relation, dst), record) in enumerate(self._edges.items()):
            edge_src[position] = src
            edge_dst[position] = dst
            edge_rel[position] = relation
            edge_level[position] = record.level
            edge_created[position] = record.created
            edge_actor[position] = record.actor

        return AccessGraph(
            node_ids=tuple(self._node_ids),
            node_index=dict(self._node_index),
            node_type=np.array(self._node_type, dtype=np.int8),
            node_created=np.array(self._node_created, dtype=np.int64),
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_rel=edge_rel,
            edge_level=edge_level,
            edge_created=edge_created,
            edge_actor=edge_actor,
        )

    def register_node(self, entity_id: str, created: int = UNKNOWN) -> int:
        """Index of a node, registering it on first sight with its creation time."""
        index = self._node_index.get(entity_id)
        if index is None:
            index = len(self._node_ids)
            self._node_index[entity_id] = index
            self._node_ids.append(entity_id)
            self._node_type.append(int(entity_type(entity_id)))
            self._node_created.append(created)
        return index
```

- [ ] **Step 4: Implement `replay.py`**

```python
# src/rga/domain/replay.py
"""Journal to graph."""

from __future__ import annotations

from collections.abc import Iterable

from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph, GraphBuilder


def replay(events: Iterable[GraphEvent], until: int | None = None) -> AccessGraph:
    """Build the graph as it stood at `until`, inclusive.

    Events must be in non-decreasing time order; iteration stops at the first
    event past the cutoff, so a sorted journal is never read in full.
    """
    builder = GraphBuilder()
    for event in events:
        if until is not None and event.ts > until:
            break
        builder.apply(event)
    return builder.build()
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/domain -v`
Expected: PASS

- [ ] **Step 6: Commit and push**

```bash
git add src/rga/domain/graph.py src/rga/domain/replay.py tests/domain/test_graph.py tests/domain/test_replay.py
git commit -m "feat: add access graph and journal replay"
git push origin main
```

---

## Task 6: Source capabilities and relation mapping

This is the portability contract from section 12 of the spec. Everything above the
adapter layer talks to `GraphSource` and asks it what it can provide; it never asks
what product is behind it.

**Files:**
- Create: `src/rga/adapters/__init__.py`, `src/rga/adapters/base.py`, `src/rga/adapters/mapping.py`, `configs/mapping/opens3.yaml`
- Test: `tests/adapters/test_base.py`, `tests/adapters/test_mapping.py`

**Interfaces:**
- Consumes: `RelationType`, `PermissionLevel`, `parse_level` (Task 3); `AccessGraph` (Task 5); `GraphEvent` (Task 4)
- Produces:
  - `Capabilities` frozen dataclass with `timestamps: bool`, `provenance: bool`, `change_log: bool` and property `level -> int`
  - `GraphSource` Protocol with `capabilities()`, `snapshot(at)`, `events(since, until)`
  - `RelationRule` frozen dataclass with `relation: RelationType`, `level: PermissionLevel | None`
  - `RelationMapping` with `from_config(cfg) -> RelationMapping`, `load(path) -> RelationMapping`, `translate(source_relation, level_property=None) -> tuple[RelationType, PermissionLevel]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/adapters/test_base.py
"""Capability levels decide which feature groups a source can fill."""

from rga.adapters.base import Capabilities


def test_bare_snapshot_is_level_zero() -> None:
    assert Capabilities(timestamps=False, provenance=False, change_log=False).level == 0


def test_timestamps_alone_are_level_one() -> None:
    assert Capabilities(timestamps=True, provenance=False, change_log=False).level == 1


def test_timestamps_with_provenance_are_level_two() -> None:
    assert Capabilities(timestamps=True, provenance=True, change_log=True).level == 2


def test_provenance_without_timestamps_is_still_level_zero() -> None:
    # Knowing who granted a right is useless without knowing when.
    assert Capabilities(timestamps=False, provenance=True, change_log=False).level == 0


def test_a_change_log_alone_does_not_raise_the_level() -> None:
    assert Capabilities(timestamps=False, provenance=False, change_log=True).level == 0
```

```python
# tests/adapters/test_mapping.py
"""Translation from a source's own vocabulary to the canonical one."""

from pathlib import Path

import pytest

from rga.adapters.mapping import RelationMapping
from rga.domain.relations import PermissionLevel, RelationType


def test_level_carrying_relation_reads_the_level_from_a_property() -> None:
    mapping = RelationMapping.from_config(
        {"HAS_PERMISSION": {"relation": "has_permission", "level": "from_property"}}
    )
    assert mapping.translate("HAS_PERMISSION", "admin") == (
        RelationType.HAS_PERMISSION,
        PermissionLevel.ADMIN,
    )


def test_a_system_without_levels_encodes_them_in_relation_names() -> None:
    # SpiceDB and OpenFGA style: each right is its own relation, no ordinal property.
    mapping = RelationMapping.from_config(
        {
            "viewer": {"relation": "has_permission", "level": "read"},
            "editor": {"relation": "has_permission", "level": "write"},
            "owner": {"relation": "owner_of"},
        }
    )
    assert mapping.translate("viewer") == (RelationType.HAS_PERMISSION, PermissionLevel.READ)
    assert mapping.translate("editor") == (RelationType.HAS_PERMISSION, PermissionLevel.WRITE)
    assert mapping.translate("owner") == (RelationType.OWNER_OF, PermissionLevel.NONE)


def test_missing_property_on_a_level_carrying_relation_yields_none() -> None:
    # An unknown level is masked downstream rather than guessed.
    mapping = RelationMapping.from_config(
        {"HAS_PERMISSION": {"relation": "has_permission", "level": "from_property"}}
    )
    assert mapping.translate("HAS_PERMISSION", None) == (
        RelationType.HAS_PERMISSION,
        PermissionLevel.NONE,
    )


def test_unknown_source_relation_is_rejected() -> None:
    mapping = RelationMapping.from_config({"viewer": {"relation": "has_permission",
                                                      "level": "read"}})
    with pytest.raises(KeyError, match="unmapped source relation"):
        mapping.translate("banana")


def test_a_fixed_level_on_a_relation_that_cannot_carry_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot carry a level"):
        RelationMapping.from_config({"boss": {"relation": "member_of", "level": "admin"}})


def test_the_shipped_opens3_mapping_loads() -> None:
    mapping = RelationMapping.load(Path("configs/mapping/opens3.yaml"))
    assert mapping.translate("MEMBER_OF") == (RelationType.MEMBER_OF, PermissionLevel.NONE)
    assert mapping.translate("HAS_PERMISSION", "write") == (
        RelationType.HAS_PERMISSION,
        PermissionLevel.WRITE,
    )
    assert mapping.translate("OWNER_OF") == (RelationType.OWNER_OF, PermissionLevel.NONE)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest tests/adapters -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.adapters'`

- [ ] **Step 3: Implement `base.py`**

```python
# src/rga/adapters/base.py
"""The contract every data source implements.

Capability levels, from section 12 of the design document:

    0  a snapshot of relation tuples — any authorization engine has this
    1  plus creation timestamps — enables the temporal feature group
    2  plus the initiator of each change — enables the provenance group

A source states what it has; feature extraction masks what it lacks. Nothing
above this layer knows which product is behind the source.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph


@dataclass(frozen=True)
class Capabilities:
    """What a source can provide.

    `change_log` is orthogonal to the level: it says whether `events()` is
    supported, which also decides whether revocations are observable at all —
    a revoked edge is simply absent from a snapshot.
    """

    timestamps: bool
    provenance: bool
    change_log: bool

    @property
    def level(self) -> int:
        """Capability level 0, 1 or 2."""
        if not self.timestamps:
            return 0
        return 2 if self.provenance else 1


@runtime_checkable
class GraphSource(Protocol):
    """A source of access-graph data."""

    def capabilities(self) -> Capabilities:
        """Describe what this source can provide."""
        ...

    def snapshot(self, at: int | None = None) -> AccessGraph:
        """The graph as it stood at `at`, or the latest state when `at` is None.

        A source without timestamps ignores `at` and returns the current state.
        """
        ...

    def events(self, since: int = 0, until: int | None = None) -> Iterator[GraphEvent]:
        """Changes in the half-open interval, in non-decreasing time order.

        Raises NotImplementedError when `capabilities().change_log` is False.
        """
        ...
```

- [ ] **Step 4: Implement `mapping.py`**

```python
# src/rga/adapters/mapping.py
"""Declarative translation of a source's relation vocabulary to the canonical one.

Vocabularies differ in kind, not just in spelling. This engine expresses a right
as one relation with an ordered level property. Most Zanzibar-like systems have
no ordinal level at all and model each right as a separate relation, with the
hierarchy between them living in schema rules. Both shapes map here:

    HAS_PERMISSION: {relation: has_permission, level: from_property}
    viewer:         {relation: has_permission, level: read}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from rga.domain.relations import (
    LEVEL_CARRYING,
    PermissionLevel,
    RelationType,
    parse_level,
    parse_relation,
)

#: Sentinel meaning "the level is a property of the edge, read it from the source".
FROM_PROPERTY = "from_property"


@dataclass(frozen=True)
class RelationRule:
    """How one source relation becomes a canonical one.

    `level` of None means the level is carried by the edge itself.
    """

    relation: RelationType
    level: PermissionLevel | None


@dataclass(frozen=True)
class RelationMapping:
    """The full vocabulary translation for one source."""

    rules: Mapping[str, RelationRule]

    @classmethod
    def from_config(cls, config: Mapping[str, Mapping[str, str]]) -> RelationMapping:
        """Build from the parsed `relation_mapping` section of a config file."""
        rules: dict[str, RelationRule] = {}
        for source_name, spec in config.items():
            relation = parse_relation(spec["relation"])
            raw_level = spec.get("level")
            if raw_level is None:
                level: PermissionLevel | None = PermissionLevel.NONE
            elif raw_level == FROM_PROPERTY:
                level = None
            else:
                level = parse_level(raw_level)

            if level is not PermissionLevel.NONE and relation not in LEVEL_CARRYING:
                raise ValueError(f"{relation.name} cannot carry a level (source {source_name!r})")
            rules[source_name] = RelationRule(relation=relation, level=level)
        return cls(rules=rules)

    @classmethod
    def load(cls, path: Path) -> RelationMapping:
        """Load from a YAML file holding a `relation_mapping` key."""
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_config(document["relation_mapping"])

    def translate(
        self, source_relation: str, level_property: str | None = None
    ) -> tuple[RelationType, PermissionLevel]:
        """Translate one source relation, with its level property when it has one."""
        try:
            rule = self.rules[source_relation]
        except KeyError:
            raise KeyError(f"unmapped source relation: {source_relation!r}") from None
        if rule.level is not None:
            return rule.relation, rule.level
        return rule.relation, parse_level(level_property)
```

- [ ] **Step 5: Write the shipped mapping for opens3-rebac**

```yaml
# configs/mapping/opens3.yaml
# Vocabulary of the opens3-rebac authorization engine.
#
# OWNER_OF and VIEWER are listed because the engine's permission check uses them,
# even though its gRPC enum currently has no value for them and they cannot be
# created through the API. See docs/opens3-rebac-findings.md, finding 2.
relation_mapping:
  MEMBER_OF:
    relation: member_of
  HAS_PERMISSION:
    relation: has_permission
    level: from_property
  PARENT_OF:
    relation: parent_of
  OWNER_OF:
    relation: owner_of
  VIEWER:
    relation: has_permission
    level: read
```

- [ ] **Step 6: Create the package init**

```python
# src/rga/adapters/__init__.py
"""Data sources and the contract they implement."""
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/adapters -v`
Expected: PASS

- [ ] **Step 8: Commit and push**

```bash
git add src/rga/adapters configs/mapping tests/adapters
git commit -m "feat: add source capability contract and relation mapping"
git push origin main
```

---

## Task 7: File-backed source

**Files:**
- Create: `src/rga/adapters/file_source.py`
- Test: `tests/adapters/test_file_source.py`

**Interfaces:**
- Consumes: `GraphSource`, `Capabilities` (Task 6); `read_events` (Task 4); `replay` (Task 5)
- Produces: `FileSource(path: Path)` implementing `GraphSource`, plus `FileSource.capabilities()` derived by inspecting the journal's first events

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/test_file_source.py
"""A journal file behaves as a full-capability source."""

from pathlib import Path

import pytest

from rga.adapters.base import GraphSource
from rga.adapters.file_source import FileSource
from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.io.jsonl import write_events


def _journal(tmp_path: Path, *, actors: bool) -> Path:
    path = tmp_path / "events.jsonl"
    actor = "user:root" if actors else None
    write_events(
        path,
        [
            GraphEvent(100, EventOp.GRANT, "user:alice", RelationType.MEMBER_OF,
                       "group:devops", actor=actor),
            GraphEvent(200, EventOp.GRANT, "group:devops", RelationType.HAS_PERMISSION,
                       "bucket:photos", PermissionLevel.WRITE, actor=actor),
            GraphEvent(300, EventOp.REVOKE, "user:alice", RelationType.MEMBER_OF,
                       "group:devops", actor=actor),
        ],
    )
    return path


def test_file_source_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(FileSource(_journal(tmp_path, actors=True)), GraphSource)


def test_a_journal_with_actors_reaches_level_two(tmp_path: Path) -> None:
    capabilities = FileSource(_journal(tmp_path, actors=True)).capabilities()
    assert capabilities.level == 2
    assert capabilities.change_log is True


def test_a_journal_without_actors_stays_at_level_one(tmp_path: Path) -> None:
    assert FileSource(_journal(tmp_path, actors=False)).capabilities().level == 1


def test_snapshot_honours_the_cutoff(tmp_path: Path) -> None:
    source = FileSource(_journal(tmp_path, actors=True))
    assert source.snapshot(at=200).num_edges == 2
    assert source.snapshot().num_edges == 1


def test_events_are_bounded_at_both_ends(tmp_path: Path) -> None:
    source = FileSource(_journal(tmp_path, actors=True))
    assert [event.ts for event in source.events(since=150, until=250)] == [200]


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.jsonl").capabilities()
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/adapters/test_file_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.adapters.file_source'`

- [ ] **Step 3: Implement it**

```python
# src/rga/adapters/file_source.py
"""A source backed by a saved journal file.

Used by the generator, by saved datasets, and by any integration that exports a
change log offline rather than being queried live.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from rga.adapters.base import Capabilities
from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph
from rga.domain.replay import replay
from rga.io.jsonl import read_events

#: How many leading events to inspect when deciding whether actors are recorded.
_PROBE_SIZE = 256


class FileSource:
    """Reads a newline-delimited journal from disk."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def capabilities(self) -> Capabilities:
        """A journal always has timestamps; provenance depends on the data."""
        probe = list(islice(read_events(self._path), _PROBE_SIZE))
        has_actor = any(event.actor is not None for event in probe)
        return Capabilities(timestamps=True, provenance=has_actor, change_log=True)

    def snapshot(self, at: int | None = None) -> AccessGraph:
        """Replay the journal up to `at`."""
        return replay(read_events(self._path), until=at)

    def events(self, since: int = 0, until: int | None = None) -> Iterator[GraphEvent]:
        """Yield journal events within the interval, stopping early past `until`."""
        for event in read_events(self._path):
            if until is not None and event.ts > until:
                return
            if event.ts >= since:
                yield event
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/adapters -v`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/adapters/file_source.py tests/adapters/test_file_source.py
git commit -m "feat: add file-backed graph source"
git push origin main
```

---

## Task 8: Generator configuration and organization model

The organization is the *static* structure: who exists, which team they belong to,
which projects that team owns. Task 9 decides *when* each of those facts becomes
true. Splitting them this way keeps the growth process readable and lets tests
check structure without simulating time.

**Files:**
- Create: `src/rga/generator/__init__.py`, `src/rga/generator/config.py`, `src/rga/generator/org.py`, `configs/generator/small.yaml`, `configs/generator/default.yaml`
- Test: `tests/generator/test_config.py`, `tests/generator/test_org.py`

**Interfaces:**
- Consumes: `PermissionLevel` (Task 3)
- Produces:
  - `OrgConfig`, `TimelineConfig`, `AnomalyConfig`, `DatasetConfig` frozen dataclasses
  - `load_dataset_config(path: Path) -> DatasetConfig`
  - `User(id, team, department)`, `Team(id, group_id, department, members, buckets)`, `Bucket(id, owner, team, objects)`, `CrossCuttingGroup(id, members, level, buckets)`
  - `Organization` with fields `users`, `teams`, `buckets`, `cross_cutting` and methods `user(entity_id)`, `team(entity_id)`, `bucket(entity_id)`, `team_of(user_id)`, `department_of(user_id)`
  - `build_organization(config: OrgConfig, rng: np.random.Generator) -> Organization`

- [ ] **Step 1: Write the failing config test**

```python
# tests/generator/test_config.py
"""Configuration loading and validation."""

from pathlib import Path

import pytest

from rga.generator.config import load_dataset_config


def test_shipped_small_config_loads() -> None:
    config = load_dataset_config(Path("configs/generator/small.yaml"))
    assert config.seed >= 0
    assert config.org.departments >= 1
    assert config.timeline.days > config.eval_window_days
    assert 0.0 < config.anomalies.rate < 1.0


def test_shipped_default_config_loads() -> None:
    config = load_dataset_config(Path("configs/generator/default.yaml"))
    assert config.org.departments >= config.org.cross_cutting_groups


def test_eval_window_must_fit_inside_the_timeline(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\nseed: 1\neval_window_days: 30\n"
        "org: {departments: 1, teams_per_department: [1, 1], users_per_team: [2, 2],\n"
        "      projects_per_team: [1, 1], objects_per_bucket: [1, 2],\n"
        "      cross_cutting_groups: 0, cross_cutting_membership_rate: 0.0,\n"
        "      legitimate_exception_rate: 0.0}\n"
        "timeline: {start_ts: 0, days: 10, working_hours: [9, 18], off_hours_rate: 0.05,\n"
        "           weekend_rate: 0.08, hires_per_day: 1.0, departures_per_day: 0.1,\n"
        "           grants_per_day: 1.0, uploads_per_day: 1.0}\n"
        "anomalies: {patterns: [], rate: 0.01, train_contamination: 0.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="eval window"):
        load_dataset_config(path)


def test_a_range_must_be_ordered(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\nseed: 1\neval_window_days: 2\n"
        "org: {departments: 1, teams_per_department: [5, 1], users_per_team: [2, 2],\n"
        "      projects_per_team: [1, 1], objects_per_bucket: [1, 2],\n"
        "      cross_cutting_groups: 0, cross_cutting_membership_rate: 0.0,\n"
        "      legitimate_exception_rate: 0.0}\n"
        "timeline: {start_ts: 0, days: 10, working_hours: [9, 18], off_hours_rate: 0.05,\n"
        "           weekend_rate: 0.08, hires_per_day: 1.0, departures_per_day: 0.1,\n"
        "           grants_per_day: 1.0, uploads_per_day: 1.0}\n"
        "anomalies: {patterns: [], rate: 0.01, train_contamination: 0.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="teams_per_department"):
        load_dataset_config(path)
```

- [ ] **Step 2: Write the failing organization test**

```python
# tests/generator/test_org.py
"""Static organization structure."""

import numpy as np

from rga.domain.entities import EntityType, entity_type
from rga.generator.config import OrgConfig
from rga.generator.org import build_organization

CONFIG = OrgConfig(
    departments=3,
    teams_per_department=(2, 4),
    users_per_team=(3, 6),
    projects_per_team=(1, 3),
    objects_per_bucket=(5, 20),
    cross_cutting_groups=2,
    cross_cutting_membership_rate=0.1,
    legitimate_exception_rate=0.05,
)


def _org(seed: int = 7):
    return build_organization(CONFIG, np.random.default_rng(seed))


def test_team_count_is_within_the_configured_range() -> None:
    org = _org()
    per_department: dict[int, int] = {}
    for team in org.teams:
        per_department[team.department] = per_department.get(team.department, 0) + 1
    assert len(per_department) == CONFIG.departments
    assert all(2 <= count <= 4 for count in per_department.values())


def test_every_user_belongs_to_exactly_one_team() -> None:
    org = _org()
    memberships = [user.id for team in org.teams for user in org.users if user.team == team.id]
    assert sorted(memberships) == sorted(user.id for user in org.users)


def test_ids_use_the_engine_format() -> None:
    org = _org()
    assert all(entity_type(user.id) is EntityType.USER for user in org.users)
    assert all(entity_type(team.group_id) is EntityType.GROUP for team in org.teams)
    assert all(entity_type(bucket.id) is EntityType.BUCKET for bucket in org.buckets)
    assert all(
        entity_type(obj) is EntityType.OBJECT for bucket in org.buckets for obj in bucket.objects
    )


def test_objects_live_in_their_own_bucket() -> None:
    org = _org()
    for bucket in org.buckets:
        prefix = f"object:{bucket.id.split(':', 1)[1]}/"
        assert all(obj.startswith(prefix) for obj in bucket.objects)


def test_each_bucket_is_owned_by_a_member_of_its_team() -> None:
    org = _org()
    for bucket in org.buckets:
        assert bucket.owner in org.team(bucket.team).members


def test_generation_is_reproducible() -> None:
    assert _org(7) == _org(7)


def test_different_seeds_give_different_organizations() -> None:
    assert _org(7) != _org(8)


def test_lookups_resolve() -> None:
    org = _org()
    user = org.users[0]
    assert org.user(user.id) is user
    assert org.team_of(user.id).id == user.team
    assert org.department_of(user.id) == user.department
```

- [ ] **Step 3: Run both to confirm they fail**

Run: `uv run pytest tests/generator -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.generator'`

- [ ] **Step 4: Implement `config.py`**

```python
# src/rga/generator/config.py
"""Typed configuration for dataset generation.

Plain dataclasses over a YAML document: the configuration is small and fully
enumerated here, so an extra validation dependency would buy nothing. Validation
is explicit and fails at load time, not halfway through a generation run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OrgConfig:
    """Shape of the modelled organization."""

    departments: int
    teams_per_department: tuple[int, int]
    users_per_team: tuple[int, int]
    projects_per_team: tuple[int, int]
    objects_per_bucket: tuple[int, int]
    cross_cutting_groups: int
    cross_cutting_membership_rate: float
    #: Share of users holding a right outside their own department. Without these
    #: the graph is a perfect hierarchy and every deviation is trivially anomalous.
    legitimate_exception_rate: float


@dataclass(frozen=True)
class TimelineConfig:
    """Intensity of the growth process."""

    start_ts: int
    days: int
    working_hours: tuple[int, int]
    off_hours_rate: float
    weekend_rate: float
    hires_per_day: float
    departures_per_day: float
    grants_per_day: float
    uploads_per_day: float


@dataclass(frozen=True)
class AnomalyConfig:
    """Which patterns to inject and how densely."""

    patterns: tuple[str, ...]
    #: Share of edges in the evaluation window that are anomalous.
    rate: float
    #: Share of anomalous edges planted in the training span, to test robustness
    #: to a training graph that is not perfectly clean.
    train_contamination: float


@dataclass(frozen=True)
class DatasetConfig:
    """A complete, reproducible dataset recipe."""

    name: str
    seed: int
    org: OrgConfig
    timeline: TimelineConfig
    anomalies: AnomalyConfig
    eval_window_days: int


def _pair(raw: object, field_name: str) -> tuple[int, int]:
    """Read an inclusive [low, high] range and check its ordering."""
    values = tuple(int(item) for item in raw)  # type: ignore[union-attr]
    if len(values) != 2:
        raise ValueError(f"{field_name} must have exactly two values, got {values}")
    low, high = values
    if low > high:
        raise ValueError(f"{field_name} range is inverted: {low} > {high}")
    return low, high


def _rate(raw: object, field_name: str) -> float:
    value = float(raw)  # type: ignore[arg-type]
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1], got {value}")
    return value


def _org_config(document: Mapping[str, object]) -> OrgConfig:
    return OrgConfig(
        departments=int(document["departments"]),  # type: ignore[arg-type]
        teams_per_department=_pair(document["teams_per_department"], "teams_per_department"),
        users_per_team=_pair(document["users_per_team"], "users_per_team"),
        projects_per_team=_pair(document["projects_per_team"], "projects_per_team"),
        objects_per_bucket=_pair(document["objects_per_bucket"], "objects_per_bucket"),
        cross_cutting_groups=int(document["cross_cutting_groups"]),  # type: ignore[arg-type]
        cross_cutting_membership_rate=_rate(
            document["cross_cutting_membership_rate"], "cross_cutting_membership_rate"
        ),
        legitimate_exception_rate=_rate(
            document["legitimate_exception_rate"], "legitimate_exception_rate"
        ),
    )


def _timeline_config(document: Mapping[str, object]) -> TimelineConfig:
    hours = _pair(document["working_hours"], "working_hours")
    if not 0 <= hours[0] < hours[1] <= 24:
        raise ValueError(f"working_hours must lie within [0, 24], got {hours}")
    return TimelineConfig(
        start_ts=int(document["start_ts"]),  # type: ignore[arg-type]
        days=int(document["days"]),  # type: ignore[arg-type]
        working_hours=hours,
        off_hours_rate=_rate(document["off_hours_rate"], "off_hours_rate"),
        weekend_rate=_rate(document["weekend_rate"], "weekend_rate"),
        hires_per_day=float(document["hires_per_day"]),  # type: ignore[arg-type]
        departures_per_day=float(document["departures_per_day"]),  # type: ignore[arg-type]
        grants_per_day=float(document["grants_per_day"]),  # type: ignore[arg-type]
        uploads_per_day=float(document["uploads_per_day"]),  # type: ignore[arg-type]
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset recipe."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    timeline = _timeline_config(document["timeline"])
    eval_window_days = int(document["eval_window_days"])
    if eval_window_days >= timeline.days:
        raise ValueError(
            f"eval window of {eval_window_days} days does not fit in {timeline.days} days"
        )
    anomalies_document = document["anomalies"]
    return DatasetConfig(
        name=str(document["name"]),
        seed=int(document["seed"]),
        org=_org_config(document["org"]),
        timeline=timeline,
        anomalies=AnomalyConfig(
            patterns=tuple(str(name) for name in anomalies_document["patterns"]),
            rate=_rate(anomalies_document["rate"], "anomalies.rate"),
            train_contamination=_rate(
                anomalies_document["train_contamination"], "anomalies.train_contamination"
            ),
        ),
        eval_window_days=eval_window_days,
    )
```

- [ ] **Step 5: Implement `org.py`**

```python
# src/rga/generator/org.py
"""The static structure of the modelled organization.

Departments hold teams, teams own projects, a project is a bucket holding
objects. A team maps to a group, and rights are normally granted to that group
rather than to individuals. Cross-cutting groups — an operations team, say —
hold elevated rights across many buckets, and a small share of users hold rights
outside their own department. Both exist so that the normal graph is not a
perfect tree: without them every deviation from the hierarchy would be anomalous
by construction and the problem would be trivial.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from rga.domain.relations import PermissionLevel
from rga.generator.config import OrgConfig


@dataclass(frozen=True)
class User:
    id: str
    team: str
    department: int


@dataclass(frozen=True)
class Team:
    id: str
    group_id: str
    department: int
    members: tuple[str, ...]
    buckets: tuple[str, ...]


@dataclass(frozen=True)
class Bucket:
    id: str
    owner: str
    team: str
    objects: tuple[str, ...]


@dataclass(frozen=True)
class CrossCuttingGroup:
    id: str
    members: tuple[str, ...]
    level: PermissionLevel
    buckets: tuple[str, ...]


@dataclass(frozen=True)
class Organization:
    """Everything that exists, without any notion of when it appeared."""

    users: tuple[User, ...]
    teams: tuple[Team, ...]
    buckets: tuple[Bucket, ...]
    cross_cutting: tuple[CrossCuttingGroup, ...]

    def user(self, entity_id: str) -> User:
        return self._users_by_id[entity_id]

    def team(self, team_id: str) -> Team:
        return self._teams_by_id[team_id]

    def bucket(self, bucket_id: str) -> Bucket:
        return self._buckets_by_id[bucket_id]

    def team_of(self, user_id: str) -> Team:
        return self.team(self.user(user_id).team)

    def department_of(self, user_id: str) -> int:
        return self.user(user_id).department

    @property
    def _users_by_id(self) -> dict[str, User]:
        return {user.id: user for user in self.users}

    @property
    def _teams_by_id(self) -> dict[str, Team]:
        return {team.id: team for team in self.teams}

    @property
    def _buckets_by_id(self) -> dict[str, Bucket]:
        return {bucket.id: bucket for bucket in self.buckets}


def _uuid(rng: np.random.Generator) -> str:
    """A deterministic UUID drawn from the seeded generator."""
    return str(uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8)), version=4))


def _between(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    """Inclusive integer draw."""
    low, high = bounds
    return int(rng.integers(low, high + 1))


def build_organization(config: OrgConfig, rng: np.random.Generator) -> Organization:
    """Construct the organization deterministically from a seeded generator."""
    users: list[User] = []
    teams: list[Team] = []
    buckets: list[Bucket] = []

    for department in range(config.departments):
        for team_number in range(_between(rng, config.teams_per_department)):
            team_id = f"team-{department}-{team_number}"
            group_id = f"group:{team_id}"

            members = tuple(
                User(id=f"user:{_uuid(rng)}", team=team_id, department=department).id
                for _ in range(_between(rng, config.users_per_team))
            )
            users.extend(
                User(id=member, team=team_id, department=department) for member in members
            )

            team_buckets: list[str] = []
            for project in range(_between(rng, config.projects_per_team)):
                bucket_id = f"bucket:{team_id}-p{project}"
                objects = tuple(
                    f"object:{team_id}-p{project}/file-{index:05d}.dat"
                    for index in range(_between(rng, config.objects_per_bucket))
                )
                owner = members[int(rng.integers(len(members)))]
                buckets.append(Bucket(id=bucket_id, owner=owner, team=team_id, objects=objects))
                team_buckets.append(bucket_id)

            teams.append(
                Team(
                    id=team_id,
                    group_id=group_id,
                    department=department,
                    members=members,
                    buckets=tuple(team_buckets),
                )
            )

    cross_cutting: list[CrossCuttingGroup] = []
    all_bucket_ids = tuple(bucket.id for bucket in buckets)
    for index in range(config.cross_cutting_groups):
        membership = tuple(
            user.id for user in users if rng.random() < config.cross_cutting_membership_rate
        )
        covered = tuple(
            bucket_id for bucket_id in all_bucket_ids if rng.random() < 0.5
        )
        cross_cutting.append(
            CrossCuttingGroup(
                id=f"group:ops-{index}",
                members=membership,
                level=PermissionLevel.WRITE if index % 2 else PermissionLevel.ADMIN,
                buckets=covered,
            )
        )

    return Organization(
        users=tuple(users),
        teams=tuple(teams),
        buckets=tuple(buckets),
        cross_cutting=tuple(cross_cutting),
    )
```

- [ ] **Step 6: Write the shipped configurations**

```yaml
# configs/generator/small.yaml
# Fast configuration used by tests and by a first end-to-end run.
name: small
seed: 20260912
eval_window_days: 7

org:
  departments: 3
  teams_per_department: [2, 3]
  users_per_team: [3, 6]
  projects_per_team: [1, 2]
  objects_per_bucket: [5, 30]
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
  grants_per_day: 4.0
  uploads_per_day: 20.0

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
```

```yaml
# configs/generator/default.yaml
# Main configuration for reported experiments.
name: default
seed: 20260912
eval_window_days: 14

org:
  departments: 8
  teams_per_department: [3, 6]
  users_per_team: [4, 14]
  projects_per_team: [1, 4]
  objects_per_bucket: [20, 400]
  cross_cutting_groups: 3
  cross_cutting_membership_rate: 0.06
  legitimate_exception_rate: 0.03

timeline:
  start_ts: 1735689600000   # 2025-01-01T00:00:00Z
  days: 180
  working_hours: [9, 19]
  off_hours_rate: 0.05
  weekend_rate: 0.08
  hires_per_day: 0.8
  departures_per_day: 0.3
  grants_per_day: 12.0
  uploads_per_day: 60.0

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
  rate: 0.01
  train_contamination: 0.0
```

- [ ] **Step 7: Create the package init**

```python
# src/rga/generator/__init__.py
"""Synthetic dataset generation."""
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/generator -v`
Expected: PASS

- [ ] **Step 9: Commit and push**

```bash
git add src/rga/generator configs/generator tests/generator
git commit -m "feat: add generator configuration and organization model"
git push origin main
```

---

## Task 9: Growth process

Turns the static organization into a journal. Intensities vary by hour and by
weekday, which matters more than it looks: without a diurnal rhythm in the normal
data, "a right granted at three in the morning" carries no information, and that is
one of the strongest signals a real audit has.

**Files:**
- Create: `src/rga/util/timeutil.py`, `src/rga/generator/timeline.py`
- Modify: `src/rga/domain/replay.py` (add `journal_issues`)
- Test: `tests/util/test_timeutil.py`, `tests/generator/test_timeline.py`

**Interfaces:**
- Consumes: `Organization` (Task 8); `TimelineConfig` (Task 8); `GraphEvent`, `EventOp` (Task 4)
- Produces:
  - `DAY_MS`, `HOUR_MS`, `MINUTE_MS`, `SECOND_MS` constants
  - `day_start(start_ts: int, day_index: int) -> int`
  - `is_weekend(ts: int) -> bool`
  - `hour_of_day(ts: int) -> int`
  - `sample_time_of_day(rng, day_start_ts: int, config: TimelineConfig) -> int`
  - `generate_normal_journal(org: Organization, config: TimelineConfig, rng, *, exception_rate: float) -> list[GraphEvent]`
  - `journal_issues(events: Iterable[GraphEvent]) -> list[str]`

- [ ] **Step 1: Write the failing time tests**

```python
# tests/util/test_timeutil.py
"""Time helpers. All timestamps in the project are UTC milliseconds."""

import numpy as np

from rga.generator.config import TimelineConfig
from rga.util.timeutil import DAY_MS, day_start, hour_of_day, is_weekend, sample_time_of_day

# 2025-01-01 was a Wednesday.
WEDNESDAY = 1_735_689_600_000

CONFIG = TimelineConfig(
    start_ts=WEDNESDAY,
    days=30,
    working_hours=(9, 19),
    off_hours_rate=0.05,
    weekend_rate=0.08,
    hires_per_day=1.0,
    departures_per_day=0.2,
    grants_per_day=5.0,
    uploads_per_day=10.0,
)


def test_day_start_advances_by_whole_days() -> None:
    assert day_start(WEDNESDAY, 0) == WEDNESDAY
    assert day_start(WEDNESDAY, 3) == WEDNESDAY + 3 * DAY_MS


def test_weekend_detection() -> None:
    assert not is_weekend(WEDNESDAY)
    assert is_weekend(day_start(WEDNESDAY, 3))  # Saturday
    assert is_weekend(day_start(WEDNESDAY, 4))  # Sunday
    assert not is_weekend(day_start(WEDNESDAY, 5))  # Monday


def test_sampled_times_stay_inside_their_day() -> None:
    rng = np.random.default_rng(0)
    start = day_start(WEDNESDAY, 2)
    for _ in range(200):
        ts = sample_time_of_day(rng, start, CONFIG)
        assert start <= ts < start + DAY_MS


def test_most_activity_lands_in_working_hours() -> None:
    rng = np.random.default_rng(0)
    hours = [hour_of_day(sample_time_of_day(rng, WEDNESDAY, CONFIG)) for _ in range(2000)]
    inside = sum(1 for hour in hours if 9 <= hour < 19)
    assert inside / len(hours) > 0.85


def test_off_hours_activity_is_present_but_rare() -> None:
    rng = np.random.default_rng(0)
    hours = [hour_of_day(sample_time_of_day(rng, WEDNESDAY, CONFIG)) for _ in range(2000)]
    outside = sum(1 for hour in hours if not 9 <= hour < 19)
    assert 0 < outside / len(hours) < 0.15
```

- [ ] **Step 2: Write the failing journal tests**

```python
# tests/generator/test_timeline.py
"""The normal growth process."""

import numpy as np

from rga.domain.events import EventOp
from rga.domain.relations import RelationType
from rga.domain.replay import journal_issues, replay
from rga.generator.config import OrgConfig, TimelineConfig
from rga.generator.org import build_organization
from rga.generator.timeline import generate_normal_journal
from rga.util.timeutil import DAY_MS, is_weekend

ORG = OrgConfig(
    departments=2,
    teams_per_department=(2, 3),
    users_per_team=(3, 5),
    projects_per_team=(1, 2),
    objects_per_bucket=(3, 10),
    cross_cutting_groups=1,
    cross_cutting_membership_rate=0.1,
    legitimate_exception_rate=0.05,
)
TIMELINE = TimelineConfig(
    start_ts=1_735_689_600_000,
    days=40,
    working_hours=(9, 19),
    off_hours_rate=0.05,
    weekend_rate=0.08,
    hires_per_day=0.5,
    departures_per_day=0.15,
    grants_per_day=3.0,
    uploads_per_day=8.0,
)


def _journal(seed: int = 3):
    rng = np.random.default_rng(seed)
    org = build_organization(ORG, rng)
    journal = generate_normal_journal(
        org, TIMELINE, rng, exception_rate=ORG.legitimate_exception_rate
    )
    return org, journal


def test_journal_is_sorted_by_time() -> None:
    _, events = _journal()
    assert [event.ts for event in events] == sorted(event.ts for event in events)


def test_journal_stays_inside_the_configured_span() -> None:
    _, events = _journal()
    end = TIMELINE.start_ts + TIMELINE.days * DAY_MS
    assert all(TIMELINE.start_ts <= event.ts < end for event in events)


def test_journal_is_internally_consistent() -> None:
    # No revoke of an edge that does not exist at that moment.
    _, events = _journal()
    assert journal_issues(events) == []


def test_every_team_group_receives_rights_on_its_own_buckets() -> None:
    org, events = _journal()
    granted = {
        (event.subject, event.object)
        for event in events
        if event.op is EventOp.GRANT and event.relation is RelationType.HAS_PERMISSION
    }
    for team in org.teams:
        for bucket in team.buckets:
            assert (team.group_id, bucket) in granted


def test_objects_are_attached_to_their_bucket() -> None:
    org, events = _journal()
    parent_edges = {
        (event.subject, event.object)
        for event in events
        if event.relation is RelationType.PARENT_OF
    }
    for bucket in org.buckets:
        attached = {obj for parent, obj in parent_edges if parent == bucket.id}
        assert attached, f"{bucket.id} has no objects attached"


def test_objects_carry_their_own_permissions() -> None:
    # The engine does not inherit rights through containment, so the normal
    # process must grant on objects explicitly. Without this, any object-level
    # right would look anomalous by itself and the bypass pattern would be
    # detectable without a model at all.
    _, events = _journal()
    with_rights = {
        event.object
        for event in events
        if event.relation is RelationType.HAS_PERMISSION
        and event.object.startswith("object:")
    }
    assert len(with_rights) > 10


def test_weekends_are_quieter_than_weekdays() -> None:
    _, events = _journal()
    weekend = sum(1 for event in events if is_weekend(event.ts))
    assert 0 < weekend < len(events) * 0.2


def test_generation_is_reproducible() -> None:
    assert _journal(3)[1] == _journal(3)[1]


def test_the_resulting_graph_is_not_trivial() -> None:
    _, events = _journal()
    graph = replay(events)
    assert graph.num_nodes > 50
    assert graph.num_edges > 50
```

- [ ] **Step 3: Run them to confirm they fail**

Run: `uv run pytest tests/util/test_timeutil.py tests/generator/test_timeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.util.timeutil'`

- [ ] **Step 4: Implement `timeutil.py`**

```python
# src/rga/util/timeutil.py
"""Time helpers. Every timestamp in this project is UTC milliseconds since epoch."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

SECOND_MS = 1_000
MINUTE_MS = 60 * SECOND_MS
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS


def day_start(start_ts: int, day_index: int) -> int:
    """Midnight of the given day, counted from `start_ts`."""
    return start_ts + day_index * DAY_MS


def hour_of_day(ts: int) -> int:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).hour


def is_weekend(ts: int) -> bool:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).weekday() >= 5


def sample_time_of_day(rng: np.random.Generator, day_start_ts: int, config) -> int:
    """Draw a moment within a day, concentrated in working hours.

    The diurnal rhythm is not decoration. "Granted at three in the morning" is
    only a signal if normal activity has a shape to deviate from.
    """
    low, high = config.working_hours
    if rng.random() < config.off_hours_rate:
        off_hours = [hour for hour in range(24) if not low <= hour < high]
        hour = int(off_hours[int(rng.integers(len(off_hours)))])
    else:
        hour = int(rng.integers(low, high))
    return (
        day_start_ts
        + hour * HOUR_MS
        + int(rng.integers(60)) * MINUTE_MS
        + int(rng.integers(60)) * SECOND_MS
    )
```

- [ ] **Step 5: Add `journal_issues` to `replay.py`**

Append to `src/rga/domain/replay.py`:

```python
def journal_issues(events: Iterable[GraphEvent]) -> list[str]:
    """Describe every internal inconsistency in a journal.

    Catches the mistakes a generator makes: revoking an edge that was never
    granted, and events arriving out of order. An empty list means the journal
    replays cleanly.
    """
    from rga.domain.events import EventOp

    issues: list[str] = []
    live: set[tuple[str, int, str]] = set()
    last_ts: int | None = None

    for position, event in enumerate(events):
        if last_ts is not None and event.ts < last_ts:
            issues.append(f"event {position} at ts={event.ts} precedes ts={last_ts}")
        last_ts = event.ts

        key = event.edge_key()
        if event.op is EventOp.GRANT:
            live.add(key)
        elif key not in live:
            issues.append(f"event {position} revokes an edge that is not live: {key}")
        else:
            live.discard(key)

    return issues
```

- [ ] **Step 6: Implement `timeline.py`**

```python
# src/rga/generator/timeline.py
"""The growth process: an organization living through time.

Day zero founds the company — a share of the staff, the team groups, the first
projects. After that each day draws hires, departures, uploads and grants from
Poisson counts, and the normal procedure for granting a right is always the same:
rights go to the team group, not to individuals. Deviations from that procedure
exist in the normal data too, at the configured rate, because an organization
where the procedure is never bypassed is not a realistic baseline.

Objects receive their own permission edges. The target engine does not inherit
rights through containment — a permission on a bucket does not cover its objects,
which is a deliberate decision of that project, verified against its test suite —
so the gateway writes an explicit permission after every upload. Reproducing that
matters: if the normal graph had no object-level rights, the bypass pattern would
be detectable by the mere presence of one.
"""

from __future__ import annotations

import numpy as np

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.config import TimelineConfig
from rga.generator.org import Organization
from rga.util.timeutil import DAY_MS, day_start, is_weekend, sample_time_of_day

#: Share of the staff already present when the timeline starts.
_FOUNDING_SHARE = 0.6

#: Levels a team group normally holds on its own buckets.
_TEAM_LEVELS = (PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.CREATE)


def generate_normal_journal(
    org: Organization,
    config: TimelineConfig,
    rng: np.random.Generator,
    *,
    exception_rate: float,
) -> list[GraphEvent]:
    """Produce the journal of a normally operating organization.

    `exception_rate` is the share of grants made outside the standard group
    procedure. It describes the company's habits rather than the simulation's
    intensity, so it lives in OrgConfig and is passed in explicitly.
    """
    events: list[GraphEvent] = []
    present: set[str] = set()
    pending_users = list(org.users)
    rng.shuffle(pending_users)  # type: ignore[arg-type]

    founding_count = max(1, int(len(pending_users) * _FOUNDING_SHARE))
    founders, newcomers = pending_users[:founding_count], pending_users[founding_count:]

    _found(events, org, founders, present, config, rng)
    _operate(events, org, newcomers, present, config, rng, exception_rate)

    events.sort(key=lambda event: event.ts)
    return events


def _found(
    events: list[GraphEvent],
    org: Organization,
    founders: list,
    present: set[str],
    config: TimelineConfig,
    rng: np.random.Generator,
) -> None:
    """Day zero: staff, groups, buckets and the standard grants."""
    start = day_start(config.start_ts, 0)

    for user in founders:
        ts = sample_time_of_day(rng, start, config)
        events.append(
            GraphEvent(ts, EventOp.GRANT, user.id, RelationType.MEMBER_OF,
                       org.team(user.team).group_id, actor="user:system")
        )
        present.add(user.id)

    for bucket in org.buckets:
        ts = sample_time_of_day(rng, start, config)
        events.append(
            GraphEvent(ts, EventOp.GRANT, bucket.owner, RelationType.OWNER_OF, bucket.id,
                       actor="user:system")
        )
        level = _TEAM_LEVELS[int(rng.integers(len(_TEAM_LEVELS)))]
        events.append(
            GraphEvent(ts, EventOp.GRANT, org.team(bucket.team).group_id,
                       RelationType.HAS_PERMISSION, bucket.id, level, actor=bucket.owner)
        )

    for group in org.cross_cutting:
        for member in group.members:
            ts = sample_time_of_day(rng, start, config)
            events.append(
                GraphEvent(ts, EventOp.GRANT, member, RelationType.MEMBER_OF, group.id,
                           actor="user:system")
            )
        for bucket_id in group.buckets:
            ts = sample_time_of_day(rng, start, config)
            events.append(
                GraphEvent(ts, EventOp.GRANT, group.id, RelationType.HAS_PERMISSION,
                           bucket_id, group.level, actor="user:system")
            )


def _operate(
    events: list[GraphEvent],
    org: Organization,
    newcomers: list,
    present: set[str],
    config: TimelineConfig,
    rng: np.random.Generator,
    exception_rate: float,
) -> None:
    """Days one onwards: hires, uploads, grants, departures."""
    queue = list(newcomers)
    pending_objects = [(bucket.id, obj) for bucket in org.buckets for obj in bucket.objects]
    rng.shuffle(pending_objects)  # type: ignore[arg-type]
    direct_grants: set[tuple[str, str]] = set()

    for day in range(1, config.days):
        start = day_start(config.start_ts, day)
        quiet = is_weekend(start)
        scale = config.weekend_rate if quiet else 1.0

        for _ in range(int(rng.poisson(config.hires_per_day * scale))):
            if not queue:
                break
            user = queue.pop()
            ts = sample_time_of_day(rng, start, config)
            events.append(
                GraphEvent(ts, EventOp.GRANT, user.id, RelationType.MEMBER_OF,
                           org.team(user.team).group_id, actor="user:system")
            )
            present.add(user.id)

        for _ in range(int(rng.poisson(config.uploads_per_day * scale))):
            if not pending_objects:
                break
            bucket_id, object_id = pending_objects.pop()
            ts = sample_time_of_day(rng, start, config)
            events.append(
                GraphEvent(ts, EventOp.GRANT, bucket_id, RelationType.PARENT_OF,
                           object_id, actor="user:system")
            )
            # Containment does not carry rights in the target engine: a
            # permission on a bucket does not cover its objects, and the gateway
            # writes an explicit one after every upload. The normal graph
            # therefore carries object-level permissions in bulk, and an
            # object-level right is not by itself unusual.
            bucket = org.bucket(bucket_id)
            events.append(
                GraphEvent(ts, EventOp.GRANT, org.team(bucket.team).group_id,
                           RelationType.HAS_PERMISSION, object_id,
                           _TEAM_LEVELS[int(rng.integers(len(_TEAM_LEVELS)))],
                           actor=bucket.owner)
            )

        for _ in range(int(rng.poisson(config.grants_per_day * scale))):
            grant = _draw_grant(org, present, config, rng, start, exception_rate)
            if grant is None:
                continue
            events.append(grant)
            if grant.subject.startswith("user:"):
                direct_grants.add((grant.subject, grant.object))

        for _ in range(int(rng.poisson(config.departures_per_day * scale))):
            if not present:
                break
            leaver = sorted(present)[int(rng.integers(len(present)))]
            ts = sample_time_of_day(rng, start, config)
            events.append(
                GraphEvent(ts, EventOp.REVOKE, leaver, RelationType.MEMBER_OF,
                           org.team_of(leaver).group_id, actor="user:system")
            )
            for subject, target in sorted(pair for pair in direct_grants if pair[0] == leaver):
                events.append(
                    GraphEvent(ts, EventOp.REVOKE, subject, RelationType.HAS_PERMISSION,
                               target, actor="user:system")
                )
                direct_grants.discard((subject, target))
            present.discard(leaver)


def _draw_grant(
    org: Organization,
    present: set[str],
    config: TimelineConfig,
    rng: np.random.Generator,
    day_start_ts: int,
    exception_rate: float,
) -> GraphEvent | None:
    """One right granted by the normal procedure, or a legitimate exception.

    The procedure grants to the team group. The exception grants directly to a
    person, sometimes across departments — rare, but normal.
    """
    if not present:
        return None
    ts = sample_time_of_day(rng, day_start_ts, config)
    bucket = org.buckets[int(rng.integers(len(org.buckets)))]
    approver = org.bucket(bucket.id).owner

    if rng.random() >= exception_rate:
        team = org.team(bucket.team)
        level = _TEAM_LEVELS[int(rng.integers(len(_TEAM_LEVELS)))]
        return GraphEvent(ts, EventOp.GRANT, team.group_id, RelationType.HAS_PERMISSION,
                          bucket.id, level, actor=approver)

    candidates = sorted(present)
    subject = candidates[int(rng.integers(len(candidates)))]
    return GraphEvent(ts, EventOp.GRANT, subject, RelationType.HAS_PERMISSION, bucket.id,
                      PermissionLevel.READ, actor=approver)
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/util tests/generator -v`
Expected: PASS. If `test_the_resulting_graph_is_not_trivial` fails, the intensities in the test's `TIMELINE` are too low for the org size — raise `grants_per_day`, not the assertion.

- [ ] **Step 8: Commit and push**

```bash
git add src/rga/util/timeutil.py src/rga/generator/timeline.py src/rga/domain/replay.py tests/util/test_timeutil.py tests/generator/test_timeline.py
git commit -m "feat: add organization growth process"
git push origin main
```

---

## Task 10: Anomaly pattern framework

Each pattern is a parameterised procedure that plants a group of related events in
the evaluation window and returns exact labels for the edges it created. Patterns
know nothing about each other and are looked up by name, so the config file lists
which ones a dataset contains.

**Files:**
- Create: `src/rga/generator/anomalies/__init__.py`, `src/rga/generator/anomalies/base.py`
- Test: `tests/generator/anomalies/test_base.py`

**Interfaces:**
- Consumes: `AccessGraph` (Task 5); `Organization` (Task 8); `GraphEvent`, `EventOp` (Task 4)
- Produces:
  - `NoCandidateError`
  - `AnomalyLabel(ts, subject, relation, object, pattern)` with `edge_key()`, `to_dict()`, `from_dict()`
  - `Injection(events, labels)`
  - `InjectionContext(rng, org, graph, window)`
  - `AnomalyPattern` Protocol with class attribute `name: str` and `inject(context) -> Injection`
  - `register(cls)` decorator, `get_pattern(name)`, `available_patterns()`
  - helpers `pick(rng, sequence)`, `sample_ts(rng, window)`, `sample_night_ts(rng, window)`, `level_on(graph, subject_id, object_id)`, `last_activity(graph, subject_id)`, `label_for(event, pattern)`

- [ ] **Step 1: Write the failing test**

```python
# tests/generator/anomalies/test_base.py
"""Pattern registry and the helpers every pattern uses."""

import numpy as np
import pytest

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.domain.replay import replay
from rga.generator.anomalies.base import (
    AnomalyLabel,
    available_patterns,
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
            GraphEvent(1_000, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
                       "bucket:x", PermissionLevel.READ),
            GraphEvent(2_000, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
                       "bucket:y", PermissionLevel.ADMIN),
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
    event = GraphEvent(5_000, EventOp.GRANT, "user:a", RelationType.HAS_PERMISSION,
                       "bucket:x", PermissionLevel.ADMIN, actor="user:a")
    label = label_for(event, "demo")
    assert label == AnomalyLabel(5_000, "user:a", RelationType.HAS_PERMISSION, "bucket:x", "demo")
    assert label.edge_key() == event.edge_key()


def test_label_round_trips_through_a_dict() -> None:
    label = AnomalyLabel(5_000, "user:a", RelationType.HAS_PERMISSION, "bucket:x", "demo")
    assert AnomalyLabel.from_dict(label.to_dict()) == label


def test_unknown_pattern_is_reported_by_name() -> None:
    with pytest.raises(KeyError, match="unknown anomaly pattern"):
        get_pattern("does_not_exist")


def test_all_eight_patterns_register() -> None:
    import rga.generator.anomalies  # noqa: F401  registers every pattern module

    assert set(available_patterns()) == {
        "self_grant_admin",
        "privileged_group_join",
        "grant_burst",
        "dormant_awakening",
        "hierarchy_bypass",
        "cross_department",
        "shadow_group",
        "delegation_cascade",
    }
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/generator/anomalies -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.generator.anomalies'`

- [ ] **Step 3: Implement `base.py`**

```python
# src/rga/generator/anomalies/base.py
"""The anomaly pattern contract.

A pattern receives the graph as it stood at the start of the evaluation window
and plants a group of related events inside that window, returning exact labels
for the edges it created. Patterns are stateless and independent; the dataset
assembler decides how many of each to inject.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

from rga.domain.events import GraphEvent
from rga.domain.graph import AccessGraph
from rga.domain.relations import PermissionLevel, RelationType, parse_relation
from rga.generator.org import Organization
from rga.util.timeutil import DAY_MS, HOUR_MS, MINUTE_MS

T = TypeVar("T")

#: How many times a pattern retries candidate selection before giving up.
ATTEMPTS = 64


class NoCandidateError(RuntimeError):
    """A pattern found no suitable place to inject itself.

    Expected on small graphs; the assembler skips the pattern and tries another.
    """


@dataclass(frozen=True)
class AnomalyLabel:
    """Ground truth: one edge that a pattern created."""

    ts: int
    subject: str
    relation: RelationType
    object: str
    pattern: str

    def edge_key(self) -> tuple[str, int, str]:
        return (self.subject, int(self.relation), self.object)

    def to_dict(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "subject": self.subject,
            "relation": self.relation.name,
            "object": self.object,
            "pattern": self.pattern,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AnomalyLabel:
        return cls(
            ts=int(payload["ts"]),  # type: ignore[arg-type]
            subject=str(payload["subject"]),
            relation=parse_relation(str(payload["relation"])),
            object=str(payload["object"]),
            pattern=str(payload["pattern"]),
        )


@dataclass(frozen=True)
class Injection:
    """What one pattern produced."""

    events: tuple[GraphEvent, ...]
    labels: tuple[AnomalyLabel, ...]


@dataclass(frozen=True)
class InjectionContext:
    """Everything a pattern may look at."""

    rng: np.random.Generator
    org: Organization
    #: The graph as it stood at the start of the window.
    graph: AccessGraph
    #: Half-open [start, end) of the evaluation window, in milliseconds.
    window: tuple[int, int]


class AnomalyPattern(Protocol):
    """One threat scenario."""

    name: str

    def inject(self, context: InjectionContext) -> Injection:
        """Plant the pattern, or raise NoCandidateError if the graph has no room."""
        ...


REGISTRY: dict[str, AnomalyPattern] = {}


def register(cls: type) -> type:
    """Class decorator registering a stateless pattern under its name."""
    instance = cls()
    if instance.name in REGISTRY:
        raise ValueError(f"duplicate anomaly pattern name: {instance.name!r}")
    REGISTRY[instance.name] = instance
    return cls


def get_pattern(name: str) -> AnomalyPattern:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown anomaly pattern: {name!r}") from None


def available_patterns() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def pick(rng: np.random.Generator, sequence: Sequence[T]) -> T:
    """Draw one element.

    Written by hand rather than with `rng.choice` because numpy coerces a
    sequence of dataclasses or strings into an object array and silently
    changes their type.
    """
    if not sequence:
        raise NoCandidateError("cannot pick from an empty sequence")
    return sequence[int(rng.integers(len(sequence)))]


def sample_ts(rng: np.random.Generator, window: tuple[int, int]) -> int:
    """A moment uniformly inside the window."""
    start, end = window
    return int(rng.integers(start, end))


def sample_night_ts(rng: np.random.Generator, window: tuple[int, int]) -> int:
    """A moment inside the window, forced into the small hours."""
    start, end = window
    day_count = max(1, (end - start) // DAY_MS)
    midnight = start - (start % DAY_MS) + int(rng.integers(day_count)) * DAY_MS
    ts = midnight + int(rng.integers(5)) * HOUR_MS + int(rng.integers(60)) * MINUTE_MS
    return int(min(max(ts, start), end - 1))


def level_on(graph: AccessGraph, subject_id: str, object_id: str) -> PermissionLevel:
    """Highest permission the subject holds directly on the object."""
    if subject_id not in graph.node_index or object_id not in graph.node_index:
        return PermissionLevel.NONE
    mask = (
        (graph.edge_src == graph.index_of(subject_id))
        & (graph.edge_dst == graph.index_of(object_id))
        & (graph.edge_rel == int(RelationType.HAS_PERMISSION))
    )
    levels = graph.edge_level[mask]
    return PermissionLevel(int(levels.max())) if levels.size else PermissionLevel.NONE


def last_activity(graph: AccessGraph, subject_id: str) -> int:
    """Creation time of the subject's newest outgoing edge, or -1 if it has none."""
    if subject_id not in graph.node_index:
        return -1
    times = graph.edge_created[graph.edge_src == graph.index_of(subject_id)]
    return int(times.max()) if times.size else -1


def label_for(event: GraphEvent, pattern: str) -> AnomalyLabel:
    """Ground-truth label for an event a pattern created."""
    return AnomalyLabel(
        ts=event.ts,
        subject=event.subject,
        relation=event.relation,
        object=event.object,
        pattern=pattern,
    )
```

- [ ] **Step 4: Write the package init that registers every pattern**

```python
# src/rga/generator/anomalies/__init__.py
"""Anomaly patterns.

Importing this package registers every pattern. Modules are imported for their
side effect, so the star of unused imports below is deliberate.
"""

from rga.generator.anomalies import (  # noqa: F401
    bypass,
    compromise,
    escalation,
    persistence,
)
from rga.generator.anomalies.base import (  # noqa: F401
    AnomalyLabel,
    AnomalyPattern,
    Injection,
    InjectionContext,
    NoCandidateError,
    available_patterns,
    get_pattern,
)
```

Note: this import will fail until Tasks 11–14 create the four modules. Write the
init now but expect `tests/generator/anomalies/test_base.py::test_all_eight_patterns_register`
to stay red until Task 14. Every other test in the file must pass.

- [ ] **Step 5: Create the four pattern modules as empty placeholders so the init imports**

```bash
for module in escalation compromise bypass persistence; do
  printf '"""Anomaly patterns: %s."""\n' "$module" > "src/rga/generator/anomalies/$module.py"
done
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/generator/anomalies -v`
Expected: all PASS except `test_all_eight_patterns_register`, which fails with an
empty set. That single failure is expected and closes in Task 14.

- [ ] **Step 7: Commit and push**

```bash
git add src/rga/generator/anomalies tests/generator/anomalies
git commit -m "feat: add anomaly pattern framework and registry"
git push origin main
```

---

## Task 11: Escalation patterns

**Files:**
- Modify: `src/rga/generator/anomalies/escalation.py`
- Test: `tests/generator/anomalies/test_escalation.py`

**Interfaces:**
- Consumes: everything from Task 10
- Produces: patterns `self_grant_admin` and `privileged_group_join`

- [ ] **Step 1: Write the failing test**

```python
# tests/generator/anomalies/test_escalation.py
"""Vertical escalation: a subject acquires rights it should not be able to give itself."""

import pytest

from rga.domain.events import EventOp
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import NoCandidateError, get_pattern


def test_self_grant_admin_makes_subject_its_own_actor(context_factory) -> None:
    context = context_factory(seed=11)
    injection = get_pattern("self_grant_admin").inject(context)

    assert len(injection.events) == 1
    event = injection.events[0]
    assert event.op is EventOp.GRANT
    assert event.relation is RelationType.HAS_PERMISSION
    assert event.level is PermissionLevel.ADMIN
    assert event.actor == event.subject


def test_self_grant_admin_targets_a_bucket_it_did_not_already_administer(context_factory) -> None:
    context = context_factory(seed=12)
    event = get_pattern("self_grant_admin").inject(context).events[0]
    from rga.generator.anomalies.base import level_on

    assert level_on(context.graph, event.subject, event.object) < PermissionLevel.ADMIN


def test_privileged_group_join_adds_a_non_member_to_a_cross_cutting_group(context_factory) -> None:
    context = context_factory(seed=13)
    injection = get_pattern("privileged_group_join").inject(context)

    event = injection.events[0]
    assert event.relation is RelationType.MEMBER_OF
    assert event.actor == event.subject
    group_ids = {group.id for group in context.org.cross_cutting}
    assert event.object in group_ids
    joined = next(group for group in context.org.cross_cutting if group.id == event.object)
    assert event.subject not in joined.members


def test_events_land_inside_the_window(context_factory) -> None:
    context = context_factory(seed=14)
    for name in ("self_grant_admin", "privileged_group_join"):
        for event in get_pattern(name).inject(context).events:
            assert context.window[0] <= event.ts < context.window[1]


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=15)
    for name in ("self_grant_admin", "privileged_group_join"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)


def test_privileged_group_join_gives_up_without_cross_cutting_groups(context_factory) -> None:
    context = context_factory(seed=16, cross_cutting_groups=0)
    with pytest.raises(NoCandidateError):
        get_pattern("privileged_group_join").inject(context)


def test_injection_is_reproducible(context_factory) -> None:
    first = get_pattern("self_grant_admin").inject(context_factory(seed=17))
    second = get_pattern("self_grant_admin").inject(context_factory(seed=17))
    assert first == second
```

- [ ] **Step 2: Write the shared test fixture**

```python
# tests/generator/anomalies/conftest.py
"""A small but realistic graph for pattern tests.

Exposed as a fixture rather than an importable helper: pytest puts this
directory on sys.path, not the repository root, so a cross-package import of a
conftest module would not resolve.
"""

import numpy as np
import pytest

import rga.generator.anomalies  # noqa: F401  registers the patterns
from rga.domain.replay import replay
from rga.generator.anomalies.base import InjectionContext
from rga.generator.config import OrgConfig, TimelineConfig
from rga.generator.org import build_organization
from rga.generator.timeline import generate_normal_journal
from rga.util.timeutil import DAY_MS

START = 1_735_689_600_000
TRAIN_DAYS = 60
WINDOW_DAYS = 14


@pytest.fixture
def context_factory():
    """Return a builder for the context patterns receive."""

    def build(seed: int, *, cross_cutting_groups: int = 2) -> InjectionContext:
        org_config = OrgConfig(
            departments=3,
            teams_per_department=(2, 4),
            users_per_team=(4, 8),
            projects_per_team=(1, 3),
            objects_per_bucket=(5, 20),
            cross_cutting_groups=cross_cutting_groups,
            cross_cutting_membership_rate=0.1,
            legitimate_exception_rate=0.04,
        )
        timeline_config = TimelineConfig(
            start_ts=START,
            days=TRAIN_DAYS,
            working_hours=(9, 19),
            off_hours_rate=0.05,
            weekend_rate=0.08,
            hires_per_day=0.6,
            departures_per_day=0.1,
            grants_per_day=5.0,
            uploads_per_day=15.0,
        )

        rng = np.random.default_rng(seed)
        org = build_organization(org_config, rng)
        journal = generate_normal_journal(
            org, timeline_config, rng, exception_rate=org_config.legitimate_exception_rate
        )
        cutoff = START + TRAIN_DAYS * DAY_MS
        return InjectionContext(
            rng=rng,
            org=org,
            graph=replay(journal, until=cutoff),
            window=(cutoff, cutoff + WINDOW_DAYS * DAY_MS),
        )

    return build
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `uv run pytest tests/generator/anomalies/test_escalation.py -v`
Expected: FAIL — `KeyError: unknown anomaly pattern: 'self_grant_admin'`

- [ ] **Step 4: Implement the patterns**

```python
# src/rga/generator/anomalies/escalation.py
"""Vertical escalation.

Both patterns model a subject acquiring rights through a path the standard
procedure does not provide. What makes them detectable structurally is not the
level itself — plenty of people legitimately hold admin — but that the right
appears without the surrounding structure that normally accompanies it.
"""

from __future__ import annotations

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import (
    ATTEMPTS,
    Injection,
    InjectionContext,
    NoCandidateError,
    label_for,
    level_on,
    pick,
    register,
    sample_night_ts,
    sample_ts,
)


@register
class SelfGrantAdmin:
    """A user grants itself administrative rights on a bucket of its own team.

    The subject already has ordinary access through the team group, so the new
    edge is not an impossible one — it is one that nobody else in a comparable
    position holds, and that the subject issued for itself.
    """

    name = "self_grant_admin"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if not users:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            user = pick(context.rng, users)
            team = context.org.team_of(user.id)
            if not team.buckets:
                continue
            bucket = pick(context.rng, team.buckets)
            if level_on(context.graph, user.id, bucket) >= PermissionLevel.ADMIN:
                continue

            event = GraphEvent(
                ts=sample_night_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=bucket,
                level=PermissionLevel.ADMIN,
                actor=user.id,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)


@register
class PrivilegedGroupJoin:
    """A user joins a cross-cutting group holding elevated rights.

    The group is legitimate and so is membership in it — for the people who went
    through the process. The anomaly is the membership appearing on its own,
    issued by the joiner, without the department or role that normally precedes it.
    """

    name = "privileged_group_join"

    def inject(self, context: InjectionContext) -> Injection:
        if not context.org.cross_cutting:
            raise NoCandidateError(self.name)
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if not users:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            group = pick(context.rng, context.org.cross_cutting)
            user = pick(context.rng, users)
            if user.id in group.members:
                continue

            event = GraphEvent(
                ts=sample_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.MEMBER_OF,
                object=group.id,
                actor=user.id,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/generator/anomalies/test_escalation.py -v`
Expected: PASS

- [ ] **Step 6: Commit and push**

```bash
git add src/rga/generator/anomalies/escalation.py tests/generator/anomalies
git commit -m "feat: add escalation anomaly patterns"
git push origin main
```

---

## Task 12: Compromise patterns

**Files:**
- Modify: `src/rga/generator/anomalies/compromise.py`
- Test: `tests/generator/anomalies/test_compromise.py`

**Interfaces:**
- Consumes: everything from Task 10; the `context_factory` fixture from `tests/generator/anomalies/conftest.py`
- Produces: patterns `grant_burst` and `dormant_awakening`

- [ ] **Step 1: Write the failing test**

```python
# tests/generator/anomalies/test_compromise.py
"""Account compromise: rights appear at a rate or after a silence that does not fit."""

from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import get_pattern, last_activity
from rga.util.timeutil import DAY_MS, MINUTE_MS


def test_grant_burst_comes_from_one_subject_in_a_short_span(context_factory) -> None:
    injection = get_pattern("grant_burst").inject(context_factory(seed=21))

    assert len(injection.events) >= 6
    assert len({event.subject for event in injection.events}) == 1
    span = max(event.ts for event in injection.events) - min(
        event.ts for event in injection.events
    )
    assert span <= 45 * MINUTE_MS


def test_grant_burst_targets_buckets_outside_the_subjects_team(context_factory) -> None:
    context = context_factory(seed=22)
    injection = get_pattern("grant_burst").inject(context)

    subject = injection.events[0].subject
    own_team = context.org.team_of(subject).id
    for event in injection.events:
        assert context.org.bucket(event.object).team != own_team


def test_grant_burst_events_are_ordered_and_inside_the_window(context_factory) -> None:
    context = context_factory(seed=23)
    events = get_pattern("grant_burst").inject(context).events
    assert [event.ts for event in events] == sorted(event.ts for event in events)
    assert all(context.window[0] <= event.ts < context.window[1] for event in events)


def test_dormant_awakening_picks_a_long_silent_account(context_factory) -> None:
    context = context_factory(seed=24)
    event = get_pattern("dormant_awakening").inject(context).events[0]

    silence = context.window[0] - last_activity(context.graph, event.subject)
    assert silence >= 21 * DAY_MS
    assert event.relation is RelationType.HAS_PERMISSION
    assert event.level >= PermissionLevel.WRITE


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=25)
    for name in ("grant_burst", "dormant_awakening"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)


def test_patterns_are_reproducible(context_factory) -> None:
    for name in ("grant_burst", "dormant_awakening"):
        assert get_pattern(name).inject(context_factory(seed=26)) == get_pattern(name).inject(
            context_factory(seed=26)
        )
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/generator/anomalies/test_compromise.py -v`
Expected: FAIL — `KeyError: unknown anomaly pattern: 'grant_burst'`

- [ ] **Step 3: Implement the patterns**

```python
# src/rga/generator/anomalies/compromise.py
"""Account compromise.

Neither pattern creates an impossible right. What marks them is the rhythm: a
volume of grants no single person produces in an hour, or activity from an
account that has been silent for weeks.
"""

from __future__ import annotations

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import (
    ATTEMPTS,
    Injection,
    InjectionContext,
    NoCandidateError,
    label_for,
    last_activity,
    level_on,
    pick,
    register,
    sample_night_ts,
    sample_ts,
)
from rga.util.timeutil import DAY_MS, MINUTE_MS

#: How many grants a compromised account issues.
_BURST_SIZE = (6, 16)
#: How long the burst lasts.
_BURST_SPAN_MS = 45 * MINUTE_MS
#: Silence after which an account counts as dormant.
_DORMANT_MS = 21 * DAY_MS


@register
class GrantBurst:
    """One subject grants itself rights on many foreign buckets within an hour."""

    name = "grant_burst"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        window_start, window_end = context.window
        if window_end - window_start <= _BURST_SPAN_MS:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            user = pick(context.rng, users)
            own_team = context.org.team_of(user.id).id
            foreign = [
                bucket.id for bucket in context.org.buckets if bucket.team != own_team
            ]
            if len(foreign) < _BURST_SIZE[0]:
                continue

            count = min(
                int(context.rng.integers(_BURST_SIZE[0], _BURST_SIZE[1] + 1)), len(foreign)
            )
            order = context.rng.permutation(len(foreign))[:count]
            start = sample_ts(context.rng, (window_start, window_end - _BURST_SPAN_MS))

            events = [
                GraphEvent(
                    ts=start + int(context.rng.integers(_BURST_SPAN_MS)),
                    op=EventOp.GRANT,
                    subject=user.id,
                    relation=RelationType.HAS_PERMISSION,
                    object=foreign[int(index)],
                    level=PermissionLevel.WRITE,
                    actor=user.id,
                )
                for index in order
            ]
            events.sort(key=lambda event: event.ts)
            return Injection(
                tuple(events), tuple(label_for(event, self.name) for event in events)
            )

        raise NoCandidateError(self.name)


@register
class DormantAwakening:
    """An account silent for weeks suddenly takes an elevated right."""

    name = "dormant_awakening"

    def inject(self, context: InjectionContext) -> Injection:
        threshold = context.window[0] - _DORMANT_MS
        dormant = [
            user
            for user in context.org.users
            if user.id in context.graph.node_index
            and 0 <= last_activity(context.graph, user.id) <= threshold
        ]
        if not dormant:
            raise NoCandidateError(self.name)

        for _ in range(ATTEMPTS):
            user = pick(context.rng, dormant)
            bucket = pick(context.rng, context.org.buckets)
            if level_on(context.graph, user.id, bucket.id) >= PermissionLevel.WRITE:
                continue

            event = GraphEvent(
                ts=sample_night_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=bucket.id,
                level=PermissionLevel.WRITE,
                actor=user.id,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/generator/anomalies/test_compromise.py -v`
Expected: PASS. If `test_dormant_awakening_picks_a_long_silent_account` raises
`NoCandidateError`, the fixture's 40-day span leaves too few silent accounts — raise
`TRAIN_DAYS` in `conftest.py`, do not lower `_DORMANT_MS`.

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/generator/anomalies/compromise.py tests/generator/anomalies/test_compromise.py
git commit -m "feat: add compromise anomaly patterns"
git push origin main
```

---

## Task 13: Bypass patterns

**Files:**
- Modify: `src/rga/generator/anomalies/bypass.py`
- Test: `tests/generator/anomalies/test_bypass.py`

**Interfaces:**
- Consumes: everything from Task 10; `parent_bucket`, `entity_type`, `EntityType` (Task 3)
- Produces: patterns `hierarchy_bypass` and `cross_department`

Both patterns deliberately use a plausible approver as the actor rather than the
subject itself. If every pattern made the subject its own actor, provenance alone
would separate anomalies from normal data and the structural model would never be
tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/generator/anomalies/test_bypass.py
"""Bypassing the delegation hierarchy, horizontally and vertically."""

from rga.domain.entities import EntityType, entity_type, parent_bucket
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import get_pattern, level_on


def test_hierarchy_bypass_grants_on_an_object_not_its_bucket(context_factory) -> None:
    context = context_factory(seed=31)
    event = get_pattern("hierarchy_bypass").inject(context).events[0]

    assert entity_type(event.object) is EntityType.OBJECT
    assert event.relation is RelationType.HAS_PERMISSION
    # Object-level rights are ordinary; holding one with nothing on the
    # containing bucket is not.
    assert level_on(context.graph, event.subject, parent_bucket(event.object)) is (
        PermissionLevel.NONE
    )


def test_hierarchy_bypass_is_not_issued_by_the_subject(context_factory) -> None:
    context = context_factory(seed=32)
    event = get_pattern("hierarchy_bypass").inject(context).events[0]
    assert event.actor is not None
    assert event.actor != event.subject


def test_cross_department_crosses_a_department_boundary(context_factory) -> None:
    context = context_factory(seed=33)
    event = get_pattern("cross_department").inject(context).events[0]

    subject_department = context.org.department_of(event.subject)
    bucket = context.org.bucket(event.object)
    target_department = context.org.team(bucket.team).department
    assert subject_department != target_department


def test_cross_department_approver_belongs_to_the_target_department(context_factory) -> None:
    context = context_factory(seed=34)
    event = get_pattern("cross_department").inject(context).events[0]

    bucket = context.org.bucket(event.object)
    target_department = context.org.team(bucket.team).department
    assert context.org.department_of(event.actor) == target_department
    assert event.actor != bucket.owner


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=35)
    for name in ("hierarchy_bypass", "cross_department"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)


def test_events_land_inside_the_window(context_factory) -> None:
    context = context_factory(seed=36)
    for name in ("hierarchy_bypass", "cross_department"):
        for event in get_pattern(name).inject(context).events:
            assert context.window[0] <= event.ts < context.window[1]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/generator/anomalies/test_bypass.py -v`
Expected: FAIL — `KeyError: unknown anomaly pattern: 'hierarchy_bypass'`

- [ ] **Step 3: Implement the patterns**

```python
# src/rga/generator/anomalies/bypass.py
"""Bypassing the delegation hierarchy.

The actor in both patterns is a plausible approver rather than the subject. If
every pattern made the subject its own actor, provenance alone would separate
anomalies from normal data and the structural model would never be exercised.
"""

from __future__ import annotations

from rga.domain.entities import EntityType, entity_type, parent_bucket
from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import (
    ATTEMPTS,
    Injection,
    InjectionContext,
    NoCandidateError,
    label_for,
    level_on,
    pick,
    register,
    sample_ts,
)


@register
class HierarchyBypass:
    """A right on an object, granted to someone with no right on its bucket.

    Object-level rights are ordinary here — the engine does not inherit through
    containment, so every upload produces one. What is not ordinary is who holds
    it: normally the right goes to the group that owns the containing bucket, and
    whoever holds it on the object holds something on the bucket too. This edge
    lands on a leaf, held by a subject with nothing above it.
    """

    name = "hierarchy_bypass"

    def inject(self, context: InjectionContext) -> Injection:
        objects = [
            node_id
            for node_id in context.graph.node_ids
            if entity_type(node_id) is EntityType.OBJECT
        ]
        if not objects:
            raise NoCandidateError(self.name)
        users = [user for user in context.org.users if user.id in context.graph.node_index]

        for _ in range(ATTEMPTS):
            target = pick(context.rng, objects)
            bucket_id = parent_bucket(target)
            user = pick(context.rng, users)
            if level_on(context.graph, user.id, bucket_id) is not PermissionLevel.NONE:
                continue
            if level_on(context.graph, user.id, target) is not PermissionLevel.NONE:
                continue
            try:
                approver = context.org.bucket(bucket_id).owner
            except KeyError:
                continue
            if approver == user.id:
                continue

            event = GraphEvent(
                ts=sample_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=target,
                level=PermissionLevel.WRITE,
                actor=approver,
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)


@register
class CrossDepartment:
    """A right on another department's bucket, with no shared structure to justify it.

    Rights across departments exist in the normal data too — that is what the
    legitimate-exception rate produces. The difference is the surrounding
    structure: a normal exception sits near shared groups and shared projects,
    this one sits alone.
    """

    name = "cross_department"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]

        for _ in range(ATTEMPTS):
            user = pick(context.rng, users)
            department = context.org.department_of(user.id)
            foreign = [
                bucket
                for bucket in context.org.buckets
                if context.org.team(bucket.team).department != department
            ]
            if not foreign:
                continue
            bucket = pick(context.rng, foreign)
            if level_on(context.graph, user.id, bucket.id) is not PermissionLevel.NONE:
                continue

            target_department = context.org.team(bucket.team).department
            approvers = [
                candidate.id
                for candidate in context.org.users
                if candidate.department == target_department
                and candidate.id != bucket.owner
                and candidate.id in context.graph.node_index
            ]
            if not approvers:
                continue

            event = GraphEvent(
                ts=sample_ts(context.rng, context.window),
                op=EventOp.GRANT,
                subject=user.id,
                relation=RelationType.HAS_PERMISSION,
                object=bucket.id,
                level=PermissionLevel.WRITE,
                actor=pick(context.rng, approvers),
            )
            return Injection((event,), (label_for(event, self.name),))

        raise NoCandidateError(self.name)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/generator/anomalies/test_bypass.py -v`
Expected: PASS

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/generator/anomalies/bypass.py tests/generator/anomalies/test_bypass.py
git commit -m "feat: add hierarchy bypass anomaly patterns"
git push origin main
```

---

## Task 14: Persistence patterns

These two are held out from the supervised baseline's training set, together with
`dormant_awakening`. They are the generalisation test of section 9 of the spec, so
they must not resemble the first five in shape — both create *new structure* rather
than a new edge between existing nodes.

**Files:**
- Modify: `src/rga/generator/anomalies/persistence.py`
- Test: `tests/generator/anomalies/test_persistence.py`

**Interfaces:**
- Consumes: everything from Task 10
- Produces: patterns `shadow_group` and `delegation_cascade`

- [ ] **Step 1: Write the failing test**

```python
# tests/generator/anomalies/test_persistence.py
"""Persistence: structure created to keep access after the entry point is closed."""

from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import get_pattern
from rga.util.timeutil import HOUR_MS


def test_shadow_group_has_exactly_one_member(context_factory) -> None:
    injection = get_pattern("shadow_group").inject(context_factory(seed=41))

    memberships = [
        event for event in injection.events if event.relation is RelationType.MEMBER_OF
    ]
    assert len(memberships) == 1
    group_id = memberships[0].object
    assert group_id.startswith("group:svc-")


def test_shadow_group_takes_admin_on_several_buckets(context_factory) -> None:
    injection = get_pattern("shadow_group").inject(context_factory(seed=42))

    grants = [
        event for event in injection.events if event.relation is RelationType.HAS_PERMISSION
    ]
    assert len(grants) >= 4
    assert all(event.level is PermissionLevel.ADMIN for event in grants)
    assert len({event.subject for event in grants}) == 1


def test_shadow_group_is_a_new_node(context_factory) -> None:
    context = context_factory(seed=43)
    injection = get_pattern("shadow_group").inject(context)
    group_id = next(
        event.object for event in injection.events if event.relation is RelationType.MEMBER_OF
    )
    assert group_id not in context.graph.node_index


def test_delegation_cascade_chains_actor_to_previous_subject(context_factory) -> None:
    injection = get_pattern("delegation_cascade").inject(context_factory(seed=44))
    events = injection.events

    assert len(events) == 3
    assert len({event.object for event in events}) == 1
    for earlier, later in zip(events, events[1:], strict=True):
        assert later.actor == earlier.subject


def test_delegation_cascade_runs_quickly_and_in_order(context_factory) -> None:
    context = context_factory(seed=45)
    events = get_pattern("delegation_cascade").inject(context).events

    assert [event.ts for event in events] == sorted(event.ts for event in events)
    assert events[-1].ts - events[0].ts <= 2 * HOUR_MS
    assert all(context.window[0] <= event.ts < context.window[1] for event in events)


def test_delegation_cascade_uses_distinct_subjects(context_factory) -> None:
    events = get_pattern("delegation_cascade").inject(context_factory(seed=46)).events
    assert len({event.subject for event in events}) == len(events)


def test_labels_cover_every_created_edge(context_factory) -> None:
    context = context_factory(seed=47)
    for name in ("shadow_group", "delegation_cascade"):
        injection = get_pattern(name).inject(context)
        assert {label.edge_key() for label in injection.labels} == {
            event.edge_key() for event in injection.events
        }
        assert all(label.pattern == name for label in injection.labels)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/generator/anomalies/test_persistence.py -v`
Expected: FAIL — `KeyError: unknown anomaly pattern: 'shadow_group'`

- [ ] **Step 3: Implement the patterns**

```python
# src/rga/generator/anomalies/persistence.py
"""Establishing durable access.

Both patterns create new structure rather than a new edge between existing
nodes, which is what makes them a fair generalisation test: a model that learned
the first five patterns has seen nothing shaped like these.
"""

from __future__ import annotations

from rga.domain.events import EventOp, GraphEvent
from rga.domain.relations import PermissionLevel, RelationType
from rga.generator.anomalies.base import (
    Injection,
    InjectionContext,
    NoCandidateError,
    label_for,
    pick,
    register,
    sample_night_ts,
    sample_ts,
)
from rga.util.timeutil import HOUR_MS, MINUTE_MS

#: How many buckets a shadow group takes rights on.
_SHADOW_BUCKETS = (4, 10)
#: How many people a delegation chain passes through.
_CHAIN_LENGTH = 4
#: Spacing between the steps of a chain.
_CHAIN_STEP_MS = 20 * MINUTE_MS


@register
class ShadowGroup:
    """A fresh group with a single member and administrative rights on many buckets.

    Groups exist to be shared. One that holds broad rights and has exactly one
    member is not serving the purpose groups serve.
    """

    name = "shadow_group"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if not users or len(context.org.buckets) < _SHADOW_BUCKETS[0]:
            raise NoCandidateError(self.name)

        owner = pick(context.rng, users)
        group_id = f"group:svc-{int(context.rng.integers(1 << 32)):08x}"
        window_start, window_end = context.window
        start = sample_night_ts(context.rng, (window_start, window_end - HOUR_MS))

        events = [
            GraphEvent(
                ts=start,
                op=EventOp.GRANT,
                subject=owner.id,
                relation=RelationType.MEMBER_OF,
                object=group_id,
                actor=owner.id,
            )
        ]

        count = min(
            int(context.rng.integers(_SHADOW_BUCKETS[0], _SHADOW_BUCKETS[1] + 1)),
            len(context.org.buckets),
        )
        for offset, index in enumerate(context.rng.permutation(len(context.org.buckets))[:count]):
            ts = min(start + (offset + 1) * int(context.rng.integers(1, 20)) * MINUTE_MS,
                     window_end - 1)
            events.append(
                GraphEvent(
                    ts=ts,
                    op=EventOp.GRANT,
                    subject=group_id,
                    relation=RelationType.HAS_PERMISSION,
                    object=context.org.buckets[int(index)].id,
                    level=PermissionLevel.ADMIN,
                    actor=owner.id,
                )
            )

        events.sort(key=lambda event: event.ts)
        return Injection(tuple(events), tuple(label_for(event, self.name) for event in events))


@register
class DelegationCascade:
    """Administrative rights on one bucket passed along a chain of people in an hour.

    Each step is a legitimate delegation in isolation. The chain is what is wrong:
    rights move hand to hand faster than any approval process works.
    """

    name = "delegation_cascade"

    def inject(self, context: InjectionContext) -> Injection:
        users = [user for user in context.org.users if user.id in context.graph.node_index]
        if len(users) < _CHAIN_LENGTH:
            raise NoCandidateError(self.name)

        window_start, window_end = context.window
        span = _CHAIN_LENGTH * _CHAIN_STEP_MS
        if window_end - window_start <= span:
            raise NoCandidateError(self.name)

        order = context.rng.permutation(len(users))[:_CHAIN_LENGTH]
        chain = [users[int(index)].id for index in order]
        bucket = pick(context.rng, context.org.buckets).id
        start = sample_ts(context.rng, (window_start, window_end - span))

        events = []
        actor = chain[0]
        for step in range(1, _CHAIN_LENGTH):
            events.append(
                GraphEvent(
                    ts=start + step * _CHAIN_STEP_MS,
                    op=EventOp.GRANT,
                    subject=chain[step],
                    relation=RelationType.HAS_PERMISSION,
                    object=bucket,
                    level=PermissionLevel.ADMIN,
                    actor=actor,
                )
            )
            actor = chain[step]

        return Injection(tuple(events), tuple(label_for(event, self.name) for event in events))
```

- [ ] **Step 4: Run the whole anomaly suite**

Run: `uv run pytest tests/generator/anomalies -v`
Expected: PASS, including `test_all_eight_patterns_register` from Task 10, which has
been red since then.

- [ ] **Step 5: Commit and push**

```bash
git add src/rga/generator/anomalies/persistence.py tests/generator/anomalies/test_persistence.py
git commit -m "feat: add persistence anomaly patterns"
git push origin main
```

---

## Task 15: Dataset assembly

Brings the pieces together: a normal journal, a temporal split, anomalies planted
in the evaluation window, and a directory on disk that reproduces exactly.

**Files:**
- Create: `src/rga/generator/dataset.py`, `src/rga/io/dataset_io.py`
- Modify: `src/rga/generator/config.py` (extract `dataset_config_from_document`)
- Test: `tests/generator/test_dataset.py`, `tests/io/test_dataset_io.py`

**Interfaces:**
- Consumes: everything from Tasks 8–14
- Produces:
  - `dataset_config_from_document(document: Mapping[str, object]) -> DatasetConfig` in `config.py`, with `load_dataset_config` reduced to reading the file and calling it
  - `Dataset(config, events, labels, split_ts, window_end)` frozen dataclass
  - `Dataset.window_events() -> tuple[GraphEvent, ...]`, `Dataset.anomaly_keys() -> set[tuple[str, int, str]]`
  - `build_dataset(config: DatasetConfig) -> Dataset`
  - `save_dataset(path: Path, dataset: Dataset) -> None`, `load_dataset(path: Path) -> Dataset`

- [ ] **Step 1: Write the failing assembly test**

```python
# tests/generator/test_dataset.py
"""Dataset assembly: split, injection, consistency."""

from pathlib import Path

from rga.domain.events import EventOp
from rga.domain.replay import journal_issues, replay
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset

CONFIG = load_dataset_config(Path("configs/generator/small.yaml"))


def test_dataset_journal_is_consistent() -> None:
    dataset = build_dataset(CONFIG)
    assert journal_issues(dataset.events) == []


def test_split_leaves_a_populated_training_graph() -> None:
    dataset = build_dataset(CONFIG)
    graph = replay(dataset.events, until=dataset.split_ts)
    assert graph.num_nodes > 50
    assert graph.num_edges > 50


def test_all_labels_fall_inside_the_evaluation_window() -> None:
    dataset = build_dataset(CONFIG)
    assert dataset.labels
    assert all(dataset.split_ts <= label.ts < dataset.window_end for label in dataset.labels)


def test_every_label_corresponds_to_a_real_event() -> None:
    dataset = build_dataset(CONFIG)
    granted = {
        event.edge_key() for event in dataset.events if event.op is EventOp.GRANT
    }
    assert {label.edge_key() for label in dataset.labels} <= granted


def test_anomaly_rate_stays_in_a_sane_band() -> None:
    # The configured rate is a target, not a guarantee: coverage of every pattern
    # comes first, and one burst incident creates a dozen edges. What must hold is
    # that anomalies are present and remain a minority of the window.
    dataset = build_dataset(CONFIG)
    window_grants = [
        event for event in dataset.window_events() if event.op is EventOp.GRANT
    ]
    observed = len(dataset.labels) / len(window_grants)
    assert 0.0 < observed < 0.25


def test_labelled_edges_are_not_also_produced_normally() -> None:
    # Ambiguous ground truth would silently corrupt every metric downstream.
    dataset = build_dataset(CONFIG)
    normal_keys = {
        event.edge_key()
        for event in dataset.events
        if event.op is EventOp.GRANT and event.ts < dataset.split_ts
    }
    assert not (dataset.anomaly_keys() & normal_keys)


def test_nearly_every_configured_pattern_is_represented() -> None:
    # A pattern may exhaust its candidates on a small graph, but most must land,
    # otherwise per-pattern evaluation in Module 2 has nothing to measure.
    dataset = build_dataset(CONFIG)
    present = {label.pattern for label in dataset.labels}
    assert len(present) >= len(CONFIG.anomalies.patterns) - 2


def test_generation_is_reproducible() -> None:
    assert build_dataset(CONFIG).events == build_dataset(CONFIG).events
```

- [ ] **Step 2: Write the failing storage test**

```python
# tests/io/test_dataset_io.py
"""A dataset directory round-trips exactly."""

from pathlib import Path

from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.io.dataset_io import load_dataset, save_dataset


def test_dataset_round_trips(tmp_path: Path) -> None:
    original = build_dataset(load_dataset_config(Path("configs/generator/small.yaml")))
    save_dataset(tmp_path / "small", original)
    restored = load_dataset(tmp_path / "small")

    assert restored.events == original.events
    assert restored.labels == original.labels
    assert restored.split_ts == original.split_ts
    assert restored.window_end == original.window_end
    assert restored.config == original.config


def test_saved_layout_is_the_expected_three_files(tmp_path: Path) -> None:
    dataset = build_dataset(load_dataset_config(Path("configs/generator/small.yaml")))
    save_dataset(tmp_path / "small", dataset)
    assert sorted(path.name for path in (tmp_path / "small").iterdir()) == [
        "events.jsonl",
        "labels.jsonl",
        "meta.json",
    ]
```

- [ ] **Step 3: Run both to confirm they fail**

Run: `uv run pytest tests/generator/test_dataset.py tests/io/test_dataset_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.generator.dataset'`

- [ ] **Step 4: Extract the document parser in `config.py`**

Replace the body of `load_dataset_config` with a thin wrapper and add the parser
so that saved metadata can be reloaded without touching a YAML file:

```python
def dataset_config_from_document(document: Mapping[str, object]) -> DatasetConfig:
    """Build and validate a config from an already-parsed document."""
    timeline = _timeline_config(document["timeline"])  # type: ignore[arg-type]
    eval_window_days = int(document["eval_window_days"])  # type: ignore[arg-type]
    if eval_window_days >= timeline.days:
        raise ValueError(
            f"eval window of {eval_window_days} days does not fit in {timeline.days} days"
        )
    anomalies_document = document["anomalies"]
    return DatasetConfig(
        name=str(document["name"]),
        seed=int(document["seed"]),  # type: ignore[arg-type]
        org=_org_config(document["org"]),  # type: ignore[arg-type]
        timeline=timeline,
        anomalies=AnomalyConfig(
            patterns=tuple(str(name) for name in anomalies_document["patterns"]),  # type: ignore[index]
            rate=_rate(anomalies_document["rate"], "anomalies.rate"),  # type: ignore[index]
            train_contamination=_rate(
                anomalies_document["train_contamination"],  # type: ignore[index]
                "anomalies.train_contamination",
            ),
        ),
        eval_window_days=eval_window_days,
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset recipe from a YAML file."""
    return dataset_config_from_document(yaml.safe_load(path.read_text(encoding="utf-8")))
```

- [ ] **Step 5: Implement `dataset.py`**

```python
# src/rga/generator/dataset.py
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
        1
        for event in normal
        if event.op is EventOp.GRANT and split_ts <= event.ts < window_end
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
```

- [ ] **Step 6: Implement `dataset_io.py`**

```python
# src/rga/io/dataset_io.py
"""Dataset directory layout.

    <name>/events.jsonl   the full journal
    <name>/labels.jsonl   ground truth for the evaluation window
    <name>/meta.json      the recipe, the split, and summary counts

Plain text throughout: a dataset is an experimental artefact that has to be
inspectable years later without this code.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rga.generator.anomalies.base import AnomalyLabel
from rga.generator.config import dataset_config_from_document
from rga.generator.dataset import Dataset
from rga.io.jsonl import read_events, write_events


def save_dataset(path: Path, dataset: Dataset) -> None:
    """Write a dataset to its directory, creating it if needed."""
    path.mkdir(parents=True, exist_ok=True)
    write_events(path / "events.jsonl", dataset.events)

    with (path / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for label in dataset.labels:
            handle.write(json.dumps(label.to_dict(), separators=(",", ":")))
            handle.write("\n")

    meta = {
        "config": asdict(dataset.config),
        "split_ts": dataset.split_ts,
        "window_end": dataset.window_end,
        "counts": {"events": len(dataset.events), "labels": len(dataset.labels)},
    }
    (path / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_dataset(path: Path) -> Dataset:
    """Read a dataset back from its directory."""
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    labels = tuple(
        AnomalyLabel.from_dict(json.loads(line))
        for line in (path / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return Dataset(
        config=dataset_config_from_document(meta["config"]),
        events=tuple(read_events(path / "events.jsonl")),
        labels=labels,
        split_ts=int(meta["split_ts"]),
        window_end=int(meta["window_end"]),
    )
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/generator/test_dataset.py tests/io/test_dataset_io.py -v`
Expected: PASS. If `test_anomaly_rate_is_close_to_the_configured_one` fails low, the
patterns are exhausting on the small config — raise the org size in
`configs/generator/small.yaml`, not the tolerance.

- [ ] **Step 8: Commit and push**

```bash
git add src/rga/generator/dataset.py src/rga/generator/config.py src/rga/io/dataset_io.py tests/generator/test_dataset.py tests/io/test_dataset_io.py
git commit -m "feat: add dataset assembly with temporal split and anomaly injection"
git push origin main
```

---

## Task 16: Command line and dataset statistics

**Files:**
- Create: `src/rga/generator/stats.py`, `src/rga/cli/__init__.py`, `src/rga/cli/main.py`
- Test: `tests/generator/test_stats.py`, `tests/cli/test_main.py`

**Interfaces:**
- Consumes: `Dataset`, `build_dataset`, `save_dataset`, `load_dataset` (Task 15); `replay` (Task 5)
- Produces:
  - `dataset_stats(dataset: Dataset) -> dict[str, object]`
  - `format_stats(stats: dict[str, object]) -> str`
  - `main(argv: list[str] | None = None) -> int` with subcommands `generate`, `stats`

- [ ] **Step 1: Write the failing tests**

```python
# tests/generator/test_stats.py
"""Dataset statistics — the acceptance check for a generated dataset."""

from pathlib import Path

from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.generator.stats import dataset_stats, format_stats


def _stats():
    return dataset_stats(build_dataset(load_dataset_config(Path("configs/generator/small.yaml"))))


def test_reports_graph_size_at_the_split() -> None:
    stats = _stats()
    assert stats["train_nodes"] > 50
    assert stats["train_edges"] > 50


def test_reports_counts_per_node_type_and_relation() -> None:
    stats = _stats()
    assert set(stats["nodes_by_type"]) <= {"USER", "GROUP", "BUCKET", "OBJECT"}  # type: ignore[arg-type]
    assert "HAS_PERMISSION" in stats["edges_by_relation"]  # type: ignore[operator]


def test_reports_anomalies_per_pattern() -> None:
    stats = _stats()
    per_pattern = stats["anomalies_by_pattern"]
    assert isinstance(per_pattern, dict)
    assert sum(per_pattern.values()) == stats["anomalies"]


def test_formats_without_raising() -> None:
    assert "train_nodes" in format_stats(_stats())
```

```python
# tests/cli/test_main.py
"""The command line runs end to end."""

from pathlib import Path

from rga.cli.main import main


def test_generate_then_stats(tmp_path: Path, capsys) -> None:
    out = tmp_path / "small"
    assert main(["generate", "--config", "configs/generator/small.yaml", "--out", str(out)]) == 0
    assert (out / "events.jsonl").exists()

    assert main(["stats", "--dataset", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "train_nodes" in printed
    assert "anomalies" in printed


def test_unknown_command_fails_cleanly(capsys) -> None:
    assert main(["nonsense"]) == 2
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest tests/generator/test_stats.py tests/cli -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.generator.stats'`

- [ ] **Step 3: Implement `stats.py`**

```python
# src/rga/generator/stats.py
"""Dataset statistics.

These numbers are the acceptance check for a generated dataset. A graph whose
degree distribution is flat, or whose anomalies all come from one pattern, is not
a usable experiment input however well the code ran.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from rga.domain.entities import EntityType
from rga.domain.events import EventOp
from rga.domain.relations import RelationType
from rga.domain.replay import replay
from rga.generator.dataset import Dataset


def dataset_stats(dataset: Dataset) -> dict[str, object]:
    """Summarise a dataset."""
    graph = replay(dataset.events, until=dataset.split_ts)
    degrees = np.bincount(graph.edge_src, minlength=graph.num_nodes)
    window_grants = [
        event for event in dataset.window_events() if event.op is EventOp.GRANT
    ]

    return {
        "name": dataset.config.name,
        "seed": dataset.config.seed,
        "events": len(dataset.events),
        "train_nodes": graph.num_nodes,
        "train_edges": graph.num_edges,
        "nodes_by_type": {
            entity.name: int((graph.node_type == int(entity)).sum()) for entity in EntityType
        },
        "edges_by_relation": {
            relation.name: int((graph.edge_rel == int(relation)).sum())
            for relation in RelationType
        },
        "out_degree_mean": float(degrees.mean()) if graph.num_nodes else 0.0,
        "out_degree_max": int(degrees.max()) if graph.num_nodes else 0,
        "window_grants": len(window_grants),
        "anomalies": len(dataset.labels),
        "anomaly_rate": (len(dataset.labels) / len(window_grants)) if window_grants else 0.0,
        "anomalies_by_pattern": dict(
            Counter(label.pattern for label in dataset.labels)
        ),
    }


def format_stats(stats: dict[str, object]) -> str:
    """Render statistics as aligned lines."""
    width = max(len(key) for key in stats)
    lines = []
    for key, value in stats.items():
        if isinstance(value, dict):
            lines.append(f"{key:<{width}} :")
            inner = max((len(str(name)) for name in value), default=0)
            lines.extend(f"  {str(name):<{inner}} : {count}" for name, count in value.items())
        elif isinstance(value, float):
            lines.append(f"{key:<{width}} : {value:.4f}")
        else:
            lines.append(f"{key:<{width}} : {value}")
    return "\n".join(lines)
```

- [ ] **Step 4: Implement `main.py`**

```python
# src/rga/cli/main.py
"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.generator.stats import dataset_stats, format_stats
from rga.io.dataset_io import load_dataset, save_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rga", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate a synthetic dataset")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)

    stats = commands.add_parser("stats", help="summarise a dataset")
    stats.add_argument("--dataset", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line. Returns the process exit code."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as exit_signal:
        return int(exit_signal.code or 0)

    if arguments.command == "generate":
        dataset = build_dataset(load_dataset_config(arguments.config))
        save_dataset(arguments.out, dataset)
        print(format_stats(dataset_stats(dataset)))
        return 0

    if arguments.command == "stats":
        print(format_stats(dataset_stats(load_dataset(arguments.dataset))))
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

```python
# src/rga/cli/__init__.py
"""Command line interface."""
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/generator/test_stats.py tests/cli -v`
Expected: PASS

- [ ] **Step 6: Generate the real datasets and look at them**

Run:
```bash
uv run rga generate --config configs/generator/small.yaml --out data/small
uv run rga generate --config configs/generator/default.yaml --out data/default
```

Read the printed statistics before moving on. The dataset is acceptable when the
maximum out-degree is well above the mean — a realistic graph has hubs — when every
configured pattern appears in `anomalies_by_pattern`, and when `anomaly_rate` is a
small minority of the window. If a pattern is missing, it exhausted its candidates:
raise the organization size rather than lowering the rate.

On `default.yaml` the achieved rate should also land near the configured 1 percent.
If it lands far above, the window is too quiet for the catalogue: one incident of
every pattern is roughly thirty edges, so the window needs a few thousand grants
for one percent to be reachable. Raise `grants_per_day` and the number of
departments rather than dropping patterns.

- [ ] **Step 7: Run the whole suite and the linter**

Run:
```bash
uv run ruff check .
uv run pytest -m "not integration and not gpu"
```
Expected: both clean.

- [ ] **Step 8: Commit and push**

```bash
git add src/rga/generator/stats.py src/rga/cli tests/generator/test_stats.py tests/cli
git commit -m "feat: add cli and dataset statistics"
git push origin main
```

---

## Task 17: opens3-rebac — timestamps on nodes and edges

**This task works in a different repository**, `/home/wexel/Data/Code/Projects/opens3-rebac`,
which four other people share. It goes on a branch and through a pull request, never
straight to its `main`.

**Files:**
- Modify: `services/authz/internal/repositories/neo4j/store.py`
- Test: `services/authz/tests/integration/test_neo4j_store.py`

**Interfaces:**
- Consumes: nothing from this repository
- Produces: `created_at` and `updated_at` in milliseconds on every relationship written by `Neo4jStore`, and `created_at` on every node it creates

- [ ] **Step 1: Branch**

```bash
cd /home/wexel/Data/Code/Projects/opens3-rebac
git checkout main
git pull
git checkout -b feat/graph-timestamps
```

- [ ] **Step 2: Write the failing integration test**

Append to `services/authz/tests/integration/test_neo4j_store.py`:

```python
def test_write_records_timestamps(neo4j_store, clean_graph):
    """Every written relationship and node carries creation time in milliseconds."""
    before = int(time.time() * 1000)
    neo4j_store.write_tuple(Tuple("user:alex", "MEMBER_OF", "group:devops", level=None))
    after = int(time.time() * 1000)

    with neo4j_store.driver.session() as session:
        record = session.run(
            """
            MATCH (s {id: 'user:alex'})-[r:MEMBER_OF]->(o {id: 'group:devops'})
            RETURN r.created_at AS created, r.updated_at AS updated,
                   s.created_at AS subject_created
            """
        ).single()

    assert record is not None
    assert before <= record["created"] <= after
    assert record["updated"] >= record["created"]
    assert before <= record["subject_created"] <= after


def test_rewrite_keeps_created_and_advances_updated(neo4j_store, clean_graph):
    """A re-grant must not look like a brand new relationship."""
    tuple_ = Tuple("group:devs", "HAS_PERMISSION", "resource:repo1", level="read")
    neo4j_store.write_tuple(tuple_)

    with neo4j_store.driver.session() as session:
        first = session.run(
            "MATCH ()-[r:HAS_PERMISSION]->() RETURN r.created_at AS created"
        ).single()["created"]

    time.sleep(0.01)
    neo4j_store.write_tuple(Tuple("group:devs", "HAS_PERMISSION", "resource:repo1",
                                  level="admin"))

    with neo4j_store.driver.session() as session:
        record = session.run(
            """
            MATCH ()-[r:HAS_PERMISSION]->()
            RETURN r.created_at AS created, r.updated_at AS updated, r.level AS level
            """
        ).single()

    assert record["created"] == first
    assert record["updated"] > first
    assert record["level"] == "admin"
```

Add `import time` at the top of that test file if it is not already imported.

- [ ] **Step 3: Start Neo4j and run the test to confirm it fails**

Run:
```bash
cd /home/wexel/Data/Code/Projects/opens3-rebac
docker compose up -d neo4j
cd services/authz && python -m pytest tests/integration -m integration -v -k timestamp
```
Expected: FAIL — `assert None is not None`, because `r.created_at` does not exist.

- [ ] **Step 4: Add timestamps to the write queries**

In `services/authz/internal/repositories/neo4j/store.py`, add `import time` at the
top, then replace the query in `_write_has_permission`:

```python
    def _write_has_permission(self, tuple_: Tuple) -> bool:
        """Create (subject)-[:HAS_PERMISSION {level: ...}]->(object)."""
        s_label = infer_node_label(tuple_.subject).value
        o_label = infer_node_label(tuple_.object).value
        query = """
        MERGE (subject:`%s` {id: $subject_id})
          ON CREATE SET subject.created_at = $now
        MERGE (object:`%s` {id: $object_id})
          ON CREATE SET object.created_at = $now
        MERGE (subject)-[r:HAS_PERMISSION]->(object)
          ON CREATE SET r.created_at = $now
        SET r.level = $level, r.updated_at = $now
        RETURN r
        """ % (s_label, o_label)
        with self.driver.session() as session:
            result = session.run(
                query,
                subject_id=tuple_.subject,
                object_id=tuple_.object,
                level=tuple_.level,
                now=int(time.time() * 1000),
            )
            return result.single() is not None
```

and the query in `_write_plain_relation`:

```python
    def _write_plain_relation(self, tuple_: Tuple) -> bool:
        """Create (subject)-[:REL_TYPE]->(object) for MEMBER_OF, OWNER_OF, etc."""
        s_label = infer_node_label(tuple_.subject).value
        o_label = infer_node_label(tuple_.object).value
        query = """
        MERGE (subject:`%s` {id: $subject_id})
          ON CREATE SET subject.created_at = $now
        MERGE (object:`%s` {id: $object_id})
          ON CREATE SET object.created_at = $now
        MERGE (subject)-[rel:`%s`]->(object)
          ON CREATE SET rel.created_at = $now
        SET rel.updated_at = $now
        RETURN rel
        """ % (s_label, o_label, tuple_.relation)
        logger.debug({}, "Neo4j write plain relation", tuple=str(tuple_))
        with self.driver.session() as session:
            result = session.run(
                query,
                subject_id=tuple_.subject,
                object_id=tuple_.object,
                now=int(time.time() * 1000),
            )
            return result.single() is not None
```

- [ ] **Step 5: Run the tests**

Run: `cd /home/wexel/Data/Code/Projects/opens3-rebac/services/authz && python -m pytest tests -v`
Expected: PASS, including the existing tests — the change adds properties and alters
no behaviour.

- [ ] **Step 6: Commit and open the pull request**

```bash
cd /home/wexel/Data/Code/Projects/opens3-rebac
git add services/authz
git commit -m "feat(authz): record created_at and updated_at on graph nodes and edges"
git push -u origin feat/graph-timestamps
gh pr create --title "authz: record timestamps on graph nodes and edges" --body "Adds created_at and updated_at in milliseconds to every relationship and created_at to every node written by Neo4jStore.

Backwards compatible: no query behaviour changes, only additional properties. Edges written before this change simply have no timestamp.

Needed for retrospective audit of the permission graph — at present the graph cannot answer when a right was granted, and the only record of that lives in the Kafka topic for as long as its retention allows."
```

- [ ] **Step 7: Return to the main repository**

```bash
cd /home/wexel/Data/Code/Projects/rebac-graph-anomaly
```

---

## Task 18: opens3-rebac — the initiator of a change

Same repository, same rules. This one touches the shared proto contract, so the Go
stubs regenerate too; adding a field is wire-compatible and the Go services keep
compiling untouched.

Note that `services/gateway` is currently empty, so nothing else in the system needs
to change: whoever writes the gateway fills the new field from the parsed token.

**Files:**
- Modify: `shared/api/authz/v1/authz.proto`, `services/authz/internal/types.py`, `services/authz/entrypoints/server/servicer.py`, `services/authz/internal/repositories/neo4j/store.py`, `services/authz/internal/repositories/kafka/producer.py`
- Test: `services/authz/tests/unit/test_permission_service.py`, `services/authz/tests/integration/test_neo4j_store.py`

**Interfaces:**
- Consumes: Task 17 merged or at least branched from
- Produces: optional `actor` carried from the gRPC request into the Neo4j edge property and the Kafka audit event

- [ ] **Step 1: Branch from the timestamps work**

```bash
cd /home/wexel/Data/Code/Projects/opens3-rebac
git checkout feat/graph-timestamps
git checkout -b feat/graph-provenance
```

- [ ] **Step 2: Write the failing integration test**

Append to `services/authz/tests/integration/test_neo4j_store.py`:

```python
def test_write_records_the_actor(neo4j_store, clean_graph):
    """The initiator of a change is stored on the edge."""
    neo4j_store.write_tuple(
        Tuple("user:alex", "MEMBER_OF", "group:devops", level=None, actor="user:root")
    )

    with neo4j_store.driver.session() as session:
        record = session.run(
            "MATCH ()-[r:MEMBER_OF]->() RETURN r.actor AS actor"
        ).single()

    assert record["actor"] == "user:root"


def test_absent_actor_leaves_no_property(neo4j_store, clean_graph):
    """Unknown is distinct from empty: no actor means no property at all."""
    neo4j_store.write_tuple(Tuple("user:alex", "MEMBER_OF", "group:devops", level=None))

    with neo4j_store.driver.session() as session:
        record = session.run(
            "MATCH ()-[r:MEMBER_OF]->() RETURN r.actor AS actor"
        ).single()

    assert record["actor"] is None
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `cd services/authz && python -m pytest tests/integration -m integration -v -k actor`
Expected: FAIL — `TypeError: Tuple.__init__() got an unexpected keyword argument 'actor'`

- [ ] **Step 4: Add the field to the proto contract**

In `shared/api/authz/v1/authz.proto`, extend the two request messages:

```proto
message WriteTupleRequest {
  string subject = 1;
  Relation relation = 2;
  string object = 3;
  PermissionLevel level = 4;

  // Кто выполняет изменение. Заполняется вызывающим сервисом из разобранного
  // токена доступа. Необязательно: пустая строка означает "неизвестно".
  // Нужно для аудита — по кортежу отношения иначе невозможно отличить право,
  // выданное администратором, от права, которое субъект выдал сам себе.
  string actor = 5;
}
```

```proto
message DeleteTupleRequest {
  string   subject  = 1;
  Relation relation = 2;
  string   object   = 3;

  // См. WriteTupleRequest.actor.
  string actor = 4;
}
```

- [ ] **Step 5: Regenerate the stubs**

```bash
cd /home/wexel/Data/Code/Projects/opens3-rebac
make generate-authz-py
make generate-authz-go
```

- [ ] **Step 6: Carry the field through the service**

In `services/authz/internal/types.py`, add the field to `Tuple`:

```python
@dataclass(frozen=True)
class Tuple:
    """Atomic relationship tuple: (subject, relation, object) with optional level."""
    subject: str
    relation: str
    object: str
    level: Optional[str] = None
    # Who performed the change. None means unknown, which is distinct from nobody.
    actor: Optional[str] = None
```

In `services/authz/entrypoints/server/servicer.py`, pass it in both mutating handlers:

```python
        tuple_ = Tuple(request.subject, relation, request.object, level=level,
                       actor=request.actor or None)
```

```python
        tuple_ = Tuple(request.subject, relation, request.object,
                       actor=request.actor or None)
```

In `store.py`, set it in both write queries by adding to the `SET` clause —
`_write_has_permission` becomes `SET r.level = $level, r.updated_at = $now, r.actor = $actor`
and `_write_plain_relation` becomes `SET rel.updated_at = $now, rel.actor = $actor`,
with `actor=tuple_.actor` added to both `session.run` calls. Cypher removes a
property that is set to null, so an unknown actor leaves no property behind.

In `producer.py`, include it in the audit event:

```python
            "tuple": {
                "subject": tuple_.subject,
                "relation": tuple_.relation,
                "object": tuple_.object,
                "actor": tuple_.actor,
            },
```

- [ ] **Step 7: Run the tests**

Run: `cd services/authz && python -m pytest tests -v`
Expected: PASS, including the pre-existing unit tests.

- [ ] **Step 8: Commit and open the pull request**

```bash
cd /home/wexel/Data/Code/Projects/opens3-rebac
git add shared services/authz
git commit -m "feat(authz): carry the initiator of a tuple change into the graph and audit log"
git push -u origin feat/graph-provenance
gh pr create --base feat/graph-timestamps --title "authz: record who changed a relation tuple" --body "Adds an optional actor field to WriteTupleRequest and DeleteTupleRequest, stored as an edge property and included in the Kafka audit event.

Wire-compatible: the field is optional and a caller that omits it behaves exactly as before. services/gateway is not implemented yet, so nothing currently fills it; whoever writes it takes the value from the parsed access token.

Without this, neither the graph nor the audit log can distinguish a right granted by an administrator from one a subject granted to itself, which is the defining signal of privilege escalation."
```

- [ ] **Step 9: Return to the main repository**

```bash
cd /home/wexel/Data/Code/Projects/rebac-graph-anomaly
```

---

## Task 19: Neo4j source

The adapter that makes a live `opens3-rebac` deployment a data source like any other.
It is written against a narrow reader interface rather than the driver so that its
logic is unit-testable without a database; a marked integration test covers the real
driver.

**Files:**
- Create: `src/rga/adapters/neo4j_source.py`
- Test: `tests/adapters/test_neo4j_source.py`

**Interfaces:**
- Consumes: `Capabilities`, `GraphSource` (Task 6); `RelationMapping` (Task 6); `GraphBuilder`, `AccessGraph`, `UNKNOWN` (Task 5)
- Produces:
  - `GraphRecordReader` Protocol with `nodes() -> Iterator[Mapping[str, object]]` and `relationships() -> Iterator[Mapping[str, object]]`
  - `BoltReader(uri, user, password)` implementing it over the official driver
  - `Neo4jSource(reader, mapping)` implementing `GraphSource`

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/test_neo4j_source.py
"""Reading a live authorization graph, including one written before the patches."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from rga.adapters.base import GraphSource
from rga.adapters.mapping import RelationMapping
from rga.adapters.neo4j_source import Neo4jSource
from rga.domain.graph import UNKNOWN
from rga.domain.relations import PermissionLevel, RelationType

MAPPING = RelationMapping.load(Path("configs/mapping/opens3.yaml"))


class FakeReader:
    def __init__(self, nodes: list[dict], relationships: list[dict]) -> None:
        self._nodes = nodes
        self._relationships = relationships

    def nodes(self) -> Iterator[Mapping[str, object]]:
        return iter(self._nodes)

    def relationships(self) -> Iterator[Mapping[str, object]]:
        return iter(self._relationships)


def _modern() -> FakeReader:
    return FakeReader(
        nodes=[
            {"id": "user:alice", "created_at": 1_000},
            {"id": "group:devops", "created_at": 900},
            {"id": "bucket:photos", "created_at": 800},
        ],
        relationships=[
            {"subject": "user:alice", "relation": "MEMBER_OF", "object": "group:devops",
             "level": None, "created_at": 1_100, "actor": "user:root"},
            {"subject": "group:devops", "relation": "HAS_PERMISSION", "object": "bucket:photos",
             "level": "write", "created_at": 1_200, "actor": "user:root"},
        ],
    )


def _legacy() -> FakeReader:
    """A graph written before the timestamp and actor patches."""
    return FakeReader(
        nodes=[{"id": "user:alice", "created_at": None},
               {"id": "group:devops", "created_at": None}],
        relationships=[
            {"subject": "user:alice", "relation": "MEMBER_OF", "object": "group:devops",
             "level": None, "created_at": None, "actor": None}
        ],
    )


def test_source_satisfies_the_protocol() -> None:
    assert isinstance(Neo4jSource(_modern(), MAPPING), GraphSource)


def test_patched_deployment_reaches_level_two() -> None:
    capabilities = Neo4jSource(_modern(), MAPPING).capabilities()
    assert capabilities.level == 2
    # A snapshot cannot show revocations, so there is no change log either way.
    assert capabilities.change_log is False


def test_unpatched_deployment_is_level_zero() -> None:
    assert Neo4jSource(_legacy(), MAPPING).capabilities().level == 0


def test_snapshot_translates_relations_and_levels() -> None:
    graph = Neo4jSource(_modern(), MAPPING).snapshot()
    assert graph.num_nodes == 3
    assert graph.num_edges == 2

    permission = graph.edge_rel == int(RelationType.HAS_PERMISSION)
    assert graph.edge_level[permission].tolist() == [int(PermissionLevel.WRITE)]
    assert graph.neighbors("user:alice", RelationType.MEMBER_OF).tolist() == [
        graph.index_of("group:devops")
    ]


def test_snapshot_marks_missing_timestamps_as_unknown() -> None:
    graph = Neo4jSource(_legacy(), MAPPING).snapshot()
    assert graph.edge_created.tolist() == [UNKNOWN]
    assert graph.node_created.tolist() == [UNKNOWN, UNKNOWN]
    assert graph.edge_actor.tolist() == [UNKNOWN]


def test_snapshot_cutoff_drops_later_edges() -> None:
    graph = Neo4jSource(_modern(), MAPPING).snapshot(at=1_150)
    assert graph.num_edges == 1


def test_cutoff_on_an_untimestamped_graph_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot honour a cutoff"):
        Neo4jSource(_legacy(), MAPPING).snapshot(at=1_000)


def test_events_are_not_available_from_a_snapshot() -> None:
    with pytest.raises(NotImplementedError, match="change log"):
        list(Neo4jSource(_modern(), MAPPING).events())


def test_unmapped_relation_is_reported_with_its_name() -> None:
    reader = FakeReader(
        nodes=[{"id": "user:alice", "created_at": 1}],
        relationships=[{"subject": "user:alice", "relation": "FRIENDS_WITH",
                        "object": "user:bob", "level": None, "created_at": 2, "actor": None}],
    )
    with pytest.raises(KeyError, match="FRIENDS_WITH"):
        Neo4jSource(reader, MAPPING).snapshot()
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/adapters/test_neo4j_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rga.adapters.neo4j_source'`

- [ ] **Step 3: Confirm the builder entry points exist**

`GraphBuilder.register_node` and `GraphBuilder.add_edge` were added in Task 5
precisely for this case: a snapshot delivers rows rather than events, and some rows
carry no timestamp. No change to `graph.py` is needed here.

Run: `uv run pytest tests/domain -v`
Expected: PASS, confirming the builder is intact before building on it.

- [ ] **Step 5: Implement `neo4j_source.py`**

```python
# src/rga/adapters/neo4j_source.py
"""A live opens3-rebac deployment as a data source.

Written against a narrow reader interface rather than the driver, so the
translation logic is testable without a database. The driver-backed reader is a
dozen lines at the bottom of this file and is exercised by a marked integration
test.

Capability level depends on the data, not on the product: a deployment whose
edges predate the timestamp patch reports level 0 and its temporal features are
masked, exactly as any third-party engine without timestamps would be.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol

from rga.adapters.base import Capabilities
from rga.adapters.mapping import RelationMapping
from rga.domain.events import GraphEvent
from rga.domain.graph import UNKNOWN, AccessGraph, GraphBuilder

#: Rows inspected when deciding what the deployment can provide.
_PROBE_SIZE = 512


class GraphRecordReader(Protocol):
    """Rows of the authorization graph, in whatever shape the store returns them."""

    def nodes(self) -> Iterator[Mapping[str, object]]:
        """Yield mappings with `id` and `created_at`."""
        ...

    def relationships(self) -> Iterator[Mapping[str, object]]:
        """Yield mappings with `subject`, `relation`, `object`, `level`, `created_at`, `actor`."""
        ...


class Neo4jSource:
    """Reads the access graph from a Neo4j-backed authorization engine."""

    def __init__(self, reader: GraphRecordReader, mapping: RelationMapping) -> None:
        self._reader = reader
        self._mapping = mapping

    def capabilities(self) -> Capabilities:
        """Probe the data to decide the level.

        `change_log` is always False: a snapshot cannot show a revocation, since
        a revoked edge is simply absent. Observing revocations needs the Kafka
        adapter, which arrives with the streaming mode in a later module.
        """
        timestamps = False
        provenance = False
        for index, row in enumerate(self._reader.relationships()):
            if index >= _PROBE_SIZE:
                break
            timestamps = timestamps or row.get("created_at") is not None
            provenance = provenance or row.get("actor") is not None
        return Capabilities(timestamps=timestamps, provenance=provenance, change_log=False)

    def snapshot(self, at: int | None = None) -> AccessGraph:
        """Build the graph, optionally as it stood at `at`."""
        if at is not None and not self.capabilities().timestamps:
            raise ValueError("cannot honour a cutoff: this deployment records no timestamps")

        builder = GraphBuilder()
        for row in self._reader.nodes():
            builder.register_node(str(row["id"]), _as_int(row.get("created_at")))

        for row in self._reader.relationships():
            created = _as_int(row.get("created_at"))
            if at is not None and created > at:
                continue
            relation, level = self._mapping.translate(
                str(row["relation"]),
                None if row.get("level") is None else str(row["level"]),
            )
            actor = row.get("actor")
            builder.add_edge(
                str(row["subject"]),
                relation,
                str(row["object"]),
                level=int(level),
                created=created,
                actor=None if actor is None else str(actor),
            )

        return builder.build()

    def events(self, since: int = 0, until: int | None = None) -> Iterator[GraphEvent]:
        """Not available: a snapshot carries no change log."""
        raise NotImplementedError(
            "a Neo4j snapshot has no change log; revocations are invisible in it"
        )


def _as_int(value: object) -> int:
    """Coerce an optional timestamp, mapping absence to the unknown sentinel."""
    return UNKNOWN if value is None else int(value)  # type: ignore[arg-type]


class BoltReader:
    """Reads graph rows over the official Neo4j driver."""

    _NODES = "MATCH (n) RETURN n.id AS id, n.created_at AS created_at"
    _RELATIONSHIPS = """
    MATCH (s)-[r]->(o)
    RETURN s.id AS subject, type(r) AS relation, o.id AS object,
           r.level AS level, r.created_at AS created_at, r.actor AS actor
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def nodes(self) -> Iterator[Mapping[str, object]]:
        with self._driver.session() as session:
            for record in session.run(self._NODES):
                yield dict(record)

    def relationships(self) -> Iterator[Mapping[str, object]]:
        with self._driver.session() as session:
            for record in session.run(self._RELATIONSHIPS):
                yield dict(record)

    def close(self) -> None:
        self._driver.close()
```

- [ ] **Step 6: Add the integration test**

Append to `tests/adapters/test_neo4j_source.py`:

```python
@pytest.mark.integration
def test_reads_a_live_deployment() -> None:
    """Requires a running opens3-rebac Neo4j: docker compose up -d neo4j."""
    import os

    from rga.adapters.neo4j_source import BoltReader

    reader = BoltReader(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password123"),
    )
    try:
        graph = Neo4jSource(reader, MAPPING).snapshot()
    finally:
        reader.close()

    assert graph.num_nodes >= 0
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/adapters -v -m "not integration"`
Expected: PASS

With Neo4j running and the patches from Tasks 17–18 applied:
```bash
uv sync --extra cpu --extra neo4j
uv run pytest tests/adapters -v -m integration
```

- [ ] **Step 8: Run the whole suite and the linter**

Run:
```bash
uv run ruff check .
uv run pytest -m "not integration and not gpu"
```
Expected: both clean.

- [ ] **Step 9: Commit and push**

```bash
git add src/rga/adapters/neo4j_source.py tests/adapters/test_neo4j_source.py
git commit -m "feat: add neo4j graph source with capability probing"
git push origin main
```

---

## Module Completion

Module 1 is done when all of the following hold:

- `uv run pytest -m "not integration and not gpu"` is green and `uv run ruff check .` is clean.
- `uv run python scripts/gpu_smoke.py` on the Windows PC reports `supported: True` and completes its matmul.
- `uv run rga generate --config configs/generator/default.yaml --out data/default` produces a dataset whose statistics show every configured pattern present, an anomaly rate close to the configured one, and a maximum out-degree well above the mean.
- Both pull requests are open against `opens3-rebac`.
- A live Neo4j snapshot reads through `Neo4jSource` and reports the expected capability level.

Module 2 begins from `Dataset` and the `AccessGraph` it replays: feature extraction,
baselines, and the metrics harness.
