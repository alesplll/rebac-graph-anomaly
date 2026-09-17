# rebac-graph-anomaly

Finds structurally unusual permission changes in a ReBAC access graph.

The system reads an access graph and its change log, scores every change, and hands a
security analyst a ranked queue of suspicious grants with grounds for each: which
features moved the score and which relationships it leaned on. It does not pronounce a
verdict — it reduces how much has to be reviewed by hand.

## What it works with

A source declares a capability level, and features it cannot supply are masked rather
than faked:

| Level | The source offers | What becomes available |
|---|---|---|
| 0 | a snapshot of subject–relation–object tuples | structural features |
| 1 | plus edge creation times | plus temporal features and a change log |
| 2 | plus the initiator of each change | plus provenance features |

Sources included: a synthetic organization generator, a journal dump, and a live
`opens3-rebac` deployment over Neo4j. Adding another authorization system means one
class implementing `GraphSource` and one relation mapping in `configs/mapping/`.

Identifiers follow the engine: `user:<uuid>`, `group:<name>`, `bucket:<name>`,
`object:<bucket>/<key>`. Relations are `MEMBER_OF`, `HAS_PERMISSION` (carries an
ordinal level `read < write < create < delete < admin`), `PARENT_OF`, `OWNER_OF`.

Graph layers are written directly on PyTorch. PyTorch Geometric and DGL are not used:
the graph is heterogeneous and multi-relational, and `HAS_PERMISSION` carries an
ordinal level that no stock layer models.

## Install

Python 3.12 through `uv`. The `cpu` and `gpu` extras conflict; install one.

```bash
uv sync --extra cpu --extra service --extra ml --extra neo4j
```

## Run

```bash
uv run rga generate --config configs/generator/small.yaml --out data/small
uv run rga evaluate --config configs/experiments/gnn.yaml --out experiments/runs/gnn
uv run rga train --config configs/train/gnn-supervised.yaml --out artifacts/gnn-supervised
uv run rga serve --config configs/service/synthetic.yaml       # http://127.0.0.1:8000
```

Pointing the service at a live engine is a configuration swap to
`configs/service/opens3.yaml`; no code changes.

```bash
uv run pytest -m "not integration and not gpu"
uv run ruff check .
```

## Documents

`docs/thesis/` holds the result tables, each reproducible by one command.
`docs/module-3-findings.md` records what the neural module measured, including three
negative results and their causes. `docs/demo.md` is the demonstration running order,
`docs/run-on-gpu.md` what to run on the GPU machine. Designs and per-module plans are
under `docs/superpowers/`.
