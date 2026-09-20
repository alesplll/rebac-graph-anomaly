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

## Run the service on synthetic data

Train the model once — `artifacts/` is not in the repository because it is
reproducible from its recipe — then start the service.

```bash
uv run rga train --config configs/train/gnn-supervised.yaml --out artifacts/gnn-supervised
uv run rga serve --config configs/service/synthetic.yaml
```

Open http://127.0.0.1:8000. Training takes a few minutes; startup takes about twenty
seconds, because the service generates the dataset, extracts candidates and scores
them before it answers. `--port` moves it off 8000.

## Run the service against a live opens3-rebac

Same code, same artefact; only the configuration differs. Neo4j must be the one the
authorization engine writes to, and its graph has to carry edge timestamps — that is
capability level 1, without which there is no change log to score.

```bash
# 1. the engine's database; main records timestamps and actors since PR #70
cd ../opens3-rebac && git checkout main
docker compose up -d --wait neo4j

# 2. a populated graph: on an empty one every change looks unusual
cd ../rebac-graph-anomaly
uv run python scripts/fill_live_graph.py --config configs/generator/small-history.yaml

# 3. the service, pointed at Neo4j instead of the generator
uv run rga serve --config configs/service/opens3.yaml
```

The status line should read source `neo4j` and capability level 2. Grant a permission
through the engine, press Refresh, and the change appears in the queue.

Connection details live in `configs/service/opens3.yaml`: `uri`, `user`, `password`,
and the relation mapping. Edit that file for a different deployment.

## Other commands

```bash
uv run rga generate --config configs/generator/small.yaml --out data/small
uv run rga evaluate --config configs/experiments/gnn.yaml --out experiments/runs/gnn
uv run python scripts/profile_epoch.py
uv run pytest -m "not integration and not gpu"
uv run ruff check .
```

`configs/experiments/` also holds `module5.yaml`, which compares the self-supervised
variants, and `generalisation.yaml`, which evaluates against hidden patterns shaped
unlike the training ones.

## Documents

`docs/thesis/` holds the result tables, each reproducible by one command.
`docs/module-3-findings.md` and `docs/module-5-findings.md` record what the neural
modules measured, negative results included, each traced to a cause. `docs/demo.md` is the demonstration running order,
`docs/run-on-gpu.md` what to run on the GPU machine. Designs and per-module plans are
under `docs/superpowers/`.
