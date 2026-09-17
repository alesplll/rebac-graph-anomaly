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
