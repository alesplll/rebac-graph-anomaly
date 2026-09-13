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
