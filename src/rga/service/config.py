"""What the service is pointed at.

The source is the only place in the running system that knows whether the graph came
from the generator or from a live authorization engine. Switching between them is a
configuration change and nothing else, which is what section 12 of the design document
demands and what the closing minute of the demonstration shows.
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
