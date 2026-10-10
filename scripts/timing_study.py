"""Measure the time and reproducibility requirements of the service.

The requirements fix three numbers: the queue for a window of up to a thousand
changes within a minute, the justification of one change within ten seconds, and
the same first fifty positions on a repeated run. This script measures all three
on both demonstration configurations and writes `docs/thesis/timing.md`.

Run: uv run python scripts/timing_study.py
"""

from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path

import torch

from rga.artifacts import load_scorer
from rga.explain.incident import build_incident
from rga.service.analysis import analyse
from rga.service.config import load_service_config

CONFIGS = [Path("configs/service/synthetic.yaml"), Path("configs/service/full.yaml")]
OUT = Path("docs/thesis/timing.md")
REPEATS = 3
CARDS = 10
TOP = 50


def _measure(path: Path) -> dict[str, object]:
    config = load_service_config(path)
    scorer = load_scorer(config.model)

    queue_times = []
    analyses = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        analyses.append(analyse(config, scorer))
        queue_times.append(time.perf_counter() - started)

    analysis = analyses[0]
    card_times = []
    for rank, position in enumerate(analysis.order[:CARDS].tolist(), start=1):
        started = time.perf_counter()
        build_incident(
            scorer,
            analysis.candidates,
            int(position),
            score=float(analysis.scores[position]),
            rank=rank,
        )
        card_times.append(time.perf_counter() - started)

    first = [analysis.candidates.keys[i] for i in analysis.order[:TOP].tolist()]
    same_set = all(
        set(first) == {other.candidates.keys[i] for i in other.order[:TOP].tolist()}
        for other in analyses[1:]
    )
    same_order = all(
        first == [other.candidates.keys[i] for i in other.order[:TOP].tolist()]
        for other in analyses[1:]
    )
    return {
        "config": path.name,
        "changes": analysis.candidates.n_candidates,
        "queue_median": statistics.median(queue_times),
        "queue_max": max(queue_times),
        "card_median": statistics.median(card_times),
        "card_max": max(card_times),
        "same_set": same_set,
        "same_order": same_order,
    }


def main() -> None:
    rows = [_measure(path) for path in CONFIGS]
    lines = [
        "### timing — requirements of section 1.3.3",
        "",
        f"Machine: {platform.processor() or platform.machine()}, "
        f"{torch.get_num_threads()} threads, CPU only, torch {torch.__version__}. "
        f"Queue: {REPEATS} runs; justification: the first {CARDS} cards of the queue.",
        "",
        "| config | changes in window | queue, median s | queue, max s "
        "| justification, median s | justification, max s "
        f"| top {TOP} same set | top {TOP} same order |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['changes']} | {row['queue_median']:.2f} "
            f"| {row['queue_max']:.2f} | {row['card_median']:.2f} | {row['card_max']:.2f} "
            f"| {'yes' if row['same_set'] else 'no'} | {'yes' if row['same_order'] else 'no'} |"
        )
    lines += [
        "",
        "The queue time includes building the window from the source: for the",
        "synthetic configurations that is generating the dataset, for a live engine it",
        "is reading the graph over Bolt. The model is trained beforehand and loaded once.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
