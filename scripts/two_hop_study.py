"""Did the two-hop sampling strategy spend its negatives on ordinary changes?

Module 3 drew its hardest negatives from the two-hop neighbourhood and measured no
benefit. The explanation it offered — that a node two steps away is the shape of a
legitimate future grant — is checked here without training anything: for every
candidate in the evaluation window, how far apart did its two ends stand in the
graph as it was before the window opened.

Writes `docs/thesis/two-hop.md`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.candidates import candidate_arrays
from rga.nn.two_hop import BEYOND, hop_distribution

_DATASET = Path("configs/generator/small-history.yaml")
_OUT = Path("docs/thesis/two-hop.md")
_SEEDS = (1, 2, 3, 4, 5)
_LABELS = {
    0: "0 — тот же узел",
    1: "1 — сосед",
    2: "2 — два шага",
    BEYOND: "3 и дальше, либо конец неизвестен",
}


def _shares(distance: np.ndarray) -> np.ndarray:
    """Share of the candidates at each of the four distances."""
    counts = np.bincount(distance, minlength=BEYOND + 1).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def main() -> None:
    base = load_dataset_config(_DATASET)
    ordinary: list[np.ndarray] = []
    anomalous: list[np.ndarray] = []

    for seed in _SEEDS:
        dataset = build_dataset(replace(base, seed=seed))
        evaluation = build_candidates(dataset, Span.EVAL)
        graph = evaluation.graph
        assert graph is not None

        arrays, _, _ = candidate_arrays(evaluation)
        distance = hop_distribution(graph, arrays.src, arrays.dst)
        flagged = evaluation.y_true().astype(bool)

        ordinary.append(_shares(distance[~flagged]))
        anomalous.append(_shares(distance[flagged]))
        print(f"seed {seed}: {int(flagged.sum())} anomalous of {flagged.size} candidates")

    normal_mean, normal_sd = np.mean(ordinary, axis=0), np.std(ordinary, axis=0)
    odd_mean, odd_sd = np.mean(anomalous, axis=0), np.std(anomalous, axis=0)

    lines = [
        "# На каком расстоянии стояли концы изменения до его появления",
        "",
        "Раздел 4.3 `docs/module-3-findings.md` объяснил, почему двухшаговые",
        "отрицательные примеры не дали выигрыша: узел в двух шагах — это форма",
        "законной будущей выдачи, и обучение отвергать такие пары учит модель",
        "занижать правдоподобие нормальным изменениям. Объяснение проверяется без",
        "обучения: достаточно посчитать расстояние между концами кандидата в графе,",
        "каким он был до открытия окна оценки.",
        "",
        f"Набор данных `{_DATASET.name}`, сиды {', '.join(str(seed) for seed in _SEEDS)},",
        "среднее и стандартное отклонение по сидам.",
        "",
        "| Расстояние до появления ребра | Доля нормальных | Доля аномальных |",
        "|---|---|---|",
    ]
    for value in (0, 1, 2, BEYOND):
        lines.append(
            f"| {_LABELS[value]} | {normal_mean[value]:.3f} ± {normal_sd[value]:.3f} "
            f"| {odd_mean[value]:.3f} ± {odd_sd[value]:.3f} |"
        )
    lines += [
        "",
        "Числа, на которые опирается разбор в `docs/module-5-findings.md`:",
        "",
        f"- на расстоянии один: нормальных {normal_mean[1]:.3f}, "
        f"аномальных {odd_mean[1]:.3f};",
        f"- на расстоянии два: нормальных {normal_mean[2]:.3f}, "
        f"аномальных {odd_mean[2]:.3f};",
        f"- на расстоянии три и дальше: нормальных {normal_mean[BEYOND]:.3f}, "
        f"аномальных {odd_mean[BEYOND]:.3f}.",
        "",
    ]

    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"written {_OUT}")
    print(
        f"normal at two hops: {normal_mean[2]:.3f} +- {normal_sd[2]:.3f}, "
        f"anomalous: {odd_mean[2]:.3f} +- {odd_sd[2]:.3f}"
    )


if __name__ == "__main__":
    main()
