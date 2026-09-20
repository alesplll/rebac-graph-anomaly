"""Each term of the self-supervised score, measured on its own.

The score of section 3 of the spec is an equal-weight average of ranked terms, and
module 3 found two of them pointing the wrong way. Module 5 changed what two of
them are made of, so the question has to be asked again, per term and per variant.

This measurement uses the anomaly labels. That makes it a diagnostic and nothing
else: choosing weights from this table would be choosing them by looking at the
answer, and the detector would no longer be one.

Writes `docs/thesis/score-terms.md`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from rga.eval.metrics import evaluate_ranking
from rga.features.build import Span, build_candidates
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.nn.config import ModelConfig
from rga.nn.scorer import GnnScorer

_DATASET = Path("configs/generator/small-history.yaml")
_OUT = Path("docs/thesis/score-terms.md")
_SEEDS = (1, 2, 3, 4, 5)
_VARIANTS = {
    "gnn": ModelConfig(),
    "gnn_neighbourhood": replace(ModelConfig(), reconstruction_target="neighbourhood"),
    "gnn_correspondence": replace(ModelConfig(), correspondence_weight=1.0),
}
_RUSSIAN = {
    "likelihood": "правдоподобие ребра",
    "subject_deviation": "отклонение профиля субъекта",
    "object_deviation": "отклонение профиля объекта",
    "correspondence": "соответствие строки и структуры",
    "combined": "итоговая оценка",
}


def _measure(config: ModelConfig) -> dict[str, tuple[float, float, float, float]]:
    """Mean and deviation of PR-AUC and ROC-AUC of every term, over the seeds."""
    gathered: dict[str, dict[str, list[float]]] = {}
    base = load_dataset_config(_DATASET)

    for seed in _SEEDS:
        dataset = build_dataset(replace(base, seed=seed))
        train = build_candidates(dataset, Span.TRAIN)
        evaluation = build_candidates(dataset, Span.EVAL)

        scorer = GnnScorer(seed=seed, config=config)
        scorer.fit(train)

        parts = dict(scorer.terms(evaluation))
        parts["combined"] = np.mean(list(parts.values()), axis=0)
        truth = evaluation.y_true()
        for name, values in parts.items():
            row = evaluate_ranking(truth, values, ks=(20,)).as_row()
            bucket = gathered.setdefault(name, {"pr_auc": [], "roc_auc": []})
            bucket["pr_auc"].append(float(row["pr_auc"]))
            bucket["roc_auc"].append(float(row["roc_auc"]))

    return {
        name: (
            float(np.mean(values["pr_auc"])),
            float(np.std(values["pr_auc"])),
            float(np.mean(values["roc_auc"])),
            float(np.std(values["roc_auc"])),
        )
        for name, values in gathered.items()
    }


def main() -> None:
    lines = [
        "# Слагаемые оценки по отдельности",
        "",
        "Оценка самообучаемой модели — равновесное среднее ранжированных слагаемых.",
        "Здесь каждое измерено само по себе, на пяти сидах набора",
        f"`{_DATASET.name}`. ROC-AUC ниже 0.5 означает, что слагаемое упорядочивает",
        "кандидатов в обратную сторону: чем оно больше, тем изменение обычнее.",
        "",
        "Таблица опирается на метки аномалий и потому является диагностикой.",
        "Выбирать по ней веса нельзя: это был бы выбор с подглядыванием в ответ.",
        "",
    ]

    for label, config in _VARIANTS.items():
        measured = _measure(config)
        lines += [
            f"## {label}",
            "",
            "| Слагаемое | PR-AUC | ROC-AUC |",
            "|---|---|---|",
        ]
        for name, (pr_mean, pr_sd, roc_mean, roc_sd) in measured.items():
            lines.append(
                f"| {_RUSSIAN.get(name, name)} | {pr_mean:.3f} ± {pr_sd:.3f} "
                f"| {roc_mean:.3f} ± {roc_sd:.3f} |"
            )
        lines.append("")
        print(f"{label}: " + ", ".join(f"{n} {v[0]:.3f}" for n, v in measured.items()))

    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"written {_OUT}")


if __name__ == "__main__":
    main()
