"""Regenerate the descriptive tables that go into the thesis.

Everything in docs/thesis is produced from code. Nothing there is written by hand,
so a number in the report can always be traced back to a command.

    uv run python scripts/thesis_tables.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rga.features.build import CANDIDATE_BLOCK, Span, build_candidates
from rga.features.edges import EDGE_BLOCK
from rga.features.nodes import NODE_BLOCK
from rga.features.spec import FeatureGroup
from rga.generator.config import load_dataset_config
from rga.generator.dataset import build_dataset
from rga.generator.stats import dataset_stats

_OUT = Path("docs/thesis")

_GROUP_TITLES = {
    FeatureGroup.STRUCTURAL: "структурные",
    FeatureGroup.TEMPORAL: "временные",
    FeatureGroup.PROVENANCE: "происхождения",
}

_PATTERN_TITLES = {
    "self_grant_admin": "Самовыдача уровня admin",
    "privileged_group_join": "Вход в привилегированную группу",
    "grant_burst": "Всплеск выдач от одного субъекта",
    "dormant_awakening": "Оживление спящей учётной записи",
    "hierarchy_bypass": "Право на объект в обход бакета",
    "cross_department": "Доступ к ресурсам чужого отдела",
    "shadow_group": "Теневая группа с одним участником",
    "delegation_cascade": "Каскад делегирований",
}


def _dataset_table() -> str:
    lines = [
        "# Характеристики наборов данных",
        "",
        "Воспроизводится: `uv run python scripts/thesis_tables.py`",
        "",
    ]
    for name in ("small", "default"):
        config = load_dataset_config(Path(f"configs/generator/{name}.yaml"))
        dataset = build_dataset(config)
        stats = dataset_stats(dataset)
        evaluation = build_candidates(dataset, Span.EVAL)
        train = build_candidates(dataset, Span.TRAIN)

        lines += [
            f"## Конфигурация `{name}`",
            "",
            "| Величина | Значение |",
            "|---|---|",
            f"| Горизонт моделирования, суток | {config.timeline.days} |",
            f"| Окно оценки, суток | {config.eval_window_days} |",
            f"| Событий в журнале | {stats['events']} |",
            f"| Узлов на срезе обучения | {stats['train_nodes']} |",
            f"| Рёбер на срезе обучения | {stats['train_edges']} |",
            f"| Средняя исходящая степень | {stats['out_degree_mean']:.2f} |",
            f"| Максимальная исходящая степень | {stats['out_degree_max']} |",
            f"| Кандидатов на обучающем срезе | {train.n_candidates} |",
            f"| Кандидатов в окне оценки | {evaluation.n_candidates} |",
            f"| Аномальных рёбер | {int(evaluation.labels.sum())} |",
            f"| Доля аномалий | {evaluation.labels.mean():.2%} |",
            "",
            "Узлы по типам:",
            "",
            "| Тип | Количество |",
            "|---|---|",
        ]
        lines += [f"| {kind} | {count} |" for kind, count in stats["nodes_by_type"].items()]
        lines += ["", "Рёбра по типам отношений:", "", "| Отношение | Количество |", "|---|---|"]
        lines += [
            f"| {relation} | {count} |" for relation, count in stats["edges_by_relation"].items()
        ]
        lines += ["", "Аномалии по паттернам:", "", "| Паттерн | Рёбер |", "|---|---|"]
        counts = Counter(evaluation.patterns)
        for pattern in sorted(_PATTERN_TITLES):
            lines.append(f"| {_PATTERN_TITLES[pattern]} | {counts.get(pattern, 0)} |")
        lines.append("")

    return "\n".join(lines)


def _feature_table() -> str:
    blocks = [
        ("Признаки изменения", EDGE_BLOCK, ""),
        ("Признаки субъекта", NODE_BLOCK, "subj_"),
        ("Признаки объекта", NODE_BLOCK, "obj_"),
    ]
    per_group = Counter(CANDIDATE_BLOCK.groups)

    lines = [
        "# Признаковое пространство",
        "",
        "Воспроизводится: `uv run python scripts/thesis_tables.py`",
        "",
        f"Всего признаков: **{len(CANDIDATE_BLOCK)}**. Каждый принадлежит одной группе, и "
        "группа определяет, какой уровень возможностей источника нужен, чтобы признак "
        "вообще можно было вычислить.",
        "",
        "| Группа | Признаков | Что требуется от системы авторизации |",
        "|---|---|---|",
        f"| Структурные | {per_group[FeatureGroup.STRUCTURAL]} | снимок кортежей отношений |",
        f"| Временные | {per_group[FeatureGroup.TEMPORAL]} | метки времени создания |",
        f"| Происхождения | {per_group[FeatureGroup.PROVENANCE]} | инициатор изменения |",
        "",
        "Признак, который источник не может дать, помечается ненаблюдаемым, а не "
        "зануляется: ноль — законное значение почти для всех этих величин, и модель, "
        "получившая нули, выучила бы «инициатора не было» вместо «мы не знаем, кто это был».",
        "",
    ]

    for title, block, prefix in blocks:
        lines += [f"## {title}", "", "| Признак | Группа |", "|---|---|"]
        for name, group in zip(block.names, block.groups, strict=True):
            lines.append(f"| `{prefix}{name}` | {_GROUP_TITLES[group]} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "dataset.md").write_text(_dataset_table() + "\n", encoding="utf-8")
    (_OUT / "features.md").write_text(_feature_table() + "\n", encoding="utf-8")
    print(f"wrote {_OUT / 'dataset.md'} and {_OUT / 'features.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
