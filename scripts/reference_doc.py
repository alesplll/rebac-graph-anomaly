"""Write the analyst's reference out as text for the report.

Run: uv run python scripts/reference_doc.py
Writes: docs/thesis/reference.md

The same texts the interface renders, so the chapter and the running system cannot
say different things.
"""

from __future__ import annotations

from pathlib import Path

from rga.explain.reference import reference

OUTPUT = Path("docs/thesis/reference.md")


def main() -> None:
    body = reference()
    lines = [
        "# Справочник аналитика",
        "",
        "Сгенерировано из `rga.explain.reference`, той же структуры, которую отдаёт",
        "сервис на странице `/reference`. Править здесь руками нечего: текст меняется",
        "в коде и появляется одновременно в интерфейсе и в работе.",
        "",
        "## Наблюдения",
        "",
        "Карточка изменения перечисляет наблюдения — то, что система заметила.",
        "Наблюдение не означает нарушения: это отличие от того, как процесс идёт",
        "обычно. Ниже сказано, как он идёт обычно и почему отличие стоит внимания.",
        "",
    ]

    for entry in body["observations"]:  # type: ignore[union-attr]
        lines += [
            f"### {entry['title']}",
            "",
            entry["why"],
            "",
            f"**На что смотреть.** {entry['look_at']}",
            "",
        ]

    lines += ["## Таблицы карточки", ""]
    for item in body["tables"].values():  # type: ignore[union-attr]
        lines += [
            f"### {item['title']}",
            "",
            item["what"],
            "",
            f"**Как читать.** {item['how']}",
            "",
        ]

    lines += [
        "## Группы признаков",
        "",
        "Группа говорит, что система авторизации должна уметь сообщить, чтобы признак",
        "вообще существовал. Признак, который источник заполнить не может, помечается",
        "недоступным, а не обнуляется.",
        "",
        "| Группа | Что требует от движка |",
        "|---|---|",
    ]
    for name, item in body["groups"].items():  # type: ignore[union-attr]
        lines.append(f"| {item['title']} (`{name}`) | {item['meaning']} |")

    features = body["features"]  # type: ignore[assignment]
    lines += ["", f"## Признаки — всего {len(features)}", ""]
    for group in ("structural", "temporal", "provenance"):
        chosen = [item for item in features if item["group"] == group]
        if not chosen:
            continue
        title = body["groups"][group]["title"]  # type: ignore[index]
        lines += [f"### {title}", "", "| Признак | Что это |", "|---|---|"]
        lines += [f"| `{item['name']}` | {item['title']} |" for item in chosen]
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}: {len(features)} features, {len(body['observations'])} observations")


if __name__ == "__main__":
    main()
