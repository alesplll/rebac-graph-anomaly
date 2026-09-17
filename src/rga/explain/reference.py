"""The help page's content, assembled from the same texts the cards use.

Kept in one place on purpose. If the reference were written separately it would drift
away from what the interface actually shows within a week, and a reference that lies
is worse than none.
"""

from __future__ import annotations

from rga.explain.glossary import GROUP_TITLES, feature_titles
from rga.explain.text import CATALOGUE
from rga.features.build import CANDIDATE_BLOCK

#: How to read the two tables of a card.
_TABLES = {
    "edges": {
        "title": "Связи, на которые опиралась оценка",
        "what": (
            "Список связей графа вокруг изменения, каждая со своей значимостью. "
            "Значимость получена прямым измерением: связь убирается из графа, "
            "изменение оценивается заново, и разница показывает, сколько эта связь "
            "давала."
        ),
        "how": (
            "Положительное число означает, что связь держала оценку вверх: без неё "
            "изменение выглядело бы обычнее. Отрицательное — наоборот, связь делала "
            "изменение более обычным, и без неё оценка выросла бы. Ноль означает, что "
            "связь на оценку не повлияла. Смотреть стоит на первые две-три строки: "
            "именно они отвечают на вопрос «почему система сочла это нетипичным»."
        ),
    },
    "features": {
        "title": "Вклад признаков",
        "what": (
            "Что система знала об изменении и насколько каждая известная величина "
            "сдвинула оценку. Вклад считается как производная оценки по признаку, "
            "умноженная на само значение признака."
        ),
        "how": (
            "Знак показывает направление: положительный вклад поднимал оценку, "
            "отрицательный опускал. Величина сравнима только внутри одной карточки — "
            "это не проценты и не вероятности. Строка курсивом с пометкой «не "
            "сообщается» означает, что движок авторизации такие данные не хранит: это "
            "не ноль, а отсутствие сведений, и модель обучена такие признаки "
            "игнорировать, а не принимать нули за осмысленные значения."
        ),
    },
}


def reference() -> dict[str, object]:
    """Everything the reference page renders."""
    titles = feature_titles()
    groups = {str(group): {"title": title, "meaning": meaning}
              for group, (title, meaning) in GROUP_TITLES.items()}

    return {
        "observations": [
            {
                "kind": entry.kind,
                "title": entry.title,
                "why": entry.why,
                "look_at": entry.look_at,
            }
            for entry in CATALOGUE
        ],
        "groups": groups,
        "features": [
            {
                "name": name,
                "title": titles[name],
                "group": str(CANDIDATE_BLOCK.groups[position]),
            }
            for position, name in enumerate(CANDIDATE_BLOCK.names)
        ],
        "tables": _TABLES,
    }
