"""Feature names in words.

The model works with short English identifiers; an analyst should not have to. Every
feature the interface can show has a phrase here, and a test walks the whole feature
space to make sure none is missing — a table row reading `adamic_adar` with no
explanation is worse than no row at all.

Endpoint features are stored twice, once per end of the change, so their phrases are
generated from one description of the quantity plus which end it belongs to.
"""

from __future__ import annotations

from rga.features.nodes import NODE_BLOCK
from rga.features.spec import FeatureGroup

#: What each group of features needs from the authorization engine, and what it costs
#: the analyst when the engine cannot supply it.
GROUP_TITLES: dict[FeatureGroup, tuple[str, str]] = {
    FeatureGroup.STRUCTURAL: (
        "структура",
        "Форма графа прав: кто с кем связан, какими отношениями и на каком уровне. "
        "Эти признаки доступны от любой системы авторизации, потому что без них она "
        "не смогла бы проверять доступ.",
    ),
    FeatureGroup.TEMPORAL: (
        "время",
        "Когда произошло изменение и как часто субъект действовал до него. Требует, "
        "чтобы движок хранил время создания связей; многие хранят.",
    ),
    FeatureGroup.PROVENANCE: (
        "происхождение",
        "Кто выполнил изменение. Требует, чтобы движок записывал инициатора, а это "
        "делают редко: кортеж отношения описывает состояние прав, но не историю его "
        "возникновения.",
    ),
}

_EDGE_TITLES = {
    "rel_member_of": "изменение — членство в группе",
    "rel_has_permission": "изменение — выдача права",
    "rel_parent_of": "изменение — вложенность ресурсов",
    "rel_owner_of": "изменение — владение ресурсом",
    "level_ordinal": "уровень выданного права, от чтения до администрирования",
    "is_permission": "это выдача права, а не членство или вложенность",
    "common_neighbours": "сколько общих соседей у субъекта и ресурса в графе",
    "adamic_adar": "вес общих соседей: редкие общие связи весомее массовых",
    "jaccard": "доля общих соседей среди всех соседей обоих концов",
    "path_hops": "длина кратчайшего пути от субъекта к ресурсу до этого изменения",
    "path_unreachable": "пути от субъекта к ресурсу не было вовсе",
    "bypasses_bucket": "право выдано на объект в обход прав на его бакет",
    "level_jump": "на сколько ступеней уровень выше прежнего у этого субъекта",
    "same_community": "субъект и ресурс принадлежат одному сообществу графа",
    "hour_sin": "час суток, синус — чтобы 23:00 и 00:00 были рядом",
    "hour_cos": "час суток, косинус — вторая половина той же пары",
    "dow_sin": "день недели, синус — чтобы воскресенье и понедельник были рядом",
    "dow_cos": "день недели, косинус — вторая половина той же пары",
    "is_off_hours": "изменение сделано вне рабочих часов",
    "is_weekend": "изменение сделано в выходной день",
    "actor_is_subject": "право выдал сам получатель, а не кто-то другой",
    "actor_level_on_object": "какой уровень на этом ресурсе был у инициатора",
}

_NODE_TITLES = {
    "type_user": "это пользователь",
    "type_group": "это группа",
    "type_bucket": "это бакет",
    "type_object": "это объект",
    "deg_member_of_out": "в скольких группах состоит",
    "deg_member_of_in": "сколько участников состоит в нём",
    "deg_has_permission_out": "на сколько ресурсов имеет права",
    "deg_has_permission_in": "скольким субъектам выданы права на него",
    "deg_parent_of_out": "сколько вложенных ресурсов содержит",
    "deg_parent_of_in": "в скольких контейнерах находится",
    "deg_owner_of_out": "сколькими ресурсами владеет",
    "deg_owner_of_in": "сколько владельцев у него",
    "max_level": "самый высокий уровень прав, который у него есть",
    "group_count": "в скольких различных группах состоит",
    "depth": "глубина в иерархии ресурсов",
    "triangles": "сколько треугольников связей проходит через него",
    "clustering": "насколько плотно связаны между собой его соседи",
    "same_community_share": "доля соседей из того же сообщества графа",
    "is_new": "появился незадолго до этого изменения",
    "age": "сколько времени существует",
    "gap": "сколько прошло с его прошлого изменения прав",
    "activity_1h": "сколько изменений прав за предыдущий час",
    "activity_24h": "сколько изменений прав за предыдущие сутки",
    "activity_7d": "сколько изменений прав за предыдущую неделю",
}

_ENDPOINTS = {"subj_": "у субъекта", "obj_": "у ресурса"}


def feature_titles() -> dict[str, str]:
    """A phrase for every feature name the candidate block can contain."""
    titles = dict(_EDGE_TITLES)
    for prefix, whose in _ENDPOINTS.items():
        for name in NODE_BLOCK.names:
            titles[f"{prefix}{name}"] = f"{whose}: {_NODE_TITLES[name]}"
    return titles


def describe_feature(name: str) -> str:
    """The phrase for one feature, or its bare name when it is not known here."""
    return feature_titles().get(name, name)
