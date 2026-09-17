"""Template sentences describing one change.

Every sentence states something observed and stops. The system hands an analyst a
ranked list of hypotheses with grounds; it does not announce a compromise, and the
wording has to hold that line even when the signal looks obvious. A test walks every
candidate of a window and fails if any forbidden word appears.

Sentences are produced only for features the source actually supplied. A right whose
initiator is unknown gets no sentence about its initiator, rather than a sentence
saying nobody granted it.
"""

from __future__ import annotations

from rga.domain.relations import PermissionLevel
from rga.features.spec import CandidateSet
from rga.util.timeutil import hour_of_day

#: Words that pronounce a verdict. None of them belongs in what an analyst is shown.
FORBIDDEN_WORDS = (
    "компрометац",
    "атака",
    "злоумышленник",
    "взлом",
    "нарушитель",
    "эскалация привилегий",
)


def _said(candidates: CandidateSet, position: int, name: str) -> float | None:
    """A feature's value, or None when the source could not supply it."""
    if not bool(candidates.matrix.observed(name)[position]):
        return None
    return float(candidates.matrix.column(name)[position])


def describe(candidates: CandidateSet, position: int) -> tuple[str, ...]:
    """What is worth saying about this change, most telling first."""
    subject, _, target = candidates.keys[position]
    lines: list[str] = []

    level = _said(candidates, position, "level_ordinal")
    if level is not None and level > 0:
        name = PermissionLevel(round(level)).name.lower()
        lines.append(f"Выдан уровень «{name}» на ресурс {target}.")
    else:
        lines.append(f"Создана связь между {subject} и {target}.")

    if _said(candidates, position, "actor_is_subject") == 1.0:
        lines.append("Право выдал сам субъект, а не кто-то другой.")

    jump = _said(candidates, position, "level_jump")
    if jump is not None and jump >= 2:
        lines.append(
            f"Уровень выше на {int(jump)} ступени, чем то, что у субъекта было "
            "в этом поддереве ресурсов."
        )

    if _said(candidates, position, "bypasses_bucket") == 1.0:
        lines.append("Право выдано на объект, хотя прав на содержащий его бакет нет.")

    common = _said(candidates, position, "common_neighbours")
    if common is not None and common == 0:
        lines.append("У субъекта и ресурса нет ни одного общего соседа в графе.")

    if _said(candidates, position, "path_unreachable") == 1.0:
        lines.append("До этого ресурса от субъекта не было пути по графу прав.")

    if _said(candidates, position, "is_off_hours") == 1.0:
        lines.append(
            f"Изменение сделано в {hour_of_day(int(candidates.ts[position]))}:00 — "
            "вне рабочих часов."
        )

    if _said(candidates, position, "is_weekend") == 1.0:
        lines.append("Изменение сделано в выходной день.")

    return tuple(lines)
