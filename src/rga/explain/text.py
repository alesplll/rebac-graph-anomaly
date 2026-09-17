"""What is said about one change, and why it is worth a look.

Each observation comes in two parts. The first states what happened: it describes,
never concludes. The second says what the usual process looks like, so that a reader
who does not work with this system every day can judge for themselves whether the
departure matters. "The subject granted the right to itself" means nothing to someone
who does not know that rights are normally granted by a resource owner.

Neither part pronounces a verdict. The system hands an analyst a ranked list of
hypotheses with grounds; deciding what happened is the analyst's job, and a test walks
every candidate of a window to make sure the wording holds that line.
"""

from __future__ import annotations

from dataclasses import dataclass

from rga.domain.relations import PermissionLevel
from rga.features.edges import decode_level
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


@dataclass(frozen=True)
class Observation:
    """One thing noticed about a change, and why it is worth noticing."""

    text: str
    why: str


def _said(candidates: CandidateSet, position: int, name: str) -> float | None:
    """A feature's value, or None when the source could not supply it."""
    if not bool(candidates.matrix.observed(name)[position]):
        return None
    return float(candidates.matrix.column(name)[position])


def describe(candidates: CandidateSet, position: int) -> tuple[Observation, ...]:
    """What is worth saying about this change, most telling first."""
    subject, _, target = candidates.keys[position]
    found: list[Observation] = []

    level = _said(candidates, position, "level_ordinal")
    if level is not None and level > 0:
        name = PermissionLevel(decode_level(level)).name.lower()
        found.append(
            Observation(
                text=f"Выдан уровень «{name}» на ресурс {target}.",
                why=(
                    "Уровни упорядочены от чтения до администрирования, и каждый "
                    "включает нижестоящие. Чем выше уровень, тем больше операций "
                    "открывает одно это ребро графа."
                ),
            )
        )
    else:
        found.append(
            Observation(
                text=f"Создана связь между {subject} и {target}.",
                why=(
                    "Членство в группе само по себе прав не даёт, но наследует все "
                    "права этой группы. Поэтому вступление в группу расширяет доступ "
                    "не хуже прямой выдачи прав, а выглядит скромнее."
                ),
            )
        )

    if _said(candidates, position, "actor_is_subject") == 1.0:
        found.append(
            Observation(
                text="Право выдал сам субъект, а не кто-то другой.",
                why=(
                    "В штатной процедуре право выдаёт владелец ресурса или "
                    "администратор. Когда субъект выдаёт право себе, решение никем "
                    "со стороны не подтверждено, и обычная проверка «вторыми глазами» "
                    "не срабатывает. Само по себе это законно — владелец часто "
                    "повышает себе уровень на своём же бакете, — но в сочетании с "
                    "остальным это первое, на что стоит посмотреть."
                ),
            )
        )

    jump = _said(candidates, position, "level_jump")
    if jump is not None and jump >= 2:
        found.append(
            Observation(
                text=(
                    f"Уровень выше на {int(jump)} ступени, чем то, что у субъекта было "
                    "в этом поддереве ресурсов."
                ),
                why=(
                    "Обычно доступ расширяется постепенно, по мере работы: сначала "
                    "чтение, потом запись. Скачок сразу через несколько ступеней "
                    "означает, что промежуточные шаги пропущены."
                ),
            )
        )

    if _said(candidates, position, "bypasses_bucket") == 1.0:
        found.append(
            Observation(
                text="Право выдано на объект, хотя прав на содержащий его бакет нет.",
                why=(
                    "Права принято выдавать на бакет целиком: так они видны в одном "
                    "месте и снимаются вместе с ним. Право прямо на объект даёт доступ "
                    "к содержимому, не появляясь в списке прав на бакет."
                ),
            )
        )

    common = _said(candidates, position, "common_neighbours")
    if common is not None and common == 0:
        found.append(
            Observation(
                text="У субъекта и ресурса нет ни одного общего соседа в графе.",
                why=(
                    "Как правило, к моменту выдачи субъект и ресурс уже связаны: общая "
                    "группа, общий проект, соседние права. Отсутствие общих соседей "
                    "означает, что доступ выдан вне сложившейся структуры."
                ),
            )
        )

    if _said(candidates, position, "path_unreachable") == 1.0:
        found.append(
            Observation(
                text="До этого ресурса от субъекта не было пути по графу прав.",
                why=(
                    "Движок авторизации отвечает на запрос, идя по графу от субъекта к "
                    "ресурсу. Если пути не было, то до этого изменения субъект не мог "
                    "добраться до ресурса никаким способом — доступ появился с нуля, а "
                    "не расширил существующий."
                ),
            )
        )

    if _said(candidates, position, "is_off_hours") == 1.0:
        found.append(
            Observation(
                text=(
                    f"Изменение сделано в {hour_of_day(int(candidates.ts[position]))}:00 — "
                    "вне рабочих часов."
                ),
                why=(
                    "Основная масса изменений прав приходится на рабочее время, когда "
                    "рядом есть кому спросить и согласовать. Ночные изменения реже "
                    "проходят через обычные согласования и позже попадаются на глаза."
                ),
            )
        )

    if _said(candidates, position, "is_weekend") == 1.0:
        found.append(
            Observation(
                text="Изменение сделано в выходной день.",
                why=(
                    "То же соображение, что и с ночным временем: в выходной меньше "
                    "свидетелей у изменения и дольше срок до его разбора."
                ),
            )
        )

    return tuple(found)
