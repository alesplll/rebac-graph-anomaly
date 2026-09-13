# Признаковое пространство

Воспроизводится: `uv run python scripts/thesis_tables.py`

Всего признаков: **70**. Каждый принадлежит одной группе, и группа определяет, какой уровень возможностей источника нужен, чтобы признак вообще можно было вычислить.

| Группа | Признаков | Что требуется от системы авторизации |
|---|---|---|
| Структурные | 52 | снимок кортежей отношений |
| Временные | 16 | метки времени создания |
| Происхождения | 2 | инициатор изменения |

Признак, который источник не может дать, помечается ненаблюдаемым, а не зануляется: ноль — законное значение почти для всех этих величин, и модель, получившая нули, выучила бы «инициатора не было» вместо «мы не знаем, кто это был».

## Признаки изменения

| Признак | Группа |
|---|---|
| `rel_member_of` | структурные |
| `rel_has_permission` | структурные |
| `rel_parent_of` | структурные |
| `rel_owner_of` | структурные |
| `level_ordinal` | структурные |
| `is_permission` | структурные |
| `common_neighbours` | структурные |
| `adamic_adar` | структурные |
| `jaccard` | структурные |
| `path_hops` | структурные |
| `path_unreachable` | структурные |
| `bypasses_bucket` | структурные |
| `level_jump` | структурные |
| `same_community` | структурные |
| `hour_sin` | временные |
| `hour_cos` | временные |
| `dow_sin` | временные |
| `dow_cos` | временные |
| `is_off_hours` | временные |
| `is_weekend` | временные |
| `actor_is_subject` | происхождения |
| `actor_level_on_object` | происхождения |

## Признаки субъекта

| Признак | Группа |
|---|---|
| `subj_type_user` | структурные |
| `subj_type_group` | структурные |
| `subj_type_bucket` | структурные |
| `subj_type_object` | структурные |
| `subj_deg_member_of_out` | структурные |
| `subj_deg_member_of_in` | структурные |
| `subj_deg_has_permission_out` | структурные |
| `subj_deg_has_permission_in` | структурные |
| `subj_deg_parent_of_out` | структурные |
| `subj_deg_parent_of_in` | структурные |
| `subj_deg_owner_of_out` | структурные |
| `subj_deg_owner_of_in` | структурные |
| `subj_max_level` | структурные |
| `subj_group_count` | структурные |
| `subj_depth` | структурные |
| `subj_triangles` | структурные |
| `subj_clustering` | структурные |
| `subj_same_community_share` | структурные |
| `subj_is_new` | структурные |
| `subj_age` | временные |
| `subj_gap` | временные |
| `subj_activity_1h` | временные |
| `subj_activity_24h` | временные |
| `subj_activity_7d` | временные |

## Признаки объекта

| Признак | Группа |
|---|---|
| `obj_type_user` | структурные |
| `obj_type_group` | структурные |
| `obj_type_bucket` | структурные |
| `obj_type_object` | структурные |
| `obj_deg_member_of_out` | структурные |
| `obj_deg_member_of_in` | структурные |
| `obj_deg_has_permission_out` | структурные |
| `obj_deg_has_permission_in` | структурные |
| `obj_deg_parent_of_out` | структурные |
| `obj_deg_parent_of_in` | структурные |
| `obj_deg_owner_of_out` | структурные |
| `obj_deg_owner_of_in` | структурные |
| `obj_max_level` | структурные |
| `obj_group_count` | структурные |
| `obj_depth` | структурные |
| `obj_triangles` | структурные |
| `obj_clustering` | структурные |
| `obj_same_community_share` | структурные |
| `obj_is_new` | структурные |
| `obj_age` | временные |
| `obj_gap` | временные |
| `obj_activity_1h` | временные |
| `obj_activity_24h` | временные |
| `obj_activity_7d` | временные |

