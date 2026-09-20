### module5 — 5 seeds

| scorer | pr_auc | roc_auc | precision_at_20 | precision_at_50 | precision_at_100 |
|---|---|---|---|---|---|
| gnn | 0.266 ± 0.167 | 0.708 ± 0.136 | 0.300 ± 0.207 | 0.128 ± 0.077 | 0.100 ± 0.049 |
| gnn_correspondence | 0.071 ± 0.015 | 0.593 ± 0.067 | 0.090 ± 0.073 | 0.064 ± 0.045 | 0.058 ± 0.031 |
| gnn_neighbourhood | 0.203 ± 0.162 | 0.731 ± 0.073 | 0.170 ± 0.196 | 0.132 ± 0.071 | 0.100 ± 0.040 |
| isolation_forest | 0.719 ± 0.118 | 0.989 ± 0.005 | 0.760 ± 0.107 | 0.488 ± 0.039 | 0.248 ± 0.022 |

### Recall at 50, by pattern

| scorer | cross_department | delegation_cascade | dormant_awakening | grant_burst | hierarchy_bypass | privileged_group_join | self_grant_admin | shadow_group |
|---|---|---|---|---|---|---|---|---|
| gnn | 0.00 | 0.20 | 0.80 | 0.40 | 0.40 | 1.00 | 0.00 | 0.00 |
| gnn_correspondence | 0.00 | 0.07 | 0.00 | 0.22 | 0.20 | 0.20 | 0.00 | 0.20 |
| gnn_neighbourhood | 0.60 | 0.33 | 0.40 | 0.35 | 0.50 | 0.40 | 0.20 | 0.02 |
| isolation_forest | 1.00 | 0.93 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 |

### Recall at 50: known patterns against hidden ones

| scorer | known | hidden | gap |
|---|---|---|---|
| gnn | 0.36 | 0.33 | +0.03 |
| gnn_correspondence | 0.12 | 0.09 | +0.04 |
| gnn_neighbourhood | 0.41 | 0.25 | +0.16 |
| isolation_forest | 0.98 | 0.98 | +0.00 |

Известными считаются паттерны 1-5 из раздела 5.3, на которых обучается
супервизорный вариант; скрытыми — паттерны 6-8, которых он не видел.
Самообучаемая модель не видит меток вообще, поэтому для неё все восемь
одинаково незнакомы, и разрыв у неё должен быть близок к нулю.
