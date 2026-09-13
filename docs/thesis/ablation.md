### ablation — 5 seeds

| scorer | pr_auc | roc_auc | precision_at_20 | precision_at_50 | precision_at_100 |
|---|---|---|---|---|---|
| isolation_forest | 0.581 ± 0.102 | 0.982 ± 0.006 | 0.563 ± 0.136 | 0.468 ± 0.030 | 0.248 ± 0.022 |
| lof | 0.463 ± 0.232 | 0.844 ± 0.158 | 0.400 ± 0.235 | 0.329 ± 0.140 | 0.195 ± 0.059 |
| rules | 0.498 ± 0.141 | 0.949 ± 0.015 | 0.613 ± 0.159 | 0.341 ± 0.088 | 0.230 ± 0.031 |

### Recall at 50, by pattern

| scorer | cross_department | delegation_cascade | dormant_awakening | grant_burst | hierarchy_bypass | privileged_group_join | self_grant_admin | shadow_group |
|---|---|---|---|---|---|---|---|---|
| isolation_forest | 0.87 | 0.93 | 1.00 | 1.00 | 1.00 | 0.87 | 1.00 | 0.87 |
| lof | 0.27 | 0.58 | 0.33 | 0.56 | 0.07 | 0.83 | 0.87 | 0.99 |
| rules | 0.87 | 0.51 | 0.89 | 0.75 | 1.00 | 0.10 | 0.73 | 0.62 |

### ablation — feature groups

| feature groups | isolation_forest | lof | rules |
|---|---|---|---|
| structural | 0.517 ± 0.062 | 0.293 ± 0.200 | 0.328 ± 0.035 |
| structural+temporal | 0.583 ± 0.078 | 0.533 ± 0.193 | 0.576 ± 0.115 |
| structural+temporal+provenance | 0.642 ± 0.116 | 0.564 ± 0.200 | 0.591 ± 0.034 |

pr_auc, mean ± standard deviation over seeds.
