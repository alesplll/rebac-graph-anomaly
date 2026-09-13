### baselines — 5 seeds

| scorer | pr_auc | roc_auc | precision_at_20 | precision_at_50 | precision_at_100 |
|---|---|---|---|---|---|
| isolation_forest | 0.642 ± 0.116 | 0.985 ± 0.006 | 0.670 ± 0.129 | 0.464 ± 0.027 | 0.248 ± 0.022 |
| lof | 0.564 ± 0.200 | 0.925 ± 0.037 | 0.490 ± 0.248 | 0.368 ± 0.119 | 0.220 ± 0.035 |
| rules | 0.591 ± 0.034 | 0.947 ± 0.012 | 0.660 ± 0.058 | 0.364 ± 0.085 | 0.214 ± 0.039 |

### Recall at 50, by pattern

| scorer | cross_department | delegation_cascade | dormant_awakening | grant_burst | hierarchy_bypass | privileged_group_join | self_grant_admin | shadow_group |
|---|---|---|---|---|---|---|---|---|
| isolation_forest | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 0.83 |
| lof | 0.20 | 0.53 | 0.40 | 0.72 | 0.10 | 1.00 | 1.00 | 1.00 |
| rules | 0.60 | 0.33 | 1.00 | 1.00 | 1.00 | 0.10 | 1.00 | 0.46 |
