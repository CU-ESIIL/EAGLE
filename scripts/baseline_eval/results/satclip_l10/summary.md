# satclip_l10: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.402 | 0.00 - 0.54            |          20 | location  |       443 |      449 |                0 |               0.187 |     0.402 |      0.372 | 0.42  |    0.125 |           8 |
| nlcd                |   0.202 | 0.00 - 0.43            |           1 | location  |       443 |      449 |                0 |               0.187 |     0.202 |      0.184 | 0.241 |    0.067 |          15 |
| lfrdb_ecosys        |   0.253 | 0.00 - 0.80            |          20 | location  |      4041 |     3986 |                0 |               0.996 |     0.253 |      0.172 | 0.302 |    0.045 |          22 |
| lfrdb_ecosys        |   0.085 | 0.00 - 0.67            |           1 | location  |      4041 |     3986 |                0 |               0.996 |     0.085 |      0.035 | 0.042 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.198 | 0.00 - 0.60            |          20 | location  |      4041 |     3985 |                0 |               0.996 |     0.198 |      0.129 | 0.272 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.099 | 0.00 - 0.67            |           1 | location  |      4041 |     3985 |                0 |               0.996 |     0.099 |      0.041 | 0.043 |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.277 | 0.00 - 0.68            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.277 |      0.226 | 0.337 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.185 | 0.00 - 0.94            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.185 |      0.106 | 0.222 |    0.018 |          57 |
| nps_vmi_group       |   0.323 | 0.00 - 1.00            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.323 |      0.25  | 0.327 |    0.043 |          23 |
| nps_vmi_group       |   0.128 | 0.00 - 0.94            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.128 |      0.063 | 0.095 |    0.01  |          96 |
| vegbench_formation  |   0.257 | 0.00 - 0.46            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.257 |      0.129 | 0.18  |    0.091 |          11 |
| vegbench_formation  |   0.175 | 0.00 - 0.42            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.175 |      0.081 | 0.152 |    0.062 |          16 |
| vegbench_division   |   0.208 | 0.00 - 0.40            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.208 |      0.133 | 0.158 |    0.062 |          16 |
| vegbench_division   |   0.122 | 0.00 - 0.42            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.122 |      0.079 | 0.112 |    0.048 |          21 |
| vegbench_macrogroup |   0.178 | 0.00 - 0.52            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.178 |      0.107 | 0.137 |    0.037 |          27 |
| vegbench_macrogroup |   0.132 | 0.00 - 0.37            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.132 |      0.052 | 0.056 |    0.026 |          38 |
| vegbench_group      |   0.144 | 0.00 - 0.50            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.144 |      0.079 | 0.155 |    0.024 |          41 |
| vegbench_group      |   0.127 | 0.00 - 0.56            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.127 |      0.039 | 0.053 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.868 |     0.183 | 0.61 - 0.99                 | 0.01 - 0.83              |          20 | location  |      6450 |     6484 |                0 |               0.215 |        0.868 |             0.036 |         164 |
| bbs_occurrence           |   0.842 |     0.121 | 0.01 - 1.00                 | 0.00 - 0.83              |           1 | location  |      6450 |     6484 |                0 |               0.215 |        0.842 |             0.017 |         358 |
| butterflies_occurrence   |   0.703 |     0.617 | 0.39 - 0.98                 | 0.20 - 0.94              |          20 | location  |       136 |      129 |                0 |               0.343 |        0.703 |             0.408 |          51 |
| butterflies_occurrence   |   0.784 |     0.44  | 0.09 - 1.00                 | 0.01 - 1.00              |           1 | location  |       136 |      129 |                0 |               0.343 |        0.784 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.75  |     0.262 | 0.47 - 0.95                 | 0.01 - 0.79              |          20 | location  |      1595 |     1814 |                0 |               0     |        0.75  |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.736 |     0.12  | 0.04 - 1.00                 | 0.00 - 0.79              |           1 | location  |      1595 |     1814 |                0 |               0     |        0.736 |             0.035 |         150 |
| satbird_winter_occupancy |   0.903 |     0.438 | 0.56 - 1.00                 | 0.03 - 0.95              |          20 | location  |     34327 |     6187 |                0 |               0     |        0.903 |             0.109 |         467 |
| satbird_winter_occupancy |   0.897 |     0.367 | 0.01 - 1.00                 | 0.00 - 1.00              |           1 | location  |     34327 |     6187 |                0 |               0     |        0.897 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.177 | 0.11 - 0.36             |         nan | location  |      4650 |     4583 |                0 |               0.101 |     0.177 |           0.431 |          12 |
| ofo_trees |   0.025 | -0.34 - 0.39            |         nan | location  |       146 |       51 |                0 |               0.015 |     0.025 |          -0.09  |           2 |
