# satclip_l40: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.396 | 0.10 - 0.58            |          20 | location  |       443 |      449 |                0 |               0.187 |     0.396 |      0.383 | 0.426 |    0.125 |           8 |
| nlcd                |   0.198 | 0.00 - 0.45            |           1 | location  |       443 |      449 |                0 |               0.187 |     0.198 |      0.155 | 0.246 |    0.067 |          15 |
| lfrdb_ecosys        |   0.22  | 0.00 - 0.67            |          20 | location  |      4041 |     3986 |                0 |               0.996 |     0.22  |      0.147 | 0.22  |    0.045 |          22 |
| lfrdb_ecosys        |   0.118 | 0.00 - 0.50            |           1 | location  |      4041 |     3986 |                0 |               0.996 |     0.118 |      0.047 | 0.056 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.198 | 0.00 - 0.78            |          20 | location  |      4041 |     3985 |                0 |               0.996 |     0.198 |      0.145 | 0.236 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.093 | 0.00 - 0.52            |           1 | location  |      4041 |     3985 |                0 |               0.996 |     0.093 |      0.04  | 0.039 |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.285 | 0.00 - 0.77            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.285 |      0.264 | 0.316 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.159 | 0.00 - 0.85            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.159 |      0.118 | 0.193 |    0.018 |          57 |
| nps_vmi_group       |   0.305 | 0.00 - 1.00            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.305 |      0.276 | 0.329 |    0.043 |          23 |
| nps_vmi_group       |   0.104 | 0.00 - 0.89            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.104 |      0.059 | 0.129 |    0.01  |          96 |
| vegbench_formation  |   0.237 | 0.06 - 0.39            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.237 |      0.189 | 0.231 |    0.091 |          11 |
| vegbench_formation  |   0.2   | 0.00 - 0.39            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.2   |      0.097 | 0.133 |    0.062 |          16 |
| vegbench_division   |   0.211 | 0.02 - 0.38            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.211 |      0.143 | 0.185 |    0.062 |          16 |
| vegbench_division   |   0.163 | 0.00 - 0.34            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.163 |      0.079 | 0.091 |    0.048 |          21 |
| vegbench_macrogroup |   0.166 | 0.00 - 0.55            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.166 |      0.126 | 0.153 |    0.037 |          27 |
| vegbench_macrogroup |   0.161 | 0.00 - 0.26            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.161 |      0.069 | 0.079 |    0.026 |          38 |
| vegbench_group      |   0.14  | 0.00 - 0.55            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.14  |      0.093 | 0.144 |    0.024 |          41 |
| vegbench_group      |   0.122 | 0.00 - 0.55            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.122 |      0.048 | 0.077 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.86  |     0.186 | 0.57 - 0.99                 | 0.01 - 0.83              |          20 | location  |      6450 |     6484 |                0 |               0.215 |        0.86  |             0.036 |         164 |
| bbs_occurrence           |   0.877 |     0.125 | 0.21 - 1.00                 | 0.00 - 0.84              |           1 | location  |      6450 |     6484 |                0 |               0.215 |        0.877 |             0.017 |         358 |
| butterflies_occurrence   |   0.715 |     0.624 | 0.50 - 0.98                 | 0.20 - 0.91              |          20 | location  |       136 |      129 |                0 |               0.343 |        0.715 |             0.408 |          51 |
| butterflies_occurrence   |   0.787 |     0.46  | 0.00 - 1.00                 | 0.01 - 1.00              |           1 | location  |       136 |      129 |                0 |               0.343 |        0.787 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.76  |     0.277 | 0.48 - 0.95                 | 0.01 - 0.89              |          20 | location  |      1595 |     1814 |                0 |               0     |        0.76  |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.77  |     0.142 | 0.13 - 1.00                 | 0.00 - 0.98              |           1 | location  |      1595 |     1814 |                0 |               0     |        0.77  |             0.035 |         150 |
| satbird_winter_occupancy |   0.905 |     0.455 | 0.59 - 1.00                 | 0.02 - 0.95              |          20 | location  |     34327 |     6187 |                0 |               0     |        0.905 |             0.109 |         467 |
| satbird_winter_occupancy |   0.897 |     0.37  | 0.00 - 1.00                 | 0.00 - 1.00              |           1 | location  |     34327 |     6187 |                0 |               0     |        0.897 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.231 | 0.12 - 0.42             |         nan | location  |      4650 |     4583 |                0 |               0.101 |     0.231 |           0.478 |          12 |
| ofo_trees |   0.243 | 0.05 - 0.44             |         nan | location  |       146 |       51 |                0 |               0.015 |     0.243 |           0.66  |           2 |
