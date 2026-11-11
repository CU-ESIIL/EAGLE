# location_rff: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.319 | 0.15 - 0.43            |          20 | location  |       443 |      449 |                0 |               0.187 |     0.319 |      0.311 | 0.328 |    0.125 |           8 |
| nlcd                |   0.219 | 0.00 - 0.36            |           1 | location  |       443 |      449 |                0 |               0.187 |     0.219 |      0.207 | 0.241 |    0.067 |          15 |
| lfrdb_ecosys        |   0.292 | 0.00 - 0.80            |          20 | location  |      4041 |     3986 |                0 |               0.996 |     0.292 |      0.217 | 0.351 |    0.045 |          22 |
| lfrdb_ecosys        |   0.093 | 0.00 - 0.67            |           1 | location  |      4041 |     3986 |                0 |               0.996 |     0.093 |      0.05  | 0.053 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.135 | 0.00 - 0.49            |          20 | location  |      4041 |     3985 |                0 |               0.996 |     0.135 |      0.107 | 0.134 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.097 | 0.00 - 0.67            |           1 | location  |      4041 |     3985 |                0 |               0.996 |     0.097 |      0.055 | 0.07  |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.267 | 0.00 - 0.87            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.267 |      0.225 | 0.264 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.139 | 0.00 - 0.93            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.139 |      0.117 | 0.172 |    0.018 |          57 |
| nps_vmi_group       |   0.265 | 0.00 - 1.00            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.265 |      0.241 | 0.301 |    0.043 |          23 |
| nps_vmi_group       |   0.095 | 0.00 - 0.85            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.095 |      0.068 | 0.135 |    0.01  |          96 |
| vegbench_formation  |   0.227 | 0.07 - 0.32            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.227 |      0.179 | 0.217 |    0.091 |          11 |
| vegbench_formation  |   0.175 | 0.00 - 0.27            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.175 |      0.119 | 0.16  |    0.062 |          16 |
| vegbench_division   |   0.197 | 0.00 - 0.29            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.197 |      0.141 | 0.185 |    0.062 |          16 |
| vegbench_division   |   0.141 | 0.00 - 0.37            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.141 |      0.092 | 0.123 |    0.048 |          21 |
| vegbench_macrogroup |   0.141 | 0.00 - 0.46            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.141 |      0.095 | 0.131 |    0.037 |          27 |
| vegbench_macrogroup |   0.138 | 0.00 - 0.41            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.138 |      0.086 | 0.105 |    0.026 |          38 |
| vegbench_group      |   0.114 | 0.00 - 0.43            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.114 |      0.076 | 0.129 |    0.024 |          41 |
| vegbench_group      |   0.131 | 0.00 - 0.51            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.131 |      0.069 | 0.092 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.825 |     0.146 | 0.46 - 0.98                 | 0.00 - 0.82              |          20 | location  |      6450 |     6484 |                0 |               0.215 |        0.825 |             0.036 |         164 |
| bbs_occurrence           |   0.807 |     0.103 | 0.06 - 1.00                 | 0.00 - 1.00              |           1 | location  |      6450 |     6484 |                0 |               0.215 |        0.807 |             0.017 |         358 |
| butterflies_occurrence   |   0.713 |     0.609 | 0.52 - 0.98                 | 0.24 - 0.94              |          20 | location  |       136 |      129 |                0 |               0.343 |        0.713 |             0.408 |          51 |
| butterflies_occurrence   |   0.804 |     0.463 | 0.08 - 1.00                 | 0.01 - 1.00              |           1 | location  |       136 |      129 |                0 |               0.343 |        0.804 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.757 |     0.267 | 0.50 - 0.97                 | 0.02 - 0.92              |          20 | location  |      1595 |     1814 |                0 |               0     |        0.757 |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.741 |     0.119 | 0.08 - 1.00                 | 0.00 - 1.00              |           1 | location  |      1595 |     1814 |                0 |               0     |        0.741 |             0.035 |         150 |
| satbird_winter_occupancy |   0.896 |     0.418 | 0.57 - 1.00                 | 0.01 - 0.95              |          20 | location  |     34327 |     6187 |                0 |               0     |        0.896 |             0.109 |         467 |
| satbird_winter_occupancy |   0.876 |     0.352 | 0.03 - 1.00                 | 0.00 - 1.00              |           1 | location  |     34327 |     6187 |                0 |               0     |        0.876 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.192 | 0.09 - 0.40             |         nan | location  |      4650 |     4583 |                0 |               0.101 |     0.192 |           0.439 |          12 |
| ofo_trees |  -0.004 | -0.42 - 0.41            |         nan | location  |       146 |       51 |                0 |               0.015 |    -0.004 |           0.633 |           2 |
