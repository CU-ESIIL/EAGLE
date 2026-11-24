# satclip_l10: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.402 | 0.06 - 0.54            |          20 | location  |       443 |      449 |                0 |               0.187 |     0.402 |      0.381 | 0.42  |    0.125 |           8 |
| nlcd                |   0.203 | 0.00 - 0.42            |           1 | location  |       443 |      449 |                0 |               0.187 |     0.203 |      0.186 | 0.243 |    0.067 |          15 |
| lfrdb_ecosys        |   0.244 | 0.00 - 0.53            |          20 | location  |      4041 |     3986 |                0 |               0.996 |     0.244 |      0.157 | 0.23  |    0.045 |          22 |
| lfrdb_ecosys        |   0.104 | 0.00 - 0.67            |           1 | location  |      4041 |     3986 |                0 |               0.996 |     0.104 |      0.044 | 0.053 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.179 | 0.00 - 0.60            |          20 | location  |      4041 |     3985 |                0 |               0.996 |     0.179 |      0.112 | 0.253 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.095 | 0.00 - 0.67            |           1 | location  |      4041 |     3985 |                0 |               0.996 |     0.095 |      0.043 | 0.05  |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.245 | 0.00 - 0.61            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.245 |      0.215 | 0.325 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.158 | 0.00 - 0.94            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.158 |      0.089 | 0.181 |    0.018 |          57 |
| nps_vmi_group       |   0.272 | 0.00 - 0.92            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.272 |      0.223 | 0.319 |    0.043 |          23 |
| nps_vmi_group       |   0.131 | 0.00 - 0.94            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.131 |      0.064 | 0.1   |    0.01  |          96 |
| vegbench_formation  |   0.236 | 0.01 - 0.39            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.236 |      0.146 | 0.181 |    0.091 |          11 |
| vegbench_formation  |   0.189 | 0.00 - 0.41            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.189 |      0.106 | 0.143 |    0.062 |          16 |
| vegbench_division   |   0.18  | 0.00 - 0.34            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.18  |      0.121 | 0.144 |    0.062 |          16 |
| vegbench_division   |   0.138 | 0.00 - 0.34            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.138 |      0.078 | 0.097 |    0.048 |          21 |
| vegbench_macrogroup |   0.17  | 0.00 - 0.50            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.17  |      0.105 | 0.137 |    0.037 |          27 |
| vegbench_macrogroup |   0.129 | 0.00 - 0.29            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.129 |      0.049 | 0.053 |    0.026 |          38 |
| vegbench_group      |   0.148 | 0.00 - 0.45            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.148 |      0.083 | 0.154 |    0.024 |          41 |
| vegbench_group      |   0.118 | 0.00 - 0.46            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.118 |      0.038 | 0.065 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.87  |     0.186 | 0.61 - 0.99                 | 0.01 - 0.83              |          20 | location  |      6450 |     6484 |                0 |               0.215 |        0.87  |             0.036 |         164 |
| bbs_occurrence           |   0.884 |     0.124 | 0.23 - 1.00                 | 0.00 - 1.00              |           1 | location  |      6450 |     6484 |                0 |               0.215 |        0.884 |             0.017 |         358 |
| butterflies_occurrence   |   0.704 |     0.615 | 0.47 - 0.99                 | 0.19 - 0.95              |          20 | location  |       136 |      129 |                0 |               0.343 |        0.704 |             0.408 |          51 |
| butterflies_occurrence   |   0.771 |     0.427 | 0.03 - 1.00                 | 0.01 - 1.00              |           1 | location  |       136 |      129 |                0 |               0.343 |        0.771 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.727 |     0.259 | 0.42 - 0.95                 | 0.01 - 0.78              |          20 | location  |      1595 |     1814 |                0 |               0     |        0.727 |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.768 |     0.132 | 0.05 - 1.00                 | 0.00 - 0.98              |           1 | location  |      1595 |     1814 |                0 |               0     |        0.768 |             0.035 |         150 |
| satbird_winter_occupancy |   0.901 |     0.429 | 0.56 - 1.00                 | 0.02 - 0.95              |          20 | location  |     34327 |     6187 |                0 |               0     |        0.901 |             0.109 |         467 |
| satbird_winter_occupancy |   0.899 |     0.353 | 0.04 - 1.00                 | 0.00 - 1.00              |           1 | location  |     34327 |     6187 |                0 |               0     |        0.899 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.169 | 0.10 - 0.35             |         nan | location  |      4650 |     4583 |                0 |               0.101 |     0.169 |           0.402 |          12 |
| ofo_trees |  -0.026 | -0.42 - 0.37            |         nan | location  |       146 |       51 |                0 |               0.015 |    -0.026 |          -0.11  |           2 |
