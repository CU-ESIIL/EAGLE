# location_rff: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.306 | 0.11 - 0.44            |          20 | location  |       443 |      449 |                0 |               0.187 |     0.306 |      0.266 | 0.313 |    0.125 |           8 |
| nlcd                |   0.216 | 0.00 - 0.39            |           1 | location  |       443 |      449 |                0 |               0.187 |     0.216 |      0.195 | 0.239 |    0.067 |          15 |
| lfrdb_ecosys        |   0.283 | 0.00 - 0.80            |          20 | location  |      4041 |     3986 |                0 |               0.996 |     0.283 |      0.207 | 0.343 |    0.045 |          22 |
| lfrdb_ecosys        |   0.054 | 0.00 - 0.67            |           1 | location  |      4041 |     3986 |                0 |               0.996 |     0.054 |      0.032 | 0.038 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.146 | 0.00 - 0.48            |          20 | location  |      4041 |     3985 |                0 |               0.996 |     0.146 |      0.116 | 0.147 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.094 | 0.00 - 0.67            |           1 | location  |      4041 |     3985 |                0 |               0.996 |     0.094 |      0.053 | 0.073 |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.271 | 0.00 - 0.87            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.271 |      0.227 | 0.284 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.155 | 0.00 - 0.88            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.155 |      0.112 | 0.17  |    0.018 |          57 |
| nps_vmi_group       |   0.197 | 0.00 - 1.00            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.197 |      0.184 | 0.229 |    0.043 |          23 |
| nps_vmi_group       |   0.094 | 0.00 - 0.83            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.094 |      0.07  | 0.147 |    0.01  |          96 |
| vegbench_formation  |   0.227 | 0.07 - 0.33            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.227 |      0.183 | 0.22  |    0.091 |          11 |
| vegbench_formation  |   0.172 | 0.00 - 0.25            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.172 |      0.119 | 0.162 |    0.062 |          16 |
| vegbench_division   |   0.188 | 0.00 - 0.40            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.188 |      0.136 | 0.179 |    0.062 |          16 |
| vegbench_division   |   0.169 | 0.00 - 0.41            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.169 |      0.115 | 0.122 |    0.048 |          21 |
| vegbench_macrogroup |   0.144 | 0.00 - 0.46            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.144 |      0.097 | 0.132 |    0.037 |          27 |
| vegbench_macrogroup |   0.144 | 0.00 - 0.38            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.144 |      0.086 | 0.092 |    0.026 |          38 |
| vegbench_group      |   0.116 | 0.00 - 0.40            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.116 |      0.079 | 0.125 |    0.024 |          41 |
| vegbench_group      |   0.125 | 0.00 - 0.40            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.125 |      0.052 | 0.061 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.858 |     0.176 | 0.53 - 0.99                 | 0.01 - 0.82              |          20 | location  |      6450 |     6484 |                0 |               0.215 |        0.858 |             0.036 |         164 |
| bbs_occurrence           |   0.873 |     0.123 | 0.04 - 1.00                 | 0.00 - 1.00              |           1 | location  |      6450 |     6484 |                0 |               0.215 |        0.873 |             0.017 |         358 |
| butterflies_occurrence   |   0.715 |     0.611 | 0.51 - 0.97                 | 0.24 - 0.95              |          20 | location  |       136 |      129 |                0 |               0.343 |        0.715 |             0.408 |          51 |
| butterflies_occurrence   |   0.805 |     0.463 | 0.08 - 1.00                 | 0.01 - 1.00              |           1 | location  |       136 |      129 |                0 |               0.343 |        0.805 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.761 |     0.27  | 0.51 - 0.97                 | 0.02 - 0.92              |          20 | location  |      1595 |     1814 |                0 |               0     |        0.761 |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.763 |     0.121 | 0.05 - 1.00                 | 0.00 - 0.91              |           1 | location  |      1595 |     1814 |                0 |               0     |        0.763 |             0.035 |         150 |
| satbird_winter_occupancy |   0.897 |     0.423 | 0.57 - 1.00                 | 0.02 - 0.95              |          20 | location  |     34327 |     6187 |                0 |               0     |        0.897 |             0.109 |         467 |
| satbird_winter_occupancy |   0.893 |     0.357 | 0.12 - 1.00                 | 0.00 - 1.00              |           1 | location  |     34327 |     6187 |                0 |               0     |        0.893 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.192 | 0.09 - 0.39             |         nan | location  |      4650 |     4583 |                0 |               0.101 |     0.192 |           0.437 |          12 |
| ofo_trees |   0.291 | 0.17 - 0.42             |         nan | location  |       146 |       51 |                0 |               0.015 |     0.291 |           0.713 |           2 |
