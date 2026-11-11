# satclip_l40: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.393 | 0.12 - 0.52            |          20 | location  |       443 |      449 |                0 |               0.187 |     0.393 |      0.382 | 0.411 |    0.125 |           8 |
| nlcd                |   0.242 | 0.00 - 0.48            |           1 | location  |       443 |      449 |                0 |               0.187 |     0.242 |      0.225 | 0.288 |    0.067 |          15 |
| lfrdb_ecosys        |   0.29  | 0.00 - 0.80            |          20 | location  |      4041 |     3986 |                0 |               0.996 |     0.29  |      0.2   | 0.331 |    0.045 |          22 |
| lfrdb_ecosys        |   0.117 | 0.00 - 0.67            |           1 | location  |      4041 |     3986 |                0 |               0.996 |     0.117 |      0.045 | 0.069 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.166 | 0.00 - 0.77            |          20 | location  |      4041 |     3985 |                0 |               0.996 |     0.166 |      0.129 | 0.195 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.09  | 0.00 - 0.46            |           1 | location  |      4041 |     3985 |                0 |               0.996 |     0.09  |      0.033 | 0.047 |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.274 | 0.00 - 0.75            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.274 |      0.243 | 0.327 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.173 | 0.00 - 0.94            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.173 |      0.117 | 0.213 |    0.018 |          57 |
| nps_vmi_group       |   0.271 | 0.00 - 1.00            |          20 | location  |      3441 |     2667 |                0 |               0.624 |     0.271 |      0.243 | 0.304 |    0.043 |          23 |
| nps_vmi_group       |   0.129 | 0.00 - 0.94            |           1 | location  |      3441 |     2667 |                0 |               0.624 |     0.129 |      0.062 | 0.104 |    0.01  |          96 |
| vegbench_formation  |   0.24  | 0.06 - 0.40            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.24  |      0.18  | 0.21  |    0.091 |          11 |
| vegbench_formation  |   0.195 | 0.00 - 0.41            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.195 |      0.099 | 0.12  |    0.062 |          16 |
| vegbench_division   |   0.2   | 0.01 - 0.36            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.2   |      0.153 | 0.189 |    0.062 |          16 |
| vegbench_division   |   0.149 | 0.00 - 0.34            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.149 |      0.07  | 0.084 |    0.048 |          21 |
| vegbench_macrogroup |   0.163 | 0.00 - 0.51            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.163 |      0.114 | 0.152 |    0.037 |          27 |
| vegbench_macrogroup |   0.16  | 0.00 - 0.30            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.16  |      0.072 | 0.075 |    0.026 |          38 |
| vegbench_group      |   0.139 | 0.00 - 0.55            |          20 | location  |      6163 |     3277 |                0 |               0.472 |     0.139 |      0.091 | 0.158 |    0.024 |          41 |
| vegbench_group      |   0.13  | 0.00 - 0.53            |           1 | location  |      6163 |     3277 |                0 |               0.472 |     0.13  |      0.056 | 0.083 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.853 |     0.181 | 0.55 - 0.99                 | 0.01 - 0.83              |          20 | location  |      6450 |     6484 |                0 |               0.215 |        0.853 |             0.036 |         164 |
| bbs_occurrence           |   0.806 |     0.117 | 0.00 - 1.00                 | 0.00 - 1.00              |           1 | location  |      6450 |     6484 |                0 |               0.215 |        0.806 |             0.017 |         358 |
| butterflies_occurrence   |   0.716 |     0.626 | 0.50 - 0.98                 | 0.20 - 0.92              |          20 | location  |       136 |      129 |                0 |               0.343 |        0.716 |             0.408 |          51 |
| butterflies_occurrence   |   0.787 |     0.462 | 0.00 - 1.00                 | 0.01 - 1.00              |           1 | location  |       136 |      129 |                0 |               0.343 |        0.787 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.757 |     0.276 | 0.47 - 0.97                 | 0.01 - 0.88              |          20 | location  |      1595 |     1814 |                0 |               0     |        0.757 |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.718 |     0.134 | 0.01 - 1.00                 | 0.00 - 1.00              |           1 | location  |      1595 |     1814 |                0 |               0     |        0.718 |             0.035 |         150 |
| satbird_winter_occupancy |   0.905 |     0.453 | 0.59 - 1.00                 | 0.02 - 0.96              |          20 | location  |     34327 |     6187 |                0 |               0     |        0.905 |             0.109 |         467 |
| satbird_winter_occupancy |   0.89  |     0.382 | 0.01 - 1.00                 | 0.00 - 1.00              |           1 | location  |     34327 |     6187 |                0 |               0     |        0.89  |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.204 | 0.08 - 0.41             |         nan | location  |      4650 |     4583 |                0 |               0.101 |     0.204 |           0.48  |          12 |
| ofo_trees |   0.26  | 0.07 - 0.45             |         nan | location  |       146 |       51 |                0 |               0.015 |     0.26  |           0.665 |           2 |
