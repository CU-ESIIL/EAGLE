# alphaearth_center: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year.

## multiclass

| task                | pooling   |   min_count |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|:----------|------------:|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                | center    |          20 |       443 |      449 |                0 |               0.187 |     0.751 |      0.741 | 0.748 |    0.125 |           8 |
| nlcd                | center    |           1 |       443 |      449 |                0 |               0.187 |     0.577 |      0.522 | 0.562 |    0.067 |          15 |
| lfrdb_ecosys        | center    |          20 |      4041 |     3986 |                0 |               0.996 |     0.564 |      0.54  | 0.584 |    0.045 |          22 |
| lfrdb_ecosys        | center    |           1 |      4041 |     3986 |                0 |               0.996 |     0.237 |      0.17  | 0.283 |    0.009 |         112 |
| lfrdb_nvc_group     | center    |          20 |      4041 |     3985 |                0 |               0.996 |     0.445 |      0.418 | 0.432 |    0.037 |          27 |
| lfrdb_nvc_group     | center    |           1 |      4041 |     3985 |                0 |               0.996 |     0.214 |      0.158 | 0.246 |    0.009 |         114 |
| nps_vmi_macrogroup  | center    |          20 |      3441 |     2667 |                0 |               0.624 |     0.483 |      0.479 | 0.541 |    0.042 |          24 |
| nps_vmi_macrogroup  | center    |           1 |      3441 |     2667 |                0 |               0.624 |     0.357 |      0.296 | 0.419 |    0.018 |          57 |
| nps_vmi_group       | center    |          20 |      3441 |     2667 |                0 |               0.624 |     0.566 |      0.547 | 0.58  |    0.043 |          23 |
| nps_vmi_group       | center    |           1 |      3441 |     2667 |                0 |               0.624 |     0.331 |      0.263 | 0.337 |    0.01  |          96 |
| vegbench_formation  | center    |          20 |      6163 |     3277 |                0 |               0.472 |     0.517 |      0.488 | 0.545 |    0.091 |          11 |
| vegbench_formation  | center    |           1 |      6163 |     3277 |                0 |               0.472 |     0.379 |      0.333 | 0.511 |    0.062 |          16 |
| vegbench_division   | center    |          20 |      6163 |     3277 |                0 |               0.472 |     0.469 |      0.404 | 0.449 |    0.062 |          16 |
| vegbench_division   | center    |           1 |      6163 |     3277 |                0 |               0.472 |     0.42  |      0.318 | 0.418 |    0.048 |          21 |
| vegbench_macrogroup | center    |          20 |      6163 |     3277 |                0 |               0.472 |     0.386 |      0.339 | 0.389 |    0.037 |          27 |
| vegbench_macrogroup | center    |           1 |      6163 |     3277 |                0 |               0.472 |     0.35  |      0.257 | 0.321 |    0.026 |          38 |
| vegbench_group      | center    |          20 |      6163 |     3277 |                0 |               0.472 |     0.351 |      0.31  | 0.341 |    0.024 |          41 |
| vegbench_group      | center    |           1 |      6163 |     3277 |                0 |               0.472 |     0.269 |      0.208 | 0.283 |    0.014 |          73 |

## multilabel

| task                     | pooling   |   min_count |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_ap |   mean_prevalence |   n_species |
|:-------------------------|:----------|------------:|----------:|---------:|-----------------:|--------------------:|-------------:|----------:|------------------:|------------:|
| bbs_occurrence           | center    |          20 |      6450 |     6484 |                0 |               0.215 |        0.884 |     0.221 |             0.036 |         164 |
| bbs_occurrence           | center    |           1 |      6450 |     6484 |                0 |               0.215 |        0.888 |     0.14  |             0.017 |         358 |
| butterflies_occurrence   | center    |          20 |       136 |      129 |                0 |               0.343 |        0.662 |     0.57  |             0.408 |          51 |
| butterflies_occurrence   | center    |           1 |       136 |      129 |                0 |               0.343 |        0.746 |     0.399 |             0.136 |         215 |
| snapshot_usa_occurrence  | center    |          20 |      1595 |     1814 |                0 |               0     |        0.803 |     0.322 |             0.126 |          37 |
| snapshot_usa_occurrence  | center    |           1 |      1595 |     1814 |                0 |               0     |        0.764 |     0.152 |             0.035 |         150 |
| satbird_winter_occupancy | center    |          20 |     34327 |     6187 |                0 |               0     |        0.927 |     0.509 |             0.109 |         467 |
| satbird_winter_occupancy | center    |           1 |     34327 |     6187 |                0 |               0     |        0.922 |     0.428 |             0.087 |         587 |

## regression

| task      | pooling   |   min_count |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|:----------|------------:|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   | center    |         nan |      4650 |     4583 |                0 |               0.101 |     0.511 |           0.649 |          12 |
| ofo_trees | center    |         nan |       146 |       51 |                0 |               0.015 |     0.266 |           0.661 |           2 |
