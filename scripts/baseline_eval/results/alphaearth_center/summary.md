# alphaearth_center: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.751 | 0.64 - 0.84            |          20 | center    |       443 |      449 |                0 |               0.187 |     0.751 |      0.741 | 0.748 |    0.125 |           8 |
| nlcd                |   0.577 | 0.24 - 0.84            |           1 | center    |       443 |      449 |                0 |               0.187 |     0.577 |      0.522 | 0.562 |    0.067 |          15 |
| lfrdb_ecosys        |   0.564 | 0.12 - 0.89            |          20 | center    |      4041 |     3986 |                0 |               0.996 |     0.564 |      0.54  | 0.584 |    0.045 |          22 |
| lfrdb_ecosys        |   0.237 | 0.00 - 0.75            |           1 | center    |      4041 |     3986 |                0 |               0.996 |     0.237 |      0.17  | 0.283 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.445 | 0.06 - 0.88            |          20 | center    |      4041 |     3985 |                0 |               0.996 |     0.445 |      0.418 | 0.432 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.214 | 0.00 - 1.00            |           1 | center    |      4041 |     3985 |                0 |               0.996 |     0.214 |      0.158 | 0.246 |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.483 | 0.02 - 0.93            |          20 | center    |      3441 |     2667 |                0 |               0.624 |     0.483 |      0.479 | 0.541 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.357 | 0.00 - 0.89            |           1 | center    |      3441 |     2667 |                0 |               0.624 |     0.357 |      0.296 | 0.419 |    0.018 |          57 |
| nps_vmi_group       |   0.566 | 0.00 - 1.00            |          20 | center    |      3441 |     2667 |                0 |               0.624 |     0.566 |      0.547 | 0.58  |    0.043 |          23 |
| nps_vmi_group       |   0.331 | 0.00 - 0.93            |           1 | center    |      3441 |     2667 |                0 |               0.624 |     0.331 |      0.263 | 0.337 |    0.01  |          96 |
| vegbench_formation  |   0.517 | 0.33 - 0.63            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.517 |      0.488 | 0.545 |    0.091 |          11 |
| vegbench_formation  |   0.379 | 0.00 - 0.63            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.379 |      0.333 | 0.511 |    0.062 |          16 |
| vegbench_division   |   0.469 | 0.12 - 0.61            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.469 |      0.404 | 0.449 |    0.062 |          16 |
| vegbench_division   |   0.42  | 0.00 - 0.59            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.42  |      0.318 | 0.418 |    0.048 |          21 |
| vegbench_macrogroup |   0.386 | 0.05 - 0.59            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.386 |      0.339 | 0.389 |    0.037 |          27 |
| vegbench_macrogroup |   0.35  | 0.00 - 0.65            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.35  |      0.257 | 0.321 |    0.026 |          38 |
| vegbench_group      |   0.351 | 0.05 - 0.70            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.351 |      0.31  | 0.341 |    0.024 |          41 |
| vegbench_group      |   0.269 | 0.00 - 0.65            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.269 |      0.208 | 0.283 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.884 |     0.221 | 0.64 - 0.99                 | 0.01 - 0.85              |          20 | center    |      6450 |     6484 |                0 |               0.215 |        0.884 |             0.036 |         164 |
| bbs_occurrence           |   0.888 |     0.14  | 0.37 - 1.00                 | 0.00 - 0.92              |           1 | center    |      6450 |     6484 |                0 |               0.215 |        0.888 |             0.017 |         358 |
| butterflies_occurrence   |   0.662 |     0.57  | 0.46 - 0.92                 | 0.20 - 0.92              |          20 | center    |       136 |      129 |                0 |               0.343 |        0.662 |             0.408 |          51 |
| butterflies_occurrence   |   0.746 |     0.399 | 0.00 - 1.00                 | 0.01 - 1.00              |           1 | center    |       136 |      129 |                0 |               0.343 |        0.746 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.803 |     0.322 | 0.55 - 0.98                 | 0.03 - 0.94              |          20 | center    |      1595 |     1814 |                0 |               0     |        0.803 |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.764 |     0.152 | 0.07 - 1.00                 | 0.00 - 1.00              |           1 | center    |      1595 |     1814 |                0 |               0     |        0.764 |             0.035 |         150 |
| satbird_winter_occupancy |   0.927 |     0.509 | 0.63 - 1.00                 | 0.01 - 0.96              |          20 | center    |     34327 |     6187 |                0 |               0     |        0.927 |             0.109 |         467 |
| satbird_winter_occupancy |   0.922 |     0.428 | 0.19 - 1.00                 | 0.00 - 1.00              |           1 | center    |     34327 |     6187 |                0 |               0     |        0.922 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.511 | 0.26 - 0.68             |         nan | center    |      4650 |     4583 |                0 |               0.101 |     0.511 |           0.649 |          12 |
| ofo_trees |   0.266 | 0.10 - 0.43             |         nan | center    |       146 |       51 |                0 |               0.015 |     0.266 |           0.661 |           2 |
