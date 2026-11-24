# alphaearth_center: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.738 | 0.56 - 0.82            |          20 | center    |       443 |      449 |                0 |               0.187 |     0.738 |      0.732 | 0.742 |    0.125 |           8 |
| nlcd                |   0.616 | 0.29 - 0.84            |           1 | center    |       443 |      449 |                0 |               0.187 |     0.616 |      0.571 | 0.607 |    0.067 |          15 |
| lfrdb_ecosys        |   0.571 | 0.13 - 0.87            |          20 | center    |      4041 |     3986 |                0 |               0.996 |     0.571 |      0.545 | 0.594 |    0.045 |          22 |
| lfrdb_ecosys        |   0.235 | 0.00 - 1.00            |           1 | center    |      4041 |     3986 |                0 |               0.996 |     0.235 |      0.177 | 0.286 |    0.009 |         112 |
| lfrdb_nvc_group     |   0.437 | 0.04 - 0.89            |          20 | center    |      4041 |     3985 |                0 |               0.996 |     0.437 |      0.407 | 0.437 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.221 | 0.00 - 0.75            |           1 | center    |      4041 |     3985 |                0 |               0.996 |     0.221 |      0.148 | 0.23  |    0.009 |         114 |
| nps_vmi_macrogroup  |   0.466 | 0.02 - 0.93            |          20 | center    |      3441 |     2667 |                0 |               0.624 |     0.466 |      0.466 | 0.529 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.372 | 0.00 - 0.91            |           1 | center    |      3441 |     2667 |                0 |               0.624 |     0.372 |      0.281 | 0.41  |    0.018 |          57 |
| nps_vmi_group       |   0.577 | 0.00 - 1.00            |          20 | center    |      3441 |     2667 |                0 |               0.624 |     0.577 |      0.545 | 0.576 |    0.043 |          23 |
| nps_vmi_group       |   0.297 | 0.00 - 0.93            |           1 | center    |      3441 |     2667 |                0 |               0.624 |     0.297 |      0.251 | 0.348 |    0.01  |          96 |
| vegbench_formation  |   0.519 | 0.33 - 0.63            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.519 |      0.492 | 0.548 |    0.091 |          11 |
| vegbench_formation  |   0.386 | 0.00 - 0.63            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.386 |      0.336 | 0.515 |    0.062 |          16 |
| vegbench_division   |   0.462 | 0.12 - 0.60            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.462 |      0.399 | 0.447 |    0.062 |          16 |
| vegbench_division   |   0.433 | 0.00 - 0.60            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.433 |      0.325 | 0.427 |    0.048 |          21 |
| vegbench_macrogroup |   0.404 | 0.05 - 0.62            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.404 |      0.342 | 0.382 |    0.037 |          27 |
| vegbench_macrogroup |   0.344 | 0.00 - 0.64            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.344 |      0.257 | 0.326 |    0.026 |          38 |
| vegbench_group      |   0.344 | 0.07 - 0.74            |          20 | center    |      6163 |     3277 |                0 |               0.472 |     0.344 |      0.307 | 0.334 |    0.024 |          41 |
| vegbench_group      |   0.293 | 0.00 - 0.65            |           1 | center    |      6163 |     3277 |                0 |               0.472 |     0.293 |      0.2   | 0.266 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.895 |     0.23  | 0.66 - 0.99                 | 0.01 - 0.85              |          20 | center    |      6450 |     6484 |                0 |               0.215 |        0.895 |             0.036 |         164 |
| bbs_occurrence           |   0.908 |     0.153 | 0.32 - 1.00                 | 0.00 - 0.92              |           1 | center    |      6450 |     6484 |                0 |               0.215 |        0.908 |             0.017 |         358 |
| butterflies_occurrence   |   0.664 |     0.571 | 0.45 - 0.92                 | 0.20 - 0.92              |          20 | center    |       136 |      129 |                0 |               0.343 |        0.664 |             0.408 |          51 |
| butterflies_occurrence   |   0.748 |     0.395 | 0.00 - 1.00                 | 0.01 - 1.00              |           1 | center    |       136 |      129 |                0 |               0.343 |        0.748 |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.806 |     0.325 | 0.55 - 0.98                 | 0.03 - 0.94              |          20 | center    |      1595 |     1814 |                0 |               0     |        0.806 |             0.126 |          37 |
| snapshot_usa_occurrence  |   0.799 |     0.16  | 0.15 - 1.00                 | 0.00 - 1.00              |           1 | center    |      1595 |     1814 |                0 |               0     |        0.799 |             0.035 |         150 |
| satbird_winter_occupancy |   0.927 |     0.511 | 0.63 - 1.00                 | 0.01 - 0.96              |          20 | center    |     34327 |     6187 |                0 |               0     |        0.927 |             0.109 |         467 |
| satbird_winter_occupancy |   0.925 |     0.425 | 0.16 - 1.00                 | 0.00 - 1.00              |           1 | center    |     34327 |     6187 |                0 |               0     |        0.925 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.511 | 0.26 - 0.68             |         nan | center    |      4650 |     4583 |                0 |               0.101 |     0.511 |            0.65 |          12 |
| ofo_trees |   0.295 | 0.13 - 0.46             |         nan | center    |       146 |       51 |                0 |               0.015 |     0.295 |            0.66 |           2 |
