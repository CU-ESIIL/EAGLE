# naip_clay: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.606 | 0.35 - 0.73            |          20 | naip_r15m |       443 |      449 |                0 |               0.094 |     0.606 |      0.594 | 0.613 |    0.125 |           8 |
| nlcd                |   0.524 | 0.25 - 0.76            |           1 | naip_r15m |       443 |      449 |                0 |               0.094 |     0.524 |      0.478 | 0.507 |    0.067 |          15 |
| lfrdb_ecosys        |   0.37  | 0.00 - 0.82            |          20 | naip_r50m |      3966 |     3711 |              350 |               0.448 |     0.37  |      0.354 | 0.45  |    0.045 |          22 |
| lfrdb_ecosys        |   0.129 | 0.00 - 0.67            |           1 | naip_r50m |      3966 |     3711 |              350 |               0.448 |     0.129 |      0.104 | 0.208 |    0.01  |          96 |
| lfrdb_nvc_group     |   0.264 | 0.00 - 0.84            |          20 | naip_r50m |      3966 |     3710 |              350 |               0.448 |     0.264 |      0.237 | 0.322 |    0.037 |          27 |
| lfrdb_nvc_group     |   0.129 | 0.00 - 0.74            |           1 | naip_r50m |      3966 |     3710 |              350 |               0.448 |     0.129 |      0.087 | 0.181 |    0.01  |         100 |
| nps_vmi_macrogroup  |   0.365 | 0.00 - 0.90            |          20 | naip_r30m |      3441 |     2661 |                6 |               0.084 |     0.365 |      0.354 | 0.415 |    0.042 |          24 |
| nps_vmi_macrogroup  |   0.283 | 0.00 - 0.83            |           1 | naip_r30m |      3441 |     2661 |                6 |               0.084 |     0.283 |      0.232 | 0.322 |    0.018 |          57 |
| nps_vmi_group       |   0.395 | 0.00 - 0.96            |          20 | naip_r30m |      3441 |     2661 |                6 |               0.084 |     0.395 |      0.367 | 0.409 |    0.043 |          23 |
| nps_vmi_group       |   0.23  | 0.00 - 0.90            |           1 | naip_r30m |      3441 |     2661 |                6 |               0.084 |     0.23  |      0.183 | 0.271 |    0.01  |          96 |
| vegbench_formation  |   0.412 | 0.22 - 0.54            |          20 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.412 |      0.388 | 0.446 |    0.091 |          11 |
| vegbench_formation  |   0.305 | 0.00 - 0.54            |           1 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.305 |      0.257 | 0.402 |    0.062 |          16 |
| vegbench_division   |   0.382 | 0.08 - 0.54            |          20 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.382 |      0.308 | 0.363 |    0.062 |          16 |
| vegbench_division   |   0.285 | 0.00 - 0.52            |           1 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.285 |      0.222 | 0.326 |    0.048 |          21 |
| vegbench_macrogroup |   0.33  | 0.07 - 0.45            |          20 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.33  |      0.247 | 0.275 |    0.037 |          27 |
| vegbench_macrogroup |   0.239 | 0.00 - 0.48            |           1 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.239 |      0.174 | 0.23  |    0.026 |          38 |
| vegbench_group      |   0.237 | 0.00 - 0.54            |          20 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.237 |      0.212 | 0.252 |    0.024 |          41 |
| vegbench_group      |   0.192 | 0.00 - 0.44            |           1 | naip_r50m |      6163 |     3277 |                0 |               0.186 |     0.192 |      0.131 | 0.182 |    0.014 |          73 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling    |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:-----------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.863 |     0.197 | 0.59 - 0.98                 | 0.01 - 0.84              |          20 | naip_r150m |      6412 |     6421 |              101 |               0.227 |        0.863 |             0.037 |         163 |
| bbs_occurrence           |   0.86  |     0.118 | 0.06 - 1.00                 | 0.00 - 1.00              |           1 | naip_r150m |      6412 |     6421 |              101 |               0.227 |        0.86  |             0.018 |         353 |
| butterflies_occurrence   |   0.683 |     0.577 | 0.51 - 0.95                 | 0.22 - 0.94              |          20 | naip_r50m  |       136 |      129 |                0 |               0     |        0.683 |             0.408 |          51 |
| butterflies_occurrence   |   0.74  |     0.359 | 0.26 - 1.00                 | 0.01 - 1.00              |           1 | naip_r50m  |       136 |      129 |                0 |               0     |        0.74  |             0.136 |         215 |
| snapshot_usa_occurrence  |   0.752 |     0.287 | 0.38 - 0.98                 | 0.01 - 0.91              |          20 | naip_r100m |      1595 |     1783 |               31 |               0.396 |        0.752 |             0.128 |          37 |
| snapshot_usa_occurrence  |   0.741 |     0.115 | 0.20 - 1.00                 | 0.00 - 0.92              |           1 | naip_r100m |      1595 |     1783 |               31 |               0.396 |        0.741 |             0.035 |         150 |
| satbird_winter_occupancy |   0.915 |     0.483 | 0.64 - 1.00                 | 0.01 - 0.96              |          20 | naip_r300m |     34327 |     6187 |                0 |               1     |        0.915 |             0.109 |         467 |
| satbird_winter_occupancy |   0.912 |     0.391 | 0.50 - 1.00                 | 0.00 - 0.96              |           1 | naip_r300m |     34327 |     6187 |                0 |               1     |        0.912 |             0.087 |         587 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.458 | 0.15 - 0.66             |         nan | naip_r50m |      4648 |     4583 |                2 |               0.309 |     0.458 |           0.6   |          12 |
| ofo_trees |   0.249 | 0.11 - 0.39             |         nan | naip_r30m |       146 |       51 |                0 |               0.665 |     0.249 |           0.579 |           2 |
