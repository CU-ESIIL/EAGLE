# forpt: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task                |   score | f1 range (per class)   |   min_count | pooling     |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:--------------------|--------:|:-----------------------|------------:|:------------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd                |   0.415 | 0.21 - 0.73            |          20 | cookie_r15m |       414 |      421 |               57 |               0.187 |     0.415 |      0.399 | 0.417 |    0.125 |           8 |
| nlcd                |   0.266 | 0.00 - 0.58            |           1 | cookie_r15m |       414 |      421 |               57 |               0.187 |     0.266 |      0.214 | 0.26  |    0.067 |          15 |
| lfrdb_ecosys        |   0.381 | 0.08 - 0.73            |          20 | cookie_r50m |      3752 |     3558 |              717 |               0.996 |     0.381 |      0.367 | 0.443 |    0.05  |          20 |
| lfrdb_ecosys        |   0.106 | 0.00 - 0.67            |           1 | cookie_r50m |      3752 |     3558 |              717 |               0.996 |     0.106 |      0.078 | 0.127 |    0.009 |         109 |
| lfrdb_nvc_group     |   0.289 | 0.07 - 0.51            |          20 | cookie_r50m |      3752 |     3557 |              717 |               0.996 |     0.289 |      0.268 | 0.312 |    0.04  |          25 |
| lfrdb_nvc_group     |   0.105 | 0.00 - 0.67            |           1 | cookie_r50m |      3752 |     3557 |              717 |               0.996 |     0.105 |      0.08  | 0.126 |    0.009 |         111 |
| nps_vmi_macrogroup  |   0.297 | 0.00 - 0.64            |          20 | cookie_r30m |      3334 |     2425 |              349 |               0.624 |     0.297 |      0.284 | 0.349 |    0.045 |          22 |
| nps_vmi_macrogroup  |   0.21  | 0.00 - 0.60            |           1 | cookie_r30m |      3334 |     2425 |              349 |               0.624 |     0.21  |      0.162 | 0.266 |    0.018 |          55 |
| nps_vmi_group       |   0.366 | 0.00 - 0.83            |          20 | cookie_r30m |      3334 |     2425 |              349 |               0.624 |     0.366 |      0.325 | 0.365 |    0.048 |          21 |
| nps_vmi_group       |   0.165 | 0.00 - 0.60            |           1 | cookie_r30m |      3334 |     2425 |              349 |               0.624 |     0.165 |      0.12  | 0.193 |    0.011 |          94 |
| vegbench_formation  |   0.373 | 0.18 - 0.64            |          20 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.373 |      0.344 | 0.434 |    0.091 |          11 |
| vegbench_formation  |   0.209 | 0.00 - 0.58            |           1 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.209 |      0.191 | 0.324 |    0.062 |          16 |
| vegbench_division   |   0.353 | 0.07 - 0.63            |          20 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.353 |      0.3   | 0.371 |    0.067 |          15 |
| vegbench_division   |   0.257 | 0.00 - 0.58            |           1 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.257 |      0.211 | 0.326 |    0.048 |          21 |
| vegbench_macrogroup |   0.258 | 0.00 - 0.53            |          20 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.258 |      0.227 | 0.298 |    0.04  |          25 |
| vegbench_macrogroup |   0.191 | 0.00 - 0.53            |           1 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.191 |      0.152 | 0.239 |    0.026 |          38 |
| vegbench_group      |   0.229 | 0.00 - 0.66            |          20 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.229 |      0.203 | 0.272 |    0.027 |          37 |
| vegbench_group      |   0.131 | 0.00 - 0.65            |           1 | cookie_r50m |      5843 |     3106 |              491 |               0.472 |     0.131 |      0.125 | 0.218 |    0.014 |          72 |

## multilabel (score = mean_auroc)

| task                     |   score |   mean_ap | auroc range (per species)   | ap range (per species)   |   min_count | pooling      |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_auroc |   mean_prevalence |   n_species |
|:-------------------------|--------:|----------:|:----------------------------|:-------------------------|------------:|:-------------|----------:|---------:|-----------------:|--------------------:|-------------:|------------------:|------------:|
| bbs_occurrence           |   0.718 |     0.092 | 0.49 - 0.94                 | 0.01 - 0.66              |          20 | cookie_r100m |      6450 |     6484 |                0 |               0.215 |        0.718 |             0.036 |         164 |
| bbs_occurrence           |   0.689 |     0.049 | 0.02 - 0.99                 | 0.00 - 0.66              |           1 | cookie_r100m |      6450 |     6484 |                0 |               0.215 |        0.689 |             0.017 |         358 |
| butterflies_occurrence   |   0.57  |     0.478 | 0.39 - 0.79                 | 0.16 - 0.89              |          20 | cookie_r50m  |       135 |      129 |                1 |               0.343 |        0.57  |             0.413 |          50 |
| butterflies_occurrence   |   0.54  |     0.194 | 0.00 - 1.00                 | 0.01 - 1.00              |           1 | cookie_r50m  |       135 |      129 |                1 |               0.343 |        0.54  |             0.136 |         214 |
| snapshot_usa_occurrence  |   0.68  |     0.213 | 0.48 - 0.92                 | 0.03 - 0.91              |          20 | cookie_r100m |      1568 |     1721 |              120 |               0     |        0.68  |             0.127 |          37 |
| snapshot_usa_occurrence  |   0.673 |     0.085 | 0.00 - 1.00                 | 0.00 - 0.92              |           1 | cookie_r100m |      1568 |     1721 |              120 |               0     |        0.673 |             0.035 |         148 |
| satbird_winter_occupancy |   0.802 |     0.264 | 0.49 - 0.98                 | 0.01 - 0.88              |          20 | cookie_r100m |     32887 |     5883 |             1744 |               0     |        0.802 |             0.109 |         465 |
| satbird_winter_occupancy |   0.791 |     0.207 | 0.20 - 1.00                 | 0.00 - 0.88              |           1 | cookie_r100m |     32887 |     5883 |             1744 |               0     |        0.791 |             0.087 |         584 |

## regression (score = mean_r2)

| task      |   score | r2 range (per target)   |   min_count | pooling     |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   mean_r2 |   mean_spearman |   n_targets |
|:----------|--------:|:------------------------|------------:|:------------|----------:|---------:|-----------------:|--------------------:|----------:|----------------:|------------:|
| blm_aim   |   0.246 | 0.02 - 0.71             |         nan | cookie_r50m |      4495 |     4438 |              300 |               0.101 |     0.246 |           0.378 |          12 |
| ofo_trees |   0.267 | 0.15 - 0.39             |         nan | cookie_r30m |       142 |       51 |                4 |               0.015 |     0.267 |           0.58  |           2 |
