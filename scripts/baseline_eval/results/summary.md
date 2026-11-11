# Baseline results: score per task and feature type

Test rows of the spatial hold-out. `score` names the metric (bal_acc = balanced accuracy, mean_auroc, mean_r2); `min_count` is the class-set scenario (20 = classes with >= 20 rows in train and test, 1 = all). Blank = not run for that feature type. Per-model detail: `results/<model>/summary.md`.

| task                     |   min_count (0 = n/a) | score      |   alphaearth_center |   location_rff |
|:-------------------------|----------------------:|:-----------|--------------------:|---------------:|
| nlcd                     |                    20 | bal_acc    |               0.751 |          0.319 |
| nlcd                     |                     1 | bal_acc    |               0.577 |          0.219 |
| lfrdb_ecosys             |                    20 | bal_acc    |               0.564 |          0.292 |
| lfrdb_ecosys             |                     1 | bal_acc    |               0.237 |          0.093 |
| lfrdb_nvc_group          |                    20 | bal_acc    |               0.445 |          0.135 |
| lfrdb_nvc_group          |                     1 | bal_acc    |               0.214 |          0.097 |
| nps_vmi_macrogroup       |                    20 | bal_acc    |               0.483 |          0.267 |
| nps_vmi_macrogroup       |                     1 | bal_acc    |               0.357 |          0.139 |
| nps_vmi_group            |                    20 | bal_acc    |               0.566 |          0.265 |
| nps_vmi_group            |                     1 | bal_acc    |               0.331 |          0.095 |
| vegbench_formation       |                    20 | bal_acc    |               0.517 |          0.227 |
| vegbench_formation       |                     1 | bal_acc    |               0.379 |          0.175 |
| vegbench_division        |                    20 | bal_acc    |               0.469 |          0.197 |
| vegbench_division        |                     1 | bal_acc    |               0.42  |          0.141 |
| vegbench_macrogroup      |                    20 | bal_acc    |               0.386 |          0.141 |
| vegbench_macrogroup      |                     1 | bal_acc    |               0.35  |          0.138 |
| vegbench_group           |                    20 | bal_acc    |               0.351 |          0.114 |
| vegbench_group           |                     1 | bal_acc    |               0.269 |          0.131 |
| bbs_occurrence           |                    20 | mean_auroc |               0.884 |          0.825 |
| bbs_occurrence           |                     1 | mean_auroc |               0.888 |          0.807 |
| butterflies_occurrence   |                    20 | mean_auroc |               0.662 |          0.713 |
| butterflies_occurrence   |                     1 | mean_auroc |               0.746 |          0.804 |
| snapshot_usa_occurrence  |                    20 | mean_auroc |               0.803 |          0.757 |
| snapshot_usa_occurrence  |                     1 | mean_auroc |               0.764 |          0.741 |
| satbird_winter_occupancy |                    20 | mean_auroc |               0.927 |          0.896 |
| satbird_winter_occupancy |                     1 | mean_auroc |               0.922 |          0.876 |
| blm_aim                  |                     0 | mean_r2    |               0.511 |          0.192 |
| ofo_trees                |                     0 | mean_r2    |               0.266 |         -0.004 |
