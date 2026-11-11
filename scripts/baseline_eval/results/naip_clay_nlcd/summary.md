# naip_clay_nlcd: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year. `score` is the macro average over classes / species / targets; the range columns give the lowest - highest per-class (F1), per-species (AUROC, AP) or per-target (R2) value.

## multiclass (score = bal_acc)

| task   |   score | f1 range (per class)   |   min_count | pooling   |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:-------|--------:|:-----------------------|------------:|:----------|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd   |   0.636 | 0.46 - 0.81            |          20 | naip_r15m |       443 |      449 |                0 |               0.094 |     0.636 |      0.633 | 0.647 |    0.125 |           8 |
| nlcd   |   0.499 | 0.19 - 0.80            |           1 | naip_r15m |       443 |      449 |                0 |               0.094 |     0.499 |      0.477 | 0.52  |    0.067 |          15 |
