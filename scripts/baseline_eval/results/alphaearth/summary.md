# alphaearth: linear probe results

Test rows only (the datasets' spatial hold-out). `frac clipped` = share of rows whose survey year is outside the imagery years and uses the nearest available year.

## multiclass

| task   | pooling   |   min_count |   n_train |   n_test |   n_no_embedding |   frac_year_clipped |   bal_acc |   macro_f1 |   acc |   chance |   n_classes |
|:-------|:----------|------------:|----------:|---------:|-----------------:|--------------------:|----------:|-----------:|------:|---------:|------------:|
| nlcd   | disc      |          20 |       443 |      449 |                0 |               0.187 |     0.703 |      0.705 | 0.712 |    0.125 |           8 |
| nlcd   | disc      |           1 |       443 |      449 |                0 |               0.187 |     0.555 |      0.507 | 0.549 |    0.067 |          15 |
| nlcd   | center    |          20 |       443 |      449 |                0 |               0.187 |     0.756 |      0.747 | 0.752 |    0.125 |           8 |
| nlcd   | center    |           1 |       443 |      449 |                0 |               0.187 |     0.578 |      0.518 | 0.556 |    0.067 |          15 |
