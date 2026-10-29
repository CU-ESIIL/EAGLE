# VegBench: matching California vegetation surveys to 3DEP lidar

`01_match_als.py` reads the VegBench benchmark CSV (`datasets/raw/vegbench/benchmark_splits/VegBench/VegBench.csv`, 42,393 expert vegetation surveys from CDFW DS1020, 1978-2023, NVC labels from formation class to association) and matches each survey to the 3DEP lidar product closest in time (`get_nearest_year_product`, source AWS). The benchmark's own `split` / `test_type` columns (train, val, test/strat, test/acc) are kept unchanged. Run from the repo root: `pixi run python data_cleaning/vegbench/01_match_als.py` (about 30 s).

## Outputs

- `datasets/raw/vegbench/derived/vegbench_als_covered.parquet` (gitignored): every survey covered by a 3DEP footprint (37,380), with `year_diff_AWS`, so any maximum gap N can be applied by filtering.
- `datasets/VegBench_eval/vegbench_eval.parquet`: the rows with `year_diff_AWS <= 3` (10,455). Columns: survey id, coordinates, year, split, the NVC labels at each level, CalVeg/CWHR/WHR13 crosswalks, and the lidar product (`product_name_AWS`, `collection_year_AWS`, `year_diff_AWS`, `signed_year_diff` = lidar year minus survey year, `als_site_id`).
- `catalog/year_gap_counts.csv`, `year_gap_counts_by_split.csv`, `year_gap_distribution.csv`: surveys kept for N = 0..5.

## Surveys kept by maximum year gap N

| N | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| all splits | 2,477 | 6,721 | 8,723 | **10,455** | 11,687 | 13,462 |
| train | 1,611 | 4,292 | 5,259 | 6,208 | 6,979 | 7,927 |
| val | 332 | 738 | 793 | 937 | 1,083 | 1,313 |
| test/strat | 302 | 611 | 768 | 883 | 966 | 1,240 |
| test/acc | 232 | 1,080 | 1,903 | 2,427 | 2,659 | 2,982 |

37,380 of 42,393 surveys are covered by a footprint, but most surveys predate the lidar (median gap 8 years among covered).

## Caveats

- At N = 3, 297 alliances remain (387 in the full benchmark); 13 alliances in the test splits are absent from train, and 71 associations. Association is labeled for only 6,772 of 10,455 rows.
- At N = 3 the lidar was collected before the survey in 44% of rows, after it in 32%, and in the same year in 24%, so vegetation may have changed in either direction.
- 43 kept surveys share an `als_site_id` with another survey (same product and coordinates).
- `acc` surveys are accuracy-assessment points whose split guarantees differ from the others; report the two test sets separately (see the VegBench data card).

## Label turnover at repeat-survey locations

`02_label_turnover.py` measures how often a revisited location keeps the same NVC label, per taxonomy level and years between visits. A revisit is two surveys within 5 m (EPSG:3310) and at least 1 year apart; it uses the whole benchmark. Outputs: `catalog/turnover_*.csv` and `turnover_summary.txt`, and `datasets/VegBench_eval/figures/01_label_turnover.png`.

549 pairs (747 surveys at 306 sites, 21 projects; 95% from the same project; median gap 4.9 years). Fraction with the same label:

| level | formation class | formation | macrogroup | group | alliance | association |
|---|---|---|---|---|---|---|
| all pairs | 0.97 | 0.86 | 0.85 | 0.77 | 0.73 | 0.65 |
| gap 1 yr | 0.97 | 0.92 | 0.92 | 0.86 | 0.85 | 0.76 |
| gap 11-20 yr | 1.00 | 0.80 | 0.76 | 0.61 | 0.53 | 0.32 |

- Coarse levels barely change; agreement falls at each finer level. Of the 149 pairs whose alliance changed, 86% also changed group, 56% macrogroup and 9% formation class.
- Alliance change rises with the log of the gap (slope 0.36, p = 0.02, errors clustered by site), but the bins are noisy and project mix differs between them (SUMA is 35 of 49 pairs at 11-20 years). Intervals come from resampling sites.
- Already at a 1-year gap 15% of alliances (24% of associations) differ. That mostly reflects different crews, protocols and boundary cases, not vegetation change, and is a rough ceiling on how well any model can match a second survey's label.
- Only 29 pairs are across projects, so project effects cannot be separated. At 1 m and 10 m radius the overall agreement is about the same (`catalog/turnover_by_radius.csv`).

## Samples per class, level and split

`03_class_counts.py` counts rows per class for each taxonomy level and split of the eval table (lidar within 3 years): `catalog/class_counts_by_split.csv` (also the full-benchmark counts), `catalog/class_counts_summary.csv`, and figures `02_class_counts_rank.png` (rank-abundance per level and split) and `03_class_counts_coarse.png` (per-class stacked counts, six coarsest levels) in `datasets/VegBench_eval/figures/`.

Classes present in each split (train 6,208 rows, val 937, test/strat 883, test/acc 2,427), and how many have at least 20 rows in train and in the two test sets combined ("loose") or in each test set ("strict"):

| level | classes (full benchmark) | in eval table | train | val | test/strat | test/acc | in test, not in train | loose >= 20 | strict >= 20 |
|---|---|---|---|---|---|---|---|---|---|
| formation class | 6 | 6 | 6 | 5 | 5 | 5 | 0 | 3 | 3 |
| formation subclass | 10 | 10 | 9 | 9 | 8 | 7 | 0 | 5 | 4 |
| formation | 17 | 17 | 16 | 16 | 15 | 14 | 0 | 11 | 9 |
| division | 22 | 22 | 21 | 21 | 20 | 19 | 0 | 16 | 12 |
| macrogroup | 41 | 40 | 39 | 37 | 34 | 35 | 0 | 27 | 14 |
| group | 86 | 82 | 80 | 67 | 65 | 70 | 1 | 41 | 7 |
| alliance | 387 | 297 | 282 | 180 | 169 | 185 | 13 | 33 | 2 |
| association (6,772 rows labeled) | 1,598 | 831 | 728 | 304 | 265 | 187 | 71 | 0 | 0 |

- Everything down to macrogroup is usable with a reasonable number of classes; from group down, the long tail leaves few classes with enough test rows. Alliance, the benchmark's primary target, has 33 classes with at least 20 train rows and 20 test rows (4,296 of 10,455 rows; the other 264 classes are scored only through a long-tail metric or not at all).
- Test/acc is much larger than test/strat (2,427 vs 883 rows) and comes from only 10 projects (DOLO, OC and SSNF are 65% of it; test/strat has 31 projects, train 65); the two test sets should be reported separately.
- Association is labeled for 65% of rows and no association reaches 20 rows in a test set.
