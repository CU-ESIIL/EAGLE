# NLCD land-cover evaluation task for lidar representations

A small CONUS classification benchmark for checking that representation training is progressing (e.g. a linear probe on frozen embeddings). Each row is a 30 m pixel with an analyst-interpreted land-cover label and the 3DEP lidar product closest in time to that label.

Labels are **reference labels, not the modeled NLCD map**: the NLCD 2021 accuracy-assessment points (Wickham et al. 2026, https://doi.org/10.5066/P9JZ7AO3), 3,245 CONUS pixels interpreted by trained analysts who were blind to the map labels, using high-resolution imagery. Each point has a primary and an alternate label for 2016, 2019 and 2021. Raw files are in `datasets/raw/NLCD_AA2021/` (gitignored). Code: `scripts/nlcd_eval/`.

## Files

| file | contents |
|---|---|
| `nlcd_lidar_eval.parquet` | one row per point (1,089 rows), each with a label year equal to the lidar collection year |
| `nlcd_lidar_eval_als_status.csv` | streaming check for 1,006 rows (all of `balanced_subset` plus a few others): `ok` (956), `empty` (49), `error` (1: transient S3 failure); `n_points` is the count within 50 m; no point clouds are saved |

## Columns

| column | meaning |
|---|---|
| `point_id` | id of the reference pixel (`FID` in the source) |
| `nlcd_code`, `nlcd_class` | **classification target**: primary Level II label (16 classes; perennial ice/snow has only 2 points) |
| `nlcd_L1_code`, `nlcd_L1_class` | coarser Level I grouping (8 classes), derived from the Level II code |
| `alt_nlcd_code`, `alt_nlcd_class` | alternate label the analysts gave where the class is ambiguous (about half of points differ from the primary). Score with either label if you want a lenient metric |
| `lat`, `lon` | WGS84 center of the 30 m pixel |
| `year` | year the label applies to: the date of the imagery used by the analyst when it is within 2 years of the map epoch, otherwise the epoch year |
| `label_year` | map epoch the label comes from (2016, 2019 or 2021) |
| `label_image_year` | year of the imagery the analyst interpreted, where recorded |
| `label_conf` | analyst confidence flag from the source (`Conf`) |
| `stable_2016_2021` | the primary label is the same in all three epochs; if true the label is safe even when the lidar year is not exactly `year` |
| `label_source` | `NLCD2021_AA_reference` for every row |
| `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS` | same meaning as in the other datasets: the 3DEP project containing the point whose collection year is closest to `year`, its year, and the absolute gap in years (always 0 here) |
| `als_site_id` | `<product>__<lat>_<lon>`, as built by `eagle_als.sites.add_site_ids` |
| `test_split` | `True` for about 25% of rows, assigned by whole 1-degree lat/lon blocks (spatial hold-out) |
| `balanced_subset` | `True` for the class-balanced evaluation subset (see below) |

## Class-balanced subset

`balanced_subset` has up to 100 points per Level II class (988 rows in total): stable labels first, at most 10 points per 3DEP project and class. Most classes have fewer than 100 exact-year points and keep all of them; only open-space developed, evergreen forest, shrub/scrub, grassland and cultivated crops are capped at 100. Small classes: perennial ice/snow (1), high-intensity developed (27), low-intensity developed (39), barren (35). The full table is there for anyone who prefers a different sample.

## Choices and caveats

- Only 3,245 reference points exist, so 100 per class is the ceiling and rare classes fall short; this task is small by design.
- Rows are limited to an exact match between `year` and the lidar collection year (`year_diff_AWS == 0`), which leaves 1,089 of 3,245 reference points. If a larger set is needed, set `MAX_YEAR_DIFF = 1` in `01_build_table.py` (adds about 1,300 rows). Land cover can still change within a year, so `stable_2016_2021` (80% of rows) marks the cleanest labels.
- Matching is by 3DEP footprint only. Footprints can include gaps (water, edges), so use the status file, or regenerate it on cluster compute, before relying on a row.
- The reference pixel is 30 m. Evaluation crops can be as small as that or include context (the status check uses a 50 m radius). The analysts used surrounding context to assign labels, so context helps.
- Reference labels are accurate but not perfect (the source reports about 73% Level II agreement between map and primary reference label, so many classes are genuinely ambiguous, e.g. pasture vs grassland).
- Interpretation of 2019 and 2021 used Landsat instead of high-resolution imagery for about 13% and 19% of pixels.

## Regenerating

```
python scripts/nlcd_eval/01_build_table.py                      # builds the parquet
python scripts/nlcd_eval/02_check_streaming.py --max-seconds 600  # optional, resumable, appends to the status csv
```
