# LANDFIRE Reference Database (LFRDB) habitat-type evaluation task for lidar representations

A habitat-type classification (and cover regression) benchmark for checking that representation training is progressing, e.g. a linear probe on frozen embeddings. Each row is a field plot with a **precise GPS location**, a **LANDFIRE Ecological System label assigned from the plot's field data**, and the 3DEP lidar product closest in time to the plot visit (within 3 years).

The labels are not the modeled LANDFIRE EVT raster. They come from the public LANDFIRE Reference Database (LFRDB, LF 2.0.0 / LF 2016 Remap), a compilation of field plots from agencies and partners (here mostly NPS and state vegetation surveys). LANDFIRE assigned each plot's Ecological System either with **AutoKey**, an automated program that assigns the type from species cover data (3,813 rows), or by **Crosswalk**, experts mapping the source dataset's own vegetation type to an Ecological System (2,929 rows); see `evt_method`. Raw data are in `datasets/raw/LFRDB/` (gitignored); how and when they were obtained, and the cleaning code, are in `data_cleaning/lfrdb/` ([README](../../data_cleaning/lfrdb/README.md)).

## Files

| file | contents |
|---|---|
| `lfrdb_eval.parquet` | one row per plot (6,742 rows, 222 Ecological Systems) |
| `ecosys_split_counts.csv` | plots per Ecological System in train and test: use it to choose which classes to score |
| `figures/` | `01_label_counts.png` community type counts, `02_structure_variables.png` cover/height distributions and missingness, `03_als_match.png` years and regions, `04_map.png` and `05_split_map.png` maps, `06_final_counts_by_split.png`. Figures 01-04 are drawn **before** the missing-label filter, 05-06 after |

## Columns

| column | meaning |
|---|---|
| `plot_id` | LFRDB `EventID`; unique per plot visit |
| `lat`, `lon` | WGS84 plot location (decimal degrees, at least 5 decimals) |
| `year`, `month` | visit date (`month` is null for 0.3% of rows) |
| `ecosys_code`, `ecosys` | **primary classification target**: LANDFIRE/NatureServe Ecological System (222 classes, long-tailed). Never null and never an `Unclassified ...` class |
| `ecosys_lifeform` | lifeform of that Ecological System (`Tree` 3,509, `Shrub` 1,666, `Herb` 1,216, `Sparse` 291, `N/A` 60): a coarse 5-class variant of the task |
| `nvc_group_code`, `nvc_group` | National Vegetation Classification group (198 classes), a coarser alternative; null for a few rows |
| `tree_cover_pct`, `shrub_cover_pct`, `herb_cover_pct` | **regression targets**: percent cover by lifeform, adjusted for overlap (0-100), from LANDFIRE-processed species cover. Null for about 40% of rows (no species cover recorded) |
| `tree_height_m`, `shrub_height_m`, `herb_height_m` | cover-weighted lifeform height. **Almost always null** (175, 106 and 30 rows); not usable as a task, kept for completeness |
| `dominant_lifeform`, `dominant_species` | LANDFIRE-derived dominant lifeform and species from the plot's species list (null for 43% of rows) |
| `evt_method` | how the Ecological System was assigned: `AutoKey` or `Crosswalk` (see above) |
| `source_id`, `source_agency`, `lfrdb_region` | contributing dataset id (see `lutdtVisitsSourceID` in the raw databases), its agency (NPS 4,581, State 1,893, Multipartner 142, USFS 95, USFWS 31) and the LANDFIRE region database |
| `loc_method`, `n_decimals` | location method (`G` for every row) and the number of decimals in the coarser of lat/lon (5 or 6) |
| `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS` | same meaning as in the other datasets: the 3DEP project covering the plot whose collection year is closest to `year`, its collection year, and the absolute gap in years (0 to 3) |
| `als_site_id` | `<product>__<lat>_<lon>`, as built by `eagle_als.sites.add_site_ids` |
| `test_split` | `True` for 50% of rows (3,379), assigned by whole 1-degree lat/lon blocks (spatial hold-out; see caveats) |

## How the table was built

Counts after each step (also in `data_cleaning/lfrdb/catalog/filter_counts.csv`):

| step | plots left |
|---|---|
| all plots in the nine public regional databases | 539,373 |
| has valid coordinates | 533,043 |
| `LocMeth = G` (location captured with GPS in the field) | 259,615 |
| at least 5 decimals in both lat and lon (about 1 m) | 251,959 |
| visit type is `field visit` (drops aerial and helicopter surveys, photo interpretation, remote) | 217,672 |
| visit year 1970-2025 | 215,098 |
| covered by a 3DEP footprint with a collection year | 175,279 |
| 3DEP collection year within +/-3 years of the visit | 7,783 |
| `ecosys` not missing | 7,538 |
| `ecosys` is not an `Unclassified ...` fallback | **6,742** |

The coordinate filters need both conditions. The `G` flag alone is not enough: the 6,266 plots from sources marked "YES, plot locations excluded" in the LFRDB source table have 3 or fewer decimals in every case (about 100 m or worse) even though 95% of them are flagged GPS. The data dictionary describes Lat/Long as "to the nearest 100 seconds", which appears to be a typo for 1/100 second: about 97% of all plots have at least 5 decimals.

The ALS match uses `get_nearest_year_product` from `src/streaming/3dep/get_als.py` (AWS 3DEP footprints). The +/-3 year filter is what removes most plots: LFRDB visits have a median year of about 2003 and most 3DEP projects are newer.

## Recommended use

- **Score a subset of classes.** Only 18 of the 222 Ecological Systems have at least 20 plots in both train and test, because the long tail and the spatial split put most classes on one side only (137 of 222 appear in only one split). Pick classes from `ecosys_split_counts.csv` (for example `n_train >= 20 and n_test >= 20`; those 18 classes hold 2,779 plots) or use the coarser labels: `ecosys_lifeform` has 4 classes with at least 20 plots in both splits, `nvc_group` has 21.
- Report macro-F1 or balanced accuracy for classification and R^2 for the cover columns (about 4,000 rows each).
- The label is assigned by LANDFIRE from field data, but it is still a derived label (a species-cover key or an expert crosswalk), so some classes are ambiguous, particularly the `Ruderal` ones.

## Choices and caveats

- **Small and clustered by design.** 68% of rows are NPS plots and 4,039 of the 7,783 matched rows (before the label filter) are in the Southwest; coverage is mostly parks and a few California and southeastern surveys. Rows within a source are often close together and share a lidar product (50 products in total).
- **Spatial split lumpiness.** Updated from 1-degree to 0.1 degree blocks for spatial holdout. Some classes land entirely in test. 24 classes in both splits.
- **Time gap.** Plots were visited 0 to 3 years from the lidar (1,013 exact matches, 1,560, 2,022 and 2,147 at 1, 2 and 3 years). Vegetation can change in that time, especially after fire, logging or conversion; use `year_diff_AWS == 0` for the strictest evaluation.
- **Plot geometry is unknown.** The public LFRDB gives a point but no plot size, so evaluation crops cannot be matched to a plot footprint. Plot positions are GPS positions of unknown accuracy (decimals only bound the rounding, not the GPS error).
- **Footprint matching only.** 3DEP footprints can include gaps (water, edges); no point-cloud availability check was run for this dataset. The `eagle_als` streaming check used for `NLCD_eval` can be run on `plot_id`/`lat`/`lon`/`als_site_id` if needed.
- **Dropped label classes.** `Unclassified ...` Ecological Systems (796 matched plots, e.g. "Unclassified Shrubland") are lifeform fallback buckets, not habitat types, so they were treated as missing. Rows with null cover keep their label and can be used for classification only.
- **Height columns** contain values within sanity bounds only (tree up to 120 m, shrub up to 20 m, herb up to 10 m); none of the matched rows exceeded them.
- Only the public LFRDB is used. It excludes proprietary and sensitive plots, and for some sources the species cover or GIS information was removed.

## Regenerating

```
brew install mdbtools                                  # for stage 2 only
pixi run python data_cleaning/lfrdb/01_download.py     # raw .accdb files + provenance log
pixi run python data_cleaning/lfrdb/02_extract_tables.py
pixi run python data_cleaning/lfrdb/03_filter_and_match_als.py
pixi run python data_cleaning/lfrdb/04_summarize_matched.py
pixi run python data_cleaning/lfrdb/05_finalize_eval.py
```
