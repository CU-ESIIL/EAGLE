# Collection dates and leaf-on status for the AWS 3DEP registry

Supports `datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson`. Scripts: `src/streaming/3dep/leaf_on/`
(run in order with `python -I`; each step is resumable).

## Collection dates (done)

How dates are obtained, most to least trustworthy:

1. **Measured**: `GpsTime` of the EPT root node (`01b_gpstime_dates.py`), valid for 1,830 / 2,278 products.
   The other 434 store GPS week-seconds or empty times and cannot be dated this way.
2. **WESM** `collect_start` / `collect_end` (the USGS workunit table). The old registry only matched WESM on the
   exact EPT folder name (1,033 products). `01_fill_collect_dates.py` also matches normalized workunit / project names
   (strips `USGS_LPC_` and `_LAS_<publication year>`, ignores case and punctuation), reaching 2,249 products.
3. Reconciliation (`01c_apply_collect_dates.py`): WESM dates are kept, checked against the measured median GPS date
   (+-45 d). Result in `collect_dates_check`:

| collect_dates_check | products | meaning |
|---|---|---|
| confirmed_by_gpstime | 1,755 | WESM window contains the measured date (99% of exact-name matches, 95% of normalized matches) |
| gpstime_unavailable | 443 | WESM dates only |
| conflict_wesm_vs_gpstime | 51 | disagree; WESM dates kept, inspect (`gpstime_dates.csv`) |
| gpstime_only | 24 | no WESM match; dates are the 1st-99th percentile of measured GpsTime |
| no_date | 5 | still undated, `collection_year` unchanged |

Registry columns changed: `collect_start`, `collect_end`, `collection_year` (year of `collect_start`; 789 products
changed, mostly publication year -> collection year, up to 12 years). Added: `collection_year_prev` (the old value),
`collect_dates_source`, `collect_dates_check`. Full change list: `collection_year_changes.csv`.
Caveat: for the 611 products whose window spans more than one calendar year, `collection_year` is the start year.

## Curated datasets to re-run (matching used the old `collection_year`)

Rows listed are those whose stored `collection_year_AWS` no longer equals the registry value.

| dataset | pipeline | stale rows | notes |
|---|---|---|---|
| butterflies | `data_cleaning/subset_and_split_butterfly.ipynb` | 72,728 / 170,525 (43%) | 36 of 64 products |
| BBS | `data_cleaning/subset_and_split_bbs.ipynb` | 3,432 / 13,440 (26%) | 324 products; `BBS_train_test*.parquet` |
| NLCD eval | `scripts/nlcd_eval/01_build_table.py`, `02_check_streaming.py` | 330 / 1,089 (30%) | filter is `year_diff == 0`, so membership changes; 194 products |
| vegbank | `data_cleaning/vegbank_pairing.ipynb` | 1,459 / 6,597 (22%) | 126 products |
| LFRDB eval | `data_cleaning/lfrdb/03_filter_and_match_als.py` then 04, 05 | 697 / 6,742 (10%) | 19 products; re-split after rematching |
| BLM AIM | `data_cleaning/blm_aim/blm_aim_explore_and_match.ipynb` | 468 / 9,161 (5%) | 240 products |
| OFO trees | `data_cleaning/merge_ofo_trees.ipynb` | 1 / 194 | refresh only, no real change |
| litePT pretrain sampling | `scripts/litePT/sample_pretrain_locations.py` | n/a | uses `collection_year` for `--min-year`; rerun if its output was used |

Not affected: NPS VMI (no ALS matching). Rematching can change the chosen product, `year_diff_AWS`, `als_site_id`
and the rows kept, so downstream splits and evaluation tables should be rebuilt, not patched.

## Leaf-on status

Registry columns: `leaf_on` (boolean, null when not determinable) and `leaf_on_source` (`collection_season` or null).

Built by `src/streaming/3dep/leaf_on/05_leaf_on_from_dates.py` from the collection dates alone (CONUS rule): `True` if every month touched by `collect_start`..`collect_end` is in June-September, `False` if every month is in December-March, null otherwise (spans other months, shoulder season, or undated). Result: 201 True, 541 False, 1,536 null.

An earlier approach using provider metadata and MODIS NDVI phenology was dropped: NDVI also tracks evergreen conifers (Maine in December reads ~0.6), provider statements are often templated requirements, and a binary label does not fit multi-month or shoulder-season collections. Its columns and scripts were removed; the intermediate files `leaf_on/provider_evidence.csv` and `leaf_on/modis_ndvi_samples.parquet` remain only as unused leftovers.
