# USGS 3DEP lidar product registry

Polygon registries of USGS 3DEP lidar products, used to match field datasets to airborne lidar (ALS) by location and year.

| file | features | contents |
|---|---|---|
| `usgs_3dep_resources_AWS.geojson` | 2,278 | products available as EPT on the public AWS bucket `s3://usgs-lidar-public` (streamable). **This is the registry the matching code reads** (`src/eagle_als/fetch.py`, `scripts/3dep/dataset.py`) |
| `usgs_3dep_resources_ALL.geojson` | 2,844 | every WESM lidar workunit with its staged-LAZ link (`lpc_link`). Not updated by the date work below |
| `leaf_on/` | | intermediate tables, logs and notes for collection dates and leaf-on status (see `leaf_on/README.md`) |

Code: `src/streaming/3dep/fetch_3dep_metadata.py` builds both registries from hobu's EPT boundary list plus the USGS WESM table. The scripts in `src/streaming/3dep/leaf_on/` then correct the dates and (in progress) add leaf-on status. **Re-running `fetch_3dep_metadata.py` overwrites the AWS registry, so re-run the `leaf_on` steps afterwards.**

## Key columns (AWS registry)

| column | meaning |
|---|---|
| `name` | EPT bucket folder name; the product id used everywhere (`product_name_AWS` in curated datasets) |
| `url` | `ept.json` URL |
| `count` | number of points |
| `workunit`, `project`, `ql`, `spec`, ... | WESM fields. **Only filled for the 1,033 products whose name equals a WESM workunit exactly**; null otherwise (this is not a data gap, just an unmatched name) |
| `collect_start`, `collect_end` | collection window, corrected and filled (see below) |
| `collection_year` | year of `collect_start` (name-parsed year for the 5 undated products) |
| `collection_year_prev` | the value before the 2026-10-08 correction |
| `collect_dates_source` | `wesm_exact_workunit`, `wesm_normalized_workunit`, `gpstime` or null |
| `collect_dates_check` | agreement with measured GPS time (table below) |

## Collection dates

Why they needed fixing: dates were merged from WESM on the exact folder name, which works for 1,033 of 2,278 products. For the rest, `collection_year` fell back to the last four characters of the name. Names such as `USGS_LPC_SD_NRCS_DAS_2017_LAS_2019` end in the **publication** year, so hundreds of products were off by up to 12 years, and year matching to field data was wrong for them.

How dates are now obtained (`src/streaming/3dep/leaf_on/`, run with `python -I`, each step resumable):

1. `01_fill_collect_dates.py` matches names to WESM after normalizing them (drops the `USGS_LPC_` prefix and `_LAS_<year>` suffix, ignores case and punctuation, tries workunit then project names; min start and max end over matching rows). 2,249 products get WESM dates.
2. `01b_gpstime_dates.py` measures acquisition dates from the point clouds. It downloads the EPT root node (`ept-data/0-0-0-0.laz`, tens of KB, a thinned sample spread over the whole project) and converts `GpsTime` (Adjusted Standard GPS, +1e9 s since 1980-01-06). Usable for 1,830 products; the other 434 store GPS week-seconds or empty times. The sample can slightly under-cover the true span.
3. `01c_apply_collect_dates.py` keeps the WESM dates, checks them against the measured median date (+-45 days), uses measured dates where WESM has none, and writes the registry.

| `collect_dates_check` | products | meaning |
|---|---|---|
| `confirmed_by_gpstime` | 1,755 | WESM window contains the measured date (99% of exact-name matches, 95% of normalized matches) |
| `gpstime_unavailable` | 443 | WESM dates only, no check possible |
| `conflict_wesm_vs_gpstime` | 51 | sources disagree; WESM kept, needs a look (`leaf_on/gpstime_dates.csv`). Several measured dates look like week-second artifacts rather than real conflicts |
| `gpstime_only` | 24 | no WESM match; dates are the 1st-99th percentile of measured GPS time |
| `no_date` | 5 | undated |

Result: `collection_year` changed for 789 of 2,278 products (`leaf_on/collection_year_changes.csv`). For the 611 products whose window spans more than one calendar year, `collection_year` is the start year, so choose your own rule if you need the year of the bulk of the flights.

## Curated datasets that must be re-matched

Curated datasets store `collection_year_AWS` / `year_diff_AWS` from the old registry. Rematching can change the chosen product, the year difference, `als_site_id` and which rows pass the year filter, so rebuild the tables and splits rather than patching the column. "Stale" = stored `collection_year_AWS` differs from the registry now.

| dataset | pipeline | stale rows |
|---|---|---|
| `butterflies` | `data_cleaning/subset_and_split_butterfly.ipynb` | 72,728 / 170,525 (43%) |
| `BBS` | `data_cleaning/subset_and_split_bbs.ipynb` | 3,432 / 13,440 (26%) |
| `NLCD_eval` | `scripts/nlcd_eval/01_build_table.py`, then `02_check_streaming.py` | 330 / 1,089 (30%); the filter is `year_diff == 0`, so membership changes |
| `vegbank` | `data_cleaning/vegbank_pairing.ipynb` | 1,459 / 6,597 (22%) |
| `LFRDB_eval` | `data_cleaning/lfrdb/03_filter_and_match_als.py` through `05_finalize_eval.py` | 697 / 6,742 (10%); re-split after rematching |
| `BLM_AIM` | `data_cleaning/blm_aim/blm_aim_explore_and_match.ipynb` | 468 / 9,161 (5%) |
| `OFO_trees` | `data_cleaning/merge_ofo_trees.ipynb` | 1 / 194; column refresh only |

Also `scripts/litePT/sample_pretrain_locations.py` (uses `collection_year` for `--min-year`): rerun if its output was used. `NPS_VMI` has no lidar matching and is unaffected.

## Leaf-on status (in progress, not yet in the registry)

Goal: a `leaf_on` boolean per product. Preference order: what the data provider documents, then an inference from phenology.

- `02_provider_leaf_text.py` scrapes USGS staged metadata (vendor XML and LPC reports on `prd-tnm.s3.amazonaws.com`) for "leaf-off" / "leaf-on" statements. Of 2,243 metadata links: 342 say leaf-off, 23 leaf-on, 1 conflicting. WESM itself has no leaf field. Output: `leaf_on/provider_evidence.csv` (with the quoted snippet).
- `03_modis_ndvi_samples.py` samples Terra MOD13Q1 16-day NDVI (Planetary Computer, anonymous) at 8 random points per product for the collection year(s). MCD12Q2 phenology is not on Planetary Computer and needs an Earthdata login.
- `04_merge_leaf_on.py` (not yet run on the registry) will add `leaf_on`, `leaf_on_source` (`provider_metadata` or `modis_ndvi_phenology`), `leaf_on_frac` (share of collection days with leaf-on canopy), `leaf_on_provider` and `leaf_on_note`. Rule: a day is leaf-on when NDVI is at least halfway between that year's minimum and maximum; flat curves count as leaf-on if green (NDVI >= 0.3). It prints the agreement between this rule and the provider labels.

Known limits of the phenology inference: points are random within the footprint, not restricted to forest; evergreen conifers under snow can read as leaf-off; large footprints mix phenologies, so check `leaf_on_frac` rather than only the boolean.

See `leaf_on/README.md` for the same material in step-by-step form.
