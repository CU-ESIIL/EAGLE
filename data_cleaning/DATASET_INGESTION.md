# Ingesting a new evaluation dataset

How a labeled dataset becomes an evaluation task for the lidar encoder: a table of locations with labels, each tied to the 3DEP lidar product closest in time. Worked examples to copy from: `lfrdb/` (database download, many filters), `snapshot_usa/` (counts to per-location targets), `satbird/` (labels and split from a Hugging Face repackaging of the original dataset; encounter-rate and occupancy targets), `vegbench/` (ready-made split, repeat-survey analysis), `nps_vmi/` (many tables).

## Steps

1. **Get the raw data into `datasets/raw/<name>/`** (gitignored; large data stays out of GitHub). Keep files unmodified. Record where and when they came from: source URL or DOI, version, download date (UTC), and file hashes. Scripted downloads write `catalog/download_log.csv` and a `SOURCE.md` next to the raw files (see `lfrdb/01_download.py`); for a manual download, write the same facts in the dataset README.
2. **Read the dataset's own documentation, then profile the data.** Learn what one row is (plot, visit, deployment, route), how locations are recorded (precision, method, datum), what the date means, and what the labels are. Check duplicates, missing values, repeated locations and coordinates that are rounded or obviously wrong. Put findings that affect use in the README (or `DATASET_CONCERNS.md`, as in `nps_vmi/`).
3. **Harmonize to one table: one row per labeled location and visit.** Required columns: a unique id, `lat`, `lon` (WGS84 decimal degrees), `year` (survey year). Derived tables go in `datasets/raw/<name>/derived/`. Keep the dataset's own column meaning; rename only for consistency (`lat`, `lon`, `year`). **Coordinates: check the CRS, do not guess it.** Find the source CRS in the data (`.prj`, GeoPackage, file metadata) or the provider's documentation, never assume it for plain lat/lon columns without reading what the datum is. Convert with `eagle_als.crs.to_wgs84(x, y, src_crs)` (it has no default CRS), run `crs.check_lonlat` (range, region, swapped lat/lon, lost minus sign) and save its counts to `catalog/crs_check.csv`. If the CRS is unknown, `crs.candidate_crs` shows which candidates land inside the study region; confirm with the provider. Record the source CRS and how it was verified in the README and in `datasets/registry.yaml` (`source_crs`, `crs_status`). Output is always WGS84 lat/lon (EPSG:4326); NAD83 lat/lon is accepted as is (about 1 m difference, far below the 100 m cookie), say so.
4. **Apply location filters before the lidar lookup, and log counts after every filter.** Typical: valid coordinates, enough decimals for a 100 m-radius cookie (4-5 decimals), a precise location method, a plausible year, dataset-specific quality flags. Write the counts to `catalog/filter_counts*.csv` with a `dropped` column. Say why each rule exists.
5. **Match each location to 3DEP.** `eagle_als.match.match_3dep(df, "lat", "lon", "year")` does one 3DEP lookup per unique (lat, lon, year), treats an exception (footprints without a collection year) as no match, and adds `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS` (absolute gap), `signed_year_diff` (collection minus survey year) and `als_site_id` (`<product>__<lat>_<lon>`, the cookie's key in the cache). It also returns the year-gap table (step 6). The standard rule is `apply_year_gap(matched, 3)`, i.e. `year_diff_AWS <= 3`. If the labels pool many years (eBird encounter rates, long-term counts), there is no single survey year: accept any product collected inside the pooled period, take the latest one, set `year_diff_AWS` to the gap to that period (0 inside), and document the rule and its consequence in the README (`satbird/`).
6. **Report the year-gap trade-off before choosing N.** Write how many rows (per split, per class if relevant) survive N = 0..5 (`catalog/year_gap_counts*.csv`; `vegbench/01_match_als.py` does this). Keep the table of all covered rows with `year_diff_AWS`, so other N can be applied later, and export the N = 3 rows as the eval table.
7. **Define targets and splits.** Classification: one primary label column, plus coarser or finer variants if the taxonomy has levels. Regression: numeric columns with units in the name. Prefer the dataset's published split when it has one. Otherwise use `eagle_als.splits.spatial_split(df, lat, lon, group=...)`: whole 0.25-degree blocks are assigned to test by a hash of the block id and the seed, so a split depends only on where a row is, not on which other rows exist. That makes it **stable**: rebuilding a dataset with more or fewer rows keeps every surviving row on its side, and neighbouring datasets agree on shared blocks. Pass `group` (route, site) so repeat visits to one place stay together. Do not change the block size, test fraction (0.5) or seed (`splits.py`) per dataset; if the printed test share is far from 0.5 (few or very uneven blocks), say so in the README. Splits are frozen once published: never re-split a released table, and a published split (VegBench) is kept as is. Report per-class counts in each split (`*_split_counts.csv`) so users can choose which classes to score. For datasets with many classes or species (multi-label presence, encounter rate, cover), also follow the many-classes rule below.
8. **Summarize before and after the label filters** (`figures/`: label counts, structure variables and missingness, years and match, map, final counts by split), and check for labels that are suspiciously coarse, ambiguous, or inconsistent across visits to the same place (`vegbench/02_label_turnover.py` is an example of the last).
9. **Write the evaluation table and README** (format below).
10. **Register the dataset.** Add one entry to `datasets/registry.yaml` (table path, `lat`/`lon`/`year`/id columns, how the split is stored, source CRS and whether it was verified). That single entry makes pre-training exclude the table's sites (`squares.EVAL_SITES` reads the registry, so pre-training never samples within 500 m of it) and gives `eagle_bench.tasks` its paths and column names. Check it with `python scripts/validate_datasets.py <name>` (warnings only). Build the cookie cache with `python -m eagle_als.cache` (see `src/eagle_als/README.md`). To score it during training, add an `eval_tasks` entry (README at the repo root, "Validation tasks during training"); keep a task to about 2,000 cookies or fewer.
11. **Log it.** Append an entry to `PROMPT_ACTION_LOG.md`, and update the project working notes.

## Many classes or species: publish all of them

When the label is a long list of classes or species, publish **every** class that occurs in at least 1 row of each of train and test, once, in one folder. Do not publish separate "-20" folders: how rare a class may be and still be scored is an evaluation setting (`eagle_bench.probes.evaluate(min_count=...)`, `scripts/baseline_eval/02_linear_probe.py --min-counts 20 1`), so one run reports both "all classes" and ">= 20 rows in each of train and test" from the same table.

| target type | table |
|---|---|
| presence / occupancy (0/1) | `location_occupancy.parquet`, columns `occ__<class>` |
| rate or abundance (encounter rate, detection rate) | `location_encounter_rate.parquet` or `location_detection_rate.parquet`, columns `enc__<class>` / `rate__<class>` |

Both tables carry identical location columns, rows and split. Also write `location_species_summary.csv` (every class with its train/test counts) so users can see which classes pass any threshold. The counts must use the final split, after the lidar and year-gap filters. Examples: `snapshot_usa/` and `satbird/`.

## Layout

```text
data_cleaning/<name>/
  README.md              raw data provenance, run order, cleaning rules and why, results table
  01_*.py, 02_*.py ...   one numbered script per stage; each rerunnable, run from the repo root with
                         `pixi run python data_cleaning/<name>/<script>`; module docstring lists inputs,
                         filters in order, and outputs
  catalog/               small CSVs: download log, filter counts, year-gap counts, class counts
datasets/raw/<name>/     raw files + derived/   (gitignored)
datasets/<NAME>_eval/    the evaluation-ready dataset
  <name>_eval.parquet    the table (format below)
  README.md              what a row is, label sources and caveats, columns, how it was built (filter counts),
                         recommended use (which classes to score, metrics)
  *_split_counts.csv     rows per class and split
  figures/               numbered PNGs
```

## Target format of the evaluation table

One parquet file, one row per labeled location and visit, no row without a usable label for the primary target.

| column | type | meaning |
|---|---|---|
| id column (`plot_id`, `survey_point_id`, ...) | str | unique per row; the dataset's own id |
| `lat`, `lon` | float | WGS84 decimal degrees, at the dataset's precision (no rounding) |
| `year` | int | survey year (add `month` if known) |
| label columns | str / float | primary target first. Classification: string class names, optionally with `<level>_code` columns. Regression: numeric, units in the name (`tree_cover_pct`, `tree_height_m`). Missing = null, never a placeholder string |
| split column | bool or str | `test_split` (True = test), or `split` with `train`/`val`/`test` plus `test_type` where there are several test sets. Whole spatial blocks or the source's split; never random rows |
| `product_name_AWS` | str | name of the 3DEP project covering the location |
| `collection_year_AWS` | int | collection year of that project |
| `year_diff_AWS` | int | absolute gap in years between collection and survey (0..3 in the eval table) |
| `als_site_id` | str | `<product>__<lat>_<lon>` from `add_site_ids`; the cookie's id in the cache (`id_col`, default `als_site_id`) |

Also keep columns that let users stratify or audit (source project, survey method, source split id), but do not duplicate large free text. Place-level datasets (BBS routes, camera-trap locations) pool visits first, so that every label refers to the one lidar product in its row (`snapshot_usa/04_location_targets.py`).

How the table is consumed: a `classification` eval task reads `table`, `label_col`, optional `split_col` (`train`/`test` values or bool; omit for 5-fold CV) and optional `query` (a pandas row filter such as `balanced_subset`), joins to cached cookies on `id_col`, and drops classes with fewer than 5 cookies. Rows with a null label or id are dropped.

## Checklist

- [ ] raw data in `datasets/raw/<name>/` with source, version and download date recorded. when possible, document how/from where the data is obtained.
- [ ] every filter has a reason and a count in `catalog/`
- [ ] year-gap counts for N = 0..5 reported; N = 3 unless there is a reason
- [ ] `spatial_split` (or the source's published split) with per-class counts; the split is frozen once released
- [ ] for many-class labels: all classes in one table (and both target types where possible); no "-20" copies
- [ ] `datasets/<NAME>_eval/` has the parquet, a README with columns and caveats, and figures
- [ ] source CRS read from the data or documentation (not assumed), `catalog/crs_check.csv`, output WGS84
- [ ] entry in `datasets/registry.yaml`; `scripts/validate_datasets.py` clean or its warnings explained; cookie cache built
- [ ] `PROMPT_ACTION_LOG.md` entry; working notes updated
