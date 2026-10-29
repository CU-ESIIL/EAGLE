# Ingesting a new evaluation dataset

How a labeled dataset becomes an evaluation task for the lidar encoder: a table of locations with labels, each tied to the 3DEP lidar product closest in time. Worked examples to copy from: `lfrdb/` (database download, many filters), `snapshot_usa/` (counts to per-location targets, "all"/"-20" variants), `satbird/` (API download of coordinates, encounter-rate and occupancy variants), `vegbench/` (ready-made split, repeat-survey analysis), `nps_vmi/` (many tables).

## Steps

1. **Get the raw data into `datasets/raw/<name>/`** (gitignored; large data stays out of GitHub). Keep files unmodified. Record where and when they came from: source URL or DOI, version, download date (UTC), and file hashes. Scripted downloads write `catalog/download_log.csv` and a `SOURCE.md` next to the raw files (see `lfrdb/01_download.py`); for a manual download, write the same facts in the dataset README.
2. **Read the dataset's own documentation, then profile the data.** Learn what one row is (plot, visit, deployment, route), how locations are recorded (precision, method, datum), what the date means, and what the labels are. Check duplicates, missing values, repeated locations and coordinates that are rounded or obviously wrong. Put findings that affect use in the README (or `DATASET_CONCERNS.md`, as in `nps_vmi/`).
3. **Harmonize to one table: one row per labeled location and visit.** Required columns: a unique id, `lat`, `lon` (WGS84 decimal degrees), `year` (survey year). Derived tables go in `datasets/raw/<name>/derived/`. Keep the dataset's own column meaning; rename only for consistency (`lat`, `lon`, `year`).
4. **Apply location filters before the lidar lookup, and log counts after every filter.** Typical: valid coordinates, enough decimals for a 100 m-radius cookie (4-5 decimals), a precise location method, a plausible year, dataset-specific quality flags. Write the counts to `catalog/filter_counts*.csv` with a `dropped` column. Say why each rule exists.
5. **Match each location to 3DEP.** Call `get_nearest_year_product(lat, lon, year, source="AWS")` from `src/streaming/3dep/get_als.py` once per unique (lat, lon, year); it returns `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS` (absolute gap). Treat an exception as "no match" (footprints without a collection year raise). Then `eagle_als.sites.add_site_ids(df, "lat", "lon")` adds `als_site_id` (`<product>__<lat>_<lon>`), the cookie's key in the cache. The standard rule is `year_diff_AWS <= 3`. If the labels pool many years (eBird encounter rates, long-term counts), there is no single survey year: accept any product collected inside the pooled period, take the latest one, set `year_diff_AWS` to the gap to that period (0 inside), and document the rule and its consequence in the README (`satbird/`).
6. **Report the year-gap trade-off before choosing N.** Write how many rows (per split, per class if relevant) survive N = 0..5 (`catalog/year_gap_counts*.csv`; `vegbench/01_match_als.py` does this). Keep the table of all covered rows with `year_diff_AWS`, so other N can be applied later, and export the N = 3 rows as the eval table.
7. **Define targets and splits.** Classification: one primary label column, plus coarser or finer variants if the taxonomy has levels. Regression: numeric columns with units in the name. Prefer the dataset's published split when it has one. Otherwise make a **spatial** hold-out: whole blocks (e.g. 1-degree lat/lon cells) assigned to test, or a natural coarse train/test division category, so neighbours do not straddle the split. Check that repeat visits to one place fall in the same split. Report per-class counts in each split (`*_split_counts.csv`) so users can choose which classes to score. For datasets with many classes or species (multi-label presence, encounter rate, cover), also publish the **task variants** below.
8. **Summarize before and after the label filters** (`figures/`: label counts, structure variables and missingness, years and match, map, final counts by split), and check for labels that are suspiciously coarse, ambiguous, or inconsistent across visits to the same place (`vegbench/02_label_turnover.py` is an example of the last).
9. **Write the evaluation table and README** (format below).
10. **Register the dataset.** Add its table to `squares.EVAL_SITES` in `src/eagle_als/squares.py` (table path, lat column, lon column) so pre-training never samples within 500 m of it. Build the cookie cache with `python -m eagle_als.cache` (see `src/eagle_als/README.md`). To score it during training, add an `eval_tasks` entry (README at the repo root, "Validation tasks during training"); keep a task to about 2,000 cookies or fewer.
11. **Log it.** Append an entry to `PROMPT_ACTION_LOG.md`, and update the project working notes.

## Recommended task variants (many classes or species)

When the label is a long list of classes or species, rare ones make a poor benchmark: a class with a handful of positives in train or test gives a noisy score. Publish two class sets and, where the data allow both, two target types:

| class set | rule | folder |
|---|---|---|
| **all** | classes present in at least 1 location (row) of **each** of train and test | `datasets/<NAME>_eval/` |
| **-20** | classes present in at least 20 locations (rows) of **each** of train and test | `datasets/<NAME>_eval-20/` |

| target type | table |
|---|---|
| presence / occupancy (0/1) | `location_occupancy.parquet`, columns `occ__<class>` |
| rate or abundance (encounter rate, detection rate) | `location_encounter_rate.parquet` or `location_detection_rate.parquet`, columns `enc__<class>` / `rate__<class>` |

Both folders carry identical location columns, rows and split, so results compare directly; only the class columns differ. Each also holds `location_species_summary.csv` (every class with its train/test counts and an `included` flag). Keep the class-set rule in a single script argument (`--min-locations N`, as in `snapshot_usa/04_location_targets.py`) or constant (`satbird/03_split_and_export.py`) and say in the README how many classes each variant keeps. The counts must use the final split, after the lidar and year-gap filters. Examples: `snapshot_usa/` and `satbird/`.

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

- [ ] raw data in `datasets/raw/<name>/` with source, version and download date recorded
- [ ] every filter has a reason and a count in `catalog/`
- [ ] year-gap counts for N = 0..5 reported; N = 3 unless there is a reason
- [ ] spatial split (or the source's published split) with per-class counts
- [ ] for many-class labels: the "all" and "-20" variants (and both target types where possible)
- [ ] `datasets/<NAME>_eval/` has the parquet, a README with columns and caveats, and figures
- [ ] added to `squares.EVAL_SITES`; cookie cache built
- [ ] `PROMPT_ACTION_LOG.md` entry; working notes updated
