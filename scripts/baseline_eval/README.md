# Baseline evaluation: AlphaEarth embeddings + linear probes

Scores frozen embeddings with a linear probe on the whole benchmark suite, as a reference for the lidar
encoder. First baseline: **AlphaEarth Foundations (AEF) annual embeddings** (64-d, 10 m, 2017-2025). Reusable
code is in `src/eagle_bench/` (see its README); scripts here only wire it together.

**Status (2026-10-29): code written and tested locally on synthetic embeddings and a 2-point real extraction; the
full extraction has NOT been run.** It reads about 1 TB, so it is handed off to Bridges-2.

## Tasks (16 entries, 10 datasets)

| task | kind | rows (train/test) | label | notes |
|---|---|---|---|---|
| `nlcd` | multiclass | 892 | NLCD Level II class | class-balanced subset |
| `lfrdb_ecosys`, `lfrdb_nvc_group` | multiclass | 8,027 | LFRDB ecological system / NVC group | |
| `nps_vmi_macrogroup`, `nps_vmi_group` | multiclass | 4,159 | NVC macrogroup / group | |
| `vegbench_formation`, `_division`, `_macrogroup`, `_group` | multiclass | 9,440 | USNVC level | published split, `val` unused |
| `bbs_occurrence` | multilabel | 12,934 route-years | 670 species, count > 0 | |
| `butterflies_occurrence` | multilabel | 265 sites | 387 species, any record | site level, year = mean survey year |
| `snapshot_usa_occurrence` | multilabel | 3,409 locations | 39 species | `SNAPSHOT_USA_eval-20` set |
| `satbird_occupancy` | multilabel | 48,048 hotspots | all 618 species (present in >= 1 hotspot of train and test) | `SATBIRD_eval`; year 2022 |
| `satbird_occupancy_20` | multilabel | 48,048 hotspots | 537 species with >= 20 hotspots in each split | `SATBIRD_eval-20` |
| `blm_aim` | regression | 9,233 | 12 cover / height / richness targets | |
| `ofo_trees` | regression | 197 | `basal_area_m2ha`, `total_trees` | `datasets/OFO_trees/plots_w_als.gpkg` |

Splits are each dataset's stored spatial hold-out (`test_split`, or `split == test`); nothing is re-split.
SatBird encounter rate is not included (occupancy only); the `-20` and "all" tasks share the same embeddings.

## Design decisions

- **No row is dropped for the year.** The AEF year is the survey year clipped to 2017-2025, i.e. the closest
  year with imagery; if that year has no tile for the point, the nearest year that does (`aef_year` records the
  year used). Many surveys predate 2017 (all of LFRDB, ~78% NPS VMI, ~47% VegBench, ~22% BBS), so these use 2017
  imagery; `frac_year_clipped` is reported per task. Pooled labels use a representative year (butterflies: mean
  survey year; SnapShot: midpoint of first/last year; SatBird: 2022).
- **Pooling.** `disc` = mean of the valid pixels within 50 m of the point (matches the lidar probe's central 50 m
  disc), renormalised to unit length; `center` = the single pixel. Both are reported.
- **Probe.** Standardise on train, then logistic regression (class-balanced, C in {0.01, 0.1, 1} by 3-fold CV on
  train) or ridge (RidgeCV). Multiclass classes are scored if they have >= 20 train and >= 5 test rows
  (`test_rows_scored_frac` says how much of the test set that covers); species if >= 20 positives in each split
  (1 for `satbird_occupancy`). Rows with no embedding (masked pixels) are excluded and counted in `n_no_embedding`.
- Metrics: multiclass balanced accuracy, macro F1, accuracy (chance = 1/classes); multilabel mean AUROC and
  mean average precision (AP of a random ranking = `mean_prevalence`); regression R2, Spearman, RMSE per target.

## Handoff: run on Bridges-2

1. **Sync.** Needs the repo with `src/eagle_bench/`, `scripts/baseline_eval/`, and these `datasets/` inputs (they are
   built by `data_cleaning/`; several (SatBird, VegBench) are still untracked in git, so copy them; `datasets/raw/`
   is gitignored and not needed): `NLCD_eval`, `LFRDB_eval`,
   `NPS_VMI`, `VegBench_eval`, `BBS`, `butterflies`, `SNAPSHOT_USA_eval-20`, `SATBIRD_eval`, `SATBIRD_eval-20`,
   `BLM_AIM`, `OFO_trees`. Check each exists (`ls`); load a task with
   `python -c "import sys; sys.path.insert(0,'src'); from eagle_bench.tasks import load_task; print(load_task('nlcd').df.shape)"`.
2. **Environment.** Needs `rasterio`, `pyproj`, `geopandas`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `joblib`,
   `tabulate`. The `litept-gpu` env (`scripts/litePT/env.sh`) has geopandas/pandas/scipy but **not** rasterio or
   scikit-learn: either add them to `envs/litept-gpu/pixi.toml`, or install the repo's root `pixi.toml`
   environment (`linux-64` is a supported platform) on /ocean. Test: `python -c "import rasterio, sklearn"`.
3. **Check the network first.** On a compute node: `curl -sI https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet | head -1`
   (needs HTTP 200). Use a node with internet access.
4. **Count, no download:** `python scripts/baseline_eval/01_extract_aef.py --count-only`. Expect 93,971 unique points,
   0 with no tile, ~6,800 files, ~33,400 blocks (~1 TB). The first call downloads the 80 MB index to `datasets/raw/aef/`.
5. **Small trial** (a few minutes; NLCD has ~900 points): `python scripts/baseline_eval/01_extract_aef.py --tasks nlcd --workers 16`
   then `python scripts/baseline_eval/02_linear_probe.py --tasks nlcd`.
6. **Full run:** `sbatch scripts/baseline_eval/extract_aef.sbatch` (RM-shared, 32 cores, 24 h; it also runs the probes
   at the end). Measured single-threaded here: ~17-25 s per 1024 x 1024 block; expect ~1-3 h with 32 threads on a
   fast link, and it resumes if cut off (rerun the same command; finished shards are skipped). If the throughput is
   far lower, raise `--workers`, or run per task (`--tasks ...`) in separate jobs.
7. **Results** appear in `scripts/baseline_eval/results/alphaearth/` (`summary.csv`, `per_target.csv`, `summary.md`).
   Commit those small files; leave `datasets/raw/aef/` (embeddings, ~100 MB) out of git. Then append a
   `PROMPT_ACTION_LOG.md` entry and update the working notes with the results table.

## Checks the receiving agent should do

- **Orientation.** The AEF COGs are bottom-up (row 0 at the southern edge); `aef.locate` computes rows from
  `utm_south` with +10 m rows, which matches the file's geotransform, but it has not been validated against Earth
  Engine. Spot-check ~5 points: `ee.Image('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')` filtered to the year, sampled at the
  point, should have cosine similarity ~1 with the `c00..c63` values (small differences from quantisation).
- Each stored embedding should have unit norm (`n_valid_disc` is 81 for a full 50 m disc). Points with
  `n_valid_disc == 0` (water or no imagery) are NaN and excluded from scoring; report how many per task.
- Sanity: random embeddings give chance-level scores (verified); AEF should beat chance on every task, with the
  largest gains on land cover (NLCD, VegBench formation/division).

## Local test numbers

Two OFO points, one block: `aef.extract` took 24 s on a laptop; embeddings had norm 1.0, 81 valid pixels in the disc, and
centre-vs-disc cosine 0.986. These are the only real AEF values extracted so far, so there are no probe results yet.
