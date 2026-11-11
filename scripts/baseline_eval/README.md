# Baseline evaluation: AlphaEarth embeddings + linear probes

Scores frozen embeddings with a linear probe on the whole benchmark suite, as a reference for the lidar
encoder. First baseline: **AlphaEarth Foundations (AEF) annual embeddings** (64-d, 10 m, 2017-2025). Reusable
code is in `src/eagle_bench/` (see its README); scripts here only wire it together.

**Status (2026-11-10): done for all 15 tasks, centre pixel, through Google Earth Engine** (results in
`results/alphaearth_center/`). The COG route below was only run for NLCD (`results/alphaearth/`, disc and centre).

### Earth Engine route (recommended)

The COGs are band-interleaved: each of the 64 bands is a separate ~0.5 MB zstd block of 1024 x 1024 px, so one
pixel's vector costs ~32 MB and the whole suite ~1.07 TB (33,464 blocks, ~5 h at the 2 blocks/s measured on
CyVerse). Earth Engine returns only the sampled pixels: all 95,410 points in ~7 min.

    earthengine authenticate --auth_mode=notebook        # once per machine
    python scripts/baseline_eval/01_extract_aef_ee.py --project eagle-lidar
    python scripts/baseline_eval/02_linear_probe.py --points datasets/raw/aef/points_ee --pooling center \
        --model alphaearth_center --n-jobs 4

Validated against the COG extraction: cosine similarity median 1.000 over the 892 NLCD points (5th percentile
0.988), same image year for every point, which also settles the orientation check below. No point was masked
and none needed a year other than its (clipped) survey year.

### Clay v1.5 NAIP embeddings (`09_fetch_naip_clay.py`)

Pre-computed 1024-d embeddings of 256 x 256 px NAIP chips (256 m at 1 m, ~154 m at 60 cm), averaged over a disc around each point with a radius per task (`naip_clay_radii.yaml`). Files are read whole and dropped; only the pooled
embedding per point is kept (`pooled/<task>.parquet`). Details in `src/eagle_bench/naip_clay.py` and the working doc.

## Tasks (16 entries, 10 datasets)

| task | kind | rows (train/test) | label | notes |
|---|---|---|---|---|
| `nlcd` | multiclass | 892 | NLCD Level II class | class-balanced subset |
| `lfrdb_ecosys`, `lfrdb_nvc_group` | multiclass | 8,027 | LFRDB ecological system / NVC group | |
| `nps_vmi_macrogroup`, `nps_vmi_group` | multiclass | 6,108 | NVC macrogroup / group | |
| `vegbench_formation`, `_division`, `_macrogroup`, `_group` | multiclass | 9,440 | USNVC level | published split, `val` unused |
| `bbs_occurrence` | multilabel | 12,934 route-years | 670 species, count > 0 | |
| `butterflies_occurrence` | multilabel | 265 sites | 387 species, any record | site level, year = mean survey year |
| `snapshot_usa_occurrence` | multilabel | 3,409 locations | 159 species (39 with >= 20 locations per split) | `SNAPSHOT_USA_eval` |
| `satbird_winter_occupancy` | multilabel | 40,514 hotspots (SatBird train + test; val unused) | 587 species (467 with >= 20 hotspots per split) | `SATBIRD_USA_winter_eval`; year 2022 |
| `blm_aim` | regression | 9,233 | 12 cover / height / richness targets | |
| `ofo_trees` | regression | 197 | `basal_area_m2ha`, `total_trees` | `datasets/OFO_trees/plots_w_als.gpkg` |

Splits are each dataset's stored spatial hold-out (`test_split`, or `split == test`); nothing is re-split.
SatBird encounter rate is not included (occupancy only); the class-set scenarios (all classes vs. >= 20 rows per split, `--min-counts`) share the same embeddings and tables.

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
  (1 for `satbird_winter_occupancy`). Rows with no embedding (masked pixels) are excluded and counted in `n_no_embedding`.
- Metrics: multiclass balanced accuracy, macro F1, accuracy (chance = 1/classes); multilabel mean AUROC and
  mean average precision (AP of a random ranking = `mean_prevalence`); regression R2, Spearman, RMSE per target.

## Handoff: run on Bridges-2

1. **Sync.** Needs the repo with `src/eagle_bench/`, `scripts/baseline_eval/`, and these `datasets/` inputs (they are
   built by `data_cleaning/`; several (SatBird, VegBench) are still untracked in git, so copy them; `datasets/raw/`
   is gitignored and not needed): `NLCD_eval`, `LFRDB_eval`,
   `NPS_VMI`, `VegBench_eval`, `BBS`, `butterflies`, `SNAPSHOT_USA_eval`, `SATBIRD_USA_winter_eval`,
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
