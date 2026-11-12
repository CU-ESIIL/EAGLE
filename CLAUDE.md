# CLAUDE.md: benchmarking session handoff

Read `AGENTS.md` (repo rules) and `README.md` (project) first. This file is the operational state of the
**baseline benchmarking** work on CyVerse, kept current so another agent can resume mid-experiment.
User notes and feedback for this work: `benchmarking_working_doc.md` (check it for new feedback at the start).

## Conventions for this work

- Commit regularly with descriptive messages (user's git identity is set in the repo config); pull before
  starting (`git pull --rebase`): the user pushes dataset changes in parallel.
- Append a `PROMPT_ACTION_LOG.md` entry after meaningful changes; keep `benchmarking_working_doc.md` notes and this
  file up to date.
- Large data never goes in git. Persistent cache: `$BENCH = /home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking`
  (iRODS FUSE mount: fine for large sequential files, slow for many small reads).
- Downloaded weights: always `torch.load(..., weights_only=True)`.
- Prefer the established open-source implementation (e.g. the official ForPT / SatCLIP code, installed or cloned
  outside the repo, e.g. `/home/jovyan/external/`) over custom re-implementations: it is more reproducible and
  scientifically defensible. Re-implement only when the package cannot be used, and say why. (User policy,
  2026-11-11; this replaces an earlier "downloaded code is read only, never executed" rule.)

## Machine: NCSA Delta (from 2026-11-12)

- Interactive Slurm jobs (e.g. 8 h, 32 CPUs, CPU partition, ~250 GB RAM). Everything dies at the job's end time
  (`scontrol show job $SLURM_JOB_ID | grep EndTime`), so long runs use `--max-hours` and are resumed in the next job.
- `$WORK = /work/hdd/bjud/sammlapp` (1.5 TB project quota) holds the cache: `$WORK/benchmarking` plays the role of `$BENCH`.
- No pixi on Delta yet (installing it via `curl | bash` was blocked for the agent; the user can install it). Meanwhile the
  CPU env is a conda prefix built from the `miniforge3-python` module (`module load` does not set PATH in agent shells; use
  `/sw/rh9.4/python/miniforge3/bin/mamba` directly):
  `CONDA_PKGS_DIRS=$WORK/conda_pkgs mamba create -p $WORK/envs/eagle-bench -c conda-forge --override-channels python=3.12 "python-pdal>=3.5,<4" pdal gdal rasterio pyproj shapely geopandas pandas pyarrow numpy scipy scikit-learn joblib tabulate pyyaml`.
  Unactivated use needs `PROJ_DATA=$WORK/envs/eagle-bench/share/proj` (the run script sets it). The PDAL HDF-plugin load
  errors at import are harmless.

## Machine (CyVerse JupyterHub)

- **4 CPUs** (cgroup quota; `nproc` misleadingly says 128): pass `--n-jobs 4` / `--workers` accordingly.
- GPU: NVIDIA A16, 15 GB. No flash-attn (LitePT falls back to SDPA; memory-hungry, keep ~200k points per pass).
- Environments (uv venvs, outside the repo; `$HOME` is not on the data store, so they may need rebuilding):
  - `/home/jovyan/envs/eagle-bench` (CPU): rasterio, pyproj, geopandas, scikit-learn, tabulate, earthengine-api.
    Rebuild: `uv venv -p 3.12 <dir> && uv pip install rasterio pyproj geopandas pandas pyarrow scikit-learn scipy joblib tabulate pyyaml earthengine-api`
  - `/home/jovyan/envs/eagle-gpu` (GPU): torch 2.8.0+cu126, torch-scatter (PyG wheel for torch 2.8.0+cu126),
    spconv-cu126, timm, addict + the CPU stack. Install torch **from the cu126 index last**, other packages can
    silently upgrade it (torch-scatter then segfaults). Pinned list: `/home/jovyan/envs/eagle-gpu.freeze.txt`.
  - pixi (installed by the user at `~/.pixi/bin/pixi`): `envs/eagle-bench` (CPU + PDAL, used for cookie fetching and
    lidar metrics) and `envs/lidr` (R 4.4 + lidR 4.3.3, arrow, terra; linux-64 only). Always go through
    `pixi run --manifest-path ...` (activation sets PROJ_DATA/GDAL_DATA).
- Run everything from the repo root with `PYTHONPATH=src`.
- Earth Engine: project `eagle-lidar`, credentials in `~/.config/earthengine/credentials`
  (re-auth: `earthengine authenticate --auth_mode=notebook`, user does it in a terminal).

## Status (update this table)

| step | state | how to resume / next |
|---|---|---|
| AlphaEarth, centre pixel, all tasks | **done** 2026-11-10 | `results/alphaearth_center/`; embeddings `$BENCH/aef/points_ee` (via symlink `datasets/raw/aef`) |
| AlphaEarth after SatBird → `satbird_winter_occupancy` | re-extracted (18,379 new points); probes rerun | check `results/alphaearth_center/summary.md` has `satbird_winter_occupancy` |
| Lidar site table | **done**: 86,150 unique sites in `$BENCH/als_cache/sites.parquet` | rerun `scripts/baseline_eval/03_lidar_sites.py` when a dataset changes |
| Lidar cookie cache (Delta) | **running** since 2026-11-12 02:01 CST on Delta node cn104, tmux `als_cache`, 64 workers, `--max-hours 7.3` (job ends 09:34 CST); ~11 cookies/s (CPU-bound, load ~100 on 32 CPUs), so ~2.2 h and ~650 GB expected; site table `$WORK/benchmarking/als_cache/sites.parquet` (86,150 sites, same as CyVerse) | log `$WORK/benchmarking/als_cache/logs/`. Resume in a new job: `tmux new -d -s als_cache "BENCH=$WORK/benchmarking EAGLE_ENV=$WORK/envs/eagle-bench WORKERS=64 bash scripts/baseline_eval/run_lidar_cache.sh --max-hours <job hours - 0.5>"` |
| Lidar cookie cache (CyVerse) | started 2026-11-10 in tmux `als_cache` on CyVerse (~1.1 cookies/s); state unknown from Delta | separate copy under the CyVerse `$BENCH`; large-data work now on Delta/Bridges-2 (AGENTS.md) |
| ForPT embedder | **written, smoke-tested on synthetic clouds** | weights `$BENCH/forpt_weights/epoch_30.pth`; run `scripts/baseline_eval/04_embed_forpt.py --limit 200` once cookies exist, then full |
| ForPT probes | not started | `02_linear_probe.py --embeddings $BENCH/embeddings/forpt/r50_z --model forpt_r50_z --n-jobs 4` |
| lidR stdmetrics (+terrain) | **pipeline written and trialled** (20 cookies); user approved lidR::stdmetrics with and without terrain, no GEDI | `pixi run --manifest-path envs/eagle-bench/pixi.toml env PYTHONPATH=src python scripts/baseline_eval/05_lidr_metrics.py` (resumable; ~0.18 cookies/s while the cache saturates the CPUs, so best run after it); probe `02_linear_probe.py --embeddings $BENCH/features/lidr_stdmetrics/r50 [--exclude '^terrain_|^n_ground$']` |
| Clay v1.5 NAIP embeddings | **NLCD-only trial done** 2026-11-11 (Mac, r=15 m, 892 pts: bal. acc. 0.636 vs AlphaEarth 0.751; `results/naip_clay_nlcd/`). **Full all-task run not started** (too slow from the Mac: ~1 file/s). Tables without `radius` in `datasets/registry.yaml` use `--default-radius` (50 m) | follow "Runbook: Clay NAIP embeddings" below; output `$BENCH/naip_clay/pooled/<task>.parquet`, then probe |
| Random forest / gradient boosting probe | not started (user wants linear + RF for the lidR metrics) | add to `eagle_bench/probes.py` |
| AlphaEarth geography / leakage diagnostics | **plan in working doc**, not run | tests 1, 3, 5 first (location-only baselines, displacement test, lat/lon from AEF) |

## Pipeline (scripts/baseline_eval/)

1. `01_extract_aef_ee.py --project eagle-lidar` → `$BENCH/aef/points_ee/part_<task>_<n>.parquet` (resumable).
   (`01_extract_aef.py` = COG route, ~1 TB; only used for NLCD validation.)
2. `02_linear_probe.py` → `results/<model>/{summary.csv,per_target.csv,summary.md}`. It rewrites the summary with
   only the tasks it runs: always run all tasks for a model. AEF: `--points ... --pooling center`; cookie-keyed
   features: `--embeddings <dir>`.
3. `03_lidar_sites.py` → site table for the shared cookie cache (priority order: small tasks first, SatBird last).
4. `04_embed_forpt.py` → `$BENCH/embeddings/forpt/r<radius>_<z|hag>/part_*.parquet` (1008-d, resumable).

## Runbook: Clay NAIP embeddings (run on a machine with a fast link to AWS us-west-2, e.g. CyVerse)

What it does: for each eval point, averages the Clay v1.5 NAIP chip embeddings (1024-d; chips are 256 m, or ~154 m for 60 cm NAIP) over a disc of the task's radius, using the NAIP year nearest to the survey year. Each 3.6 MB source file
is downloaded, pooled and dropped: **no tiles or chips are stored** (user requirement), only the pooled embedding per point. Code: `src/eagle_bench/naip_clay.py`, `scripts/baseline_eval/09_fetch_naip_clay.py`; background and checks in the working doc.

1. `git pull --rebase`. Check the user's feedback in `benchmarking_working_doc.md`.
2. Radii: set `radius:` (metres) for every table in `datasets/registry.yaml`. Blank ones fall back to `--default-radius` (50 m) with a printed notice: ask the user for the real values before the full run rather than inventing them. Radii below ~150 m return the chip under the point. Optional override: `--radii file.yaml` with `{task or table: metres}`.
3. Env: the CPU env `/home/jovyan/envs/eagle-bench` is enough (needs pandas, pyarrow, shapely (comes with geopandas), scipy, pyyaml). No GPU, no PDAL, no Earth Engine. Do not run it inside the tmux session that fills the lidar cache if the CPUs are saturated; the work is network-bound, but the parsing needs a core or two.
4. Dry run (about 5 min incl. the index build, ~100 files): `PYTHONPATH=src python scripts/baseline_eval/09_fetch_naip_clay.py --out $BENCH/naip_clay_test --max-points 100 --workers 16`. Expect `0 points touch no cell`, median coverage 1.00 and `|NAIP year - target|` median 0-1 for every task. Then `rm -r $BENCH/naip_clay_test`.
5. Measure the rate from the dry run (files/s). From the user's Mac it was ~1.3 files/s regardless of threads (network cap). The full run is 30-45k files (230-350 GB, depends on the radii; the plan line prints the exact count and GB), so decide whether it is acceptable and tell the user the estimate.
6. Full run in tmux (resumable, rerun the same command after any interruption; failed files are listed and retried on the next run):
   `tmux new -d -s naip_clay 'cd <repo> && PYTHONPATH=src python scripts/baseline_eval/09_fetch_naip_clay.py --out $BENCH/naip_clay --workers 16 2>&1 | tee -a $BENCH/naip_clay/fetch.log'`
   Progress: the log line `n/N files ... files/s` after every 400 files. First run builds `naip_index.parquet` (3 min, lists the bucket anonymously over HTTPS; no credentials needed). Increase `--workers` (32-64) if throughput scales.
7. When it finishes it writes `$BENCH/naip_clay/pooled/<task>.parquet` for each task and prints points with an embedding, median coverage and year gap per task. Points in no NAIP-embedded cell (about 0.5%, e.g. outside CONUS) have NaN and count as `n_no_embedding` in the probe.
8. Probe (4 CPUs): `PYTHONPATH=src python scripts/baseline_eval/02_linear_probe.py --naip-clay $BENCH/naip_clay --model naip_clay --n-jobs 4` (all tasks; writes `results/naip_clay/`).
9. Record: results table + the per-task radii and timings in the working doc, a `PROMPT_ACTION_LOG.md` entry, update the status table, commit (pooled tables are large data: do not commit them).
Changing a radius or adding points later only reads the files whose (point, radius) pairs are not in `partial/` yet; changing a radius of an already pooled task re-reads that task's files, because only sums for the old radius were kept.

## Known data issues

- **Noise points are in the cached cookies.** `eagle_als.fetch` drops classes 7/18 with the PDAL filter
  `Classification![7:7],Classification![18:18]`, but PDAL ORs ranges on the same dimension, so nothing is dropped
  (seen: 155k high-noise points in one cookie, zmax 256 m). Handled downstream in `05_lidr_metrics.py` and
  `eagle_bench/forpt.py`. Reported to the user; the fetch code (also used by pre-training) is unchanged pending their call.
- Open-water and some urban cookies have no ground points: terrain metrics are NaN there, and lidR `zentropy` is NA on
  flat cookies; the probe fills NaNs with the train median.
- Stat-ing many files on the iRODS mount is very slow: list cached cookies from the cache manifest, not `Path.exists()`.

## ForPT facts (from github.com/prs-eth/ForPT, cloned at /home/jovyan/external/ForPT)

- Teacher backbone loads into the vendored `litept.model.LitePT(in_channels=3, enc_mode=True, norm="ln",
  embedding="linear", mask_token=True)` with all 169 keys matching (prefix `module.teacher.backbone.`).
- Inputs: xyz only; CenterShift(apply_z=True); GridSample 0.05 m. Pooling as their `DefaultClassifierLinearProbeCat`:
  up-cast all stages, concat (1008-d), mean over points. Set `shuffle_orders=False` for reproducible embeddings.
- Speed (synthetic, A16, CPU busy): ~4-5 s per forward pass, mostly fixed CPU overhead, so cookies are packed
  into passes of ≤200k points. Re-measure on real cookies before estimating the full run.
- Domain gap to keep in mind: ForPT was pretrained on dense forest ALS/ULS/MLS at 5 cm; 3DEP is 2-30 pts/m².
