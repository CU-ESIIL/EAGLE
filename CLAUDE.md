# CLAUDE.md: session handoff

Read `AGENTS.md` (repo rules) and `README.md` (project) first. The project has two workstreams, each with a
local working doc that holds its status, decisions, results, next steps and the user's feedback. **Read the
doc for the workstream you are on at the start (check it for new user feedback), and update it before you
finish**: status table, results, ticked/removed next steps, a dated log line. Stale notes are worse than none.

| workstream | machine | working doc | operational notes |
|---|---|---|---|
| **Pre-training** the LitePT-S encoder (SSL on streamed 3DEP) | PSC Bridges-2 (H100) | `training_working_doc.md` | "Pre-training (Bridges-2)" below; `README.md` (running, monitoring) |
| **Baseline benchmark** (frozen features + linear probe on 15 tasks) | NCSA Delta (was CyVerse) | `benchmarking_working_doc.md` | the rest of this file |

Both share the evaluation datasets (`datasets/registry.yaml`), the eval-site exclusion, the cookie code
(`eagle_als.fetch`/`cache`) and the linear probe (`eagle_als.linear_probe`). A change to any of these affects
both: note it in both docs. (The claude.ai working-notes artifact is retired as of 2026-11-12.)

## Conventions

- Commit regularly with descriptive messages (user's git identity is set in the repo config); pull before
  starting (`git pull --rebase`): the user pushes dataset changes in parallel.
- Append a `PROMPT_ACTION_LOG.md` entry after meaningful changes; keep the working doc and this file up to date.

## Pre-training (Bridges-2)

- `$EAGLE_SCRATCH = /ocean/projects/bio260075p/sammlapp/eagle` (runs, logs, eval cookie caches). /ocean is not
  mounted on Delta, so run state can only be checked on Bridges-2.
- Env: `source scripts/litePT/env.sh` (pixi env `envs/litept-gpu`, installed on /ocean; see the working doc).
- Submit / resume: `sbatch scripts/litePT/slurm/pretrain.sbatch run_name=<name> [key=value ...]`; resubmitting the
  same `run_name` resumes. Settings go as arguments, not environment variables. Config:
  `scripts/litePT/configs/ssl_litept_s.py`.
- Monitor: TensorBoard on `$EAGLE_SCRATCH/runs`, `scripts/litePT/TENSORBOARD.md`; judge runs on the `eval/*`
  probes and `eval/embed/*`, not the loss. Score old checkpoints with `scripts/litePT/eval_checkpoints.py`.
- Eval caches for in-training tasks: `scripts/litePT/build_eval_caches.py` (rerun when an eval table changes).
- GPU runs over 30 GPU-h need the user's go-ahead (AGENTS.md); the user usually submits the full runs.

## Benchmark conventions

- User notes and feedback: `benchmarking_working_doc.md`.
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
- Probes and other batch CPU work: in the interactive job when it is free (user, 2026-11-12), else
  `sbatch scripts/baseline_eval/probe_delta.sbatch <02_linear_probe.py args>` (account `bjud-delta-cpu`, partition `cpu`,
  16 cores x 4 h = 64 CPU-h max by default). AGENTS.md compute policy:
  ask the user before a job over 300 CPU-h or 30 GPU-h, or when a partition's credits are 80% used (`accounts`).
- `$WORK = /work/hdd/bjud/sammlapp` (1.5 TB project quota) holds the cache: `$WORK/benchmarking` plays the role of `$BENCH`.
- pixi: `~/.pixi/bin/pixi` (installed by the user). Envs `envs/eagle-bench`, `envs/lidr`, `envs/litept-gpu` (13 GB; GPU env needs
  `CONDA_OVERRIDE_CUDA=12.6` for `pixi install`/`pixi run` on CPU nodes). Older fallback, no longer used: a conda prefix built from the `miniforge3-python` module (`module load` does not set PATH in agent shells; use
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
| AlphaEarth, centre pixel, all tasks | **done** 2026-11-10 (torch probe rerun 2026-11-12) | `results/alphaearth_center/`; embeddings `$WORK/benchmarking/aef/points_ee` on Delta (via symlink `datasets/raw/aef`) |
| AlphaEarth after SatBird → `satbird_winter_occupancy` | re-extracted (18,379 new points); probes rerun | check `results/alphaearth_center/summary.md` has `satbird_winter_occupancy` |
| Lidar site table | **done**: 86,150 unique sites in `$BENCH/als_cache/sites.parquet` | rerun `scripts/baseline_eval/03_lidar_sites.py` when a dataset changes |
| Lidar cookie cache (Delta) | **done** 2026-11-12 04:43 CST in `$WORK/benchmarking/als_cache`: 82,160 ok, 3,892 empty, 98 too few points, 0 errors (the 46 noise-rejected sites were refetched with the fixed noise filter) | rerun `run_lidar_cache.sh` (BENCH=$WORK/benchmarking) only when the site table changes |
| Lidar cookie cache (CyVerse) | started 2026-11-10 in tmux `als_cache` on CyVerse (~1.1 cookies/s); state unknown from Delta | separate copy under the CyVerse `$BENCH`; large-data work now on Delta/Bridges-2 (AGENTS.md) |
| ForPT embedder | **done** 2026-11-12 at the task radii capped at 100 m: jobs 23829140 (r15/30/50, 58 min) and 23829141 (r100, 6.4 h); rows r15 875, r30 6,061, r50 25,227, r100 51,533; 131 empty-disc cookies failed (expected, edge sites) | output `$WORK/benchmarking/embeddings/forpt/r<R>_z`; logs `$WORK/benchmarking/logs/forpt-embed-<job>.out`; rerun `sbatch -p gpuA100x4 -t 16:00:00 scripts/baseline_eval/embed_forpt_delta.sbatch --radius-m task` only if the cache changes |
| ForPT probes | **done** 2026-11-25 (torch probe on the interactive A100, env `envs/litept-gpu`; that env lacks `tabulate`, so `summary.md` was written with `envs/eagle-bench`) | `scripts/baseline_eval/results/forpt/`; log `$WORK/benchmarking/logs/probe_forpt.log`; comparison in the working doc |
| lidR stdmetrics (+terrain) | **running**: r15, r30, r50 done; r100 in Slurm 24526193 (`--workers 64 --batch 2`; the first try 24522657 was OOM-killed at r100 with 92 workers x batch 10 in 180 GB). First job 23829139 failed on a terra error in the terrain step, fixed 2026-11-25 (terrain errors -> NA columns; a failed batch is logged and skipped, retried on the next run); also fixed: all-empty-disc batches wrote 0-column parquet files that broke the resume scan, and shard numbering by file count overwrote shards after an interrupted run (now max index + 1) | check `$WORK/benchmarking/logs/lidr-metrics-24526193.out` (`batch ... failed` lines); resubmit `sbatch scripts/baseline_eval/lidr_metrics_delta.sbatch --radius-m task --workers 64 --batch 2` if it times out; output `$WORK/benchmarking/features/lidr_stdmetrics/r<R>`; probe with `--embeddings '.../lidr_stdmetrics/r{radius}'` (with and without `--exclude "^terrain_\|^n_ground$"`) |
| Clay v1.5 NAIP embeddings | **done** 2026-11-12 on Delta: fetch (32,593 files / 253 GB streamed, 0 failed; 87,298/87,749 points with an embedding; pooled tables `$WORK/benchmarking/naip_clay/pooled/`) and probe with the torch linear probe (all 15 tasks, 40 min on 32 CPUs) | `results/naip_clay/summary.md`; results table in the working doc |
| Linear probe = torch (2026-11-12) | **all probes use `eagle_als.linear_probe`** (one `nn.Linear`, full-batch L-BFGS, class-balanced, l2 from 1e2..1e-5 by 3-fold CV). Reruns **done** 2026-11-12 on Delta for alphaearth_center, location_rff, satclip_l10/l40 (logs `$WORK/benchmarking/logs/probe_*.log`) and naip_clay; comparison table in the working doc | ForPT and lidR still to probe |
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

- **Noise points (classes 7/18) in cookies fetched before 2026-11-12.** The old PDAL filter
  `Classification![7:7],Classification![18:18]` dropped nothing (PDAL ORs ranges on the same dimension). Fixed 2026-11-12:
  `eagle_als.fetch.NOISE_FILTER` (`filters.expression`, also used by `eagle_als.coreg.data`). The Delta eval cache was
  fetched with the old filter except the 46 sites retried after the fix, so `05_lidr_metrics.py` and `eagle_bench/forpt.py`
  must keep dropping 7/18 downstream. `scripts/3dep/dataset.py` uses the same filter.
- **Sites at collection edges (accepted, documented; user 2026-11-12).** Matching (`eagle_als.match`) uses footprint
  polygons, which are coarser than the point clouds, so a site can get a cookie with points on one side only or none in
  the disc. `scripts/baseline_eval/10_cookie_qc.py` -> `$WORK/benchmarking/als_cache/qc_r50.parquet` (`low_points` =
  fewer than 50% of the 3DEP minimum: QL0/1 8, QL2 2, QL3/unknown 0.5 pulses/m2 x disc area; `cell_coverage` = share
  of 5 m cells with a point). 1,702 of 82,160 flagged (SatBird 3.2%, others <= 1.5%), 436 with an empty disc.
  Empty discs give no ForPT / lidR row (missing embedding in the probe, `n_no_embedding`); partly covered ones are
  kept. Open water is also flagged (few returns), so do not drop flagged cookies blindly.
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
