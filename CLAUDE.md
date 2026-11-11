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
| Lidar cookie cache | **running** since 2026-11-10 13:51 UTC in tmux session `als_cache` (`scripts/baseline_eval/run_lidar_cache.sh`); ~1.1 cookies/s, ~21 h, ~7.7 MB/cookie (~650 GB) | check: `tmux ls`; log `$BENCH/als_cache/logs/`; progress = manifest status counts (see the script header). If it died: rerun the same tmux command (resumes) |
| ForPT embedder | **written, smoke-tested on synthetic clouds** | weights `$BENCH/forpt_weights/epoch_30.pth`; run `scripts/baseline_eval/04_embed_forpt.py --limit 200` once cookies exist, then full |
| ForPT probes | not started | `02_linear_probe.py --embeddings $BENCH/embeddings/forpt/r50_z --model forpt_r50_z --n-jobs 4` |
| lidR stdmetrics (+terrain) | **pipeline written and trialled** (20 cookies); user approved lidR::stdmetrics with and without terrain, no GEDI | `pixi run --manifest-path envs/eagle-bench/pixi.toml env PYTHONPATH=src python scripts/baseline_eval/05_lidr_metrics.py` (resumable; ~0.18 cookies/s while the cache saturates the CPUs, so best run after it); probe `02_linear_probe.py --embeddings $BENCH/features/lidr_stdmetrics/r50 [--exclude '^terrain_|^n_ground$']` |
| Clay v1.5 NAIP embeddings | **code written and validated on 40 points; full run waiting for the user's per-task disc radii** (`scripts/baseline_eval/naip_clay_radii.yaml` is a placeholder) | `python scripts/baseline_eval/09_fetch_naip_clay.py --radii scripts/baseline_eval/naip_clay_radii.yaml --workers 16` (resumable; 30-45k files, 230-350 GB download, run on CyVerse) then `02_linear_probe.py --naip-clay datasets/raw/naip_clay --model naip_clay --n-jobs 4`; no tiles kept, output `pooled/<task>.parquet` |
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
