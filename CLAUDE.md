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
- Downloaded weights: always `torch.load(..., weights_only=True)`. Downloaded code (e.g. `/home/jovyan/external/ForPT`)
  is read only, never executed.

## Machine (CyVerse JupyterHub)

- **4 CPUs** (cgroup quota; `nproc` misleadingly says 128): pass `--n-jobs 4` / `--workers` accordingly.
- GPU: NVIDIA A16, 15 GB. No flash-attn (LitePT falls back to SDPA; memory-hungry, keep ~200k points per pass).
- Environments (uv venvs, outside the repo; `$HOME` is not on the data store, so they may need rebuilding):
  - `/home/jovyan/envs/eagle-bench` (CPU): rasterio, pyproj, geopandas, scikit-learn, tabulate, earthengine-api.
    Rebuild: `uv venv -p 3.12 <dir> && uv pip install rasterio pyproj geopandas pandas pyarrow scikit-learn scipy joblib tabulate pyyaml earthengine-api`
  - `/home/jovyan/envs/eagle-gpu` (GPU): torch 2.8.0+cu126, torch-scatter (PyG wheel for torch 2.8.0+cu126),
    spconv-cu126, timm, addict + the CPU stack. Install torch **from the cu126 index last**, other packages can
    silently upgrade it (torch-scatter then segfaults). Pinned list: `/home/jovyan/envs/eagle-gpu.freeze.txt`.
  - **No PDAL yet** (no pip wheel; pixi install script and conda were not approved). Lidar cookie fetching
    (`eagle_als.cache`) needs it. Waiting on the user (see Status).
- Run everything from the repo root with `PYTHONPATH=src`.
- Earth Engine: project `eagle-lidar`, credentials in `~/.config/earthengine/credentials`
  (re-auth: `earthengine authenticate --auth_mode=notebook`, user does it in a terminal).

## Status (update this table)

| step | state | how to resume / next |
|---|---|---|
| AlphaEarth, centre pixel, all tasks | **done** 2026-11-10 | `results/alphaearth_center/`; embeddings `$BENCH/aef/points_ee` (via symlink `datasets/raw/aef`) |
| AlphaEarth after SatBird → `satbird_winter_occupancy` | re-extracted (18,379 new points); probes rerun | check `results/alphaearth_center/summary.md` has `satbird_winter_occupancy` |
| Lidar site table | **done**: 86,150 unique sites in `$BENCH/als_cache/sites.parquet` | rerun `scripts/baseline_eval/03_lidar_sites.py` when a dataset changes |
| Lidar cookie cache | **blocked: needs PDAL** | `python -m eagle_als.cache --sites $BENCH/als_cache/sites.parquet --out $BENCH/als_cache --id-col als_site_id --workers 8` (resumable; run in background, log to `$BENCH/als_cache/logs/`) |
| ForPT embedder | **written, smoke-tested on synthetic clouds** | weights `$BENCH/forpt_weights/epoch_30.pth`; run `scripts/baseline_eval/04_embed_forpt.py --limit 200` once cookies exist, then full |
| ForPT probes | not started | `02_linear_probe.py --embeddings $BENCH/embeddings/forpt/r50_z --model forpt_r50_z --n-jobs 4` |
| Lidar metrics (derived products) | **proposal awaiting user review** in `benchmarking_working_doc.md` | implement after approval |

## Pipeline (scripts/baseline_eval/)

1. `01_extract_aef_ee.py --project eagle-lidar` → `$BENCH/aef/points_ee/part_<task>_<n>.parquet` (resumable).
   (`01_extract_aef.py` = COG route, ~1 TB; only used for NLCD validation.)
2. `02_linear_probe.py` → `results/<model>/{summary.csv,per_target.csv,summary.md}`. It rewrites the summary with
   only the tasks it runs: always run all tasks for a model. AEF: `--points ... --pooling center`; cookie-keyed
   features: `--embeddings <dir>`.
3. `03_lidar_sites.py` → site table for the shared cookie cache (priority order: small tasks first, SatBird last).
4. `04_embed_forpt.py` → `$BENCH/embeddings/forpt/r<radius>_<z|hag>/part_*.parquet` (1008-d, resumable).

## ForPT facts (from github.com/prs-eth/ForPT, cloned at /home/jovyan/external/ForPT)

- Teacher backbone loads into the vendored `litept.model.LitePT(in_channels=3, enc_mode=True, norm="ln",
  embedding="linear", mask_token=True)` with all 169 keys matching (prefix `module.teacher.backbone.`).
- Inputs: xyz only; CenterShift(apply_z=True); GridSample 0.05 m. Pooling as their `DefaultClassifierLinearProbeCat`:
  up-cast all stages, concat (1008-d), mean over points. Set `shuffle_orders=False` for reproducible embeddings.
- Speed (synthetic, A16, CPU busy): ~4-5 s per forward pass, mostly fixed CPU overhead, so cookies are packed
  into passes of ≤200k points. Re-measure on real cookies before estimating the full run.
- Domain gap to keep in mind: ForPT was pretrained on dense forest ALS/ULS/MLS at 5 cm; 3DEP is 2-30 pts/m².
