# Prompt Action Log

## 2026-06-26

### Prompt

User asked: "This repo is behind on some of the standard updates we've been making to other template repos like it needs the agents.md file and the prompt log and the ESIIL branding. Can you do a comprehensive update of the postdoc oasis so that it matches the working group oasis in style and setup but still geared toward postdocs instead of working groups. CU-ESIIL/Working_group_OASIS"

### Files and folders inspected

- Local Postdoc_OASIS repository structure
- Working_group_OASIS reference repository
- `mkdocs.yml`
- `README.md`
- `.github/workflows/`
- `docs/`
- `docker/`

### Actions taken

- Added `AGENTS.md` with postdoc-specific agent guidance.
- Added this prompt action log.
- Updated MkDocs navigation, branding paths, Material theme settings, CSS, and ESIIL visual styling.
- Reworked the homepage around the same repository-plus-website model used by Working_group_OASIS, adapted for postdoctoral researchers.
- Added postdoc work-plan and "how this postdoc project works" pages.
- Added ESIIL and team resource pages, public-facing site guidance, cloud triangle guidance, cite/reuse guidance, and practical GitHub/storage instructions.
- Added semantic image slots, process gallery folders, generated include files, and image slot generation scripts.
- Added non-blocking site health checks and a local template integrity check.
- Moved the JupyterLab runtime folder from `docker/` to `containers/` and updated the Docker image workflow.
- Updated GitHub Pages automation to generate image references, generate the site health report, and build with strict MkDocs before deploying.
- Added local artifact ignores for `.DS_Store`, `site/`, virtual environments, Node modules, and test results.
- Removed tracked `.DS_Store` files from the template.

### Verification

- Ran `python3 scripts/generate_image_slots.py`.
- Ran `python3 scripts/site_health.py`; remaining warnings are expected template placeholders.
- Ran `python3 scripts/check_template.py`.
- Ran `/private/tmp/postdoc_oasis_mkdocs_venv/bin/mkdocs build --strict --clean`.
- Confirmed strict build succeeds after fixing legacy missing asset links in older resource pages.

### Open questions and follow-up

- Confirm whether this template should keep the older resource guides in the main navigation or leave them available under the broader resources folder only.

## 2026-10-05

### Prompt

Investigate why SSL data loading looked slow on the GPU node (12 s/sample/worker in benchmark job 47443176), how throughput scales with workers, and how to make cookie fetching and sampling faster.

### Actions taken

- Added `scripts/litePT/profile_dataloader.py`: per-transform CPU cost and DataLoader throughput vs. worker count.
- Profiled the pipeline and grouped-vs-single EPT fetches; no pipeline code was changed.

### Verification

- The 12 s figure was worker start-up (spawn + imports) amortized over only 16 samples. Steady state is ~0.33 s/sample/worker and scales linearly (7 workers: 18-21 samples/s on RM-shared).
- One PDAL pipeline per tile (multiple polygons) fetched the same points 1.4-2.7x faster than one pipeline per cookie.

### Open questions and follow-up

- Decide whether to adopt the loader optimizations (pre-thinned cookies with cached intensity rank, no deepcopy in the view generator, faster elastic distortion) and per-tile grouped fetching in `eagle_als.cache`.

## 2026-10-05 (streaming design, raster HAG)

### Prompt

Design single-node streaming of 3DEP data (500 m squares, rolling buffer on node-local NVMe) and implement a cheaper height-above-ground calculation.

### Actions taken

- Measured PDAL fetch cost vs. read size, scattered vs. contiguous reads, concurrent decode throughput, and /ocean write/read throughput; wrote the single-node streaming design into the working-notes doc (Preprocessing efficiency section).
- `src/eagle_als/fetch.py`: `height_above_ground` now defaults to `method="raster"` (1 m mean-ground grid, nearest-cell gap fill, bilinear lookup; new helper `ground_raster_z`). The previous IDW k-NN method stays available as `method="idw"`.

### Verification

- 60 cached cookies (24.7 M points): raster 0.19 vs IDW 2.08 core-s per M points; median |difference| 1 cm, 99th percentile 13 cm per cookie.
- Live fetch of 3 random pre-training locations: ground-point HAG median 1-2 cm.

### Open questions and follow-up

- No-ground-class fallback (0.1% of cookies) differs from IDW by up to ~1 m at the 99th percentile; acceptable for now.
- Build the producer and `StreamingChunkDataset` per the design.

## 2026-10-06 (streaming implementation plan)

### Prompt

Read the working notes and `src/eagle_als/`, and prepare to implement the single-node streaming design.

### Actions taken

- Reviewed `fetch.py`, `cache.py`, `data.py`, `transforms.py`, `train_ssl.py` and the SSL config against the design.
- Added an Implementation plan to the working-notes doc (files to add/change, shard format, five steps with tests). No code changed yet.

### Verification

- `sinfo`/`scontrol`: H100 nodes w001-w010 have 104 cores and 2 TB RAM; `/local` on the interactive node is 6.1 TB.

### Open questions and follow-up

- Check `$LOCAL` size on an H100 node in the first job; measure S3 concurrency limits with the producer.

## 2026-10-06 (streaming plan A vs B)

### Prompt

Document plan A (fetch inside DataLoader workers) as the chosen streaming plan with plan B (separate producer, being built by another agent) as the alternative; implement A in separate modules and profile A against B.

### Actions taken

- Working-notes doc: added "Streaming plan A (chosen)" and relabelled B's sections as the alternative.
- Added `src/eagle_als/stream_pool.py` (SquarePoolDataset, FetchHelper subprocess, RoundRobinBatches, build_pool_loader), reusing B's sampler, square fetch and cookie cutting from `eagle_als.stream`.
- Added `scripts/litePT/profile_stream_compare.py` and `scripts/litePT/slurm/stream_compare.sbatch`.

### Verification

- Job 47472580 (64 cores): A 86 samples/s and B 103 samples/s uncapped; both sustain 40 samples/s capped; distinct squares per batch 1.00 (A) vs 0.99 (B).
- Start-up (5.5-44 min) was dominated by cold imports and SquareSampler init on /ocean, before any fetch.

### Open questions and follow-up

- Job 47476010: capped reruns with a warm environment, window-only CPU, and A at 12 uses per square.
- The trainer is not switched to plan A yet (train_ssl.py belongs to the other agent's work).

## 2026-10-06 (adopt plan A, remove plan B, node-local staging)

### Prompt

Switch the training workflow to streaming plan A, remove the plan B code (it is in git history, commit 895bafe), update the planning document, and get ready for a full training run.

### Actions taken

- `src/eagle_als/stream.py` -> `squares.py` (git mv): kept the square sampler, shard format, cookie cutting and square fetch, plus the fetch-helper `serve` loop; removed the producer, eviction and shard listing. It no longer pulls in torch, so fetch helpers start fast.
- `src/eagle_als/stream_pool.py`: uses `squares.py`; `build_pool_loader` offsets the square sequence by the start step and uses `in_order=False`.
- Removed `StreamingChunkDataset` / `build_ssl_dataset` (data.py), `slurm/producer_sweep.sbatch`, `slurm/stream_compare.sbatch`, and the stream mode of `profile_dataloader.py`; `profile_stream_compare.py` -> `profile_pool.py` (pool only).
- `train_ssl.py` and config: `data_mode="pool"` is the default and builds the loader once per job; `"cache"` stays for the overfit test (overfit_test.sbatch now sets it).
- `scripts/litePT/env.sh`: `PROJ_NETWORK=OFF`. PROJ's network grid cache is one SQLite file in $HOME; fetch helpers waited on its lock over Lustre, which stalled data loading for up to an hour. Also `EAGLE_ENV_PREFIX` to use a staged environment copy.
- New `stage_local.sh` / `pack_for_local.sh`: extract the environment (one zstd archive, `pixi_envs/eagle-litept-gpu-env.tar.zst`) and evaluation cookies (`cache/sites.tar`) onto `$LOCAL` at job start.
- New `slurm/pretrain.sbatch`: full 8-H100 run (batch 128 = 8 GPUs x 8 x accum 2), stops cleanly 30 min before the time limit, resumable.
- Working-notes doc: plan A marked adopted, plan B sections removed (measurements and the square fetch check kept), node-local staging and the PROJ finding recorded.

### Verification

- Pool loader smoke test on 8 cores: first batch 15.7 s (was 40 s), clean shutdown.
- Staged environment from a different path: extraction 19 s, imports 7 s (torch, spconv, flash_attn, pdal), live square fetch OK.
- 1-GPU end-to-end smoke test: job 47479028.

### Open questions and follow-up

- /jet home is 97% full (831 MB free).

## 2026-10-06 (README project documentation)

### Prompt

Write higher-level project documentation in README.md: training design, how to monitor training, planned experiments.

### Actions taken

- README.md: filled the empty "Project description" and added "Pre-training design", "Running pre-training", "Monitoring training" (run names, TensorBoard, metrics to watch) and "Planned experiments". Website-template sections left unchanged.

## 2026-10-05 - USNVC full taxonomy

### Prompt

Convert the USNVC catalog so every row includes its higher-level designations (Biome through Alliance).

### Actions

- Added `data_cleaning/usnvc_full_taxonomy.py`, which fills down each level and clears deeper levels at each row. It checks the result against `parent elcode`.
- Output: `datasets/vegbank/USNVC_catalog_full_taxonomy.csv`.

## 2026-10-06 - VegBank class quiz notebook

### Prompt

Design a notebook that shows a few example ALS tiles for a chosen classification level (e.g. Subbiome), then quizzes the user on a fresh tile from one of the shown classes. Level and number of classes configurable (default 5).

### Actions

- Added `scripts/3dep/vegbank_class_quiz.ipynb`. Config cell sets `LEVEL`, `N_CLASSES`, `N_EXAMPLES`, `RADIUS`, etc. It streams tiles with `load_als_cookie` (cached in `~/.cache/eagle_als_quiz`), shows 3D plotly examples per class, then runs an ipywidgets quiz with score and answer reveal. Class names for Division and below come from `USNVC_catalog_full_taxonomy.csv`.
- Tested the loading and class-selection logic headlessly (plotly stubbed). The plotly and widget quiz cells have not been run.

## 2026-10-06 - Quiz notebook: remove terrain slope

### Prompt

Tiles in `vegbank_class_quiz.ipynb` look very tilted (e.g. 50 m relief on a 50 m radius tile); suspected a processing error.

### Actions

- Checked the point clouds: not a processing error. Class-2 ground points alone span up to ~36 m across a 100 m-wide tile on steep terrain. The notebook was plotting raw elevation.
- `vegbank_class_quiz.ipynb` now plots height above ground by default (`HEIGHT_MODE = 'above_ground'`), interpolated from ground points. `'elevation'` restores raw Z. Cache key bumped to `_v2` (old cached tiles are ignored).

## 2026-10-06 - Quiz notebook: color by intensity

### Prompt

Add an option to color by intensity.

### Actions

- `vegbank_class_quiz.ipynb`: new config `COLOR_BY = 'height' | 'intensity'`. Intensity is stored in the tile cache (key bumped to `_v3`) and clipped to the 2nd-98th percentile per tile, since scales differ between sensors; no colorbar is shown in that mode.

## 2026-10-06 - False-color blends of HAG, intensity, return type

### Prompt

Develop several false-color options that blend height above ground, return type (only / first / intermediate / last) and intensity, in a separate .py module.

### Actions

- Added `scripts/3dep/false_color.py` with `false_color(hag, intensity, return_type, mode)` and five variants: `hsv_return_hue`, `hsv_height_hue`, `rgb_channels`, `ternary`, `palette_shaded` (`MODES`, `DESCRIPTIONS`).
- `vegbank_class_quiz.ipynb`: `COLOR_BY` accepts any variant, tiles now cache `ReturnType` (cache key `_v4`), and a new "Color options" cell draws one tile in every option.
- Checked on real tiles by rendering side views with matplotlib; the plotly figures have not been run.
