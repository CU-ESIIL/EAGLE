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

## 2026-10-07 - BLM AIM LMF dataset summary notebook

### Prompt

Look at `datasets/raw/BLM...`, make a python notebook that loads and summarizes the dataset and provide a high level description of contents.

### Actions

- Added `scripts/blm_aim/blm_aim_lmf_summary.ipynb`: lists the 20 geodatabase layers, loads `I_Indicators`, `I_Species`, `F_POINT`, `F_POINTCOORDINATES`, and samples `F_PINTERCEPT` / `F_SOILHORIZON`, with summary tables, plots, a join example and usage notes.
- All cells were executed as a script and ran without errors; the notebook is saved without outputs.

## 2026-10-07 - BLM AIM tile viewer with measured cover

### Prompt

Read `scripts/3dep/vegbank_class_quiz.ipynb`, then make a notebook that visualizes tiles from the matched BLM AIM plots (`datasets/BLM_AIM/BLM_AIM_plots.gpkg`, same AWS product-matching columns); each tile should report the cover % from the AIM plot.

### Actions

- Added `scripts/3dep/blm_aim_tile_viewer.ipynb`. Tile loading, caching (shared cache dir) and false-color rendering are copied from the quiz notebook, keyed on `PrimaryKey` and `Latitude_WGS84`/`Longitude_WGS84`.
- Shows one row of tiles per quantile bin of a chosen cover column (`SORT_BY`), titled with state, visit date and shrub / grass / forb / tree / bare-soil cover %, plus a table of all cover columns per row and a single-plot view (`KEY`).
- Filters: `SPLIT` (train/test), `STATE`, `MAX_YEAR_DIFF`. Lives in `scripts/3dep` because `dataset.py` loads the tile index by relative path.
- Ran all cells as a script on real tiles (figures built, not displayed in Jupyter); notebook saved without outputs.

## 2026-10-07 - Convert PLOTS v4 Access database to parquet

### Prompt

The NPS PLOTS database (`datasets/raw/PLOTS_v4_Distribute_64`) is for MS Access and I'm on a Mac; convert it to parquet file(s).

### Actions

- Installed `mdbtools` via Homebrew (not on conda-forge) and added `data_cleaning/plots_accdb_to_parquet.py`, which writes one typed parquet per table of `PLOTS_v4_BE.accdb` to `datasets/PLOTS_v4/` (27 files, 2.2 MB).
- Finding: the distributed back-end is an empty template. The data tables (`tPlots`, `tPlotEvents`, `tPlotEventSpecies`, `tSpecies`, `tPhotos`, ...) have 0 rows; only the `x*_lu` lookups are populated (e.g. `xPLANTS_lu` 34,626 rows, `xVegAssociation_lu` 5,863 NVC types). `PLOTS_v4-64.accdb` is only the front end (forms/queries), with no survey data.

## 2026-10-07 - Crawl and harmonize NPS Vegetation Mapping Inventory plot data

### Prompt

From https://www.nps.gov/im/vmi-products.htm, find each park's vegetation inventory data product (park page -> IRMA DataStore profile -> open-format data package CSVs), download them, harmonize the plot data, and compile dataset questions/concerns per park plus a summary table of which products were accessible.

### Actions

- Found that the park pages and IRMA profiles are JavaScript-rendered, so used the IRMA REST API (`irmaservices.nps.gov/datastore/v8/rest/Profile/<id>`). Added `data_cleaning/nps_vmi/01_catalog.py` (141 park pages -> 824 products -> 103 "Field data ... Open Format Data Package" packages; every HTTP access logged in `catalog/access_log.csv`: 2 park pages 404, 3 profile 500s).
- `02_download.py` fetched 3,746 CSVs (401 MB) into `datasets/raw/nps_vmi/` (gitignored); a transient DNS failure was recovered by rerun (script skips existing files).
- `03_profile.py` profiled all packages; `04_harmonize_v4.py` and `05_build_core.py` (with shared `coords.py`) build `datasets/NPS_VMI/vmi_core_events|species.parquet` for 91 packages in 3 schema families (PLOTS v3/v4, NCPN `tbl*`, NER `Plots`/`Plots-Species`): ~56k events (98.7% georeferenced), ~560k species records.
- Coordinate repairs, each flagged in `qa_flags`: wrong/missing UTM zone (resolved against the park bounding box from IRMA), swapped easting/northing (GUMO), swapped lat/lon with lost longitude sign (CHAT, HOBE, OCMU, BUFF).
- `06_status_and_concerns.py` writes `catalog/package_status.csv` (all parks, with access/harmonization status) and `DATASET_CONCERNS.md` (cross-cutting concerns + generated per-park notes). `datasets/NPS_VMI/README.md` documents outputs.
- Not done: 12 custom-schema packages (9 Alaska parks, BIHO, CIRO, SCPN/WUPA); plot size/shape and stratum heights are not harmonized in the core tables; taxonomy not reconciled.

## 2026-10-07 - NPS VMI: ingest NVC community assignments, ALS-matching columns

### Prompt

Ingest community assignments for any datasets where they are provided, and make the unified parquet easy to match to ALS: community name (or null), freeform community name, year, WGS84 lat/lon, location accuracy flag.

### Actions

- `05_build_core.py`: gathers each event's final classification (PLOTS `NVC_Elcode`/`Classified_Code`, NER `NVC.ELCODE`, NCPN `tblFinalAssociationNames*`/`tblFinalClassification`/`tbl_Final_Classification`/`tblVegetation`), AA field call (`Primary_Code`), and freeform provisional names; resolves codes (and association names, matched on scientific or common name) against the USNVC catalog and rolls up to division/macrogroup/group.
- Added v3 `tAA`/`tAAEvents` accuracy-assessment points (19.6k events, no species lists), which carry many of the community calls; APPA and NATR `tAAEvents` skipped (different shape).
- Core events now ~75.8k rows; 24.8k resolve to an NVC community (was 10.4k), 28.1k have `community_name`, 41k+ have a community or freeform name.
- Added `community_name`, `freeform_community_name`, `year`, `lat_wgs84`, `lon_wgs84`, `location_accuracy_flag` (first columns); renamed `lon`/`lat`. `coords.py` now treats UTM zones outside 1-60 as missing.
- Updated both READMEs and `DATASET_CONCERNS.md` (per-park label coverage and sources).
- Caveats: park-local, provisional (CEPP/CEPS) and retired CEGL codes do not resolve; AA field calls are lower confidence (`nvc_code_source = aa_field_call`).

## 2026-10-08 - NPS VMI plot geometry assessment

### Prompt

Assess whether raw NPS VMI plot data can give a geo-referenced area per plot (GeoPackage), else add area/shape columns to the unified parquet; then update the docs only.

### Actions

- Assessed whether raw VMI plot data can give a georeferenced footprint per plot. Result: no (area/shape on ~40% of events; point-vs-center/corner undocumented; rectangle azimuth ~5%), so the unified format stays a parquet with area/shape columns.
- Documented the existing `plot_shape`, `plot_area_m2`, `plot_equiv_radius_m`, `plot_area_basis`, `plot_dims_flag`, `plot_azimuth_deg`, `gps_error_m` columns in `datasets/NPS_VMI/README.md`, and corrected the plot-size concern in `DATASET_CONCERNS.md` and its generator `06_status_and_concerns.py`.

## 2026-10-08 - LFRDB habitat-type evaluation dataset (GPS plots matched to ALS)

### Prompt

Ingest the public LANDFIRE Reference Database into datasets/raw and document how and when it was obtained; filter for GPS-based coordinates with enough decimals; match to ALS product and year with the helper functions and keep records within +/-3 years; plot summaries of community types and structure variables; drop records missing the primary evaluation column; make a 50% test holdout; export a parquet like the BBS, butterfly and OFO examples; write a dataset README.

### Actions

- Explored the public LFRDB first: coordinates are not rounded in general (about 97% of plots have 5+ decimals), but the 6,266 plots from sources flagged "YES, plot locations excluded" have 3 or fewer decimals while 95% of them are flagged GPS, so `LocMeth = G` alone is not a sufficient filter. The data dictionary's "nearest 100 seconds" wording looks like a typo.
- Added `data_cleaning/lfrdb/` (`01_download.py` to `05_finalize_eval.py`, README, `catalog/`). Downloaded nine regional `.accdb` files (2.1 GB) to `datasets/raw/LFRDB/` (gitignored) on 2026-10-08, with a provenance log (URL, UTC time, Last-Modified, sha256) in `catalog/download_log.csv`.
- Filters (539,373 -> 6,742 plots): valid coordinates, GPS, 5+ decimals, `field visit` type (added beyond the request: drops aerial, helicopter, photo-interpreted and remote visits), visit year, 3DEP coverage via `get_nearest_year_product`, collection year within +/-3 years (the big drop, since plots are mostly from about 2003), non-missing and non-`Unclassified` `ecosys` label.
- Wrote `datasets/LFRDB_eval/lfrdb_eval.parquet` (primary label `ecosys`; cover regression columns; ALS columns; `test_split` by 1-degree blocks, 50.1% test), `ecosys_split_counts.csv`, six figures and `README.md`.
- Caveats noted in the README: only 18 of 222 classes have 20+ plots in both splits; heights are almost always null; plot size is unknown; no point-cloud streaming check was run; 68% of rows are NPS plots.

## 2026-10-08 - NLCD land-cover evaluation task for lidar representations

### Prompt

Develop a simple evaluation task for lidar representations using NLCD classes as the target, preferably from manually verified points rather than modeled NLCD; table of target, coordinate and matching ALS product, with the label year matching the ALS collection year. Follow-ups: match column names of other datasets, more classes the better plus a coarser level, CONUS, stream 3DEP only to check availability without saving anything, and keep only rows where the label year equals the 3DEP collection year (widen to +/- 1 year only if under 1k rows).

### Actions

- Used the NLCD 2021 accuracy-assessment reference points (3,245 CONUS 30 m pixels, analyst-interpreted, primary + alternate Level II labels for 2016, 2019, 2021 with imagery dates); raw files in `datasets/raw/NLCD_AA2021/` (gitignored).
- `scripts/nlcd_eval/01_build_table.py` matches each point to the nearest-year 3DEP product (`product_name_AWS`, `collection_year_AWS`, `year_diff_AWS`, `als_site_id`, `test_split` as in other datasets), keeps `year_diff_AWS == 0` (1,089 rows, 16 classes + Level I column), and flags a class-balanced subset (988 rows, max 100 per class, max 10 per product).
- `scripts/nlcd_eval/02_check_streaming.py` streamed a 50 m crop for 1,006 rows including every balanced row (nothing cached): 956 ok, 49 empty, 1 transient S3 error; resumable, appends results (an uncaught PDAL error killed the first version before it saved).
- Wrote `datasets/NLCD_eval/` (parquet, status csv, README). Caveats: only 3,245 reference points exist so most classes have under 100; perennial ice/snow has 1 point.

## 2026-10-06 (eagle_als package README)

### Actions taken

- Added `src/eagle_als/README.md`: module map, per-module purpose and entry points, cookie / shard / sample formats, conventions (coordinates, HAG, crash isolation, evaluation-site exclusion, PROJ offline), and how to add transforms or datasets.
- `cache.py` docstring now points to that README (it referenced a non-existent `scripts/litePT/README.md`).

## 2026-10-06 (pretrain smoke test)

### Verification

- Job 47479028 (1 H100 on w004, 12 CPUs): staging to /local worked (environment 16 s, evaluation cookies 207 s); 220 steps of batch 128 (8 x accum 16) at ~13 s/step, data_frac ~0.06; clean timed stop with checkpoint; 1,184 squares fetched (69 empty, 2 errors), retired after a median 24 cookies.
- Exported RUN_NAME/BATCH did not reach the job, so it ran as ssl_s_pool_v1; its output was moved to runs/ssl_s_pool_smoke_47479028. `pretrain.sbatch` now reads run_name= and batch_size= from its arguments; README updated.

## 2026-10-06 (learning-rate warmup)

### Actions taken

- `configs/ssl_litept_s.py`: `warmup_steps = 2_500` (linear from 0 over the first 5% of 50k steps, then constant 1e-3). Previously there was no learning-rate warmup. The overfit test with a 100-step warmup (job 47450794) reached loss 7.59 vs 7.80 without (job 47450403).
- README: noted the schedule.

## 2026-10-06 (TensorBoard in VS Code)

### Actions taken

- Root `pixi.toml`: `tensorboard` and `torch-tb-profiler` (conda-forge) for VS Code's TensorBoard integration; removed a duplicate PyPI entry for `torch-tb-profiler`.

### Verification

- VS Code's own probe (`pixi run ... get_output_via_markers.py`) returns tensorboard 2.21.0 and torch_tb_profiler 0.4.3, which meet its requirements (>= 2.4.1, >= 0.2.0); TensorBoard served the smoke-test run's scalars from this environment.
- Root cause of VS Code's "TensorBoard is required" prompt: pixi printed a warning on stderr about its repodata cache being on a network filesystem ($HOME), and VS Code treated it as a failed probe. Added `~/.pixi/config.toml` with `[cache] repodata = "/tmp/pixi-cache-sammlapp/repodata"`; VS Code's probe now has empty stderr.

## 2026-10-07 (pre-training on any number of GPUs)

### Actions taken

- The 8-H100 job was waiting in the GPU queue (Priority), so pre-training now adapts to the GPUs and CPUs a job gets, for 1-2 GPU runs on GPU-shared.
- `configs/ssl_litept_s.py`: `grad_accum = "auto"`, `num_workers = "auto"`, new `max_batch_per_gpu = 8`.
- `train_utils.resolve_batching` (called from `train_ssl.py`): keeps the global `batch_size`; grad_accum = fewest micro-batches with at most 8 samples per GPU (8 GPUs: 2, 2 GPUs: 8, 1 GPU: 16); workers per GPU = (cores per GPU - 1) // 2 in pool mode (6 on a full node, as before). Explicit values still override. Clear error when batch_size does not split over the GPUs.
- `slurm/pretrain.sbatch`: no longer computes grad_accum; usage lines for 2 GPUs (`--cpus-per-task=26`) and 1 GPU (13 CPUs per GPU). README updated.

### Verification

- Resolver checked with the config for 8 / 2 / 1 GPUs (104 / 26 / 13 cores), 1 GPU with batch_size=16, an explicit grad_accum, cache mode with explicit num_workers, and 3 GPUs (error). `bash -n` on the sbatch. Not yet run on GPUs.

## 2026-10-08 (reading the ssl_s_2gpu curves; probe evaluation during training)

### Prompt

User asked how to interpret the TensorBoard curves of `ssl_s_2gpu` (loss flat after ~2k steps, `mem_gb` stepping up every hour or two), whether to lower the mask ratio and size, and to prepare a periodic classification probe on a fixed labeled eval set.

### Findings (ssl_s_2gpu, 2 H100s, steps 0-3780; last checkpoint step 3000)

- `loss` = `target_entropy` + `mask_kl`. The drop from 8.1 to 6.7 is almost all `target_entropy` (7.67 to 6.4); `mask_kl` stays at 0.25-0.33 and drifts down slowly (0.31 at steps 1500-2500, 0.26 at 3000-3800).
- The small loss rise at steps 1750-2500 matches the rise in `target_entropy` from the teacher-temperature ramp (0.04 to 0.07). `mask_kl` did not rise when the mask ramp ended, and `unmask_loss` (local views, never masked) flattened at the same time, so harder masking is not what flattened the loss.
- `mem_gb` was `max_memory_allocated()` since the job started, so it can only step up (on rare large batches, up to ~510k points against a ~310k median). It has been flat at 51.3 GB since step 2300. Not a leak.
- The run predates the Sinkhorn memory bank (no `sinkhorn_cookies` in its config).

### Actions taken

- New `scripts/litePT/TENSORBOARD.md`: how to read every logged metric. `train_ssl.py` writes it to each run's TensorBoard Text tab; also added to `ssl_s_2gpu/tb` as a separate event file.
- README "What to watch" table revised (flat loss is expected; judge runs on probes) and linked to the guide; new "Probe evaluation during training" paragraph.
- `train_ssl.py`: `mem_gb` is now the peak since the previous log line, plus new `mem_max_gb` (peak since the job started).
- New `src/eagle_als/probe.py`: fixed labeled cookie sets (table + cookie cache), one deterministic view per cookie prepared at startup, mean-pooled up-cast teacher features, logistic regression + cosine kNN (balanced accuracy, macro F1; train/test split column or 5-fold CV).
- `train_ssl.py`: with `eval_sets` in the config, rank 0 runs the probes every `eval_every` steps (default 1000) and at step 0 of a fresh run; the other ranks wait at a barrier. Logged as `eval/<set>/*`. Config keys `eval_sets = []`, `eval_every`, `eval_knn_k`.
- `check_embeddings.py` now imports `eval_transform` from `eagle_als.probe`.

### Verification

- 1-GPU smoke job 48795262 (cache mode, 40 steps, a 24-cookie dummy set with 3 cyclic labels): probes ran at steps 0, 20 and 40 (34 s at step 0 including CUDA warm-up, under 1 s after); `eval/*` scalars, `mem_gb` / `mem_max_gb` and the `guide` text appear in TensorBoard; training continued in train mode. Test run and label file deleted afterwards.
- On the command line, `eval_sets` needs dict literals (`{'name': ...}`), not `dict(...)`.

## 2026-10-08 (NLCD land cover as an in-training validation task)

### Prompt

User asked to add the NLCD_eval classification task as a periodic in-training check of the frozen feature extractor (linear probe and kNN, macro and per-class F1 in TensorBoard), in a way that makes it easy to add more evaluation tasks; then asked to score existing checkpoints on it.

### Actions taken

- New `src/eagle_als/evaluation.py`: task registry (`EvalTask`, `@register`), `build_tasks` from the config list `eval_tasks`, `run_tasks` (eval mode, no gradients, train mode restored), `tb_tag` (headline metrics `eval/<task>/*`, detail metrics such as per-class F1 in `eval_<task>/*`).
- `src/eagle_als/probe.py`: `ProbeSet` / `run_probes` replaced by the registered `classification` task `ClassificationProbe`: relative table paths from the repo root, `query` row filter, boolean or train/test `split_col`, `probes` choice, per-class F1 (`<probe>_f1/<class>`); macro F1 averages the classes present in the test rows.
- Config: `eval_sets` renamed `eval_tasks`; default task `nlcd` (label `nlcd_class`, `query="balanced_subset"`, `split_col="test_split"`), cookie cache chosen by `_cache_dir` (node-local copy if staged).
- Cookie cache `$EAGLE_SCRATCH/cache/nlcd` built for all 1,089 rows (1,035 ok, 51 empty, 3 too few points; 7.0 GB) and packed as `cache/nlcd.tar`; `slurm/pretrain.sbatch` stages it (`stage_local.sh sites nlcd`).
- `squares.EVAL_SITES` now includes the NLCD table (excluded sites 3,229 -> 4,318). Runs started before this change may have seen NLCD sites.
- New `scripts/litePT/eval_checkpoints.py`: scores a run's saved checkpoints (and a random init) on the eval tasks, writes `<run>/eval_checkpoints.csv` and optionally TensorBoard.
- README, `TENSORBOARD.md`, `src/eagle_als/README.md` updated.

### Verification

- Task build on a CPU node: 942 of 988 balanced rows usable, 15 classes (perennial ice/snow dropped), 726 train / 216 test, 74 s and 3.3 GB on rank 0.
- 1-GPU smoke job 48873059 (pool mode, 40 steps, `eval_every=20`): NLCD cache staged in 18 s; evaluation at steps 0/20/40, 6-7 s each; training continued.
- Job 48939251 scored checkpoints (linear macro F1, test split): `ssl_s_2gpu` init 0.285, 1000 0.318, 2000 0.341, 3000 0.302; `ssl_s_1gpu_test` init 0.281, 500 0.245, 1000 0.244, 1500 0.253, 2000 0.238. kNN macro F1 stays at or below init (0.21) in both runs. Chance 0.067.
- GPU-shared rejects `--cpus-per-task=13` per GPU (maximum 12).
- Follow-up: user set `eval_every = 300` (was 1000), since one evaluation takes about 7 s; at batch 128 that is about every 40 min on 2 H100s.
