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
