# Baseline benchmarking: working doc

Claude: use this doc to keep notes and look for my feedback during this session.

Operational state (machines, environments, paths, commands to resume) is in `CLAUDE.md`; this doc holds decisions,
open questions, results and the dated log. Newest first in each log.

## Open questions for the user

- **Tiling vs capping for BBS (150 m) and SatBird (300 m)** in a later run (proposal below); this run caps at 100 m.

## Next steps

1. ForPT probe (embeddings done at all four radii): `02_linear_probe.py --embeddings '$WORK/benchmarking/embeddings/forpt/r{radius}_z' --model forpt`.
2. lidR stdmetrics: **job 23829139 failed** in r30 on a terra error in the terrain metrics
   (`[terrain] too few values for writing: 16 < 64`, a cookie with very few ground points). Fix `lidr_metrics.R` (guard
   the terrain step, NaN when too few ground points), then resubmit; r15 is done, r30 6,041 of 6,099, r50/r100 not
   started. Then linear probe and gradient boosting.
4. AlphaEarth diagnostics (plan at the end), tests 1, 3, 5 first.
5. Later: lidar-nuisance-only baseline (point density, year, quality level: are labels confounded with acquisition?),
   AEF + lidar metrics concatenated (does lidar add anything beyond the satellite embedding?).

## Decisions (user)

- **Radius (2026-11-12):** one radius per task for every area-based feature set (point-only ones, AEF centre pixel
  and location/SatCLIP, excepted), following the task's `radius` in `datasets/registry.yaml`; tables without one use
  the documented 50 m default.
- **Lidar capped at 100 m for this evaluation run (2026-11-12):** ForPT and lidR features use each task's registry
  radius capped at the cached cookie's 100 m (`eagle_bench.tasks.lidar_radius`): nlcd 15 m; nps_vmi, ofo_trees 30 m;
  lfrdb, vegbench, butterflies, blm_aim 50 m (default); snapshot_usa 100 m; **bbs 100 m (registry 150 m) and
  satbird_winter 100 m (registry 300 m) are capped**, so on those two tasks lidar sees a smaller area than Clay NAIP.
  Features are computed from the points in the disc only (crop, then embed / compute metrics). Output dirs per radius
  (`r15`, `r30`, `r50`, `r100`); the probe picks each task's with `--embeddings '<dir>/r{radius}...'`.
- **One linear probe for everything (2026-11-12):** `eagle_als.linear_probe`, a single torch `nn.Linear` on standardized
  features, full-batch L-BFGS from zero weights; class-balanced softmax (multiclass), per-target sigmoid BCE
  (multilabel), squared error on standardized targets (regression); NaN labels masked; `l2/2 ||W||^2` with l2 from
  1e2 to 1e-5 chosen by 3-fold CV on train (balanced accuracy / mean AUROC / mean R2). Consistency across experiments
  matters more than matching sklearn. Used by the benchmark (`eagle_bench.probes`) and the pre-training evaluation
  (`eagle_als.probe`, `probe_table`). Synthetic check vs sklearn: bal. acc. 0.510 vs 0.506, AUROC 0.925 vs 0.925,
  R2 0.962 vs 0.962.
- **Lidar derived metrics (2026-11-10):** `lidR::stdmetrics` (Roussel et al. 2020) on height above ground in the
  analysis disc, with and without a few terrain metrics (elevation, slope, aspect from ground points; TWI from a
  provider). No intensity, no GEDI simulation. Probes: linear and gradient boosting. `lidRmetrics` (Tompalski et al.)
  is a later candidate. Reason: standard feature sets over custom code.
- **NAIP imagery (2026-11-11):** Clay v1.5 NAIP embeddings, averaged over a disc of the task's radius; nothing but the
  pooled vector per point is stored. DINOv3-Sat is a possible later option.
- **Official implementations (2026-11-11):** use the authors' code (ForPT, SatCLIP), cloned or installed outside the
  repo, rather than re-implementations.
- **Sites at collection edges (2026-11-12):** accepted and documented, not rematched (see Known data issues).

## Proposal: lidar embeddings for radii above 100 m (Claude, 2026-11-12, for review)

The cache holds 100 m-radius cookies; ForPT / lidR / QC currently cut a fixed 50 m disc from them. lidR metrics on a
larger disc only need more points (download). The point encoders (ForPT, EAGLE) are trained on 100 m cookies, so a
300 m area in one pass is out of distribution and ~9x the memory. Proposal:

1. The unit of embedding is always a training-sized cookie (100 m radius); centres on a square grid of spacing s over
   the task disc (s = 100 m: every point within 71 m of its nearest centre, so >= ~30 m of context).
2. Each point takes the up-cast encoder features (concatenated stage features of the voxels it belongs to; no decoder
   needed, this is what the cookie mean already pools) from its nearest-centre cookie (square cells).
3. Mean over points per 5 m cell, then over the cells inside the exact disc (area weighting instead of point-density
   weighting; no partial-square weights).

For radius <= ~30 m this is one centred cookie, as now. Cost for SatBird at 300 m: ~35-40 cookies/site, ~1.5 M passes,
~75 GPU-h at the measured 5.5 cookies/s (over the 30 GPU-h threshold; s = 141 m halves it), and the 400 m read around
each site is several TB, so it would stream (fetch, embed, keep only the pooled vector) like the NAIP fetch.
Suggested order: ForPT / lidR now at min(task radius, 100 m) from the existing cache (13 of 15 tasks exact, BBS and
SatBird labelled as capped); then the tiled path for BBS and SatBird, first checked on SnapShot (100 m: one cookie vs
tiles).

## Feature sets

| model | input | dim | state |
|---|---|---|---|
| AlphaEarth (AEF) | annual embedding, centre pixel (Earth Engine) | 64 | **done** (torch probe) |
| Location (RFF) | lat/lon, random Fourier features at several km scales | | **done** (torch probe) |
| SatCLIP L10 / L40 | lat/lon, official package + Hugging Face checkpoints | 256 | **done** (torch probe) |
| Clay v1.5 NAIP | NAIP chip embeddings averaged over the task disc | 1024 | **done** (results below) |
| ForPT | 3DEP cookie, task disc (<= 100 m), frozen LitePT-S teacher | 1008 | embedded at all radii (r15 875, r30 6,061, r50 25,227, r100 51,533); probe not run |
| lidR stdmetrics | 3DEP cookie, HAG in the task disc (+ 7 terrain columns) | 56 (+7) | r15 done (875), r30 6,041 of 6,099; job failed (see Next steps) |

Notes per model:

- **AlphaEarth:** trained on Sentinel-1/2, Landsat, GEDI, DEM, climate and geotagged text. Year = survey year clipped to
  2017-2025. Validated against the COG files (cosine median 1.000). Points on Delta: `$WORK/benchmarking/aef/points_ee`
  (repo symlink `datasets/raw/aef`).
- **ForPT** (`prs-eth/ForPT`, `epoch_30.pth`, LitePT-S 12.4M params): the teacher backbone loads into EAGLE's vendored
  LitePT with every key matching. Recipe from ForPT's linear-probe configs: xyz only, CenterShift, 5 cm grid, all 5
  encoder stages up-cast and concatenated, mean over points. Domain gap: pretrained on dense forest scans (ALS/ULS/MLS);
  3DEP is 2-30 pts/m² and mostly not forest, so expect it to do best on forest-structure tasks.
- **Clay NAIP:** one 1024-d embedding per non-overlapping 256x256 px chip: 256 m at 1 m NAIP (2010-2017), ~154 m at
  60 cm (2016+). Discs below ~150 m return the chip under the point, not centred on it. Year = NAIP year nearest the
  survey year (clipped to 2010-2021). File of a point = arithmetic on lat/lon (USGS quarter-quads), no STAC search.
  Code: `src/eagle_bench/naip_clay.py`, `scripts/baseline_eval/09_fetch_naip_clay.py`.

## Results

### Comparison, torch probe (test split, min_count=20)

Combined table: `scripts/baseline_eval/results/summary.md` (`07_unified_summary.py`); per model
`results/<model>/summary.md`. AEF, location and SatCLIP use the centre point; Clay NAIP the task disc. ForPT and lidR to
add.

| task | metric | AlphaEarth | location RFF | SatCLIP L10 | SatCLIP L40 | Clay NAIP |
|---|---|---|---|---|---|---|
| nlcd | bal. acc. | **0.738** | 0.306 | 0.402 | 0.396 | 0.606 |
| lfrdb_ecosys | bal. acc. | **0.571** | 0.283 | 0.244 | 0.220 | 0.370 |
| lfrdb_nvc_group | bal. acc. | **0.437** | 0.146 | 0.179 | 0.198 | 0.264 |
| nps_vmi_macrogroup | bal. acc. | **0.466** | 0.271 | 0.245 | 0.285 | 0.365 |
| nps_vmi_group | bal. acc. | **0.577** | 0.197 | 0.272 | 0.305 | 0.395 |
| vegbench_formation | bal. acc. | **0.519** | 0.227 | 0.236 | 0.237 | 0.412 |
| vegbench_division | bal. acc. | **0.462** | 0.188 | 0.180 | 0.211 | 0.382 |
| vegbench_macrogroup | bal. acc. | **0.404** | 0.144 | 0.170 | 0.166 | 0.330 |
| vegbench_group | bal. acc. | **0.344** | 0.116 | 0.148 | 0.140 | 0.237 |
| bbs_occurrence | mean AUROC | **0.895** | 0.858 | 0.870 | 0.860 | 0.863 |
| butterflies_occurrence | mean AUROC | 0.664 | **0.715** | 0.704 | **0.715** | 0.683 |
| snapshot_usa_occurrence | mean AUROC | **0.806** | 0.761 | 0.727 | 0.760 | 0.752 |
| satbird_winter_occupancy | mean AUROC | **0.927** | 0.897 | 0.901 | 0.905 | 0.915 |
| blm_aim | mean R2 | **0.511** | 0.192 | 0.169 | 0.231 | 0.458 |
| ofo_trees | mean R2 | **0.295** | 0.291 | -0.026 | 0.243 | 0.249 |

AlphaEarth is best on 14 of 15 tasks; on butterflies, location alone beats every image embedding. Location-only is
close to AEF on the bird/mammal occurrence tasks (AUROC within 0.03-0.05) and on OFO, far below it on the vegetation
classifications. Compared with the old sklearn probes most scores moved by a few hundredths either way (AEF NLCD 0.751
-> 0.738; SatCLIP L40 lfrdb_ecosys 0.290 -> 0.220), the exception being location RFF on OFO (-0.004 -> 0.291).

### Clay NAIP, all tasks (torch probe, 2026-11-12)

`scripts/baseline_eval/results/naip_clay/summary.md`.

| task | radius | metric | score | chance / prevalence |
|---|---|---|---|---|
| nlcd | 15 m | bal. acc. | 0.606 | 0.125 |
| lfrdb_ecosys | 50 m* | bal. acc. | 0.370 | 0.045 |
| lfrdb_nvc_group | 50 m* | bal. acc. | 0.264 | 0.037 |
| nps_vmi_macrogroup | 30 m | bal. acc. | 0.365 | 0.042 |
| nps_vmi_group | 30 m | bal. acc. | 0.395 | 0.043 |
| vegbench_formation | 50 m* | bal. acc. | 0.412 | 0.091 |
| vegbench_division | 50 m* | bal. acc. | 0.382 | 0.062 |
| vegbench_macrogroup | 50 m* | bal. acc. | 0.330 | 0.037 |
| vegbench_group | 50 m* | bal. acc. | 0.237 | 0.024 |
| bbs_occurrence (163 spp.) | 150 m | mean AUROC / AP | 0.863 / 0.197 | AP 0.037 |
| butterflies_occurrence (51 spp.) | 50 m* | mean AUROC / AP | 0.683 / 0.577 | AP 0.408 |
| snapshot_usa_occurrence (37 spp.) | 100 m | mean AUROC / AP | 0.752 / 0.287 | AP 0.128 |
| satbird_winter_occupancy (467 spp.) | 300 m | mean AUROC / AP | 0.915 / 0.483 | AP 0.109 |
| blm_aim (12 targets) | 50 m* | mean R2 | 0.458 | |
| ofo_trees (2 targets) | 30 m | mean R2 | 0.249 | |

\* no radius in the registry, 50 m default (no effect for Clay below ~130 m). NAIP years were clipped to 2021 for all
SatBird rows, 67% of OFO and 45% of LFRDB. 87,298 of 87,749 points have an embedding (451 touch no NAIP cell, 350 of
them LFRDB plots, probably outside CONUS). OFO needed the strong end of the l2 grid (l2 0.1 gave R2 -1.2).

## Known data issues

Details and paths in `CLAUDE.md` (Known data issues).

- **Noise points:** the old PDAL filter `Classification![7:7],Classification![18:18]` dropped nothing (PDAL ORs ranges
  on one dimension). Fixed 2026-11-12 with `eagle_als.fetch.NOISE_FILTER` (`filters.expression`), also used by the
  pre-training data code and `scripts/3dep/dataset.py`. Most of the Delta cache was fetched before the fix, so the
  feature scripts keep dropping classes 7/18.
- **Sites at collection edges:** matching uses footprint polygons, which are coarser than the point clouds, so a site
  can get a cookie with points on one side only or none in the disc. `10_cookie_qc.py` flags cookies with fewer than
  50% of the 3DEP minimum points (QL0/1 8, QL2 2, QL3/unknown 0.5 pulses/m² x disc area): 1,702 of 82,160 (SatBird
  3.2%, others <= 1.5%), 436 with an empty disc. 1,263 of the flagged cover < 50% of the disc's 5 m cells (edges);
  208 cover >= 90% but are sparse (mostly open water, a real signal), so flagged cookies are not dropped automatically.
  Empty discs give no ForPT / lidR features (`n_no_embedding` in the probe).
- Open water and some urban cookies have no ground points: terrain metrics NaN, lidR `zentropy` NA on flat cookies;
  the probe fills NaNs with the train median.

## Log

### 2026-11-24 (NCSA Delta, status check)

- **Torch-probe reruns done** (2026-11-12, 07:45-08:37): AlphaEarth centre, location RFF, SatCLIP L10/L40; results
  table above (results files committed 2026-11-24). Logs `$WORK/benchmarking/logs/probe_*.log`.
- **ForPT embeddings done** at all task radii: job 23829140 (r15/30/50, 58 min, ~7.7 cookies/s) and 23829141 (r100,
  6.4 h, ~2.25 cookies/s). 131 cookies failed with an empty disc (`zero-size array`; 7 r15, 38 r30, 84 r50, 2 r100),
  as expected from the edge-site QC; they count as `n_no_embedding`.
- **lidR job 23829139 failed** after 2 min (it waited ~a day in the queue, ran 2026-11-13 13:35): r15 done, r30 stopped
  on `[terrain] too few values for writing` in `lidr_metrics.R`; one failed batch kills the whole run.

### 2026-11-12 (NCSA Delta)

- **ForPT and lidR full runs submitted at the task radii (capped at 100 m).** lidR on 100 m discs costs ~7 core-s per
  cookie vs ~1.2 at 50 m (smoke test of 40 cookies per radius), so ~125 core-h in all: one 96-core job, 3 h limit.
  ForPT on two A100s: r15/30/50 (~32k cookies, 4 h limit) and r100 (~54k cookies, 16 h limit); at most 20 GPU-h.

- **Torch-probe reruns** of AlphaEarth centre, location RFF and SatCLIP L10/L40 started (AEF points copied from
  CyVerse by the user; SatCLIP in its own pixi env `envs/satclip`, checkpoints in `$WORK/benchmarking/satclip_weights`).
- **Speed tests:** ForPT on one A100 ~5.5 cookies/s, GPU busy ~47% (CPU preprocessing limits it) -> ~4-5 GPU-h for
  82k cookies. lidR stdmetrics ~1.4 s/cookie in R single-threaded; batches in parallel (`05_lidr_metrics.py
  --workers 28`) give 14 cookies/s, ~1.6 h for all cookies.
- **Cookie coverage QC** (`10_cookie_qc.py`), see Known data issues. Found because 9 of the first 1,500 ForPT cookies
  had no point in the disc (points only 87-100 m to the north of the site).
- **Clay NAIP fetch and probe**, all tasks: 32,593 files (253 GB streamed, nothing stored) in 14 min at ~40 files/s,
  0 failed; probe 40 min on 32 CPUs (torch). NLCD 0.606 vs 0.631 with the old sklearn probe and 0.636 in the Mac trial.
- **Lidar eval cache done** in `$WORK/benchmarking/als_cache`: 86,150 sites -> 82,160 cookies, 3,892 empty, 98 too
  few points, 0 errors (the 46 sites rejected for a 3-4 km elevation range were noise; all fetched after the filter
  fix). ~11 cookies/s with 64 workers on 32 CPUs (1.1/s on CyVerse), ~650 GB.
- **PDAL thread oversubscription:** `readers.ept` sizes its thread pool from the node's 128 hardware threads (load 517,
  rate 11 -> 7 cookies/s). `EAGLE_PDAL_THREADS` (set to 4 by `run_lidar_cache.sh`) fixes it.

### 2026-11-11

- Clay NAIP chosen; code written and validated (radius 0 = containing chip; 40 random points at 100/250 m: coverage
  1.00). NLCD-only trial on the Mac: bal. acc. 0.636 (AlphaEarth 0.751). Download from the Mac was network-capped
  (~1.3 files/s), so the full run moved to a cluster.

### 2026-11-10

- AlphaEarth centre pixel extracted through Earth Engine for all tasks (~7 min for 95k points; the COG route would read
  ~1 TB) and probed with the old sklearn probe. Re-extracted after SatBird changed to the USA winter task.
- ForPT embedder written and smoke-tested; lidR stdmetrics pipeline trialled on 20 cookies (56 stdmetrics + 7 terrain
  columns; heights < 0 clamped to 0).
- Lidar cache started on CyVerse (~1.1 cookies/s); moved to Delta on 2026-11-12.

## Why does AlphaEarth classify so well? (diagnostics plan, 2026-11-10, not run)

Two hypotheses: (H1) the embedding mostly encodes geographic position / regional context; (H2) label information leaked
in through AEF's training data (secondary summaries of the AEF paper list GBIF species records and geotagged Wikipedia
as text sources, alongside Sentinel-1/2, Landsat, GEDI, DEM, ERA5-Land, GRACE; to confirm in arXiv 2507.22291).

Tests for H1 (geography):
1. *Location-only baselines* on the same splits (location RFF and SatCLIP, now being rerun). AEF minus location-only =
   what AEF adds beyond position.
2. *AEF + location* vs location: incremental gain; and AEF residualized on location features.
3. *Displacement test* (cheap with Earth Engine): use AEF from points moved 30 m, 100 m, 500 m, 2 km, 10 km in a random
   direction at test time (train on true positions). Scores that survive 2-10 km displacement are regional context,
   not the site itself; the decay curve shows the spatial scale each task is really using.
4. *Within-region discrimination*: score only among test sites within the same ~50 km block (or pairs of nearby sites
   with different labels). Geography cannot separate neighbours; site-level information can.
5. *How much position is in the embedding*: predict lat/lon from AEF on the spatial hold-out (R², error in km).

Tests for H2 (training-data leakage):
6. *Map-vs-truth test (NLCD)*: NLCD reference points carry a field/photo-interpreted reference class; the NLCD 2021 map
   class at the same pixel sometimes disagrees. On the disagreeing points, does AEF predict the map class or the
   reference class? Predicting the map's errors means it has learned a map product rather than the land cover.
7. *Bird tasks vs GBIF*: eBird (SatBird) and possibly BBS records flow into GBIF. Compare AEF's gain over location-only
   on bird occupancy with its gain on tasks whose labels are not in GBIF (BLM AIM, OFO, NPS VMI plots, SnapShot only
   partly). A much larger gain for GBIF-sourced labels is consistent with leakage (not proof).
8. *Year test*: AEF embeddings of the survey year vs other years (2017 vs 2024). Static geography gives the same score in
   every year; real land-cover signal should degrade where the site changed.

Proposed order: 1, 3, 5 (cheap, most informative), then 6, then 2, 4, 7, 8.
