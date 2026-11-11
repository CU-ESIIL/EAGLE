Claude: use this doc to keep notes and look for my feedback during this session. 

Next steps:
cache the lidar tiles for the evaluation tasks
embed tiles with ForPT and evaluate linear probing on the benchmark https://github.com/prs-eth/ForPT
generate a suite of lidar summary metrics / derived products for each tile and evaluate linear probing + random forest on the benchmark

## Feature sets

### AlphaEarth
AlphaEarth is an embedding model that was trained on satellite imagery, geo-referenced text, Radar/SAR, thermal landsat, GEDI lidar, climate variables; at inference time 
center-point embedding, 64 dim -> linear probing
accessed via google earth engine
cached in `/home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking/aef/points_ee/` (repo path `datasets/raw/aef/points_ee` via symlink)

### ForPT Embeddings
The ForPT model pre-trains LitePT Point Transformer self-supervised, and demonstrated better transfer to forest ecology tasks than task-specific models. It can be fine-tuned, but here we use it only as a frozen backbone for feature extraction. 
Lidar point cloud cookie -> 1d embedding vector

*Claude notes (2026-11-10):*
- Pretrained backbone: `prs-eth/ForPT` on Hugging Face, `00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth` (LitePT-S, 12.4M params), cached in `benchmarking/forpt_weights/`. Its teacher backbone loads into EAGLE's vendored LitePT with every key matching, so no ForPT code is needed.
- Recipe copied from ForPT's own linear-probe configs: xyz only (no intensity / returns), CenterShift (z from the lowest point), 5 cm grid, features of all 5 encoder stages up-cast and concatenated (1008-d), mean over points.
- Our choices (to compare): crop radius (default 50 m disc, like the EAGLE probes), z = elevation (ForPT) vs height above ground.
- Domain gap: ForPT pretrained on dense forest scans (ALS/ULS/MLS); 3DEP is 2-30 pts/m² and covers non-forest too. Expect it to do best on forest-structure tasks (OFO, forest VegBench classes).
- Code: `src/eagle_bench/forpt.py`, `scripts/baseline_eval/04_embed_forpt.py`. Smoke-tested on synthetic clouds; waiting for the lidar cache (needs PDAL).

### Lidar Derived Metrics

**Decision (2026-11-10, user): use `lidR::stdmetrics`** (Roussel et al. 2020), on height above ground in the central 50 m
disc, with and without a small set of lidR terrain/ground metrics (elevation, slope, aspect from the ground points or fetched from a provider; TWI fetched from a provider).
No GEDI simulation. `lidRmetrics` (Tompalski et al.) kept as a future candidate. The custom proposal below is superseded
except for the location-only and acquisition-only baselines, which are kept. 

<details><summary>Superseded: custom-metrics proposal</summary>

**Claude proposal for review (2026-11-10).** Hand-crafted structure metrics per cookie, computed from the cached points
(central 50 m disc by default, with 25 m and 100 m versions to test scale), then linear probe and random forest / gradient
boosting on the same splits. About 120 features; all from xyz, height above ground (HAG), return numbers and class.

1. Height distribution (points with HAG > 2 m, i.e. vegetation; also first returns only): percentiles p05-p99, max, mean,
   SD, CV, skewness, kurtosis, canopy relief ratio (mean - min) / (max - min). Standard area-based forest inventory metrics,
   the counterpart of `lidR::stdmetrics`.
2. Cover and gaps: share of first returns above 1, 2, 5, 10, 20 m (canopy cover at several thresholds); gap fraction;
   from a 1 m canopy height model (CHM): share of cells < 2 m, size of the largest gap.
3. Vertical profile: share of returns in height bins (0-0.5, 0.5-1, 1-2, 2-5, 5-10, 10-20, 20-30, > 30 m), foliage
   height diversity (Shannon entropy of the bins), vertical complexity index, leaf/plant area density per bin and total
   plant area index (MacArthur-Horn gap-fraction method).
4. Understory: share of returns in 0.5-2 m and 2-5 m among returns that reach below that height (relative density,
   corrects for occlusion by the canopy).
5. Canopy surface texture (1 m CHM): mean, SD, rumple index (surface area / ground area), local maxima per ha
   (tree density), crown size distribution from a simple watershed or local-maximum segmentation.
6. Terrain (ground points, 1 m DEM): mean elevation, slope, northness / eastness, curvature, topographic position
   (cookie centre vs. mean), roughness. Terrain is a big driver of vegetation and birds and is not in ForPT's input.
7. Returns: share of single returns, mean number of returns, share of last-of-many returns (penetration).
   Intensity left out by default (uncalibrated across 3DEP collections); a within-cookie-ranked intensity version is optional.
8. Class shares where reliable: ground, building (6), water (9) fractions.

Additions: Topographic Wetness Index TWI (derived from a larger-area DEM than our tiles, but I imagine we can just pull the TWI directly as a derived product from somewhere)

Covariates reported but not used as features (for confound checks): point density, pulse density, collection year,
quality level, leaf-on/off, years between lidar and survey.

**Baselines worth adding alongside** (cheap, and important for interpreting everything else):
- *Location only*: lat, lon (and elevation) with the same probes / random forest. With spatial hold-outs this shows how
  much of each task is geography rather than structure. SatCLIP (above) is the learned version of this.
- *Lidar nuisance only*: point density, collection year, quality level. If these score well, the labels are confounded with
  acquisition, and the lidar features need checking.
- *AEF + lidar metrics* concatenated: does lidar add information beyond the satellite embedding?

Implementation: numpy / scipy on the cached cookies, CPU only (~0.2-1 s per cookie: 5-20 h on 4 CPUs for 86k cookies;
run per task, small tasks first). Random forest: scikit-learn `RandomForestClassifier/Regressor` (500 trees, class-balanced),
hyperparameters fixed or tuned by CV on train only, like the linear probes.

</details>

*Questions for you (still open):* (a) is the 50 m disc the right default scale (plot data vs. BBS routes / eBird hotspots, which are much
larger)? (b) include intensity? (c) random forest or gradient boosting (HistGradientBoosting is faster on 4 CPUs)?
Let's try 100 m radius as a starting point
Leave out intensity from derived metrics
Gradient boost works. 
Main critique: lots of custom code; is there a standard feature set or sets that are available, or that existing packages extract? 


### SatCLIP
SatCLIP is a location encoder trained on satellite imagery. 
(lat, lon) -> embedding

### Embeddings of NAIP imagery

**Decision (2026-11-11, user): Clay v1.5 NAIP embeddings** (`source.coop/clay/clay-v1-5-naip-2`, pre-computed for all NAIP 2010-2021); DINOv3-Sat kept as a possible later option.
Probe input = the embedding averaged over a disc around the point, disc radius per task. No chips/tiles are stored, only the summary per point.

*Claude notes (2026-11-11):*
- Dataset: 1.2 M parquet files (one per NAIP quarter-quad and year, 3.6 MB), each with one 1024-d embedding (Clay class token) per non-overlapping 256x256 px chip: **256 m chips at 1 m NAIP (2010-2017), ~154 m at 60 cm NAIP (2016+)**. So the finest spatial resolution is a chip; discs below ~150 m mostly return the chip under the point (radius 0 = that chip, and the point can sit at the chip edge). Embeddings are not centred on the point.
- Code: `src/eagle_bench/naip_clay.py`, `scripts/baseline_eval/09_fetch_naip_clay.py` (disc radius per task = `radius` of the task's table in `datasets/registry.yaml`; the run stops if one is blank), probe via `02_linear_probe.py --naip-clay`. File of a point = arithmetic on its lat/lon (USGS quarter-quad naming; checked against the chip footprints), so no STAC search. Year = the NAIP year nearest to the survey year (clipped to 2010-2021; ties to the later), per quarter-quad cell. Disc = sampled grid, each sample takes the chip it falls in, from the file of the cell it falls in (NAIP images overlap their neighbours, so no double counting). Output also has `coverage` and `naip_year`.
- Validated: radius 0 equals the containing chip exactly (6 points), cells contained in chip footprints (10 files), 40 random points at 100/250 m: coverage 1.00.
- **Cost:** 88,255 unique points (survey years clipped to 2010-2021). Files to read = download volume (a file is read whole): 29.8k files / 230 GB at radius 0, 31k at 250 m, 37k / 288 GB at 500 m, 45k / 346 GB at 1 km, 62k / 468 GB at 2 km, 118k / 878 GB at 5 km. 99.5% of centre points have NAIP embeddings. From the Mac the network gave only ~5-7 MB/s regardless of threads (~1.3 files/s, 7-10 h): run on CyVerse. Radii must be fixed before the run (a new radius re-reads the files).

## Status and results (Claude, kept current)

### 2026-11-10
- **Lidar cache running** (tmux `als_cache`, survives the session): ~1.1 cookies/s → ~21 h for 86k sites; cookies ~7.7 MB
  (~650 GB total on the data store). 1,434 ok / 63 empty after the first 25 min; 99% of cookies have ground points.
- **Bug found (pre-training code too): noise points are not removed.** `eagle_als.fetch` uses
  `Classification![7:7],Classification![18:18]`; PDAL keeps a point if it passes *any* range of a dimension, so classes
  7 and 18 stay (one NLCD water cookie: 155k high-noise points, max height 256 m). The benchmark drops them downstream.
  Fix in fetch.py: one expression `Classification[0:6],Classification[8:17],Classification[19:255]` (still ORed, but
  the ranges now cover exactly the kept classes) or a `filters.expression` `Classification != 7 && Classification != 18`.
  Your call, since it changes pre-training data.
- **lidR stdmetrics pipeline trialled** (20 cookies): 56 stdmetrics + 7 terrain columns. Heights < 0 clamped to 0
  (lidR convention). Open water: no ground → terrain NaN; zentropy NA on flat cookies → train-median imputation in the probe.
- **AlphaEarth (centre pixel, Earth Engine): done for all tasks.** Results: `scripts/baseline_eval/results/alphaearth_center/summary.md`.
  Validated against the COG files (cosine 1.000 median). Re-extracted after the SatBird → USA winter change.
- **Lidar cookie cache: blocked on PDAL.** Site table ready (86,150 unique sites, `benchmarking/als_cache/sites.parquet`);
  PDAL has no pip wheel, and the pixi installer download was not approved. Needs: pixi (preferred) or conda for `python-pdal`.
- **ForPT: embedder ready**, weights cached, smoke-tested on synthetic clouds (1008-d, 0.75 GB GPU per 50k-point cloud).
  Waiting for cookies.
- **Lidar metrics: proposal above**, waiting for your review.

## Why does AlphaEarth classify so well? (diagnostics plan, 2026-11-10)

Two hypotheses: (H1) the embedding mostly encodes geographic position / regional context; (H2) label information leaked
in through AEF's training data (secondary summaries of the AEF paper list GBIF species records and geotagged Wikipedia
as text sources, alongside Sentinel-1/2, Landsat, GEDI, DEM, ERA5-Land, GRACE; to confirm in arXiv 2507.22291).

Tests for H1 (geography):
1. *Location-only baselines* on the same splits: lat/lon through random Fourier features / spherical harmonics + the
   same linear probe, and kNN / random forest on lat/lon. AEF minus location-only = what AEF adds beyond position.
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
