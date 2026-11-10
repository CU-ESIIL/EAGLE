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

*Questions for you:* (a) is the 50 m disc the right default scale (plot data vs. BBS routes / eBird hotspots, which are much
larger)? (b) include intensity? (c) random forest or gradient boosting (HistGradientBoosting is faster on 4 CPUs)?


### SatCLIP
SatCLIP is a location encoder trained on satellite imagery. 
(lat, lon) -> embedding

### Embeddings of NAIP imagery

## Status and results (Claude, kept current)

### 2026-11-10
- **AlphaEarth (centre pixel, Earth Engine): done for all tasks.** Results: `scripts/baseline_eval/results/alphaearth_center/summary.md`.
  Validated against the COG files (cosine 1.000 median). Re-extracted after the SatBird → USA winter change.
- **Lidar cookie cache: blocked on PDAL.** Site table ready (86,150 unique sites, `benchmarking/als_cache/sites.parquet`);
  PDAL has no pip wheel, and the pixi installer download was not approved. Needs: pixi (preferred) or conda for `python-pdal`.
- **ForPT: embedder ready**, weights cached, smoke-tested on synthetic clouds (1008-d, 0.75 GB GPU per 50k-point cloud).
  Waiting for cookies.
- **Lidar metrics: proposal above**, waiting for your review.
