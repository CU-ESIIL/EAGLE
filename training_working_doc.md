# EAGLE pre-training: working doc

Working notes for **self-supervised pre-training** of the LitePT-S lidar encoder (runs on PSC Bridges-2).
The baseline benchmark has its own doc, `benchmarking_working_doc.md`. Both docs follow the same rules:

- Keep this file current: update the status table, record decisions and results, and tick or remove next steps
  as they change. Stale notes are worse than none. Put a dated line in "Log" for anything meaningful.
- The user writes feedback and notes here too: read it at the start of a session, newest first.
- One entry per code change also goes in `PROMPT_ACTION_LOG.md`; the repo-wide description lives in `README.md`.

Migrated 2026-11-12 from the claude.ai working-notes artifact (last edited about 2026-10-11), which is now
retired. Facts from that doc that later commits changed have been updated; anything not verifiable from the
repo is marked **unverified**.

## Status

| item | state | next / how to resume |
|---|---|---|
| GPU env (`envs/litept-gpu`, LitePT vendored in `src/litept/`) | done | `source scripts/litePT/env.sh` |
| Streamed pre-training data (500 m squares inside DataLoader workers) | done 2026-10-06, in use | |
| Node-local staging of env + eval cookies | done (`stage_local.sh`) | check `$LOCAL` size on H100 nodes |
| Training on any GPU count (`grad_accum`/workers "auto") | done 2026-10-07 | |
| In-training validation: `nlcd` probe + `embed` collapse check | done 2026-10-08/09 | |
| Multi-target `probe_table` task + `build_eval_caches.py` | code committed 2026-10-29 | not yet in the default `eval_tasks`; build caches for LFRDB, VegBench, BBS, SnapShot, OFO, ... and add tasks |
| Eval-site exclusion from pre-training | done: `squares.EVAL_SITES` reads `datasets/registry.yaml` (all 10 tables, 500 m) | runs started before 2026-10-29 may have seen some eval sites |
| Sinkhorn memory bank (`sinkhorn_cookies=128`) | on by default; A/B (`ssl_s_1gpu_sk128` vs `ssl_s_1gpu_test`) **result not recorded** | compare steps 500-2000 (loss, `mask_kl`, `student_protos_used`, `check_embeddings.py`) |
| Run `ssl_s_2gpu_2` | last recorded at step 9,200 (2026-10-09): no collapse, NLCD probes drift below init | **state after 2026-10-09 unverified** (Bridges-2 is not mounted on Delta): check `$EAGLE_SCRATCH/runs/` |
| Full 8-H100 run (`ssl_s_pool_v1`, batch 128, 50k steps) | not started (job 47486115 sat in the queue) | `sbatch scripts/litePT/slurm/pretrain.sbatch` |
| ForPT details vs authors' code | code now public (github.com/prs-eth/ForPT; a copy is cloned at `/home/jovyan/external/ForPT` on CyVerse) | compare warmups, Sinkhorn, local-view matching, EMA schedule (see "Differences from ForPT") |
| Downstream evaluation of EAGLE embeddings on the benchmark | not started | reuse `scripts/baseline_eval/` (same probe and splits as the baselines) |

`$EAGLE_SCRATCH` = `/ocean/projects/bio260075p/sammlapp/eagle`. Run outputs: `$EAGLE_SCRATCH/runs/<run_name>/`,
slurm logs `$EAGLE_SCRATCH/logs/eagle-pretrain-<jobid>.out`. How to run and monitor: `README.md`
("Running pre-training", "Monitoring training") and `scripts/litePT/TENSORBOARD.md`.

## Goal and plan

Pre-train LitePT-S (12.4M parameters) with self-supervised learning on 100 m-radius "cookies" of USGS 3DEP
aerial lidar, following ForPT (Yue et al. 2026, arXiv:2609.24787: Sonata-style self-distillation of the same
encoder on forest lidar). Evaluate the frozen and fine-tuned embeddings on the 15 benchmark tasks
(`datasets/registry.yaml`, `scripts/baseline_eval/README.md`) against the baselines.

1. GPU environment, LitePT vendored. **Done.**
2. Cookie fetching/caching from 3DEP EPT with PDAL; lidar availability of the labeled sites. **Done** (the
   eval cookie cache for all benchmark sites now exists on Delta, see `CLAUDE.md`).
3. Pre-training data. **Done**: streamed squares (the earlier plan of 60k cached random cookies was superseded).
4. ForPT / Sonata recipe implemented, benchmarked on one H100, overfit test passed. **Done.**
5. Full pre-training on 8 H100s (or 2 on GPU-shared). **In progress / unverified** (see status).
6. Downstream evaluation, then the ablations under "Next steps".

## Environment (Bridges-2)

PyTorch 2.13 (CUDA 12.9), flash-attn 2.8.3, spconv 2.3.8, torch_scatter and PDAL, built for sm_50 to sm_120.

- Manifest `envs/litept-gpu/pixi.toml`; the env (~14 GB) is installed on `/ocean/projects/bio260075p/sammlapp/pixi_envs`
  (the first install filled the 25 GB home quota). Set `PIXI_CACHE_DIR=/ocean/projects/bio260075p/sammlapp/.pixi_cache`
  before installing; on CPU nodes also `CONDA_OVERRIDE_CUDA=12.6`.
- `source scripts/litePT/env.sh` sets PATH, PROJ_DATA/GDAL_DATA, PYTHONPATH and `EAGLE_SCRATCH` (safe under `set -u`).
- LitePT is vendored in `src/litept/` from upstream commit `436d048` (MIT). EAGLE patches are marked in comments:
    - flash-attn optional; without it (V100) PyTorch SDPA is used (matches to 2e-7). `EAGLE_ATTN=sdpa` forces it.
    - `LitePT(norm="ln", embedding="linear", mask_token=True)` = the ForPT / Sonata pre-training variant
      (LayerNorm, linear stem, learnable mask token). Upstream behaviour stays the default.
    - PointROPE uses its pure-PyTorch version unless the CUDA kernel is compiled.
    - Package-relative imports; `colorhash` optional.
- Sonata reference code (Pointcept `1342eda`) is cloned at `/ocean/projects/bio260075p/sammlapp/Pointcept`, reference only.
- Pass training settings as `key=value` arguments, not environment variables (exports do not reach Bridges-2 jobs).
- In interactive jobs the shell sees all 8 GPUs: set `CUDA_VISIBLE_DEVICES` to the job's GPU
  (`scontrol show job <id> -d`, IDX). Some 2026-10-07 runs used GPU 0, which was not in the allocation.
- GPU-shared allows at most 12 CPUs per GPU (`--cpus-per-task=13` per GPU is rejected).

## Data for pre-training

**Cookies.** `src/eagle_als/fetch.py` reads a 100 m-radius cylinder from a 3DEP EPT tile, drops noise classes
7 and 18 (`NOISE_FILTER`; before 2026-11-12 the PDAL filter dropped nothing, see `CLAUDE.md` "Known data
issues"), reprojects to local UTM centred on the site and computes height above ground (HAG) with a 1 m raster
of class-2 points (lowest point per 2 m cell when a tile has no ground class). Typical cookie: 0.15-1.2M points.

**Cache** (`src/eagle_als/cache.py`, used for evaluation cookies): parallel, resumable, one manifest record per
site; survives PDAL worker crashes (sliding window, crashed sites re-run one per process).

**Streamed squares (adopted 2026-10-06).** Each DataLoader worker keeps a pool of P = 8 squares (500 m) in RAM,
cuts random 100 m cookies from them and retires a square after N = 24 cookies (~3 uses per cookie area) once its
replacement has arrived. Fetches run in one crash-isolated helper subprocess per worker, writing shards to
`/dev/shm`; samples are yielded round-robin and the main process collates per-GPU batches from different
workers. A resumed run offsets the square sequence by its start step. Code: `squares.py` (`SquareSampler`, shard
format, `cut_cookie`, fetch helper; no torch import), `stream_pool.py` (`SquarePoolDataset`, `FetchHelper`,
`RoundRobinBatches`, `build_pool_loader`), `profile_pool.py`.

- Squares are weighted by sqrt(tile area) and redrawn if any evaluation site is within 500 m of the centre
  (half-diagonal 354 m + 100 m cookie, rounded up).
- Shard: one directory per square, `points.npy` (structured array, 19 bytes/point, ~57 MB for 3M points, sorted
  into a 10 x 10 grid of 50 m blocks) + `meta.json`; workers memory-map it and read only the blocks a cookie
  touches. Raw intensity is stored; `ALSFeatures` ranks it within each cookie, as for cached cookies.
- Cookies cut from a square match `fetch_cookie()` at the same centre (17/18 identical point counts); HAG differs
  by a median 0.14 cm (99th percentile 3 cm).

Measurements behind the design (RM-shared, 2026-10-05/06):

| Measurement | Result |
|---|---|
| Fetch cost vs read size | ~1.5 s fixed per PDAL query; a 1.6 km square costs 2-13x a 200 m square for 64x the area |
| 16 scattered patches vs one square of equal area | scattered 2.0-5.3x slower (median ~3.7x) |
| PDAL decode | CPU-bound, ~0.45 M points/s per core |
| HAG | raster HAG 0.19 core-s per M points, 11x faster than IDW k-NN; median difference 1 cm, 99th pct 13 cm |
| 500 m square fetch / 100 m cookie fetch | 7.8 s (1.9-64) / 2.3 s (1.4-4.2) |
| Preprocessing per sample | 0.3-0.45 s per worker, linear in workers |
| /ocean I/O | writes ~1 GB/s, large reads 2.9-5 GB/s, but many small reads can stall for minutes to an hour |

- Pool design A vs the separate-producer design B (job 47472580): both keep up with the GPUs; B ~20% faster
  uncapped (103 vs 86 samples/s). A was kept as simpler; B is in commit 895bafe.
- Start-up was the real problem: imports 81 s and `SquareSampler()` 466 s cold on a fresh node; one run
  (47476010) blocked on /ocean reads for an hour. Hence node-local staging: each job extracts the env (13 GB,
  one zstd archive) and the eval cookies (tar) to `$LOCAL` (`stage_local.sh`); checkpoints and logs stay on /ocean.
- H100 nodes (w001-w010): 104 cores, 2 TB RAM.
- Dense squares dominate cost (32-37 pts/m²: 63 s each, two-thirds of points discarded by thinning to 12 pts/m²):
  reading a coarser EPT level for dense tiles is worth testing.

## Evaluation datasets and 3DEP registry

The labeled datasets are now defined in `datasets/registry.yaml` (10 tables) and documented in each
`datasets/*/README.md`, `data_cleaning/DATASET_INGESTION.md` and `scripts/baseline_eval/README.md`. Splits are
frozen since 2026-10-29 (`eagle_als.splits`: 0.25-degree spatial blocks, or design units for BBS routes and BLM
AIM). Notes that matter for training:

- VegBank was replaced by **VegBench** (California, published split; `datasets/VegBench_eval/`, 10,455 surveys
  with lidar within 3 years).
- **SatBird** was reworked on 2026-11-10: USA winter only, SatBird's own labels and split
  (`satbird_winter_occupancy`).
- **sPlotOpen** was dropped (82 usable US plots). **NestWatch** is exploration only (notebooks in
  `data_cleaning/data_exploration/`), not an evaluation dataset.
- 3DEP registry: collection dates corrected from WESM + EPT GpsTime (789 of 2,278 products changed year;
  51 disagree, 434 without usable GpsTime). `leaf_on` is now a collection-season rule (June-September only =
  True, December-March only = False, else null: 201 / 541 / 1,536); the NDVI/provider approach was removed.
  Details in `datasets/USGS_3dep/README.md`.

## In-training validation

`train_ssl.py` runs the config's `eval_tasks` on the frozen teacher every `eval_every` = 300 steps and at step 0
(`src/eagle_als/evaluation.py`, task registry). Defaults:

- `nlcd` (`classification`): NLCD 2021 reference points, balanced subset, 942 usable rows, 15 classes, 726 train /
  216 test, linear probe and cosine kNN on mean-pooled up-cast features. Cache `$EAGLE_SCRATCH/cache/nlcd`
  (7 GB, staged by `pretrain.sbatch`). 6-7 s per evaluation on one H100 plus ~75 s of setup at job start.
  The NLCD split was redone on 2026-10-29 (0.25-degree blocks, 50% test, was 25%): **rebuild `cache/nlcd`
  and its tar if rows or `als_site_id` changed**, and note that later NLCD scores are not comparable to the ones below.
- `embed` (`embedding_check`): collapse statistics on 256 held-out NLCD cookies (`eval/embed/*`).
- `probe_table` (2026-10-29, not in the defaults yet): several targets (classification, regression, presence)
  on one labeled table; caches built with `scripts/litePT/build_eval_caches.py`. All linear probes now use
  `eagle_als.linear_probe` (the same probe as the benchmark).
- Scoring saved checkpoints: `scripts/litePT/eval_checkpoints.py`, `check_embeddings.py` (GPU needed, spconv).

## Results so far

**One-H100 benchmark** (job 47448878, OFO cookies): batch 8 per GPU fits in 52 GB (16 runs out of memory); step
time 1.01 s at batch 8, throughput flat beyond batch 4 (8 samples/s). Starting loss 8.31 = ln(4096). Batch 128
on one 8-GPU node is ~2 s/step, so 50k steps ≈ 28 h. On 1 GPU (batch 128 as 8 x 16 accumulation) ~13 s/step.
Overfit test (job 47450403) passed.

**NLCD linear macro F1 of saved checkpoints** (old 25% test split; chance 0.067; kNN never above its init 0.21):

| Run | init | 500 | 1000 | 1500 | 2000 | 3000 |
|---|---|---|---|---|---|---|
| `ssl_s_2gpu` (batch 128) | 0.285 |  | 0.318 |  | 0.341 | 0.302 |
| `ssl_s_1gpu_test` (batch 32) | 0.281 | 0.245 | 0.244 | 0.253 | 0.238 |  |

Random-init features already reach 0.28, and with 216 test rows a few points of difference is noise.

**`ssl_s_2gpu_2` at step 9,200 (2026-10-09): no collapse, but the probes drift down.** `mask_kl` rose
(0.36 at steps 3000-3500, 0.41 at 8500-9000) while `target_entropy` fell (5.47 to 5.00): the teacher sharpens
faster than the student follows. NLCD linear F1 peaked at 0.345 (steps 900, 4200, 5100), then 0.21-0.27 after
step 6000 (init 0.289). Embedding check (job 48979752):

| Step | cookie_erank | cookie_cos | point_erank | point_cos_within | point_std |
|---|---|---|---|---|---|
| init | 69.1 | 0.912 | 626 | 0.746 | 0.0171 |
| 7000 | 92.7 | 0.217 | 321 | 0.783 | 0.0264 |
| 8000 | 92.4 | 0.213 | 310 | 0.794 | 0.0259 |
| 9000 | 90.8 | 0.233 | 301 | 0.809 | 0.0251 |

Cookie embeddings are well spread; point features within a cookie slowly become more alike (loss of spatial
detail). Hypothesis: the fixed EMA momentum 0.994 (no ramp to 1.0 as in Sonata) with a constant lr keeps the
teacher drifting. Not yet tested.

**`ssl_s_1gpu_test`** (batch 32, 2026-10-07): by step 1000 `protos_used` fell 4,093 → ~350 with `target_entropy`
~7.8; cookie cosine 0.94 → 0.31, point effective rank 630 → 146, cookie effective rank 74 → 28 (falling).
Stopped at step 2010 (resumable) as the baseline of the Sinkhorn-bank A/B.

## Model and training code

`src/eagle_als/ssl.py` ports Sonata's reference implementation (Pointcept `sonata_v1m1_base.py`) with ForPT's
settings wherever the paper states them. Component-level explanation: `src/eagle_als/README.md` ("SSL components").

- **Inputs** (`transforms.py`), 8 channels: xyz relative to median ground, HAG, intensity percentile within the
  cookie, return features (return_number / number_of_returns, is_first, is_last). 0.4 m grid.
- **Augmentations** per view: Sonata's set with metric values x20 for our grid; plus thinning to 2-12 pts/m²,
  intensity gamma/gain/drop, a smooth HAG offset, occasional dropping of all non-first returns.
- **Views:** 2 global (40-100% of a 50 m-radius disc), 4 local (10-40%), capped at 65k points each.
- **Losses** (point level, 4096 prototypes, Sinkhorn targets, match radius 6.4 m by pre-augmentation coordinates):
  masked-to-global (each masked student global view vs the teacher's same and other global view, weights 2/8 +
  2/8) and local-to-global (weight 4/8, against the teacher's first global view).
- **Sinkhorn memory bank** (2026-10-07, on by default): each normalization covers 128 cookies on any GPU count
  (`ssl.sinkhorn_cookies=128`, `sinkhorn_points_per_cookie=2048`, weighted subsample; identical to Sonata with no
  queue). Logged: `sk_cookies`, `protos_soft_used`, `target_max`, `student_protos_used`.
- **Trainer** (`scripts/litePT/train_ssl.py`, config `configs/ssl_litept_s.py`): DDP via torchrun, gradient
  accumulation, AdamW with layer-wise lr decay, bf16 (fp16 on V100), automatic resume, clean stop 30 min before
  the time limit, exports `backbone_teacher.pth`, collapse diagnostics in the log.
- **Downstream embedding:** mean over points of the up-cast, concatenated multi-scale features.

### Differences from ForPT

| Aspect | ForPT | Ours |
|---|---|---|
| Data | 160 km² of forest, mostly European ALS/UAV/MLS, ~200 pts/m², 20 x 20 m patches | US-wide 3DEP, all land cover, thinned to 2-12 pts/m², 100 m cookies, views within 50 m |
| Grid | 5 cm | 0.4 m |
| Inputs | xyz | 8 channels (xyz, HAG, intensity, returns) |
| Masks | 5 cm patches, ratio 0.7 | 2→6 m xy columns, ratio 0.3→0.7 over the first 5% |
| Metric settings (jitter, elastic, match radius) | Sonata's indoor values | x20, e.g. 6.4 m match radius |
| Height | Sonata re-centres z per view | z relative to median ground (`CenterShift(apply_z=False)`) |
| LR warmup | none stated | linear over 2,500 steps (5%), then constant |
| Sinkhorn batch | per-GPU batch | 128 cookies via the memory bank |

Shared with ForPT: LitePT-S encoder (LN, linear stem, mask token, no teacher drop path), lr 1e-3 constant,
weight decay 1e-4, layer decay 0.9, batch 128, EMA momentum 0.994 fixed, teacher/student temperatures 0.07 / 0.1.
From Sonata's code: bf16, grad clip 3.0, 4096 prototypes, 5% warmups for teacher temperature and mask ratio.

To check against the now-public ForPT code: whether it keeps Sonata's warmups and Sinkhorn targets; whether local
views match one global view (Sonata's code) or both (ForPT eq. 2); its EMA momentum schedule. The benchmark's
ForPT embedder already uses their released teacher weights (`CLAUDE.md`, "ForPT facts"): inputs xyz only,
CenterShift with z, 5 cm grid.

## Alternative architectures and training paradigms

The current recipe learns ~1-10 m structure (LitePT-S stages at 0.4, 0.8, 1.6, 3.2, 6.4 m) but never trains a
100 m-scale representation: both losses are point-level and global views are at most 50 m in radius.

| Option | Memory | Scales | Effort | Notes |
|---|---|---|---|---|
| Multi-scale heads on the current recipe | same | 1 m, 10 m, view (50-100 m) | low | DINO loss and head per pooled scale; global views covering the full cookie, coarser EPT levels far from the centre |
| Vernata memory bank | lower | — | low (done, see Sinkhorn bank) | [paper](https://arxiv.org/html/2608.06919), [code](https://github.com/rai-opensource/vernata) |
| Two-level: LitePT on 10-20 m tiles + 2D transformer over tile tokens | bounded | 1 m to 1 km | medium | precedent: pathology (HIPT, Prov-GigaPath); suits BBS and butterflies |
| Pillar/BEV tokens + 2D ViT | very low | 1, 10, 100 m maps | medium | reuses 2D SSL; aligns with NAIP/Sentinel; loses within-column 3D detail |
| Masked reconstruction (MAE family) | encoder sees 30-40% | depends | medium | mask 10 m blocks, reconstruct height-binned occupancy; usually weaker frozen probes than DINO |
| Distilling 2D foundation models | + teacher | 1 m (NAIP), 10 m (satellite) | medium | e.g. Clay NAIP or AlphaEarth embeddings as targets (both are now benchmark baselines) |
| Mamba / linear-attention backbones | lower | same | high | low priority |

Suggested order (2026-10): (1) baselines (now largely done in the benchmark: AlphaEarth, Clay NAIP, ForPT,
lidR metrics); (2) multi-scale heads and 100 m views in the current run; (3) the two-level model, with pillar/BEV
+ 2D ViT as the cheaper comparison. Utonia ([arXiv:2603.03283](https://arxiv.org/html/2603.03283v1)) suggests a
fixed grid with rescaled coordinates rather than jittering the grid when mixing densities. Precedents named
without links are cited from memory and not checked.

## Next steps

- [ ] Find out the state of `ssl_s_2gpu_2` and the Sinkhorn-bank A/B on Bridges-2; record results here.
- [ ] Test the teacher-drift hypothesis: EMA momentum ramp 0.994 → 1.0 (Sonata) and/or a cosine lr decay.
- [ ] Launch the full run on 8 H100s, or 2 on GPU-shared (`--gpus=h100-80:2 --cpus-per-task=24`).
- [ ] Rebuild `cache/nlcd` after the NLCD re-split if `als_site_id` or rows changed; consider standardizing
      embeddings before cosine kNN, and a Level I (8-class) NLCD task for less noisy scores.
- [ ] Add `probe_table` tasks for more datasets (LFRDB, VegBench, ...) via `build_eval_caches.py`, keeping each task
      to ~2,000 cookies.
- [ ] Check the released ForPT code for the unconfirmed details above.
- [ ] Evaluate EAGLE checkpoints on the full benchmark with `scripts/baseline_eval/` (same probe, splits, radii).
- [ ] V100 benchmark of the SDPA fallback.
- [ ] Ablation: height inputs (z and HAG, z only, HAG only).
- [ ] Ablation: return features as a 4-way one-hot (only, first of many, intermediate, last of many) plus clipped
      `number_of_returns`.
- [ ] Data efficiency: coarser EPT levels for dense tiles; `pool_uses_per_square`; scale `pool_size` with
      8 / GPUs so small-GPU runs see as many squares as a full node.
- [ ] BBS: 3 pairs of renumbered routes share a start point (and a cookie) across splits (e.g. 840_27_019 /
      840_27_219). The 2026-10-29 re-split is still by `RouteID`, so this probably remains; `probe_table` drops
      cookies found in both splits. Fix by merging renumbered routes or splitting by start location.

## Future training ideas (user)

- [ ] Pull a larger-radius point cloud at much lower density, e.g. a 1 km x 1 km tile at 0.1 pts/m², as a
      landscape-context view processed in parallel with the detailed cookie; or train a separate model on these
      coarse tiles.
- [ ] Contrastive loss on `point_source_id`:
    - the point cloud often has complete coverage from 2+ overlapping flight lines;
    - these are two literal views of the same habitat on the same date;
    - make a view per `point_source_id` and pull [1], [2] and [1 & 2] together.
- [ ] Supervised tasks during pre-training: eBird, iNaturalist observation prediction; land-cover / habitat
      classification.
- [ ] Fast compression of flat regions: many ALS tiles are mostly or entirely ground. Rather than run the heavy
      pipeline, expand such regions into a sparse surface cheaply. Whether this uses HAG or height matters: a
      region with no points above ground can be a surface + intensity, and if planar, just elevation,
      intensity, slope and aspect.

## Log

- 2026-11-12: migrated from the claude.ai artifact to this file; updated for later changes (VegBench, SatBird
  rework, sPlotOpen dropped, leaf-on rule, registry-based eval-site exclusion, frozen splits, torch linear
  probe, `probe_table`, ForPT code release, noise-filter fix).
- 2026-10-29: `probe_table` multi-target eval task, `build_eval_caches.py`, eval caches/checkpoint slurm scripts;
  `EVAL_SITES` from the registry; splits frozen.
- 2026-10-09: `embed` collapse check as an eval task; `ssl_s_2gpu_2` analysed at step 9,200; leaf-on rule.
- 2026-10-08: NLCD in-training validation, `eval_checkpoints.py`; 3DEP collection dates corrected.
- 2026-10-07: training on any GPU count; Sinkhorn memory bank; `ssl_s_1gpu_test` collapse test.
- 2026-10-06: streamed squares (design A) adopted, node-local staging, lr warmup.
- 2026-10-05: ForPT / Sonata port, one-H100 benchmark, streaming measurements, raster HAG.
