# `eagle_als`

Python package for EAGLE: getting 3DEP aerial lidar from USGS EPT tiles, turning it into training
samples, and pre-training the LitePT encoder with self-supervision. Scripts in `scripts/litePT/`
use it; the project-level overview (training design, monitoring, planned experiments) is in the
repository `README.md`.

Activate the environment first; it puts `src/` on `PYTHONPATH`:

```bash
source scripts/litePT/env.sh
python -c "import eagle_als"
```

The LitePT model and its generic point-cloud transforms are vendored separately in `src/litept/`.

## How the modules fit together

```text
                 fetch.py  (PDAL reads from 3DEP, reprojection, height above ground)
                 /                                   \
   cookies (100 m circles)                     squares (500 m)
   cache.py  -> .npz cache on /ocean           squares.py  -> shards on /dev/shm
   data.py   -> CookiePoolDataset /            stream_pool.py -> SquarePoolDataset
                LabeledCookieDataset                              (pre-training)
   (evaluation sets, overfit test)                 \
                 \                                  \
                  transforms.py  (features, views, augmentations)
                         |
                  ssl.py  (SonataLitePT: student/teacher self-distillation)
                         |
                  train_utils.py  (config, DDP, optimizer groups, checkpoints)
                         |
             scripts/litePT/train_ssl.py
```

There are two ways data reaches the model:

- **Streamed squares** (pre-training, `data_mode="pool"`): each DataLoader worker fetches 500 m
  squares and cuts random 100 m cookies from them. Nothing is cached between jobs.
- **Cached cookies** (evaluation sets and tests, `data_mode="cache"`): one 100 m cookie per site,
  fetched once by `cache.py` and saved as `.npz`.

Both produce the same cookie dictionary (below), so the same transforms and model run on either.

## Modules

| Module | Purpose | Main entry points |
| --- | --- | --- |
| `fetch.py` | Read lidar from a 3DEP EPT tile with PDAL: drop noise classes, reproject to local UTM, compute height above ground. Save and load cookies. | `fetch_cookie`, `fetch_square`, `height_above_ground`, `load_cookie`, `save_cookie`, `tile_index` |
| `squares.py` | 500 m squares for streamed pre-training: seeded square locations (evaluation sites excluded), the shard format, cutting cookies from a shard, and the fetch-helper process. No torch import, so helpers start quickly. | `SquareSampler`, `fetch_square_shard`, `cut_cookie`, `python -m eagle_als.squares serve` |
| `stream_pool.py` | Pre-training DataLoader: per-worker pool of squares, crash-isolated fetch helper, and batching across workers. | `build_pool_loader`, `SquarePoolDataset`, `FetchHelper`, `RoundRobinBatches` |
| `cache.py` | Fetch and cache one cookie per site for a table of sites, in parallel and resumable, with a status manifest. | `python -m eagle_als.cache`, `build_cache`, `read_manifest`, `cookie_path`, `imap_isolated` |
| `data.py` | Datasets over cached cookies, and the collate function shared by all loaders. | `CookiePoolDataset`, `LabeledCookieDataset`, `collate_points` |
| `transforms.py` | ALS transforms registered in LitePT's transform registry, used from configs by name. | `ALSFeatures`, `RandomPointThinning`, `ALSMultiViewGenerator`, `PackFeat`, `IntensityJitter`, `HAGJitter`, `ReturnDrop`, `XYCrop` |
| `ssl.py` | Self-supervised model: LitePT-S student and EMA teacher, prototype heads, masking, masked-to-global and local-to-global losses, collapse diagnostics (ForPT / Sonata recipe). | `SonataLitePT` |
| `evaluation.py` | Validation tasks on the frozen teacher encoder, run by `train_ssl.py` every `eval_every` steps: task registry, build from the `eval_tasks` config, TensorBoard tags. | `EvalTask`, `register`, `build_tasks`, `run_tasks`, `tb_tag` |
| `probe.py` | The `classification` task: frozen-embedding probes on fixed labeled cookie sets (logistic regression + kNN on mean-pooled teacher features; balanced accuracy, macro and per-class F1). | `ClassificationProbe`, `probe_scores`, `eval_transform` |
| `train_utils.py` | Shared training helpers: python config files with `key=value` overrides, DDP setup, layer-wise learning-rate groups, schedules, checkpoints, clean stop on time limit or signal. | `load_config`, `setup_distributed`, `layerwise_param_groups`, `save_checkpoint`, `latest_checkpoint`, `StopFlag` |
| `sites.py` | Site tables: read csv / parquet / geo files, and stable site ids from (tile, lat, lon). | `read_table`, `site_id`, `add_site_ids` |

## Data formats

**Cookie** (a dict of numpy arrays, from `fetch.load_cookie`, `fetch.fetch_cookie` or
`squares.cut_cookie`):

| Key | Type | Meaning |
| --- | --- | --- |
| `xyz` | float32 [N, 3] | x, y relative to the cookie centre in UTM metres; z is elevation |
| `hag` | float32 [N] | height above ground (m) |
| `intensity` | uint16 [N] | raw intensity (`ALSFeatures` ranks it within each cookie) |
| `return_number`, `number_of_returns` | uint8 [N] | |
| `classification` | uint8 [N] | ASPRS class; noise classes 7 and 18 are already dropped |

Cached cookies are `.npz` files that also hold a `meta` JSON string (tile, lat, lon, EPSG, radius,
point count, whether the tile has ground points). The cache layout is
`<cache>/cookies/<shard>/<site_id>.npz` plus `<cache>/manifest/part-*.parquet`.

**Square shard** (`squares.py`): a directory `<index:09d>/` holding `points.npy`, one structured
array of 21 bytes per point sorted into 50 m blocks, and `meta.json` with the block offsets, tile,
centre and timings. Square `i` depends only on `(seed, i)`. `cut_cookie` reads only the blocks a
cookie overlaps.

**Training sample** (output of the transforms, input to `SonataLitePT`): `global_*` and `local_*`
arrays for the 2 global and 4 local views (`coord`, `origin_coord`, `feat`, `offset`).
`collate_points` concatenates samples into a batch and shifts the offsets.

## SSL components

`ssl.py` holds `SonataLitePT`, the self-supervised model. It ports the Sonata recipe (Wu et al. 2025,
Pointcept `sonata_v1m1_base.py`) with the settings ForPT (arXiv:2609.24787) reports, plus a few
changes for aerial lidar. Numbers below are the defaults in `scripts/litePT/configs/ssl_litept_s.py`.

### The idea in one paragraph

There are no labels. Instead, two copies of the same network are trained together: a **student**
that is updated by gradient descent, and a **teacher** that is a slowly moving average of the
student. Both see different versions ("views") of the same cookie: crops, rotations, jitter, and for
the student, parts hidden by a mask. For every student point, the matching point in the teacher's view
is found by location, and the student is trained to give the same answer as the teacher for that
point. "The same answer" means a probability distribution over 4096 learned **prototypes**
(cluster centres). To match the teacher, the student must work out what a place looks like from
partial, distorted or smaller views, so its features have to describe structure, not raw input
values. After pre-training only the teacher's encoder is kept (`backbone_teacher.pth`).

### Inputs and views

Each training sample is one cookie, turned into 6 views by `transforms.ALSMultiViewGenerator`:

- **2 global views**: discs covering 40–100% of the area of a 50 m-radius disc (radius 32–50 m).
  The first ("principal") view is centred near the middle of the cookie; the second is centred on a
  random point of the first, so the two overlap but are not identical.
- **4 local views**: small discs covering 10–40% of that area (radius 16–32 m), centred on random
  points of the principal view.

Each view is augmented independently (scaling, rotation about z and small tilts, flips, jitter,
elastic distortion, intensity and height-above-ground jitter), then voxelised to a 0.4 m grid
(`GridSample`, one point kept per occupied 0.4 m cell). Each kept point has 8 input channels: x, y, z
(divided by 50 m; z is height relative to the cookie's median ground), height above ground,
intensity rank within the cookie, and 3 return features.

Every point also carries `origin_coord`, its position **before** augmentation. Augmentations move
`coord` differently in each view, but `origin_coord` is the same real-world location in all views, so
it is what the losses use to find which points correspond.

### The encoder (LitePT-S)

LitePT is a point transformer that works on sparse voxels. It has an embedding layer and 5 stages;
between stages, a pooling layer merges points. `enc_mode=True`: the pre-training model has only the
encoder, without the decoder LitePT normally uses for per-point segmentation.

| Stage | Cell size | Channels | Blocks | Block type |
| --- | --- | --- | --- | --- |
| embedding | 0.4 m | 36 | – | linear layer (8 → 36) + LayerNorm + GELU |
| 0 | 0.4 m | 36 | 2 | sparse convolution |
| 1 | 0.8 m | 72 | 2 | sparse convolution |
| 2 | 1.6 m | 144 | 2 | sparse convolution |
| 3 | 3.2 m | 252 | 6 | attention (14 heads) |
| 4 | 6.4 m | 504 | 2 | attention (28 heads) |

- **Pooling (`GridPooling`, stride 2).** At the start of stages 1–4, every point's features are
  projected to the new channel count, then all points in the same 2×2×2 block of cells are merged
  into one point: features by max, coordinates by mean. The cell size doubles, so the number of
  points falls at each stage (up to 8×; less where cells are sparse). The pooling layer stores
  which parent point each merged point came from (`pooling_parent`, `pooling_inverse`); `up_cast`
  uses this later.
- **Convolution blocks (stages 0–2).** A 3×3×3 sparse convolution (only occupied cells are
  computed) + linear layer + LayerNorm, added back to the input (a residual connection). These are
  cheap and capture local shape: a point and its immediate neighbours.
- **Attention blocks (stages 3–4).** Attention lets every point exchange information with many
  others, but costs grow with the square of the number of points. LitePT puts it only in the deep
  stages, where few points are left. Points are put in a 1D order along a space-filling curve
  (z-order or Hilbert, in 4 variants), so points next to each other in the list are close in space;
  the list is cut into patches of 1024 points and attention runs within each patch. Successive blocks
  use different curve orders, so the patches differ and information mixes across patch borders.
  At 3.2–6.4 m cells, one patch covers a large part of a view or all of it, so attention here is
  close to global within the view. Each attention block is LayerNorm → attention → residual, then
  LayerNorm → MLP (4× wider) → residual. Position enters the attention through PointROPE (rotary
  position encoding on the grid coordinates), so the attention knows how far apart two points are.
- **Mask token.** The embedding layer has a learnable vector, `mask_token`. For masked points (see
  below) the embedded features are replaced by this vector. The point stays in place; only what it
  "says" about itself is hidden.
- **Drop path** (randomly skipping a block's residual branch during training; 0 → 0.3 from the
  first block to the last) regularises the student. The teacher has drop path and dropout off
  (`teacher_custom`).

### `up_cast`: from the deepest stage back to point features

The encoder ends at stage 4, with one point per 6.4 m cell: too coarse to compare points between
views, and 504 channels with only the deepest context. Sonata does not use a decoder for
pre-training (its authors argue that a decoder lets the network solve the task from low-level geometry
without learning good encoder features). Instead, `up_cast` goes back up `up_cast_level` = 2
stages, using the parent links stored by pooling:

1. Each stage-3 point (3.2 m) gets the 504 features of the stage-4 point it was merged into,
   appended to its own 252 → 756 channels.
2. Each stage-2 point (1.6 m) gets the 756 features of its stage-3 point, appended to its own 144
   → **900 channels**.

The result is one point per occupied 1.6 m cell, with 900 features that combine local detail (stage
2), medium context (stage 3) and wide context (stage 4). No parameters are learned in `up_cast`; it
only copies and concatenates. These are the features the losses compare, and the features used for
evaluation.

### Projection heads and prototypes (`OnlineCluster`)

Each of the student and teacher has two heads, `mask_head` and `unmask_head` (one per kind of loss;
same design, separate weights). A head:

1. MLP 900 → 4096 → 256, then L2-normalise, giving a 256-d unit vector per point (`embed`).
2. `prototype`: a linear layer 256 → 4096 without bias whose weight rows are kept at unit length
   (weight normalisation with the length fixed to 1). Each row is one prototype, so the output is the
   cosine similarity between the point and each of the 4096 prototypes, a number between -1 and 1.

The prototypes are learned and act as a soft clustering of point types (e.g. "canopy top of a
closed broadleaf forest", "flat open ground"). The heads exist only to define the training task and
are thrown away afterwards.

### Teacher targets: temperatures and Sinkhorn-Knopp

The student's prediction for a point is `softmax(similarities / student_temp)` with
`student_temp` = 0.1: dividing by a small temperature makes the distribution peakier.

The teacher's similarities are turned into the target with **Sinkhorn-Knopp** (`sinkhorn_knopp`),
not a plain softmax. The reason is **collapse**: the easiest way for the student to agree with the
teacher is for both to put every point on the same prototype, which gives zero loss and useless
features. Sinkhorn-Knopp prevents this by making the targets for a batch use all prototypes about
equally:

1. Take the teacher's similarities for all target points in the batch, a matrix of points ×
   prototypes. Compute `exp(similarity / teacher_temp)`.
2. Rescale each prototype's column so every prototype has the same total over the batch.
3. Rescale each point's row so it sums to 1 (a probability distribution).
4. Repeat 2–3 three times. The result is close to a matrix whose rows are distributions and whose
   columns have equal totals.

Each point still goes mostly to the prototypes it is most similar to, but a prototype that many
points prefer is scaled down and rarely used ones are scaled up. This is an "equal partition" of
the batch among prototypes (as in SwAV and DINOv2). The sums run over all GPUs (`all_reduce`), so
the partition covers the whole batch, not one GPU's share.

`teacher_temp` rises from 0.04 to 0.07 over the first 5% of steps. A lower teacher temperature
gives sharper targets (fewer prototypes per point); the teacher is sharper than the student (0.07
vs 0.1), which pushes the student toward confident assignments.

**Sinkhorn memory bank.** The equal partition only makes sense when the batch is large enough to
contain many kinds of places. With 8 cookies per GPU per micro-batch and gradient accumulation, one
normalisation would see few cookies. Each GPU therefore keeps a queue of the teacher's head
embeddings (the 256-d vectors) from recent cookies, up to `sinkhorn_points_per_cookie` = 2048
random points per cookie. At each step the queued embeddings are scored against the current
prototypes and join the normalisation, so it spans `sinkhorn_cookies` = 128 cookies over all GPUs
(the full ForPT batch). Queued points are weighted so a queued cookie counts as much as if all its
points were present. Only the current points' targets are returned; the queue only shapes the
partition. With 8 GPUs and 2 accumulation steps, the queue is mostly the other micro-batch of the
same optimizer step.

### Masking (`generate_mask`)

Only the student's copies of the global views are masked; the teacher sees them complete.

- Each global view is divided into xy squares of side `mask_size`, and a fraction `mask_ratio` of
  the squares is chosen at random. Every point in a chosen square (the whole vertical column, ground
  to canopy top) is masked.
- Mask size grows from 2 m to 6 m and the ratio from 30% to 70% over the first 5% of steps, so the
  task starts easy.
- Masked points get the mask token instead of their features, and their coordinates are jittered
  (Gaussian, σ = 0.2 m, `mask_jitter`) so their exact positions give away less of the hidden shape.

Why columns rather than Sonata's 3D cubes (`mask_dims=2`): aerial lidar is close to 2.5D: most of the
information is in how points are stacked vertically at each xy location. A 3D cube mask would leave
the rest of the column visible above and below it, and the student could fill in the gap from the
same column. Masking whole columns forces it to infer a location's vertical structure from the
surrounding area. Larger masks than ForPT's 5 cm are used because 3DEP is far sparser than the
terrestrial scans ForPT used.

### Matching points between views (`match_neighbour`)

The losses compare a student point with "the same place" in a teacher view. After `up_cast`, both
have one point per occupied 1.6 m cell, but the cells do not line up between views (different crops
and augmentations). For each student point, the nearest teacher point of the same cookie is found by
`origin_coord` (pre-augmentation location), with a KD-tree. Pairs further apart than `match_max_r` =
6.4 m (one stage-4 cell) are dropped; this happens where a student view extends outside the teacher
view. The fraction of student points that found a match is logged (`mask_match`, `roll_match`,
`unmask_match`).

### The three losses

Each loss is a cross-entropy between the teacher's target distribution `t` and the student's
prediction `p` for every matched point: `-Σ t · log p` over the 4096 prototypes. It is averaged
over points within each view, then over views (`distill_loss`). The teacher side has no gradients.

| Loss | Student sees | Teacher sees | Weight | What it teaches |
| --- | --- | --- | --- | --- |
| `mask_loss` (masked-to-global) | masked global view A | full global view A | 2/8 | fill in hidden columns from their surroundings |
| `roll_mask_loss` (masked-to-global, crossed) | masked global view A | full global view B | 2/8 | same, plus agree across a different crop and augmentation of the same place |
| `unmask_loss` (local-to-global) | full local view (no mask) | full principal global view | 4/8 | describe a place from a small crop as the teacher does with wide context |

- **Mask loss.** Matching runs over all student points, not only the masked ones. For masked points
  the student must predict from context; for visible points it must still give features that agree
  with the teacher's.
- **Roll loss.** `roll_point` swaps the two global views of each cookie in the teacher output
  ([A, B] → [B, A]), so masked view A is compared with view B and masked B with A. Because A and B
  were cropped and augmented independently, matching them teaches invariance to those changes
  (rotation, scaling, jitter, intensity and height noise, where the crop edge falls).
- **Local-to-global loss.** A local view is a 16–32 m disc. The teacher sees the same location as
  part of a 32–50 m view. To match, the student must infer from little context what the teacher sees
  with wider context, e.g. that a patch of trees is part of a forest edge. All 4 local views of a
  cookie are matched to its principal global view. This loss uses the `unmask_head` heads and has
  its own Sinkhorn queue.
- Total: `loss = 2/8 · mask_loss + 2/8 · roll_mask_loss + 4/8 · unmask_loss`, ForPT's
  L = L_m2g + L_l2g with Sonata's weights.

### Student-teacher updates (`update_teacher`)

After each optimizer step on the student, every teacher weight becomes
`teacher = m · teacher + (1 − m) · student` with momentum `m` = 0.994 (ForPT keeps it fixed; Sonata
raises it to 1). The teacher never gets gradients. This exponential moving average (EMA) changes
slowly, so:

- the targets are stable: the student is not chasing a target that moves as fast as it does, which
  would make collapse or oscillation easy;
- the teacher averages many recent students, which usually makes it slightly better than any one
  of them, so the student keeps learning from a better version of itself.

This is why evaluation and the exported weights use the teacher.

### Schedules (`set_step`)

`train_ssl.py` calls `set_step(step, total_steps)` each step. Mask size, mask ratio and teacher
temperature warm up linearly over the first 5% of steps and then stay constant; momentum is constant
(`cosine_schedule` with equal base and final values is a constant).

### Diagnostics (`diagnostics`)

Computed on the mask loss and logged: `target_entropy` (how spread the teacher targets are),
`mask_kl` (how far the student is from the targets; the loss equals entropy + KL, so only KL can be
learned away), `protos_used` / `student_protos_used` (how many prototypes are anyone's top choice),
`protos_soft_used`, `argmax_agree`, `target_max`. See `scripts/litePT/TENSORBOARD.md`.

### From encoder to cookie embedding

For evaluation (`probe.py`, `embedding_check.py`) and downstream use:

1. Turn the cookie into one fixed view: thinned to 8 pts/m², central 50 m disc, no augmentation,
   0.4 m grid, no mask (`probe.eval_transform`).
2. Run the teacher encoder and `up_cast`: 900 features per occupied 1.6 m cell.
3. Average over all cells of the cookie: one 900-d vector per cookie.

The heads are not used. Because the average is over occupied 3D cells, a tall multi-layer canopy
(many occupied cells per column) weighs more in the mean than flat open ground (about one cell per
column). The per-cell features from step 2 can also be used directly for maps at 1.6 m resolution.

## Conventions

- **Coordinates.** Each cookie or square is in its local UTM zone, centred on its query point.
  EPT tiles are stored in Web Mercator, so reads are padded and cropped exactly after reprojection.
- **Height above ground.** By default from a 1 m raster of ground-class points, filled and
  interpolated (`method="raster"`); IDW over the nearest ground points is available
  (`method="idw"`). Tiles without ground classes fall back to a lowest-point-per-2 m surface.
- **Crash isolation.** PDAL can abort a whole process on a bad tile, so fetches run in separate
  processes: `cache.imap_isolated` for caching, and one `FetchHelper` subprocess per DataLoader
  worker for streaming.
- **Evaluation sites.** `squares.EVAL_SITES` lists the labeled site tables; squares within 500 m of
  any of them are redrawn. Add new evaluation datasets (e.g. VegBank) there.
- **Transforms are registered by name.** Importing `eagle_als.transforms` (or `data`, or
  `stream_pool`) registers them, so configs can refer to them as `dict(type="ALSFeatures")`.
- **PROJ stays offline.** `scripts/litePT/env.sh` sets `PROJ_NETWORK=OFF`. PROJ's network grid
  cache is a single SQLite file in `$HOME`, and many processes locking it over Lustre stalled data
  loading. None of the transforms here need downloaded grids.

## Adding things

- **A new augmentation or feature:** add a class to `transforms.py` with
  `@TRANSFORMS.register_module()` and a `__call__(self, data_dict)` that returns the dict. Index
  per-point arrays together (`litept.transform.index_operator`) so keys stay aligned. If it adds
  input channels, update `PackFeat` / `feat_channels` and `in_channels` in the config.
- **A new labeled dataset:** build its cookie cache with `python -m eagle_als.cache`, load it with
  `LabeledCookieDataset`, and add its site table to `squares.EVAL_SITES` so pre-training avoids it.
- **A new data source for pre-training:** produce the cookie dict above, and the transforms, model
  and trainer work unchanged.
