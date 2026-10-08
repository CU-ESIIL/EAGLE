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
