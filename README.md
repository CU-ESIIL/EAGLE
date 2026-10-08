# EAGLE: Extracting Airborne Gestalts from Lidar for Ecology

## Project description

EAGLE pre-trains a point-cloud encoder (LitePT-S, 12.4M parameters) on USGS 3DEP aerial lidar with
self-supervised learning, so that a 100 m-radius patch of lidar ("cookie") can be summarised as an
embedding useful for ecology. The embeddings are evaluated on labeled sites: VegBank habitat
classification (main target), Breeding Bird Survey routes, butterfly observations and OFO forest plots.

Detailed, dated working notes (measurements, decisions, open issues) live in the project's working
notes document; `PROMPT_ACTION_LOG.md` records each change to the code.

## Pre-training design

**Model and loss.** The recipe follows ForPT (arXiv:2609.24787), a Sonata-style self-distillation
of the LitePT-S encoder (`src/eagle_als/ssl.py`). A student network learns to match an
exponential-moving-average teacher across 2 global and 4 local views of each cookie, with columns of
the global views masked out. Inputs are 8 channels per point (xyz, height above ground, intensity
rank, return features) on a 0.4 m grid. Augmentations include thinning to 2–12 points/m², which
simulates the range of 3DEP quality levels. The learning rate (1e-3, AdamW with layer-wise decay)
warms up linearly over the first 2,500 steps (5%) and then stays constant, as in ForPT.
`scripts/litePT/configs/ssl_litept_s.py` holds every setting.

**Data: streamed, not cached.** Training draws on all of 3DEP rather than a fixed dataset:

```text
3DEP EPT tiles (S3)
   │  one PDAL read per 500 m square, thinned to 12 pts/m², height above ground from a 1 m ground raster
   ▼
fetch helper process (one per DataLoader worker; a PDAL crash only restarts the helper)
   │  shard written to /dev/shm
   ▼
DataLoader worker: pool of 8 squares in RAM → random 100 m cookie → views + augmentations
   │  each square retired after 24 cookies (~3 uses per cookie-sized area)
   ▼
batches mixed from several workers → 8 H100s (DDP)
```

- Squares come from a seeded sampler (`src/eagle_als/squares.py`): tiles are weighted by
  sqrt(area), and squares within 500 m of any evaluation site are redrawn so evaluation data is never
  seen in pre-training.
- One large read per square is 2–5× cheaper per km² than many cookie-sized reads, which is why the
  workers fetch squares and cut several cookies from each.
- Each job copies the Python environment and the evaluation cookies onto node-local disk first
  (`scripts/litePT/stage_local.sh`). /ocean reads large files quickly but can stall for minutes on
  the many small reads that Python imports make.
- Code: `src/eagle_als/stream_pool.py` (pool dataset, fetch helper, batching), `squares.py`
  (sampler, square fetch, cookie cutting), `fetch.py` (PDAL reads, height above ground),
  `transforms.py` (features and views).

## Running pre-training

All commands run from the repository root on Bridges-2.

```bash
# full run: one 8-H100 node, batch 128 (8 per GPU x 2 accumulation steps), up to 48 h
sbatch scripts/litePT/slurm/pretrain.sbatch

# choose the run name (default ssl_s_pool_v1); arguments are config overrides (key=value)
sbatch scripts/litePT/slurm/pretrain.sbatch run_name=ssl_s_pool_v2 total_steps=80000

# 2 H100s on GPU-shared (shorter queue): same batch 128, as 8 per GPU x 8 accumulation steps
sbatch -p GPU-shared --gpus=h100-80:2 --cpus-per-task=24 scripts/litePT/slurm/pretrain.sbatch run_name=ssl_s_2gpu

# 1-GPU smoke test
sbatch -p GPU-shared --gpus=h100-80:1 --cpus-per-task=13 -t 1:00:00 scripts/litePT/slurm/pretrain.sbatch \
    run_name=ssl_s_test batch_size=16 total_steps=200 log_every=10
```

The same script runs on any number of GPUs. `train_ssl.py` keeps the global `batch_size` and works out
the gradient accumulation (at most `max_batch_per_gpu` = 8 samples per GPU per micro-batch) and the
DataLoader workers per GPU (from the CPU cores per GPU) for the job it is in, so the training recipe
is unchanged; only the time per step changes (about 13 s on 1 GPU). On GPU-shared, ask for 13 CPUs per
GPU, the full-node share (104 cores for 8 GPUs). `batch_size` must divide by the number of GPUs.

Pass settings as arguments, not environment variables: exported variables do not reach the job on
Bridges-2. A run stops itself with a checkpoint 30 minutes before the job's time limit.
Resubmitting the same command (same `run_name`) resumes from the latest checkpoint and draws new
squares.

Everything for a run is written to `$EAGLE_SCRATCH/runs/<run_name>/`
(`$EAGLE_SCRATCH` = `/ocean/projects/bio260075p/sammlapp/eagle`):

| Path | Contents |
| --- | --- |
| `config.json` | the resolved config |
| `log.txt` | one JSON line per logged step |
| `tb/` | TensorBoard logs |
| `checkpoints/` | the last 3 checkpoints (resume points) |
| `backbone_teacher.pth` | exported encoder weights for downstream use |
| `pool/` | one record per fetched and retired square, per worker |

The slurm log is `$EAGLE_SCRATCH/logs/eagle-pretrain-<jobid>.out`.

## Monitoring training

**Finding the run name.** It is the `run_name=` you submitted with (default `ssl_s_pool_v1`). The
slurm log also prints it on the line starting `[pretrain] run`:

```bash
grep "\[pretrain\] run" $EAGLE_SCRATCH/logs/eagle-pretrain-<jobid>.out
ls $EAGLE_SCRATCH/runs/          # all runs
```

**Live dashboard (TensorBoard).** Curves update every `log_every` steps (default 20).

- VS Code: command palette → "Python: Launch TensorBoard" → `/ocean/projects/bio260075p/sammlapp/eagle/runs`. VS Code forwards the port.
- Terminal: `source scripts/litePT/env.sh && tensorboard --logdir $EAGLE_SCRATCH/runs --port 6006`,  
then forward port 6006.

**What to watch.** `scripts/litePT/TENSORBOARD.md` explains every curve. It is also shown in each
run's TensorBoard **Text** tab. The short version:

| Metric | Healthy | Warning sign |
| --- | --- | --- |
| `eval/*` validation probes (e.g. `eval/nlcd/linear_f1`) | rise above the step-0 (random init) value | flat: the encoder is not learning anything useful for the labels |
| `loss` = `target_entropy` + `mask_kl` | starts near ln(4096) ≈ 8.3, falls, then flattens (normal) | NaN, or rising after the warmups end |
| `mask_kl` | falls slowly (the student is learning the teacher's targets) | rising |
| `protos_used`, `student_protos_used` (of 4096) | stable, hundreds to thousands | dropping toward a few: collapse (confirm with `check_embeddings.py`) |
| `grad_norm` | O(0.1–1) | → 0 (collapse), or often above `clip_grad` |
| `data_frac` | near 0 | above ~0.2: GPUs are waiting for data |
| `mem_gb` (peak since last log), `mem_max_gb` (peak since start) | well under 80; `mem_max_gb` steps up on rare large batches | close to 80: out-of-memory risk |

A flat loss is not a reason to stop a run. The loss cannot fall below `target_entropy`, and the
teacher-temperature warmup raises that floor. Judge runs on the probes.

Without TensorBoard: `tail -f $EAGLE_SCRATCH/logs/eagle-pretrain-<jobid>.out`, or
`python scripts/litePT/plot_ssl_log.py $EAGLE_SCRATCH/runs/<run_name>` for a PNG of the curves.

**Validation tasks during training.** Every `eval_every` steps (default 300), and once at step 0 for
the random-initialisation baseline, rank 0 runs the tasks listed in `eval_tasks` on the frozen
teacher encoder (`src/eagle_als/evaluation.py`); the other GPUs wait. The default task is `nlcd`:
NLCD 2021 reference land cover (`datasets/NLCD_eval/README.md`), 15 Level II classes in the
class-balanced subset, scored on a spatial hold-out (1-degree blocks). Its cookies are cached in
`$EAGLE_SCRATCH/cache/nlcd` and copied to node-local disk by the slurm job.

A `classification` task (`src/eagle_als/probe.py`) is a cookie cache (`python -m eagle_als.cache`) plus
a table with one labeled row per cookie:

```python
eval_tasks = [dict(type="classification", name="nlcd", table="datasets/NLCD_eval/nlcd_lidar_eval.parquet",
                   cache_dir=_cache_dir("nlcd"), label_col="nlcd_class",
                   split_col="test_split",       # train/test values or bool (True = test); omit for 5-fold CV
                   query="balanced_subset",      # optional pandas row filter
                   probes=("linear", "knn"))]    # optional, default both
```

`id_col` (default `als_site_id`) is the cookie's id in the cache. Classes with fewer than 5 cached
cookies are dropped. Rank 0 embeds every cookie (one fixed view: 8 pts/m², central 50 m disc, no
augmentation, mean of the up-cast point features) and fits logistic regression and cosine kNN. In
TensorBoard, `eval/<name>/` holds balanced accuracy and macro F1 of each probe, and `eval_<name>/`
holds the F1 of each class. Keep a task to roughly 2,000 cookies or fewer. Its sites must be listed in
`squares.EVAL_SITES`, so pre-training never sees them. To add a new kind of task, subclass `EvalTask`
and register it (see the docstring of `evaluation.py`). To score the saved checkpoints of an earlier
run, use `python scripts/litePT/eval_checkpoints.py $EAGLE_SCRATCH/runs/<run_name> --init --tb` on a GPU.

## Planned experiments

1. **Baseline pre-training** with the current config (`ssl_s_pool_v1`).
2. **Downstream evaluation** of frozen and fine-tuned embeddings (linear probe, kNN, fine-tuning):
   VegBank habitat classification first, then BBS, butterflies and OFO.
3. **Height inputs:** z and HAG together vs z only vs HAG only.
4. **Return features:** a 4-way one-hot (only, first of many, intermediate, last of many) plus a
   clipped number of returns, vs the current encoding.
5. **ForPT details** to confirm against the authors' code when it is released: warmups, Sinkhorn
   targets, and whether local views are matched to one or both global views.
6. **Data efficiency:** reading coarser EPT levels for dense tiles; reuse per square
   (`pool_uses_per_square`).
7. **Further ideas:** a landscape-context view (e.g. 1 km × 1 km at ~0.1 pts/m²) alongside the
   detailed cookie, and contrastive views from overlapping flight lines (`point_source_id`) of the
   same place.
