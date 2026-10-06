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
simulates the range of 3DEP quality levels. `scripts/litePT/configs/ssl_litept_s.py` holds every
setting.

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

# choose the run name (default ssl_s_pool_v1); extra arguments override config values
RUN_NAME=ssl_s_pool_v2 sbatch scripts/litePT/slurm/pretrain.sbatch total_steps=80000

# 1-GPU smoke test
RUN_NAME=ssl_s_test BATCH=16 sbatch -p GPU-shared --gpus=h100-80:1 --cpus-per-task=12 -t 1:00:00 \
    scripts/litePT/slurm/pretrain.sbatch total_steps=200 log_every=10
```

A run stops itself with a checkpoint 30 minutes before the job's time limit. Resubmitting the same
command (same `RUN_NAME`) resumes from the latest checkpoint and draws new squares.

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

**Finding the run name.** It is the `RUN_NAME` you submitted with (default `ssl_s_pool_v1`). The
slurm log also prints it on the line starting `[pretrain] run`:

```bash
grep "\[pretrain\] run" $EAGLE_SCRATCH/logs/eagle-pretrain-<jobid>.out
ls $EAGLE_SCRATCH/runs/          # all runs
```

**Live dashboard (TensorBoard).** Curves update every `log_every` steps (default 20).

- VS Code: command palette → "Python: Launch TensorBoard" → `$EAGLE_SCRATCH/runs/<run_name>/tb`
  (or `$EAGLE_SCRATCH/runs` to compare runs). VS Code forwards the port.
- Terminal: `source scripts/litePT/env.sh && tensorboard --logdir $EAGLE_SCRATCH/runs --port 6006`,
  then forward port 6006.

**What to watch.**

| Metric | Healthy | Warning sign |
| --- | --- | --- |
| `loss` | starts near ln(4096) ≈ 8.3 and falls | flat, or NaN |
| `mask_kl` | falls (the student is learning the teacher's targets) | flat |
| `target_entropy`, `protos_used` (of 4096) | stay well above 0 | dropping toward 0: collapse onto a few outputs |
| `argmax_agree` | rises | stuck near 0 |
| `data_frac` | near 0 | above ~0.2: GPUs are waiting for data (more workers, or larger `pool_uses_per_square`) |
| `step_s` | steady | slowly rising |
| `mem_gb` | well under 80 | close to 80: out-of-memory risk |

Without TensorBoard: `tail -f $EAGLE_SCRATCH/logs/eagle-pretrain-<jobid>.out`, or
`python scripts/litePT/plot_ssl_log.py $EAGLE_SCRATCH/runs/<run_name>` for a PNG of the curves.

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



## Details on the website organization

The repository has two connected layers. Top-level files configure the project and its automation. The `docs/` folder contains the website content. `mkdocs.yml` tells MkDocs how to turn that content into the public site. Analysis folders hold the working scientific materials that generate the results shown on the website.

```text
.
├── README.md              # Repository overview and setup notes
├── AGENTS.md              # Guidance for coding agents and future maintainers
├── PROMPT_ACTION_LOG.md   # Record of template-level prompt-driven changes
├── mkdocs.yml             # Website navigation, theme, plugins, and edit links
├── docs/                  # Markdown source for the public website
├── scripts/               # Data processing, ingestion, and analysis scripts
├── src/                   # Re-usable code for analyses
└── datasets/              # Curated and cleaned datasets for LiDAR
```


## Preview locally

```bash
pip install -r requirements.txt
python scripts/generate_image_slots.py
python scripts/site_health.py
mkdocs serve
```

## Build site

```bash
python scripts/generate_image_slots.py
python scripts/site_health.py
mkdocs build --strict --clean
```

## Swapping homepage images

The site uses semantic image slots so postdocs do not need to edit Markdown links every time an image changes.

1. Open the relevant folder in `docs/assets/images/slots/`.
2. Delete the old image file.
3. Add one new `.png`, `.jpg`, `.jpeg`, `.webp`, or `.svg` file.
4. Run `python scripts/generate_image_slots.py`.
5. Commit the image change and the regenerated slot references.

If a slot folder contains multiple images, the generator uses the first image alphabetically and the site health report will warn you to clean it up. The cleanest workflow is still one image per slot folder.

## Using process galleries

Process galleries are folder-driven. Add files to a gallery folder, commit them, and the site updates automatically.

1. Open the relevant folder in `docs/assets/images/process/`.
2. Add images or supported deliverable files.
3. Optionally add a `captions.txt` file with lines like `filename.png | Caption text`.
4. Run `python scripts/generate_image_slots.py`.
5. Commit the new files and the regenerated gallery includes.

Supported image files:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.svg`

Supported linked deliverable files:

- `.pdf`
- `.html`
- `.csv`
- `.xlsx`
- `.docx`
- `.pptx`

## Site Health

The site generates a non-blocking health report during the build.

The report flags common issues such as missing files, placeholder links, outdated navigation, or incomplete template fields.

Warnings do not prevent the site from publishing. They are intended to help postdocs and maintainers improve the site.