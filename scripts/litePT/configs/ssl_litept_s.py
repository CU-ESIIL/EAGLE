"""Self-supervised pre-training of LitePT-S on 3DEP cookies, mirroring ForPT / Sonata.

Architecture and loss follow ForPT (arXiv:2609.24787) and Sonata's reference code
(Pointcept sonata_v1m1_base.py), see src/eagle_als/ssl.py. Kept ALS-specific on purpose:
0.4 m grid, larger (column) masks, extra lidar channels, density thinning, area-based views.
Sonata's metric settings (jitter, elastic distortion, match radius) are scaled by our grid
ratio 0.4 / 0.02 = 20 so they are the same in voxel units.

Override any top-level value from the command line, e.g.
    torchrun --nproc-per-node 8 scripts/litePT/train_ssl.py --config scripts/litePT/configs/ssl_litept_s.py \
        --opts batch_size=128 total_steps=50000 run_name=ssl_s_v1
"""

import os

scratch = os.environ.get("EAGLE_SCRATCH", "/ocean/projects/bio260075p/sammlapp/eagle")
run_name = "ssl_litept_s"
out_dir = f"{scratch}/runs/{run_name}"  # resolved again after --opts (see train_ssl.py)

# ---------------- data ----------------
# "pool": each DataLoader worker streams 500 m squares from 3DEP and cuts cookies from them
# (eagle_als.stream_pool, slurm/pretrain.sbatch); "cache": cookies cached with eagle_als.cache
data_mode = "pool"
cache_dirs = [f"{scratch}/cache/sites"]  # data_mode="cache" only (e.g. the overfit test)
min_points = 2000          # skip near-empty cookies (water, data gaps)
pool_size = 8              # squares held in RAM per worker (~60 MB each, on /dev/shm)
pool_uses_per_square = 24  # cookies cut from a square before it is replaced (~3 uses per cookie area)
pool_min = 2               # squares a worker needs before it starts yielding samples
pool_shm_dir = None        # default /dev/shm/eagle_pool_<pid>
pool_log_dir = None        # default <out_dir>/pool: one record per fetched / retired square
batch_size = 128           # ForPT: total over all GPUs (samples; each = 2 global + 4 local views)
max_batch_per_gpu = 8      # samples per GPU per micro-batch on an H100 (16 runs out of memory)
grad_accum = "auto"        # micro-batches per optimizer step; "auto": fewest with <= max_batch_per_gpu per GPU
                           # (8 GPUs: 2, 2 GPUs: 8, 1 GPU: 16), so batch_size stays the same on any GPU count
num_workers = "auto"       # per GPU; "auto": from the CPU cores per GPU (pool mode: each worker also runs a
                           # fetch helper, so (cores - 1) // 2; 6 on a full 104-core 8-GPU node)
prefetch_factor = 4        # batches per worker (cache mode)
pool_prefetch = 4          # samples per worker (pool mode)

grid_size = 0.4            # voxel size (m) for the network input (ForPT: 0.05 m on dense forest scans)
view_radius = 50.0         # global view of scale 1.0 covers a disc of this radius (m)
feat_keys = ("coord", "hag", "intensity", "returns")

# Sonata view augmentations (probabilities as in Sonata, metric values x20) + ALS-specific ones
view_aug = [
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.1, clip=0.4),
    dict(type="ElasticDistortion", distortion_params=[[4.0, 8.0], [16.0, 32.0]]),
    dict(type="IntensityJitter", p=0.8),
    dict(type="HAGJitter", p=0.5),
]
view_post = [
    dict(type="CenterShift", apply_z=False),  # keep z = height above median ground (Sonata: apply_z=True)
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train"),
    dict(type="PackFeat", keys=feat_keys),
]
train_transform = [
    dict(type="ALSFeatures"),
    # normalize pulse density into a common range (also an augmentation over 3DEP quality levels)
    dict(type="RandomPointThinning", target_density=(2.0, 12.0), radius=100.0, p=1.0),
    dict(type="ReturnDrop", p=0.05),
    dict(
        type="ALSMultiViewGenerator",
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),   # Sonata's local scale (here: fraction of area, not of points)
        max_radius=view_radius,
        max_size=65536,
        global_transform=view_aug + view_post,
        local_transform=view_aug + view_post,
        keep_keys=("coord", "origin_coord", "feat"),
    ),
]

# ---------------- model ----------------
backbone = dict(
    type="LitePT",
    in_channels=8,  # = eagle_als.transforms.feat_channels(feat_keys)
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(36, 72, 144, 252, 504),
    enc_num_head=(2, 4, 8, 14, 28),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    enc_conv=(True, True, True, False, False),
    enc_attn=(False, False, False, True, True),
    enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
    mlp_ratio=4,
    qkv_bias=True,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enc_mode=True,            # ForPT: pre-train the encoder only
    norm="ln",                # ForPT / Sonata: LayerNorm instead of BatchNorm
    embedding="linear",       # ForPT / Sonata: linear embedding instead of sparse conv stem
    mask_token=True,
)
ssl = dict(
    teacher_custom=dict(attn_drop=0.0, proj_drop=0.0, drop_path=0.0),
    head_in_channels=144 + 252 + 504,   # up-cast stage 2 + 3 + 4 features
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_dims=2,              # ALS: mask xy columns (Sonata: 3D cubes)
    mask_size_start=2.0,      # m; larger masks than ForPT's 5 cm (sparse ALS)
    mask_size_base=6.0,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,      # ForPT: 0.7
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.2,          # Sonata 0.01 m x20
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,   # ForPT: 0.07
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,         # ForPT: 0.1
    mask_loss_weight=2 / 8,   # ForPT L = L_l2g + L_m2g with Sonata's weights
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,      # ForPT: fixed 0.994 (Sonata: cosine 0.994 -> 1)
    momentum_final=0.994,
    match_max_r=6.4,          # Sonata 0.32 m x20
    up_cast_level=2,
    grid_size=grid_size,
    sinkhorn_cookies=128,     # cookies per Sinkhorn normalization over all GPUs (ForPT batch), via a
                              # Vernata-style memory bank of recent cookies on each GPU; 0 = micro-batch only
    sinkhorn_points_per_cookie=2048,  # target points queued per cookie (weighted up to its full mass)
)

# ---------------- optimization (ForPT) ----------------
total_steps = 50_000
lr = 0.001                 # ForPT; no batch-size scaling
lr_schedule = "constant"   # "constant" (ForPT, after warmup), "cosine" (decay after warmup) or "onecycle" (Sonata)
warmup_steps = 2_500       # linear warmup from 0 over the first 5% of training, like Sonata and the
                           # teacher-temperature / mask warmups; the overfit tests fitted better with one
layer_decay = 0.9          # layer-wise lr decay over encoder blocks (enc{e}.block{b})
weight_decay = 1e-4        # ForPT (Sonata: 0.04 -> 0.2 cosine)
clip_grad = 3.0            # Sonata
amp_dtype = "auto"         # bf16 on Ampere/Hopper (Sonata), fp16 + GradScaler on V100
empty_cache = False

# ---------------- validation tasks on the frozen teacher encoder (eagle_als.evaluation) ----------------
# run on rank 0 every eval_every steps and at step 0 of a fresh run (random-init baseline); headline
# metrics in tensorboard eval/<name>/*, per-class scores in eval_<name>/*. Each entry: dict(type=, name=,
# ...); type "classification" (eagle_als.probe) takes table=, cache_dir=, label_col=, id_col="als_site_id",
# split_col=None (-> 5-fold CV), query=None, probes=("linear", "knn"). Empty list: no evaluation.
def _cache_dir(name):
    """The node-local copy staged by stage_local.sh when the job has one, else the cache on /ocean."""
    local = os.path.join(os.environ.get("EAGLE_LOCAL_CACHE", ""), name)
    return local if os.environ.get("EAGLE_LOCAL_CACHE") and os.path.isdir(local) else f"{scratch}/cache/{name}"


eval_tasks = [
    # NLCD 2021 reference land cover (datasets/NLCD_eval/README.md): 16 Level II classes, class-balanced
    # subset (<= 100 per class), spatial hold-out by 1-degree blocks; perennial ice/snow (1 row) is dropped
    dict(type="classification", name="nlcd", table="datasets/NLCD_eval/nlcd_lidar_eval.parquet",
         cache_dir=_cache_dir("nlcd"), label_col="nlcd_class", split_col="test_split", query="balanced_subset"),
]
eval_every = 1000
eval_knn_k = 20

# ---------------- logging / checkpoints ----------------
log_every = 20
ckpt_every = 1000
keep_ckpts = 3
seed = 0
max_hours = None           # stop cleanly (with checkpoint) after this many hours; slurm scripts set it
