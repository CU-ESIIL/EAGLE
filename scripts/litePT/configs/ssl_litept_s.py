"""Self-supervised pre-training of LitePT-S on 3DEP cookies (SonataLite, see src/eagle_als/ssl.py).

Override any top-level value from the command line, e.g.
    torchrun --nproc-per-node 4 scripts/litePT/train_ssl.py --config scripts/litePT/configs/ssl_litept_s.py \
        --opts batch_size=32 total_steps=100000 run_name=ssl_s_v1
"""

import os

scratch = os.environ.get("EAGLE_SCRATCH", "/ocean/projects/bio260075p/sammlapp/eagle")
run_name = "ssl_litept_s"
out_dir = f"{scratch}/runs/{run_name}"  # resolved again after --opts (see train_ssl.py)

# ---------------- data ----------------
# one or more cookie caches built with eagle_als.cache (see slurm/cache_pretrain.sbatch)
cache_dirs = [f"{scratch}/cache/pretrain_v1"]
min_points = 2000          # skip near-empty cookies (water, data gaps)
refresh_pool = True        # re-scan the cache every epoch, so a concurrent cache job grows the pool
batch_size = 16            # total over all GPUs (samples; each sample = 2 global + 4 local views)
num_workers = 5            # per GPU (Bridges-2: 5 CPUs per V100)
prefetch_factor = 4

grid_size = 0.4            # voxel size (m) for the network input
view_radius = 50.0         # global view of scale 1.0 covers a disc of this radius (m)
feat_keys = ("coord", "hag", "intensity", "returns")

view_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.02, clip=0.1),
    dict(type="IntensityJitter", p=0.8),
    dict(type="HAGJitter", p=0.5),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
]
view_post = [
    dict(type="CenterShift", apply_z=False),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
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
        local_view_scale=(0.05, 0.2),
        max_radius=view_radius,
        max_size=65536,
        global_transform=view_aug + view_post,
        local_transform=view_aug + view_post,
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
)
ssl = dict(
    upcast_levels=2,          # point features = concat(stage4, stage3, stage2) at stage-2 resolution
    num_prototypes=4096,
    scene_prototypes=4096,
    mask_grid=6.0,            # masked columns of 6 m x 6 m on student global views
    mask_ratio=(0.4, 0.7),    # linearly increased over the first half of training
    match_grid=2.0,           # student/teacher points matched within 2 m cells (origin coords)
    teacher_temp=(0.04, 0.07),
    student_temp=0.1,
    momentum=(0.994, 1.0),
    scene_weight=0.5,
)

# ---------------- optimization ----------------
total_steps = 100_000
warmup_steps = 2_000
lr = 0.002                 # for batch_size 16, scaled linearly with batch size (lr * batch_size / 16)
block_lr_scale = 0.1       # LitePT/PTv3 recipe: lower lr for transformer/conv "block" params
weight_decay = 0.04
clip_grad = 3.0
amp_dtype = "auto"         # bf16 on Ampere/Hopper, fp16 + GradScaler on V100
empty_cache = False

# ---------------- logging / checkpoints ----------------
log_every = 20
ckpt_every = 1000
keep_ckpts = 3
seed = 0
max_hours = None           # stop cleanly (with checkpoint) after this many hours; slurm scripts set it
