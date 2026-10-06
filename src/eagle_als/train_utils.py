"""Small training utilities shared by the SSL and classification scripts (config, DDP, checkpoints)."""

import ast
import math
import os
import random
import runpy
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


# ---------------- config ----------------
def load_config(path, opts=()):
    """Load a python config file (LitePT style) and apply `key=value` overrides (python literals)."""
    cfg = {k: v for k, v in runpy.run_path(str(path)).items() if not k.startswith("_") and not callable(v)}
    cfg = {k: v for k, v in cfg.items() if not hasattr(v, "__file__")}  # drop imported modules
    for opt in opts:
        key, val = opt.split("=", 1)
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass  # plain string
        d, parts = cfg, key.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = val
    if "run_name" in cfg and "scratch" in cfg and not any(o.startswith("out_dir=") for o in opts):
        cfg["out_dir"] = f"{cfg['scratch']}/runs/{cfg['run_name']}"
    return cfg


# ---------------- distributed ----------------
def setup_distributed():
    """Init torch.distributed from torchrun env vars. Returns (rank, local_rank, world_size)."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return rank, local_rank, world


def is_main():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_mean(t):
    if dist.is_available() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return t


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


class InfiniteRandomSampler(torch.utils.data.Sampler):
    """Endless random indices, different per rank; dataset size may change between epochs."""

    def __init__(self, dataset, seed=0, rank=0, world=1):
        self.dataset, self.seed, self.rank, self.world = dataset, seed, rank, world

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000 + self.rank)
        while True:
            yield from torch.randperm(len(self.dataset), generator=g).tolist()


# ---------------- optimization ----------------
def amp_dtype(name="auto"):
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def param_groups(model, lr, weight_decay, block_lr_scale=1.0, keyword="block"):
    """AdamW groups: no decay for norms/biases/tokens; lower lr for backbone `block` params (LitePT recipe)."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "token" in name
        is_block = keyword in name
        key = (no_decay, is_block)
        if key not in groups:
            groups[key] = dict(
                params=[], weight_decay=0.0 if no_decay else weight_decay,
                lr=lr * (block_lr_scale if is_block else 1.0), base_lr=lr * (block_lr_scale if is_block else 1.0),
            )
        groups[key]["params"].append(p)
    return list(groups.values())


def layerwise_param_groups(model, lr, weight_decay, enc_depths, layer_decay=0.9):
    """Pointcept/Sonata layer-wise lr decay: parameters of encoder block `enc{e}.block{b}.` get
    lr * layer_decay ** (blocks after it); everything else (embedding, pooling, heads) gets lr.
    Weight decay applies to all parameters, as in Pointcept."""
    n_blocks = sum(enc_depths)
    keywords = {}
    for e in range(len(enc_depths)):
        for b in range(enc_depths[e]):
            keywords[f"enc{e}.block{b}."] = lr * layer_decay ** (n_blocks - sum(enc_depths[:e]) - b - 1)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group_lr = next((v for k, v in keywords.items() if k in name), lr)
        g = groups.setdefault(group_lr, dict(params=[], lr=group_lr, base_lr=group_lr, weight_decay=weight_decay))
        g["params"].append(p)
    return list(groups.values())


def lr_factor(step, total, warmup=0, schedule="constant"):
    """Multiplier on each group's base lr. 'onecycle' mimics Sonata's OneCycleLR
    (pct_start=0.05, cosine, div_factor=10, final_div_factor=1000)."""
    if schedule == "constant":
        return min(1.0, (step + 1) / warmup) if warmup else 1.0
    if schedule == "onecycle":
        up = max(1, int(0.05 * total))
        if step < up:
            return 0.1 + 0.9 * 0.5 * (1 - math.cos(math.pi * step / up))
        final = 0.1 / 1000
        prog = min(1.0, (step - up) / max(1, total - up))
        return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * prog))
    if schedule == "cosine":
        return cosine_lr(step, total, warmup)
    raise ValueError(schedule)


def cosine_lr(step, total, warmup, final_ratio=1e-3):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def move_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


# ---------------- checkpoints ----------------
def save_checkpoint(out_dir, state, step, keep=3):
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pth"
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)
    (ckpt_dir / "latest.txt").write_text(path.name)
    old = sorted(ckpt_dir.glob("step_*.pth"))[:-keep] if keep else []
    for p in old:
        p.unlink(missing_ok=True)
    return path


def latest_checkpoint(out_dir):
    f = Path(out_dir) / "checkpoints" / "latest.txt"
    if f.exists():
        p = f.parent / f.read_text().strip()
        if p.exists():
            return p
    return None


class StopFlag:
    """Set when slurm sends SIGUSR1/SIGTERM (use `#SBATCH --signal=B:USR1@300`) or max_hours elapses."""

    def __init__(self, max_hours=None):
        self.flag = False
        self.t0 = time.time()
        self.max_hours = max_hours
        for s in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(s, self._handler)

    def _handler(self, *_):
        self.flag = True

    def should_stop(self):
        if self.max_hours is not None and time.time() - self.t0 > self.max_hours * 3600:
            self.flag = True
        t = torch.tensor([float(self.flag)], device="cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())
