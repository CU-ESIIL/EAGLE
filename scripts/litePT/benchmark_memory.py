"""Benchmark GPU memory and step time of SSL pre-training (or classification) vs. batch size.

Samples are produced by the real data pipeline (config transforms on cached cookies), pre-computed
once on CPU and re-used, so the numbers reflect realistic point counts. For each batch size it runs
a few full training steps (forward + backward + optimizer + EMA) and records the peak allocated
memory; it stops at the first out-of-memory.

    python scripts/litePT/benchmark_memory.py --config scripts/litePT/configs/ssl_litept_s.py \
        --batch-sizes 2 4 8 12 16 24 32 --out $EAGLE_SCRATCH/benchmarks/ssl_s_h100

Writes <out>/benchmark.csv and <out>/benchmark.png. The `peak_gb` of the largest batch size that fits
with ~15% headroom is a good per-GPU batch size (point counts vary between batches).
"""

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eagle_als.data import CookiePoolDataset, collate_points
from eagle_als.train_utils import amp_dtype, load_config, param_groups

_DS = None


def _init(cfg):
    global _DS
    import eagle_als.transforms  # noqa: F401

    _DS = CookiePoolDataset(cfg["cache_dirs"], cfg["train_transform"], min_points=cfg["min_points"])


def _sample(i):
    np.random.seed(i)
    import random

    random.seed(i)
    return _DS[i % len(_DS)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--opts", nargs="*", default=[])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 12, 16, 24, 32])
    p.add_argument("--steps", type=int, default=6, help="steps per batch size (first 2 are warm-up)")
    p.add_argument("--n-samples", type=int, default=96)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = load_config(a.config, a.opts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = amp_dtype(cfg.get("amp_dtype", "auto"))
    gpu = torch.cuda.get_device_name()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[bench] {gpu} ({total_gb:.0f} GB), amp={dtype}", flush=True)

    t0 = time.time()
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        samples = pool.map(_sample, range(a.n_samples))
    print(f"[bench] prepared {len(samples)} samples in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) * a.workers / len(samples):.2f} s/sample/worker)", flush=True)

    from eagle_als.ssl import SonataLite

    model = SonataLite(cfg["backbone"], **cfg["ssl"]).to(device)
    student = torch.nn.ModuleList(model.student_modules())
    opt = torch.optim.AdamW(param_groups(student, 1e-4, 0.04, cfg.get("block_lr_scale", 0.1)))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    rows = []
    for bs in a.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times, npts, status = [], [], "ok"
        try:
            for s in range(a.steps):
                idx = [(s * bs + j) % len(samples) for j in range(bs)]
                batch = collate_points([samples[i] for i in idx])
                npts.append(int(batch["global_coord"].shape[0] + batch.get("local_coord", torch.zeros(0)).shape[0]))
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                torch.cuda.synchronize()
                t = time.time()
                with torch.autocast("cuda", dtype=dtype):
                    losses = model(batch, progress=0.5)
                opt.zero_grad(set_to_none=True)
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
                scaler.step(opt)
                scaler.update()
                model.update_teacher(0.5)
                torch.cuda.synchronize()
                if s >= 2:
                    times.append(time.time() - t)
                if not torch.isfinite(losses["loss"]):
                    status = "nan_loss"
        except torch.OutOfMemoryError:
            status = "oom"
        except RuntimeError as e:  # spconv / flash-attn raise RuntimeError on OOM
            status = "oom" if "out of memory" in str(e).lower() else f"error: {e}"[:200]
        peak = torch.cuda.max_memory_allocated() / 1e9
        row = dict(gpu=gpu, batch_size=bs, status=status, peak_gb=round(peak, 2),
                   step_s=round(float(np.mean(times)), 3) if times else None,
                   samples_per_s=round(bs / float(np.mean(times)), 2) if times else None,
                   mean_points=int(np.mean(npts)) if npts else None,
                   loss=float(losses["loss"]) if status == "ok" else None)
        rows.append(row)
        print("[bench]", json.dumps(row), flush=True)
        del batch
        if status != "ok":
            opt.zero_grad(set_to_none=True)
            if status == "oom":
                break

    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark.csv", index=False)
    ok = df[df.status == "ok"]
    if len(ok):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(ok.batch_size, ok.peak_gb, "o-")
        ax[0].axhline(total_gb, color="r", ls="--", label="GPU memory")
        ax[0].set(xlabel="batch size (samples / GPU)", ylabel="peak memory (GB)", title=gpu)
        ax[0].legend()
        ax[1].plot(ok.batch_size, ok.samples_per_s, "o-")
        ax[1].set(xlabel="batch size (samples / GPU)", ylabel="samples / s")
        fig.tight_layout()
        fig.savefig(out / "benchmark.png", dpi=120)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
