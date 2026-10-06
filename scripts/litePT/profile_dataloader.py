"""Profile the SSL data pipeline: per-transform CPU cost and DataLoader throughput vs. worker count.

    python scripts/litePT/profile_dataloader.py --config scripts/litePT/configs/ssl_litept_s.py \
        --opts "cache_dirs=['$EAGLE_SCRATCH/cache/sites']" --stages 40 --workers 1 2 4 8 --loader-samples 64

Streamed squares (data_mode="pool") are profiled with profile_pool.py instead.

stages: one process, `--stages` cookies; time of np.load and of every transform (inside the view
        generator, each view transform is summed over the 6 views), plus point counts.
loader: a torch DataLoader per worker count (spawn, like train_ssl.py); reports the time to the
        first batch (worker start-up + imports) and the steady-state samples/s after it.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from eagle_als.train_utils import load_config


class Timed:
    def __init__(self, name, fn, acc):
        self.name, self.fn, self.acc = name, fn, acc

    def __call__(self, d):
        t = time.perf_counter()
        out = self.fn(d)
        self.acc[self.name] += time.perf_counter() - t
        return out


def profile_stages(cfg, n):
    from eagle_als.data import CookiePoolDataset
    from eagle_als.fetch import load_cookie

    ds = CookiePoolDataset(cfg["cache_dirs"], cfg["train_transform"], min_points=cfg["min_points"])
    rng = np.random.default_rng(0)
    paths = [ds.paths[i] for i in rng.choice(len(ds), min(n, len(ds)), replace=False)]
    acc = defaultdict(float)
    tf = ds.transform.transforms
    for i, t in enumerate(tf):
        name = type(t).__name__
        if name == "ALSMultiViewGenerator":
            for comp, prefix in ((t.global_transform, "global"), (t.local_transform, "local")):
                comp.transforms = [Timed(f"  {prefix}.{type(s).__name__}", s, acc) for s in comp.transforms]
        tf[i] = Timed(name, t, acc)
    rows = []
    for k, p in enumerate(paths):
        before = dict(acc)
        t0 = time.perf_counter()
        cookie = load_cookie(p)[0]
        t_load = time.perf_counter() - t0
        n_raw = len(cookie["xyz"])
        out = ds.transform(cookie)
        t_total = time.perf_counter() - t0
        row = dict(i=k, n_points=n_raw, mb=os.path.getsize(p) / 1e6, load_s=t_load, total_s=t_total,
                   n_global=int(out["global_offset"][-1]), n_local=int(out["local_offset"][-1]))
        row.update({name: acc[name] - before.get(name, 0.0) for name in acc})
        rows.append(row)
    import pandas as pd

    df = pd.DataFrame(rows)
    print(f"\n[stages] {len(df)} cookies, single process")
    print(df[["n_points", "mb", "load_s", "total_s", "n_global", "n_local"]].describe().round(3).to_string())
    stage_cols = [c for c in df.columns if c not in ("i", "n_points", "mb", "load_s", "total_s", "n_global", "n_local")]
    summ = pd.DataFrame({"mean_s": df[["load_s"] + stage_cols].mean(), "first_s": df[["load_s"] + stage_cols].iloc[0]})
    summ["pct_of_total"] = 100 * summ.mean_s / df.total_s.mean()
    print(summ.round(4).to_string())
    print(f"first sample {df.total_s.iloc[0]:.3f}s, rest mean {df.total_s.iloc[1:].mean():.3f}s "
          f"(corr total_s ~ n_points: {np.corrcoef(df.n_points, df.total_s)[0, 1]:.2f})")
    return df


def profile_loader(cfg, workers, n_samples, batch_size):
    import torch
    from torch.utils.data import DataLoader

    from eagle_als.data import CookiePoolDataset, collate_points
    from eagle_als.train_utils import InfiniteRandomSampler, worker_init_fn

    ds = CookiePoolDataset(cfg["cache_dirs"], cfg["train_transform"], min_points=cfg["min_points"])
    rows = []
    for w in workers:
        t0 = time.perf_counter()
        loader = DataLoader(
            ds, batch_size=batch_size, sampler=InfiniteRandomSampler(ds, seed=w), num_workers=w,
            collate_fn=collate_points, drop_last=True, worker_init_fn=worker_init_fn,
            prefetch_factor=cfg["prefetch_factor"], multiprocessing_context="spawn", persistent_workers=False,
        )
        it = iter(loader)
        next(it)
        t_first = time.perf_counter() - t0
        # discard one batch per worker (the remaining start-up skew), then time the steady state
        for _ in range(w - 1):
            next(it)
        t1 = time.perf_counter()
        n = 0
        while n < n_samples:
            n += next(it)["num_samples"]
        dt = time.perf_counter() - t1
        del it, loader
        row = dict(workers=w, first_batch_s=round(t_first, 1), steady_samples_per_s=round(n / dt, 2),
                   s_per_sample_per_worker=round(dt * w / n, 3))
        rows.append(row)
        print("[loader]", json.dumps(row), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--opts", nargs="*", default=[])
    p.add_argument("--stages", type=int, default=40, help="cookies for the per-transform profile (0: skip)")
    p.add_argument("--workers", type=int, nargs="*", default=[1, 2, 4, 8])
    p.add_argument("--loader-samples", type=int, default=64, help="steady-state samples timed per worker count")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", default=None, help="optional directory for stages.csv / loader.csv")
    a = p.parse_args()
    cfg = load_config(a.config, a.opts)
    print(f"[profile] host {os.uname().nodename}, {len(os.sched_getaffinity(0))} cpus, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    df = profile_stages(cfg, a.stages) if a.stages else None
    rows = profile_loader(cfg, a.workers, a.loader_samples, a.batch_size) if a.workers else []
    if a.out:
        import pandas as pd

        Path(a.out).mkdir(parents=True, exist_ok=True)
        if df is not None:
            df.to_csv(Path(a.out) / "stages.csv", index=False)
        pd.DataFrame(rows).to_csv(Path(a.out) / "loader.csv", index=False)


if __name__ == "__main__":
    main()
