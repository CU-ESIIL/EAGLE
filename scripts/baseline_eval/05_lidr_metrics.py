"""Standard lidar metrics (lidR::stdmetrics + terrain) for every cached benchmark cookie (CPU; resumable).

    pixi run --manifest-path envs/eagle-bench/pixi.toml python scripts/baseline_eval/05_lidr_metrics.py
    ... --limit 50                      # trial
    ... --radius-m 50                   # analysis disc (default 50 m, as for ForPT and the EAGLE probes)
    ... --workers 28                    # batches in parallel (one single-threaded Rscript each; ~1.4 s/cookie in R)

For each batch of cookies: crop to the disc, drop noise points (classes 7, 18), clamp heights below ground to 0, write the points to a local temporary parquet, run
scripts/baseline_eval/lidr_metrics.R in the lidR pixi env (envs/lidr), and save
<out>/r<radius>/part_*.parquet (one row per als_site_id; see the R script for the columns).
Probe with and without the terrain columns:

    python scripts/baseline_eval/02_linear_probe.py --embeddings $BENCH/features/lidr_stdmetrics/r50 --model lidr_std_terrain_r50
    python scripts/baseline_eval/02_linear_probe.py --embeddings $BENCH/features/lidr_stdmetrics/r50 --exclude "^terrain_|^n_ground$" --model lidr_std_r50
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als.cache import cookie_path, read_manifest  # noqa: E402
from eagle_als.fetch import load_cookie  # noqa: E402

BENCH = Path("/home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking")
PIXI = Path.home() / ".pixi/bin/pixi"
NOISE_CLASSES = (7, 18)  # ASPRS low / high noise


def cookie_points(sid, cache, radius):
    c, meta = load_cookie(cookie_path(cache, sid))
    xyz = c["xyz"]
    # noise classes 7 / 18 are still in the cache (the PDAL range filter in eagle_als.fetch ORs its ranges)
    keep = (xyz[:, 0] ** 2 + xyz[:, 1] ** 2 <= radius**2) & ~np.isin(c["classification"], NOISE_CLASSES)
    return pd.DataFrame({
        "site": sid, "X": xyz[keep, 0].astype(np.float64), "Y": xyz[keep, 1].astype(np.float64),
        # heights below ground clamped to 0, as in lidR practice (normalized Z < 0 breaks e.g. zentropy)
        "Z": np.clip(c["hag"][keep], 0, None).astype(np.float64), "Zabs": xyz[keep, 2].astype(np.float64),
        "Intensity": c["intensity"][keep].astype(np.int32), "ReturnNumber": c["return_number"][keep].astype(np.int32),
        "NumberOfReturns": c["number_of_returns"][keep].astype(np.int32),
        "Classification": c["classification"][keep].astype(np.int32)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(BENCH / "als_cache"))
    ap.add_argument("--out", default=str(BENCH / "features/lidr_stdmetrics"))
    ap.add_argument("--radius-m", type=float, default=50.0)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="batches processed in parallel")
    a = ap.parse_args()

    out = Path(a.out) / f"r{a.radius_m:g}"
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.glob("part_*.parquet"):
        done |= set(pd.read_parquet(f, columns=["als_site_id"])["als_site_id"])
    sites = pd.read_parquet(Path(a.cache) / "sites.parquet")["als_site_id"]
    # cached = status "ok" in the cache manifest (stat-ing 86k files on the iRODS mount takes many minutes)
    cached = set(read_manifest(a.cache).query("status == 'ok'")["site_id"].astype(str))
    todo = [s for s in sites if s not in done and s in cached]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[lidr] {len(done)} done, {len(todo)} cached cookies to process -> {out}", flush=True)
    t0, first_shard = time.time(), len(list(out.glob("part_*.parquet")))
    batches = [todo[s : s + a.batch] for s in range(0, len(todo), a.batch)]
    env = dict(os.environ, OMP_NUM_THREADS="1")  # one core per Rscript; parallelism comes from --workers

    def run(k, ids, tmp):
        frames = []
        for sid in ids:
            try:
                frames.append(cookie_points(sid, a.cache, a.radius_m))
            except Exception as e:  # unreadable cookie: skipped, reported
                print(f"[lidr] {sid}: {type(e).__name__}: {e}", flush=True)
        if not frames:
            return 0
        src, dst = Path(tmp) / f"points_{k}.parquet", Path(tmp) / f"metrics_{k}.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(src, index=False)
        subprocess.run([str(PIXI), "run", "--manifest-path", str(ROOT / "envs/lidr/pixi.toml"), "Rscript",
                        str(ROOT / "scripts/baseline_eval/lidr_metrics.R"), str(src), str(dst)], check=True, env=env)
        pd.read_parquet(dst).to_parquet(out / f"part_{first_shard + k:05d}.parquet", index=False)
        src.unlink()
        dst.unlink()
        return len(ids)

    n = 0
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(run, k, ids, tmp) for k, ids in enumerate(batches)]):
            n += f.result()
            rate = n / (time.time() - t0)
            print(f"[lidr] {n}/{len(todo)}, {rate:.2f} cookies/s, ETA {(len(todo) - n) / rate / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
