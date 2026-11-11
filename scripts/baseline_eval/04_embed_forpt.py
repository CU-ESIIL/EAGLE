"""Embed every cached benchmark cookie with the frozen ForPT backbone (GPU; resumable).

    python scripts/baseline_eval/04_embed_forpt.py                            # default variant: 50 m disc, z
    python scripts/baseline_eval/04_embed_forpt.py --height hag --radius-m 50   # variant: height above ground
    python scripts/baseline_eval/04_embed_forpt.py --limit 200                  # quick trial

Reads cookies from <cache>/cookies (eagle_als.cache layout, built from 03_lidar_sites.py's table) and writes
<out>/<variant>/part_*.parquet: als_site_id, n_input (points after crop + 5 cm grid), f0000..f1007.
Cookies already in a shard are skipped. Cookies are packed into forward passes of up to --point-budget points
(the per-pass overhead dominates on small clouds). See eagle_bench/forpt.py for the preprocessing.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als.cache import cookie_path, read_manifest  # noqa: E402
from eagle_als.fetch import load_cookie  # noqa: E402
from eagle_bench import forpt  # noqa: E402

BENCH = Path("/home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(BENCH / "als_cache"))
    ap.add_argument("--out", default=str(BENCH / "embeddings/forpt"))
    ap.add_argument("--radius-m", type=float, default=50.0)
    ap.add_argument("--height", default="z", choices=["z", "hag"])
    ap.add_argument("--max-points", type=int, default=400_000, help="thin denser crops before the grid")
    ap.add_argument("--point-budget", type=int, default=200_000, help="points per forward pass")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--io-workers", type=int, default=3)
    a = ap.parse_args()

    out = Path(a.out) / f"r{a.radius_m:g}_{a.height}"
    out.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(Path(a.cache) / "sites.parquet")
    done = set()
    for f in out.glob("part_*.parquet"):
        done |= set(pd.read_parquet(f, columns=["als_site_id"])["als_site_id"])
    # cached = status "ok" in the cache manifest (stat-ing 86k files on the iRODS mount takes many minutes)
    cached = set(read_manifest(a.cache).query("status == 'ok'")["site_id"].astype(str))
    todo = [s for s in sites["als_site_id"] if s not in done and s in cached]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[forpt] {len(sites)} sites, {len(done)} embedded, {len(todo)} cached cookies to embed -> {out}", flush=True)
    if not todo:
        return

    import torch

    model = forpt.load_backbone(device="cuda")

    def prep(sid):
        try:
            return sid, forpt.preprocess(load_cookie(cookie_path(a.cache, sid))[0], a.radius_m, a.height, a.max_points)
        except Exception as e:  # unreadable cookie or empty crop: skipped, reported
            print(f"[forpt] {sid}: {type(e).__name__}: {e}", flush=True)
            return sid, None

    t0, n_done, shard_i = time.time(), 0, len(list(out.glob("part_*.parquet")))
    with ThreadPoolExecutor(a.io_workers) as ex:
        for s in range(0, len(todo), a.shard_size):
            ids, feats, sizes, batch, batch_ids = [], [], [], [], []

            def flush():
                if batch:
                    feats.append(forpt.embed_coords(model, batch, "cuda"))
                    ids.extend(batch_ids)
                    sizes.extend(len(c) for c in batch)
                    batch.clear()
                    batch_ids.clear()

            for sid, coord in ex.map(prep, todo[s : s + a.shard_size]):
                if coord is None or len(coord) == 0:
                    continue
                if batch and sum(map(len, batch)) + len(coord) > a.point_budget:
                    flush()
                batch.append(coord)
                batch_ids.append(sid)
            flush()
            torch.cuda.empty_cache()
            if not ids:
                continue
            X = np.concatenate(feats).astype(np.float32)
            df = pd.DataFrame(X, columns=[f"f{i:04d}" for i in range(X.shape[1])])
            df.insert(0, "n_input", sizes)
            df.insert(0, "als_site_id", ids)
            df.to_parquet(out / f"part_{shard_i:05d}.parquet", index=False)
            shard_i += 1
            n_done += len(ids)
            rate = n_done / (time.time() - t0)
            print(f"[forpt] {n_done}/{len(todo)} embedded, {rate:.2f} cookies/s, "
                  f"ETA {(len(todo) - n_done) / max(rate, 1e-9) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
