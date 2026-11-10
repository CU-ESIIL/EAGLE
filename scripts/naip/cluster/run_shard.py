"""Step 2: process one shard of the unit table on one node (many worker processes). Resumable and idempotent.

  python run_shard.py units.parquet OUT_DIR --shard 3 --n-shards 40 --workers 24 --jobs 5

Units are dealt round-robin (unit i belongs to shard i % n_shards) after sorting by project so that each project's lidar tiles
are read by few workers. Each worker runs one unit at a time (`--jobs` processes inside a unit for the surface fits), so
CPU use is workers x jobs (aim for the node's core count). Memory per worker at 1 m, 6 km window: roughly 8 GB.
Finished units are skipped (out/done/*.json); failures are written to out/failed/*.json and retried with --retry-failed.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "1")  # 100+ processes: no hidden BLAS threads
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def work(args):
    unit, out, kw = args
    from eagle_als.coreg import pipeline as P

    t = time.time()
    status = P.process_unit(unit, out, **kw)
    return unit["unit_id"], status, time.time() - t


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=5, help="processes per unit for the block fits")
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--buffer-m", type=float, default=500)
    ap.add_argument("--naip-years", type=int, nargs="*")
    ap.add_argument("--max-surveys", type=int)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, help="process at most N units of the shard (testing)")
    a = ap.parse_args()
    out = Path(a.out)
    df = pd.read_parquet(a.units).sort_values(["project", "cell_id"]).reset_index(drop=True)
    mine = df.iloc[a.shard :: a.n_shards]
    if a.retry_failed:
        for f in (out / "failed").glob("*.json"):
            f.unlink() if f.stem in set(mine.unit_id) else None
    todo = [r for _, r in mine.iterrows() if not (out / "done" / f"{r.unit_id}.json").exists() and not (out / "failed" / f"{r.unit_id}.json").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"shard {a.shard}/{a.n_shards}: {len(mine)} units, {len(todo)} to do, workers={a.workers} jobs={a.jobs}", flush=True)
    kw = dict(buffer_m=a.buffer_m, res=a.res, n_jobs=a.jobs, naip_years=a.naip_years, max_surveys=a.max_surveys)
    t0, n = time.time(), 0
    with cf.ProcessPoolExecutor(a.workers, mp_context=__import__("multiprocessing").get_context("spawn")) as ex:
        futs = [ex.submit(work, (r.to_dict(), str(out), kw)) for r in todo]
        for f in cf.as_completed(futs):
            try:
                uid, status, dt = f.result()
            except Exception as e:  # noqa: BLE001  (worker crash, e.g. OOM kill)
                print("worker error", repr(e)[:200], flush=True)
                continue
            n += 1
            print(f"[{n}/{len(todo)}] {uid} {status} {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
