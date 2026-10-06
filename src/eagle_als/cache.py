"""Build a local cache of ALS cookies for a table of sites (parallel + resumable).

Usage (see scripts/litePT/README.md):

    python -m eagle_als.cache --sites sites.parquet --out $EAGLE_SCRATCH/cache/<name> \
        --id-col site_id --lat-col lat --lon-col lon --tile-col product_name_AWS \
        --workers 32 [--task-index $SLURM_ARRAY_TASK_ID --num-tasks N]

Outputs, under --out:
    cookies/<shard>/<site_id>.npz     one file per successfully fetched site
    manifest/part-XXXX.parquet        one status record per attempted site:
                                      status in {ok, empty, too_few_points, error, unknown_tile}

Re-running skips sites that already have a cookie or a non-error manifest record, so a job that
hits its time limit can simply be resubmitted. Use --retry-errors to re-attempt failed fetches.
"""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import fetch_and_save


def cookie_path(out, site_id):
    site_id = str(site_id)
    shard = f"{abs(hash_str(site_id)) % 256:02x}"
    return Path(out) / "cookies" / shard / f"{site_id}.npz"


def hash_str(s):
    # stable across processes (python's hash() is salted)
    import zlib

    return zlib.crc32(s.encode())


def _isolated(ctx, fn, args):
    """Run one call in its own process; returns (result, crashed)."""
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            return pool.submit(fn, *args).result(), False
    except BrokenProcessPool:
        return None, True


def imap_isolated(fn, jobs, workers, max_tasks_per_child=200):
    """Run fn(*args) for (key, args) in `jobs` (any iterable, may be endless) in a spawn process pool.

    Yields (key, result, crashed) in completion order. A sliding window of 2 * workers in-flight
    jobs keeps memory bounded; when a worker crashes (e.g. an uncaught PDAL C++ exception aborts the
    process) the in-flight jobs are re-run one per process, so only the culprit yields crashed=True
    (result None). Breaking out of the loop cancels queued jobs and waits for running ones.
    """
    # 'spawn' avoids PDAL/GDAL state and temp-dir conflicts seen with fork
    ctx = mp.get_context("spawn")
    jobs = iter(jobs)
    exhausted = False
    while not exhausted:
        window = {}
        broken = False
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=max_tasks_per_child)
        try:
            while not broken:
                while not exhausted and len(window) < 2 * workers:
                    try:
                        key, args = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    window[pool.submit(fn, *args)] = (key, args)
                if not window:
                    break
                done, _ = wait(window, return_when=FIRST_COMPLETED)
                for fut in done:
                    key, args = window.pop(fut)
                    try:
                        result = fut.result()
                    except BrokenProcessPool:
                        window[fut] = (key, args)
                        broken = True
                        continue
                    yield key, result, False
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if broken:
            suspects = list(window.values())
            print(f"[pool] worker crashed; re-running {len(suspects)} in-flight jobs in isolation", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as tp:
                for (key, _), (result, crashed) in zip(suspects, tp.map(lambda j: _isolated(ctx, fn, j[1]), suspects)):
                    yield key, result, crashed


def read_manifest(out):
    parts = sorted((Path(out) / "manifest").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["site_id", "status"])
    m = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # keep the most recent attempt per site
    return m.sort_values("timestamp").drop_duplicates("site_id", keep="last").reset_index(drop=True)


def build_cache(
    sites,
    out,
    id_col="site_id",
    lat_col="lat",
    lon_col="lon",
    tile_col="product_name_AWS",
    radius=100.0,
    min_points=100,
    max_points=None,
    workers=16,
    task_index=0,
    num_tasks=1,
    retry_errors=False,
    limit=None,
    flush_every=100,
    max_hours=None,
):
    out = Path(out)
    (out / "manifest").mkdir(parents=True, exist_ok=True)
    sites = sites.drop_duplicates(id_col)
    sites = sites[sites[tile_col].notna()]
    sites = sites.iloc[task_index::num_tasks]

    done = read_manifest(out)
    skip_status = {"ok", "empty", "too_few_points", "unknown_tile"}
    if not retry_errors:
        skip_status.add("error")
    done_ids = set(done.loc[done.status.isin(skip_status), "site_id"].astype(str))
    todo = [r for r in sites.itertuples(index=False) if str(getattr(r, id_col)) not in done_ids]
    # cookies saved by a run that died before flushing its manifest: record them instead of refetching
    orphans = [r for r in todo if cookie_path(out, getattr(r, id_col)).exists()]
    if orphans:
        recs = []
        for r in orphans:
            path = cookie_path(out, getattr(r, id_col))
            try:
                with np.load(path) as f:
                    meta = json.loads(str(f["meta"]))
                recs.append(dict(path=str(path), tile=meta["tile"], lat=meta["lat"], lon=meta["lon"], status="ok",
                                 n_points=meta["n_points"], error="", has_ground=meta["has_ground"], seconds=0.0,
                                 site_id=str(getattr(r, id_col)), timestamp=time.time()))
            except Exception as e:  # unreadable partial file: refetch
                print(f"[cache] removing unreadable cookie {path}: {e}", flush=True)
                path.unlink(missing_ok=True)
        if recs:
            pd.DataFrame(recs).to_parquet(out / "manifest" / f"part-orphans-{int(time.time())}.parquet", index=False)
            print(f"[cache] recorded {len(recs)} cookies that were missing from the manifest", flush=True)
    todo = [r for r in todo if not cookie_path(out, getattr(r, id_col)).exists()]
    if limit is not None:
        todo = todo[:limit]
    print(f"[cache] task {task_index}/{num_tasks}: {len(sites)} sites, {len(todo)} to fetch", flush=True)
    if not todo:
        return

    part = out / "manifest" / f"part-{task_index:04d}-{int(time.time())}.parquet"
    records, t_start = [], time.time()
    deadline = None if max_hours is None else t_start + max_hours * 3600

    def flush():
        if records:
            pd.DataFrame(records).to_parquet(part, index=False)

    jobs = [
        (str(getattr(r, id_col)), (cookie_path(out, getattr(r, id_col)), getattr(r, tile_col),
                                   float(getattr(r, lat_col)), float(getattr(r, lon_col)), radius, min_points, max_points))
        for r in todo
    ]
    args_by_id = dict(jobs)
    n_done = 0
    for sid, rec, crashed in imap_isolated(fetch_and_save, jobs, workers):
        if crashed:
            path, tile, lat, lon = args_by_id[sid][:4]
            rec = dict(path=str(path), tile=tile, lat=lat, lon=lon, status="error",
                       n_points=0, error="worker process crashed (PDAL abort)", seconds=0.0)
        rec["site_id"] = sid
        rec["timestamp"] = time.time()
        records.append(rec)
        n_done += 1
        if n_done % flush_every == 0:
            flush()
            n_ok = sum(r["status"] == "ok" for r in records)
            rate = n_done / (time.time() - t_start)
            print(f"[cache] {n_done}/{len(todo)} done, {n_ok} ok, {rate:.2f} sites/s", flush=True)
        if deadline is not None and time.time() > deadline:
            print("[cache] reached --max-hours, stopping early (resubmit to continue)", flush=True)
            break
    flush()
    summary = pd.DataFrame(records).status.value_counts().to_dict()
    print(f"[cache] finished: {summary}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", required=True, help="parquet/csv table of sites")
    p.add_argument("--out", required=True)
    p.add_argument("--id-col", default="site_id")
    p.add_argument("--lat-col", default="lat")
    p.add_argument("--lon-col", default="lon")
    p.add_argument("--tile-col", default="product_name_AWS")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--min-points", type=int, default=100)
    p.add_argument("--max-points", type=int, default=None, help="randomly thin denser cookies to this many points")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    a = p.parse_args()
    sites = pd.read_parquet(a.sites) if a.sites.endswith(".parquet") else pd.read_csv(a.sites)
    build_cache(
        sites, a.out, a.id_col, a.lat_col, a.lon_col, a.tile_col, a.radius, a.min_points,
        a.max_points, a.workers, a.task_index, a.num_tasks, a.retry_errors, a.limit, max_hours=a.max_hours,
    )


if __name__ == "__main__":
    main()
