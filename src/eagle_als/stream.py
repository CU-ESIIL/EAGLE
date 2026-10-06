"""Stream 3DEP lidar into a rolling buffer of 500 m squares on node-local disk (single-node pre-training).

A producer process (this module's CLI) draws squares in a seeded order, reads each with one PDAL
query, thins it, computes HAG over the whole square and writes it as a shard. DataLoader workers
(eagle_als.data.StreamingChunkDataset) cut random 100 m cookies from whatever shards are present.
The producer is the only writer; it evicts the oldest shards beyond --max-shards.

    python -m eagle_als.stream --buffer $LOCAL/buffer --state-dir $RUN/producer --workers 48 --max-shards 2000

Shard layout (one directory per square, renamed into place when complete):
    <buffer>/<index:09d>/points.npy   structured array SHARD_DTYPE, sorted into `block` m blocks
    <buffer>/<index:09d>/meta.json    tile, lat, lon, epsg, half_size, block, block_offsets, timings
x, y are relative to the square center (UTM, m); z is elevation; blocks are numbered row-major
(by * n_blocks + bx) from the square's lower-left corner.

State in --state-dir: producer_state.json (next square index, so a resubmitted job continues the
sequence) and producer.jsonl (one record per square: status, timings, arrival and eviction times).
"""

import argparse
import json
import os
import shutil
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np

from .fetch import EmptyCookieError, fetch_square, tile_index, validate_cookie

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARD_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("hag", "<f4"), ("intensity", "<u2"),
    ("return_number", "u1"), ("number_of_returns", "u1"), ("classification", "u1"),
])
# evaluation sites kept out of pre-training: (table, lat column, lon column)
EVAL_SITES = [
    ("datasets/BBS/BBS_train_test_als_available.parquet", "Latitude", "Longitude"),
    ("datasets/butterflies/species_observations_als_available.parquet", "lat", "lon"),
    ("datasets/OFO_trees/plots_w_als_als_available.parquet", "plot_lat", "plot_lon"),
]


# ---------------- square sampling ----------------
class SquareSampler:
    """Deterministic sequence of square centers: square i depends only on (seed, i).

    Tiles are drawn with probability proportional to footprint area ** area_power; the center is
    uniform inside the tile footprint. Centers within `exclude_dist` m of an evaluation site are
    redrawn (default 500 m = half-diagonal of a 500 m square + a 100 m cookie, rounded up).
    """

    def __init__(self, seed=0, area_power=0.5, min_year=None, eval_sites=EVAL_SITES, exclude_dist=500.0):
        import shapely
        from pyproj import Transformer
        from scipy.spatial import cKDTree

        tiles = tile_index()
        tiles = tiles[tiles["url"].notna() & tiles.geometry.notna() & ~tiles.geometry.is_empty]
        if min_year is not None:
            tiles = tiles[tiles.collection_year >= min_year]
        w = (tiles.to_crs("EPSG:6933").area.to_numpy() / 1e6) ** area_power
        self.names = tiles.index.to_numpy()
        self.geoms = tiles.geometry.to_numpy()
        self.bounds = tiles.bounds.to_numpy()
        self.cum_w = np.cumsum(w / w.sum())
        self.seed = seed
        self.exclude_dist = exclude_dist
        self._to_albers = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        self._prepared = set()
        self._shapely = shapely
        xy = []
        for path, lat_col, lon_col in eval_sites or []:
            import pandas as pd

            df = pd.read_parquet(REPO_ROOT / path, columns=[lat_col, lon_col]).dropna().drop_duplicates()
            xy.append(np.stack(self._to_albers.transform(df[lon_col].to_numpy(), df[lat_col].to_numpy()), 1))
        self.n_eval_sites = int(sum(len(a) for a in xy))
        self.tree = cKDTree(np.concatenate(xy)) if xy else None

    def excluded(self, lon, lat):
        if self.tree is None:
            return False
        return self.tree.query(self._to_albers.transform(lon, lat), distance_upper_bound=self.exclude_dist)[0] < np.inf

    def __call__(self, i):
        """(tile name, lat, lon) of square i."""
        rng = np.random.default_rng([self.seed, i])
        for _ in range(1000):
            t = int(np.searchsorted(self.cum_w, rng.random()))
            g = self.geoms[t]
            if t not in self._prepared:
                self._shapely.prepare(g)
                self._prepared.add(t)
            x0, y0, x1, y1 = self.bounds[t]
            for _ in range(20):  # rejection sampling in the tile's bounding box
                lon, lat = rng.uniform(x0, x1, 64), rng.uniform(y0, y1, 64)
                inside = np.flatnonzero(self._shapely.contains_xy(g, lon, lat))
                if len(inside):
                    lon, lat = float(lon[inside[0]]), float(lat[inside[0]])
                    break
            else:
                continue
            if not self.excluded(lon, lat):
                return str(self.names[t]), lat, lon
        raise RuntimeError(f"could not draw square {i}")


# ---------------- shards ----------------
def build_shard(cookie, half_size, block=50.0):
    """Points of a square as a SHARD_DTYPE array sorted by block, plus block offsets [n_blocks**2 + 1]."""
    nb = int(round(2 * half_size / block))
    xyz = cookie["xyz"]
    bxy = np.clip(np.floor((xyz[:, :2] + half_size) / block).astype(np.int64), 0, nb - 1)
    key = bxy[:, 1] * nb + bxy[:, 0]
    order = np.argsort(key, kind="stable")
    pts = np.empty(len(xyz), SHARD_DTYPE)
    pts["x"], pts["y"], pts["z"] = xyz[order, 0], xyz[order, 1], xyz[order, 2]
    for k in ("hag", "intensity", "return_number", "number_of_returns", "classification"):
        pts[k] = cookie[k][order]
    offsets = np.concatenate([[0], np.cumsum(np.bincount(key, minlength=nb * nb))])
    return pts, offsets


def write_shard(buffer, name, pts, meta):
    """Write a shard directory atomically (readers never see partial shards)."""
    buffer = Path(buffer)
    tmp = buffer / f".tmp-{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "points.npy", pts)
    (tmp / "meta.json").write_text(json.dumps(meta))
    os.rename(tmp, buffer / name)


def list_shards(buffer):
    """Names of complete shards (sorted = arrival order of square indices)."""
    try:
        return sorted(e.name for e in os.scandir(buffer) if e.is_dir() and not e.name.startswith("."))
    except FileNotFoundError:
        return []


def read_shard_meta(shard_dir):
    return json.loads((Path(shard_dir) / "meta.json").read_text())


def cut_cookie(shard_dir, cx, cy, radius=100.0, meta=None):
    """Cookie of `radius` m around (cx, cy) (square coordinates, m) in the cached-cookie format of
    eagle_als.fetch: x, y relative to (cx, cy), z elevation. Reads only the blocks the cookie overlaps."""
    shard_dir = Path(shard_dir)
    meta = meta or read_shard_meta(shard_dir)
    half, block, off = meta["half_size"], meta["block"], meta["block_offsets"]
    nb = int(round(2 * half / block))
    pts_all = np.load(shard_dir / "points.npy", mmap_mode="r")
    bx0, bx1 = (np.clip(np.floor((np.array([cx - radius, cx + radius]) + half) / block), 0, nb - 1)).astype(int)
    by0, by1 = (np.clip(np.floor((np.array([cy - radius, cy + radius]) + half) / block), 0, nb - 1)).astype(int)
    # each block row is contiguous in the file: one read per row
    pts = np.concatenate([pts_all[off[by * nb + bx0]: off[by * nb + bx1 + 1]] for by in range(by0, by1 + 1)])
    dx, dy = pts["x"] - np.float32(cx), pts["y"] - np.float32(cy)
    keep = dx * dx + dy * dy <= radius * radius
    pts, dx, dy = pts[keep], dx[keep], dy[keep]
    cookie = dict(
        xyz=np.stack([dx, dy, pts["z"]], 1),
        hag=pts["hag"].copy(),
        intensity=pts["intensity"].copy(),
        return_number=pts["return_number"].copy(),
        number_of_returns=pts["number_of_returns"].copy(),
        classification=pts["classification"].copy(),
    )
    return cookie


# ---------------- producer ----------------
def produce_square(buffer, name, tile, lat, lon, half_size=250.0, block=50.0, max_density=12.0,
                   min_points=1000, timeout=None, seed=0):
    """Fetch, thin and write one square. Returns a status record (never raises)."""
    t0 = time.time()
    rec = dict(name=name, tile=tile, lat=lat, lon=lon, status="ok", n_points=0, n_raw=0, error="")
    try:
        cookie, meta = fetch_square(tile, lat, lon, half_size=half_size, timeout=timeout)
        t_fetch = time.time()
        n = len(cookie["xyz"])
        rec["n_raw"] = n
        if n < min_points:
            rec["status"] = "too_few_points"
        else:
            validate_cookie(cookie)
            n_max = int(max_density * (2 * half_size) ** 2) if max_density else n
            if n > n_max:  # uniform random thinning to the densest density the augmentations use
                keep = np.sort(np.random.default_rng([seed, int(name)]).choice(n, n_max, replace=False))
                cookie = {k: v[keep] for k, v in cookie.items()}
            pts, offsets = build_shard(cookie, half_size, block)
            meta.update(name=name, block=block, block_offsets=offsets.tolist(), n_points=len(pts), n_raw=n,
                        max_density=max_density, fetch_s=round(t_fetch - t0, 2))
            write_shard(buffer, name, pts, meta)
            rec.update(n_points=len(pts), has_ground=meta["has_ground"], fetch_s=meta["fetch_s"])
    except EmptyCookieError as e:
        rec["status"], rec["error"] = "empty", str(e)
    except Exception as e:  # network / PDAL / corrupted tile errors
        rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"[:500]
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def run_producer(buffer, state_dir, workers=48, max_shards=2000, seed=0, half_size=250.0, block=50.0,
                 max_density=12.0, max_squares=None, max_hours=None, timeout=None, sampler_kw=None):
    """Fill `buffer` with squares and keep it at <= max_shards, until stopped (SIGTERM / SIGINT,
    max_squares, max_hours, or a file named STOP in the buffer directory)."""
    from .cache import imap_isolated

    buffer, state_dir = Path(buffer), Path(state_dir)
    buffer.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    for p in buffer.glob(".*"):  # leftovers of a previous producer
        shutil.rmtree(p, ignore_errors=True)
    state_file = state_dir / "producer_state.json"
    start = json.loads(state_file.read_text())["next_index"] if state_file.exists() else 0
    sampler = SquareSampler(seed=seed, **(sampler_kw or {}))
    print(f"[producer] {len(sampler.names)} tiles, {sampler.n_eval_sites} evaluation sites excluded, "
          f"starting at square {start}, {workers} workers, buffer {buffer} (max {max_shards})", flush=True)

    stop = {"flag": False}
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(s, lambda *_: stop.update(flag=True))
    deadline = None if max_hours is None else time.time() + max_hours * 3600
    end = None if max_squares is None else start + max_squares
    next_index = {"i": start}

    def jobs():
        i = start
        while end is None or i < end:
            name = f"{i:09d}"
            tile, lat, lon = sampler(i)
            next_index["i"] = i + 1
            yield name, (buffer, name, tile, lat, lon, half_size, block, max_density, 1000, timeout, seed)
            i += 1

    present = deque(list_shards(buffer))  # shards surviving from earlier in this job
    log = open(state_dir / "producer.jsonl", "a")
    t0, n_done, n_ok, n_pts = time.time(), 0, 0, 0
    try:
        for name, rec, crashed in imap_isolated(produce_square, jobs(), workers):
            if crashed:
                rec = dict(name=name, status="error", error="worker process crashed (PDAL abort)")
            rec["arrival"] = time.time()
            n_done += 1
            if rec["status"] == "ok":
                n_ok += 1
                n_pts += rec["n_points"]
                present.append(name)
            evicted = []
            while len(present) > max_shards:
                old = present.popleft()
                trash = buffer / f".evict-{old}"
                try:
                    os.rename(buffer / old, trash)  # readers that already opened it keep their mmap
                    shutil.rmtree(trash, ignore_errors=True)
                except FileNotFoundError:
                    pass
                evicted.append(old)
            rec["evicted"] = evicted
            log.write(json.dumps(rec) + "\n")
            if n_done % 20 == 0:
                log.flush()
                state_file.write_text(json.dumps(dict(next_index=next_index["i"])))
                dt = time.time() - t0
                print(f"[producer] {n_done} squares ({n_ok} ok) in {dt:.0f}s: {n_ok / dt:.2f} ok squares/s, "
                      f"{n_pts / dt / 1e6:.2f} M pts/s, buffer {len(present)}", flush=True)
            if (stop["flag"] or (deadline and time.time() > deadline) or (buffer / "STOP").exists()):
                print("[producer] stop requested", flush=True)
                break
    finally:
        log.close()
        state_file.write_text(json.dumps(dict(next_index=next_index["i"])))
    dt = time.time() - t0
    print(f"[producer] finished: {n_done} squares, {n_ok} ok, {n_ok / max(dt, 1e-9):.2f} ok squares/s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", required=True, help="shard directory (node-local disk)")
    p.add_argument("--state-dir", required=True, help="producer state and log (persistent storage)")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--max-shards", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--half-size", type=float, default=250.0)
    p.add_argument("--block", type=float, default=50.0)
    p.add_argument("--max-density", type=float, default=12.0, help="thin squares to this many points/m^2")
    p.add_argument("--max-squares", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--timeout", type=float, default=None, help="PDAL readers.ept timeout (s)")
    p.add_argument("--no-exclude", action="store_true", help="do not exclude evaluation sites")
    a = p.parse_args()
    run_producer(a.buffer, a.state_dir, a.workers, a.max_shards, a.seed, a.half_size, a.block, a.max_density,
                 a.max_squares, a.max_hours, a.timeout, sampler_kw=dict(eval_sites=None) if a.no_exclude else None)


if __name__ == "__main__":
    main()
