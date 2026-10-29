"""Extract AlphaEarth annual embeddings at every benchmark location (resumable; network-bound).

Reads the public AEF COGs on Source Cooperative (no account) for the unique (lat, lon, emb_year) points of
all tasks in `eagle_bench.tasks`. A point uses the AEF file of its survey year, clipped to 2017-2025, and the
nearest year with a tile covering it if its own year has none, so no row is dropped for the year.
Per point it stores the centre pixel (`c00..c63`) and the mean of the valid pixels in a 50 m disc (`d00..d63`).

Run from the repo root (about 1 TB is read, 33,386 blocks of 1024 x 1024 px; keep only ~100 MB):

    python scripts/baseline_eval/01_extract_aef.py                       # all tasks
    python scripts/baseline_eval/01_extract_aef.py --tasks nlcd ofo_trees --workers 16
    python scripts/baseline_eval/01_extract_aef.py --count-only          # how many files/blocks, no download

Output: datasets/raw/aef/points/part_*.parquet (gitignored), one shard per batch of COG files. Rerunning
skips points already in a shard, so an interrupted job just continues.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_bench import aef  # noqa: E402
from eagle_bench.tasks import TASKS, load_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--out", default=str(aef.CACHE_DIR / "points"))
    ap.add_argument("--workers", type=int, default=16, help="COG files read in parallel")
    ap.add_argument("--radius-m", type=float, default=50.0)
    ap.add_argument("--files-per-shard", type=int, default=100)
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pts = pd.concat([load_task(n).df[["lat", "lon", "emb_year"]] for n in a.tasks]).drop_duplicates()
    pts = pts.reset_index(drop=True)
    pts["key"] = aef.point_key(pts)
    done = aef.load_points(out)
    if len(done):
        pts = pts[~pts["key"].isin(set(done["key"]))].reset_index(drop=True)
    print(f"[extract] {len(pts)} points to do ({len(done)} already extracted)", flush=True)
    if pts.empty:
        return
    located = aef.locate(pts, aef.load_index())
    print(f"[extract] no tile in any year: {int(located['url'].isna().sum())}; other year used: "
          f"{int((located['url'].notna() & (located['aef_year'] != located['emb_year'])).sum())}; "
          f"{located['url'].nunique()} files, {aef.n_blocks(located)} blocks", flush=True)
    if a.count_only:
        return

    urls = sorted(located["url"].dropna().unique())
    for s in range(0, len(urls), a.files_per_shard):
        batch = located[located["url"].isin(urls[s : s + a.files_per_shard])]
        centre, disc, n_valid = aef.extract(batch, a.radius_m, a.workers)
        shard = batch[["key", "lat", "lon", "emb_year", "aef_year"]].copy()
        shard["n_valid_disc"] = n_valid
        shard = pd.concat([shard.reset_index(drop=True),
                           pd.DataFrame(centre, columns=aef.embed_columns("c")),
                           pd.DataFrame(disc, columns=aef.embed_columns("d"))], axis=1)
        shard.to_parquet(out / f"part_{s // a.files_per_shard:05d}_{len(done)}.parquet")
        print(f"[extract] shard {s // a.files_per_shard + 1}/{-(-len(urls) // a.files_per_shard)} written", flush=True)
    miss = located[located["url"].isna()]
    if len(miss):  # points no tile covers: stored as NaN rows so they are not retried
        shard = miss[["key", "lat", "lon", "emb_year", "aef_year"]].reset_index(drop=True)
        shard["n_valid_disc"] = 0
        nan = pd.DataFrame(np.nan, index=shard.index, columns=aef.embed_columns("c") + aef.embed_columns("d"))
        pd.concat([shard, nan], axis=1).to_parquet(out / f"part_nocover_{len(done)}.parquet")


if __name__ == "__main__":
    main()
