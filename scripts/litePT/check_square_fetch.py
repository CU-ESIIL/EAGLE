"""Check streamed squares against directly fetched cookies (eagle_als.squares vs eagle_als.fetch).

For N pre-training locations: fetch the 500 m square around the location (unthinned) and write it
as a shard, cut the 100 m cookie at its center from the shard, and fetch the same cookie directly
with fetch_cookie. Reports point counts, matched-point HAG differences (square-wide vs cookie-wide
ground model) and fetch times.

    python scripts/litePT/check_square_fetch.py --locations $EAGLE_SCRATCH/pretrain/locations_v1.parquet \
        --n 20 --workers 4 --out $EAGLE_SCRATCH/benchmarks/square_check
"""

import argparse
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd


def check_one(row, tmp):
    from scipy.spatial import cKDTree

    from eagle_als.fetch import fetch_cookie, fetch_square
    from eagle_als.squares import build_shard, cut_cookie, write_shard

    rec = dict(tile=row["product_name_AWS"], lat=row["lat"], lon=row["lon"])
    try:
        t0 = time.time()
        sq, meta = fetch_square(rec["tile"], rec["lat"], rec["lon"], half_size=250.0)
        t1 = time.time()
        ck, _ = fetch_cookie(rec["tile"], rec["lat"], rec["lon"], radius=100.0)
        t2 = time.time()
    except Exception as e:
        rec.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
        return rec
    name = f"{abs(hash((rec['lat'], rec['lon']))) % 10**9:09d}"
    t3 = time.time()
    pts, off = build_shard(sq, 250.0, 50.0)
    meta.update(block=50.0, block_offsets=off.tolist())
    write_shard(tmp, name, pts, meta)
    t4 = time.time()
    cut = cut_cookie(Path(tmp) / name, 0.0, 0.0, 100.0)
    t5 = time.time()
    # match points by coordinates (identical source points, float32 rounding only)
    d, j = cKDTree(ck["xyz"]).query(cut["xyz"], distance_upper_bound=0.01)
    m = np.isfinite(d)
    dh = np.abs(cut["hag"][m] - ck["hag"][j[m]])
    rec.update(
        status="ok", n_square=len(sq["xyz"]), n_cut=len(cut["xyz"]), n_cookie=len(ck["xyz"]),
        frac_matched=float(m.mean()) if len(m) else np.nan,
        hag_med_cm=100 * float(np.median(dh)) if len(dh) else np.nan,
        hag_p99_cm=100 * float(np.percentile(dh, 99)) if len(dh) else np.nan,
        hag_max_cm=100 * float(dh.max()) if len(dh) else np.nan,
        same_class=bool((cut["classification"][m] == ck["classification"][j[m]]).all()),
        has_ground=meta["has_ground"], square_s=t1 - t0, cookie_s=t2 - t1, write_s=t4 - t3, cut_s=t5 - t4,
        density=len(sq["xyz"]) / 500**2,
    )
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locations", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    locs = pd.read_parquet(a.locations).sample(a.n, random_state=a.seed)
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as pool:
        recs = list(pool.map(check_one, [r for _, r in locs.iterrows()], [tmp] * len(locs)))
    df = pd.DataFrame(recs)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["lat", "lon"]).round(3).to_string())
    ok = df[df.status == "ok"]
    summary = dict(
        n=len(df), n_ok=len(ok), n_cut_equals_cookie=int((ok.n_cut == ok.n_cookie).sum()),
        frac_matched_min=ok.frac_matched.min(), hag_med_cm_median=ok.hag_med_cm.median(),
        hag_p99_cm_median=ok.hag_p99_cm.median(), hag_p99_cm_max=ok.hag_p99_cm.max(),
        square_s_median=ok.square_s.median(), cookie_s_median=ok.cookie_s.median(),
        square_over_cookie_time=(ok.square_s / ok.cookie_s).median(), area_ratio=500**2 / (np.pi * 100**2),
        cut_s_median=ok.cut_s.median(), write_s_median=ok.write_s.median(),
        density_median=ok.density.median(),
    )
    print(json.dumps({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in summary.items()}, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(a.out) / "square_check.csv", index=False)


if __name__ == "__main__":
    main()
