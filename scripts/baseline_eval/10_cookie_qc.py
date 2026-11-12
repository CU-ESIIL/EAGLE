"""Coverage QC of every cached cookie (eagle_als.cookie_qc): points in the analysis disc vs the 3DEP minimum.

    pixi run --manifest-path envs/eagle-bench/pixi.toml env PYTHONPATH=src \
        python scripts/baseline_eval/10_cookie_qc.py --cache $WORK/benchmarking/als_cache --workers 28

Writes <cache>/qc_r<radius>.parquet: als_site_id, tile, n_disc, ql, expected_min, density_ratio, cell_coverage,
low_points (density_ratio < 0.5), plus the dataset of the site; and prints flagged counts per dataset.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als.cache import cookie_path, read_manifest  # noqa: E402
from eagle_als.cookie_qc import disc_qc  # noqa: E402
from eagle_als.fetch import load_cookie, tile_index  # noqa: E402


def _one(args):
    cache, sid, tile, ql, radius = args
    try:
        c = load_cookie(cookie_path(cache, sid))
        c = c[0] if isinstance(c, tuple) else c
        return dict(als_site_id=sid, tile=tile, **disc_qc(c, ql, radius))
    except Exception as e:
        return dict(als_site_id=sid, tile=tile, error=f"{type(e).__name__}: {e}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--radius-m", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    sites = pd.read_parquet(Path(a.cache) / "sites.parquet").set_index("als_site_id")
    ok = read_manifest(a.cache).query("status == 'ok'")["site_id"].astype(str).drop_duplicates().tolist()
    ok = ok[: a.limit] if a.limit else ok
    ql = tile_index()["ql"]
    jobs = [(a.cache, s, sites.loc[s, "product_name_AWS"], ql.get(sites.loc[s, "product_name_AWS"]), a.radius_m)
            for s in ok]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=64))
    qc = pd.DataFrame(rows).join(sites[["dataset"]], on="als_site_id")
    out = Path(a.cache) / f"qc_r{a.radius_m:g}.parquet"
    qc.to_parquet(out, index=False)
    print(f"[qc] {len(qc)} cookies -> {out}")
    print(qc.groupby("dataset").agg(n=("als_site_id", "size"), low_points=("low_points", "sum"),
                                    empty_disc=("n_disc", lambda s: int((s == 0).sum())),
                                    coverage_lt_50pct=("cell_coverage", lambda s: int((s < 0.5).sum())),
                                    median_ratio=("density_ratio", "median")).to_string())


if __name__ == "__main__":
    main()
