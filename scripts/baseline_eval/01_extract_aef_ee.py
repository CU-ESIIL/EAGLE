"""Extract AlphaEarth centre-pixel embeddings at every benchmark location through Google Earth Engine.

Faster alternative to 01_extract_aef.py (which reads ~1 TB of COG blocks): Earth Engine returns only the sampled
pixels. Centre pixel only (columns c00..c63); disc columns are not produced, so probe with `--pooling center`.

    python scripts/baseline_eval/01_extract_aef_ee.py --project eagle-lidar             # all tasks
    python scripts/baseline_eval/01_extract_aef_ee.py --project eagle-lidar --tasks nlcd
    python scripts/baseline_eval/02_linear_probe.py --points datasets/raw/aef/points_ee --pooling center \
        --model alphaearth_center --n-jobs 4

Output: one shard per task and run in <out>/part_<task>_<n already done>.parquet (key, lat, lon, emb_year, aef_year, c00..c63).
Rerunning skips points already extracted.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_bench import aef, aef_ee  # noqa: E402
from eagle_bench.tasks import TASKS, load_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Earth Engine cloud project id")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--out", default=str(aef.CACHE_DIR / "points_ee"))
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8, help="concurrent Earth Engine requests")
    a = ap.parse_args()

    import ee

    ee.Initialize(project=a.project)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.tasks:
        pts = load_task(name).df[["lat", "lon", "emb_year"]].drop_duplicates().reset_index(drop=True)
        pts["key"] = aef.point_key(pts)
        done = aef.load_points(out)
        if len(done):
            pts = pts[~pts["key"].isin(set(done["key"]))].reset_index(drop=True)
        print(f"[extract_ee] {name}: {len(pts)} points to do", flush=True)
        if pts.empty:
            continue
        res = aef_ee.sample_points(pts, a.batch_size, a.workers)
        res = pts[["key", "lat", "lon", "emb_year"]].merge(res.drop(columns=["lat", "lon", "emb_year"]), on="key")
        n_nan = int(res["c00"].isna().sum())
        n_other = int(((res["aef_year"] != res["emb_year"]) & (res["aef_year"] > 0)).sum())
        res.to_parquet(out / f"part_{name}_{len(done)}.parquet")  # new shard per run: never overwrites earlier points
        print(f"[extract_ee] {name}: wrote {len(res)} points ({n_other} from another year, {n_nan} with no valid pixel)",
              flush=True)


if __name__ == "__main__":
    main()
