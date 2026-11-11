"""Write the site table for the shared lidar cookie cache of the benchmark (one row per unique als_site_id).

Every registry table with a 3DEP product contributes its sites (BLM AIM's ids are built from tile + lat/lon by
eagle_bench.tasks). Rows are ordered by task priority (small, high-value tasks first; SatBird last), so a cache
build that is cut off still covers whole tasks. Then fetch with eagle_als.cache (needs PDAL):

    python scripts/baseline_eval/03_lidar_sites.py                       # -> $BENCH/als_cache/sites.parquet
    python -m eagle_als.cache --sites $BENCH/als_cache/sites.parquet --out $BENCH/als_cache \
        --id-col als_site_id --workers 8                                # resumable; rerun after interruption

$BENCH = /home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking (shared, persistent).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als import registry  # noqa: E402
from eagle_bench.tasks import _table  # noqa: E402

BENCH = Path("/home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking")
PRIORITY = ["nlcd", "ofo_trees", "butterflies", "vegbench", "nps_vmi", "lfrdb", "snapshot_usa", "blm_aim", "bbs",
            "satbird_winter"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BENCH / "als_cache/sites.parquet"))
    a = ap.parse_args()
    names = [n for n in PRIORITY if n in registry.load_registry()]
    names += [n for n in registry.load_registry() if n not in names]
    parts = []
    for n in names:
        df, spec, _ = _table(n)
        if "product_name_AWS" not in df or spec["als_id"] not in df:
            print(f"[sites] {n}: no 3DEP match columns, skipped")
            continue
        s = df[[spec["als_id"], spec["lat"], spec["lon"], "product_name_AWS"]].dropna()
        s.columns = ["als_site_id", "lat", "lon", "product_name_AWS"]
        s = s.drop_duplicates("als_site_id").assign(dataset=n)
        print(f"[sites] {n}: {len(s)} sites")
        parts.append(s)
    sites = pd.concat(parts, ignore_index=True)
    first = sites.drop_duplicates("als_site_id")  # a site shared by datasets keeps its highest-priority dataset
    print(f"[sites] {len(first)} unique sites ({len(sites) - len(first)} shared between datasets)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    first.to_parquet(a.out, index=False)
    print(f"[sites] wrote {a.out}")


if __name__ == "__main__":
    main()
