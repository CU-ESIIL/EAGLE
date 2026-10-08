"""Step 1: write the table of work units (3DEP project x cell) for the coregistration database.

Examples
  # demand-driven: only the cells containing our plots (+1 ring), newest lidar only
  python plan_units.py units.parquet --points-file ../../../datasets/OFO_trees/plots_w_als.gpkg --k-ring 1
  # sparse wall-to-wall first pass (every 3rd cell in each direction), then refine_plan.py fills in
  python plan_units.py units.parquet --stride 3
  # one state / box
  python plan_units.py units.parquet --bbox -105.5 39.9 -105.0 40.2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from eagle_als.coreg import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--points-file", help="vector file/parquet with point (or any) geometries; cell centroids of these are planned")
    ap.add_argument("--k-ring", type=int, default=0)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    ap.add_argument("--cell-m", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-naip-year", type=int, default=2010)
    ap.add_argument("--newest-only", action="store_true", help="only the newest 3DEP project per cell")
    ap.add_argument("--projects", nargs="*")
    a = ap.parse_args()
    aoi = points = None
    if a.bbox:
        from shapely.geometry import box

        aoi = box(*a.bbox)
    if a.points_file:
        import geopandas as gpd

        g = gpd.read_file(a.points_file).to_crs(4269)
        points = [(p.x, p.y) for p in g.geometry.representative_point()]
    df = P.plan_units(aoi=aoi, points=points, min_year=a.min_naip_year, cell_m=a.cell_m, stride=a.stride, projects=a.projects, newest_only=a.newest_only, k_ring=a.k_ring)
    df.to_parquet(a.out)
    print(f"{len(df)} units, {df.cell_id.nunique()} cells, {df.project.nunique()} projects -> {a.out}")


if __name__ == "__main__":
    main()
