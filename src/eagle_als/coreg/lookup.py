"""Lookup of precomputed NAIP <-> 3DEP coregistration calibrations.

    db = CalibrationDB("coreg_db")            # directory written by scripts/naip/cluster/merge_db.py
    db.lookup(lat=40.0, lon=-105.28)          # dict: 3DEP projects, NAIP surveys, calibration per pairing

Calibration = smooth displacement field of the NAIP relative to the lidar (see api.py):
    d(E, N, h) = a + A p + h (t0 + G p),   p = ((E, N) - cell centre) / 1 km,  h = height above ground of the feature (m)
    theta = [ax, ay, tx0, ty0, Axx, Axy, Ayx, Ayy, Gxx, Gxy, Gyx, Gyy]  (last 8 only if model == 'affine').
To correct a NAIP pixel (or a tree top at height h), move it by -d. `eval_displacement` evaluates d.
"""

import json
from pathlib import Path

import numpy as np

from . import pipeline as P


def eval_displacement(res, e, n, h=0.0):
    """(dx, dy) in metres (east, north) of NAIP relative to lidar at UTM (e, n) and feature height h, from one calibration record."""
    th = res["theta"]
    px, py = (e - res["centre_e"]) / 1000.0, (n - res["centre_n"]) / 1000.0
    dx, dy = th[0] + h * th[2], th[1] + h * th[3]
    if res.get("model") == "affine" and len(th) >= 12:
        dx += th[4] * px + th[5] * py + h * (th[8] * px + th[9] * py)
        dy += th[6] * px + th[7] * py + h * (th[10] * px + th[11] * py)
    return float(dx), float(dy)


class CalibrationDB:
    def __init__(self, db_dir):
        import pandas as pd

        d = Path(db_dir)
        self.cal = pd.read_parquet(d / "calibrations.parquet")
        self.cov = pd.read_parquet(d / "coverage.parquet")
        self.units = pd.read_parquet(d / "units.parquet")
        self.meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        self.cell_m = int(self.meta.get("cell_m", 5000))

    def lookup(self, lat, lon, heights=(0.0, 10.0, 25.0)):
        from pyproj import Transformer
        from shapely.geometry import Point

        from eagle_als.fetch import tile_index

        epsg, e0, n0, cid = P.cell_of_point(lon, lat, self.cell_m)
        e, n = Transformer.from_crs("EPSG:4269", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
        t = tile_index()
        proj = t[t.geometry.contains(Point(lon, lat)) & t.collection_year.notna()]
        out = dict(lat=lat, lon=lon, cell_id=cid, utm_epsg=epsg, easting=e, northing=n,
                   dist_to_cell_edge_m=float(min(e - e0, e0 + self.cell_m - e, n - n0, n0 + self.cell_m - n)),
                   lidar=[dict(project=str(i), year=int(r.collection_year), ql=r.ql if isinstance(r.ql, str) else None, ept=r.url) for i, r in proj.iterrows()],
                   naip=[], note="displacement d = NAIP relative to lidar (m, east/north); correct by moving NAIP by -d")
        cov = self.cov[self.cov.cell_id == cid]
        for _, c in cov.drop_duplicates("survey").iterrows():
            item = dict(survey=c.survey, items=json.loads(c["items"]) if isinstance(c["items"], str) else c["items"], calibrations=[])
            for _, r in self.cal[(self.cal.cell_id == cid) & (self.cal.survey == c.survey)].iterrows():
                rec = r.to_dict()
                rec["theta"] = list(rec["theta"])
                cal = dict(project=rec["project"], status="ok" if rec["ok"] else "failed", quality=rec.get("quality"), model=rec.get("model"),
                           r2=rec.get("r2"), sd_xy_m=rec.get("sd_xy"), sun=(rec.get("sun_az"), rec.get("sun_el")))
                if rec["ok"]:
                    cal["ground_shift_m"] = dict(zip(("dx", "dy"), eval_displacement(rec, e, n, 0.0)))
                    cal["by_height_m"] = {str(h): dict(zip(("dx", "dy"), eval_displacement(rec, e, n, h))) for h in heights}
                    cal["theta"] = rec["theta"]
                    cal["centre"] = (rec["centre_e"], rec["centre_n"], rec["crs"])
                item["calibrations"].append(cal)
            if not item["calibrations"]:
                item["status"] = "not_computed"
            out["naip"].append(item)
        if not len(cov):
            out["status"] = "cell_not_in_database"
        return out


if __name__ == "__main__":
    import sys

    print(json.dumps(CalibrationDB(sys.argv[1]).lookup(float(sys.argv[2]), float(sys.argv[3])), indent=1, default=str))
