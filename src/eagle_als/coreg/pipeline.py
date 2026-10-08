"""Batch pipeline pieces for the coregistration database: cell grid, planning, and per-unit processing.

Unit of work = (3DEP project, cell). A cell is a `cell_m` x `cell_m` square on the UTM (NAD83) km grid of the zone containing
its centre; it is fitted on the cell plus a `buffer_m` margin, so neighbouring fits overlap. The lidar of a unit is streamed
once and re-used for every NAIP survey (vintage) that covers the cell, because the lidar is the dominant I/O cost.

All outputs are plain JSON files (one per unit) so that a run is resumable, auditable and trivial to merge.
"""

import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from . import api, data as D

CODE_VERSION = None


def code_version():
    global CODE_VERSION
    if CODE_VERSION is None:
        try:
            CODE_VERSION = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
        except Exception:  # noqa: BLE001
            CODE_VERSION = "unknown"
    return CODE_VERSION


# ----------------------------------------------------------------------------- cell grid


def utm_epsg_nad83(lon):
    return 26900 + int((lon + 180) // 6) + 1


def cell_id(epsg, e0, n0, cell_m):
    return f"{epsg}_{int(cell_m // 1000)}km_{int(e0 // 1000)}_{int(n0 // 1000)}"


def cell_of_point(lon, lat, cell_m=5000):
    """(epsg, e0, n0, cell_id) of the cell whose core contains lon/lat."""
    from pyproj import Transformer

    epsg = utm_epsg_nad83(lon)
    e, n = Transformer.from_crs("EPSG:4269", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    e0, n0 = np.floor(e / cell_m) * cell_m, np.floor(n / cell_m) * cell_m
    return epsg, float(e0), float(n0), cell_id(epsg, e0, n0, cell_m)


def cell_grid(epsg, e0, n0, cell_m=5000, buffer_m=500, res=1.0):
    n = int(round((cell_m + 2 * buffer_m) / res))
    return D.Grid(f"EPSG:{epsg}", float(e0 - buffer_m), float(n0 + cell_m + buffer_m), res, n, n)


def cells_for_geometry(geom, cell_m=5000, min_cover=0.25):
    """Cells (epsg, e0, n0) of the UTM km grid whose area is >= `min_cover` covered by the lon/lat geometry."""
    import shapely
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    out = []
    geom = shapely.make_valid(geom)
    w, s, e, n = geom.bounds
    for z in range(int((w + 180) // 6) + 1, int((e + 180) // 6) + 2):
        g = shapely.make_valid(geom.intersection(box(-180 + 6 * (z - 1), -90, -180 + 6 * z, 90)))
        if g.is_empty:
            continue
        epsg = 26900 + z
        tr = Transformer.from_crs("EPSG:4269", f"EPSG:{epsg}", always_xy=True)
        gu = shapely.make_valid(transform(tr.transform, g))
        x0, y0, x1, y1 = gu.bounds
        xs, ys = np.meshgrid(np.arange(np.floor(x0 / cell_m) * cell_m, x1, cell_m), np.arange(np.floor(y0 / cell_m) * cell_m, y1, cell_m))
        xs, ys = xs.ravel(), ys.ravel()
        boxes = shapely.box(xs, ys, xs + cell_m, ys + cell_m)
        shapely.prepare(gu)
        hit = shapely.intersects(gu, boxes)
        area = np.zeros(len(boxes))
        area[hit] = shapely.area(shapely.intersection(gu, boxes[hit]))
        for x, y in zip(xs[area >= min_cover * cell_m**2], ys[area >= min_cover * cell_m**2]):
            out.append((epsg, float(x), float(y)))
    return out


# ----------------------------------------------------------------------------- planning


def plan_units(aoi=None, points=None, min_year=2010, cell_m=5000, stride=1, projects=None, newest_only=False, k_ring=0, min_project_year=2008):
    """Build the table of work units (pandas DataFrame, one row per (cell, 3DEP project)).

    Modes: `points` (iterable of lon, lat): only the cells containing them (+ `k_ring` rings of neighbours), i.e. a demand-driven
    plan for the plots that matter first; otherwise every cell covered by each 3DEP project (optionally clipped to the lon/lat
    polygon `aoi`).
    stride:      keep only cells with (i % stride == 0 and j % stride == 0) in cell units: a sparse first pass; `refine_plan`
                 adds intermediate cells where neighbouring fits disagree.
    newest_only: keep only the newest 3DEP project per cell.
    """
    import pandas as pd
    from pyproj import Transformer
    from shapely.geometry import Point

    from eagle_als.fetch import tile_index

    t = tile_index().reset_index()
    t = t[t.collection_year.notna() & (t.collection_year >= min_project_year)]
    if projects:
        t = t[t["name"].isin(projects)]
    rows = []

    def add(epsg, e0, n0, r):
        i, j = int(e0 // cell_m), int(n0 // cell_m)
        if stride > 1 and (i % stride or j % stride):
            return
        cid = cell_id(epsg, e0, n0, cell_m)
        rows.append(dict(unit_id=f"{cid}__{r['name']}", cell_id=cid, epsg=epsg, e0=float(e0), n0=float(n0), cell_m=cell_m, project=r["name"], url=r["url"],
                         project_year=int(r["collection_year"]), ql=r["ql"] if isinstance(r["ql"], str) else "", min_naip_year=min_year))

    if points is not None:
        want = set()
        for lon, lat in points:
            epsg, e0, n0, _ = cell_of_point(lon, lat, cell_m)
            for di in range(-k_ring, k_ring + 1):
                for dj in range(-k_ring, k_ring + 1):
                    want.add((epsg, e0 + di * cell_m, n0 + dj * cell_m))
        for epsg, e0, n0 in sorted(want):
            lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4269", always_xy=True).transform(e0 + cell_m / 2, n0 + cell_m / 2)
            for _, r in t[t.geometry.contains(Point(lon, lat))].iterrows():
                add(epsg, e0, n0, r)
    else:
        for _, r in t.iterrows():
            g = r.geometry if aoi is None else r.geometry.intersection(aoi)
            if g.is_empty:
                continue
            for epsg, e0, n0 in cells_for_geometry(g, cell_m):
                add(epsg, e0, n0, r)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["cell_id", "project_year"], ascending=[True, False])
        if newest_only:
            df = df.drop_duplicates("cell_id")
        df = df.reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- processing


def _atomic_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def process_unit(unit, out_dir, buffer_m=500, res=1.0, n_jobs=1, naip_years=None, min_lidar_cover=0.4, max_surveys=None):
    """Fit all NAIP surveys of one (cell, project) unit and write out_dir/done/<unit_id>.json. Idempotent."""
    out_dir = Path(out_dir)
    done = out_dir / "done" / f"{unit['unit_id']}.json"
    if done.exists():
        return "skipped"
    t0 = time.time()
    rec = dict(unit={k: (v.item() if hasattr(v, "item") else v) for k, v in unit.items()}, code_version=code_version(), res=res, buffer_m=buffer_m, results=[], naip_available=[])
    try:
        grid = cell_grid(unit["epsg"], unit["e0"], unit["n0"], unit["cell_m"], buffer_m, res)
        surveys = D.search_naip(grid, years=naip_years)
        surveys = {k: v for k, v in surveys.items() if int(k[:4]) >= int(unit.get("min_naip_year", 2010))}
        for k, items in surveys.items():
            rec["naip_available"].append(dict(survey=k, items=[D.naip_meta(i) for i in items]))
        if not surveys:
            rec["status"] = "no_naip"
        else:
            lr = D.lidar_rasters_tiled(unit["url"], grid, tile=1000, log=lambda m: None)
            cover = float(np.mean(np.isfinite(lr.chm)))
            rec["lidar_cover"] = cover
            rec["t_lidar"] = time.time() - t0
            if cover < min_lidar_cover:
                rec["status"] = "no_lidar"
            else:
                rec["status"] = "ok"
                centre_e = grid.x0 + grid.width * res / 2
                centre_n = grid.y0 - grid.height * res / 2
                for k, items in list(surveys.items())[:max_surveys]:
                    t1 = time.time()
                    try:
                        naip = D.read_naip(items, grid)
                        r = api.estimate_field(naip, lr, res=res, n_jobs=n_jobs)
                        rec["results"].append(dict(survey=k, ok=True, model=r.model, theta=r.theta, centre_e=centre_e, centre_n=centre_n, crs=grid.crs, res=res,
                                                   sun_az=r.sun_az, sun_el=r.sun_el, r2=r.r2, cv_r2=r.cv_r2, n_surfaces=r.n_surfaces, sd_xy=r.sd_xy,
                                                   sd_lean=r.sd_lean, tall_frac=r.tall_frac, valid_frac=r.valid_frac, quality=r.quality, seconds=time.time() - t1))
                    except Exception as e:  # noqa: BLE001  one bad survey must not lose the others
                        rec["results"].append(dict(survey=k, ok=False, error=repr(e)[:300], seconds=time.time() - t1))
        rec["seconds"] = time.time() - t0
        _atomic_write(done, rec)
        return rec["status"]
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()[-2000:]
        rec["seconds"] = time.time() - t0
        _atomic_write(out_dir / "failed" / f"{unit['unit_id']}.json", rec)
        return "failed"
