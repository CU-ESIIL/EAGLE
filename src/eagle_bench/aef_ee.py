"""AlphaEarth annual embeddings at points, sampled through Google Earth Engine (centre pixel only).

Same embeddings as `aef.py` (Source Cooperative COGs) but returns only the sampled pixels, so no blocks are
downloaded: ~96k points take minutes instead of ~1 TB of reads. Needs `earthengine-api`, a registered Earth
Engine project and credentials (`earthengine authenticate`).

    import ee; ee.Initialize(project="eagle-lidar")
    df = sample_points(points)       # points: lat, lon, emb_year, key -> adds aef_year, c00..c63

Each point uses the image of its `emb_year`; points with no valid pixel that year (no tile, or masked) are
retried with the nearest other year (own year first, then |dy| = 1, 2, ... ; ties go to the earlier year),
matching `aef.locate`. Validated against the COG extraction: cosine 1.000 on 22/30 NLCD points, >= 0.975 on all.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from eagle_bench.aef import N_DIM, embed_columns

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
YEARS = list(range(2017, 2026))
EE_BANDS = [f"A{i:02d}" for i in range(N_DIM)]


def _sample_batch(batch, year, retries=5):
    """Centre-pixel embeddings for one batch of points from the `year` mosaic: DataFrame key + EE bands."""
    import ee

    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"key": r.key})
                               for r in batch.itertuples()])
    img = ee.ImageCollection(COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
    for attempt in range(retries):
        try:
            res = img.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()
            rows = [f["properties"] for f in res["features"]]
            return pd.DataFrame(rows, columns=["key"] + EE_BANDS)
        except Exception as e:  # quota / transient errors: back off and retry
            print(f"[aef_ee] year {year} batch of {len(batch)} attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Earth Engine sampling failed for a batch of {len(batch)} points, year {year}")


def sample_points(points, batch_size=1000, workers=8):
    """Return `points` (needs key, lat, lon, emb_year) with aef_year and unit-length c00..c63 (NaN = never valid)."""
    todo = points[["key", "lat", "lon", "emb_year"]].drop_duplicates("key").reset_index(drop=True)
    found = []
    pending = todo.copy()
    for step in range(len(YEARS)):  # step 0: own year; then nearest other years
        if pending.empty:
            break
        jobs = []
        for yr, g in pending.groupby("emb_year"):
            cands = sorted(YEARS, key=lambda c: (abs(c - yr), c))
            if step >= len(cands):
                continue
            year = cands[step]
            jobs += [(g.iloc[s : s + batch_size], year) for s in range(0, len(g), batch_size)]
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda j: _sample_batch(*j).assign(aef_year=j[1]), jobs))
        got = pd.concat(res, ignore_index=True).drop_duplicates("key") if res else pd.DataFrame()
        print(f"[aef_ee] pass {step}: {len(got)}/{len(pending)} points sampled from {len(jobs)} requests, "
              f"{time.time() - t0:.0f} s", flush=True)
        if len(got):
            found.append(got)
            pending = pending[~pending["key"].isin(set(got["key"]))]
    emb = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["key", "aef_year"] + EE_BANDS)
    out = todo.merge(emb, on="key", how="left")
    X = out[EE_BANDS].to_numpy(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(n > 0, n, np.nan)
    out = out.drop(columns=EE_BANDS)
    out["aef_year"] = out["aef_year"].fillna(0).astype(int)
    return pd.concat([out, pd.DataFrame(X, columns=embed_columns("c"))], axis=1)
