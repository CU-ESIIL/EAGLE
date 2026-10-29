"""AlphaEarth Foundations (AEF) annual embeddings at points, read from the public Source Cooperative COGs.

Source: https://source.coop/tge-labs/aef (CC-BY 4.0: "The AlphaEarth Foundations Satellite Embedding dataset is
produced by Google and Google DeepMind."). One 64-d unit-length embedding per 10 m pixel and year (2017-2025),
stored as int8 in 8192 x 8192 px COGs per UTM zone (1024 x 1024 px blocks, bottom-up rows). No account needed.

    points = locate(points_df, load_index())            # adds url, row, col for each (lat, lon, emb_year)
    center, disc, n_valid = extract(points)             # centre pixel, mean over a disc, valid pixels in the disc

A point's embedding is read from the file of the year `emb_year` (or, where that year has no tile
covering it, the nearest year that does; see `locate`). Two summaries are returned: the centre
pixel and the mean of the valid pixels in a disc (radius `radius_m`, default 50 m = the central disc the lidar
probe uses), renormalised to unit length. Points on masked pixels (-128: no imagery) come back as NaN.

Reads are grouped by file and block so that each 1024 x 1024 block (tens of MB) is fetched once.
"""

import math
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "datasets/raw/aef"  # gitignored
INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
HTTP_ROOT = "https://data.source.coop/"
S3_ROOT = "s3://us-west-2.opendata.source.coop/"
PIXEL_M = 10.0
BLOCK = 1024
N_DIM = 64
NODATA = -128


def load_index(cache_dir=CACHE_DIR):
    """File index (bounds per COG), downloaded once to `cache_dir` (about 80 MB)."""
    path = Path(cache_dir) / "aef_index.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[aef] downloading {INDEX_URL} -> {path}", flush=True)
        req = urllib.request.Request(INDEX_URL, headers={"User-Agent": "curl/8"})  # default urllib agent gets a 403
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    cols = ["crs", "path", "year", "utm_zone", "utm_west", "utm_south", "utm_east", "utm_north"]
    return pd.read_parquet(path, columns=cols)


def utm_epsg(lat, lon):
    zone = int(math.floor((lon + 180) / 6)) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _locate_year(pts, index, year, mask):
    """Fill url/row/col for the rows `mask` of `pts` from the files of one year; returns rows now located."""
    import pyproj

    sub = pts[mask]
    zone = np.floor((sub["lon"].to_numpy() + 180) / 6).astype(int) % 60 + 1
    epsg = np.where(sub["lat"].to_numpy() >= 0, 32600, 32700) + zone
    for code in np.unique(epsg):
        files = index[(index["year"] == year) & (index["crs"] == f"EPSG:{code}")]
        grp = sub[epsg == code]
        x, y = pyproj.Transformer.from_crs(4326, f"EPSG:{code}", always_xy=True).transform(
            grp["lon"].to_numpy(), grp["lat"].to_numpy())
        for f in files.itertuples():
            inside = (x >= f.utm_west) & (x < f.utm_east) & (y >= f.utm_south) & (y < f.utm_north)
            if inside.any():
                ids = grp.index[inside]
                pts.loc[ids, "url"] = f.path.replace(S3_ROOT, HTTP_ROOT)
                pts.loc[ids, "col"] = np.floor((x[inside] - f.utm_west) / PIXEL_M).astype(int)
                pts.loc[ids, "row"] = np.floor((y[inside] - f.utm_south) / PIXEL_M).astype(int)
                pts.loc[ids, "aef_year"] = year


def locate(points, index):
    """Add `url`, `row`, `col`, `aef_year` (pixel in the COG, and the year of the file used) to `points`.

    The file of `emb_year` is used when it covers the point; otherwise the nearest year that does (so no
    point is dropped for want of imagery in its own year). Points covered in no year keep url = None.
    """
    pts = points.copy()
    pts["url"] = None
    pts["row"] = -1
    pts["col"] = -1
    pts["aef_year"] = 0
    years = sorted(index["year"].unique())
    for yr in sorted(pts["emb_year"].unique()):
        for cand in sorted(years, key=lambda c: (abs(c - yr), c)):  # own year first, then nearest
            todo = (pts["emb_year"] == yr) & pts["url"].isna()
            if not todo.any():
                break
            _locate_year(pts, index, cand, todo)
    return pts


def n_blocks(located):
    """Number of distinct (file, block) pairs that `extract` has to fetch."""
    ok = located[located["url"].notna()]
    return len(set(zip(ok["url"], ok["row"] // BLOCK, ok["col"] // BLOCK)))


def dequantize(raw):
    v = raw.astype(np.float32)
    return np.where(raw == NODATA, np.nan, (v / 127.5) ** 2 * np.sign(v))


def _disc_offsets(radius_m):
    r = int(round(radius_m / PIXEL_M))
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    return r, (dy**2 + dx**2) * PIXEL_M**2 <= radius_m**2 + 1e-6


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, np.nan)


def _read_file(url, rows, cols, radius_m, retries=3):
    """Centre-pixel and disc-mean embeddings for pixels (rows[i], cols[i]) of one COG."""
    import rasterio
    from rasterio.windows import Window

    r, disc = _disc_offsets(radius_m)
    centre = np.full((len(rows), N_DIM), np.nan, np.float32)
    mean = np.full((len(rows), N_DIM), np.nan, np.float32)
    n_valid = np.zeros(len(rows), np.int16)
    order = np.lexsort((cols, rows // BLOCK * 100 + cols // BLOCK))  # points of one block are read together
    for attempt in range(retries):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
                              GDAL_CACHE_MAX="1500", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
                with rasterio.open(url) as ds:
                    h, w = ds.height, ds.width
                    for i in order:
                        r0, c0 = max(rows[i] - r, 0), max(cols[i] - r, 0)
                        r1, c1 = min(rows[i] + r + 1, h), min(cols[i] + r + 1, w)
                        raw = ds.read(window=Window(c0, r0, c1 - c0, r1 - r0))  # (64, h, w) int8
                        emb = dequantize(raw)
                        ci, cj = rows[i] - r0, cols[i] - c0
                        centre[i] = emb[:, ci, cj]
                        sel = disc[r0 - rows[i] + r : r1 - rows[i] + r, c0 - cols[i] + r : c1 - cols[i] + r]
                        valid = sel & (raw[0] != NODATA)
                        n_valid[i] = valid.sum()
                        if n_valid[i]:
                            mean[i] = emb[:, valid].mean(axis=1)
            return _unit(centre), _unit(mean), n_valid
        except Exception as e:  # network hiccup: retry the whole file, then give up (NaN rows)
            print(f"[aef] {url[-40:]} attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(5)
    return centre, mean, n_valid


def extract(located, radius_m=50.0, workers=8, log_every=20):
    """(centre, disc_mean, n_valid_in_disc) arrays aligned to the rows of `located` (NaN where unavailable)."""
    n = len(located)
    centre = np.full((n, N_DIM), np.nan, np.float32)
    mean = np.full((n, N_DIM), np.nan, np.float32)
    n_valid = np.zeros(n, np.int16)
    groups = [(url, g.index.to_numpy()) for url, g in located[located["url"].notna()].groupby("url")]
    pos = {idx: k for k, idx in enumerate(located.index)}
    t0, done = time.time(), 0

    def work(item):
        url, idx = item
        sub = located.loc[idx]
        return idx, _read_file(url, sub["row"].to_numpy(), sub["col"].to_numpy(), radius_m)

    with ThreadPoolExecutor(workers) as ex:
        for idx, (c, m, v) in ex.map(work, groups):
            k = [pos[i] for i in idx]
            centre[k], mean[k], n_valid[k] = c, m, v
            done += 1
            if done % log_every == 0 or done == len(groups):
                print(f"[aef] {done}/{len(groups)} files, {time.time() - t0:.0f} s", flush=True)
    return centre, mean, n_valid


def point_key(df):
    """Stable key of a (lat, lon, emb_year) point; embeddings are stored under it."""
    return df["lat"].map("{:.6f}".format) + "_" + df["lon"].map("{:.6f}".format) + "_" + df["emb_year"].astype(str)


def embed_columns(prefix):
    return [f"{prefix}{i:02d}" for i in range(N_DIM)]


def load_points(shard_dir):
    """All extracted points under `shard_dir` (parquet shards written by scripts/baseline_eval/01_extract_aef.py)."""
    files = sorted(Path(shard_dir).glob("part_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def matrix_for(task_df, points, pooling="disc"):
    """(n, 64) embedding matrix aligned to `task_df` rows (NaN where missing). pooling: "disc" or "center"."""
    cols = embed_columns("d" if pooling == "disc" else "c")
    table = points.drop_duplicates("key").set_index("key")[cols]
    return table.reindex(point_key(task_df)).to_numpy(np.float32)
