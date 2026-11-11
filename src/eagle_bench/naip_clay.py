"""Clay v1.5 NAIP embeddings (Source Cooperative `clay/clay-v1-5-naip-2`) averaged over a disc around a point.

Source: https://source.coop/clay/clay-v1-5-naip-2 (Development Seed for Clay; embeddings CC-BY 4.0). Every NAIP
quarter-quad image 2010-2021 is cut into non-overlapping 256 x 256 px chips (256 m at 1 m GSD, ~154 m at 60 cm) and
each chip has one 1024-d Clay v1.5 class embedding. One parquet per image: `geometry` (chip footprint, lon/lat) and
`embeddings`. Files are 3-4 MB and cannot be read in part, so a file is downloaded whole, used, and dropped:

    index = load_index(cache)                                   # every file, with the 3.75' cell it covers
    pairs = plan(points, index)                                 # which (point, file) to read
    partial = process_file(file_key, cell, points_of_file, radii, ...)   # sums over chips, per (point, radius)
    pooled = merge(partials, points, ...)                       # mean embedding per (point, radius)

No chip is kept: only per-(point, file) weighted sums (resume checkpoints) and the final per-point means.

How a disc is averaged: the disc is sampled on a regular grid (spacing radius/12, 4-40 m); each sample takes the
chip that contains it, in the file for the quarter-quad cell the sample falls in (so the padding that NAIP images
have beyond their quad is not double counted) and the NAIP year nearest to the point's target year that the cell
has. The embedding is the sample-weighted mean of the chips (about area weighting). Radius 0 = the chip under the
point. A chip is >= 154 m wide, so radii below ~150 m mostly return the chip under the point or its neighbours.
"""

import io
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "datasets/raw/naip_clay"  # gitignored
BUCKET = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
PREFIX = "clay/clay-v1-5-naip-2/"
HTTP_ROOT = "https://data.source.coop/"
CELL = 0.0625  # quarter-quad = 3.75 arc minutes
M_LAT = 110574.0  # metres per degree latitude (mid-latitudes; the error is < 1% over the US)
M_LON = 111320.0  # metres per degree longitude at the equator
N_DIM = 1024
YEARS = (2010, 2021)  # NAIP years in the dataset
UA = {"User-Agent": "curl/8"}  # the default urllib agent is refused by some hosts

_NAME = re.compile(r"^m_(\d{2})(\d{3})(\d{2})_(ne|nw|se|sw)_")


def embed_columns():
    return [f"e{i:04d}" for i in range(N_DIM)]


# ---------------------------------------------------------------- index of the files


def _get(url, tries=6, timeout=120):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def _list(prefix, delimiter=None):
    import urllib.parse
    import xml.etree.ElementTree as ET

    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, prefixes, tok = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if tok:
            q["continuation-token"] = tok
        root = ET.fromstring(_get(BUCKET + "?" + urllib.parse.urlencode(q)))
        keys += [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)) for c in root.findall("s:Contents", ns)]
        prefixes += [c.find("s:Prefix", ns).text for c in root.findall("s:CommonPrefixes", ns)]
        if root.find("s:IsTruncated", ns).text != "true":
            return keys, prefixes
        tok = root.find("s:NextContinuationToken", ns).text


def cell_of_name(name):
    """(iy, ix) of the 3.75' cell a USGS-named file covers (`m_3211002_se_12_1_...`), or None.

    Name = lat of the 1-degree block's south edge (2 digits), lon of its east edge (3), 7.5' quad number 01-64 counted
    from the NW corner along rows (2), quarter (ne/nw/se/sw). Checked against the chip footprints of
    az/2015/100cm/32110/m_3211002_se_12_1_20150530 (cell lon -110.8125..-110.75, lat 32.875..32.9375).
    """
    m = _NAME.match(name)
    if not m:
        return None
    la, lo, n, q = int(m[1]), int(m[2]), int(m[3]), m[4]
    r, c = divmod(n - 1, 8)
    south = la + 1 - (r + 1) * 0.125 + (CELL if q[0] == "n" else 0.0)
    west = -(lo + 1) + c * 0.125 + (CELL if q[1] == "e" else 0.0)
    return int(round(south / CELL)), int(round(west / CELL))


def build_index(cache_dir=CACHE_DIR, workers=16):
    """List the bucket (about 1.2 M files, 3 min) and save `naip_index.parquet` in `cache_dir`."""
    t0 = time.time()
    _, states = _list(PREFIX, "/")
    with ThreadPoolExecutor(workers) as ex:
        sy = [y for _, ys in ex.map(lambda p: _list(p, "/"), states) for y in ys]
        rows = [k for ks, _ in ex.map(_list, sy) for k in ks]
    df = pd.DataFrame(rows, columns=["key", "size"])
    df = df[df["key"].str.endswith(".parquet")].copy()
    parts = df["key"].str[len(PREFIX):].str.split("/", expand=True)
    df["year"] = pd.to_numeric(parts[1], errors="coerce")
    cells = [cell_of_name(n) for n in parts[5].fillna("")]
    df["iy"] = [c[0] if c else -1 for c in cells]
    df["ix"] = [c[1] if c else 0 for c in cells]
    n_all = len(df)
    df = df[df["iy"] != -1].dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    # a cell-year can have several images (e.g. a quad split by a UTM zone): keep the largest
    df = df.sort_values("size", ascending=False).drop_duplicates(["iy", "ix", "year"]).sort_values("key")
    df = df.reset_index(drop=True)
    df.insert(0, "file_id", np.arange(len(df)))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(cache_dir) / "naip_index.parquet")
    print(f"[naip_clay] index: {len(df)} files with a cell ({n_all - len(df)} skipped: unrecognised name or "
          f"duplicate cell-year), {df['size'].sum() / 1e12:.2f} TB, {time.time() - t0:.0f}s", flush=True)
    return df


def load_index(cache_dir=CACHE_DIR):
    path = Path(cache_dir) / "naip_index.parquet"
    return pd.read_parquet(path) if path.exists() else build_index(cache_dir)


class CellFiles:
    """(iy, ix) -> {year: file_id}, with the year choice used everywhere: nearest to the target, ties to the later."""

    def __init__(self, index):
        self.files = {}
        for fid, iy, ix, y in zip(index["file_id"], index["iy"], index["ix"], index["year"]):
            self.files.setdefault((int(iy), int(ix)), {})[int(y)] = int(fid)

    def pick(self, iy, ix, target):
        ys = self.files.get((int(iy), int(ix)))
        if not ys:
            return None
        y = min(ys, key=lambda y: (abs(y - target), -y))
        return ys[y], y


# ---------------------------------------------------------------- sampling a disc


_OFFSETS = {}


def disc_offsets(radius):
    """(n, 2) east/north offsets in metres of the samples that stand for a disc (one at 0 for radius 0)."""
    r = float(radius)
    if r not in _OFFSETS:
        if r <= 0:
            _OFFSETS[r] = np.zeros((1, 2))
        else:
            s = min(max(r / 12.0, 4.0), 40.0)
            g = np.arange(-int(r // s), int(r // s) + 1) * s
            dx, dy = np.meshgrid(g, g)
            m = dx ** 2 + dy ** 2 <= r * r + 1e-6
            _OFFSETS[r] = np.stack([dx[m], dy[m]], 1)
    return _OFFSETS[r]


def samples(lat, lon, radius):
    """Sample positions (lat, lon arrays) of the disc around a point."""
    o = disc_offsets(radius)
    return lat + o[:, 1] / M_LAT, lon + o[:, 0] / (M_LON * math.cos(math.radians(lat)))


# ---------------------------------------------------------------- plan: which (point, file) pairs


def plan(points, index, cells=None):
    """(point row, file) pairs: for each point, the file of every cell that its largest disc touches.

    `points` needs `lat`, `lon`, `emb_year` (target NAIP year) and `rmax` (largest radius needed, metres).
    Returns a DataFrame with `pt` (row in `points`), `file_id`, `year`, `iy`, `ix`.
    """
    cells = cells or CellFiles(index)
    lat, lon, r = (points[c].to_numpy(float) for c in ("lat", "lon", "rmax"))
    cos = np.cos(np.radians(lat))
    dlat, dlon = r / M_LAT, r / (M_LON * cos)
    iy0, iy1 = np.floor((lat - dlat) / CELL).astype(int), np.floor((lat + dlat) / CELL).astype(int)
    ix0, ix1 = np.floor((lon - dlon) / CELL).astype(int), np.floor((lon + dlon) / CELL).astype(int)
    out = []
    for a in range(int((iy1 - iy0).max()) + 1):
        for b in range(int((ix1 - ix0).max()) + 1):
            iy, ix = iy0 + a, ix0 + b
            dy = np.maximum.reduce([iy * CELL - lat, np.zeros_like(lat), lat - (iy + 1) * CELL]) * M_LAT
            dx = np.maximum.reduce([ix * CELL - lon, np.zeros_like(lat), lon - (ix + 1) * CELL]) * M_LON * cos
            for i in np.flatnonzero((iy <= iy1) & (ix <= ix1) & (np.hypot(dx, dy) <= r + 1e-6)):
                hit = cells.pick(iy[i], ix[i], points["emb_year"].iat[i])
                if hit:
                    out.append((i, hit[0], hit[1], int(iy[i]), int(ix[i])))
    return pd.DataFrame(out, columns=["pt", "file_id", "year", "iy", "ix"])


# ---------------------------------------------------------------- one file


def read_chips(key):
    """Download one embedding file: chip bounds (n, 4: west, south, east, north) and embeddings (n, 1024) float32."""
    import pyarrow.parquet as pq
    import shapely

    t = pq.read_table(io.BytesIO(_get(HTTP_ROOT + key)))
    bounds = shapely.bounds(shapely.from_wkb(t["geometry"].to_numpy(zero_copy_only=False)))
    emb = t["embeddings"].combine_chunks().values.to_numpy().reshape(len(bounds), -1).astype(np.float32)
    return bounds, emb


def chip_sums(bounds, emb, cell, lat, lon, radii):
    """Per radius: (matched samples, weighted sum of chip embeddings) for the samples that fall in `cell` and in a chip."""
    from scipy.spatial import cKDTree

    iy, ix = cell
    lat0 = float((bounds[:, 1].mean() + bounds[:, 3].mean()) / 2)
    kx, ky = M_LON * math.cos(math.radians(lat0)), M_LAT
    cx, cy = (bounds[:, 0] + bounds[:, 2]) / 2 * kx, (bounds[:, 1] + bounds[:, 3]) / 2 * ky
    hw, hh = (bounds[:, 2] - bounds[:, 0]) / 2 * kx, (bounds[:, 3] - bounds[:, 1]) / 2 * ky
    tree = cKDTree(np.stack([cx, cy], 1))
    res = []
    for r in radii:
        slat, slon = samples(lat, lon, r)
        own = (slat >= iy * CELL) & (slat < (iy + 1) * CELL) & (slon >= ix * CELL) & (slon < (ix + 1) * CELL)
        n, tot = 0, np.zeros(emb.shape[1], np.float32)
        if own.any():
            sx, sy = slon[own] * kx, slat[own] * ky
            d, j = tree.query(np.stack([sx, sy], 1), p=np.inf, distance_upper_bound=float(max(hw.max(), hh.max())) * 1.5)
            ok = np.isfinite(d)
            j = np.where(ok, j, 0)  # nearest chip centre; kept only if the sample lies inside that chip
            ok &= (np.abs(sx - cx[j]) <= hw[j] * 1.02) & (np.abs(sy - cy[j]) <= hh[j] * 1.02)
            if ok.any():
                cnt = np.bincount(j[ok], minlength=len(bounds))
                n = int(cnt.sum())
                nz = np.flatnonzero(cnt)
                tot = (cnt[nz, None].astype(np.float32) * emb[nz]).sum(0)
        res.append((n, tot))
    return res


def n_samples(radius):
    return len(disc_offsets(radius))
