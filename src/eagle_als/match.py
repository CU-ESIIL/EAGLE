"""Match labeled locations to the 3DEP lidar product closest in time (the one place this logic lives).

    from eagle_als.match import match_3dep, apply_year_gap
    matched, gaps = match_3dep(df, "lat", "lon", "year")      # every covered row, with its gap
    eval_df = apply_year_gap(matched, 3)                       # the evaluation table: |gap| <= N

Adds `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS` (absolute gap, years), `signed_year_diff`
(collection year minus survey year; negative = lidar flown before the survey) and `als_site_id`. One 3DEP lookup
per unique (lat, lon, year). A lookup that raises (a footprint without a collection year) counts as no match.
`gaps` is the table to save as catalog/year_gap_counts.csv: rows surviving for N = 0..5.

Coverage edge case (accepted, user 2026-11-12): a site is matched to a product whose footprint polygon contains it, and
footprints are coarser than the point clouds, so at collection edges the cookie can hold few or no points around the site
(benchmark eval cache: 436 of 82,160 cookies have no point within 50 m, 1,702 have < 50% of the 3DEP minimum density).
Such sites are not rematched; eagle_als.cookie_qc flags them (scripts/baseline_eval/10_cookie_qc.py).

Pooled labels (eBird encounter rates, long-term counts) have no single survey year; compute the gap to the pooled
period yourself, as in data_cleaning/satbird/02_filter_and_match_als.py, and document the rule.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eagle_als.sites import add_site_ids

_3DEP = str(Path(__file__).resolve().parents[1] / "streaming/3dep")
GAP_RANGE = range(0, 6)


def _lookup():
    if _3DEP not in sys.path:
        sys.path.insert(0, _3DEP)
    from get_als import get_nearest_year_product

    return get_nearest_year_product


def match_3dep(df, lat_col="lat", lon_col="lon", year_col="year", source="AWS", split_col=None):
    """Returns (matched, gap_counts). `matched` keeps the covered rows of `df` (input row order and index);
    rows with no footprint or a missing year are dropped. split_col (e.g. "test_split") adds per-split counts."""
    find = _lookup()
    d = df.copy()
    keys = d[[lat_col, lon_col, year_col]].dropna().drop_duplicates()
    rows = []
    for la, lo, yr in keys.itertuples(index=False):
        try:
            r = find(float(la), float(lo), int(round(yr)), source=source)
        except Exception:  # footprint without a collection year, bad geometry, ...
            r = None
        rows.append(r or {})
    hits = pd.concat([keys.reset_index(drop=True), pd.DataFrame(rows, index=range(len(keys)))], axis=1)
    need = [f"product_name_{source}", f"collection_year_{source}", f"year_diff_{source}"]
    for c in need:
        if c not in hits:
            hits[c] = np.nan
    n_in = len(d)
    d = d.reset_index().merge(hits, on=[lat_col, lon_col, year_col], how="left").set_index("index")
    d.index.name = df.index.name
    d = d[d[need[0]].notna()].copy()
    d[need[1]] = d[need[1]].astype(int)
    d[need[2]] = d[need[2]].astype(int)
    d["signed_year_diff"] = d[need[1]] - d[year_col].round().astype(int)
    d = add_site_ids(d, lat_col, lon_col, tile_col=need[0])
    print(f"[match_3dep] {len(d)} of {n_in} rows covered by a 3DEP product with a collection year")
    return d, gap_counts(d, split_col, f"year_diff_{source}")


def gap_counts(d, split_col=None, gap_col="year_diff_AWS"):
    out = pd.DataFrame({"max_year_diff_N": list(GAP_RANGE), "n_rows": [int((d[gap_col] <= n).sum()) for n in GAP_RANGE]})
    if split_col:
        for val, name in ((False, "n_train"), (True, "n_test")):
            sub = d[d[split_col] == val]
            out[name] = [int((sub[gap_col] <= n).sum()) for n in GAP_RANGE]
    return out


def apply_year_gap(matched, max_year_diff=3, gap_col="year_diff_AWS"):
    """The evaluation rows: gap <= max_year_diff (3 is the protocol default; document any other)."""
    return matched[matched[gap_col] <= max_year_diff].copy()
