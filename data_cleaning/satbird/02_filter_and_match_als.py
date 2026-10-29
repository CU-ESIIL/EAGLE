"""Stage 2: attach eBird hotspot coordinates to the SatBird targets, filter, and match each hotspot to a
3DEP lidar (ALS) product collected inside the period the labels cover.

One row = one eBird hotspot (SatBird "location"). Its labels POOL the complete June-July checklists of 2010-2023
(Teng et al. 2023), so there is no single survey year. A lidar product is therefore a valid match when its
collection year lies anywhere in LABEL_YEARS = 2010..2023, not within N years of one date. Among the products
covering a point, the one with the smallest gap to the window is kept (gap 0 = inside); ties go to the LATEST
collection year (the preference rule: newest lidar inside the window, closest to the 2022 Sentinel-2 image and the
most recent checklists).
`year_diff_AWS` is this gap: 0 = collected inside 2010-2023, otherwise years between the collection year and the nearest
window edge (the eval tables keep gap 0 only). `year` = REFERENCE_YEAR (2022, the paired Sentinel-2 year) is kept for the standard table; it is not a
survey date. Caveat: with a window of 14 years the lidar can be up to 13 years from some of the checklists.

Filters, in order (counts after each step go to catalog/filter_counts.csv):
  1. hotspot has a target vector in the SatBird pickle (all 122,593 do)
  2. hotspot is still returned by the eBird API `ref/hotspot/US-<state>` (stage 1). Hotspots that eBird has since
     merged, hidden or turned into personal locations have no public coordinates; eBird does not serve them
  3. valid coordinates
  4. not "moved": if the optional cross-check file (SatBird authors' split CSVs, see CROSSCHECK) lists the hotspot,
     its 2023 coordinates must be within MAX_SHIFT_M of eBird's current ones; otherwise the labels belong to
     a different place than the coordinates now say. Hotspots absent from that file cannot be checked and are kept.
     The CSVs are used ONLY to drop rows, never as a source of coordinates
  5. a 3DEP footprint covers the point with a collection year
  6. a covering product was collected inside LABEL_YEARS (gap 0)

Reads datasets/raw/satbird/{satbird_usa_summer_targets.pkl, ebird_hotspots/*.json} (stage 1).
Writes datasets/raw/satbird/derived/hotspots_als_covered.parquet (every covered hotspot with its gap, so a looser
rule can be applied later), catalog/filter_counts.csv, catalog/year_gap_counts.csv (rows kept for gap <= N, N = 0..5).
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/streaming/3dep"))
from eagle_als.sites import add_site_ids  # noqa: E402
from get_als import lookup_3dep_by_latlon  # noqa: E402

RAW = ROOT / "datasets/raw/satbird"
DERIVED = RAW / "derived"
CAT = Path(__file__).parent / "catalog"
LABEL_YEARS = (2010, 2023)  # checklist years pooled in the SatBird labels
REFERENCE_YEAR = 2022  # paired Sentinel-2 year; only fills the `year` column
MAX_YEAR_DIFF = 0  # gap to LABEL_YEARS allowed in the eval table
MAX_REPORT = 5
MAX_SHIFT_M = 100.0  # = radius of a lidar cookie
CROSSCHECK = sorted(glob.glob(str(RAW / "companion_splits/*/*_split.csv")))  # optional; skipped if absent


def shift_m(lat1, lon1, lat2, lon2):
    return np.hypot(lat1 - lat2, (lon1 - lon2) * np.cos(np.radians(lat1))) * 111_320


def main():
    CAT.mkdir(exist_ok=True)
    DERIVED.mkdir(exist_ok=True)
    ids = list(pickle.load(open(RAW / "satbird_usa_summer_targets.pkl", "rb")))
    counts = [("1. hotspots with SatBird targets", len(ids))]
    d = pd.DataFrame({"hotspot_id": ids})

    hs = pd.concat([pd.DataFrame(json.load(open(f))) for f in sorted(glob.glob(str(RAW / "ebird_hotspots/*.json")))])
    hs = hs.drop_duplicates("locId").rename(columns={"locId": "hotspot_id", "locName": "hotspot_name", "subnational1Code": "state_code",
                                                     "subnational2Code": "county_code", "numChecklistsAllTime": "n_checklists_all_time"})
    d = d.merge(hs[["hotspot_id", "hotspot_name", "state_code", "county_code", "lat", "lng", "n_checklists_all_time"]], on="hotspot_id", how="inner")
    d = d.rename(columns={"lng": "lon"})
    counts.append(("2. still a public eBird hotspot (coordinates from the eBird API)", len(d)))
    d = d[d.lat.between(-90, 90) & d.lon.between(-180, 180) & ~((d.lat == 0) & (d.lon == 0))]
    counts.append(("3. valid coordinates", len(d)))

    d["shift_m_vs_2023"] = np.nan
    if CROSSCHECK:
        old = pd.concat([pd.read_csv(f, usecols=["hotspot_id", "lat", "lon"]) for f in CROSSCHECK]).drop_duplicates("hotspot_id")
        m = d[["hotspot_id", "lat", "lon"]].merge(old, on="hotspot_id", suffixes=("", "_old"))
        d["shift_m_vs_2023"] = d.hotspot_id.map(dict(zip(m.hotspot_id, shift_m(m.lat, m.lon, m.lat_old, m.lon_old))))
        print(f"cross-check: {d.shift_m_vs_2023.notna().sum()} hotspots checked, {(d.shift_m_vs_2023 > MAX_SHIFT_M).sum()} moved > {MAX_SHIFT_M:.0f} m")
        d = d[~(d.shift_m_vs_2023 > MAX_SHIFT_M)]
    counts.append((f"4. not moved > {MAX_SHIFT_M:.0f} m since 2023 (where checkable)", len(d)))

    d["year"] = REFERENCE_YEAR
    keys = d[["lat", "lon"]].drop_duplicates()
    res = []
    for la, lo in keys.itertuples(index=False):
        m = lookup_3dep_by_latlon(float(la), float(lo), source="AWS")
        m = m[m.collection_year.notna()]
        if m.empty:
            res.append({})
            continue
        cy = m.collection_year.astype(int).to_numpy()
        gap = np.maximum(LABEL_YEARS[0] - cy, 0) + np.maximum(cy - LABEL_YEARS[1], 0)
        i = np.lexsort((-cy, gap))[0]  # smallest gap, then latest collection year
        res.append({"product_name_AWS": m.name.iloc[i], "collection_year_AWS": int(cy[i]), "year_diff_AWS": int(gap[i])})
    keys = pd.concat([keys.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    d = d.merge(keys, on=["lat", "lon"], how="left")
    d = d[d.product_name_AWS.notna() & d.year_diff_AWS.notna()].copy()
    counts.append(("5. covered by a 3DEP footprint with a collection year", len(d)))
    d["label_year_min"], d["label_year_max"] = LABEL_YEARS
    d["collection_year_AWS"] = d.collection_year_AWS.astype(int)
    d["year_diff_AWS"] = d.year_diff_AWS.astype(int)
    d = add_site_ids(d, "lat", "lon")
    d.to_parquet(DERIVED / "hotspots_als_covered.parquet", index=False)

    gap = pd.DataFrame({"max_years_outside_window_N": range(MAX_REPORT + 1)})
    gap["n_hotspots"] = [int((d.year_diff_AWS <= n).sum()) for n in gap.max_years_outside_window_N]
    gap.to_csv(CAT / "year_gap_counts.csv", index=False)
    d = d[d.year_diff_AWS <= MAX_YEAR_DIFF]
    counts.append((f"6. 3DEP collection year inside the label years {LABEL_YEARS[0]}-{LABEL_YEARS[1]}", len(d)))

    fc = pd.DataFrame(counts, columns=["step", "n_hotspots"])
    fc["dropped"] = (-fc.n_hotspots.diff()).fillna(0).astype(int)
    fc.to_csv(CAT / "filter_counts.csv", index=False)
    print(fc.to_string(index=False))
    print(gap.to_string(index=False))
    print(f"{d.als_site_id.nunique()} unique als_site_id, {d.product_name_AWS.nunique()} lidar products")


if __name__ == "__main__":
    main()
