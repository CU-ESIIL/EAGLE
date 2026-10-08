"""Stage 3: build one table of LFRDB plots, keep those with trustworthy coordinates, and match
each to the 3DEP lidar (ALS) product closest in time.

Filters, in order (counts after each step go to catalog/filter_counts_stage3.csv):
  1. has coordinates
  2. LocMeth == 'G' (location captured with a GPS unit in the field)
  3. at least MIN_DECIMALS decimals in both lat and lon (5 decimals ~ 1 m). Needed even for
     GPS plots: sources flagged "YES, plot locations excluded" are marked GPS but rounded to <=3 decimals
  4. visit Type == 'field visit' (drops aerial/helicopter surveys, photo interpretation, remote)
  5. a plausible visit year
  6. a 3DEP footprint covers the point (src/streaming/3dep/get_als.get_nearest_year_product,
     which returns the covering product whose collection year is closest to the visit year)
  7. |collection year - visit year| <= MAX_YEAR_DIFF

Output (gitignored, regenerable): datasets/raw/LFRDB/derived/lfrdb_gps_als_matched.parquet.
Labels with missing values are NOT dropped here; stage 4 summarizes them and stage 5 filters.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/streaming/3dep"))
from eagle_als.sites import add_site_ids  # noqa: E402
from get_als import get_nearest_year_product  # noqa: E402

RAW = ROOT / "datasets/raw/LFRDB"
DERIVED = RAW / "derived"
CAT = Path(__file__).parent / "catalog"
MIN_DECIMALS = 5
MAX_YEAR_DIFF = 3
YEAR_RANGE = (1970, 2025)
# plausibility bounds for lifeform heights (m); larger values are unit/typo errors in the source
MAX_HEIGHT = {"tree": 120.0, "shrub": 20.0, "herb": 10.0}


def read_all(table):
    parts = []
    for f in sorted(RAW.glob(f"*/tables/{table}.parquet")):
        parts.append(pd.read_parquet(f).assign(region=f.parts[-3]))
    return pd.concat(parts, ignore_index=True)


def n_decimals(s):
    """Decimals in the exported text of a coordinate, ignoring trailing zeros."""
    return s.fillna("").str.split(".").str[1].fillna("").str.rstrip("0").str.len()


def build_plot_table():
    pts = read_all("dtPoints")
    vis = read_all("dtVisits").drop(columns="region")
    com = read_all("dtCommunities").drop(columns="region")
    sta = read_all("dtStands").drop(columns="region")
    src = read_all("lutdtVisitsSourceID").drop(columns="region").drop_duplicates("SourceID")  # same table in every region
    for name, t in [("dtPoints", pts), ("dtVisits", vis), ("dtCommunities", com), ("dtStands", sta)]:
        assert not t.EventID.duplicated().any(), f"{name}: EventID not unique"

    d = pts.merge(vis, on="EventID", how="left").merge(com, on="EventID", how="left").merge(sta, on="EventID", how="left")
    d = d.merge(src[["SourceID", "AgencyCd", "PublicAccess"]], on="SourceID", how="left")
    d["n_decimals"] = np.minimum(n_decimals(d.Lat), n_decimals(d.Long))
    d["lat"] = pd.to_numeric(d.Lat, errors="coerce")
    d["lon"] = pd.to_numeric(d.Long, errors="coerce")
    d["year"] = pd.to_numeric(d.YYYY, errors="coerce")
    d["month"] = pd.to_numeric(d.MM, errors="coerce")
    return d


def tidy_columns(d):
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    out = pd.DataFrame({
        "plot_id": d.EventID,
        "lat": d.lat, "lon": d.lon,
        "year": d.year.astype("Int64"), "month": d.month.astype("Int64"),
        "lfrdb_region": d.region, "source_id": d.SourceID, "source_agency": d.AgencyCd,
        "loc_method": d.LocMeth, "n_decimals": d.n_decimals.astype(int),
        # classification columns
        "ecosys_code": num("EcoSysCd").astype("Int64"), "ecosys": d.EcoSys,
        "ecosys_lifeform": d.EcoSysLifeformPrimary,
        "nvc_group_code": num("NVCSGroupCd").astype("Int64"), "nvc_group": d.NVCSGroup,
        "evt_method": d.EVTMeth, "dominant_lifeform": d.DomLifeform, "dominant_species": d.DomSp,
        # regression columns: percent cover adjusted for overlap (0-100), height in m
        "tree_cover_pct": num("LFTreeCovAdj"), "shrub_cover_pct": num("LFShrubCovAdj"),
        "herb_cover_pct": num("LFHerbCovAdj"),
        "tree_height_m": num("LFTreeHgt"), "shrub_height_m": num("LFShrubHgt"), "herb_height_m": num("LFHerbHgt"),
    })
    notes = []
    for k, lim in MAX_HEIGHT.items():
        c = f"{k}_height_m"
        bad = out[c] > lim
        notes.append((c, f"> {lim} m set to null", int(bad.sum())))
        out.loc[bad, c] = np.nan
    return out, pd.DataFrame(notes, columns=["column", "rule", "n_values_nulled"])


def main():
    DERIVED.mkdir(exist_ok=True)
    CAT.mkdir(exist_ok=True)
    d = build_plot_table()
    counts = [("all plots in the nine public regional databases", len(d))]

    def keep(mask, label):
        nonlocal d
        d = d[mask.fillna(False)]
        counts.append((label, len(d)))

    keep(d.lat.between(-90, 90) & d.lon.between(-180, 180) & ~((d.lat == 0) & (d.lon == 0)), "1. has valid coordinates")
    keep(d.LocMeth == "G", "2. LocMeth = G (GPS in the field)")
    keep(d.n_decimals >= MIN_DECIMALS, f"3. >= {MIN_DECIMALS} decimals in lat and lon")
    keep(d.Type == "field visit", "4. visit type = field visit")
    keep(d.year.between(*YEAR_RANGE), f"5. visit year in {YEAR_RANGE[0]}-{YEAR_RANGE[1]}")

    # ALS lookup once per unique (lat, lon, year)
    keys = d[["lat", "lon", "year"]].drop_duplicates()
    res = []
    for la, lo, yr in keys.itertuples(index=False):
        try:
            r = get_nearest_year_product(float(la), float(lo), int(yr), source="AWS")
        except Exception:  # footprint without a collection year
            r = None
        res.append(r or {})
    hits = pd.concat([keys.reset_index(drop=True), pd.DataFrame(res, index=range(len(keys)))], axis=1)
    d = d.merge(hits, on=["lat", "lon", "year"], how="left")
    keep(d.product_name_AWS.notna(), "6. covered by a 3DEP footprint with a collection year")
    keep(d.year_diff_AWS <= MAX_YEAR_DIFF, f"7. 3DEP collection year within +/-{MAX_YEAR_DIFF} yr of visit")

    out, height_notes = tidy_columns(d)
    for c in ["product_name_AWS", "collection_year_AWS", "year_diff_AWS"]:
        out[c] = d[c].values
    out["collection_year_AWS"] = out.collection_year_AWS.astype(int)
    out["year_diff_AWS"] = out.year_diff_AWS.astype(int)
    out = add_site_ids(out, "lat", "lon")
    out.to_parquet(DERIVED / "lfrdb_gps_als_matched.parquet", index=False)

    fc = pd.DataFrame(counts, columns=["step", "n_plots"])
    fc["dropped"] = (-fc.n_plots.diff()).fillna(0).astype(int)
    fc.to_csv(CAT / "filter_counts_stage3.csv", index=False)
    height_notes.to_csv(CAT / "height_nulling.csv", index=False)
    print(fc.to_string(index=False))
    print(height_notes.to_string(index=False))


if __name__ == "__main__":
    main()
