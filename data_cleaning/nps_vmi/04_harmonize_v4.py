"""Stage 4: harmonize the PLOTS-v4-schema packages into two tidy parquet tables.

Outputs (datasets/NPS_VMI/):
  vmi_plot_events.parquet   one row per plot visit (plot + event + project metadata + lon/lat)
  vmi_plot_species.parquet  one row per species x stratum x event (cover as reported + midpoint %)
Adds `qa_flags` (semicolon-separated) rather than dropping anything; see README.md in this folder for flag meanings.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt

RAW = Path("datasets/raw/nps_vmi")
OUT = Path("datasets/NPS_VMI")
OUT.mkdir(exist_ok=True)
prof = pd.read_csv(Path(__file__).parent / "catalog" / "package_profile.csv")
packages = sorted(set(prof[prof.file.str.contains(r"(?:^|_)tPlotEvents\.csv$")].package))


def load(pk, table):
    f = sorted(glob.glob(f"{RAW}/{pk}/*_{table}.csv") + glob.glob(f"{RAW}/{pk}/{table}.csv"), key=len)  # shortest = exact table
    if not f:
        return pd.DataFrame()
    return pd.read_csv(f[0], dtype=str, keep_default_na=False, na_values=["NA", ""], encoding_errors="replace")


def num(s):
    return pd.to_numeric(s, errors="coerce")


events, species = [], []
for pk in packages:
    unit = pk.split("_")[0]
    ev, pl, md = load(pk, "tPlotEvents"), load(pk, "tPlots"), load(pk, "tProjectMetadata")
    sp, tx = load(pk, "tPlotEventSpecies"), load(pk, "tSpecies")
    md1 = md.iloc[0] if len(md) else pd.Series(dtype=str)
    e = ev.merge(pl, on="Plot_Code", how="left", suffixes=("", "_plot"))
    e.insert(0, "package", pk)
    e.insert(1, "unit_code", md1.get("Park_Code", unit) if pd.notna(md1.get("Park_Code", unit)) else unit)
    e["event_key"] = pk + "|" + e["Plot_Event"].astype(str)
    for c in ["Project_Name", "Coordinate_System", "Datum", "Coordinate_Method", "Sampling_Method", "Date_Created"]:
        e["md_" + c.lower()] = md1.get(c)
    events.append(e)
    if len(sp):
        s = sp.merge(tx.rename(columns={"Concept_Reference": "Taxon_Reference"}), on="Spp_Code", how="left", suffixes=("", "_tx"))
        s.insert(0, "package", pk)
        s["event_key"] = pk + "|" + s["Plot_Event"].astype(str)
        species.append(s)

E = pd.concat(events, ignore_index=True)
S = pd.concat(species, ignore_index=True)

# ---- coordinates -> lon/lat (shared repair logic, see coords.py) --------------------------
sys.path.insert(0, str(Path(__file__).parent))
from coords import to_lonlat  # noqa: E402

E = E.reset_index(drop=True)
nad83 = ~E.md_datum.fillna("").str.upper().str.contains("WGS")
zone = E.UTM_Zone.str.extract(r"(\d{1,2})")[0]
lon, lat, flags, dist = (np.full(len(E), np.nan), np.full(len(E), np.nan), [""] * len(E), np.full(len(E), np.nan))
for pk, idx in E.groupby("package").indices.items():
    lo, la, fl, d = to_lonlat(pk, E.Field_X.iloc[idx], E.Field_Y.iloc[idx], zone.iloc[idx], nad83.iloc[idx])
    lon[idx], lat[idx], dist[idx] = lo, la, d
    for j, f in zip(idx, fl):
        flags[j] = f
is_ll = (num(E.Field_X).abs() <= 180) & (num(E.Field_Y).abs() <= 90)
E["bbox_dist_deg"] = dist
flags = pd.Series(flags, index=E.index)
flags[is_ll & E.md_coordinate_system.fillna("").str.contains("UTM")] += ";ll_but_metadata_says_utm;"
lon, lat = pd.Series(lon, index=E.index), pd.Series(lat, index=E.index)
E["lon"], E["lat"], E["coord_from_latlon"] = lon, lat, is_ll
E["qa_flags"] = flags.map(lambda t: ";".join(dict.fromkeys(f for f in t.split(";") if f)))
E["event_date"] = pd.to_datetime(E.Event_Date, errors="coerce")
E["year"] = E.event_date.dt.year
E["aa_plot"] = E["AA_Plot"].map({"1": True, "0": False, "True": True, "False": False}) if "AA_Plot" in E else pd.NA
for c in ["Public_X", "Public_Y", "Field_X", "Field_Y", "Plot_Radius", "Plot_Width", "Plot_Length", "GPS_Error", "Elevation",
          "Slope_Degrees", "Aspect_Degrees"]:
    if c in E:
        E[c] = num(E[c])

# ---- species -----------------------------------------------------------------------------
S["cover_pct"] = num(S["Real_Cover"]) if "Real_Cover" in S else np.nan
S["cover_pct_source"] = np.where(S.cover_pct.notna(), "reported", None)
for alt in ("Real_Plot_Cover", "Total_Plot_Cover"):  # variants of tPlotEventSpecies that store % cover under another name
    if alt in S:
        v = num(S[alt])
        use = S.cover_pct.isna() & v.notna()
        S.loc[use, "cover_pct"], S.loc[use, "cover_pct_source"] = v[use], f"reported({alt})"
S["cover_code"] = S["Cover_Code"] if "Cover_Code" in S else S.get("Obs_Cover_Code")
S["stratum"] = S["Stratum"] if "Stratum" in S else pd.NA
S["species_name"] = S["Field_Name"]
S["qa_flags"] = np.where(S.cover_pct.isna(), "no_real_cover;", "")
S["qa_flags"] = np.where(S.cover_pct > 100, S.qa_flags + "cover_gt_100;", S.qa_flags)

E.to_parquet(OUT / "vmi_plot_events.parquet", index=False)
S.to_parquet(OUT / "vmi_plot_species.parquet", index=False)
print(len(packages), "packages;", len(E), "plot events;", len(S), "species records")
print(E.qa_flags.str.split(";").explode().value_counts().head(10))
print("events with lon/lat:", E.lon.notna().sum())
