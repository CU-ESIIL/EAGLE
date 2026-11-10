"""Stage 5: stack all harmonizable packages into one 'core' events table + species table.

Families handled
  v4    PLOTS v3/v4 schema (tPlots/tPlotEvents/tPlotEventSpecies)      <- from 04_harmonize_v4 output
  utah  NCPN-style tblPlotLocation/tblPlotDetails/tblVegetationDetails (+ AA* tables for accuracy-assessment pts)
  nerplots  'Plots' / 'Plots-Species' / 'AA_Observations' / 'AA-Species' (dotted or underscored column names)
Packages in other families (Alaska custom DBs, BIHO/CIRO aadata, ...) are NOT here; see catalog/package_status.csv.

Outputs (datasets/NPS_VMI/): vmi_core_events.parquet, vmi_core_species.parquet
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from coords import to_lonlat  # noqa: E402

RAW = Path("datasets/raw/nps_vmi")
OUT = Path("datasets/NPS_VMI")
prof = pd.read_csv(Path(__file__).parent / "catalog" / "package_profile.csv")


def norm(c):
    return re.sub(r"[^a-z0-9]", "", c.lower())


def load(pk, table):
    """Load table whose file name ends in `table` (any prefix), columns normalized to lowercase alnum."""
    cands = sorted(glob.glob(f"{RAW}/{pk}/*{table}.csv"), key=len)
    cands = [c for c in cands if re.search(rf"(^|[_/]){re.escape(table)}\.csv$", c)]
    if not cands:
        return None
    df = pd.read_csv(cands[0], dtype=str, keep_default_na=False, na_values=["NA", ""], encoding_errors="replace")
    df.columns = [norm(c) for c in df.columns]
    return df


def col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(np.nan, index=df.index, dtype=object)


def range_mid(s):
    """'5-15' -> 10, '<1' -> 0.5, '>95-100' -> 97.5, '>75%' -> 87.5, 'trace' -> 0.25, '10' -> 10."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).replace("%", "").replace(" ", "").lower()
    if t in ("trace", "tr", "few"):
        return 0.25
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", t)]
    if not nums:
        return np.nan
    if t.startswith("<"):
        return nums[0] / 2
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    return (nums[0] + 100) / 2 if t.startswith((">", ">=")) else nums[0]


def cover_lookup(df):
    """CoverClass -> midpoint % from a park's cover-class lookup table (several layouts exist)."""
    if df is None or "coverclass" not in df:
        return {}
    mid = next((c for c in ("midpoint", "mid", "coverclassveg", "coverclassmidpoint") if c in df), None)
    rng = next((c for c in ("coverclasspercent", "rangecover", "range", "coverclassdesc") if c in df), None)
    vals = df[mid].map(pd.to_numeric) if mid else df[rng].map(range_mid) if rng else None
    return dict(zip(df.coverclass.astype(str), vals)) if vals is not None else {}


def code_to_pct(code, lookup):
    """Resolve a cover-class code via the park lookup (tolerating zero padding), else parse it as a range string."""
    if code is None or (isinstance(code, float) and np.isnan(code)) or code is pd.NA:
        return np.nan, None
    c = str(code)
    for k in (c, c.zfill(2), c.lstrip("0") or "0"):
        v = lookup.get(k)
        if v is not None and not pd.isna(v):
            return v, "class_midpoint"
    v = range_mid(c) if re.search(r"[<>\-]|trace", c, re.I) else np.nan
    return v, ("class_midpoint_parsed_from_range" if not pd.isna(v) else None)


def ev_frame(pk, fam, plot_type, d):
    d = d.copy()
    d["package"], d["schema_family"], d["plot_type"] = pk, fam, plot_type
    d["event_key"] = pk + "|" + plot_type + "|" + d["event_id"].astype(str)
    return d


# NCPN cover-class fallback for parks whose lookup table lacks percentages (ARCH / GLCA schemes; ASSUMED, flagged)
NCPN_FALLBACK = {"01": 10, "02": 20, "03": 30, "04": 40, "05": 50, "06": 60, "07": 70, "08": 80, "09": 90, "10": 97.5, "P": 0.5, "T": 3,
                 "(t)": 0.5, "0": 0, "01-a": 7.5, "01-b": 12.5, "05-a": 47.5, "05-b": 52.5}
EV, SP = [], []
STATUS = []  # (package, family, kind, note)
_UNUSED_EVCOLS = ["event_key", "package", "unit_code", "schema_family", "plot_type", "plot_code", "event_date", "x_raw", "y_raw", "utm_zone_raw",
          "datum_raw", "elevation_m", "qa_flags"]

# ---------------------------------------------------------------- v4 family (already harmonized in stage 4)
E4 = pd.read_parquet(OUT / "vmi_plot_events.parquet")
S4 = pd.read_parquet(OUT / "vmi_plot_species.parquet")
e = pd.DataFrame({
    "event_key": E4.event_key, "package": E4.package, "unit_code": E4.unit_code, "schema_family": "plots_v4",
    "plot_type": np.where(E4.aa_plot.fillna(False).astype(bool), "AA", "plot"), "plot_code": E4.Plot_Code, "event_date": E4.event_date,
    "x_raw": E4.Field_X, "y_raw": E4.Field_Y, "utm_zone_raw": E4.UTM_Zone, "datum_raw": E4.md_datum, "lon": E4.lon, "lat": E4.lat,
    "bbox_dist_deg": E4.bbox_dist_deg, "freeform_community_name": E4.get("Provisional_Community").where(lambda v: ~v.isin(["??", "?"])) if "Provisional_Community" in E4 else None,
    "cand_code_final": (E4["Classified_Code"].where(E4["Classified_Code"].notna(), E4.get("NVC_Elcode")) if "Classified_Code" in E4 else E4.get("NVC_Elcode")),
    "cand_code_field": E4.get("Primary_Code"),
    "elevation_m": E4.get("Elevation"), "shape_raw": E4.get("Plot_Shape"), "dim_radius": E4.get("Plot_Radius"), "dim_len": E4.get("Plot_Length"),
    "dim_wid": E4.get("Plot_Width"), "gps_error_m": E4.get("GPS_Error"), "qa_flags": E4.qa_flags})
EV.append(e)
cc = {}
for pk in E4.package.unique():
    lk = load(pk, "xCoverClass_lu")
    cc[pk] = cover_lookup(lk.rename(columns={"coverclass": "coverclass"}) if lk is not None else None)
s = pd.DataFrame({
    "event_key": S4.event_key, "package": S4.package, "species_name": S4.species_name, "plants_symbol": S4.get("PLANTS_Symbol"),
    "stratum": S4.stratum, "cover_code": S4.cover_code, "cover_pct": S4.cover_pct, "cover_pct_source": S4.cover_pct_source,
    "qa_flags": S4.qa_flags})
miss = s.cover_pct.isna()
res = [code_to_pct(c, cc.get(p, {})) for p, c in zip(s.package[miss], s.cover_code[miss])]
s.loc[miss, "cover_pct"] = [r[0] for r in res]
s.loc[miss, "cover_pct_source"] = [r[1] for r in res]
s["qa_flags"] = np.where(s.cover_pct.isna(), "no_cover_value;", "")
SP.append(s)


# ---------------------------------------------------------------- v3 tAA / tAAEvents accuracy-assessment points
# (separate tables in PLOTS v3-schema packages; the only place their community calls live)
have = {(r.package) for r in E4[["package"]].drop_duplicates().itertuples()}
for pk in sorted(have):
    aae, aa = load(pk, "tAAEvents"), load(pk, "tAA")
    if aae is None or aae.empty:
        continue
    if "aacode" not in aae or "aaevent" not in aae:
        STATUS.append((pk, "plots_v3_aa", "AA", f"tAAEvents not ingested; unexpected columns {sorted(aae.columns)[:8]}"))
        continue
    unit = pk.split("_")[0]
    md = load(pk, "tProjectMetadata")
    datum = md.datum.iloc[0] if md is not None and len(md) and "datum" in md else None
    nad83 = pd.Series(not (isinstance(datum, str) and "WGS" in datum.upper()), index=aae.index)
    already = set(E4.loc[E4.package == pk, "Plot_Code"].astype(str))
    aae = aae[~aae.aacode.astype(str).isin(already)].copy()
    if aae.empty:
        continue
    if aa is not None:
        aae = aae.merge(aa[[c for c in ("aacode", "elevation", "elevunits") if c in aa]].drop_duplicates("aacode"), on="aacode", how="left")
    lon, lat, fl, dist = to_lonlat(pk, col(aae, "fieldx"), col(aae, "fieldy"), col(aae, "utmzone").astype(str).str.extract(r"(\d+)")[0], nad83)
    elev = pd.to_numeric(col(aae, "elevation"), errors="coerce")
    elev = elev.where(~col(aae, "elevunits").fillna("").str.lower().str.startswith("f"), elev * 0.3048)
    fin = col(aae, "classifiedcode")
    EV.append(ev_frame(pk, "plots_v3_aa", "AA", pd.DataFrame({
        "event_id": aae.aaevent, "unit_code": unit, "plot_code": aae.aacode, "event_date": pd.to_datetime(col(aae, "eventdate"), errors="coerce"),
        "x_raw": col(aae, "fieldx"), "y_raw": col(aae, "fieldy"), "utm_zone_raw": col(aae, "utmzone"), "datum_raw": datum,
        "lon": lon, "lat": lat, "bbox_dist_deg": dist, "cand_code_final": fin, "cand_code_field": col(aae, "primarycode"),
        "elevation_m": elev, "shape_raw": col(aae, "aaplotshape"), "gps_error_m": col(aae, "gpserror"), "qa_flags": fl})))
    STATUS.append((pk, "plots_v3_aa", "AA", f"{len(aae)} AA points from tAA/tAAEvents added (no species lists for these)"))

# ---------------------------------------------------------------- utah (NCPN tbl*) family
v4pk = set(E4.package)
utah = sorted(set(prof[prof.file.str.contains(r"tblVegetationDetails\.csv$|tblAAVegetationDetails\.csv$")].package) - v4pk)
for pk in utah:
    unit = pk.split("_")[0]
    taxa = load(pk, "tblNCPNPlants") if load(pk, "tblNCPNPlants") is not None else load(pk, "tblPlantCodes")
    tmap = {}
    if taxa is None and (t := load(pk, "tlkp_PLANTS")) is not None:
        t = t.rename(columns={"itistsn": "tsn", "sciname": "latinname", "plantssymbol": "plantscode"})
        taxa = t
    if taxa is not None and "tsn" in taxa:
        nm = taxa.get("fullname", taxa.get("latinname", pd.Series(dtype=object)))
        tmap = {t: (n, c) for t, n, c in zip(taxa.tsn, nm.fillna(taxa.get("latinname", nm)), taxa.get("plantscode", taxa.get("speciescode", pd.Series(np.nan, index=taxa.index))))}
    lk = cover_lookup(load(pk, "tlkpCover"))
    for kind in ("plot", "AA"):
        pre = "tblAA" if kind == "AA" else "tbl"
        loc = load(pk, "tblAALocation" if kind == "AA" else "tblPlotLocation")
        det = load(pk, "tblAADetails" if kind == "AA" else "tblPlotDetails")
        veg = load(pk, "tblAAVegetation" if kind == "AA" else "tblVegetation")
        vd = load(pk, "tblAAVegetationDetails" if kind == "AA" else "tblVegetationDetails")
        if loc is None or vd is None:
            continue
        codecol = "aacode" if kind == "AA" else "plotcode"
        idcol = "aaid" if kind == "AA" else "plotid"
        ev = loc.copy()
        if det is not None:
            ev = ev.merge(det[[c for c in det.columns if c in (codecol, idcol, "surveydate", "plotshape", "plotlength", "plotwidth", "plotdiam", "plotazimuth")]].drop_duplicates(codecol), on=codecol, how="left")
        ev["event_id"] = ev[codecol]
        nad83 = ~col(ev, "datum").fillna("").str.upper().str.contains("WGS")
        lon, lat, fl, dist = to_lonlat(pk, col(ev, "utmeasting"), col(ev, "utmnorthing"), col(ev, "utmzone").astype(str).str.extract(r"(\d+)")[0], nad83)
        out = ev_frame(pk, "ncpn_tbl", kind, pd.DataFrame({
            "event_id": ev.event_id, "unit_code": unit, "plot_code": ev[codecol], "event_date": pd.to_datetime(col(ev, "surveydate"), errors="coerce"),
            "x_raw": col(ev, "utmeasting"), "y_raw": col(ev, "utmnorthing"), "utm_zone_raw": col(ev, "utmzone"), "datum_raw": col(ev, "datum"),
            "lon": lon, "lat": lat, "bbox_dist_deg": dist, "shape_raw": col(ev, "plotshape"), "dim_len": col(ev, "plotlength"), "dim_wid": col(ev, "plotwidth"),
            "dim_diam": col(ev, "plotdiam"), "azimuth_deg": col(ev, "plotazimuth"), "gps_error_m": col(ev, "utmerror"), "qa_flags": fl}))

        # community assignments (final association codes/names) + provisional field names
        asg = {}  # code -> dict(cand_code_final, cand_name_final)
        def put(c, code=None, name=None):
            if c is None or pd.isna(c):
                return
            d = asg.setdefault(str(c), {})
            if code is not None and pd.notna(code) and str(code).strip().lower() not in ("new", "nan", ""):
                d.setdefault("code", str(code).strip())
            if name is not None and pd.notna(name):
                d.setdefault("name", str(name).strip())
        prov = {}
        if kind == "plot":
            for t in (load(pk, "tblFinalAssociationNamesPlots"),):
                if t is not None:
                    for c, e_, n in zip(col(t, "plotcode", "aacode"), col(t, "elccode"), col(t, "associationname", "vegassocname")):
                        put(c, e_, n)
            t = load(pk, "tblFinalClassification")
            if t is not None:
                for c, e_, n in zip(t.plotcode, col(t, "elcode"), col(t, "finaljcname")):
                    put(c, e_, n)
            t = load(pk, "tbl_Final_Classification")
            if t is not None:
                for c, n in zip(t.plotcode, col(t, "finalnvcname")):
                    put(c, None, n)
            if veg is not None and "plotid" in veg and idcol in ev:
                m = veg.drop(columns=[codecol], errors="ignore").merge(ev[[idcol, codecol]].drop_duplicates(), on=idcol, how="left")
                for c, e_, n, a_, p_ in zip(m[codecol], col(m, "ceglcode"), col(m, "finalnvcname"), col(m, "associations"), col(m, "provcommunityname")):
                    put(c, e_, n if pd.notna(n) else a_)
                    if pd.notna(p_):
                        prov.setdefault(str(c), p_)
        else:
            t = load(pk, "tblFinalAssociationNamesAA")
            if t is not None:
                t = t.assign(_p=(col(t, "vegassocrank").fillna("") != "Primary")).sort_values("_p")
                for c, e_, n in zip(col(t, "aacode", "plotcode", "aaobscode"), col(t, "elccode"), col(t, "vegassocname", "associationname")):
                    put(c, e_, n)
            if veg is not None and idcol in veg and idcol in ev:
                m = veg.drop(columns=[codecol], errors="ignore").merge(ev[[idcol, codecol]].drop_duplicates(), on=idcol, how="left")
                for c, a_, p_ in zip(m[codecol], col(m, "associations"), col(m, "provcommunityname")):
                    put(c, None, a_)
                    if pd.notna(p_):
                        prov.setdefault(str(c), p_)
        out["cand_code_final"] = out.plot_code.map(lambda c: asg.get(str(c), {}).get("code"))
        out["cand_name_final"] = out.plot_code.map(lambda c: asg.get(str(c), {}).get("name"))
        out["freeform_community_name"] = out.plot_code.map(lambda c: prov.get(str(c)))
        EV.append(out)
        if veg is None:
            continue
        # vegetation (per visit) -> details (per species/stratum)
        vk = veg[["vegetationid", idcol]].drop_duplicates() if idcol in veg else None
        if vk is None:
            continue
        if "vegetationid" not in vd and "plotaacode" in vd:
            continue
        if "vegetationid" not in vd and "plotid" in vd and kind == "plot":
            vd = vd.assign(vegetationid=vd.plotid)
            veg = veg.assign(vegetationid=veg.plotid) if "plotid" in veg else veg
        if "vegetationid" not in vd or "vegetationid" not in veg:
            STATUS.append((pk, "ncpn_tbl", kind, f"events loaded; species skipped (vd cols {sorted(vd.columns)[:6]}, veg has vegetationid={'vegetationid' in veg})"))
            continue
        if idcol not in ev:
            STATUS.append((pk, "ncpn_tbl", kind, f"events loaded but no {idcol} link to vegetation; species skipped"))
            continue
        d = vd.drop(columns=[idcol, codecol], errors="ignore").merge(vk, on="vegetationid", how="left").merge(ev[[idcol, codecol]].drop_duplicates(), on=idcol, how="left")
        pct = pd.to_numeric(col(d, "percentcover", "coverpct"), errors="coerce")
        cls = col(d, "coverclass").astype(object)
        res = cls.map(lambda c: code_to_pct(c, lk))
        mid = res.map(lambda r: r[0])
        rng_parsed = res.map(lambda r: r[1] == "class_midpoint_parsed_from_range")
        assumed = mid.isna() & cls.notna() & cls.map(lambda c: str(c) in NCPN_FALLBACK)
        mid = mid.where(~assumed, cls.map(lambda c: NCPN_FALLBACK.get(str(c), np.nan)))
        cp = pct.where(pct.notna(), mid)
        src = np.where(pct.notna(), "reported", np.where(assumed & cp.notna(), "class_midpoint_assumed_ncpn_scheme", np.where(rng_parsed, "class_midpoint_parsed_from_range", np.where(mid.notna(), "class_midpoint", None))))
        tsn = col(d, "tsn")
        names = col(d, "species").where(col(d, "species").notna(), tsn.map(lambda t: tmap.get(t, (np.nan, np.nan))[0]))
        sym = tsn.map(lambda t: tmap.get(t, (np.nan, np.nan))[1])
        SP.append(pd.DataFrame({
            "event_key": pk + "|" + kind + "|" + d[codecol].astype(str), "package": pk, "species_name": names, "plants_symbol": sym, "tsn": tsn,
            "stratum": col(d, "stratum"), "cover_code": cls, "cover_pct": cp, "cover_pct_source": src,
            "qa_flags": np.where(cp.isna(), "no_cover_value;", "")}))

# ---------------------------------------------------------------- NER 'Plots'/'Plots-Species' family
ner = sorted(set(prof[prof.file.str.contains(r"_Plots-Species\.csv$")].package) - v4pk - set(utah))
for pk in ner:
    unit = pk.split("_")[0]
    for kind, evt, spt, codecols in (("plot", "Plots", "Plots-Species", ("plotcode",)), ("AA", "AA_Observations", "AA-Species", ("aaobscode", "aaobscode"))):
        ev = load(pk, evt)
        sp = load(pk, spt)
        if sp is None and kind == "AA":
            sp = load(pk, "AA_Species")
        if ev is None:
            continue
        code = next((c for c in codecols if c in ev), ev.columns[0])
        ev = ev.copy()
        ev["event_id"] = ev[code]
        def pick(corr, field):
            c, f_ = col(ev, corr), col(ev, field)
            return c.where(pd.to_numeric(c, errors="coerce").notna(), f_)

        xs, ys = pick("correctedutmx", "fieldutmx"), pick("correctedutmy", "fieldutmy")
        lat_c, lon_c = pick("correctedlat", "fieldlat"), pick("correctedlong", "fieldlong")
        use_ll = pd.to_numeric(xs, errors="coerce").isna() & pd.to_numeric(lat_c, errors="coerce").notna()
        X = xs.where(~use_ll, lon_c)
        Y = ys.where(~use_ll, lat_c)
        nad83 = ~col(ev, "gpsdatum").fillna("").str.upper().str.contains("WGS")
        lon, lat, fl, dist = to_lonlat(pk, X, Y, col(ev, "utmzone").astype(str).str.extract(r"(\d+)")[0], nad83)
        EV.append(ev_frame(pk, "ner_plots", kind, pd.DataFrame({
            "event_id": ev.event_id, "unit_code": unit, "plot_code": ev[code], "event_date": pd.to_datetime(col(ev, "surveydate"), errors="coerce"),
            "x_raw": X, "y_raw": Y, "utm_zone_raw": col(ev, "utmzone"), "datum_raw": col(ev, "gpsdatum"),
            "lon": lon, "lat": lat, "bbox_dist_deg": dist, "cand_code_final": col(ev, "nvcelcode"), "cand_name_final": col(ev, "classifiedcommunityname"),
            "freeform_community_name": col(ev, "provisionalcommunityname", "primarycommunityname"), "elevation_m": pd.to_numeric(col(ev, "elevation"), errors="coerce"),
            "shape_raw": col(ev, "plotshape"), "dim_len": col(ev, "xdimension"), "dim_wid": col(ev, "ydimension"),
            "gps_error_m": col(ev, "gpsaccuracy"), "qa_flags": fl})))
        if sp is None:
            continue
        scode = next((c for c in ("plotcode", "aaobscode", "aaobs", "plot") if c in sp), sp.columns[0])
        rc = pd.to_numeric(col(sp, "realcover"), errors="coerce")
        SP.append(pd.DataFrame({
            "event_key": pk + "|" + kind + "|" + sp[scode].astype(str), "package": pk, "species_name": col(sp, "scientificname", "fieldname"),
            "plants_symbol": col(sp, "plantsymbol", "plantssymbol"), "stratum": col(sp, "stratum"), "cover_code": col(sp, "rangecover"),
            "cover_pct": rc, "cover_pct_source": np.where(rc.notna(), "reported", None), "qa_flags": np.where(rc.isna(), "no_cover_value;", "")}))

core_e = pd.concat(EV, ignore_index=True)
core_s = pd.concat(SP, ignore_index=True)
core_e["event_date"] = pd.to_datetime(core_e.event_date, errors="coerce")
bad = core_e.event_date.notna() & ~core_e.event_date.dt.year.between(1975, 2025)  # typos like 1195 or 2105
core_e.loc[bad, "qa_flags"] = core_e.loc[bad, "qa_flags"].fillna("") + ";implausible_date_nulled;"
core_e.loc[bad, "event_date"] = pd.NaT
core_e["qa_flags"] = core_e.qa_flags.fillna("").map(lambda t: ";".join(dict.fromkeys(x for x in t.split(";") if x)))
core_e["year"] = core_e.event_date.dt.year
for c in ("x_raw", "y_raw", "utm_zone_raw", "datum_raw", "shape_raw", "cand_code_final", "cand_code_field", "cand_name_final",
          "freeform_community_name"):
    if c not in core_e:
        core_e[c] = pd.NA
    core_e[c] = core_e[c].astype("string").str.strip()
    core_e.loc[core_e[c] == "", c] = pd.NA

# ---- community assignment against the US NVC catalog ------------------------------------------------
CAT_NVC = pd.read_csv("datasets/vegbank/USNVC_catalog_full_taxonomy.csv", dtype=str).drop_duplicates("ELCODE").set_index("ELCODE")
NAME_COL = {"Association": "Scientific name", "Alliance": "alliance commname", "Group": "group_name",
            "Macrogroup": "macrogroup_name", "Division": "division code and name"}
lvl = CAT_NVC.LEVEL.str.replace(r"^\d", "", regex=True)
cat_name = pd.Series({k: CAT_NVC.at[k, NAME_COL[l]] if l in NAME_COL else CAT_NVC.at[k, "Biome"] for k, l in lvl.items()})


def norm_name(x):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\[provisional\]|\(provisional\)", "", str(x).lower())) if pd.notna(x) else None


assoc_by_name = {}
for k in CAT_NVC.index[lvl == "Association"]:
    for nc in ("Scientific name", "Association name"):
        assoc_by_name.setdefault(norm_name(CAT_NVC.at[k, nc]), k)
code = core_e.cand_code_final.copy()
src = pd.Series(np.where(code.notna(), "final_classification", None), index=core_e.index, dtype=object)
fld = code.isna() & core_e.cand_code_field.notna()
code[fld], src[fld] = core_e.cand_code_field[fld], "aa_field_call"
byname = code.isna() & core_e.cand_name_final.notna()
mapped = core_e.cand_name_final[byname].map(lambda n: assoc_by_name.get(norm_name(n)))
code[mapped.index], src[mapped.index] = mapped.where(mapped.notna(), pd.NA), np.where(mapped.notna(), "name_match_to_catalog", None)
core_e["nvc_code"] = code
core_e["nvc_code_source"] = src
core_e["nvc_resolved"] = code.isin(CAT_NVC.index)
core_e["nvc_level"] = code.map(lvl)
core_e["nvc_catalog_name"] = code.map(cat_name)
core_e["nvc_common_name"] = code.map(CAT_NVC["Association name"]).where(core_e.nvc_level.eq("Association"))
for out_c, in_c in (("nvc_division", "division code and name"), ("nvc_macrogroup", "macrogroup_name"), ("nvc_group", "group_name")):
    core_e[out_c] = code.map(CAT_NVC[in_c])
# community_name: the NVC community (catalog name if the code resolves, else the source's own classified name), else NULL
core_e["community_name"] = core_e.nvc_catalog_name.where(core_e.nvc_resolved, core_e.cand_name_final)
core_e["community_name_source"] = np.where(core_e.nvc_resolved, "nvc_catalog", np.where(core_e.cand_name_final.notna(), "source_classified_name", None))
core_e.loc[core_e.nvc_code_source.eq("aa_field_call"), "community_name_source"] = np.where(core_e.nvc_resolved, "nvc_catalog(aa_field_call)", None)[core_e.nvc_code_source.eq("aa_field_call")]
core_e["freeform_community_name"] = core_e.freeform_community_name


# ---- plot footprint descriptors (area / shape); NOT a georeferenced footprint -----------------------------
def numc(c):
    """Numeric value of a raw dimension column; tolerates '20 m' style text. Raw column is replaced by the numeric version."""
    if c not in core_e:
        return pd.Series(np.nan, index=core_e.index)
    v = pd.to_numeric(core_e[c].astype("string").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce").astype("float64")
    core_e[c] = v
    return v


rad, dia, ln, wd = numc("dim_radius"), numc("dim_diam"), numc("dim_len"), numc("dim_wid")
az = numc("azimuth_deg")
sh_l = core_e.shape_raw.fillna("").str.lower()
shape = pd.Series("unknown", index=core_e.index)
shape[sh_l.str.contains("rect|quickplot|belt|recatng")] = "rectangle"
shape[sh_l.str.contains("square")] = "square"
shape[sh_l.str.contains("circ")] = "circle"
shape[sh_l.str.contains("transect|linear|lineear|strip")] = "transect"
shape[sh_l.str.contains("obs")] = "point"
shape[(shape == "unknown") & (sh_l != "") & ~sh_l.isin(["na", "#n/a", "other", "estimate"])] = "other"
shape[sh_l.str.contains("irregular|oval|triangle|crescent|banana|boot|amoeba|pac man|semi|half|shaped")] = "other"
dim_bad = (rad > 100) | (dia > 200) | (ln > 200) | (wd > 200)  # typos such as 812568; real plots are < ~1 ha here
rad, dia, ln, wd = (x.where((x > 0) & ~dim_bad) for x in (rad, dia, ln, wd))
area = pd.Series(np.nan, index=core_e.index)
basis = pd.Series("none", index=core_e.index, dtype=object)
use = rad.notna() & shape.isin(["circle", "unknown"])
area[use], basis[use] = np.pi * rad[use] ** 2, "radius"
use = area.isna() & dia.notna() & shape.isin(["circle", "unknown"])
area[use], basis[use] = np.pi * (dia[use] / 2) ** 2, "diameter"
use = area.isna() & ln.notna() & wd.notna() & ~shape.isin(["circle", "point"])
area[use], basis[use] = ln[use] * wd[use], "length_x_width"
use = area.isna() & shape.isin(["square"]) & (ln.notna() | wd.notna())
area[use], basis[use] = (ln.fillna(wd)[use]) ** 2, "side"
use = area.isna() & rad.notna()  # shape missing/other but a radius exists
area[use], basis[use] = np.pi * rad[use] ** 2, "radius"
core_e["plot_shape"] = shape
core_e["plot_area_m2"] = area
core_e["plot_area_basis"] = basis
core_e["plot_equiv_radius_m"] = np.sqrt(area / np.pi)
core_e["plot_azimuth_deg"] = az.where((az > 0) & (az <= 360))  # 0 / 'n/s' are placeholders in the source
core_e["plot_dims_flag"] = np.select([dim_bad, area.isna()], ["implausible_dimension_ignored", "no_usable_dimensions"], default="ok")
core_e["gps_error_m"] = pd.to_numeric(core_e.gps_error_m, errors="coerce").where(lambda v: (v > 0) & (v < 1000))

# ---- ALS-matching columns ------------------------------------------------------------------------------
core_e = core_e.rename(columns={"lon": "lon_wgs84", "lat": "lat_wgs84"})
f_ = core_e.qa_flags.fillna("")
core_e["location_accuracy_flag"] = np.select(
    [core_e.lon_wgs84.isna(), f_.str.contains("outside_park_bbox"),
     f_.str.contains("utm_zone_inferred|utm_zone_corrected|utm_xy_swapped|latlon_swapped|lon_sign_assumed")],
    ["missing", "suspect_outside_park", "repaired"], default="as_recorded")
first = ["event_key", "package", "unit_code", "plot_type", "plot_code", "community_name", "freeform_community_name", "year", "lat_wgs84",
         "lon_wgs84", "location_accuracy_flag", "nvc_code", "nvc_code_source", "nvc_resolved", "nvc_level", "nvc_common_name", "nvc_macrogroup", "nvc_group", "nvc_division",
         "plot_shape", "plot_area_m2", "plot_equiv_radius_m", "plot_area_basis", "plot_dims_flag", "gps_error_m"]
core_e = core_e[first + [c for c in core_e.columns if c not in first and not c.startswith("cand_")] + ["cand_code_final", "cand_code_field", "cand_name_final"]]
for c in ("species_name", "plants_symbol", "stratum", "cover_code", "tsn", "cover_pct_source"):
    if c in core_s:
        core_s[c] = core_s[c].astype("string")
core_s["cover_pct"] = pd.to_numeric(core_s.cover_pct, errors="coerce")
orph = ~core_s.event_key.isin(core_e.event_key)
for pk, n in core_s[orph].groupby("package").size().items():
    STATUS.append((pk, "all", "species", f"{n} species rows dropped: their event is not in this package (belongs to another park's package)"))
core_s = core_s[~orph]
core_e.to_parquet(OUT / "vmi_core_events.parquet", index=False)
core_s.to_parquet(OUT / "vmi_core_species.parquet", index=False)
pd.DataFrame(STATUS, columns=["package", "family", "kind", "note"]).to_csv(Path(__file__).parent / "catalog" / "core_build_notes.csv", index=False)
print(STATUS)
print(core_e.groupby(["schema_family", "plot_type"]).agg(packages=("package", "nunique"), events=("event_key", "size"), with_lonlat=("lon_wgs84", "count")))
print(core_s.groupby(core_s.event_key.str.split("|").str[0].map(dict(zip(core_e.package, core_e.schema_family))) ).agg(rows=("cover_pct", "size"), cover=("cover_pct", "count")))

