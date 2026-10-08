"""Step 4: combine dates, provider statements and MODIS phenology into leaf-on fields on the AWS registry.

Adds to datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson (existing columns are untouched except that
null collect_start / collect_end are filled from step 1):
  leaf_on                  bool or null. Provider statement if there is one, else phenology (>= 50% of
                           the collection days leaf-on), else null (no dates / no usable NDVI).
  leaf_on_source           'provider_metadata' | 'modis_ndvi_phenology' | null
  leaf_on_frac             0-1, share of collection days (over sampled points) with the canopy leaf-on
                           according to MODIS NDVI; null if not computed. Always filled when possible,
                           even when the provider statement decides leaf_on.
  leaf_on_provider         'on' | 'off' | null  (what the provider documents say)
  leaf_on_note             short evidence snippet from the provider documents

Phenology rule per sample point and year: 16-day NDVI -> daily by linear interpolation; a day is
leaf-on when NDVI >= min + 0.5 * (max - min) of that year's curve (White et al. 1997 style threshold).
Curves with amplitude < 0.15 are non-seasonal: leaf-on when max NDVI >= 0.3 (evergreen / always green),
otherwise dropped (bare / barren). Prints a validation of the phenology rule against provider labels.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.signal import medfilt

REPO = Path(__file__).resolve().parents[4]
D = REPO / "datasets/USGS_3dep/leaf_on"
AWS = REPO / "datasets/USGS_3dep/usgs_3dep_resources_AWS.geojson"
THRESH, MIN_AMP, MIN_GREEN, MIN_VALID = 0.5, 0.15, 0.3, 15


def leaf_on_days(doy, ndvi, n_days):
    """Boolean array (n_days) of leaf-on state for one point-year, NaN where not classifiable."""
    ok = ~np.isnan(ndvi)
    if ok.sum() < MIN_VALID:
        return None
    day = np.arange(1, n_days + 1)
    curve = np.interp(day, doy[ok], ndvi[ok])
    curve = medfilt(curve, 15)  # ~half a composite; removes single-composite cloud/snow dips
    lo, hi = curve.min(), curve.max()
    if hi - lo < MIN_AMP:
        return np.full(n_days, hi >= MIN_GREEN) if hi >= MIN_GREEN else None
    return curve >= lo + THRESH * (hi - lo)


def phenology_fraction(dates, samples):
    """dates: DataFrame[name, collect_start, collect_end]; samples: ndvi parquet. -> Series name -> frac."""
    groups = {k: g for k, g in samples.groupby(["name", "pt", "year"])}
    out = {}
    for r in dates.itertuples():
        on = total = 0
        for (name, pt, year), g in ((k, g) for k, g in groups.items() if k[0] == r.name):
            n_days = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
            s = max(r.collect_start, pd.Timestamp(year=year, month=1, day=1)).dayofyear
            e = min(r.collect_end, pd.Timestamp(year=year, month=12, day=31)).dayofyear
            if e < s:
                continue
            state = leaf_on_days(g["doy"].values, g["ndvi"].values, n_days)
            if state is None:
                continue
            on += state[s - 1 : e].sum()
            total += e - s + 1
        out[r.name] = on / total if total else np.nan
    return pd.Series(out)


def main():
    aws = gpd.read_file(AWS)
    dates = pd.read_csv(D / "collect_dates_final.csv", parse_dates=["collect_start", "collect_end"]).set_index("name")
    ev = pd.read_csv(D / "provider_evidence.csv").set_index("link")
    samples = pd.read_parquet(D / "modis_ndvi_samples.parquet")

    # 2. provider statements: a product may map to several metadata links
    prov, note = {}, {}
    for name, links in dates["metadata_links"].dropna().items():
        rows = ev.reindex(links.split("|")).dropna(subset=["provider_leaf"])
        labels = set(rows["provider_leaf"]) - {""}
        if labels == {"off"} or labels == {"on"}:
            prov[name] = labels.pop()
            note[name] = rows.loc[rows["provider_leaf"] == prov[name], "evidence"].iloc[0]
    aws["leaf_on_provider"] = aws["name"].map(prov)
    aws["leaf_on_note"] = aws["name"].map(note)

    # 3. phenology
    dd = aws.loc[aws.collect_start.notna(), ["name", "collect_start", "collect_end"]]
    frac = phenology_fraction(dd, samples)
    aws["leaf_on_frac"] = aws["name"].map(frac).round(3)

    # 4. combine
    prov_bool = aws["leaf_on_provider"].map({"on": True, "off": False})
    pheno_bool = aws["leaf_on_frac"].map(lambda f: np.nan if pd.isna(f) else f >= 0.5)
    leaf_on = prov_bool.where(prov_bool.notna(), pheno_bool)
    aws["leaf_on"] = leaf_on.astype("boolean")
    aws["leaf_on_source"] = np.where(
        prov_bool.notna(), "provider_metadata", np.where(pheno_bool.notna(), "modis_ndvi_phenology", None)
    )

    # validation of the phenology rule against provider labels
    v = aws[prov_bool.notna() & pheno_bool.notna()]
    agree = (v["leaf_on_provider"].map({"on": True, "off": False}) == v["leaf_on_frac"].ge(0.5)).mean()
    print(f"phenology vs provider: n={len(v)}, agreement={agree:.3f}")
    print(pd.crosstab(v["leaf_on_provider"], pd.cut(v["leaf_on_frac"], [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])))
    print(aws["leaf_on_source"].value_counts(dropna=False))
    print(aws["leaf_on"].value_counts(dropna=False))

    aws.to_file(AWS, driver="GeoJSON")


if __name__ == "__main__":
    main()
