"""Health report for the evaluation tables in datasets/registry.yaml. Prints warnings; never fails a build.

    pixi run python scripts/validate_datasets.py            # all tables
    pixi run python scripts/validate_datasets.py lfrdb nlcd

Per table: file and columns present, coordinates (range, region, swapped lat/lon), CRS status recorded in the
registry, null ids, split column (both sides present, test share), lidar columns (year gap <= 3, cookie id).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als import crs, registry  # noqa: E402


def check(name):
    spec, warn = registry.get(name), []
    path = ROOT / spec["path"]
    if not path.exists():
        return [f"file missing: {spec['path']}"], {}
    df = registry.read(name)
    need = [spec[k] for k in ("lat", "lon", "year", "id") if spec.get(k)] + [spec["split"]["col"]]
    miss = [c for c in need if c not in df]
    if miss:
        return [f"columns missing: {miss}"], {}
    info = {"rows": len(df)}
    c = crs.check_lonlat(df[spec["lat"]], df[spec["lon"]]).set_index("check").n_rows
    for k in c.index[1:]:
        if c[k] and not k.startswith("  "):
            warn.append(f"coordinates: {c[k]} rows '{k}'")
    if spec.get("crs_status") != "verified":
        warn.append("source CRS not verified (registry crs_status)")
    if spec.get("id") and (df[spec["id"]].isna().any()):
        warn.append(f"{int(df[spec['id']].isna().sum())} null ids")
    if spec.get("year") and df[spec["year"]].isna().any():
        warn.append(f"{int(df[spec['year']].isna().sum())} null years")
    t = registry.is_test(df, spec)
    info["test_share"] = round(float(t.mean()), 3)
    if t.nunique() < 2:
        warn.append("split has only one side")
    elif abs(t.mean() - 0.5) > 0.15:
        warn.append(f"test share {t.mean():.2f} is far from 0.5")
    for col in ("product_name_AWS", "collection_year_AWS", "year_diff_AWS", spec["als_id"]):
        if col not in df and not (col == "year_diff_AWS" and "year_diff_AWS_max" in df):
            warn.append(f"no {col} column")
    gap = df.get("year_diff_AWS", df.get("year_diff_AWS_max"))
    if gap is not None and (gap.max() > 3):
        warn.append(f"year_diff_AWS up to {int(gap.max())} (protocol: <= 3; document if intended)")
    if spec["als_id"] in df and df[spec["als_id"]].isna().any():
        warn.append(f"{int(df[spec['als_id']].isna().sum())} rows without {spec['als_id']}")
    return warn, info


if __name__ == "__main__":
    names = sys.argv[1:] or list(registry.load_registry())
    for n in names:
        w, info = check(n)
        print(f"{n:14s} {info}" + ("  OK" if not w else ""))
        for m in w:
            print(f"    WARN {m}")
