"""Step 3: merge per-unit JSON results into the lookup database (parquet tables + meta.json).

  python merge_db.py OUT_DIR DB_DIR [--units units.parquet]

Tables: calibrations (one row per unit x NAIP survey: model, theta, QC), coverage (cell x NAIP survey: items/dates),
units (what was planned, with status done / failed / pending), failures. Re-run any time; it only reads OUT_DIR.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("db_dir")
    ap.add_argument("--units")
    a = ap.parse_args()
    out, db = Path(a.out_dir), Path(a.db_dir)
    db.mkdir(parents=True, exist_ok=True)
    cal, cov, status = [], [], {}
    for f in (out / "done").glob("*.json"):
        r = json.loads(f.read_text())
        u = r["unit"]
        status[u["unit_id"]] = r.get("status", "ok")
        for s in r["naip_available"]:
            cov.append(dict(cell_id=u["cell_id"], survey=s["survey"], items=json.dumps(s["items"]), date=s["items"][0]["date"], gsd=s["items"][0]["gsd"]))
        for x in r["results"]:
            row = dict(unit_id=u["unit_id"], cell_id=u["cell_id"], project=u["project"], project_year=u["project_year"], code_version=r["code_version"],
                       lidar_cover=r.get("lidar_cover"), **x)
            cal.append(row)
    fails = []
    for f in (out / "failed").glob("*.json"):
        r = json.loads(f.read_text())
        status[r["unit"]["unit_id"]] = "failed"
        fails.append(dict(unit_id=r["unit"]["unit_id"], error=r.get("error", "")[-500:]))
    cal_df = pd.DataFrame(cal)
    if len(cal_df):
        cal_df["ok"] = cal_df["ok"].astype(bool)
    cal_df.to_parquet(db / "calibrations.parquet")
    pd.DataFrame(cov).drop_duplicates(["cell_id", "survey"]).to_parquet(db / "coverage.parquet") if cov else pd.DataFrame(columns=["cell_id", "survey", "items"]).to_parquet(db / "coverage.parquet")
    pd.DataFrame(fails).to_parquet(db / "failures.parquet")
    units = pd.read_parquet(a.units) if a.units else pd.DataFrame({"unit_id": list(status)})
    units["status"] = units.unit_id.map(status).fillna("pending")
    units.to_parquet(db / "units.parquet")
    q = cal_df[cal_df.ok].quality.value_counts().to_dict() if len(cal_df) else {}
    meta = dict(cell_m=int(units.cell_m.iloc[0]) if "cell_m" in units and len(units) else 5000, n_units=len(units), status=units.status.value_counts().to_dict(), quality=q,
                code_versions=sorted(cal_df.code_version.unique().tolist()) if len(cal_df) else [])
    (db / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
