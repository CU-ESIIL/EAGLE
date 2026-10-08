"""Stage 3: profile every downloaded package -> catalog/package_profile.csv (one row per package/table)."""
from pathlib import Path

import pandas as pd

RAW = Path("datasets/raw/nps_vmi")
rows = []
for d in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name != "api_cache"):
    for f in sorted(d.glob("*.csv")):
        try:
            df = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace", low_memory=False)
            n, cols = len(df), ";".join(df.columns)
        except Exception as e:  # empty file, parse error
            n, cols = -1, f"ERR {e!r}"[:100]
        rows.append({"package": d.name, "table": f.stem.split("_", 1)[1] if f.stem.split("_")[0] == d.name.split("_")[0] else f.stem,
                     "file": f.name, "rows": n, "columns": cols})
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / "catalog" / "package_profile.csv", index=False)
print(out.package.nunique(), "packages,", len(out), "tables")
