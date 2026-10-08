"""Stage 2: export the LFRDB tables used for the evaluation task from each regional .accdb
to parquet, one folder per region: datasets/raw/LFRDB/<REGION>/tables/<table>.parquet.

Requires `mdb-export` from mdbtools (macOS: `brew install mdbtools`; not in the pixi env).
Everything is kept as text exactly as exported, so coordinate strings keep their original
number of decimals (stage 3 counts them). Types are applied in stage 3.

Tables: dtPoints (coordinates, location method), dtVisits (date, source), dtCommunities
(Ecological System / NVC group labels, dominant species), dtStands (cover and height),
lutdtVisitsSourceID (source datasets, agency, public-access flag).
"""
import io
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets/raw/LFRDB"
TABLES = ["dtPoints", "dtVisits", "dtCommunities", "dtStands", "lutdtVisitsSourceID"]


def main():
    if shutil.which("mdb-export") is None:
        raise SystemExit("mdb-export not found; install mdbtools (brew install mdbtools)")
    for region_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        accdb = next(region_dir.glob("*.accdb"))
        out = region_dir / "tables"
        out.mkdir(exist_ok=True)
        for t in TABLES:
            csv = subprocess.run(["mdb-export", str(accdb), t], capture_output=True, check=True).stdout
            df = pd.read_csv(io.BytesIO(csv), dtype=str, keep_default_na=False, na_values=[""])
            df.to_parquet(out / f"{t}.parquet", index=False)
            print(f"{region_dir.name:3s} {t:20s} {len(df):8d} rows")


if __name__ == "__main__":
    main()
