"""Stage 2: download VMI data-package CSVs listed in catalog/files.csv.

Skips Access system tables (MSys*) and the shared USDA PLANTS lookup (xPLANTS_lu).
Files go to datasets/raw/nps_vmi/<UNIT>_<product_id>/ (gitignored). Results (including
failures) are written to catalog/download_log.csv. Reruns skip files already present.
"""
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

CAT = Path(__file__).parent / "catalog"
RAW = Path("datasets/raw/nps_vmi")
SKIP = ("MSys", "xPLANTS_lu")

files = pd.read_csv(CAT / "files.csv")
prods = pd.read_csv(CAT / "products.csv")[["product_id", "units"]].drop_duplicates("product_id")
files = files[(files.product_type == "Data Package") & (files.ext == "csv")].merge(prods, on="product_id")
files = files[~files.file_name.str.contains("|".join(SKIP), na=False)]
files["dir"] = files.units.str.split(";").str[0] + "_" + files.product_id.astype(str)
print(len(files), "files to fetch")


def fetch(r):
    dest = RAW / r.dir / r.file_name
    if dest.exists() and dest.stat().st_size > 0:
        return r.url, "cached", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        try:
            req = urllib.request.Request(r.url, headers={"User-Agent": "Mozilla/5.0 (EAGLE research crawl)"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return r.url, "ok", dest.stat().st_size
        except Exception as e:
            err = repr(e)[:150]
            time.sleep(2 * (i + 1))
    return r.url, err, 0


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(fetch, files.itertuples()))
log = pd.DataFrame(res, columns=["url", "result", "bytes"]).merge(files[["url", "dir", "file_name"]], on="url")
log.to_csv(CAT / "download_log.csv", index=False)
print(log.result.value_counts().head(), log.bytes.sum() / 1e6, "MB")
