"""Stage 1: download the public LANDFIRE Reference Database (LFRDB, LF 2.0.0 / LF 2016 Remap)
regional Access databases.

Source page: https://landfire.gov/reference/lfrdb/lfrdb_data
Files: https://landfire.gov/data-downloads/zip/LFRDBs_and_Photos/<REGION>/<REGION>_Public_LFRDB_LF2.0.0.zip
Each zip holds one .accdb and the data dictionary PDF. Files are unzipped to
datasets/raw/LFRDB/<REGION>/ (gitignored). The provenance of every download (URL, UTC
download time, server Last-Modified, size, sha256) is written to catalog/download_log.csv,
which is tracked, and to datasets/raw/LFRDB/SOURCE.md for anyone looking at the raw folder.
Reruns skip regions whose .accdb is already present (delete the folder to refetch).
"""
import hashlib
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets/raw/LFRDB"
CAT = Path(__file__).parent / "catalog"
BASE = "https://landfire.gov/data-downloads/zip/LFRDBs_and_Photos"
# LANDFIRE geographic areas: AK, HI, insular areas, and the six CONUS regions
REGIONS = ["AK", "HI", "IA", "SW", "SC", "SE", "NE", "NC", "NW"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(region):
    url = f"{BASE}/{region}/{region}_Public_LFRDB_LF2.0.0.zip"
    out = RAW / region
    if list(out.glob("*.accdb")):
        return None  # already downloaded; keep the existing log row
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{region}_Public_LFRDB_LF2.0.0.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EAGLE research download)"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        last_modified = resp.headers.get("Last-Modified")
        zpath.write_bytes(resp.read())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
        members = z.namelist()
    row = dict(
        region=region, url=url, downloaded_utc=stamp, server_last_modified=last_modified,
        zip_bytes=zpath.stat().st_size, zip_sha256=sha256(zpath), members=";".join(members),
    )
    zpath.unlink()  # the extracted .accdb is the raw data; the hash above identifies the zip
    # the extracted files may sit in a sub-folder; flatten to datasets/raw/LFRDB/<REGION>/
    for p in out.rglob("*"):
        if p.is_file() and p.parent != out:
            p.rename(out / p.name)
    for d in sorted(out.rglob("*"), reverse=True):
        if d.is_dir():
            d.rmdir()
    return row


def write_source_note(log):
    lines = [
        "# LFRDB raw data (gitignored)",
        "",
        "Public LANDFIRE Reference Database, version LF 2.0.0 (LF 2016 Remap), regional Access databases.",
        "Page: https://landfire.gov/reference/lfrdb/lfrdb_data . Fetched by `data_cleaning/lfrdb/01_download.py`;",
        "the tracked copy of this table is `data_cleaning/lfrdb/catalog/download_log.csv`.",
        "",
        "| region | downloaded (UTC) | server Last-Modified | zip MB | sha256 of zip |",
        "|---|---|---|---|---|",
    ]
    for r in log.itertuples():
        lines.append(
            f"| {r.region} | {r.downloaded_utc} | {r.server_last_modified} | {r.zip_bytes / 1e6:.1f} | `{r.zip_sha256[:16]}...` |"
        )
    (RAW / "SOURCE.md").write_text("\n".join(lines) + "\n")


def main():
    CAT.mkdir(exist_ok=True)
    log_path = CAT / "download_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    for region in REGIONS:
        row = fetch(region)
        if row is not None:
            log = pd.concat([log[log.region != region] if len(log) else log, pd.DataFrame([row])])
            print(f"{region}: downloaded {row['zip_bytes'] / 1e6:.1f} MB")
        else:
            print(f"{region}: already present")
    log = log.sort_values("region").reset_index(drop=True)
    log.to_csv(log_path, index=False)
    write_source_note(log)


if __name__ == "__main__":
    main()
