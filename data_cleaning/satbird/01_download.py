"""Stage 1: fetch SatBird-USA-summer targets (Hugging Face) and eBird hotspot coordinates (eBird API).

SatBird (Teng et al. 2023, https://arxiv.org/abs/2311.00936) labels are eBird encounter rates per
hotspot. The targets pickle is a dict {hotspot_id: float vector of 670 species} for 122,593
hotspots and carries no coordinates. Coordinates come from the eBird API v2 bulk endpoint
`ref/hotspot/<region>` (one call per US state + DC, all hotspots of that region), which needs a
free API key in the environment variable EBIRD_API_KEY (https://ebird.org/api/keygen).

Outputs (datasets/raw/satbird/, gitignored):
  satbird_usa_summer_targets.pkl        from huggingface.co/datasets/cisosdm/SatBird (USA_summer/)
  songbird_indices.npy, nonsongbird_indices.npy
  ebird_hotspots/<region>.json           raw API response per region
Provenance goes to catalog/download_log.csv (URL, UTC time, size, sha256) and a SOURCE.md in the raw folder.
Reruns skip files that exist (delete to refetch). Hotspot coordinates are eBird's current values,
which can differ slightly from those used when SatBird was built (2023).
"""
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets/raw/satbird"
CAT = Path(__file__).parent / "catalog"
HF = "https://huggingface.co/datasets/cisosdm/SatBird/resolve/main/USA_summer"
HF_FILES = {
    "satbird_usa_summer_targets.pkl": f"{HF}/satbird_usa_summer_targets.pkl",
    "songbird_indices.npy": f"{HF}/stats/songbird_indices.npy",
    "nonsongbird_indices.npy": f"{HF}/stats/nonsongbird_indices.npy",
}
STATES = (
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
    "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()
EBIRD = "https://api.ebird.org/v2/ref/hotspot/US-{}?fmt=json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, out, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "EAGLE research download"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out.write_bytes(resp.read())
    return dict(
        file=str(out.relative_to(RAW)), url=url.split("?")[0],
        downloaded_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bytes=out.stat().st_size, sha256=sha256(out),
    )


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "ebird_hotspots").mkdir(exist_ok=True)
    log_path = CAT / "download_log.csv"
    rows = [] if not log_path.exists() else pd.read_csv(log_path).to_dict("records")
    done = {r["file"] for r in rows}
    for name, url in HF_FILES.items():
        out = RAW / name
        if not out.exists():
            rows.append(download(url, out))
        elif name not in done:  # present from a manual download: record hash, source URL
            rows.append(dict(file=name, url=url, downloaded_utc="manual (before logging)",
                             bytes=out.stat().st_size, sha256=sha256(out)))
    key = os.environ.get("EBIRD_API_KEY")
    for st in STATES:
        out = RAW / "ebird_hotspots" / f"US-{st}.json"
        if out.exists():
            continue
        if not key:
            raise SystemExit("set EBIRD_API_KEY (free key: https://ebird.org/api/keygen)")
        rows.append(download(EBIRD.format(st), out, {"X-eBirdApiToken": key}))
        print(st, out.stat().st_size, flush=True)
    log = pd.DataFrame(rows).drop_duplicates("file", keep="last")
    log.to_csv(log_path, index=False)
    (RAW / "SOURCE.md").write_text(
        "# SatBird raw data (gitignored)\n\n"
        "Targets: SatBird-USA-summer encounter rates from https://huggingface.co/datasets/cisosdm/SatBird "
        "(CC BY-NC 4.0, derived from Teng et al. 2023; the full 152 GB set is on Google Drive, see the SatBird repo).\n"
        "Coordinates: eBird API v2 `ref/hotspot/US-<state>` (https://documenter.getpostman.com/view/664302/S1ENwy59), "
        "fetched by `data_cleaning/satbird/01_download.py`.\n"
        "Hashes and fetch times: `data_cleaning/satbird/catalog/download_log.csv`.\n"
    )
    print(log[["file", "bytes", "downloaded_utc"]].tail(8).to_string(index=False), f"\n{len(log)} files logged")


if __name__ == "__main__":
    main()
