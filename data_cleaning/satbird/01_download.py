"""Stage 1: fetch SatBird labels and metadata (no imagery) from the Hugging Face repo sammlapp/SatBird-labels.

SatBird (Teng et al. 2023, https://arxiv.org/abs/2311.00936) labels are eBird encounter rates per hotspot. The HF repo
repackages the original dataset's non-image files unmodified: per season (`USA_winter`, `USA_summer`) a
`hotspots.parquet` (hotspot_id, lat, lon, SatBird's own `split` train/valid/test, `probs` = encounter rate of every
species, in the order of `species_list.txt`), SatBird's original split CSVs, and the species list. Kenya is not used.
Nothing else (coordinates, split) comes from anywhere else.

Usage (from the repo root):
  pixi run python data_cleaning/satbird/01_download.py                 # runs `hf download` (needs the hf CLI)
  pixi run python data_cleaning/satbird/01_download.py --source DIR    # copy from an existing `hf download` snapshot

Outputs (datasets/raw/satbird/SatBird-labels/<season>/, gitignored), a download log in catalog/download_log.csv and
a SOURCE.md in the raw folder. Seasons missing from the repo (USA_summer until it is released) are skipped.
"""
import argparse
import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets/raw/satbird/SatBird-labels"
CAT = Path(__file__).parent / "catalog"
REPO = "sammlapp/SatBird-labels"
SEASONS = ["USA_winter", "USA_summer"]
FILES = ["hotspots.parquet", "species_list.txt", "train_split.csv", "valid_split.csv", "test_split.csv"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="snapshot folder of an earlier `hf download`; default: download now")
    src = ap.parse_args().source
    if src is None:
        src = RAW / "_hf_download"
        subprocess.run(["hf", "download", REPO, "--repo-type", "dataset", "--local-dir", str(src),
                        *sum([["--include", f"{s}/*"] for s in SEASONS], [])], check=True)
    CAT.mkdir(exist_ok=True)
    rows = []
    for season in SEASONS:
        for name in FILES:
            f = src / season / name
            if not f.exists():
                print(f"skip {season}/{name} (not in the repo)")
                continue
            out = RAW / season / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if f.resolve() != out.resolve():
                shutil.copyfile(f.resolve(), out)
            rows.append(dict(file=f"{season}/{name}", url=f"https://huggingface.co/datasets/{REPO}/resolve/main/{season}/{name}",
                             downloaded_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             bytes=out.stat().st_size, sha256=sha256(out)))
    if src == RAW / "_hf_download":
        shutil.rmtree(src)
    pd.DataFrame(rows).to_csv(CAT / "download_log.csv", index=False)
    (RAW / "SOURCE.md").write_text(
        "# SatBird labels (gitignored)\n\n"
        f"From https://huggingface.co/datasets/{REPO} (CC BY-NC 4.0): the label and metadata files of the original SatBird "
        "dataset (Teng et al. 2023), without imagery. Fetched by `data_cleaning/satbird/01_download.py`; hashes and fetch times in "
        "`data_cleaning/satbird/catalog/download_log.csv`. Use must also comply with the eBird Terms of Use.\n")
    print(pd.DataFrame(rows)[["file", "bytes"]].to_string(index=False))


if __name__ == "__main__":
    main()
