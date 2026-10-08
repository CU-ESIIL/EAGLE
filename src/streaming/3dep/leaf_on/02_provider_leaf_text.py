"""Step 2: look for provider-stated leaf-on / leaf-off conditions in the USGS staged metadata.

WESM has no leaf-condition field, but vendor metadata XML and LPC reports on the public
prd-tnm S3 bucket often state it ("All data was collected during leaf-off conditions", "Leaf off"
in a collection-conditions list). For each metadata link from step 1 we list the bucket (no auth),
read up to MAX_DOCS small .xml/.txt files and regex for leaf statements.

Output: datasets/USGS_3dep/leaf_on/provider_evidence.csv, one row per metadata link:
  n_docs_read, n_off, n_on, provider_leaf ('off' | 'on' | 'conflict' | ''), evidence (snippet)
Resumable: links already in the csv are skipped. Run with `python -I` from anywhere.
"""

import concurrent.futures as cf
import re
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[4]
DATES = REPO / "datasets/USGS_3dep/leaf_on/collect_dates.csv"
OUT = REPO / "datasets/USGS_3dep/leaf_on/provider_evidence.csv"
BUCKET = "https://prd-tnm.s3.amazonaws.com/"
MAX_DOCS = 8  # documents read per link
MAX_KEYS = 1500  # listing cap per sub-folder (some projects hold thousands of tile files)

OFF = re.compile(r"leaf[\s_-]*off|leaf[\s_-]*less|leaves[\s_-]*(off|absent)|without leaves|during dormant", re.I)
ON = re.compile(r"leaf[\s_-]*on(?![a-z])|full[\s_-]*leaf|fully leafed", re.I)
DOC_PRIORITY = re.compile(r"classified|lpc|report|project|metadata|\.xml$", re.I)
SKIP = re.compile(r"\.gdb/|tile|index|breakline|lift|calibration|control|ground_survey", re.I)

session = requests.Session()


def get(url, **kw):
    for attempt in range(4):
        try:
            r = session.get(url, timeout=40, **kw)
            if r.status_code < 500:
                return r
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    return None


def list_keys(prefix, delimiter=None):
    keys, token = [], None
    while len(keys) < MAX_KEYS:
        params = {"list-type": 2, "prefix": prefix, "max-keys": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = get(BUCKET, params=params)
        if r is None or r.status_code != 200:
            break
        keys += re.findall(r"<Key>(.*?)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return keys


def scan_link(link):
    prefix = link.split("prefix=")[1].rstrip("/") + "/"
    keys = list_keys(prefix, "/")
    for sub in ("reports/", "spatial_metadata/"):
        keys += list_keys(prefix + sub)
    docs = [k for k in keys if re.search(r"\.(xml|txt)$", k, re.I) and not SKIP.search(k)]
    # prefer one doc per distinct stem so we do not read 8 near-identical per-tile files
    seen, picked = set(), []
    for k in sorted(docs, key=lambda k: (not DOC_PRIORITY.search(k), len(k))):
        stem = re.sub(r"\d+", "#", k.rsplit("/", 1)[-1])
        if stem not in seen:
            seen.add(stem)
            picked.append(k)
        if len(picked) == MAX_DOCS:
            break
    n_off = n_on = 0
    evidence = ""
    for k in picked:
        r = get(BUCKET + k)
        if r is None or r.status_code != 200:
            continue
        t = re.sub(r"<[^>]+>", " ", r.text)
        t = re.sub(r"\s+", " ", t)
        for pat, label in ((OFF, "off"), (ON, "on")):
            m = pat.search(t)
            if m:
                n_off += label == "off"
                n_on += label == "on"
                if not evidence:
                    evidence = f"{k.rsplit('/', 1)[-1]}: ...{t[max(0, m.start() - 90):m.end() + 60]}..."
    if n_off and n_on:
        leaf = "conflict"
    else:
        leaf = "off" if n_off else "on" if n_on else ""
    return dict(link=link, n_docs_read=len(picked), n_off=n_off, n_on=n_on, provider_leaf=leaf, evidence=evidence)


def main():
    links = DATES["metadata_links"].dropna().str.split("|").explode().unique()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["link"])
    todo = [l for l in links if l not in set(done["link"])]
    print(f"{len(links)} links, {len(todo)} to scan")
    rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        for i, row in enumerate(ex.map(scan_link, todo), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(todo):
                pd.concat([done, pd.DataFrame(rows)]).to_csv(OUT, index=False)
                print(f"{i}/{len(todo)}", flush=True)
    res = pd.read_csv(OUT)
    print(res["provider_leaf"].fillna("none").value_counts())


DATES = pd.read_csv(DATES)
if __name__ == "__main__":
    main()
