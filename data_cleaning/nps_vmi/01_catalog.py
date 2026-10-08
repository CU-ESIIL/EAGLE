"""Stage 1: catalog NPS Vegetation Mapping Inventory (VMI) products.

park list page -> park page -> IRMA project ReferenceId(s) -> IRMA REST Profile -> products
-> data-package file lists. Every HTTP access is logged to catalog/access_log.csv; API JSON
is cached in datasets/raw/nps_vmi/api_cache so reruns are cheap.

Outputs (data_cleaning/nps_vmi/catalog/): parks.csv, products.csv, files.csv, access_log.csv
"""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CAT = HERE / "catalog"
CACHE = Path("datasets/raw/nps_vmi/api_cache")
CAT.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
LIST_URL = "https://www.nps.gov/im/vmi-products.htm"
API = "https://irmaservices.nps.gov/datastore/v8/rest/Profile/"
log = []


def get(url, cache_name=None, retries=3):
    """GET with caching + logging. Returns text or None."""
    if cache_name and (CACHE / cache_name).exists():
        return (CACHE / cache_name).read_text()
    status, text, note = None, None, ""
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EAGLE research crawl)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                status, text = r.status, r.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as e:
            status, note = e.code, str(e)
            if e.code in (404, 403, 401):
                break
        except Exception as e:  # timeouts, DNS, etc.
            status, note = None, repr(e)
        time.sleep(2 * (i + 1))
    log.append({"url": url, "status": status, "note": note})
    if text is not None and cache_name:
        (CACHE / cache_name).write_text(text)
    time.sleep(0.3)
    return text


def profile(ref_id):
    t = get(f"{API}{ref_id}", f"profile_{ref_id}.json")
    try:
        return json.loads(t) if t else None
    except json.JSONDecodeError:
        log.append({"url": f"{API}{ref_id}", "status": None, "note": "invalid JSON"})
        return None


# --- park list -------------------------------------------------------------------------
html = get(LIST_URL, "vmi-products.html")
parks = {}
for url, name in re.findall(r'href="(https?://(?:www\.)?nps\.gov/im/vmi-[^"]+\.htm)"[^>]*>([^<]+)', html):
    url = url.replace("http://", "https://").replace("//nps.gov", "//www.nps.gov")
    key = url.lower()
    if key not in parks:
        parks[key] = {"park_page": url, "park_name": re.sub(r"\s+", " ", name).strip()}
print(len(parks), "park pages")

# --- park page -> project ids ----------------------------------------------------------
for p in parks.values():
    slug = p["park_page"].rsplit("/", 1)[1].replace(".htm", "")
    page = get(p["park_page"], f"page_{slug}.html")
    p["page_ok"] = page is not None
    ids = re.findall(r"ReferenceId\s*=\s*(\d+)", page or "")
    ids += re.findall(r"Reference/Profile/(\d+)", page or "")
    p["project_ids"] = sorted(set(ids), key=ids.index)

# --- project profile -> products -> files ----------------------------------------------
prod_rows, file_rows = [], []
for p in parks.values():
    for pid in p["project_ids"]:
        proj = profile(pid)
        if not proj:
            continue
        units = [u["unitCode"] for u in (proj.get("units") or [])]
        for prod in proj.get("products") or []:
            row = {
                "park_name": p["park_name"], "park_page": p["park_page"], "project_id": pid,
                "units": ";".join(units), "product_id": prod["referenceId"],
                "type": prod.get("typeName") or prod.get("referenceType"),
                "title": prod.get("title"), "issued": prod.get("dateOfIssue"),
                "file_access": prod.get("fileAccess"), "n_files_listed": len(prod.get("linkedResources") or []),
            }
            prod_rows.append(row)
            files = prod.get("linkedResources") or []
            # data packages: the project listing may omit files, so fetch the full profile
            if row["type"] in ("Data Package", "Dataset") or "Data Package" in (row["title"] or ""):
                full = profile(prod["referenceId"])
                if full:
                    files = [dict(f, url=f.get("url")) for f in (full.get("filesAndLinks") or [])]
            for f in files:
                file_rows.append({
                    "project_id": pid, "product_id": prod["referenceId"], "product_type": row["type"],
                    "product_title": row["title"], "file_name": f.get("fileName"),
                    "description": f.get("description"), "url": f.get("url"),
                    "ext": (f.get("type") or Path(f.get("fileName") or "").suffix.lstrip(".")).lower(),
                    "size": f.get("fileSize"),
                })

pd.DataFrame(parks.values()).assign(project_ids=lambda d: d.project_ids.map(";".join)).to_csv(CAT / "parks.csv", index=False)
pd.DataFrame(prod_rows).to_csv(CAT / "products.csv", index=False)
pd.DataFrame(file_rows).to_csv(CAT / "files.csv", index=False)
pd.DataFrame(log).to_csv(CAT / "access_log.csv", index=False)
print(len(prod_rows), "products;", len(file_rows), "files;", len(log), "http requests")
