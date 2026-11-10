"""BBS route-level presence/absence table for the embedding probes (one row per route = one lidar cookie).

BBS_train_test.parquet has one row per route and survey year (stop 1 counts, surveys within +/-3 years
of the route's 3DEP product); every route has one location, one product and one split. A species is
present on a route if it was counted at stop 1 in any of those years (as the Snapshot USA location
occupancy pools deployments). Only species present on at least one route are kept; the probe task
keeps those with >= 20 presences and absences in both splits.

    python data_cleaning/bbs/route_presence.py   # -> datasets/BBS/BBS_route_presence.parquet
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "datasets/BBS/BBS_train_test.parquet"
OUT = REPO / "datasets/BBS/BBS_route_presence.parquet"
META = ["RouteID", "RouteDataID", "Year", "Latitude", "Longitude", "product_name_AWS", "collection_year_AWS",
        "test_split"]

d = pd.read_parquet(SRC)
species = [c for c in d.columns if c not in META]
site = d.groupby("RouteID")[["Latitude", "Longitude", "product_name_AWS", "collection_year_AWS", "test_split"]]
assert (site.nunique() <= 1).all().all(), "a route has more than one location, product or split"
routes = site.first()
routes["n_years"] = d.groupby("RouteID").Year.nunique()
routes["first_year"] = d.groupby("RouteID").Year.min()
routes["last_year"] = d.groupby("RouteID").Year.max()
pres = (d.groupby("RouteID")[species].max() > 0).astype("int8")
pres = pres.loc[:, pres.sum() > 0]
pres.columns = [f"pres__{c}" for c in pres.columns]
out = routes.join(pres).reset_index()
out.to_parquet(OUT, index=False)

n_tr = pres[~routes.test_split].sum()
n_te = pres[routes.test_split].sum()
n_ok = ((n_tr >= 20) & (n_te >= 20) & ((~routes.test_split).sum() - n_tr >= 20)
        & (routes.test_split.sum() - n_te >= 20)).sum()
print(f"{len(d)} surveys -> {len(out)} routes ({(~routes.test_split).sum()} train / {routes.test_split.sum()} test), "
      f"{pres.shape[1]} species present on >= 1 route, {n_ok} with >= 20 presences and absences in both splits "
      f"-> {OUT.relative_to(REPO)}")
