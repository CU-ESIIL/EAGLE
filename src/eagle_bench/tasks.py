"""Benchmark task definitions: one loader per task, all returning the same `Task` structure.

A task is a table of labeled locations with a fixed train/test split and one or more targets:

    kind="multiclass"   one label column ("label"); strings
    kind="multilabel"   one 0/1 column per class (species occurrence)
    kind="regression"   one numeric column per target (NaN = not measured)

`Task.df` has one row per labeled location with `lat`, `lon`, `year` (survey year), `emb_year` (year of
annual imagery to use: `year` clipped to the range given by `clip_years`, default 2017-2025 for
AlphaEarth), `is_test` and, where the lidar match exists, `als_site_id`. `Task.y` is aligned to `df`.

Every split is the one stored with the dataset (spatial hold-out); nothing here re-splits.
Add a task by writing a loader and decorating it with `@task("<name>")`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "datasets"

# A class (or species) is scored only if it has enough rows in both parts of the split; this follows the
# "-20" convention in data_cleaning/DATASET_INGESTION.md (classes with >= 20 rows in each of train and test).
MIN_TRAIN = 20
MIN_TEST = 20

TASKS = {}


@dataclass
class Task:
    name: str
    kind: str  # multiclass | multilabel | regression
    df: pd.DataFrame  # lat, lon, year, emb_year, is_test, als_site_id (if known)
    y: pd.DataFrame
    description: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self):
        return int((~self.df["is_test"]).sum())

    @property
    def n_test(self):
        return int(self.df["is_test"].sum())


def task(name):
    def deco(fn):
        TASKS[name] = fn
        return fn

    return deco


def load_task(name, clip_years=(2017, 2025)):
    t = TASKS[name]()
    lo, hi = clip_years
    t.df["emb_year"] = t.df["year"].round().clip(lo, hi).astype(int)
    t.meta["frac_year_clipped"] = float((t.df["year"].round() != t.df["emb_year"]).mean())
    t.df = t.df.reset_index(drop=True)
    t.y = t.y.reset_index(drop=True)
    return t


def _frame(df, lat, lon, year, is_test, als="als_site_id", extra=()):
    out = pd.DataFrame({
        "lat": df[lat].to_numpy(float),
        "lon": df[lon].to_numpy(float),
        "year": df[year].to_numpy(float),
        "is_test": np.asarray(is_test, dtype=bool),
    })
    if als in df:
        out["als_site_id"] = df[als].to_numpy()
    for c in extra:
        out[c] = df[c].to_numpy()
    return out


def _multiclass(name, df, label_col, lat, lon, year, is_test, description):
    """Build a multiclass Task from rows with a label; `is_test` is a boolean Series aligned to `df`."""
    keep = df[label_col].notna()
    df, is_test = df[keep], is_test[keep]
    return Task(name, "multiclass", _frame(df, lat, lon, year, is_test.to_numpy()),
                pd.DataFrame({"label": df[label_col].astype(str).to_numpy()}), description)


@task("nlcd")
def _nlcd():
    df = pd.read_parquet(DATASETS / "NLCD_eval/nlcd_lidar_eval.parquet")
    df = df[df["balanced_subset"]]
    return _multiclass("nlcd", df, "nlcd_class", "lat", "lon", "year", df["test_split"],
                       "NLCD 2021 reference land cover (Level II), class-balanced subset")


@task("lfrdb_ecosys")
def _lfrdb_ecosys():
    df = pd.read_parquet(DATASETS / "LFRDB_eval/lfrdb_eval.parquet")
    return _multiclass("lfrdb_ecosys", df, "ecosys", "lat", "lon", "year", df["test_split"],
                       "LANDFIRE reference database: ecological system")


@task("lfrdb_nvc_group")
def _lfrdb_group():
    df = pd.read_parquet(DATASETS / "LFRDB_eval/lfrdb_eval.parquet")
    return _multiclass("lfrdb_nvc_group", df, "nvc_group", "lat", "lon", "year", df["test_split"],
                       "LANDFIRE reference database: NVC group")


def _nps(level):
    df = pd.read_parquet(DATASETS / "NPS_VMI/vmi_classification_task.parquet")
    df = df[df["product_name_AWS"].notna()]
    return _multiclass(f"nps_vmi_{level}", df, f"nvc_{level}", "latitude", "longitude", "year", df["test_split"],
                       f"NPS Vegetation Mapping Inventory plots: NVC {level}")


task("nps_vmi_macrogroup")(lambda: _nps("macrogroup"))
task("nps_vmi_group")(lambda: _nps("group"))


def _vegbench(level):
    df = pd.read_parquet(DATASETS / "VegBench_eval/vegbench_eval.parquet")
    df = df[df["split"].isin(["train", "test"])]  # the validation split is not used
    return _multiclass(f"vegbench_{level}", df, f"survey_{level}", "lat", "lon", "year", df["split"] == "test",
                       f"VegBench survey plots: USNVC {level} (published train/test split; val unused)")


for _lvl in ("formation", "division", "macrogroup", "group"):
    task(f"vegbench_{_lvl}")(lambda _lvl=_lvl: _vegbench(_lvl))


@task("bbs_occurrence")
def _bbs():
    df = pd.read_parquet(DATASETS / "BBS/BBS_train_test_als_available.parquet")
    species = [c for c in df.columns[7:] if c not in ("test_split", "als_site_id", "als_n_points")]
    y = (df[species] > 0).astype(int)
    return Task("bbs_occurrence", "multilabel", _frame(df, "Latitude", "Longitude", "Year", df["test_split"]), y,
                "Breeding Bird Survey route-years: species occurrence (count > 0)")


@task("butterflies_occurrence")
def _butterflies():
    df = pd.read_parquet(DATASETS / "butterflies/species_observations_als_available.parquet")
    df = df[df["count"] > 0]
    sites = df.groupby("EAGLE_site_index").agg(
        lat=("lat", "first"), lon=("lon", "first"), year=("year", "mean"), test_split=("test_split", "first"),
        als_site_id=("als_site_id", "first"))
    y = pd.crosstab(df["EAGLE_site_index"], df["Reported species name"]).reindex(sites.index).gt(0).astype(int)
    sites = sites.reset_index(drop=True)
    return Task("butterflies_occurrence", "multilabel", _frame(sites, "lat", "lon", "year", sites["test_split"]),
                y.reset_index(drop=True),
                "Butterfly count sites: species occurrence over all surveys (site-level; year = mean survey year)")


@task("snapshot_usa_occurrence")
def _snapshot():
    df = pd.read_parquet(DATASETS / "SNAPSHOT_USA_eval-20/location_occupancy.parquet")
    df["year"] = (df["first_year"] + df["last_year"]) / 2
    cols = [c for c in df.columns if c.startswith("occ__")]
    return Task("snapshot_usa_occurrence", "multilabel", _frame(df, "lat", "lon", "year", df["test_split"]),
                df[cols].astype(int).rename(columns=lambda c: c[5:]),
                "SnapShot USA camera-trap locations: species occupancy (SNAPSHOT_USA_eval-20 species set)")


def _satbird(folder, name, min_pos):
    df = pd.read_parquet(DATASETS / folder / "location_occupancy.parquet")
    cols = [c for c in df.columns if c.startswith("occ__")]
    t = Task(name, "multilabel", _frame(df, "lat", "lon", "year", df["test_split"]),
             df[cols].astype(np.int8).rename(columns=lambda c: c[5:]),
             f"SatBird eBird hotspots (summer): species occupancy, {len(cols)} species ({folder})")
    t.meta["min_pos"] = min_pos
    return t


# year = 2022 (the paired Sentinel-2 year); the labels pool 2010-2023 checklists, so the year is a reference only
task("satbird_occupancy")(lambda: _satbird("SATBIRD_eval", "satbird_occupancy", (1, 1)))
task("satbird_occupancy_20")(lambda: _satbird("SATBIRD_eval-20", "satbird_occupancy_20", (20, 20)))


BLM_TARGETS = ["TotalFoliarCover", "BareSoilCover", "AH_ShrubCover", "AH_SagebrushCover", "AH_GrassCover",
               "AH_ForbCover", "FH_TreeCover", "AH_InvasiveCover", "AH_PJCover", "Hgt_Woody_Avg",
               "Hgt_Herbaceous_Avg", "NumSpp_Total"]


@task("blm_aim")
def _blm():
    import geopandas as gpd

    df = pd.DataFrame(gpd.read_file(DATASETS / "BLM_AIM/BLM_AIM_plots.gpkg").drop(columns="geometry"))
    return Task("blm_aim", "regression",
                _frame(df, "Latitude_WGS84", "Longitude_WGS84", "Year", df["test_split"]), df[BLM_TARGETS],
                "BLM AIM rangeland plots: cover (%), heights (cm) and species richness")


OFO_TARGETS = ["basal_area_m2ha", "total_trees"]


@task("ofo_trees")
def _ofo():
    import geopandas as gpd

    df = pd.DataFrame(gpd.read_file(DATASETS / "OFO_trees/plots_w_als.gpkg").drop(columns="geometry"))
    df = df[df["product_name_AWS"].notna()]
    return Task("ofo_trees", "regression", _frame(df, "plot_lat", "plot_lon", "survey_year", df["test_split"]),
                df[OFO_TARGETS], "Open Forest Observatory plots: basal area (m2/ha) and tree count per plot")
