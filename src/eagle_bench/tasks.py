"""Benchmark task definitions: one loader per task, all returning the same `Task` structure.

A task is a table of labeled locations with a fixed train/test split and one or more targets:

    kind="multiclass"   one label column ("label"); strings
    kind="multilabel"   one 0/1 column per class (species occurrence)
    kind="regression"   one numeric column per target (NaN = not measured)

`Task.df` has one row per labeled location with `lat`, `lon`, `year` (survey year), `emb_year` (year of
annual imagery to use: `year` clipped to the range given by `clip_years`, default 2017-2025 for
AlphaEarth), `is_test` and, where the lidar match exists, `als_site_id`. `Task.y` is aligned to `df`.

Every split is the one stored with the dataset (spatial hold-out); nothing here re-splits.
Paths, column names and the split rule come from datasets/registry.yaml (`eagle_als.registry`); a loader only
says which label columns make the task. Add a task by writing a loader and decorating it with `@task("<name>")`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eagle_als import registry

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "datasets"

# Datasets ship every class; how many rows a class needs in each of train and test to be scored is a
# parameter of the evaluation (eagle_bench.probes.evaluate(min_count=...)), not of the dataset.

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


def _table(name, query=None):
    """Registry table `name` with its row filter applied: (df, spec, is_test aligned to df)."""
    spec = registry.get(name)
    df = registry.read(name)
    if query is not None:
        df = df.query(query)
    return df, spec, registry.is_test(df, spec)


def _std(df, spec, is_test, year=None):
    return _frame(df, spec["lat"], spec["lon"], year or spec["year"], is_test, als=spec["als_id"])


def _multiclass(name, table, label_col, description, query=None):
    """Build a multiclass Task from the rows of registry table `table` that have a label."""
    df, spec, is_test = _table(table, query)
    keep = df[label_col].notna()
    df, is_test = df[keep], is_test[keep]
    return Task(name, "multiclass", _std(df, spec, is_test.to_numpy()),
                pd.DataFrame({"label": df[label_col].astype(str).to_numpy()}), description)


@task("nlcd")
def _nlcd():
    return _multiclass("nlcd", "nlcd", "nlcd_class", "NLCD 2021 reference land cover (Level II), class-balanced subset",
                       query="balanced_subset")


@task("lfrdb_ecosys")
def _lfrdb_ecosys():
    return _multiclass("lfrdb_ecosys", "lfrdb", "ecosys", "LANDFIRE reference database: ecological system")


@task("lfrdb_nvc_group")
def _lfrdb_group():
    return _multiclass("lfrdb_nvc_group", "lfrdb", "nvc_group", "LANDFIRE reference database: NVC group")


def _nps(level):
    return _multiclass(f"nps_vmi_{level}", "nps_vmi", f"nvc_{level}", f"NPS Vegetation Mapping Inventory plots: NVC {level}",
                       query="product_name_AWS.notna()")


task("nps_vmi_macrogroup")(lambda: _nps("macrogroup"))
task("nps_vmi_group")(lambda: _nps("group"))


def _vegbench(level):
    return _multiclass(f"vegbench_{level}", "vegbench", f"survey_{level}",
                       f"VegBench survey plots: USNVC {level} (published train/test split; val unused)",
                       query="split in ['train', 'test']")  # the validation split is not used


for _lvl in ("formation", "division", "macrogroup", "group"):
    task(f"vegbench_{_lvl}")(lambda _lvl=_lvl: _vegbench(_lvl))


@task("bbs_occurrence")
def _bbs():
    df, spec, is_test = _table("bbs")
    species = [c for c in df.columns[7:] if c not in ("test_split", "als_site_id", "als_n_points")]
    y = (df[species] > 0).astype(int)
    return Task("bbs_occurrence", "multilabel", _std(df, spec, is_test), y,
                "Breeding Bird Survey route-years: species occurrence (count > 0)")


@task("butterflies_occurrence")
def _butterflies():
    df, spec, _ = _table("butterflies")
    df = df[df["count"] > 0]
    sites = df.groupby("EAGLE_site_index").agg(
        lat=(spec["lat"], "first"), lon=(spec["lon"], "first"), year=(spec["year"], "mean"),
        test_split=("test_split", "first"), als_site_id=(spec["als_id"], "first"))
    y = pd.crosstab(df["EAGLE_site_index"], df["Reported species name"]).reindex(sites.index).gt(0).astype(int)
    sites = sites.reset_index(drop=True)
    return Task("butterflies_occurrence", "multilabel", _frame(sites, "lat", "lon", "year", sites["test_split"]),
                y.reset_index(drop=True),
                "Butterfly count sites: species occurrence over all surveys (site-level; year = mean survey year)")


@task("snapshot_usa_occurrence")
def _snapshot():
    df, spec, is_test = _table("snapshot_usa")
    df["year"] = (df["first_year"] + df["last_year"]) / 2
    cols = [c for c in df.columns if c.startswith("occ__")]
    return Task("snapshot_usa_occurrence", "multilabel", _std(df, spec, is_test, year="year"),
                df[cols].astype(int).rename(columns=lambda c: c[5:]),
                "SnapShot USA camera-trap locations: species occupancy")


# year = 2022 (the Sentinel-2 year of SatBird); the labels pool 2010-2023 checklists, so the year is a reference only
@task("satbird_winter_occupancy")
def _satbird_winter():
    df, spec, is_test = _table("satbird_winter", query="split in ['train', 'test']")  # SatBird's validation split is not used
    cols = [c for c in df.columns if c.startswith("occ__")]
    return Task("satbird_winter_occupancy", "multilabel", _std(df, spec, is_test), df[cols].astype(np.int8).rename(columns=lambda c: c[5:]),
                f"SatBird eBird hotspots (USA winter): species occupancy, {len(cols)} species")


BLM_TARGETS = ["TotalFoliarCover", "BareSoilCover", "AH_ShrubCover", "AH_SagebrushCover", "AH_GrassCover",
               "AH_ForbCover", "FH_TreeCover", "AH_InvasiveCover", "AH_PJCover", "Hgt_Woody_Avg",
               "Hgt_Herbaceous_Avg", "NumSpp_Total"]


@task("blm_aim")
def _blm():
    df, spec, is_test = _table("blm_aim")
    return Task("blm_aim", "regression", _std(df, spec, is_test), df[BLM_TARGETS],
                "BLM AIM rangeland plots: cover (%), heights (cm) and species richness")


OFO_TARGETS = ["basal_area_m2ha", "total_trees"]


@task("ofo_trees")
def _ofo():
    df, spec, is_test = _table("ofo_trees", query="product_name_AWS.notna()")
    return Task("ofo_trees", "regression", _std(df, spec, is_test), df[OFO_TARGETS],
                "Open Forest Observatory plots: basal area (m2/ha) and tree count per plot")
