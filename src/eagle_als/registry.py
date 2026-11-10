"""Registry of evaluation tables (datasets/registry.yaml): paths and column names in one place.

    from eagle_als import registry
    spec = registry.get("lfrdb")             # dict: path, lat, lon, year, id, split, ...
    df = registry.read("lfrdb")              # the table
    is_test = registry.is_test(df, spec)     # bool Series, whatever the table's split column looks like
    registry.eval_sites()                    # [(path, lat_col, lon_col), ...] for pre-training exclusion
"""

from pathlib import Path

import pandas as pd
import yaml

from eagle_als.sites import read_table

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "datasets/registry.yaml"


def load_registry():
    return yaml.safe_load(REGISTRY_PATH.read_text())


def get(name):
    reg = load_registry()
    if name not in reg:
        raise KeyError(f"{name!r} is not in datasets/registry.yaml (known: {', '.join(reg)})")
    return {"als_id": "als_site_id", **reg[name]}


def read(name, columns=None):
    path = REPO_ROOT / get(name)["path"]
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    df = pd.DataFrame(read_table(path).drop(columns="geometry", errors="ignore"))
    return df if columns is None else df[columns]


def is_test(df, spec):
    col, test = spec["split"]["col"], spec["split"]["test"]
    if isinstance(test, bool):
        return df[col].astype(bool)
    return df[col].isin(test if isinstance(test, list) else [test])


def eval_sites():
    """(path, lat_col, lon_col) for every registered table; the format squares.SquareSampler takes."""
    return [(s["path"], s["lat"], s["lon"]) for s in load_registry().values()]
