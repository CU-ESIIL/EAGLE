"""Registry of evaluation tables (datasets/registry.yaml): paths and column names in one place.

    from eagle_als import registry
    spec = registry.get("lfrdb")             # dict: path, lat, lon, year, id, split, ...
    df = registry.read("lfrdb")              # the table
    is_test = registry.is_test(df, spec)     # bool Series, whatever the table's split column looks like
    registry.eval_sites()                    # [(path, lat_col, lon_col), ...] for pre-training exclusion
    registry.task_radius("vegbench_group")   # analysis disc radius (m) of a benchmark task's table
"""

from pathlib import Path

import pandas as pd
import yaml

from eagle_als.sites import read_table

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "datasets/registry.yaml"
DEFAULT_RADIUS = 50.0  # metres, for tables without a `radius` (documented in registry.yaml)


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


def table_of_task(task):
    """Registry table of a benchmark task (longest table name that prefixes the task: lfrdb_ecosys -> lfrdb)."""
    reg = load_registry()
    table = max((k for k in reg if task.startswith(k)), key=len, default=None)
    if table is None:
        raise KeyError(f"task {task!r} matches no table in datasets/registry.yaml")
    return table


def table_radius(table, default=DEFAULT_RADIUS):
    r = load_registry()[table].get("radius")
    return float(default if r is None else r)


def task_radius(task, default=DEFAULT_RADIUS):
    return table_radius(table_of_task(task), default)
