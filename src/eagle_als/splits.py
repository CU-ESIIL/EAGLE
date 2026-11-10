"""Stable spatial train/test splits.

A block is a `block_deg` x `block_deg` lat/lon cell. Whether a block is test depends only on the block's id and
the seed (a hash), never on which rows a dataset has. So a dataset rebuilt with more or fewer rows keeps every
surviving row on the same side, and two datasets sharing a block put it on the same side. The realized test
fraction is approximate (it is the share of blocks, weighted by whatever falls in them); `split_report` prints it.

Defaults are the protocol (data_cleaning/DATASET_INGESTION.md): changing BLOCK_DEG, TEST_FRACTION or SEED changes
every split, so only do it for a deliberate new benchmark version. Datasets with a published split keep it and do
not use this module.

Repeat visits to one entity (a BBS route, a camera-trap site) must stay together: pass `group` and every row of a
group goes with the block of the group's mean location.
"""

import hashlib

import numpy as np
import pandas as pd

BLOCK_DEG = 0.25  # ~28 km. 1 degree gave 0.29-0.76 test shares on clustered datasets (NPS, LFRDB); 0.25 stays within 0.07 of 0.5
TOLERANCE = 0.05
TEST_FRACTION = 0.5
SEED = 20261029


def _u01(key):
    """Deterministic uniform [0, 1) from a string (not Python's per-process salted hash())."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") / 2.0**64


def block_ids(lat, lon, block_deg=BLOCK_DEG):
    i = np.floor(np.asarray(lat, dtype=float) / block_deg).astype(int)
    j = np.floor(np.asarray(lon, dtype=float) / block_deg).astype(int)
    return pd.Series([f"{a}_{b}" for a, b in zip(i, j)])


def spatial_split(df, lat_col="lat", lon_col="lon", group=None, block_deg=BLOCK_DEG, test_fraction=TEST_FRACTION,
                  seed=SEED):
    """Return (is_test, block) as Series aligned to df.index. `group` = column name keeping entities together."""
    lat, lon = df[lat_col].to_numpy(float), df[lon_col].to_numpy(float)
    if group is not None:
        g = df[group]
        lat = g.map(df.groupby(group)[lat_col].mean()).to_numpy(float)
        lon = g.map(df.groupby(group)[lon_col].mean()).to_numpy(float)
    block = block_ids(lat, lon, block_deg)
    u = block.map(lambda b: _u01(f"{seed}:{block_deg}:{b}"))
    is_test = pd.Series((u < test_fraction).to_numpy(), index=df.index)
    if abs(is_test.mean() - test_fraction) > TOLERANCE:
        print(f"WARNING spatial_split: test share {is_test.mean():.2f} vs target {test_fraction}; few or very uneven "
              f"blocks. Try a smaller block_deg or another seed, and record the choice in the README.")
    return is_test, pd.Series(block.to_numpy(), index=df.index)


def split_report(is_test, label=None):
    """Rows and share per split; with `label` (Series), classes per split. Print, and write to *_split_counts.csv."""
    is_test = pd.Series(np.asarray(is_test, dtype=bool))
    print(f"test fraction: {is_test.mean():.3f} ({int(is_test.sum())} of {len(is_test)} rows)")
    if label is None:
        return None
    t = pd.crosstab(pd.Series(np.asarray(label)), is_test.map({False: "train", True: "test"}).to_numpy())
    for c in ("train", "test"):
        if c not in t:
            t[c] = 0
    return t[["train", "test"]].sort_values("train", ascending=False).rename_axis("class").reset_index()


def unit_split(df, unit_col, test_fraction=TEST_FRACTION, seed=SEED):
    """Return is_test (Series aligned to df.index) with whole sampling-design units on one side.

    For datasets whose design already defines geographically distinct sampling areas (BBS routes, BLM AIM
    primary sampling units): each unit goes to test if a hash of its id and the seed falls below test_fraction,
    so the assignment never depends on which other units are in the table (stable), and no unit is ever split.
    """
    u = df[unit_col].map(lambda v: _u01(f"{seed}:unit:{v}"))
    is_test = pd.Series((u < test_fraction).to_numpy(), index=df.index)
    if abs(is_test.mean() - test_fraction) > TOLERANCE:
        print(f"WARNING unit_split: test share {is_test.mean():.2f} vs target {test_fraction}; few or very uneven units.")
    return is_test
