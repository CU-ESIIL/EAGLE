"""Expand the USNVC catalog so every row carries its full hierarchy.

In the raw catalog each row only fills the columns for its own LEVEL (e.g. a
Division row has Division columns but blank Biome/Subbiome/Ecobiome). Rows are
ordered depth-first, so we fill down, but whenever we hit a row at level k we
clear every level deeper than k so specific names never leak into more general
rows. Levels can be skipped (e.g. associations with parent "A-" have no
alliance); those stay blank.

The result is checked against the `parent elcode` column wherever it points to
a real ELCODE.

Usage: python usnvc_full_taxonomy.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "datasets/raw/vegbank/USNVC_catalog.csv"
OUT_PATH = ROOT / "datasets/vegbank/USNVC_catalog_full_taxonomy.csv"

# level number -> catalog columns belonging to that level (after stripping whitespace)
LEVEL_COLS = {
    1: ["Biome"],
    2: ["Subbiome"],
    3: ["Ecobiome"],
    4: ["Division", "division code and name"],
    5: ["Macrogroup", "macrogroup_name"],
    6: ["Group", "group_name"],
    7: ["Alliance", "alliance commname"],
    8: ["Association", "Association name"],
}
LEVEL_NAMES = {1: "Biome", 2: "Subbiome", 3: "Ecobiome", 4: "Division",
               5: "Macrogroup", 6: "Group", 7: "Alliance", 8: "Association"}


def build_full_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [" ".join(c.split()) for c in df.columns]  # tidy whitespace/newlines
    levels = df["LEVEL"].str[0].astype(int)

    # current lineage: level -> (elcode, {col: value})
    lineage: dict[int, tuple[str, dict]] = {}
    filled = {f"{LEVEL_NAMES[k]}_elcode": [] for k in LEVEL_COLS}
    filled.update({c: [] for cols in LEVEL_COLS.values() for c in cols})

    for lvl, (_, row) in zip(levels, df.iterrows()):
        for k in [k for k in lineage if k >= lvl]:
            del lineage[k]
        lineage[lvl] = (row["ELCODE"], {c: row[c] for c in LEVEL_COLS[lvl]})
        for k, cols in LEVEL_COLS.items():
            code, vals = lineage.get(k, (None, {}))
            filled[f"{LEVEL_NAMES[k]}_elcode"].append(code)
            for c in cols:
                filled[c].append(vals.get(c))

    for c, v in filled.items():
        df[c] = v
    df["level_num"] = levels

    # sanity check: the filled-in ancestor at level (k-1 or higher) must equal parent elcode
    code_cols = [f"{LEVEL_NAMES[k]}_elcode" for k in LEVEL_COLS]
    valid_parent = df["parent elcode"].isin(df["ELCODE"])
    for i in df.index[valid_parent]:
        ancestors = df.loc[i, code_cols[: df.at[i, "level_num"] - 1]].dropna()
        assert ancestors.iloc[-1] == df.at[i, "parent elcode"], f"lineage mismatch at row {i}"

    return df


if __name__ == "__main__":
    raw = pd.read_csv(IN_PATH)
    full = build_full_taxonomy(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(full)} rows to {OUT_PATH}")
