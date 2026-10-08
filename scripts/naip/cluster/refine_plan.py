"""Step 2b (optional): adaptive refinement of a sparse pass (plan_units.py --stride S).

For every pair of finished cells S cells apart (E-W or N-S) with the same 3DEP project and NAIP survey, both calibrations are
evaluated at the midpoint (ground level). If they differ by more than --thr metres, or either is quality 'poor', the S-1 cells
in between are added to a new unit table (rows copied from the sparse table, new cell coordinates).

  python refine_plan.py units_sparse.parquet OUT_DIR units_refine.parquet --stride 3 --thr 0.7
Run run_shard.py on the new table (same OUT_DIR); repeat with --stride 1 style halving if desired.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from eagle_als.coreg import lookup as LK, pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("out_dir")
    ap.add_argument("new_units")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.7)
    a = ap.parse_args()
    units = pd.read_parquet(a.units)
    cm = int(units.cell_m.iloc[0])
    recs = {}
    for f in (Path(a.out_dir) / "done").glob("*.json"):
        r = json.loads(f.read_text())
        u = r["unit"]
        for res in r["results"]:
            if res.get("ok"):
                recs[(u["epsg"], u["e0"], u["n0"], u["project"], res["survey"])] = res
    flagged = set()
    for (epsg, e0, n0, proj, surv), res in recs.items():
        for de, dn in ((a.stride * cm, 0), (0, a.stride * cm)):
            other = recs.get((epsg, e0 + de, n0 + dn, proj, surv))
            if other is None:
                continue
            em, nm = e0 + de / 2 + cm / 2, n0 + dn / 2 + cm / 2
            gap = np.hypot(*(np.array(LK.eval_displacement(res, em, nm)) - np.array(LK.eval_displacement(other, em, nm))))
            if gap > a.thr or "poor" in (res["quality"], other["quality"]):
                for k in range(1, a.stride):
                    flagged.add((epsg, e0 + (k * cm if de else 0), n0 + (k * cm if dn else 0), proj))
    base = units.drop_duplicates("project").set_index("project")
    rows = []
    for epsg, e0, n0, proj in sorted(flagged):
        r = base.loc[proj].to_dict()
        r.update(project=proj, epsg=epsg, e0=e0, n0=n0, cell_id=P.cell_id(epsg, e0, n0, cm))
        r["unit_id"] = f"{r['cell_id']}__{proj}"
        rows.append(r)
    pd.DataFrame(rows).to_parquet(a.new_units)
    print(f"{len(rows)} cells added for refinement -> {a.new_units}")


if __name__ == "__main__":
    main()
