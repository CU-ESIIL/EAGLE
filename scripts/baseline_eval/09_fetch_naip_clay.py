"""Clay v1.5 NAIP embeddings averaged over a disc around every benchmark point, one disc radius per task.

    python scripts/baseline_eval/09_fetch_naip_clay.py --out $BENCH/naip_clay --workers 16
    python scripts/baseline_eval/02_linear_probe.py --naip-clay $BENCH/naip_clay --model naip_clay --n-jobs 4

Radii (metres, 0 = the chip under the point) are the `radius` of each task's table in datasets/registry.yaml (task ->
table by name prefix: lfrdb_ecosys -> lfrdb); blank ones use --default-radius (50 m). A file of
~3.6 MB is downloaded for every NAIP quarter-quad a disc touches, used for the points that need it and dropped: the
chips are not stored. What is kept in <out>:
  naip_index.parquet           list of the ~1.2 M files (built once, 3 min)
  partial/part_<n>.npz         resume checkpoint: per (point, radius, file) matched samples and weighted sum
  pooled/<task>.parquet        the result: key, lat, lon, emb_year, radius, naip_year, coverage, e0000..e1023
Changing a radius or adding points only fetches the files whose (point, radius) pairs are not in partial/ yet.
`coverage` = share of the disc's samples that fell on a chip (NaN-free rows need > 0). `naip_year` is the image year
at the point. Rerunning with everything done just rewrites pooled/.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eagle_als import registry  # noqa: E402
from eagle_bench import aef, naip_clay as N  # noqa: E402
from eagle_bench.tasks import TASKS, load_task  # noqa: E402


def task_radii(tasks, override=None, default=50.0):
    """{task: disc radius in metres}: the `radius` of the task's registry table (or the override yaml: task or table -> radius), else `default`."""
    reg, over, out = registry.load_registry(), (override or {}), {}
    for t in tasks:
        table = max((k for k in reg if t.startswith(k)), key=len, default=None)
        if table is None:
            raise SystemExit(f"task {t} matches no table in datasets/registry.yaml")
        r = over.get(t, over.get(table, reg[table].get("radius")))
        out[t] = float(default if r is None else r)
        if r is None:
            print(f"[naip] no radius for {t} (table {table}) in the registry: using the default {default:.0f} m", flush=True)
    return out


def load_partials(pdir):
    keep = {k: [] for k in ("key", "radius", "file_id", "year", "centre", "n", "sum")}
    for f in sorted(pdir.glob("part_*.npz")):
        z = np.load(f, allow_pickle=False)
        for k in keep:
            keep[k].append(z[k])
    return {k: (np.concatenate(v) if v else None) for k, v in keep.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default=None, help="optional yaml overriding registry radii: {task or table: metres}")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--out", default=str(N.CACHE_DIR))
    ap.add_argument("--workers", type=int, default=16, help="concurrent downloads")
    ap.add_argument("--chunk", type=int, default=400, help="files per checkpoint")
    ap.add_argument("--max-points", type=int, default=None, help="random subset of points (testing)")
    ap.add_argument("--default-radius", type=float, default=50.0, help="metres, for tables without a registry `radius`")
    ap.add_argument("--rebuild-index", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "partial").mkdir(parents=True, exist_ok=True)
    (out / "pooled").mkdir(exist_ok=True)
    radii = task_radii(a.tasks, yaml.safe_load(open(a.radii)) if a.radii else None, a.default_radius)

    # ---- points: one row per (lat, lon, target year), with every radius some task wants for it
    frames = {}
    for name in a.tasks:
        t = load_task(name, clip_years=N.YEARS)
        df = t.df[["lat", "lon", "emb_year"]].copy()
        df["key"] = aef.point_key(df)
        df["radius"] = radii[name]
        frames[name] = df
    allp = pd.concat(frames.values(), ignore_index=True)
    pts = allp.drop_duplicates("key").drop(columns="radius").reset_index(drop=True)
    if a.max_points:
        pts = pts.sample(min(a.max_points, len(pts)), random_state=0).reset_index(drop=True)
    want = allp[allp["key"].isin(set(pts["key"]))].drop_duplicates(["key", "radius"])  # (key, radius) to compute
    pts["rmax"] = pts["key"].map(want.groupby("key")["radius"].max())
    print(f"[naip] {len(pts)} points, {len(want)} (point, radius) pairs; radii: "
          f"{radii}", flush=True)

    index = N.build_index(out) if a.rebuild_index else N.load_index(out)
    cells = N.CellFiles(index)
    pairs = N.plan(pts, index, cells)
    no_cover = len(pts) - pairs["pt"].nunique()
    print(f"[naip] {len(pairs)} (point, file) pairs over {pairs['file_id'].nunique()} files; "
          f"{no_cover} points touch no cell with NAIP embeddings", flush=True)

    # ---- what is already done
    part = load_partials(out / "partial")
    done = set()
    if part["key"] is not None:
        done = set(zip(part["key"].tolist(), part["radius"].tolist(), part["file_id"].tolist()))
    rad_of = want.groupby("key")["radius"].apply(list).to_dict()
    jobs = {}  # file_id -> [(pt row, [radii still to do])]
    for pt, fid in zip(pairs["pt"], pairs["file_id"]):
        k = pts["key"].iat[pt]
        todo = [r for r in rad_of[k] if (k, r, fid) not in done]
        if todo:
            jobs.setdefault(int(fid), []).append((int(pt), todo))
    gb = index.set_index("file_id").loc[list(jobs), "size"].sum() / 1e9 if jobs else 0
    print(f"[naip] {len(jobs)} files to read ({gb:.0f} GB), {len(pairs['file_id'].unique()) - len(jobs)} already done",
          flush=True)

    meta = index.set_index("file_id")

    def work(fid):
        row = meta.loc[fid]
        try:
            bounds, emb = N.read_chips(row["key"])
        except Exception as e:  # not recorded as done: retried on the next run
            return fid, None, repr(e)
        cell, recs = (int(row["iy"]), int(row["ix"])), []
        for pt, radii in jobs[fid]:
            lat, lon = float(pts["lat"].iat[pt]), float(pts["lon"].iat[pt])
            centre = int(row["year"]) if (int(np.floor(lat / N.CELL)), int(np.floor(lon / N.CELL))) == cell else -1
            for r, (n, s) in zip(radii, N.chip_sums(bounds, emb, cell, lat, lon, radii)):
                recs.append((pts["key"].iat[pt], r, fid, int(row["year"]), centre, n, s))
        return fid, recs, None

    def write_shard(recs):
        n = len(list(out.glob("partial/part_*.npz")))
        np.savez(out / f"partial/part_{n:05d}.npz", key=np.array([r[0] for r in recs]),
                 radius=np.array([r[1] for r in recs], np.float64), file_id=np.array([r[2] for r in recs], np.int64),
                 year=np.array([r[3] for r in recs], np.int16), centre=np.array([r[4] for r in recs], np.int16),
                 n=np.array([r[5] for r in recs], np.int32), sum=np.stack([r[6] for r in recs]).astype(np.float32))

    fids, t0, n_done, n_fail = list(jobs), time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i in range(0, len(fids), a.chunk):
            recs = []
            for fid, r, err in ex.map(work, fids[i:i + a.chunk]):
                if err:
                    n_fail += 1
                    print(f"[naip] file {fid} failed: {err}", flush=True)
                else:
                    recs += r
                    n_done += 1
            if recs:
                write_shard(recs)
            el = time.time() - t0
            print(f"[naip] {n_done}/{len(fids)} files, {n_fail} failed, {el / 60:.1f} min, "
                  f"{n_done / max(el, 1e-9):.1f} files/s", flush=True)

    # ---- pool: sum partials per (key, radius), write one table per task
    part = load_partials(out / "partial")
    if part["key"] is None:
        print("[naip] nothing fetched", flush=True)
        return
    kr = pd.DataFrame({"key": part["key"], "radius": part["radius"]})
    codes, uniq = pd.factorize(pd.MultiIndex.from_frame(kr))
    tot_sum = np.zeros((len(uniq), N.N_DIM), np.float32)
    np.add.at(tot_sum, codes, part["sum"])
    tot_n = np.bincount(codes, weights=part["n"], minlength=len(uniq))
    year = np.full(len(uniq), -1)
    np.maximum.at(year, codes, np.where(part["centre"] > 0, part["year"], -1))
    res = pd.DataFrame({"key": [u[0] for u in uniq], "radius": [u[1] for u in uniq], "naip_year": year, "n": tot_n})
    res["coverage"] = res["n"] / res["radius"].map(lambda r: N.n_samples(r))
    cols = N.embed_columns()
    emb = pd.DataFrame(tot_sum / np.maximum(tot_n, 1)[:, None].astype(np.float32), columns=cols)
    emb[tot_n == 0] = np.nan  # a disc with no chip under any sample has no embedding
    res = pd.concat([res, emb], axis=1).set_index(["key", "radius"])
    for name, df in frames.items():
        sub = df.drop_duplicates("key")
        tab = sub.join(res, on=["key", "radius"], how="left").drop(columns="n")
        tab.to_parquet(out / f"pooled/{name}.parquet")
        have = tab[cols[0]].notna()
        print(f"[naip] pooled {name} (radius {sub['radius'].iat[0]:.0f} m): {have.sum()}/{len(tab)} points with an embedding; "
              f"median coverage {tab.loc[have, 'coverage'].median():.2f}; "
              f"|NAIP year - target| median {(tab.loc[have, 'naip_year'] - tab.loc[have, 'emb_year']).abs().median():.0f}",
              flush=True)


if __name__ == "__main__":
    main()
