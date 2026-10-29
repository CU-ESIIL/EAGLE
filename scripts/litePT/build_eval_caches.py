"""Fetch (or refresh) the cookie cache of every evaluation task in the pre-training config.

The one-time data step for the validation tasks: re-run it whenever an evaluation table changes (new
rows, or a new 3DEP product match, which changes `als_site_id`). For each task with a `table` and a
`cache_dir`, every row with a 3DEP product is fetched into $EAGLE_SCRATCH/cache/<name> (the last part
of `cache_dir`) with eagle_als.cache; cookies already there are kept. Cookies with the same id in the
caches named by --reuse are hard-linked in first instead of fetched again. Then the caches are packed
into <name>.tar for node-local staging (pack_for_local.sh), unless --no-pack.

    source scripts/litePT/env.sh
    python scripts/litePT/build_eval_caches.py --workers 48                 # all tasks
    python scripts/litePT/build_eval_caches.py --tasks ofo bbs --no-pack
    sbatch scripts/litePT/slurm/eval_caches.sbatch                         # the same as a slurm job

Prints, per task, how many of its rows have a usable cookie.
"""

import argparse
import os
import subprocess
from pathlib import Path

from eagle_als.cache import build_cache, cookie_path, read_manifest
from eagle_als.probe_table import load_task_table
from eagle_als.train_utils import load_config

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "ssl_litept_s.py"))
    p.add_argument("--tasks", nargs="*", help="task names (default: all with a table)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--reuse", nargs="*", default=["sites", "nlcd"], help="caches to hard-link existing cookies from")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--no-pack", action="store_true", help="do not write the <cache>.tar archives")
    a = p.parse_args()

    os.environ.pop("EAGLE_LOCAL_CACHE", None)  # build the caches on /ocean, never in a node-local copy
    cfg = load_config(a.config)
    scratch = Path(cfg["scratch"]) / "cache"
    specs = [s for s in cfg["eval_tasks"] if s.get("table") and s.get("cache_dir")]
    if a.tasks:
        specs = [s for s in specs if s["name"] in a.tasks]
    packed = []
    for spec in specs:
        name, cache = spec["name"], scratch / Path(spec["cache_dir"]).name
        id_col, tile_col = spec.get("id_col", "als_site_id"), spec.get("tile_col", "product_name_AWS")
        table = load_task_table(spec)
        sites = table[table[id_col].notna()].drop_duplicates(id_col)
        print(f"[{name}] {len(table)} rows, {len(sites)} sites with a 3DEP product -> {cache}", flush=True)

        linked = 0
        for other in a.reuse:
            if other == cache.name or not (scratch / other).is_dir():
                continue
            for sid in sites[id_col]:
                src, dst = cookie_path(scratch / other, sid), cookie_path(cache, sid)
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(src, dst)
                    linked += 1
        if linked:
            print(f"[{name}] hard-linked {linked} existing cookies", flush=True)

        build_cache(sites, cache, id_col=id_col, lat_col=spec["lat_col"], lon_col=spec["lon_col"],
                    tile_col=tile_col, workers=a.workers, retry_errors=a.retry_errors)
        man = read_manifest(cache)
        status = man[man.site_id.isin(sites[id_col].astype(str))].status.value_counts().to_dict()
        n_ok = sum(cookie_path(cache, s).exists() for s in table[id_col].dropna())
        print(f"[{name}] site status {status}; {n_ok} of {len(table)} rows have a cookie", flush=True)
        if cache.name not in packed:
            packed.append(cache.name)

    if packed and not a.no_pack:
        subprocess.run(["bash", str(HERE / "pack_for_local.sh"), "cache", *packed], check=True)


if __name__ == "__main__":
    main()
