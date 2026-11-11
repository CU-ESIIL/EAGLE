"""SatCLIP location embeddings (lat, lon -> 256-d) linear-probed on every benchmark task.

    python scripts/baseline_eval/08_satclip_probe.py --ckpt <dir>/satclip-vit16-l10.ckpt --model satclip_l10

Same probes, splits and result layout as 02_linear_probe.py / 06_location_baseline.py; no imagery or lidar needed.
Uses the official SatCLIP package (github.com/microsoft/satclip, cloned at ~/external/satclip and pip-installed
editable into the venv this runs in; checkpoints from Hugging Face microsoft/SatCLIP-ViT16-L10 / -L40).
Embedding = `get_satclip(ckpt, device)` (the location tower) applied to (lon, lat) in degrees.
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from satclip import get_satclip

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/baseline_eval"))
from eagle_bench import probes  # noqa: E402
from eagle_bench.tasks import TASKS, load_task  # noqa: E402

HEADLINE = import_module("02_linear_probe").HEADLINE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--min-counts", nargs="*", type=int, default=[20, 1])
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    out = ROOT / "scripts/baseline_eval/results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    enc = get_satclip(a.ckpt, "cpu")
    write_markdown = import_module("02_linear_probe").write_markdown

    summaries, per_target = [], []
    for name in a.tasks:
        task = load_task(name)
        ll = task.df[["lon", "lat"]].to_numpy(float)
        good = np.isfinite(ll).all(1)
        X = np.full((len(ll), 256), np.nan, np.float32)
        with torch.no_grad():
            X[good] = enc(torch.from_numpy(ll[good])).float().numpy()
        for min_count in ([None] if task.kind == "regression" else a.min_counts):
            s, per = probes.evaluate(task, X, n_jobs=a.n_jobs, min_count=min_count)
            s.update(pooling="location", frac_year_clipped=task.meta["frac_year_clipped"], description=task.description)
            per.insert(1, "pooling", "location")
            per.insert(2, "min_count", min_count)
            summaries.append(s)
            per_target.append(per)
            print(f"[satclip] {name} (min_count={min_count}): " + ", ".join(
                f"{k}={s[k]:.3f}" for k in HEADLINE[task.kind] if k in s), flush=True)
            pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
            pd.concat(per_target, ignore_index=True).to_csv(out / "per_target.csv", index=False)
    write_markdown(pd.DataFrame(summaries), out / "summary.md", a.model)
    import_module("07_unified_summary").main()


if __name__ == "__main__":
    main()
