"""Check an SSL run's teacher encoder for representation collapse on held-out evaluation cookies.

Runs the embedding check (eagle_als.embedding_check, which explains the metrics) on every checkpoint of
a run and on a randomly initialised model as the baseline. Training also runs it every `eval_every`
steps when `eval_tasks` includes a `type="embedding_check"` task (TensorBoard eval/embed/*). Needs a
GPU (sparse convolutions).

    python scripts/litePT/check_embeddings.py $EAGLE_SCRATCH/runs/<run_name> [--n-cookies 256] [--cache DIR]
Writes <run>/embedding_check.csv and embedding_check.png. Use --cache when the run's cache_dirs[0] was
a node-local copy (/local/...) that no longer exists.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eagle_als.embedding_check import embedding_stats, sample_cookies
from eagle_als.ssl import SonataLitePT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--n-cookies", type=int, default=256)
    p.add_argument("--cache", default=None, help="evaluation cookie cache (default: the run's cache_dirs[0])")
    a = p.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = sample_cookies(a.cache or cfg["cache_dirs"][0], cfg, a.n_cookies)
    print(f"[embed] {len(samples)} evaluation cookies, "
          f"{np.mean([len(s['coord']) for s in samples]):.0f} voxels each on average", flush=True)

    torch.manual_seed(cfg["seed"])
    model = SonataLitePT(cfg["backbone"], **cfg["ssl"]).to(device).eval()
    ckpts = [(0, None)] + sorted((int(c.stem.split("_")[1]), c) for c in (run / "checkpoints").glob("step_*.pth"))
    rows = []
    for step, ckpt in ckpts:
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        row = dict(step=step, **embedding_stats(model, samples, device))
        rows.append(row)
        print("[embed] " + " ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "embedding_check.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in df.columns if c != "step"]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.2))
    for ax, c in zip(axes, cols):
        ax.plot(df.step, df[c], "o-")
        ax.set_title(c)
        ax.set_xlabel("step (0 = random init)")
    fig.tight_layout()
    fig.savefig(run / "embedding_check.png", dpi=110)
    print(f"[embed] wrote {run / 'embedding_check.csv'} and .png", flush=True)


if __name__ == "__main__":
    main()
