"""Check an SSL run's teacher encoder for representation collapse on held-out evaluation cookies.

The training log's protos_used / target_entropy describe the prototype heads, whose Sinkhorn targets
are balanced by construction, so they can look healthy while the encoder collapses. This script
embeds the same evaluation cookies with every checkpoint of a run (and a randomly initialised model
as the baseline) and reports, for the features the heads see (up-cast stage 2 + 3 + 4, 900-d):

- cookie_erank: effective rank (exp entropy of singular values) of mean-pooled cookie embeddings
- cookie_cos: mean pairwise cosine similarity between cookie embeddings (-> 1 = all cookies alike)
- point_erank: effective rank of point features (sampled across cookies)
- point_cos_within: mean cosine similarity between points of the same cookie (-> 1 = no spatial detail)
- point_std: mean per-dimension std of L2-normalised point features (-> 0 = collapse)

    python scripts/litePT/check_embeddings.py $EAGLE_SCRATCH/runs/<run_name> [--n-cookies 256]
Writes <run>/embedding_check.csv and embedding_check.png.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eagle_als.data import collate_points, pool_paths
from eagle_als.fetch import load_cookie
from eagle_als.probe import eval_transform
from eagle_als.ssl import SonataLitePT
from eagle_als.train_utils import move_to
from litept.model import Point

POINTS_PER_COOKIE = 64


def effective_rank(x):
    s = torch.linalg.svdvals(x - x.mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def mean_offdiag_cos(x):
    x = torch.nn.functional.normalize(x, dim=1)
    n = len(x)
    return float(((x @ x.T).sum() - n) / (n * (n - 1)))


@torch.no_grad()
def embed(model, samples, device):
    """Mean-pooled cookie embeddings and sampled point features from the teacher backbone."""
    cookies, points, within = [], [], []
    g = torch.Generator().manual_seed(0)
    for s in samples:
        b = move_to(collate_points([s]), device)
        point = Point(feat=b["feat"], coord=b["coord"], offset=b["offset"], grid_size=model.grid_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feat = model.up_cast(model.teacher.backbone(point)).feat.float()
        cookies.append(feat.mean(0))
        idx = torch.randperm(len(feat), generator=g)[:POINTS_PER_COOKIE].to(device)
        points.append(feat[idx])
        within.append(mean_offdiag_cos(feat[idx]))
    cookies, points = torch.stack(cookies), torch.cat(points)
    return dict(
        cookie_erank=effective_rank(cookies), cookie_cos=mean_offdiag_cos(cookies),
        point_erank=effective_rank(points), point_cos_within=float(np.mean(within)),
        point_std=float(torch.nn.functional.normalize(points, dim=1).std(0).mean()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--n-cookies", type=int, default=256)
    p.add_argument("--cache", default=None, help="evaluation cookie cache (default: the run's cache_dirs[0])")
    a = p.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = pool_paths(a.cache or cfg["cache_dirs"][0], min_points=cfg["min_points"])
    paths = random.Random(0).sample(sorted(paths), min(a.n_cookies, len(paths)))
    tf, samples = eval_transform(cfg), []
    for i, path in enumerate(paths):
        random.seed(i)
        np.random.seed(i)
        samples.append(tf(load_cookie(path)[0]))
    print(f"[embed] {len(samples)} evaluation cookies, "
          f"{np.mean([len(s['coord']) for s in samples]):.0f} voxels each on average", flush=True)

    torch.manual_seed(cfg["seed"])
    model = SonataLitePT(cfg["backbone"], **cfg["ssl"]).to(device).eval()
    ckpts = [(0, None)] + sorted((int(c.stem.split("_")[1]), c) for c in (run / "checkpoints").glob("step_*.pth"))
    rows = []
    for step, ckpt in ckpts:
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        row = dict(step=step, **embed(model, samples, device))
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
