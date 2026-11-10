"""ForPT embeddings of cached 3DEP cookies (frozen pretrained backbone, no fine-tuning).

ForPT (Yue et al. 2026, arXiv:2609.24787, https://github.com/prs-eth/ForPT) is a LitePT-S encoder pretrained with
Sonata-style self-distillation on forest ALS / ULS / MLS scans. Its teacher backbone loads into the vendored
`litept.model.LitePT(in_channels=3, enc_mode=True, norm="ln", embedding="linear", mask_token=True)` with every key
matching (checked 2026-11-10 on `00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth`).

Preprocessing follows ForPT's own downstream configs (configs/03_forspecies20k/cls-forpt-lin-*.py):
    CenterShift(apply_z=True)            x, y minus the bbox centre; z minus the lowest point
    GridSample(grid_size=0.05, train)    one point per 5 cm voxel
    feat = coord                         xyz only (in_channels = 3)
and its classifier's pooling (`DefaultClassifierLinearProbeCat`): the features of every encoder stage are up-cast to
the input points and concatenated (36 + 72 + 144 + 252 + 504 = 1008 dims), then averaged over the points.

EAGLE-specific choices (parameters, so they can be compared):
    radius_m   crop of the 100 m cookie around the site (default 50 m: the central disc the EAGLE probes use)
    height     "z" (ForPT: elevation minus the lowest point) or "hag" (height above ground from the cookie)
    max_points random thinning before the grid (bounds GPU memory on very dense tiles)

    model = load_backbone(WEIGHTS, device)
    x = embed_cookie(model, cookie, device)            # (1008,) float32
"""

from pathlib import Path

import numpy as np
import torch

FORPT_URL = "https://huggingface.co/prs-eth/ForPT/resolve/main/00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth"
WEIGHTS = Path("/home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking/forpt_weights/epoch_30.pth")
GRID = 0.05
EMBED_DIM = 1008


def load_backbone(weights=WEIGHTS, device="cuda"):
    """ForPT teacher backbone in eval mode. Loads tensors only (weights_only=True)."""
    from litept.model import LitePT

    model = LitePT(in_channels=3, enc_mode=True, norm="ln", embedding="linear", mask_token=True)
    sd = torch.load(weights, map_location="cpu", weights_only=True)["state_dict"]
    prefix = "module.teacher.backbone."
    missing, unexpected = model.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)},
                                                strict=True)
    model.shuffle_orders = False  # fixed serialization orders: repeat embeddings agree to cosine 0.9999 (bf16)
    return model.to(device).eval()


def preprocess(cookie, radius_m=50.0, height="z", max_points=None, seed=0):
    """(coord float32 [N, 3]) ready for the backbone: crop, optional thinning, centre shift, 5 cm grid sample."""
    xyz = cookie["xyz"].astype(np.float64).copy()
    if height == "hag":
        xyz[:, 2] = cookie["hag"]
    elif height != "z":
        raise ValueError(f"height must be 'z' or 'hag', got {height!r}")
    keep = (xyz[:, 0] ** 2 + xyz[:, 1] ** 2) <= radius_m**2
    xyz = xyz[keep]
    rng = np.random.default_rng(seed)
    if max_points is not None and len(xyz) > max_points:
        xyz = xyz[np.sort(rng.choice(len(xyz), max_points, replace=False))]
    lo, hi = xyz.min(0), xyz.max(0)
    xyz -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]  # CenterShift(apply_z=True)
    # GridSample(mode="train"): one random point per occupied voxel (seeded, so embeddings are reproducible)
    vox = np.floor((xyz - xyz.min(0)) / GRID).astype(np.int64)
    order = rng.permutation(len(xyz))
    _, first = np.unique(vox[order], axis=0, return_index=True)
    return xyz[np.sort(order[first])].astype(np.float32)


@torch.no_grad()
def embed_coords(model, coords, device="cuda", amp=True):
    """Mean-pooled 1008-d up-cast features for a list of preprocessed clouds (one batch); (B, 1008) numpy."""
    from litept.model import Point

    coord = torch.from_numpy(np.concatenate(coords)).to(device)
    offset = torch.tensor(np.cumsum([len(c) for c in coords]), device=device)
    point = Point(feat=coord.clone(), coord=coord, offset=offset, grid_size=GRID)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")):
        point = model(point)
        while "pooling_parent" in point.keys():  # up-cast every stage to the input points, concatenating
            parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
    feat, batch = point.feat.float(), point.batch
    sums = torch.zeros(len(coords), feat.shape[1], device=feat.device).index_add_(0, batch, feat)
    return (sums / torch.bincount(batch, minlength=len(coords)).clamp(min=1)[:, None]).cpu().numpy()


def embed_cookie(model, cookie, device="cuda", **kw):
    return embed_coords(model, [preprocess(cookie, **kw)], device)[0]
