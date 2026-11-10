"""Collapse check on the frozen teacher encoder: spread of cookie embeddings and point features.

Registered as `type="embedding_check"` in `eval_tasks` (eagle_als.evaluation), and used by
`scripts/litePT/check_embeddings.py` to score a run's saved checkpoints. No labels are needed:

    dict(type="embedding_check",
         name="embed",                     # tensorboard: eval/embed/*
         cache_dir=".../cache/nlcd",       # any cookie cache; held-out evaluation cookies are best
         n_cookies=256)                    # optional (default 256), a fixed random sample of the cache

The training log's protos_used / target_entropy describe the prototype heads, whose Sinkhorn targets
are balanced by construction, so they can look healthy while the encoder collapses. This looks at the
features the heads see (up-cast stage 2 + 3 + 4, 900-d), one fixed view per cookie (as in the probes):

- cookie_erank: effective rank (exp entropy of singular values) of mean-pooled cookie embeddings
- cookie_cos: mean pairwise cosine similarity between cookie embeddings (-> 1 = all cookies alike)
- point_erank: effective rank of point features (sampled across cookies)
- point_cos_within: mean cosine similarity between points of the same cookie (-> 1 = no spatial detail)
- point_std: mean per-dimension std of L2-normalised point features (-> 0 = collapse)
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from litept.model import Point

from .data import collate_points, pool_paths
from .evaluation import EvalTask, register
from .probe import prepare_samples
from .train_utils import move_to

POINTS_PER_COOKIE = 64


def effective_rank(x):
    s = torch.linalg.svdvals(x - x.mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def mean_offdiag_cos(x):
    x = F.normalize(x, dim=1)
    n = len(x)
    return float(((x @ x.T).sum() - n) / (n * (n - 1)))


def sample_cookies(cache_dir, cfg, n_cookies=256):
    """Fixed model inputs for a seeded random sample of a cache's usable cookies."""
    paths = pool_paths(cache_dir, min_points=cfg["min_points"])
    paths = random.Random(0).sample(sorted(paths), min(n_cookies, len(paths)))
    return prepare_samples(paths, cfg)


@torch.no_grad()
def embedding_stats(model, samples, device):
    """{metric: value} from the teacher backbone's up-cast features, one cookie at a time."""
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
        point_std=float(F.normalize(points, dim=1).std(0).mean()),
    )


@register("embedding_check")
class EmbeddingCheck(EvalTask):
    """A fixed sample of cookies, preprocessed once; run() reports the collapse statistics."""

    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        self.samples = sample_cookies(spec["cache_dir"], cfg, spec.get("n_cookies", 256))
        if len(self.samples) < 2:
            raise ValueError(f"eval task {self.name}: fewer than 2 usable cookies in {spec['cache_dir']}")
        print(f"[eval] {self.name}: embedding check on {len(self.samples)} cookies from {spec['cache_dir']}, "
              f"{np.mean([len(s['coord']) for s in self.samples]):.0f} voxels per cookie", flush=True)

    def run(self, model, device):
        return embedding_stats(model, self.samples, device)
