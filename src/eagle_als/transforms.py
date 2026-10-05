"""ALS-specific data transforms, registered into the vendored LitePT TRANSFORMS registry.

They can be freely mixed with the LitePT transforms (src/litept/transform.py) in config files:

    dict(type="ALSFeatures"),            # cookie arrays -> coord / per-point feature keys
    dict(type="RandomPointThinning", ...),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1),   # LitePT
    ...

Per-point keys produced by ALSFeatures (all indexed consistently by LitePT transforms):
    coord          [N, 3] x, y relative to the cookie center, z relative to median ground (m)
    origin_coord   [N, 3] copy of coord before augmentation (used for view matching / plotting)
    hag            [N, 1] height above ground (m) / hag_scale, clipped
    intensity      [N, 1] per-cookie percentile rank of intensity in [0, 1] (sensor-agnostic)
    returns        [N, 3] (return_number / number_of_returns, is_first, is_last)
"""

import copy
import random

import numpy as np

from litept.transform import TRANSFORMS, Compose, index_operator

ALS_POINT_KEYS = ["coord", "origin_coord", "hag", "intensity", "returns", "classification", "segment"]


@TRANSFORMS.register_module()
class ALSFeatures:
    """Convert a raw cookie (see eagle_als.fetch) into LitePT-style point keys."""

    def __init__(self, hag_scale=30.0, hag_clip=(-5.0, 100.0), keep_classification=False):
        self.hag_scale = hag_scale
        self.hag_clip = hag_clip
        self.keep_classification = keep_classification

    def __call__(self, d):
        xyz = d.pop("xyz").astype(np.float32)
        hag = d.pop("hag").astype(np.float32)
        ground_z = np.median(xyz[:, 2] - hag)
        coord = xyz.copy()
        coord[:, 2] -= ground_z
        inten = d.pop("intensity").astype(np.float32)
        # percentile rank within the cookie: robust to the sensor-specific intensity scaling
        rank = np.empty(len(inten), dtype=np.float32)
        rank[np.argsort(inten, kind="stable")] = np.linspace(0, 1, len(inten), dtype=np.float32)
        rn = d.pop("return_number").astype(np.float32)
        nr = np.maximum(d.pop("number_of_returns").astype(np.float32), 1)
        rn = np.clip(rn, 1, nr)
        returns = np.stack([rn / nr, rn == 1, rn == nr], axis=1).astype(np.float32)
        cls = d.pop("classification")
        d.update(
            coord=coord,
            origin_coord=coord.copy(),
            hag=(np.clip(hag, *self.hag_clip) / self.hag_scale)[:, None],
            intensity=rank[:, None],
            returns=returns,
        )
        if self.keep_classification:
            d["classification"] = cls.astype(np.int64)
        d["index_valid_keys"] = [k for k in ALS_POINT_KEYS if k in d]
        return d


@TRANSFORMS.register_module()
class RandomPointThinning:
    """Randomly keep a fraction of points to simulate lower pulse densities (QL0 -> QL2 -> older data).

    keep ratio ~ U(ratio_range) when applied (probability p). With target_density (pts/m^2) set,
    the ratio is chosen so the output density falls in that range instead (never upsamples).
    """

    def __init__(self, ratio_range=(0.25, 1.0), target_density=None, radius=100.0, p=0.5):
        self.ratio_range = ratio_range
        self.target_density = target_density
        self.radius = radius
        self.p = p

    def __call__(self, d):
        if random.random() > self.p:
            return d
        n = len(d["coord"])
        if self.target_density is not None:
            density = n / (np.pi * self.radius**2)
            ratio = min(1.0, np.random.uniform(*self.target_density) / max(density, 1e-6))
        else:
            ratio = np.random.uniform(*self.ratio_range)
        keep = max(int(n * ratio), 1)
        if keep >= n:
            return d
        idx = np.sort(np.random.choice(n, keep, replace=False))
        return index_operator(d, idx)


@TRANSFORMS.register_module()
class XYCrop:
    """Keep points within `radius` (m) of `center` in the horizontal plane.

    radius may be a (min, max) range to randomly vary the spatial context; center=None keeps (0, 0)
    (the cookie / plot center). With `random_center` the center is drawn uniformly so the crop stays
    inside the cookie of radius `cookie_radius`.
    """

    def __init__(self, radius=50.0, random_center=False, cookie_radius=100.0):
        self.radius = radius
        self.random_center = random_center
        self.cookie_radius = cookie_radius

    def __call__(self, d):
        r = np.random.uniform(*self.radius) if isinstance(self.radius, (list, tuple)) else self.radius
        c = np.zeros(2, dtype=np.float32)
        if self.random_center and r < self.cookie_radius:
            rho = (self.cookie_radius - r) * np.sqrt(np.random.rand())
            phi = np.random.uniform(0, 2 * np.pi)
            c = np.array([rho * np.cos(phi), rho * np.sin(phi)], dtype=np.float32)
        ref = d["origin_coord"] if "origin_coord" in d else d["coord"]
        keep = np.where(((ref[:, :2] - c) ** 2).sum(1) <= r * r)[0]
        if len(keep) == 0:
            return d
        return index_operator(d, keep)


@TRANSFORMS.register_module()
class IntensityJitter:
    """Perturb the (rank-normalized) intensity: random gamma, gain and gaussian noise; or drop it."""

    def __init__(self, gamma=(0.7, 1.4), gain=(0.9, 1.1), std=0.03, drop_p=0.1, p=0.8):
        self.gamma, self.gain, self.std, self.drop_p, self.p = gamma, gain, std, drop_p, p

    def __call__(self, d):
        if "intensity" not in d or random.random() > self.p:
            return d
        x = d["intensity"]
        if random.random() < self.drop_p:
            d["intensity"] = np.full_like(x, 0.5)
            return d
        x = np.clip(x, 0, 1) ** np.random.uniform(*self.gamma) * np.random.uniform(*self.gain)
        x = x + np.random.normal(0, self.std, x.shape).astype(np.float32)
        d["intensity"] = np.clip(x, 0, 1).astype(np.float32)
        return d


@TRANSFORMS.register_module()
class HAGJitter:
    """Simulate ground-classification error: a smooth random offset added to height above ground."""

    def __init__(self, std=0.15, slope_std=0.002, p=0.5):
        self.std, self.slope_std, self.p = std, slope_std, p

    def __call__(self, d):
        if "hag" not in d or random.random() > self.p:
            return d
        hag_scale = 30.0
        a = np.random.normal(0, self.slope_std, 2)
        off = np.random.normal(0, self.std) + d["origin_coord"][:, :2] @ a
        d["hag"] = d["hag"] + (off[:, None] / hag_scale).astype(np.float32)
        return d


@TRANSFORMS.register_module()
class ReturnDrop:
    """Randomly remove all non-first returns (mimics sensors / processing that keep fewer returns)."""

    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, d):
        if "returns" not in d or random.random() > self.p:
            return d
        keep = np.where(d["returns"][:, 1] > 0.5)[0]
        return index_operator(d, keep) if len(keep) > 0 else d


@TRANSFORMS.register_module()
class ALSMultiViewGenerator:
    """Sonata-style global/local views for ALS, defined by horizontal *area* instead of point count.

    Each view is the set of points within an xy-radius of a center, where the radius is drawn so the
    view covers `scale` x the area of a disc of radius `max_radius`. The view is capped at `max_size`
    points (closest to the center), so dense and sparse tiles give views of similar spatial extent.
    Output keys follow LitePT's MultiViewGenerator: global_<key>, global_offset, local_<key>, local_offset.
    """

    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.05, 0.2),
        max_radius=60.0,
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        view_keys=("coord", "origin_coord", "hag", "intensity", "returns"),
        keep_keys=("coord", "origin_coord", "grid_coord", "feat"),
    ):
        self.global_view_num = global_view_num
        self.global_view_scale = global_view_scale
        self.local_view_num = local_view_num
        self.local_view_scale = local_view_scale
        self.max_radius = max_radius
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.view_keys = view_keys
        self.keep_keys = keep_keys

    def get_view(self, point, center, scale):
        r = self.max_radius * np.sqrt(np.random.uniform(*scale))
        d2 = np.sum(np.square(point["origin_coord"][:, :2] - center[:2]), axis=-1)
        index = np.where(d2 <= r * r)[0]
        if len(index) > self.max_size:
            index = index[np.argpartition(d2[index], self.max_size)[: self.max_size]]
        view = {key: point[key][index] for key in self.view_keys if key in point}
        view["index"] = index
        view["index_valid_keys"] = [k for k in point.get("index_valid_keys", self.view_keys) if k in view]
        return view

    def __call__(self, data_dict):
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        oc = point["origin_coord"]
        # major view centered somewhere within the inner part of the cookie
        inner = np.where((oc[:, :2] ** 2).sum(1) <= (0.5 * np.abs(oc[:, :2]).max()) ** 2)[0]
        major_center = oc[np.random.choice(inner if len(inner) else np.arange(len(oc)))]
        major = self.get_view(point, major_center, self.global_view_scale)
        mc = major["origin_coord"]
        global_views = [major] + [
            self.get_view(point, mc[np.random.randint(len(mc))], self.global_view_scale)
            for _ in range(self.global_view_num - 1)
        ]
        local_views = [
            self.get_view(point, mc[np.random.randint(len(mc))], self.local_view_scale)
            for _ in range(self.local_view_num)
        ]
        out = {}
        for prefix, views, tf in (("global", global_views, self.global_transform), ("local", local_views, self.local_transform)):
            if not views:
                continue
            views = [tf({k: v for k, v in view.items() if k != "index"}) for view in views]
            for key in self.keep_keys:
                out[f"{prefix}_{key}"] = np.concatenate([v[key] for v in views], axis=0)
            out[f"{prefix}_offset"] = np.cumsum([len(v["coord"]) for v in views])
        return out


@TRANSFORMS.register_module()
class PackFeat:
    """Concatenate per-point keys into the model input `feat` (float32 [N, C])."""

    def __init__(self, keys=("coord", "hag", "intensity", "returns"), coord_scale=50.0):
        self.keys = keys
        self.coord_scale = coord_scale

    def __call__(self, d):
        parts = []
        for k in self.keys:
            v = d[k]
            if k == "coord":
                v = v / self.coord_scale  # keep coordinate features O(1)
            parts.append(v.reshape(len(v), -1).astype(np.float32))
        d["feat"] = np.concatenate(parts, axis=1)
        d["coord"] = d["coord"].astype(np.float32)
        if "index_valid_keys" in d and "feat" not in d["index_valid_keys"]:
            d["index_valid_keys"].append("feat")
        return d


def feat_channels(keys=("coord", "hag", "intensity", "returns")):
    """Number of input channels produced by PackFeat(keys)."""
    sizes = dict(coord=3, hag=1, intensity=1, returns=3)
    return sum(sizes[k] for k in keys)
