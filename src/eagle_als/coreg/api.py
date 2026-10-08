"""High-level NAIP -> 3DEP coregistration.

    from eagle_als.coreg import api, data
    grid = data.Grid.from_center(lon, lat, 2000)            # NAD83 UTM, 1 m
    lr   = <LidarRasters on grid>                            # data.read_lidar_points + data.rasterize_lidar
    naip = data.read_naip(items, grid)                       # [4, H, W]
    res  = api.estimate(naip, lr)                            # CoregResult
    out  = api.apply(naip, lr, res)                          # NAIP resampled into the lidar frame, relief-corrected

Model: NAIP(p) = f(lidar features at q), p = q + (dx, dy) + h(q) * (tx, ty)
  (dx, dy)  ground-plane displacement of the NAIP relative to the lidar, metres east / north
  (tx, ty)  relief lean per metre of height above ground (orthorectification is to bare earth, so tall things lean from the
            frame nadir); for a 20 m tree and t = 0.1 the crown is displaced 2 m relative to its base
  f         linear map from lidar features (CHM, intensity, slope, edges, cast shadows for a *fitted* sun azimuth/elevation) to
            the NAIP bands and band edges; fitted by least squares, so no radiometric calibration is needed.
Sign: to correct NAIP, move it by (-dx, -dy) (and by -h * (tx, ty) for relief). `apply` does both.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from . import features as F
from . import model as M
from . import register as R


@dataclass
class CoregResult:
    dx: float
    dy: float
    tx: float
    ty: float
    sun_az: float | None
    sun_el: float | None
    r2: float  # R^2 of lidar-synthesised NAIP at the solution (window median)
    r2_naive: float  # same with zero shift / zero lean
    sd: float  # spread (m, root of var_x + var_y) of (dx, dy) over independent windows: reproducibility, not accuracy
    n_windows: int
    windows: list = field(default_factory=list)

    def asdict(self):
        return asdict(self)


def tall_fraction(chm, h=2.0):
    return float(np.mean(np.nan_to_num(chm) > h))


def build(naip, lr, sun=None):
    """Target and feature stacks. `sun` = (az, el) adds shadow features."""
    X = np.concatenate([F.lidar_stack(lr), F.lidar_edge_stack(lr)])
    if sun is not None:
        X = M.with_shadows(X, lr.dsm, *sun)
    Y = np.concatenate([naip, F.naip_edge_stack(naip)])
    return Y.astype(np.float32), X.astype(np.float32)


def estimate(naip, lr, res=1.0, radius=12, sun="auto", fit_lean=True, block=300, n_boot=30, seed=0, verbose=False, n_jobs=1):
    """Estimate the geometric model (see module docstring) for one NAIP image and lidar rasters on the same grid.

    sun: 'auto' (fit az/el if the scene has tall objects), None (no shadow model), or (az, el).
    Steps: (1) sun fit, (2) pooled integer search over blocks -> initial shift, (3) regression surfaces per (block, height
    class), (4) joint fit of ground shift + lean to those surfaces, (5) block bootstrap for the reproducibility `sd`.
    """
    rng = np.random.default_rng(seed)
    H, W = naip.shape[1:]
    Y, X = build(naip, lr)
    az = el = None
    if sun not in (None, "auto"):
        az, el = sun
    elif sun == "auto" and tall_fraction(lr.chm) > 0.05:
        az, el, _ = M.fit_sun(Y, lr.dsm, X, radius=8, window=min(500, H, W))
    if az is not None:
        Y, X = build(naip, lr, (az, el))
        if verbose:
            print(f"sun az={az:.0f} el={el:.0f}")
    blocks = R.block_estimates(Y, X, 320, radius, res)
    init, _ = R.pool_surfaces(blocks, radius, res)
    surfs = M.class_surfaces(Y, X, lr.chm, (init.dx, init.dy), block=block, radius=10, res=res, n_jobs=n_jobs)
    fit = M.fit_lean_model(surfs, fit_lean=fit_lean, res=res)
    # bootstrap over blocks (all classes of a block move together)
    keys = sorted({(s["row"], s["col"]) for s in surfs})
    boots = []
    for _ in range(n_boot):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        by = {}
        for s in surfs:
            by.setdefault((s["row"], s["col"]), []).append(s)
        sub = [s for kx in pick for s in by[kx]]
        f = M.fit_lean_model(sub, fit_lean=fit_lean, res=res)
        boots.append((f["dx"], f["dy"], f["tx"], f["ty"]))
    boots = np.array(boots)
    sd = float(np.hypot(*boots[:, :2].std(0)))
    r2_naive = float(np.mean([M.surface_value(s, 0.0, 0.0, 0.0, 0.0, res) for s in surfs]))  # unweighted; for reference only
    return CoregResult(fit["dx"], fit["dy"], fit["tx"], fit["ty"], az, el, fit["r2"], r2_naive, sd, len(keys), [dict(zip(("dx", "dy", "tx", "ty"), b)) for b in boots[:5]])


@dataclass
class FieldResult:
    """Smooth large-area model: d(p, h) = a + A (p - c) + h (t0 + G (p - c)); p in km east/north of the grid centre."""

    theta: list
    model: str  # 'const' or 'affine'
    centre: tuple  # (row, col) px
    sun_az: float | None
    sun_el: float | None
    r2: float
    cv_r2: dict  # held-out R^2 of the const and affine models
    n_surfaces: int
    sd_xy: float = float("nan")  # block-bootstrap sd (m) of the ground shift at the centre: reproducibility, not accuracy
    sd_lean: float = float("nan")  # same for the lean (per m of height)
    tall_frac: float = float("nan")  # fraction of the area with lidar height > 2 m
    valid_frac: float = float("nan")  # fraction of pixels with both NAIP and lidar

    @property
    def quality(self):
        """Provisional QC class from fit strength and bootstrap spread (thresholds to be re-tuned on landmark results)."""
        if not np.isfinite(self.sd_xy):
            return "poor"
        if self.r2 >= 0.10 and self.sd_xy <= 0.5:
            return "good"
        if self.r2 >= 0.06 and self.sd_xy <= 1.0:
            return "fair"
        return "poor"

    @property
    def dx(self):  # ground shift at the grid centre
        return self.theta[0]

    @property
    def dy(self):
        return self.theta[1]

    def asdict(self):
        return asdict(self)


def estimate_field(naip, lr, res=1.0, radius=12, sun="auto", block=250, cv_gain=0.005, n_jobs=1, verbose=False, return_surfaces=False, n_boot=20):
    """Large-area estimate (several km): one smooth model for the whole grid, chosen between 'const' and 'affine' by
    spatial cross-validation (affine is used only if it improves held-out R^2 by `cv_gain`)."""
    rng = np.random.default_rng(0)
    H, W = naip.shape[1:]
    Y, X = build(naip, lr)
    az = el = None
    if sun not in (None, "auto"):
        az, el = sun
    elif sun == "auto" and tall_fraction(lr.chm) > 0.05:
        c = (H // 2, W // 2)
        sl = (slice(max(c[0] - 400, 0), c[0] + 400), slice(max(c[1] - 400, 0), c[1] + 400))
        az, el, _ = M.fit_sun(Y[:, sl[0], sl[1]], lr.dsm[sl], X[:, sl[0], sl[1]], radius=8, window=500)
    if az is not None:
        Y, X = build(naip, lr, (az, el))
    blocks = R.block_estimates(Y, X, 320, radius, res)
    blocks = [blocks[i] for i in rng.choice(len(blocks), min(25, len(blocks)), replace=False)]
    init, _ = R.pool_surfaces(blocks, radius, res)
    surfs = M.class_surfaces(Y, X, lr.chm, (init.dx, init.dy), block=block, radius=10, res=res, n_jobs=n_jobs)
    centre = (H / 2, W / 2)
    cv = {m: M.cv_field_model(surfs, centre, m, res=res) for m in ("const", "affine")}
    model = "affine" if cv["affine"] - cv["const"] >= cv_gain else "const"
    fit = M.fit_field_model(surfs, centre, model, res=res)
    # block bootstrap (all classes of a block move together)
    by = {}
    for s_ in surfs:
        by.setdefault((s_["row"], s_["col"]), []).append(s_)
    keys = list(by)
    boots = []
    for _ in range(n_boot):
        sub = [s_ for k in (keys[i] for i in rng.integers(0, len(keys), len(keys))) for s_ in by[k]]
        boots.append(M.fit_field_model(sub, centre, model, res=res)["theta"][:4])
    boots = np.array(boots)
    valid = float(np.mean(np.isfinite(naip).all(0) & np.isfinite(lr.chm)))
    out = FieldResult(fit["theta"], model, centre, az, el, fit["r2"], cv, len(surfs),
                      sd_xy=float(np.hypot(*boots[:, :2].std(0))), sd_lean=float(np.hypot(*boots[:, 2:].std(0))),
                      tall_frac=tall_fraction(lr.chm), valid_frac=valid)
    return (out, surfs) if return_surfaces else out


def displacement_maps(result, lr, res=1.0, shape=None):
    """Per-pixel (Dx, Dy) [m] of the NAIP relative to the lidar at lidar pixel q: ground field + height * lean field."""
    H, W = shape or lr.chm.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float64)
    px = (cc - result.centre[1]) * res / 1000.0
    py = -(rr - result.centre[0]) * res / 1000.0
    h = np.nan_to_num(lr.chm).astype(np.float64)
    th = result.theta
    Dx = th[0] + h * th[2]
    Dy = th[1] + h * th[3]
    if result.model == "affine":
        Dx = Dx + th[4] * px + th[5] * py + h * (th[8] * px + th[9] * py)
        Dy = Dy + th[6] * px + th[7] * py + h * (th[10] * px + th[11] * py)
    return Dx, Dy


def apply_field(naip, lr, result, res=1.0, order=1):
    """Resample `naip` into the lidar frame with the per-pixel displacement of `estimate_field`."""
    H, W = naip.shape[1:]
    Dx, Dy = displacement_maps(result, lr, res, (H, W))
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float64)
    coords = [rr - Dy / res, cc + Dx / res]
    return np.stack([ndimage.map_coordinates(b, coords, order=order, mode="constant", cval=np.nan) for b in naip])


def apply(naip, lr, result, res=1.0, resampling_order=1):
    """Resample `naip` [B,H,W] into the lidar frame: out(q) = naip(q + (dx,dy) + h(q) (tx,ty)). NaN where outside."""
    H, W = naip.shape[1:]
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float64)
    h = np.nan_to_num(lr.chm).astype(np.float64)
    Dx = result.dx + h * result.tx
    Dy = result.dy + h * result.ty
    coords = [rr - Dy / res, cc + Dx / res]
    return np.stack([ndimage.map_coordinates(b, coords, order=resampling_order, mode="constant", cval=np.nan) for b in naip])
