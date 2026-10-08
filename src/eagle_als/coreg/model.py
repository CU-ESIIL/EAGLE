"""Physically informed registration: lidar -> synthetic NAIP with fitted sun angle, then shift search.

Why: the cross-modal appearance gap is not noise. NAIP has cast shadows (displaced from the objects that cast them by
h / tan(sun elevation), several metres for buildings and trees) and relief displacement; lidar has neither. Fitting the sun
(azimuth, elevation) per NAIP vintage and giving the regression shadow features removes the dominant bias of plain matching.
"""

import numpy as np

from . import features as F
from . import register as R


def with_shadows(X, dsm, az, el):
    sf = [F.standardize(a) for a in F.shadow_features(dsm, az, el)]
    return np.concatenate([X, np.stack(sf)]).astype(np.float32)


def fit_sun(naip, dsm, X, radius=10, window=800, azs=range(90, 271, 30), els=(30, 45, 60), refine=True):
    """Pick sun (az, el) maximising the regression R^2 on a central window. Returns (az, el, r2)."""
    H, W = naip.shape[1:]
    r0, c0 = max((H - window) // 2, 0), max((W - window) // 2, 0)
    sl = (slice(None), slice(r0, r0 + window), slice(c0, c0 + window))
    n, x = naip[sl], X[sl]
    d = dsm[r0 : r0 + window, c0 : c0 + window]
    best = (-1, None, None)

    def score(az, el):
        return R.regression_search(n, with_shadows(x, d, az % 360, el), radius).score

    for el in els:
        for az in azs:
            s = score(az, el)
            if s > best[0]:
                best = (s, az, el)
    if refine:
        s0, az0, el0 = best
        for daz, dele in [(-15, 0), (15, 0), (0, -10), (0, 10), (-15, -10), (15, 10), (-15, 10), (15, -10)]:
            az, el = az0 + daz, float(np.clip(el0 + dele, 15, 80))
            s = score(az, el)
            if s > best[0]:
                best = (s, az, el)
    return best[1] % 360, best[2], best[0]


# ----------------------------------------------------------------------------- geometry fit: ground shift + relief lean


def warp_coords(shape_window, origin, height, params, res=1.0, stride=1):
    """Source coordinates (rows, cols in the lidar rasters) for each NAIP pixel of a window.

    NAIP(p) ~ F(q) with  p = q + D(q),  D = (dx, dy) + h(q) * (tx, ty)   [m, east/north; t = relief lean per metre of height].
    Solved for q by two fixed-point iterations. `origin` = (row0, col0) of the window in the full rasters.
    """
    from scipy import ndimage

    dx, dy, tx, ty = params
    rr, cc = np.mgrid[0 : shape_window[0], 0 : shape_window[1]].astype(np.float64) * stride
    rr += origin[0]
    cc += origin[1]
    qr, qc = rr + dy / res, cc - dx / res  # p = q + (dx, dy): north = -rows
    for _ in range(2):
        h = ndimage.map_coordinates(height, [qr, qc], order=1, mode="nearest")
        qr = rr + (dy + h * ty) / res
        qc = cc - (dx + h * tx) / res
    return qr, qc


def fixed_r2(naip, feats, weight=None, ridge=1e-3):
    """R^2 (mean over bands) of the OLS fit naip ~ feats (+intercept) over valid pixels. Cheap closed form."""
    from numpy.linalg import solve

    valid = np.isfinite(naip).all(0)
    if weight is not None:
        valid &= weight > 0
    Y = naip[:, valid].astype(np.float64)
    X = np.concatenate([np.ones((1, valid.sum())), feats[:, valid]], 0).astype(np.float64)
    Y = (Y - Y.mean(1, keepdims=True)) / np.maximum(Y.std(1, keepdims=True), 1e-9)
    G = X @ X.T / X.shape[1]
    G[1:, 1:] += ridge * np.eye(len(G) - 1)
    C = X @ Y.T / X.shape[1]
    B = solve(G, C)
    return float(((B * C).sum(0) / 1.0).mean())


def fit_geometry(naip, feats, height, init=(0.0, 0.0), window=800, center=None, lean0=(0.0, 0.0), res=1.0, max_iter=150, fit_lean=True, stride=2):
    """Fit (dx, dy, tx, ty) by maximising the regression R^2 in a window (Nelder-Mead from `init`).

    feats [K,H,W] lidar features (incl. shadows), height [H,W] canopy height used for the lean term.
    Returns dict(dx, dy, tx, ty, r2). With fit_lean=False tx, ty are held at lean0.
    """
    from scipy import ndimage, optimize

    H, W = naip.shape[1:]
    if center is None:
        center = (H // 2, W // 2)
    half = window // 2
    r0 = int(np.clip(center[0] - half, 0, H - window))
    c0 = int(np.clip(center[1] - half, 0, W - window))
    n = naip[:, r0 : r0 + window : stride, c0 : c0 + window : stride]  # subsampled pixels: ~stride^2 faster, little information lost
    hgt = np.nan_to_num(height)
    # Feature crop with a margin + cubic-spline coefficients computed once. Plain bilinear sampling blurs features at
    # fractional offsets and biases the optimum towards half-pixel positions (seen as shifts clustering at x.5 m).
    m = 40
    fr0, fc0 = max(r0 - m, 0), max(c0 - m, 0)
    fr1, fc1 = min(r0 + window + m, H), min(c0 + window + m, W)
    coef = np.stack([ndimage.spline_filter(f[fr0:fr1, fc0:fc1].astype(np.float64), order=3) for f in feats])

    def cost(theta):
        params = (theta[0], theta[1], theta[2], theta[3]) if fit_lean else (theta[0], theta[1], lean0[0], lean0[1])
        qr, qc = warp_coords(n.shape[1:], (r0, c0), hgt, params, res, stride)
        qr, qc = qr - fr0, qc - fc0
        if qr.min() < 0 or qc.min() < 0 or qr.max() > fr1 - fr0 - 1 or qc.max() > fc1 - fc0 - 1:
            return 1.0  # sampling outside the feature crop: reject candidate
        w = np.stack([ndimage.map_coordinates(f, [qr, qc], order=3, prefilter=False, mode="nearest") for f in coef])
        pen = 2e-4 * (params[2] ** 2 + params[3] ** 2) / 0.01 if fit_lean else 0.0  # keep lean ~0 where unconstrained
        return 1.0 - fixed_r2(n, w) + pen

    x0 = np.array([init[0], init[1], lean0[0], lean0[1]] if fit_lean else [init[0], init[1]])
    scale = np.array([0.5, 0.5, 0.03, 0.03][: len(x0)])
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * scale[i] for i in range(len(x0))])
    r = optimize.minimize(cost, x0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxiter=max_iter, xatol=0.02, fatol=1e-5))
    th = r.x
    out = dict(dx=float(th[0]), dy=float(th[1]), tx=float(th[2]) if fit_lean else lean0[0], ty=float(th[3]) if fit_lean else lean0[1], r2=float(1 - r.fun - 0.0))
    return out


def fixed_r2_at(naip, feats, params, height, window, center, res=1.0):
    """R^2 for given (dx, dy, tx, ty) in a window (used for before/after reporting)."""
    from scipy import ndimage

    H, W = naip.shape[1:]
    half = window // 2
    r0 = int(np.clip(center[0] - half, 0, H - window))
    c0 = int(np.clip(center[1] - half, 0, W - window))
    n = naip[:, r0 : r0 + window, c0 : c0 + window]
    qr, qc = warp_coords(n.shape[1:], (r0, c0), np.nan_to_num(height), params, res)
    qr, qc = np.clip(qr, 0, H - 1), np.clip(qc, 0, W - 1)
    w = np.stack([ndimage.map_coordinates(f, [qr, qc], order=1, mode="nearest") for f in feats])
    return fixed_r2(n, w)


# ----------------------------------------------------------------------------- stratified (height-class) surfaces: ground shift + lean


CLASS_EDGES = (0.5, 3.0, 8.0, 15.0)  # lidar height classes (m): ground | shrub/low | building/small tree | tree | tall tree


_G = {}


def _block_job(rc):
    """Worker: class surfaces for one block (state in module global _G so forked workers share the rasters)."""
    r0, c0 = rc
    Y, Xs, cls, chm = _G["Y"], _G["Xs"], _G["cls"], _G["chm"]
    radius, size, res, min_px, nk = _G["radius"], _G["size"], _G["res"], _G["min_px"], _G["nk"]
    y = Y[:, r0 : r0 + size, c0 : c0 + size]
    if np.isfinite(y).all(0).mean() < 0.5:
        return []
    x = Xs[:, r0 : r0 + size, c0 : c0 + size]
    kk = cls[r0 : r0 + size, c0 : c0 + size]
    inner = np.zeros_like(kk, bool)
    inner[radius:-radius, radius:-radius] = True
    out = []
    for k in range(nk):
        m = (kk == k) & inner
        if m.sum() < min_px:
            continue
        try:
            r = R.regression_search(y, x, radius, res, weight=m.astype(float))
        except ValueError:
            continue
        h = float(chm[r0 : r0 + size, c0 : c0 + size][m].mean())
        out.append(dict(k=k, h=h, n=int(m.sum()), row=r0 + size / 2, col=c0 + size / 2, surface=r.surface.astype(np.float32), di0=_G["di0"], dj0=_G["dj0"], radius=radius))
    return out


def class_surfaces(Y, X, chm, s0, edges=CLASS_EDGES, block=250, radius=10, res=1.0, min_px=4000, n_jobs=1):
    """Per (block, height class) regression-R^2 surfaces over integer shifts around the initial shift `s0=(dx, dy)` [m].

    The lidar features are first moved by the integer part of s0 so the +-radius search is centred on it. A pixel belongs to
    a class by the (shifted) lidar CHM at its location. Returns list of dicts: class k, mean height h, n pixels, block centre
    (row, col) and the surface (same convention as register.regression_search, offset by (di0, dj0)).
    `n_jobs` > 1 runs blocks in forked worker processes.
    """
    H, W = Y.shape[1:]
    di0, dj0 = int(round(s0[1])), int(round(-s0[0]))  # displacement (dx, dy) <-> search offset (di, dj) = (dy, -dx)
    Xs = np.roll(X, (-di0, -dj0), axis=(1, 2))  # Xs(p) = X(p + (di0, dj0))
    chm0 = np.nan_to_num(chm)
    cs = np.roll(chm0, (-di0, -dj0), axis=(0, 1))
    size = block + 2 * radius
    _G.update(Y=Y, Xs=Xs, cls=np.digitize(cs, edges), chm=cs, radius=radius, size=size, res=res, min_px=min_px, nk=len(edges) + 1, di0=di0, dj0=dj0)
    jobs = [(r0, c0) for r0 in range(0, H - size + 1, block) for c0 in range(0, W - size + 1, block)]
    if n_jobs > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(n_jobs) as pool:
            res_ = pool.map(_block_job, jobs, chunksize=1)
    else:
        res_ = [_block_job(j) for j in jobs]
    return [s for r in res_ for s in r]


def fit_lean_model(surfs, fit_lean=True, lean_prior=0.1, res=1.0):
    """Joint fit of ground shift (dx, dy) and relief lean (tx, ty) to the class surfaces.

    Class k in a block observed at displacement  d(h) = (dx, dy) + h (tx, ty)  (NAIP rel. to lidar, m). Maximises
    sum_b,k n_bk * R2_bk(d) with each surface sub-pixel interpolated by a cubic spline. Returns dict incl. per-class optima.
    """
    from scipy import interpolate, optimize

    splines = []
    for s in surfs:
        N = s["surface"].shape[0]
        ax = np.arange(N) - s["radius"]
        splines.append(interpolate.RectBivariateSpline(ax, ax, s["surface"], kx=3, ky=3))
    n = np.array([s["n"] for s in surfs], float)
    h = np.array([s["h"] for s in surfs], float)
    di0 = np.array([s["di0"] for s in surfs], float)
    dj0 = np.array([s["dj0"] for s in surfs], float)

    def lookup(dx, dy, tx, ty):
        # displacement (dx, dy) -> search offset (di, dj) = (dy, -dx) [m/res]; surface axes are relative to (di0, dj0)
        vals = np.empty(len(surfs))
        for i, sp in enumerate(splines):
            ddx, ddy = dx + h[i] * tx, dy + h[i] * ty
            di, dj = ddy / res - di0[i], -ddx / res - dj0[i]
            r = sp.bounds if hasattr(sp, "bounds") else None
            rad = surfs[i]["radius"]
            di, dj = np.clip(di, -rad, rad), np.clip(dj, -rad, rad)
            vals[i] = sp(di, dj)[0, 0]
        return vals

    def cost(th):
        tx, ty = (th[2], th[3]) if fit_lean else (0.0, 0.0)
        pen = 0.0
        if fit_lean:  # prior: lean is physically < ~0.3 (off-nadir < 17 deg) and ~0 unless height contrast demands it
            pen = 1e-3 * (tx**2 + ty**2) / lean_prior**2 + 10.0 * max(np.hypot(tx, ty) - 0.35, 0) ** 2
        return -(n * lookup(th[0], th[1], tx, ty)).sum() / n.sum() + pen

    # start from the best single-shift: grid the pooled surface
    best = None
    for dx in np.arange(-6, 6.1, 1.0):
        for dy in np.arange(-6, 6.1, 1.0):
            c = cost([dx, dy, 0, 0])
            if best is None or c < best[0]:
                best = (c, dx, dy)
    x0 = np.array([best[1], best[2], 0.0, 0.0] if fit_lean else [best[1], best[2]])
    if not fit_lean:
        f = lambda t: cost([t[0], t[1], 0, 0])  # noqa: E731
    else:
        f = cost
    r = optimize.minimize(f, x0, method="Nelder-Mead", options=dict(xatol=0.01, fatol=1e-7, maxiter=400))
    th = r.x
    return dict(dx=float(th[0]), dy=float(th[1]), tx=float(th[2]) if fit_lean else 0.0, ty=float(th[3]) if fit_lean else 0.0, r2=float(-r.fun))


def surface_value(s, dx, dy, tx, ty, res=1.0):
    """Interpolated R^2 of one class surface at the displacement implied by (dx, dy) + h (tx, ty)."""
    from scipy import interpolate

    N = s["surface"].shape[0]
    ax = np.arange(N) - s["radius"]
    sp = interpolate.RectBivariateSpline(ax, ax, s["surface"], kx=3, ky=3)
    ddx, ddy = dx + s["h"] * tx, dy + s["h"] * ty
    rad = s["radius"]
    return float(sp(np.clip(ddy / res - s["di0"], -rad, rad), np.clip(-ddx / res - s["dj0"], -rad, rad))[0, 0])


# ----------------------------------------------------------------------------- vectorised surface stack + smooth field models


class SurfaceStack:
    """All class surfaces upsampled x`up` (cubic) so that sub-pixel lookups are vectorised bilinear gathers."""

    def __init__(self, surfs, up=4, res=1.0):
        from scipy import ndimage

        self.res, self.up = res, up
        self.n = np.array([s["n"] for s in surfs], float)
        self.h = np.array([s["h"] for s in surfs], float)
        self.row = np.array([s["row"] for s in surfs], float)
        self.col = np.array([s["col"] for s in surfs], float)
        self.di0 = np.array([s["di0"] for s in surfs], float)
        self.dj0 = np.array([s["dj0"] for s in surfs], float)
        self.rad = surfs[0]["radius"]
        N = surfs[0]["surface"].shape[0]
        z = np.stack([ndimage.zoom(s["surface"].astype(np.float64), up, order=3, grid_mode=False, mode="nearest") for s in surfs])
        self.z = z  # [S, N', N'] sample (i') at offset  i'/up - rad  (zoom with grid_mode=False maps endpoints)
        self.N, self.Np = N, z.shape[1]
        self.scale = (self.Np - 1) / (N - 1)

    def lookup(self, dx, dy):
        """Interpolated R^2 of every surface at displacement (dx, dy) [m] (arrays, one per surface)."""
        di = np.clip(dy / self.res - self.di0, -self.rad, self.rad) + self.rad
        dj = np.clip(-dx / self.res - self.dj0, -self.rad, self.rad) + self.rad
        fi, fj = di * self.scale, dj * self.scale
        i0 = np.clip(np.floor(fi).astype(int), 0, self.Np - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, self.Np - 2)
        wi, wj = fi - i0, fj - j0
        s = np.arange(len(self.n))
        z = self.z
        return (z[s, i0, j0] * (1 - wi) * (1 - wj) + z[s, i0 + 1, j0] * wi * (1 - wj) + z[s, i0, j0 + 1] * (1 - wi) * wj + z[s, i0 + 1, j0 + 1] * wi * wj)


def field_displacement(theta, stack, centre, km=1000.0, model="affine", res=1.0):
    """Displacement (dx, dy) of each surface under the smooth field model (positions relative to `centre` (row, col) px).

    d(p, h) = a + A (p - c) + h (t0 + G (p - c)),  p in km, h = class height.
    theta = [ax, ay, tx0, ty0] (+ [Axx, Axy, Ayx, Ayy, Gxx, Gxy, Gyx, Gyy] for 'affine').
    x = east = col, y = north = -row.
    """
    px = (stack.col - centre[1]) * res / km
    py = -(stack.row - centre[0]) * res / km
    ax, ay, tx0, ty0 = theta[:4]
    dx = ax + stack.h * tx0
    dy = ay + stack.h * ty0
    if model == "affine":
        Axx, Axy, Ayx, Ayy, Gxx, Gxy, Gyx, Gyy = theta[4:12]
        dx = dx + Axx * px + Axy * py + stack.h * (Gxx * px + Gxy * py)
        dy = dy + Ayx * px + Ayy * py + stack.h * (Gyx * px + Gyy * py)
    return dx, dy


def fit_field_model(surfs, centre, model="affine", lean_prior=0.1, stack=None, x0=None, res=1.0):
    """Fit a smooth large-area model (constant or affine ground shift + constant or linearly varying lean) to class surfaces."""
    from scipy import optimize

    stack = stack or SurfaceStack(surfs)
    npar = 12 if model == "affine" else 4

    def cost(th):
        dx, dy = field_displacement(th, stack, centre, model=model, res=res)
        score = (stack.n * stack.lookup(dx, dy)).sum() / stack.n.sum()
        pen = 1e-3 * (th[2] ** 2 + th[3] ** 2) / lean_prior**2
        if model == "affine":  # G (lean gradient) physically ~1/H_AGL ~ 1e-4/m = 0.1/km; A (stretch) should be small
            pen += 1e-3 * (np.sum(np.square(th[8:12])) / 0.25**2) + 1e-4 * np.sum(np.square(th[4:8])) / 1.0**2
        pen += 10.0 * max(np.hypot(th[2], th[3]) - 0.35, 0) ** 2
        return -score + pen

    if x0 is None:
        best = None
        for dx in np.arange(-6, 6.1, 1.0):
            for dy in np.arange(-6, 6.1, 1.0):
                c = cost(np.r_[dx, dy, np.zeros(npar - 2)])
                if best is None or c < best[0]:
                    best = (c, dx, dy)
        x0 = np.r_[best[1], best[2], np.zeros(npar - 2)]
    r = optimize.minimize(cost, x0, method="L-BFGS-B", options=dict(maxiter=500))
    r = optimize.minimize(cost, r.x, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-3, fatol=1e-8))
    return dict(theta=r.x.tolist(), r2=float(-r.fun), model=model)


def cv_field_model(surfs, centre, model, win=1250, folds=4, res=1.0):
    """Spatial k-fold CV (checkerboard of `win`-px windows): mean held-out pixel-weighted R^2 of the model displacement."""
    rows = np.array([s["row"] for s in surfs])
    cols = np.array([s["col"] for s in surfs])
    fold = ((rows // win).astype(int) + 2 * (cols // win).astype(int)) % folds
    scores = []
    for f in range(folds):
        tr = [s for s, k in zip(surfs, fold) if k != f]
        te = [s for s, k in zip(surfs, fold) if k == f]
        if len(te) < 5 or len(tr) < 10:
            continue
        fit = fit_field_model(tr, centre, model, res=res)
        st = SurfaceStack(te, res=res)
        dx, dy = field_displacement(fit["theta"], st, centre, model=model, res=res)
        scores.append(float((st.n * st.lookup(dx, dy)).sum() / st.n.sum()))
    return float(np.mean(scores)) if scores else float("nan")
