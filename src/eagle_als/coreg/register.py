"""Translation estimators between a NAIP image and lidar-derived layers on the same grid.

Convention (used everywhere): the result `(dx, dy)` is the displacement in metres (east, north) of the NAIP content
relative to the lidar, i.e. a NAIP feature that sits at lidar position P is found at P + (dx, dy) in the NAIP.
To correct the NAIP, move it by (-dx, -dy).

Internally the search offsets (di, dj) = (rows, cols) are where NAIP(p) is compared with lidar(p + (di, dj)); that is the
opposite of the NAIP displacement, and north = -rows, so dx = -dj * res, dy = +di * res.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal


@dataclass
class ShiftResult:
    dx: float  # m east of NAIP relative to lidar
    dy: float  # m north
    score: float  # estimator specific; higher is better (R^2 / NCC / MI)
    surface: np.ndarray | None = None  # score over the integer search grid [(2R+1), (2R+1)] rows=di, cols=dj
    radius: int = 0
    method: str = ""
    n_px: int = 0

    def __iter__(self):
        return iter((self.dx, self.dy))


# ----------------------------------------------------------------------------- helpers


def fill_nan(a, sigma=2.0):
    """Replace NaN by the nearest finite value (then light blur only where filled)."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    if bad.all():
        return np.zeros_like(a)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def standardize(a, mask=None):
    m = np.isfinite(a) if mask is None else mask & np.isfinite(a)
    mu, sd = a[m].mean(), a[m].std()
    return (a - mu) / (sd if sd > 0 else 1.0)


def quad_peak(surface, i, j, half=2, maximize=True):
    """Sub-sample location of an extremum of `surface` near integer (i, j) from a 2-D quadratic least-squares fit."""
    H, W = surface.shape
    i0, i1, j0, j1 = max(i - half, 0), min(i + half + 1, H), max(j - half, 0), min(j + half + 1, W)
    ii, jj = np.mgrid[i0:i1, j0:j1]
    z = surface[i0:i1, j0:j1].ravel()
    ii, jj = (ii - i).ravel().astype(float), (jj - j).ravel().astype(float)
    A = np.stack([np.ones_like(ii), ii, jj, ii * ii, jj * jj, ii * jj], 1)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    Hm = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
    try:
        d = -np.linalg.solve(Hm, c[1:3])
    except np.linalg.LinAlgError:
        return float(i), float(j)
    ok = (np.linalg.eigvalsh(Hm) < 0).all() if maximize else (np.linalg.eigvalsh(Hm) > 0).all()
    if not ok or np.abs(d).max() > half:
        return float(i), float(j)
    return i + d[0], j + d[1]


def _to_xy(di, dj, res):
    return -dj * res, di * res


def _crop(a, R):
    return a[..., R : a.shape[-2] - R, R : a.shape[-1] - R]


# ----------------------------------------------------------------------------- baseline 1: NCC


def ncc_search(target, source, radius=10, res=1.0, mask=None):
    """Zero-mean normalised cross-correlation of one NAIP-derived image (`target`) with one lidar-derived image (`source`).

    `target` is cropped by `radius` so every shift has full overlap. Returns ShiftResult (score = max NCC).
    """
    t = np.where(np.isfinite(target), target, np.nanmean(target)).astype(np.float64)
    s = fill_nan(source).astype(np.float64)
    tw = _crop(t, radius)
    w = np.ones_like(tw) if mask is None else _crop(mask, radius).astype(np.float64)
    n = w.sum()
    tm = (tw * w).sum() / n
    tz = (tw - tm) * w
    tvar = (tz**2).sum()
    ones = np.ones_like(tw)
    cross = signal.correlate(s, tz, mode="valid", method="fft")
    sw = signal.correlate(s, w, mode="valid", method="fft")
    s2w = signal.correlate(s * s, w, mode="valid", method="fft")
    svar = s2w - sw**2 / n
    ncc = cross / np.sqrt(np.maximum(svar, 1e-12) * tvar)
    ncc = np.nan_to_num(ncc, nan=-1)
    return _finish(ncc, radius, res, "ncc", int(n))


def _finish(score, radius, res, method, n_px, maximize=True):
    f = score if maximize else -score
    k = np.unravel_index(np.argmax(f), f.shape)
    si, sj = quad_peak(f, k[0], k[1])
    di, dj = si - radius, sj - radius
    dx, dy = _to_xy(di, dj, res)
    return ShiftResult(dx, dy, float(f[k]), score, radius, method, n_px)


# ----------------------------------------------------------------------------- baseline 2: mutual information


def mi_search(target, source, radius=10, res=1.0, bins=24, step=1):
    """Mutual information over integer shifts (brute force, `step` pixel stride on the images for speed)."""
    t = np.where(np.isfinite(target), target, np.nanmean(target))
    s = fill_nan(source)
    tw = _crop(t, radius)[::step, ::step]
    tq = _quantile_bin(tw, bins)
    sq_full = _quantile_bin(s, bins)
    N = 2 * radius + 1
    mi = np.zeros((N, N))
    h, w = _crop(t, radius).shape
    for a in range(N):
        for b in range(N):
            sw = sq_full[a : a + h : step, b : b + w : step]
            joint = np.bincount((tq * bins + sw).ravel(), minlength=bins * bins).reshape(bins, bins).astype(float)
            p = joint / joint.sum()
            px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
            nz = p > 0
            mi[a, b] = (p[nz] * np.log(p[nz] / (px @ py)[nz])).sum()
    return _finish(mi, radius, res, "mi", tw.size)


def _quantile_bin(a, bins):
    qs = np.quantile(a[np.isfinite(a)], np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(qs, a).astype(np.int64)


# ----------------------------------------------------------------------------- main: multi-channel regression


def regression_search(naip, feats, radius=10, res=1.0, weight=None, ridge=1e-3, return_coef=False):
    """Find the shift at which a linear model of lidar features best explains the NAIP bands.

    naip   [B, H, W]  NAIP bands (any radiometry; each is standardised)
    feats  [K, H, W]  lidar-derived features (no NaN; an intercept is added)
    weight [H, W]     optional 0/1 or soft pixel weights on the NAIP side (e.g. valid pixels, a height stratum)

    For every integer shift s the model  naip(p) ~ sum_k b_k feats_k(p + s)  is fit by least squares *for all shifts at once*
    through FFT cross-correlations of the normal equations. Score = R^2 of the best-fitting model (mean over bands).
    A pixel-synthesis view of coregistration: a lidar-based prediction of the image is sharpest when aligned.
    """
    B, H, W = naip.shape
    K = feats.shape[0]
    R = radius
    X = np.concatenate([np.ones((1, H, W), np.float32), feats.astype(np.float32)], 0).astype(np.float64)
    K1 = K + 1
    Y = naip.astype(np.float64)
    valid = np.isfinite(Y).all(0)
    Y = np.where(valid, Y, 0.0)
    w = valid.astype(np.float64) if weight is None else (weight * valid).astype(np.float64)
    wc = _crop(w, R)
    n = wc.sum()
    if n < 100:
        raise ValueError("too few valid pixels")
    # standardise bands on the weighted window
    Yc = _crop(Y, R)
    mu = (Yc * wc).sum((1, 2)) / n
    sd = np.sqrt((((Yc - mu[:, None, None]) ** 2) * wc).sum((1, 2)) / n)
    Yc = (Yc - mu[:, None, None]) / np.where(sd > 0, sd, 1)[:, None, None]
    N = 2 * R + 1
    cor = lambda a, b: signal.correlate(a, b, mode="valid", method="fft")  # noqa: E731
    G = np.zeros((N, N, K1, K1))
    for k in range(K1):
        for l in range(k, K1):
            g = cor(X[k] * X[l], wc)
            G[:, :, k, l] = g
            G[:, :, l, k] = g
    # ridge relative to the diagonal scale (not on the intercept)
    diag = np.einsum("abkk->abk", G)
    G = G + ridge * n * np.einsum("abk,kl->abkl", np.ones_like(diag), np.eye(K1)) * np.r_[0.0, np.ones(K)][None, None, :, None]
    explained = np.zeros((N, N))
    coefs = []
    for b in range(B):
        c = np.stack([cor(X[k], Yc[b] * wc) for k in range(K1)], -1)  # [N, N, K1]
        sol = np.linalg.solve(G, c[..., None])[..., 0]
        explained += (sol * c).sum(-1) / n  # = c^T G^-1 c / n  (band variance is 1)
        if return_coef:
            coefs.append(sol)
    r2 = explained / B
    res_ = _finish(r2, R, res, "regression", int(n))
    if return_coef:
        res_.coef = np.stack(coefs)
    return res_


# ----------------------------------------------------------------------------- block-wise field


def block_estimates(naip, feats, block=256, radius=8, res=1.0, min_valid=0.5, estimator=None, weight=None):
    """Independent shift estimates on a regular grid of blocks (each block is `block` px, searched +/- `radius` px).

    Returns a list of dicts (row, col centre in px; dx, dy, score, surface). `weight` [H, W] (optional) is passed to the estimator per block. Blocks are non-overlapping apart from the search margin.
    """
    H, W = naip.shape[1:]
    est = estimator or (lambda n, x, w=None: regression_search(n, x, radius, res, weight=w))
    out = []
    size = block + 2 * radius
    for r0 in range(0, H - size + 1, block):
        for c0 in range(0, W - size + 1, block):
            n = naip[:, r0 : r0 + size, c0 : c0 + size]
            x = feats[:, r0 : r0 + size, c0 : c0 + size]
            if np.isfinite(n).all(0).mean() < min_valid:
                continue
            try:
                r = est(n, x) if weight is None else est(n, x, weight[r0 : r0 + size, c0 : c0 + size])
            except ValueError:
                continue
            out.append(dict(row=r0 + size / 2, col=c0 + size / 2, dx=r.dx, dy=r.dy, score=r.score, surface=r.surface))
    return out


def pool_surfaces(blocks, radius, res=1.0, n_boot=200, seed=0):
    """Combine block score surfaces into one estimate: peak of the mean surface (every block votes with its whole
    score surface, so blocks with weak/ambiguous evidence cannot pull the answer to a random peak).

    Returns (ShiftResult, sd_xy): sd from a bootstrap over blocks (a reproducibility, not an accuracy, measure).
    """
    S = np.stack([b["surface"] for b in blocks])
    mean = S.mean(0)
    pooled = _finish(mean, radius, res, "pooled", len(blocks))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        m = S[rng.integers(0, len(S), len(S))].mean(0)
        r = _finish(m, radius, res, "boot", len(S))
        bs.append((r.dx, r.dy))
    return pooled, np.std(bs, 0)
