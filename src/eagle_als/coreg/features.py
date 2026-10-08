"""Lidar-derived feature stacks and NAIP-derived images used by the estimators."""

import numpy as np
from scipy import ndimage

from .register import fill_nan, standardize


def ndvi(naip):
    r, n = naip[0], naip[3]
    return (n - r) / np.maximum(n + r, 1e-3)


def luminance(naip):
    return naip[:3].mean(0)


def grad_mag(a, sigma=1.0):
    a = fill_nan(a)
    gy, gx = np.gradient(ndimage.gaussian_filter(a, sigma))
    return np.hypot(gx, gy)


def lidar_stack(lr, kinds=("chm", "intensity", "slope"), sigma=0.7):
    """Stack of NaN-free float32 feature rasters [K, H, W], each standardised.

    chm        canopy height, its square (leaves saturate / shadows grow with height) and a 'ground' indicator
    intensity  first-return 1064 nm intensity (comparable to NAIP NIR)
    slope      x/y derivatives of the lightly smoothed DSM (a linear model of these spans any sun azimuth hillshade)
    edges      gradient magnitude of the CHM
    """
    out = []
    chm = fill_nan(lr.chm)
    dsm = fill_nan(lr.dsm)
    if "chm" in kinds:
        c = np.clip(chm, 0, 60)
        out += [c, c**2 / 30.0, np.log1p(c), (c < 0.5).astype(np.float32)]
    if "intensity" in kinds:
        out.append(np.clip(fill_nan(lr.intensity), 0, 4))
    if "slope" in kinds:
        d = ndimage.gaussian_filter(dsm, sigma)
        gy, gx = np.gradient(d)
        out += [np.clip(gx, -3, 3), np.clip(gy, -3, 3)]
    if "edges" in kinds:
        out.append(grad_mag(chm, 1.0))
    return np.stack([standardize(o.astype(np.float32)) for o in out]).astype(np.float32)


def lidar_edge_stack(lr, sigma=1.0):
    """Edge-domain lidar features: gradient magnitudes of CHM, first-return intensity and DSM, plus slope magnitude of the DTM.

    Edge maps are largely insensitive to the sign/size of radiometric contrast between sensors and seasons, which is
    what makes them usable on bare-soil-lidar vs summer-crop-NAIP cropland where raw values are uncorrelated.
    """
    chm = np.clip(fill_nan(lr.chm), 0, 60)
    layers = [grad_mag(chm, sigma), grad_mag(np.clip(fill_nan(lr.intensity), 0, 4), sigma), grad_mag(fill_nan(lr.dsm), sigma)]
    layers.append(grad_mag(fill_nan(lr.dtm), 2.0))
    layers = [np.clip(a, 0, np.nanpercentile(a, 99.5)) for a in layers]
    return np.stack([standardize(a.astype(np.float32)) for a in layers]).astype(np.float32)


def naip_edge_stack(naip, sigma=1.0):
    """Gradient magnitudes of NAIP NIR, luminance and NDVI (shadow-robust), clipped at the 99.5th percentile."""
    layers = [grad_mag(naip[3], sigma), grad_mag(luminance(naip), sigma), grad_mag(ndvi(naip), sigma)]
    return np.stack([np.clip(a, 0, np.nanpercentile(a, 99.5)) for a in layers]).astype(np.float32)


def shadow_depth(dsm, az_deg, el_deg, res=1.0, max_h=60.0):
    """Depth (m) by which each cell lies below the sun ray grazing the surrounding DSM (0 = sunlit).

    az_deg is the compass azimuth of the sun (0 = N, 90 = E); shadows fall the opposite way. Brute-force ray march of the
    DSM raster in steps of one pixel, so cost ~ (max_h / tan(el)) raster shifts.
    """
    dsm = fill_nan(dsm).astype(np.float32)
    t = np.tan(np.radians(el_deg))
    ux, uy = np.sin(np.radians(az_deg)), np.cos(np.radians(az_deg))  # toward the sun, east / north
    best = np.full_like(dsm, -np.inf)
    n_steps = int(min(max_h / t, 150.0) / res)
    for k in range(1, n_steps + 1):
        dist = k * res
        # value of DSM at p + dist*u, expressed at p: rows go south so north = -rows
        q = ndimage.shift(dsm, (uy * dist / res, -ux * dist / res), order=1, mode="nearest")
        best = np.maximum(best, q - t * dist)
    return np.clip(best - dsm, 0, None)


def shadow_features(dsm, az_deg, el_deg, res=1.0):
    """Shadow indicator and a soft (saturating) depth, both standardised-ready."""
    s = shadow_depth(dsm, az_deg, el_deg, res)
    return [(s > 0.3).astype(np.float32), np.minimum(s, 5.0)]
