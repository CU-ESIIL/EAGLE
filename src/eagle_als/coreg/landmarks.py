"""Find building landmarks (roof footprints) in lidar rasters: compact, planar-roofed, rectangular, isolated from trees.

A landmark is used as an independent tie point: its displacement between lidar and NAIP is measured on a small chip around
it, and compared with what a model fitted on *other* parts of the scene predicts at that location.
"""

import numpy as np
from scipy import ndimage


def find_buildings(chm, dsm, min_area=40, max_area=500, min_h=2.5, max_h=15.0, fill=0.88, clear_m=10, rough_max=0.35):
    """Return list of dicts (row, col, area, height, angle, comp_slice) for isolated rectangular flat-ish roofs."""
    chm = np.nan_to_num(chm)
    mask = chm > min_h
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    objs = ndimage.find_objects(lab)
    tall = chm > 2.0
    out = []
    stats = find_buildings.stats = {}
    smooth = ndimage.gaussian_filter(np.nan_to_num(dsm), 1.0)
    for i, sl in enumerate(objs, 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        pad = clear_m + 3
        if sl[0].start < pad or sl[1].start < pad or sl[0].stop > chm.shape[0] - pad or sl[1].stop > chm.shape[1] - pad:
            continue
        stats['bbox'] = stats.get('bbox', 0) + 1
        if h * w < min_area or h > 60 or w > 60:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if not (min_area <= area <= max_area):
            stats['area'] = stats.get('area', 0) + 1
            continue
        hh = chm[sl][m]
        if hh.mean() > max_h or hh.std() > 1.5:
            stats['hstd'] = stats.get('hstd', 0) + 1
            continue
        rr, cc = np.nonzero(m)
        pts = np.c_[rr, cc].astype(float)
        pts -= pts.mean(0)
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        p = pts @ evec
        rect = (np.ptp(p[:, 0]) + 1) * (np.ptp(p[:, 1]) + 1)
        if area / rect < fill:
            stats['fill'] = stats.get('fill', 0) + 1
            continue
        # roof planarity: residual of a plane fit to the roof DSM
        zz = np.nan_to_num(dsm)[sl][m]
        A = np.c_[np.ones(area), rr, cc]
        c, *_ = np.linalg.lstsq(A, zz, rcond=None)
        if np.std(zz - A @ c) > rough_max:
            stats['rough'] = stats.get('rough', 0) + 1
            continue
        # isolation: no other tall pixels in the ring [3, clear_m] m around the footprint
        r0, r1 = max(sl[0].start - clear_m - 3, 0), sl[0].stop + clear_m + 3
        c0, c1 = max(sl[1].start - clear_m - 3, 0), sl[1].stop + clear_m + 3
        full = np.zeros((r1 - r0, c1 - c0), bool)
        full[sl[0].start - r0 : sl[0].stop - r0, sl[1].start - c0 : sl[1].stop - c0] = m
        near = ndimage.binary_dilation(full, iterations=clear_m)
        inner = ndimage.binary_dilation(full, iterations=3)
        ring = near & ~inner
        if tall[r0:r1, c0:c1][ring].any():
            stats['isolated'] = stats.get('isolated', 0) + 1
            continue
        out.append(dict(row=sl[0].start + rr.mean(), col=sl[1].start + cc.mean(), area=area, height=float(hh.mean()),
                        angle=float(np.degrees(np.arctan2(evec[0, 1], evec[1, 1]))), roof_resid=float(np.std(zz - A @ c))))
    return out


def candidates(chm, min_area=25, max_area=900, min_h=2.5, max_h=20.0, pad=15, min_fill=0.6):
    """Cheap first pass on the 1 m CHM: compact tall components. Returns list of (row, col, area, height)."""
    chm = np.nan_to_num(chm)
    lab, n = ndimage.label(chm > min_h)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl[0].start < pad or sl[1].start < pad or sl[0].stop > chm.shape[0] - pad or sl[1].stop > chm.shape[1] - pad:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh > 45 or ww > 45:
            continue
        m = lab[sl] == i
        a = int(m.sum())
        if not (min_area <= a <= max_area) or a / (hh * ww) < min_fill:
            continue
        h = float(chm[sl][m].mean())
        if h > max_h:
            continue
        rr, cc = np.nonzero(m)
        out.append((sl[0].start + rr.mean(), sl[1].start + cc.mean(), a, h))
    return out


def verify_chip(pts, grid, min_area=30, max_area=700, single_frac=0.9, fill=0.88, ring_clear=0.03):
    """Confirm a building in a small high-resolution lidar chip (grid res ~0.5 m, building near the centre).

    pts: structured array (X, Y, Z, ReturnNumber, NumberOfReturns, Classification). Returns dict with the roof footprint
    mask on `grid`, roof height above local ground, area, orientation, or None if it does not look like a building.
    A roof is: one compact component of high points, >= single_frac single-return, filling >= `fill` of its oriented rectangle,
    nearly planar (<= 2 planes), and with no other tall points within a 3 m ring (trees would show multi-returns there).
    """
    H, W, res = grid.height, grid.width, grid.res
    col = np.floor((pts["X"] - grid.x0) / res).astype(int)
    row = np.floor((grid.y0 - pts["Y"]) / res).astype(int)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    pts, row, col = pts[ok], row[ok], col[ok]
    g = pts["Classification"] == 2
    if g.sum() < 20:
        return None
    ground = np.median(pts["Z"][g])
    z = pts["Z"] - ground
    hi = z > 2.5
    if hi.sum() < 50:
        return None
    flat = row * W + col
    cnt = np.bincount(flat[hi], minlength=H * W).reshape(H, W)
    dsm = np.full(H * W, -np.inf)
    np.maximum.at(dsm, flat[hi], z[hi])
    dsm = dsm.reshape(H, W)
    mask = ndimage.binary_closing(cnt > 0, iterations=2)
    lab, n = ndimage.label(mask)
    c0 = lab[H // 2 - 6 : H // 2 + 6, W // 2 - 6 : W // 2 + 6]
    ids = np.unique(c0[c0 > 0])
    if len(ids) == 0:
        return None
    best = max(ids, key=lambda i: (lab == i).sum())
    m = ndimage.binary_fill_holes(lab == best)
    area = m.sum() * res**2
    if not (min_area <= area <= max_area):
        return None
    # single-return fraction of the points on the roof
    on = hi & m[row, col]
    single = (pts["NumberOfReturns"] == 1)[on].mean() if on.any() else 0
    if single < single_frac:
        return None
    rr, cc = np.nonzero(m)
    P = np.c_[rr, cc].astype(float)
    P -= P.mean(0)
    ev, evec = np.linalg.eigh(np.cov(P.T))
    p = P @ evec
    rect = (np.ptp(p[:, 0]) + 1) * (np.ptp(p[:, 1]) + 1)
    if m.sum() / rect < fill:
        return None
    zz = dsm[m]
    zz = zz[np.isfinite(zz)]
    if zz.size < 20 or np.std(zz) > 2.0:
        return None
    # isolation: other high points in a 3 m ring
    ring = ndimage.binary_dilation(m, iterations=int(3 / res)) & ~ndimage.binary_dilation(m, iterations=2)
    if (cnt[ring] > 0).mean() > ring_clear:
        return None
    return dict(mask=m, area=float(area), height=float(np.median(zz)), angle=float(np.degrees(np.arctan2(evec[0, 1], evec[1, 1]))), single=float(single), ground=float(ground))


def measure_displacement(naip_lum, chip_grid, roof_mask, search=6.0, step=0.25):
    """Displacement (dx east, dy north, m) of a NAIP roof relative to its lidar footprint, from roof-edge contrast.

    naip_lum: NAIP luminance on `chip_grid` (res ~0.5 m). roof_mask: lidar roof footprint on the same grid.
    Score(d) = mean gradient magnitude of the NAIP image sampled on the footprint outline shifted by d, plus the absolute
    inside-vs-ring brightness contrast of the shifted footprint. Returns dict(dx, dy, quality) where quality is the peak
    over the search-window mean of the score (>~1.5 = a clear, unambiguous outline).
    """
    res = chip_grid.res
    img = ndimage.gaussian_filter(np.nan_to_num(naip_lum), 0.7)
    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy)
    edge = roof_mask & ~ndimage.binary_erosion(roof_mask)
    er, ec = np.nonzero(edge)
    inside = ndimage.binary_erosion(roof_mask, iterations=2)
    ring = ndimage.binary_dilation(roof_mask, iterations=int(4 / res)) & ~ndimage.binary_dilation(roof_mask, iterations=int(1.5 / res))
    ir, ic = np.nonzero(inside)
    rr_, rc_ = np.nonzero(ring)

    def samp(a, rows, cols, dx, dy):
        return ndimage.map_coordinates(a, [rows - dy / res, cols + dx / res], order=1, mode="nearest")

    def score(dx, dy):
        e = samp(grad, er, ec, dx, dy).mean()
        c = abs(samp(img, ir, ic, dx, dy).mean() - samp(img, rr_, rc_, dx, dy).mean())
        return e, c

    ds = np.arange(-search, search + 1e-6, step)
    E = np.array([[score(dx, dy)[0] for dx in ds] for dy in ds])
    C = np.array([[score(dx, dy)[1] for dx in ds] for dy in ds])
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)  # noqa: E731
    S = z(E) + z(C)
    i, j = np.unravel_index(np.argmax(S), S.shape)
    # sub-step refinement (quadratic on 3x3)
    from .register import quad_peak

    si, sj = quad_peak(S, i, j, half=1)
    dy, dx = np.interp(si, np.arange(len(ds)), ds), np.interp(sj, np.arange(len(ds)), ds)
    # quality: peak vs best score outside a 2.5 m radius of the peak
    D2 = (ds[None, :] - dx) ** 2 + (ds[:, None] - dy) ** 2
    far = S[D2 > 2.5**2]
    q = float(S[i, j] - far.max()) if far.size else 0.0
    return dict(dx=float(dx), dy=float(dy), quality=q)
