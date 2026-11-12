"""Coverage check of cached cookies: does the analysis disc actually contain lidar data?

A site can lie inside a 3DEP tile's footprint polygon but outside the point cloud (collection edges: the footprint is
coarser than the data), so its cookie holds points on one side only, or none within the analysis disc. Such
cookies give misleading features. For each cookie this computes, for the disc of `radius` m (noise classes dropped):

    n_disc          points in the disc
    expected_min    3DEP minimum aggregate nominal pulse density of the tile's quality level x disc area
                    (QL0/QL1 8, QL2 2, QL3 0.5 pulses/m2; unknown QL: 0.5, the most lenient). Pulses <= points, so
                    a fully covered disc has at least this many points.
    density_ratio   n_disc / expected_min
    cell_coverage   share of the disc's 5 m cells that hold at least one point (independent of density)
    low_points      density_ratio < 0.5 (user rule, 2026-11-12)

Caveat: open water returns few or no points, so water sites are flagged although their cookie is not wrong;
decide per task whether to drop flagged cookies (they are flagged, not removed).
"""

import numpy as np

from .fetch import DROP_CLASSES

QL_MIN_DENSITY = {"QL 0": 8.0, "QL 1": 8.0, "QL 2": 2.0, "QL 3": 0.5}  # 3DEP Lidar Base Specification, pulses/m2
UNKNOWN_QL_DENSITY = 0.5
LOW_RATIO = 0.5
CELL = 5.0


def disc_qc(cookie, ql, radius=50.0):
    """QC record (dict) of one cookie's analysis disc; `ql` is the tile's quality level ("QL 2", NaN, ...)."""
    xyz = cookie["xyz"]
    keep = (xyz[:, 0] ** 2 + xyz[:, 1] ** 2) <= radius**2
    if "classification" in cookie:
        keep &= ~np.isin(cookie["classification"], DROP_CLASSES)
    xy = xyz[keep, :2]
    density = QL_MIN_DENSITY.get(ql, UNKNOWN_QL_DENSITY)
    expected = density * np.pi * radius**2
    # 5 m cells whose centre lies in the disc, and how many of them hold a point
    c = np.arange(-radius + CELL / 2, radius, CELL)
    gx, gy = np.meshgrid(c, c)
    inside = gx**2 + gy**2 <= radius**2
    occupied = np.zeros_like(inside)
    if len(xy):
        ij = np.clip(((xy + radius) // CELL).astype(int), 0, len(c) - 1)
        occupied[ij[:, 1], ij[:, 0]] = True
    return dict(n_disc=int(len(xy)), ql=ql if isinstance(ql, str) else None, expected_min=float(expected),
                density_ratio=float(len(xy) / expected), cell_coverage=float((occupied & inside).sum() / inside.sum()),
                low_points=bool(len(xy) / expected < LOW_RATIO))
