"""Evaluation of coregistration estimates (all values in metres).

No independent ground truth is available at scale, so we use several complementary checks:

* block reproducibility: spread of independent block estimates around the pooled value (precision)
* vintage closure: lidar-derived shifts of two NAIP vintages must differ by the directly measured NAIP-vs-NAIP shift
  (compares two same-modality images, free of lidar/NAIP cross-modal bias; sensitive to vintage-specific errors)
* known-shift recovery: displace the lidar by a known sub-pixel offset and recover it (estimator noise floor)
"""

import itertools

import numpy as np


def closure_errors(shifts, pair_shifts):
    """shifts: {vintage: (dx, dy)} NAIP-vs-lidar. pair_shifts: {(a, b): (dx, dy)} displacement of NAIP a relative to NAIP b.

    Returns (errors [n_pairs, 2] = (s_a - s_b) - m_ab, labels).
    """
    errs, labels = [], []
    for (a, b), m in pair_shifts.items():
        if a in shifts and b in shifts:
            e = np.array(shifts[a]) - np.array(shifts[b]) - np.array(m)
            errs.append(e)
            labels.append((a, b))
    return np.array(errs), labels


def rms(errs):
    errs = np.asarray(errs)
    return float(np.sqrt((errs**2).sum(1).mean())) if len(errs) else float("nan")


def all_pairs(keys):
    return list(itertools.combinations(sorted(keys), 2))
