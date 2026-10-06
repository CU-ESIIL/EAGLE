"""False-color mappings that blend per-point lidar metadata into one RGB color.

Inputs per point:
    hag          height above ground (m)
    intensity    return intensity (any scale; rescaled per tile)
    return_type  int code from `return_type_codes`: ONLY, FIRST, INTERMEDIATE, LAST

Usage:
    rt = return_type_codes(points['ReturnNumber'], points['NumberOfReturns'])
    rgb = false_color(hag, intensity, rt, mode='hsv_return_hue')   # (N, 3) uint8
    colors = to_plotly_colors(rgb)                                 # list of 'rgb(r,g,b)'

`MODES` lists the variants; `DESCRIPTIONS[mode]` says what each channel means.
"""
import numpy as np

# return type codes
ONLY, FIRST, INTERMEDIATE, LAST = 0, 1, 2, 3
RETURN_NAMES = {ONLY: 'only', FIRST: 'first of many', INTERMEDIATE: 'intermediate', LAST: 'last of many'}

# Return type -> hue (0-1). Spread around the wheel so the four types are easy to tell apart.
RETURN_HUE = {ONLY: 0.58, FIRST: 0.33, INTERMEDIATE: 0.13, LAST: 0.88}   # blue, green, orange-yellow, magenta
# Return type -> categorical RGB (0-1), Dark2-like
RETURN_RGB = {ONLY: (0.40, 0.40, 0.40), FIRST: (0.11, 0.62, 0.47), INTERMEDIATE: (0.85, 0.37, 0.01), LAST: (0.46, 0.44, 0.70)}
# Return type -> how deep into the canopy / how "solid" the target is (0-1), used where a scalar is needed.
# Both ONLY and LAST returns come off solid surfaces (ground, roofs); FIRST is the top of whatever was hit first.
RETURN_DEPTH = {ONLY: 0.85, FIRST: 0.0, INTERMEDIATE: 0.5, LAST: 1.0}
# Return type -> brightness multiplier for the variants that use it for value
RETURN_VALUE = {ONLY: 1.0, FIRST: 0.85, INTERMEDIATE: 0.65, LAST: 0.45}

# corner colors for the ternary blend (hag, intensity, return depth)
TERNARY_CORNERS = np.array([(0.11, 0.62, 0.47), (0.95, 0.55, 0.05), (0.45, 0.40, 0.85)])


def return_type_codes(return_number, number_of_returns):
    """Classify each point as ONLY / FIRST / INTERMEDIATE / LAST from LAS ReturnNumber and NumberOfReturns."""
    rn = np.asarray(return_number)
    nr = np.asarray(number_of_returns)
    out = np.full(rn.shape, INTERMEDIATE, dtype=np.int8)
    out[nr <= 1] = ONLY
    out[(nr > 1) & (rn == 1)] = FIRST
    out[(nr > 1) & (rn >= nr)] = LAST
    return out


def normalize_hag(hag, hag_max=None):
    """Height above ground -> 0-1. hag_max defaults to the 99th percentile; pass a shared value to compare tiles."""
    hag = np.clip(np.asarray(hag, dtype=float), 0, None)
    hag_max = hag_max or max(np.percentile(hag, 99), 1.0)
    return np.clip(hag / hag_max, 0, 1)


def normalize_intensity(intensity, lo_pct=2, hi_pct=98, gamma=1.0):
    """Intensity -> 0-1 using per-tile percentile clipping (sensors differ, so scale is per tile)."""
    i = np.asarray(intensity, dtype=float)
    lo, hi = np.percentile(i, [lo_pct, hi_pct])
    return np.clip((i - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma


def hsv_to_rgb(h, s, v):
    """Vectorized HSV -> RGB, all inputs 0-1, returns (N, 3) floats 0-1."""
    h, s, v = np.broadcast_arrays(np.asarray(h, float), np.asarray(s, float), np.asarray(v, float))
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _lookup(table, codes):
    """Map return-type codes through a {code: value} dict. Values may be scalars or tuples."""
    arr = np.array([table[k] for k in sorted(table)], dtype=float)
    return arr[np.asarray(codes)]


# ---- variants (each takes normalized h, i in 0-1 and integer return codes; returns (N, 3) floats 0-1) ----

def _hsv_return_hue(h, i, rt):
    # hue = return type, saturation = intensity, value = height above ground
    return hsv_to_rgb(_lookup(RETURN_HUE, rt), 0.25 + 0.75 * i, 0.25 + 0.75 * h)


def _hsv_height_hue(h, i, rt):
    # hue = height (blue low -> red high), saturation = intensity, value = return type (only brightest, last darkest)
    return hsv_to_rgb(0.66 * (1 - h), 0.25 + 0.75 * i, _lookup(RETURN_VALUE, rt))


def _rgb_channels(h, i, rt):
    # direct channel mapping: R = height, G = intensity, B = return depth
    return np.stack([h, i, _lookup(RETURN_DEPTH, rt)], axis=-1)


def _ternary(h, i, rt):
    # color triangle: corners are the three variables; a point's color is the blend of corners weighted by its values
    v = np.stack([h, i, _lookup(RETURN_DEPTH, rt)], axis=-1) + 0.05
    w = v ** 2   # sharpen so the dominant variable pulls the color toward its corner
    w = w / w.sum(axis=1, keepdims=True)
    brightness = 0.35 + 0.65 * v.max(axis=1, keepdims=True).clip(0, 1)   # keep weak points from all collapsing to the middle
    return (w @ TERNARY_CORNERS) * brightness


def _palette_shaded(h, i, rt):
    # categorical return-type color, darkened toward black for low points, washed toward grey for low intensity
    base = _lookup(RETURN_RGB, rt)
    grey = base.mean(axis=1, keepdims=True)
    sat = 0.3 + 0.7 * i[:, None]
    return (grey + (base - grey) * sat) * (0.5 + 0.5 * h[:, None])


MODES = {
    'hsv_return_hue': _hsv_return_hue,
    'hsv_height_hue': _hsv_height_hue,
    'rgb_channels': _rgb_channels,
    'ternary': _ternary,
    'palette_shaded': _palette_shaded,
}

DESCRIPTIONS = {
    'hsv_return_hue': 'hue = return type (only=blue, first=green, intermediate=yellow, last=magenta); saturation = intensity; value = height above ground',
    'hsv_height_hue': 'hue = height above ground (blue low to red high); saturation = intensity; value = return type (only brightest, last-of-many darkest)',
    'rgb_channels': 'red = height above ground; green = intensity; blue = return depth (first=0, intermediate=0.5, only=0.85, last=1)',
    'ternary': 'color triangle: corners are height (teal), intensity (orange), return depth (violet); a point blends the corners by its values',
    'palette_shaded': 'color = return type (only=grey, first=teal, intermediate=orange, last=violet); brightness = height above ground; vividness = intensity',
}


def false_color(hag, intensity, return_type, mode='hsv_return_hue', hag_max=None, intensity_gamma=1.0):
    """Blend height above ground, intensity and return type into (N, 3) uint8 RGB using the named variant."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {list(MODES)}")
    h = normalize_hag(hag, hag_max)
    i = normalize_intensity(intensity, gamma=intensity_gamma)
    rgb = MODES[mode](h, i, np.asarray(return_type))
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


def to_plotly_colors(rgb):
    """(N, 3) uint8 -> list of 'rgb(r,g,b)' strings usable as plotly marker colors."""
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]
