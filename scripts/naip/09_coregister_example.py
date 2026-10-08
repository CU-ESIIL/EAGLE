"""End-to-end example: coregister one NAIP vintage to 3DEP lidar around a point, write the corrected GeoTIFF and a figure.

Usage: python scripts/naip/09_coregister_example.py LON LAT SIZE_M NAIP_YEAR OUT_DIR [--field] [--jobs N]
  SIZE_M <= ~2000 m: one window (api.estimate; ground shift + lean).  --field (use for >= ~3 km): api.estimate_field.
Outputs in OUT_DIR: result.json, naip_coreg.tif (4 bands, lidar-frame grid, relief-corrected), before_after.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eagle_als.coreg import api, data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lon", type=float)
    ap.add_argument("lat", type=float)
    ap.add_argument("size", type=float)
    ap.add_argument("year", type=int)
    ap.add_argument("out")
    ap.add_argument("--field", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = D.Grid.from_center(a.lon, a.lat, a.size, res=1.0)
    surveys = D.search_naip(grid, years=[a.year])
    key, items = next(iter(surveys.items()))
    print("NAIP survey", key, [i.id for i in items])
    naip = D.read_naip(items, grid)
    tile = D.tile_for(a.lon, a.lat).iloc[0]
    print("3DEP project", tile.name, int(tile.collection_year))
    lr = D.lidar_rasters_tiled(tile.url, grid) if a.size > 2000 else D.rasterize_lidar(D.read_lidar_points(tile.url, grid), grid)

    if a.field:
        res = api.estimate_field(naip, lr, n_jobs=a.jobs)
        corrected = api.apply_field(naip, lr, res)
        print(f"model={res.model} ground shift at centre ({res.dx:+.2f}, {res.dy:+.2f}) m, CV R2 {res.cv_r2}")
    else:
        res = api.estimate(naip, lr, n_jobs=a.jobs)
        corrected = api.apply(naip, lr, res)
        print(f"ground shift ({res.dx:+.2f}, {res.dy:+.2f}) m  lean ({res.tx:+.3f}, {res.ty:+.3f}) /m  sd {res.sd:.2f} m  sun {res.sun_az}/{res.sun_el}")
    (out / "result.json").write_text(json.dumps(dict(naip=key, lidar=tile.name, **res.asdict()), indent=1, default=float))

    import rasterio

    with rasterio.open(out / "naip_coreg.tif", "w", driver="GTiff", height=grid.height, width=grid.width, count=4, dtype="float32",
                       crs=grid.crs, transform=grid.transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(corrected.astype("float32"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eagle_als.coreg import features as F

    h, w = grid.height, grid.width
    sl = (slice(h // 2 - 250, h // 2 + 250), slice(w // 2 - 250, w // 2 + 250))
    edge = F.grad_mag(lr.chm, 1.0)[sl] > 1.0
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    for k, (img, title) in enumerate([(naip, "supplied coordinates"), (corrected, "coregistered")]):
        rgb = np.clip(np.moveaxis(img[:3][:, sl[0], sl[1]], 0, -1) / 255.0, 0, 1)
        ax[k].imshow(np.nan_to_num(rgb))
        ov = np.zeros(edge.shape + (4,))
        ov[edge] = (1, 0.1, 0.1, 0.55)  # lidar CHM edges (>1 m height step)
        ax[k].imshow(ov)
        ax[k].set_title(f"NAIP {key}: {title}\nred = lidar canopy/building edges")
        ax[k].axis("off")
    plt.tight_layout()
    plt.savefig(out / "before_after.png", dpi=110)


if __name__ == "__main__":
    main()
