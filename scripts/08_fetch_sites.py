"""Download inputs for every block in autoates_bc.sites into data/sites/<id>/.

Same layers as 01_fetch_data.py, plus SCANFI crown closure (NRCan, 30 m, 2020),
which exists for all of forested Canada and is tested as a replacement for VRI.
Skips blocks that are already complete.
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin

from autoates_bc.sites import SITES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch", ROOT / "scripts" / "01_fetch_data.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

SCANFI = ("https://ftp.maps.canada.ca/pub/nrcan_rncan/Forests_Foret/SCANFI/v1/"
          "SCANFI_att_closure_SW_2020_v1.2.tif")
COP = "https://copernicus-dem-30m.s3.amazonaws.com/{0}/{0}.tif"
DONE = ["dem_lidar_10m.tif", "pcc_lidar_10m.tif", "pcc_lidar_30m.tif", "pcc_vri_10m.tif",
        "pcc_vri_30m.tif", "pcc_scanfi_30m.tif", "dem_cop30_30m.tif"]


def cop_tiles(bounds):
    t = Transformer.from_crs(fetch.CRS, "EPSG:4326", always_xy=True)
    w, s, e, n = bounds
    lons, lats = t.transform([w, e, w, e], [s, s, n, n])
    out = []
    for lat in range(math.floor(min(lats)), math.floor(max(lats)) + 1):
        for lon in range(math.floor(min(lons)), math.floor(max(lons)) + 1):
            out.append(f"Copernicus_DSM_COG_10_N{lat:02d}_00_W{-lon:03d}_00_DEM")
    return out


def main():
    for sid, (bounds, label, _) in SITES.items():
        out = ROOT / "data" / "sites" / sid
        out.mkdir(parents=True, exist_ok=True)
        if all((out / f).exists() for f in DONE):
            print("skip", sid, flush=True)
            continue
        print("==", sid, label, flush=True)
        w, s, e, n = bounds
        layers = {}
        for kind in ("dtm", "dsm"):
            arr, _ = fetch.warp(fetch.HRDEM.format(kind), bounds, 2.0, Resampling.bilinear)
            layers[kind] = arr
        dtm, dsm = layers["dtm"], layers["dsm"]
        print("  lidar valid", round(float(np.isfinite(dtm).mean()), 4), flush=True)
        chm = dsm - dtm
        gy, gx = np.gradient(np.where(np.isfinite(dtm), dtm, np.nanmean(dtm)), 2.0)
        relief = 2.0 * np.sqrt(gx ** 2 + gy ** 2)
        canopy = np.where(np.isfinite(chm), (chm > 2.0 + relief).astype(np.float32), np.nan)
        for res in (10, 30):
            tr = from_origin(w, n, res, res)
            pcc = fetch.block_mean(canopy, res // 2) * 100
            fetch.write(out / f"pcc_lidar_{res}m.tif",
                        np.where(np.isnan(pcc), -9999, np.round(pcc)).astype(np.int16), tr)
        d = fetch.block_mean(dtm, 5)
        fetch.write(out / "dem_lidar_10m.tif", np.where(np.isnan(d), -9999, d).astype(np.float32),
                    from_origin(w, n, 10, 10))

        # a block can straddle two 1-degree Copernicus tiles; outside its own
        # tile a warp returns 0, so each cell is taken from the tile it lies in
        t = Transformer.from_crs(fetch.CRS, "EPSG:4326", always_xy=True)
        xs = w + 30.0 * (np.arange(int(round((e - w) / 30))) + 0.5)
        ys = n - 30.0 * (np.arange(int(round((n - s) / 30))) + 0.5)
        lon, lat = t.transform(*np.meshgrid(xs, ys))
        cop = np.full(lon.shape, np.nan, np.float32)
        for tile in cop_tiles(bounds):
            a, tr30 = fetch.warp(COP.format(tile), bounds, 30.0, Resampling.bilinear)
            tlat, tlon = int(tile.split("_N")[1][:2]), -int(tile.split("_W")[1][:3])
            # half a cell of slack so bilinear edges do not leave a seam
            own = (lon >= tlon) & (lon < tlon + 1) & (lat >= tlat) & (lat < tlat + 1)
            # within about 100 m of a tile edge the bilinear warp blends in the
            # zeros beyond the tile; leave those cells empty and fill them below
            edge = np.minimum.reduce([np.abs(lon - tlon), np.abs(lon - tlon - 1)]) < 1.5e-3
            edge |= np.minimum.reduce([np.abs(lat - tlat), np.abs(lat - tlat - 1)]) < 1.0e-3
            cop = np.where(own & ~edge & np.isnan(cop), a, cop)
        if np.isnan(cop).any():
            from rasterio.fill import fillnodata
            cop = fillnodata(np.nan_to_num(cop, nan=0.0), mask=np.isfinite(cop), max_search_distance=10)
        fetch.write(out / "dem_cop30_30m.tif", np.where(np.isnan(cop), -9999, cop).astype(np.float32), tr30)
        print("  cop30 valid", round(float(np.isfinite(cop).mean()), 4), flush=True)

        sc, _ = fetch.warp(SCANFI, bounds, 30.0, Resampling.bilinear)
        sc = np.where(np.isfinite(sc) & (sc <= 100), sc, 0)   # 255 = no data: rock, ice, water
        fetch.write(out / "pcc_scanfi_30m.tif", np.round(sc).astype(np.int16), tr30)

        params = dict(service="WFS", version="2.0.0", request="GetFeature",
                      typeName="pub:WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY", outputFormat="json",
                      srsName=fetch.CRS, propertyName="CROWN_CLOSURE,GEOMETRY",
                      bbox=f"{w},{s},{e},{n},urn:ogc:def:crs:{fetch.CRS}")
        r = requests.get(fetch.VRI_WFS, params=params, timeout=600)
        r.raise_for_status()
        feats = json.loads(r.text)["features"]
        shapes = [(f["geometry"], int(f["properties"].get("CROWN_CLOSURE") or 0)) for f in feats]
        for res in (10, 30):
            tr = from_origin(w, n, res, res)
            shape = (int(round((n - s) / res)), int(round((e - w) / res)))
            ras = rasterize(shapes, out_shape=shape, transform=tr, fill=-9999, dtype="int16")
            fetch.write(out / f"pcc_vri_{res}m.tif", ras, tr)
        print("  VRI polygons", len(feats), "covered", round(float((ras >= 0).mean()), 4),
              "| mean cover lidar/scanfi/vri:", round(float(np.nanmean(canopy) * 100), 1),
              round(float(sc.mean()), 1), round(float(ras[ras >= 0].mean()), 1), flush=True)
    print("done")


if __name__ == "__main__":
    main()
