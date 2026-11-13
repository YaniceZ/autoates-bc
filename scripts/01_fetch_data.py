"""Download the open input data for the study area and put it on a common grid.

Sources (all open, no login):
  * NRCan HRDEM Mosaic 2 m, lidar DTM and DSM (CanElevation, AWS open data)
  * Copernicus GLO-30 DSM (AWS open data)
  * BC Vegetation Resources Inventory (VRI) polygons, crown closure (BC WFS)

Outputs in data/processed (EPSG:3157, NAD83(CSRS) / UTM 10N):
  dem_lidar_10m.tif    lidar DTM averaged to 10 m
  pcc_lidar_10m.tif    canopy cover %, share of 2 m cells with DSM-DTM > 2 m
                       (plus a slope allowance, see below)
  pcc_vri_10m.tif      VRI crown closure % rasterised to 10 m
  dem_cop30_30m.tif    Copernicus GLO-30 warped to 30 m
  pcc_lidar_30m.tif, pcc_vri_30m.tif   the same forest layers at 30 m
"""
import json
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CRS = "EPSG:3157"

# 10.5 x 8.4 km block east of the Mamquam Icefield, Garibaldi Provincial Park
# (Coast Mountains, about 30 km north-east of Squamish). Chosen because the
# public lidar mosaic covers it completely. Bounds are in EPSG:3157, multiples
# of 30 m so that the 10 m and 30 m grids nest.
BOUNDS = (510600.0, 5510700.0, 521100.0, 5519100.0)

HRDEM = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/hrdem-mosaic-2m/2_4-mosaic-2m-{}.tif"
COP30 = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N49_00_W123_00_DEM/"
         "Copernicus_DSM_COG_10_N49_00_W123_00_DEM.tif")
VRI_WFS = "https://openmaps.gov.bc.ca/geo/pub/WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY/ows"


def utm_bounds():
    return BOUNDS


def write(path, arr, transform, nodata=-9999):
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype=arr.dtype, crs=CRS, transform=transform, nodata=nodata,
                   compress="deflate", tiled=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def warp(url, bounds, res, resampling):
    w, s, e, n = bounds
    width, height = int(round((e - w) / res)), int(round((n - s) / res))
    transform = from_origin(w, n, res, res)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5",
                      GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(url) as src:
            with WarpedVRT(src, crs=CRS, transform=transform, width=width, height=height,
                           resampling=resampling) as vrt:
                arr = vrt.read(1).astype(np.float32)
            nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -1000] = np.nan
    return arr, transform


def block_mean(a, k):
    h, w = a.shape[0] // k, a.shape[1] // k
    return np.nanmean(a[: h * k, : w * k].reshape(h, k, w, k), axis=(1, 3))


def fetch_vri(bounds):
    path = RAW / "vri.geojson"
    if not path.exists():
        w, s, e, n = bounds
        params = dict(service="WFS", version="2.0.0", request="GetFeature",
                      typeName="pub:WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY",
                      outputFormat="json", srsName=CRS,
                      propertyName="CROWN_CLOSURE,BCLCS_LEVEL_1,BCLCS_LEVEL_2,BCLCS_LEVEL_4,"
                                   "PROJ_HEIGHT_1,PROJECTED_DATE,REFERENCE_YEAR,GEOMETRY",
                      bbox=f"{w},{s},{e},{n},urn:ogc:def:crs:{CRS}")
        r = requests.get(VRI_WFS, params=params, timeout=300)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf8")
    return json.loads(path.read_text(encoding="utf8"))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = utm_bounds()
    w, s, e, n = bounds
    print("UTM bounds", bounds, "size km", (e - w) / 1000, (n - s) / 1000)

    # --- lidar DTM / DSM at 2 m
    layers = {}
    for kind in ("dtm", "dsm"):
        p = RAW / f"hrdem_{kind}_2m.tif"
        if not p.exists():
            arr, tr = warp(HRDEM.format(kind), bounds, 2.0, Resampling.bilinear)
            write(p, np.where(np.isnan(arr), -9999, arr).astype(np.float32), tr)
        with rasterio.open(p) as src:
            a = src.read(1)
            layers[kind] = np.where(a == -9999, np.nan, a)
        print(kind, "valid", np.isfinite(layers[kind]).mean())
    dtm, dsm = layers["dtm"], layers["dsm"]

    chm = dsm - dtm
    # A 2 m DSM cell keeps the highest return, so on a slope it sits above the
    # DTM by up to (cell size x tan slope) with no vegetation at all. Without
    # this allowance cliffs above treeline come out as one third "forest".
    gy, gx = np.gradient(np.where(np.isfinite(dtm), dtm, np.nanmean(dtm)), 2.0)
    relief = 2.0 * np.sqrt(gx ** 2 + gy ** 2)
    canopy = np.where(np.isfinite(chm), (chm > 2.0 + relief).astype(np.float32), np.nan)
    for res in (10, 30):
        k = res // 2
        tr = from_origin(w, n, res, res)
        pcc = block_mean(canopy, k) * 100
        write(OUT / f"pcc_lidar_{res}m.tif",
              np.where(np.isnan(pcc), -9999, np.round(pcc)).astype(np.int16), tr)
        if res == 10:
            d = block_mean(dtm, k)
            write(OUT / "dem_lidar_10m.tif",
                  np.where(np.isnan(d), -9999, d).astype(np.float32), tr)
            write(OUT / "chm_max_10m.tif",
                  np.where(np.isnan(d), -9999,
                           np.nanmax(np.where(np.isnan(chm), -1, chm)[: d.shape[0] * k, : d.shape[1] * k]
                                     .reshape(d.shape[0], k, d.shape[1], k), axis=(1, 3))).astype(np.float32), tr)

    # --- Copernicus GLO-30
    cop, tr30 = warp(COP30, bounds, 30.0, Resampling.bilinear)
    write(OUT / "dem_cop30_30m.tif", np.where(np.isnan(cop), -9999, cop).astype(np.float32), tr30)
    print("cop30", cop.shape, np.nanmin(cop), np.nanmax(cop))

    # --- VRI crown closure
    vri = fetch_vri(bounds)
    feats = vri["features"]
    print("VRI polygons", len(feats))
    shapes = []
    for f in feats:
        cc = f["properties"].get("CROWN_CLOSURE")
        shapes.append((f["geometry"], int(cc) if cc is not None else 0))
    for res in (10, 30):
        tr = from_origin(w, n, res, res)
        shape = (int(round((n - s) / res)), int(round((e - w) / res)))
        r = rasterize(shapes, out_shape=shape, transform=tr, fill=-9999, dtype="int16")
        write(OUT / f"pcc_vri_{res}m.tif", r, tr)
        print("VRI raster", res, "covered", (r >= 0).mean(), "mean cc", r[r >= 0].mean())


if __name__ == "__main__":
    main()
