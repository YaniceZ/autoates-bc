"""One AutoATES run: PRA -> Flow-Py -> classifier, all in memory."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.fill import fillnodata

from . import classifier as C
from . import flowpy as F
from . import pra as P

NODATA = -9999


@dataclass
class Grid:
    dem: np.ndarray        # float32, gaps filled
    cell_size: float
    transform: object
    crs: object
    valid: np.ndarray      # False where the source DEM had no data


def load_dem(path) -> Grid:
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        valid = dem != src.nodata
        if not valid.all():
            # small lidar gaps (water, data voids): interpolate so that slope and
            # the flow model do not see -9999 cliffs; `valid` keeps track of them
            dem = fillnodata(dem, mask=valid, max_search_distance=50)
        return Grid(dem, float(src.transform[0]), src.transform, src.crs, valid)


def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)


def run(dem, pcc, cell_size, params: C.Params | None = None, alpha=24.0, exp=8,
        flux=0.003, max_z=270.0, pra_thd=0.15, ws=None, return_layers=False):
    """Run AutoATES v2.0 with a percent-canopy-cover forest layer.

    dem : float array without nodata
    pcc : canopy cover in percent (0-100)
    Returns the ATES raster as int8 with 0 = non-avalanche terrain, 1-4 = simple
    to extreme. Flow-Py and PRA settings default to the published AutoATES values.
    """
    p = params or C.Params.for_forest("pcc")
    pcc = np.clip(np.asarray(pcc), 0, 100).astype(np.int16)
    radius = max(1, int(round(60.0 / cell_size)))
    if ws is None:
        ws = P.windshelter(dem, cell_size, radius=radius)
    pra_cont, pra_bin = P.pra(dem, cell_size, pcc, "pcc", radius=radius,
                              pra_thd=pra_thd, ws=ws)
    fp = F.flowpy(dem, pra_bin, cell_size, forest=pcc / 100.0, alpha=alpha, exp=exp,
                  flux=flux, max_z=max_z, nodata=None)
    overhead = C.overhead_exposure(fp["z_delta"], fp["cell_counts"], max_z_delta=max_z)
    slope = C.horn_slope(dem, cell_size, cell_size, nodata=None)
    ates = C.classify(slope, fp["fp_travel_angle"].astype(np.int16), overhead, pcc,
                      pra_bin, cell_size * cell_size, p)
    ates = np.where(ates == NODATA, 0, ates)
    ates = np.clip(ates, 0, 4).astype(np.int8)
    if return_layers:
        return ates, {"slope": slope, "windshelter": ws, "pra": pra_cont,
                      "pra_binary": pra_bin, "overhead": overhead, **fp}
    return ates


def write_like(path, arr, grid: Grid, nodata=None):
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype=arr.dtype, crs=grid.crs, transform=grid.transform,
                   compress="deflate", tiled=True)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
