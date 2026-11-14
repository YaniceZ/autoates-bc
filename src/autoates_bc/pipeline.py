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


def run(dem, forest, cell_size, params: C.Params | None = None, alpha=24.0, exp=8,
        flux=0.003, max_z=270.0, pra_thd=0.15, ws=None, return_layers=False,
        forest_type="pcc", forest_density=None, nodata=None, yres=None,
        clip_density=True):
    """Run AutoATES v2.0.

    dem : float array. Cells equal to ``nodata`` are outside the study area.
    forest : forest layer in the units of ``forest_type`` (``pcc`` percent canopy
        cover, ``bav`` basal area m2/ha, ``stems`` stems/ha).
    forest_density : forest density 0-1 for the Flow-Py detrainment module.
        Defaults to canopy cover / 100 for ``pcc``; required for other types.
    clip_density : set False to pass ``forest_density`` to Flow-Py untouched,
        including -9999 cells, as the published Bow Summit run did.
    Returns the ATES raster as int8 with 0 = non-avalanche terrain or outside,
    1-4 = simple to extreme. Flow-Py and PRA settings default to the published
    AutoATES values.
    """
    p = params or C.Params.for_forest(forest_type)
    if forest_type == "pcc":
        forest = np.clip(np.asarray(forest), 0, 100).astype(np.int16)
        if forest_density is None:
            forest_density = forest / 100.0
    elif forest_density is None:
        raise ValueError("forest_density is required unless forest_type is 'pcc'")
    pra_nodata = P.NODATA if nodata is None else nodata
    radius = max(1, int(round(60.0 / cell_size)))
    if ws is None:
        ws = P.windshelter(dem, cell_size, radius=radius, nodata=pra_nodata)
    pra_cont, pra_bin = P.pra(dem, cell_size, forest, forest_type, radius=radius,
                              pra_thd=pra_thd, ws=ws, nodata=pra_nodata)
    if nodata is not None:
        pra_bin = np.where(dem == nodata, 0, pra_bin).astype(np.int16)
    if clip_density:
        forest_density = np.clip(forest_density, 0, 1)
    fp = F.flowpy(dem, pra_bin, cell_size, forest=forest_density, alpha=alpha,
                  exp=exp, flux=flux, max_z=max_z, nodata=nodata)
    overhead = C.overhead_exposure(fp["z_delta"], fp["cell_counts"], max_z_delta=max_z)
    slope = C.horn_slope(dem, cell_size, yres or cell_size, nodata=nodata)
    ates = C.classify(slope, fp["fp_travel_angle"].astype(np.int16), overhead, forest,
                      pra_bin, cell_size * (yres or cell_size), p)
    ates = np.where(ates == NODATA, 0, ates)
    ates = np.clip(ates, 0, 4).astype(np.int8)
    if nodata is not None:
        ates[dem == nodata] = 0
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
