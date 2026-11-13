"""Potential release area (PRA) model.

Array-in / array-out port of ``PRA/PRA_AutoATES-v2.0.py`` from AutoATES v2.0
(Toft et al. 2024, GPL-3.0), which itself reimplements Veitinger et al. (2016)
and Sharp (2018). The arithmetic is kept identical to the published script,
including its quirks, so that the port can be checked against the rasters that
ship with the upstream repository (see ``tests/test_port_matches_upstream.py``).

Differences from upstream, all outside the arithmetic:
  * no files are written, so it can be called thousands of times in a
    Monte Carlo loop;
  * the wind-shelter moving window is compiled with numba;
  * GDAL's sieve filter is called through rasterio instead of ``osgeo``;
  * ``bav`` and ``sen2cc`` forest inputs are actually read (upstream only reads
    the forest raster for ``pcc`` and ``stems``, and fails for the other two).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from rasterio.features import sieve

NODATA = -9999

# Cauchy membership parameters (a, b, c) for the forest term, per forest input.
FOREST_CAUCHY = {
    "stems": (350.0, 2.0, -120.0),
    "bav": (20.0, 3.5, -10.0),
    "sen2cc": (50.0, 1.5, 0.0),
    "pcc": (40.0, 3.5, -15.0),
    "no_forest": (40.0, 3.5, -15.0),
}
SLOPE_CAUCHY = (11.0, 4.0, 43.0)
WINDSHELTER_CAUCHY = (3.0, 10.0, 3.0)


def sector_mask(shape, centre, radius, angle_range):
    """Boolean mask of a circular sector (upstream function, unchanged)."""
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)
    if tmax < tmin:
        tmax += 2 * np.pi
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin
    theta %= 2 * np.pi
    return (r2 <= radius * radius) * (theta <= (tmax - tmin))


@njit(parallel=True, cache=True)
def _windshelter(dem, dist, mask, radius, prob, nodata):
    """Quantile of arctan(dz / distance) inside the sector window."""
    h, w = dem.shape
    out = np.full((h, w), -9999.0)
    k = 2 * radius + 1
    for i in prange(radius, h - radius):
        buf = np.empty(k * k)
        for j in range(radius, w - radius):
            centre = dem[i, j]
            n = 0
            # upstream turns nodata and exact zeros into NaN; a NaN centre
            # makes the whole window NaN
            if centre != nodata and centre != 0.0:
                for a in range(k):
                    for b in range(k):
                        if not mask[a, b] or (a == radius and b == radius):
                            continue
                        v = dem[i - radius + a, j - radius + b]
                        if v == nodata or v == 0.0:
                            continue
                        buf[n] = np.arctan((v - centre) / dist[a, b])
                        n += 1
            if n == 0:
                continue  # NaN upstream, written as -9999
            vals = np.sort(buf[:n])
            pos = (n - 1) * prob
            lo = int(np.floor(pos))
            hi = min(lo + 1, n - 1)
            out[i, j] = vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return out


def windshelter(dem, cell_size, radius=6, prob=0.5, winddir=0, windtol=180, nodata=NODATA):
    """Wind-shelter index, float32, -9999 on the border and where undefined."""
    k = 2 * radius + 1
    xa, ya = np.mgrid[0:k, 0:k]
    dist = np.sqrt((xa - radius) ** 2 + (ya - radius) ** 2) * cell_size
    mask = sector_mask(dist.shape, (radius, radius), radius,
                       (winddir - windtol + 270, winddir + windtol + 270))
    mask[radius, radius] = True
    dist[radius, radius] = 1.0  # never used, avoids a divide by zero
    ws = _windshelter(np.ascontiguousarray(dem, dtype=np.float64), dist, mask,
                      radius, prob, float(nodata))
    return ws.astype(np.float32)  # upstream round-trips through a float32 GeoTIFF


def cauchy(x, a, b, c):
    return 1.0 / (1.0 + ((x - c) / a) ** (2 * b))


def pra(dem, cell_size, forest=None, forest_type="pcc", radius=6, prob=0.5,
        winddir=0, windtol=180, pra_thd=0.15, sf=3, ws=None,
        slope_cauchy=SLOPE_CAUCHY, forest_cauchy=None, nodata=NODATA):
    """Run the PRA model.

    Parameters
    ----------
    dem : 2-D array. Upstream expects int16 with -9999 nodata.
    forest : 2-D array in the units of ``forest_type`` (ignored for no_forest).
    ws : optional precomputed wind-shelter raster (it only depends on the DEM).

    Returns
    -------
    continuous : int16, 0-100 (upstream ``PRA_continous.tif``)
    binary : int16, 0/1 after the sieve filter (upstream ``PRA_binary.tif``)
    """
    dem = np.asarray(dem)
    slope_src = dem.copy()
    slope_src[slope_src < -100] = NODATA
    px, py = np.gradient(slope_src, cell_size)
    slope_deg = np.degrees(np.arctan(np.sqrt(px ** 2 + py ** 2)))

    if ws is None:
        ws = windshelter(dem, cell_size, radius, prob, winddir, windtol, nodata)

    slope_c = cauchy(slope_deg, *slope_cauchy)
    with np.errstate(over="ignore"):  # -9999 border cells
        ws_c = cauchy(ws, *WINDSHELTER_CAUCHY)

    if forest_type == "no_forest" or forest is None:
        forest_arr = np.where(dem > -100, 0, dem)
        forest_type = "no_forest"
    else:
        forest_arr = np.asarray(forest)
    a, b, c = forest_cauchy or FOREST_CAUCHY[forest_type]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        forest_c = cauchy(forest_arr, a, b, c)
    # nodata forest gives a value near zero ("dense forest"); upstream resets it
    forest_c[~(forest_c > 0.00001)] = 1

    slope_c = np.round(slope_c, 5)
    ws_c = np.round(ws_c, 5)
    forest_c = np.round(forest_c, 5)

    minvar = np.minimum(np.minimum(slope_c, ws_c), forest_c)
    out = (1 - minvar) * minvar + minvar * (slope_c + ws_c + forest_c) / 3
    out = np.round(out, 5) * 100
    continuous = out.astype(np.int16)

    thd = pra_thd * 100
    binary = out.copy()
    binary[(0 <= binary) & (binary < thd)] = 0
    binary[(thd < binary) & (binary <= 100)] = 1
    binary = binary.astype(np.int16)
    if sf and sf > 0:
        binary = sieve(binary, size=sf + 1, connectivity=8)
    return continuous, binary
