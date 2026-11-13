"""AutoATES v2.0 classifier.

Array-in / array-out port of ``AutoATES_classifier.py`` (Toft et al. 2024,
GPL-3.0). The reclassification order and the lookup table are copied from the
upstream script so the output can be compared cell by cell with the Bow Summit
rasters that ship with the upstream repository.

Classes: 1 simple, 2 challenging, 3 complex, 4 extreme. Upstream writes class 0
(non-avalanche terrain) as nodata in the final raster and so does this port.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.ndimage
from rasterio.fill import fillnodata
from skimage import morphology

NODATA = -9999

# Upper bounds of the "open", "sparse" and "mixed" forest classes per input.
TREE_DEFAULTS = {
    "pcc": (10, 50, 65),
    "bav": (10, 20, 25),
    "stems": (100, 250, 500),
    "sen2cc": (20, 60, 85),
}

# (slope/runout/overhead class + forest code + 100 if release area) -> ATES
_LOOKUP = {
    10: 0, 11: 1, 12: 2, 13: 3, 14: 4,
    20: 0, 21: 1, 22: 1, 23: 2, 24: 3,
    30: 0, 31: 1, 32: 1, 33: 1, 34: 3,
    40: 0, 41: 1, 42: 1, 43: 1, 44: 2,
    110: 0, 111: 1, 112: 2, 113: 3, 114: 4,
    120: 0, 121: 1, 122: 1, 123: 2, 124: 3,
    130: 0, 131: 1, 132: 1, 133: 2, 134: 3,
    140: 0, 141: 1, 142: 1, 143: 2, 144: 2,
}


@dataclass(frozen=True)
class Params:
    """Classifier thresholds. Defaults are the values in the upstream script."""
    SAT01: float = 15
    SAT12: float = 18
    SAT23: float = 28
    SAT34: float = 39
    AAT1: float = 18
    AAT2: float = 24
    AAT3: float = 33
    TREE1: float = 10
    TREE2: float = 50
    TREE3: float = 65
    CC1: float = 5
    CC2: float = 40
    ISL_SIZE: float = 30000
    WIN_SIZE: int = 3

    @classmethod
    def for_forest(cls, forest_type: str, **kw) -> "Params":
        t1, t2, t3 = TREE_DEFAULTS[forest_type]
        return cls(TREE1=t1, TREE2=t2, TREE3=t3, **kw)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


def horn_slope(dem, xres, yres, nodata=NODATA):
    """Slope in degrees, same as ``gdaldem slope`` (Horn, no edge computation)."""
    z = np.asarray(dem, dtype=np.float64)
    out = np.full(z.shape, float(NODATA), dtype=np.float32)
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dx = ((c + 2 * f + i) - (a + 2 * d + g)) / xres
    dy = ((g + 2 * h + i) - (a + 2 * b + c)) / yres
    slope = np.degrees(np.arctan(np.sqrt(dx * dx + dy * dy) / 8.0))
    bad = np.zeros(z.shape, dtype=bool) if nodata is None else (z == nodata)
    win = scipy.ndimage.maximum_filter(bad, size=3, mode="constant", cval=False)
    inner = slope.astype(np.float32)
    inner[win[1:-1, 1:-1]] = NODATA
    out[1:-1, 1:-1] = inner
    return out


def overhead_exposure(z_delta, cell_counts, max_z_delta=270.0, cc_max=None):
    """Overhead exposure 0-100: mean of scaled z_delta and scaled ln(cell count).

    The formula is not in the upstream code. It is described in Sykes et al.
    (2024, sect. 4.1.1) and this form reproduces the upstream Bow Summit
    ``Overhead.tif`` from ``z_delta.tif`` and ``cell_counts.tif`` to within the
    int16 truncation. ``cc_max`` is the scene maximum of the cell count unless
    given, which makes the layer depend on the extent of the study area.
    """
    cc = np.asarray(cell_counts, dtype=np.float64)
    zd = np.asarray(z_delta, dtype=np.float64)
    if cc_max is None:
        cc_max = float(cc.max())
    if cc_max <= 1:
        return np.zeros(cc.shape, dtype=np.int16)
    log_cc = np.log(np.maximum(cc, 1.0)) / np.log(cc_max) * 100.0
    oe = (zd / max_z_delta * 100.0 + log_cc) / 2.0
    return np.where(cc > 0, oe, 0).astype(np.int16)


def slope_classes(slope_deg, p: Params):
    """Slope raster (degrees, -9999 nodata) -> classes 0-4 and smoothed slope."""
    slope = np.asarray(slope_deg).astype(np.int16)[None]
    slope_nd = np.where(slope < 0, 0, slope)
    smooth = scipy.ndimage.uniform_filter(slope_nd, size=p.WIN_SIZE, mode="nearest")
    slope[(0 < slope) & (slope <= p.SAT01)] = 0
    slope[(p.SAT01 < slope) & (slope <= p.SAT12)] = 1
    slope[(p.SAT12 < slope) & (slope <= p.SAT23)] = 2
    slope[(p.SAT23 < slope) & (slope <= 100)] = 3
    slope[(p.SAT34 < smooth) & (smooth <= 100)] = 4
    return slope, smooth


def runout_classes(fp, p: Params):
    """Flow-Py travel angle -> runout classes 1-3."""
    fp = np.asarray(fp).astype(np.int16)[None]
    c1 = fp.copy()
    c1[(c1 >= 0) & (c1 < 90)] = 1
    c2 = fp.copy()
    c2[c2 < p.AAT2] = 0
    c2[(c2 >= p.AAT2) & (c2 < 90)] = 2
    c3 = fp.copy()
    c3[c3 < p.AAT3] = 0
    c3[(c3 >= p.AAT3) & (c3 < 90)] = 3
    return np.maximum(np.maximum(c1, c2), c3)


def overhead_classes(overhead, p: Params):
    a = np.asarray(overhead).astype(np.int16)[None]
    a[a == NODATA] = 0
    a[(0 <= a) & (a <= p.CC1)] = 1
    a[(p.CC1 < a) & (a <= p.CC2)] = 2
    a[(p.CC2 < a) & (a <= 20000)] = 3
    return a


def forest_codes(canopy, p: Params):
    """Forest raster -> 10 open, 20 sparse, 30 mixed, 40 dense, -1 otherwise."""
    raw = np.asarray(canopy)[None]

    f_open = raw.copy()
    f_open[f_open > p.TREE1] = -1
    f_open[(f_open >= 0) & (f_open <= p.TREE1)] = 10

    f_sparse = raw.copy()
    f_sparse[f_sparse > p.TREE2] = -1
    f_sparse[f_sparse <= p.TREE1] = -1
    f_sparse[(f_sparse > p.TREE1) & (f_sparse <= p.TREE2)] = 20

    f_dense = raw.copy()
    f_dense[f_dense > p.TREE3] = -1
    f_dense[f_dense <= p.TREE2] = -1
    f_dense[(f_dense > p.TREE2) & (f_dense <= p.TREE3)] = 30

    f_vdense = raw.copy()
    f_vdense[f_vdense < p.TREE3] = -1
    f_vdense[f_vdense >= p.TREE3] = 40

    return np.maximum(np.maximum(f_open, f_sparse), np.maximum(f_dense, f_vdense))


def generalize(merged, cell_area, p: Params):
    """Remove clusters smaller than ISL_SIZE and fill them from neighbours."""
    num_cells = np.around(p.ISL_SIZE / cell_area)
    src = (merged + 1).astype(np.int16)[None]
    lab, _ = morphology.label(src, connectivity=2, return_num=True)
    sizes = np.bincount(lab.ravel())
    small = sizes < num_cells
    small[0] = False
    lab[small[lab]] = 0
    lab = lab.astype(np.int16)
    data = fillnodata(src, lab, max_search_distance=num_cells / 4, smoothing_iterations=0)
    data = data - 1
    data[data == 0] = NODATA
    return data.astype(np.int16)[0]


def classify(slope_deg, fp, overhead, canopy, pra_binary, cell_area, p: Params,
             return_steps=False):
    """Full classifier. All inputs are 2-D arrays on the same grid.

    slope_deg : Horn slope in degrees (-9999 nodata), see :func:`horn_slope`
    fp : Flow-Py flow-path travel angle (degrees)
    overhead : overhead exposure 0-100, see :func:`overhead_exposure`
    canopy : forest raster in the units the TREE thresholds expect
    pra_binary : 0/1 release areas
    """
    slope, smooth = slope_classes(slope_deg, p)
    flow = runout_classes(fp, p)
    cc = overhead_classes(overhead, p)
    merge_new = np.maximum(np.maximum(slope, flow), cc)

    forest = forest_codes(canopy, p)
    sz = np.asarray(pra_binary).copy()[None]
    sz[sz == 1] = 100

    total = merge_new.astype(np.int64) + forest + sz
    out = total.copy()
    for key, val in _LOOKUP.items():
        out[total == key] = val
    out[out < 0] = 0
    merge_all = out.astype(np.int16)[0]

    ates = generalize(merge_all, cell_area, p)
    if return_steps:
        return ates, {"slope": slope[0], "slope_smooth": smooth[0], "flowpy": flow[0],
                      "cellcount_reclass": cc[0], "merge_new": merge_new[0],
                      "forest_reclass": forest[0], "merge_all": merge_all}
    return ates
