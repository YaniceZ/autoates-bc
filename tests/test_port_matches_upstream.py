"""Check the port against the unmodified AutoATES v2.0 repository.

Needs ``vendor/AutoATES-v2.0`` (git clone of github.com/AutoATES/AutoATES-v2.0).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

from autoates_bc import classifier as C
from autoates_bc import flowpy as F
from autoates_bc import pra as P

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "AutoATES-v2.0"
BOW = VENDOR / "test-data" / "Bow Summit"
pytestmark = pytest.mark.skipif(not VENDOR.exists(), reason="upstream repo not cloned")


def read(path):
    with rasterio.open(path) as src:
        return src.read(1), src


def test_classifier_reproduces_bow_summit_outputs():
    dem, src = read(BOW / "dem.tif")
    xres, yres = src.res
    forest, _ = read(BOW / "forest.tif")
    overhead, _ = read(BOW / "Overhead.tif")
    fp, _ = read(BOW / "FP_int16.tif")
    sz, _ = read(BOW / "pra_binary.tif")
    p = C.Params.for_forest("bav")  # the values in outputs/inputpara.csv

    slope = C.horn_slope(dem, xres, yres)
    ates, steps = C.classify(slope, fp, overhead, forest, sz, xres * yres, p,
                             return_steps=True)
    for name, arr in steps.items():
        ref, _ = read(BOW / "outputs" / f"{name}.tif")
        assert np.array_equal(arr, ref), name
    ref, _ = read(BOW / "outputs" / "ates_gen.tif")
    assert np.array_equal(ates, ref)


def test_overhead_formula_matches_bow_summit():
    zd, _ = read(BOW / "z_delta.tif")
    cc, _ = read(BOW / "cell_counts.tif")
    ref, _ = read(BOW / "Overhead.tif")
    oe = C.overhead_exposure(zd, cc)
    assert np.abs(oe.astype(int) - ref).max() <= 1


def test_pra_reproduces_upstream_rasters():
    dem, src = read(VENDOR / "PRA" / "DEM.tif")
    forest, _ = read(VENDOR / "PRA" / "FOREST.tif")
    cs = src.transform[0]
    # parameters from PRA/log.txt
    ws = P.windshelter(dem, cs, radius=6, prob=0.5, winddir=0, windtol=180,
                       nodata=src.nodata)
    ref_ws, _ = read(VENDOR / "PRA" / "windshelter.tif")
    assert np.allclose(ws, ref_ws, atol=1e-5)
    cont, binary = P.pra(dem, cs, forest, "stems", pra_thd=0.15, sf=3, ws=ref_ws)
    ref_c, _ = read(VENDOR / "PRA" / "PRA_continous.tif")
    ref_b, _ = read(VENDOR / "PRA" / "PRA_binary.tif")
    # Identical on the Windows build used for the paper. Elsewhere, libm
    # differences in pow/arctan can move a value across the 5-decimal rounding
    # step, so allow a 1-unit change on a tiny share of cells.
    d_cont = np.abs(cont.astype(int) - ref_c)
    d_bin = binary != ref_b
    print(f"PRA continuous: {np.count_nonzero(d_cont)} of {d_cont.size} cells differ, max {d_cont.max()}; "
          f"binary: {np.count_nonzero(d_bin)} cells differ")
    assert d_cont.max() <= 1
    assert np.count_nonzero(d_cont) / d_cont.size < 1e-4
    assert np.count_nonzero(d_bin) / d_bin.size < 1e-4


def test_flowpy_matches_upstream_python():
    sys.path.insert(0, str(VENDOR / "FlowPy_detrainment"))
    import flow_core  # upstream, unmodified

    dem, src = read(BOW / "dem.tif")
    rel, _ = read(BOW / "pra_binary.tif")
    forest, _ = read(BOW / "forest.tif")
    win = (slice(60, 150), slice(40, 130))
    dem = dem[win].astype(np.float64)
    rel = rel[win].copy()
    rel[rel < 0] = 0
    keep = np.zeros_like(rel)
    rr, cc = np.where(rel > 0)
    keep[rr[::7], cc[::7]] = 1  # thin the release cells, upstream is slow
    forest = np.clip(forest[win], 0, None) / 50.0
    cs = float(src.transform[0])
    header = {"cellsize": cs, "noDataValue": -9999.0}

    ref = flow_core.calculation_effect(
        [dem.copy(), header, forest, keep.astype(np.float64), 24, 8, 0.003, 270])
    z_ref, flux_ref, cnt_ref, zsum_ref, _, ta_ref, dis_ref = ref
    out = F.flowpy(dem, keep, cs, forest, alpha=24, exp=8, flux=0.003, max_z=270)

    assert keep.sum() > 50
    assert np.array_equal(out["cell_counts"], cnt_ref.astype(np.int64))
    assert np.allclose(out["z_delta"], z_ref, atol=1e-3)
    assert np.allclose(out["flux"], flux_ref, atol=1e-5)
    assert np.allclose(out["fp_travel_angle"], ta_ref, atol=1e-3)
    assert np.allclose(out["z_delta_sum"], zsum_ref, rtol=1e-4, atol=1e-2)
