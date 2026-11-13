"""Out-of-sample check of the 30 m error model.

In 02_run_experiments.py the Copernicus error statistics come from the whole
study area, which is also where the ensemble is scored. Here the statistics
(elevation bias and spread per stratum, correlation length, VRI-to-cover
scale) are fitted on the western half only, and the ensemble is scored on the
eastern half, which the fit never saw. A 1.5 km strip between them is dropped.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from autoates_bc import classifier as C
from autoates_bc import pipeline as PL
from autoates_bc import uncertainty as U

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed"
R = ROOT / "results"
N = 200


def main():
    g10 = PL.load_dem(D / "dem_lidar_10m.tif")
    g30 = PL.load_dem(D / "dem_cop30_30m.tif")
    vri = np.clip(PL.load_raster(D / "pcc_vri_30m.tif"), 0, 100)
    lid = np.clip(PL.load_raster(D / "pcc_lidar_30m.tif"), 0, 100)
    ref = np.load(R / "reference30.npy")
    det = np.load(R / "deterministic.npz")["cop30_vripcc"]

    h, w = g30.dem.shape
    cols = np.arange(w)[None, :].repeat(h, 0)
    train = cols < w // 2 - 25
    test = cols >= w // 2 + 25
    test[:5] = test[-5:] = False
    test[:, -5:] = False

    lidar30 = g10.dem.reshape(h, 3, w, 3).mean(axis=(1, 3))
    err = g30.dem - lidar30
    slope_c = C.horn_slope(g30.dem, 30, 30, nodata=None)
    slope_l = C.horn_slope(lidar30, 30, 30, nodata=None)
    s30 = np.where(slope_c < 0, 0, slope_c)
    sd_map = np.zeros_like(err)
    bias_map = np.zeros_like(err)
    bands = [0, 15, 30, 45, 91]
    for fm in (vri < 10, vri >= 10):
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = fm & (s30 >= lo) & (s30 < hi)
            fit = m & train
            sd_map[m] = err[fit].std()        # statistics from the west only
            bias_map[m] = err[fit].mean()
    sd_map = gaussian_filter(sd_map, 4)
    bias_map = gaussian_filter(bias_map, 4)
    inner = train.copy()
    inner[:2] = inner[-2:] = False
    inner[:, :2] = False
    target = float((slope_c - slope_l)[inner].std())
    rng = np.random.default_rng(0)
    best = None
    for corr in np.arange(1.4, 4.01, 0.2):
        sim = np.mean([(C.horn_slope(g30.dem + sd_map * U.random_field(err.shape, corr, rng), 30, 30,
                                     nodata=None) - slope_c)[inner].std() for _ in range(5)])
        if best is None or abs(sim - target) < best[0]:
            best = (abs(sim - target), float(corr))
    corr = best[1]
    fm = (vri > 0) & train
    scale = float((lid[fm].astype(float) * vri[fm]).sum() / (vri[fm].astype(float) ** 2).sum())
    vri_scaled = np.clip(np.round(vri * scale), 0, 100)

    model = U.ErrorModel(dem_sd=sd_map, dem_corr_px=corr, dem_bias=bias_map, forest_sd=15,
                         forest_alternatives=[vri_scaled])
    counts = U.monte_carlo(g30.dem, vri, 30.0, model, N, seed=7,
                           progress=lambda i, n: print(i, n, flush=True) if i % 20 == 0 else None)
    np.savez_compressed(R / "mc_transfer.npz", counts=counts)
    s = U.summarize(counts)
    ph = U.prob_higher(counts, det)

    def score(mask):
        prob = s["prob"][:, mask]
        r = ref[mask]
        d = det[mask]
        onehot = np.stack([r == k for k in range(5)]).astype(np.float32)
        det1h = np.stack([d == k for k in range(5)]).astype(np.float32)
        edges = np.linspace(0, 1, 6)
        p = prob[1:].ravel()
        o = onehot[1:].ravel()
        idx = np.clip(np.digitize(p, edges) - 1, 0, 4)
        rel = [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
                "mean_predicted": float(p[idx == b].mean()), "observed": float(o[idx == b].mean())}
               for b in range(5)]
        under = d < r
        flag = ph[mask] > 0.2
        conf = s["confidence"][mask]
        return {"cells": int(mask.sum()),
                "deterministic_accuracy": float((d == r).mean()),
                "most_likely_accuracy": float((s["mode"][mask] == r).mean()),
                "brier_deterministic": float(((det1h - onehot) ** 2).sum(0).mean()),
                "brier_ensemble": float(((prob - onehot) ** 2).sum(0).mean()),
                "reliability": rel,
                "area_flagged_p_higher_0.2": float(flag.mean()),
                "underrated_caught": float(flag[under].mean()),
                "det_accuracy_conf_ge_0.9": float((d == r)[conf >= 0.9].mean()),
                "share_conf_ge_0.9": float((conf >= 0.9).mean()),
                "det_accuracy_conf_lt_0.5": float((d == r)[conf < 0.5].mean()),
                "share_conf_lt_0.5": float((conf < 0.5).mean())}

    out = {"fit": {"corr_px": corr, "vri_to_cover_scale": round(scale, 2),
                   "slope_error_sd_deg_west": round(target, 2)},
           "east_unseen": score(test), "west_fitted": score(train & (cols > 5))}
    (R / "transfer_test.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
