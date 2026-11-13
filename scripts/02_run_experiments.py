"""Run the deterministic maps and the Monte Carlo experiments.

Two configurations:
  rich : lidar DEM (10 m) + lidar canopy cover. What you can do where LidarBC exists.
  poor : Copernicus GLO-30 (30 m) + VRI crown closure. What exists province-wide.

For each, one factor is perturbed at a time (DEM, forest, thresholds) and then
all together. Results go to results/*.npz and results/metrics.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from autoates_bc import classifier as C
from autoates_bc import pipeline as PL
from autoates_bc import uncertainty as U

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed"
R = ROOT / "results"
R.mkdir(exist_ok=True)

N = {"rich": {"dem": 30, "forest": 30, "params": 30, "all": 60},
     "poor": {"dem": 150, "forest": 150, "params": 150, "all": 200, "inputs": 200}}
if "--quick" in sys.argv:
    N = {c: {k: 6 for k in v} for c, v in N.items()}


def progress(tag):
    t0 = time.time()

    def cb(i, n):
        if i % 10 == 0 or i == n:
            print(f"  {tag}: {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    return cb


def main():
    metrics = {}
    g10 = PL.load_dem(D / "dem_lidar_10m.tif")
    g30 = PL.load_dem(D / "dem_cop30_30m.tif")
    pcc_lidar10 = np.clip(PL.load_raster(D / "pcc_lidar_10m.tif"), 0, 100)
    pcc_vri10 = np.clip(PL.load_raster(D / "pcc_vri_10m.tif"), 0, 100)
    pcc_lidar30 = np.clip(PL.load_raster(D / "pcc_lidar_30m.tif"), 0, 100)
    pcc_vri30 = np.clip(PL.load_raster(D / "pcc_vri_30m.tif"), 0, 100)

    # ---------------- deterministic maps
    det = {}
    t = time.time()
    det["lidar10_lidarpcc"], layers = PL.run(g10.dem, pcc_lidar10, 10.0, return_layers=True)
    metrics["seconds_per_run_10m"] = round(time.time() - t, 1)
    det["lidar10_vripcc"] = PL.run(g10.dem, pcc_vri10, 10.0)
    t = time.time()
    det["cop30_vripcc"] = PL.run(g30.dem, pcc_vri30, 30.0)
    metrics["seconds_per_run_30m"] = round(time.time() - t, 1)
    det["cop30_lidarpcc"] = PL.run(g30.dem, pcc_lidar30, 30.0)
    np.savez_compressed(R / "deterministic.npz", **det,
                        slope10=layers["slope"], pra10=layers["pra_binary"],
                        overhead10=layers["overhead"], cell_counts10=layers["cell_counts"],
                        z_delta10=layers["z_delta"])
    ref30 = U.block_mode(det["lidar10_lidarpcc"], 3)
    np.save(R / "reference30.npy", ref30)
    for k, v in det.items():
        metrics[f"class_share_{k}"] = (np.bincount(v.ravel(), minlength=5) / v.size).round(4).tolist()
    metrics["agree_forest_swap_10m"] = float((det["lidar10_lidarpcc"] == det["lidar10_vripcc"]).mean())
    metrics["agree_cop30vri_vs_ref"] = float((det["cop30_vripcc"] == ref30).mean())
    metrics["agree_cop30lidarpcc_vs_ref"] = float((det["cop30_lidarpcc"] == ref30).mean())

    # ---------------- error models
    # Copernicus error is measured against the lidar DTM, in strata that can be
    # built without lidar: Copernicus slope band x VRI forest / open.
    lidar30 = g10.dem.reshape(280, 3, 350, 3).mean(axis=(1, 3))
    err = g30.dem - lidar30
    slope_c = C.horn_slope(g30.dem, 30, 30, nodata=None)
    slope_l = C.horn_slope(lidar30, 30, 30, nodata=None)
    inner = np.zeros(err.shape, bool)
    inner[2:-2, 2:-2] = True
    s30 = np.where(slope_c < 0, 0, slope_c)
    bands = [0, 15, 30, 45, 91]
    sd_map = np.zeros_like(err)
    bias_map = np.zeros_like(err)
    em = {"bias_m": float(err.mean()), "sd_m": float(err.std()), "strata": [],
          "slope_error_sd_deg": float((slope_c - slope_l)[inner].std()),
          "slope_error_bias_deg": float((slope_c - slope_l)[inner].mean())}
    for fname, fm in (("open", pcc_vri30 < 10), ("forest", pcc_vri30 >= 10)):
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = fm & (s30 >= lo) & (s30 < hi)
            sd_map[m] = err[m].std()
            bias_map[m] = err[m].mean()
            em["strata"].append({"cover": fname, "slope": f"{lo}-{hi}", "share": round(float(m.mean()), 3),
                                 "bias_m": round(float(err[m].mean()), 1), "sd_m": round(float(err[m].std()), 1)})
    sd_map = gaussian_filter(sd_map, 4)
    bias_map = gaussian_filter(bias_map, 4)
    # correlation length: pick the one whose simulated slope error matches the
    # measured slope error, since slope is what the classifier reads
    rng = np.random.default_rng(0)
    best = None
    for corr in np.arange(1.4, 4.01, 0.2):
        sim = np.mean([(C.horn_slope(g30.dem + sd_map * U.random_field(err.shape, corr, rng), 30, 30, nodata=None)
                        - slope_c)[inner].std() for _ in range(5)])
        if best is None or abs(sim - em["slope_error_sd_deg"]) < best[0]:
            best = (abs(sim - em["slope_error_sd_deg"]), float(corr), float(sim))
    corr = best[1]
    em["corr_px"] = round(corr, 2)
    em["simulated_slope_error_sd_deg"] = round(best[2], 2)
    em["half_correlation_px"] = round(float(U.corr_px_from_errors(err - bias_map)), 2)

    # VRI crown closure vs lidar canopy cover: not the same quantity
    dpcc = pcc_vri10.astype(float) - pcc_lidar10
    em["vri_minus_lidar_pcc_mean"] = float(dpcc.mean())
    em["vri_minus_lidar_pcc_sd"] = float(dpcc.std())
    fm = pcc_vri30 > 0
    scale = float((pcc_lidar30[fm].astype(float) * pcc_vri30[fm]).sum() / (pcc_vri30[fm].astype(float) ** 2).sum())
    em["vri_to_cover_scale"] = round(scale, 2)
    pcc_vri30_scaled = np.clip(np.round(pcc_vri30 * scale), 0, 100)
    det["cop30_vriscaled"] = PL.run(g30.dem, pcc_vri30_scaled, 30.0)
    metrics["agree_cop30vriscaled_vs_ref"] = float((det["cop30_vriscaled"] == ref30).mean())
    np.save(R / "cop30_vriscaled.npy", det["cop30_vriscaled"])
    metrics["error_model"] = em
    print(json.dumps(em, indent=1))

    models = {
        # lidar DTM: nominal 0.5 m vertical error (assumption, no local check data)
        "rich": {
            "dem": U.ErrorModel(dem_sd=0.5, dem_corr_px=1.5),
            "forest": U.ErrorModel(forest_sd=10, forest_alternatives=[pcc_vri10]),
            "params": U.ErrorModel(params=True),
            "all": U.ErrorModel(dem_sd=0.5, dem_corr_px=1.5, forest_sd=10,
                                forest_alternatives=[pcc_vri10], params=True),
        },
        "poor": {
            "dem": U.ErrorModel(dem_sd=sd_map, dem_corr_px=corr, dem_bias=bias_map),
            "forest": U.ErrorModel(forest_sd=15, forest_alternatives=[pcc_vri30_scaled]),
            "params": U.ErrorModel(params=True),
            "inputs": U.ErrorModel(dem_sd=sd_map, dem_corr_px=corr, dem_bias=bias_map, forest_sd=15,
                                   forest_alternatives=[pcc_vri30_scaled]),
            "all": U.ErrorModel(dem_sd=sd_map, dem_corr_px=corr, dem_bias=bias_map, forest_sd=15,
                                forest_alternatives=[pcc_vri30_scaled], params=True),
        },
    }
    data = {"poor": (g30.dem, pcc_vri30, 30.0), "rich": (g10.dem, pcc_lidar10, 10.0)}

    (R / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf8")

    # one file per experiment, so an interrupted run can be resumed
    for cfg in ("poor", "rich"):
        dem, pcc, cs = data[cfg]
        for name, model in models[cfg].items():
            path = R / f"mc_{cfg}_{name}.npz"
            if path.exists():
                print("skip", path.name, flush=True)
                continue
            n = N[cfg][name]
            print(cfg, name, n, flush=True)
            counts = U.monte_carlo(dem, pcc, cs, model, n, seed=sum(map(ord, cfg + name)),
                                   progress=progress(f"{cfg}/{name}"))
            np.savez_compressed(path, counts=counts)

    (R / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf8")
    print("done")


if __name__ == "__main__":
    main()
