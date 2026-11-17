"""Leave-one-area-out test over the blocks in autoates_bc.sites.

For each held-out block the Copernicus error model (bias and spread per slope
band and forest / open, correlation length) and the crown-closure-to-cover
scale are fitted on the other blocks only. The 30 m ensemble is then run on the
held-out block and scored against its own lidar run.

Two province-wide forest layers are compared as the 30 m forest input: VRI crown
closure and SCANFI crown closure.

Resumable: one file per block and experiment in results/sites/.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from autoates_bc import classifier as C
from autoates_bc import pipeline as PL
from autoates_bc import uncertainty as U
from autoates_bc.sites import SITES

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "sites"
R = ROOT / "results" / "sites"
R.mkdir(parents=True, exist_ok=True)
N = 6 if "--quick" in sys.argv else 120
BANDS = [0, 15, 30, 45, 91]
FOREST_SD = 15


def load(sid):
    d = D / sid
    g10 = PL.load_dem(d / "dem_lidar_10m.tif")
    g30 = PL.load_dem(d / "dem_cop30_30m.tif")
    clip = lambda name: np.clip(PL.load_raster(d / name), 0, 100)
    s = dict(dem10=g10.dem, cop=g30.dem, lid10=clip("pcc_lidar_10m.tif"), lid30=clip("pcc_lidar_30m.tif"),
             vri=clip("pcc_vri_30m.tif"), scanfi=clip("pcc_scanfi_30m.tif"))
    h, w = s["cop"].shape
    s["lidar30"] = s["dem10"].reshape(h, 3, w, 3).mean(axis=(1, 3))
    s["err"] = s["cop"] - s["lidar30"]
    slope_c = C.horn_slope(s["cop"], 30, 30, nodata=None)
    s["slope_c"] = np.where(slope_c < 0, 0, slope_c)
    slope_l = C.horn_slope(s["lidar30"], 30, 30, nodata=None)
    inner = np.zeros((h, w), bool)
    inner[2:-2, 2:-2] = True
    s["inner"] = inner
    s["slope_err_sd"] = float((slope_c - slope_l)[inner].std())
    return s


def strata(s):
    for cover, fm in (("open", s["vri"] < 10), ("forest", s["vri"] >= 10)):
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            yield (cover, lo), fm & (s["slope_c"] >= lo) & (s["slope_c"] < hi)


def fit(train):
    """Pooled error statistics and forest scales from the training blocks."""
    stats = {}
    for key in [k for k, _ in strata(train[0])]:
        vals = np.concatenate([s["err"][m] for s in train for k, m in strata(s) if k == key])
        stats[key] = (float(vals.mean()), float(vals.std())) if vals.size > 30 else None
    allv = np.concatenate([s["err"].ravel() for s in train])
    for key, v in stats.items():
        if v is None:
            stats[key] = (float(allv.mean()), float(allv.std()))
    scale = {}
    for layer in ("vri", "scanfi"):
        x = np.concatenate([s[layer].ravel() for s in train]).astype(float)
        y = np.concatenate([s["lid30"].ravel() for s in train]).astype(float)
        m = x > 0
        scale[layer] = float((x[m] * y[m]).sum() / (x[m] ** 2).sum())
    return stats, scale


def maps(s, stats):
    sd = np.zeros(s["cop"].shape, np.float32)
    bias = np.zeros(s["cop"].shape, np.float32)
    for key, m in strata(s):
        bias[m], sd[m] = stats[key]
    return gaussian_filter(sd, 4), gaussian_filter(bias, 4)


def calibrate_corr(s, sd_map):
    rng = np.random.default_rng(0)
    base = C.horn_slope(s["cop"], 30, 30, nodata=None)
    best = None
    for corr in np.arange(1.4, 4.01, 0.2):
        sim = np.mean([(C.horn_slope(s["cop"] + sd_map * U.random_field(sd_map.shape, corr, rng), 30, 30,
                                     nodata=None) - base)[s["inner"]].std() for _ in range(4)])
        if best is None or abs(sim - s["slope_err_sd"]) < best[0]:
            best = (abs(sim - s["slope_err_sd"]), float(corr))
    return best[1]


def score(counts, det, ref):
    su = U.summarize(counts)
    ph = U.prob_higher(counts, det)
    prob, conf = su["prob"], su["confidence"]
    onehot = np.stack([ref == k for k in range(5)]).astype(np.float32)
    det1h = np.stack([det == k for k in range(5)]).astype(np.float32)
    correct, under = det == ref, det < ref
    edges = np.linspace(0, 1, 6)
    p, o = prob[1:].ravel(), onehot[1:].ravel()
    idx = np.clip(np.digitize(p, edges) - 1, 0, 4)
    flag = ph > 0.2
    out = dict(single_matches_lidar=float(correct.mean()),
               most_likely_matches_lidar=float((su["mode"] == ref).mean()),
               brier_single=float(((det1h - onehot) ** 2).sum(0).mean()),
               brier_ensemble=float(((prob - onehot) ** 2).sum(0).mean()),
               underrated_share=float(under.mean()),
               area_flagged=float(flag.mean()), underrated_caught=float(flag[under].mean()),
               reliability=[dict(bin=f"{edges[b]:.1f}-{edges[b + 1]:.1f}", n=int((idx == b).sum()),
                                 mean_predicted=float(p[idx == b].mean()), observed=float(o[idx == b].mean()))
                            for b in range(5) if (idx == b).any()],
               accuracy_by_confidence=[])
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        m = (conf >= lo) & (conf < hi)
        out["accuracy_by_confidence"].append(dict(confidence=f"{lo}-{min(hi, 1.0)}", share=float(m.mean()),
                                                  single_accuracy=float(correct[m].mean()) if m.any() else None))
    return out


def main():
    ids = [s for s in SITES if (D / s / "pcc_scanfi_30m.tif").exists()]
    data = {s: load(s) for s in ids}
    out = {}

    # per block: lidar reference, own error statistics, own correlation length
    for sid in ids:
        s = data[sid]
        f = R / f"{sid}_ref.npz"
        if not f.exists():
            print("reference", sid, flush=True)
            ref10 = PL.run(s["dem10"], s["lid10"], 10.0)
            own_stats, _ = fit([s])
            sd_map, _ = maps(s, own_stats)
            np.savez_compressed(f, ref10=ref10, ref30=U.block_mode(ref10, 3), corr=calibrate_corr(s, sd_map))
        z = np.load(f)
        s["ref"], s["ref10"], s["corr"] = z["ref30"], z["ref10"], float(z["corr"])

    for sid in ids:
        s = data[sid]
        train = [data[o] for o in ids if o != sid]
        stats, scale = fit(train)
        corr = float(np.mean([t["corr"] for t in train]))
        sd_map, bias_map = maps(s, stats)
        res = dict(label=SITES[sid][1], range=SITES[sid][2],
                   class_share_lidar=(np.bincount(s["ref"].ravel(), minlength=5) / s["ref"].size).round(4).tolist(),
                   copernicus_error=dict(bias_m=float(s["err"].mean()), sd_m=float(s["err"].std()),
                                         slope_error_sd_deg=s["slope_err_sd"], own_corr_px=s["corr"]),
                   transferred=dict(corr_px=corr, vri_scale=round(scale["vri"], 2),
                                    scanfi_scale=round(scale["scanfi"], 2)),
                   forest_layers={}, ensembles={})
        layers = {"vri": s["vri"], "vri_scaled": np.clip(np.round(s["vri"] * scale["vri"]), 0, 100),
                  "scanfi": s["scanfi"], "scanfi_scaled": np.clip(np.round(s["scanfi"] * scale["scanfi"]), 0, 100),
                  "lidar_cover": s["lid30"]}
        dets = {}
        fdet = R / f"{sid}_det.npz"
        if fdet.exists():
            dets = dict(np.load(fdet))
        else:
            for name, layer in layers.items():
                dets[name] = PL.run(s["cop"], layer, 30.0)
            np.savez_compressed(fdet, **dets)
        for name, layer in layers.items():
            diff = layer.astype(float) - s["lid30"]
            res["forest_layers"][name] = dict(
                single_matches_lidar=float((dets[name] == s["ref"]).mean()),
                underrated_share=float((dets[name] < s["ref"]).mean()),
                overrated_share=float((dets[name] > s["ref"]).mean()),
                cover_bias_vs_lidar=float(diff.mean()), cover_rmse_vs_lidar=float(np.sqrt((diff ** 2).mean())))
        for base in ("vri", "scanfi"):
            f = R / f"{sid}_mc_{base}.npz"
            if not f.exists():
                print("ensemble", sid, base, N, flush=True)
                model = U.ErrorModel(dem_sd=sd_map, dem_corr_px=corr, dem_bias=bias_map, forest_sd=FOREST_SD,
                                     forest_alternatives=[layers[f"{base}_scaled"]])
                counts = U.monte_carlo(s["cop"], layers[base], 30.0, model, N, seed=sum(map(ord, sid + base)))
                np.savez_compressed(f, counts=counts)
            res["ensembles"][base] = score(np.load(f)["counts"], dets[base], s["ref"])
        out[sid] = res
        (ROOT / "results" / "multi_area.json").write_text(json.dumps(out, indent=1), encoding="utf8")
        print("scored", sid, flush=True)
    print("done")


if __name__ == "__main__":
    main()
