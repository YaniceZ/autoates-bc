"""Turn the Monte Carlo counts into numbers (results/summary.json) and GeoTIFFs.

Figures are drawn by 05_figures.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap

from autoates_bc import pipeline as PL
from autoates_bc import uncertainty as U

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed"
R = ROOT / "results"
FIG = R / "figures"
MAPS = R / "maps"
FIG.mkdir(exist_ok=True)
MAPS.mkdir(exist_ok=True)

# ATES convention: simple green, challenging blue, complex black, extreme red
ATES_COLORS = ["#d9d9d9", "#3aa655", "#2f6fd0", "#1a1a1a", "#c0392b"]
ATES_NAMES = ["0 non-avalanche", "1 simple", "2 challenging", "3 complex", "4 extreme"]
INK, MUTED = "#1f2328", "#667085"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.facecolor": "white", "savefig.dpi": 160})


def stats(counts, baseline):
    s = U.summarize(counts)
    ph = U.prob_higher(counts, baseline)
    out = {"mean_confidence": float(s["confidence"].mean()),
           "share_conf_below_0.8": float((s["confidence"] < 0.8).mean()),
           "share_conf_below_0.6": float((s["confidence"] < 0.6).mean()),
           "share_p_higher_above_0.2": float((ph > 0.2).mean()),
           "mode_equals_baseline": float((s["mode"] == baseline).mean()),
           "by_baseline_class": {}}
    for k in range(1, 5):
        m = baseline == k
        if m.any():
            out["by_baseline_class"][str(k)] = {
                "share_of_area": float(m.mean()),
                "mean_confidence": float(s["confidence"][m].mean()),
                "share_conf_below_0.8": float((s["confidence"][m] < 0.8).mean()),
                "share_p_higher_above_0.2": float((ph[m] > 0.2).mean())}
    return out, s, ph


def reliability(prob, ref, bins=10):
    """Pooled over classes 1-4: predicted probability vs observed frequency."""
    edges = np.linspace(0, 1, bins + 1)
    p = np.concatenate([prob[k].ravel() for k in range(1, 5)])
    o = np.concatenate([(ref == k).ravel() for k in range(1, 5)]).astype(float)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() > 0:
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "observed": float(o[m].mean())})
    return rows


def brier(prob, ref):
    onehot = np.stack([(ref == k) for k in range(5)]).astype(np.float32)
    return float(((prob - onehot) ** 2).sum(axis=0).mean())


def hillshade(dem, cs):
    return LightSource(azdeg=315, altdeg=45).hillshade(dem, vert_exag=1, dx=cs, dy=cs)


def class_map(ax, cls, hs, title, extent):
    ax.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=1)
    ax.imshow(cls, cmap=ListedColormap(ATES_COLORS), norm=BoundaryNorm(np.arange(-0.5, 5), 5),
              alpha=0.62, extent=extent, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def scalar_map(ax, a, hs, title, extent, vmin=0, vmax=1, cmap="Blues", label=""):
    ax.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=1)
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.75, extent=extent, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, color=MUTED)
    cb.outline.set_visible(False)


def legend(fig, y=0.02):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75) for c in ATES_COLORS[1:]]
    fig.legend(handles, ATES_NAMES[1:], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, y))


def main():
    metrics = json.loads((R / "metrics.json").read_text(encoding="utf8"))
    det = np.load(R / "deterministic.npz")
    ref30 = np.load(R / "reference30.npy")
    g10 = PL.load_dem(D / "dem_lidar_10m.tif")
    g30 = PL.load_dem(D / "dem_cop30_30m.tif")
    hs10, hs30 = hillshade(g10.dem, 10), hillshade(g30.dem, 30)
    ext = (0, 10.5, 0, 8.4)
    base = {"rich": det["lidar10_lidarpcc"], "poor": det["cop30_vripcc"]}
    grids = {"rich": g10, "poor": g30}
    summ = {}
    out = {"experiments": {}}

    for cfg in ("rich", "poor"):
        mc = {f.stem.split("_", 2)[2]: np.load(f)["counts"] for f in sorted(R.glob(f"mc_{cfg}_*.npz"))}
        for name in mc:
            st, s, ph = stats(mc[name], base[cfg])
            st["n_runs"] = int(mc[name][:, 0, 0].sum())
            out["experiments"][f"{cfg}/{name}"] = st
            summ[(cfg, name)] = (s, ph)
        s, ph = summ[(cfg, "all")]
        PL.write_like(MAPS / f"{cfg}_ates_deterministic.tif", base[cfg].astype(np.uint8), grids[cfg], nodata=0)
        PL.write_like(MAPS / f"{cfg}_ates_most_likely.tif", s["mode"].astype(np.uint8), grids[cfg], nodata=0)
        PL.write_like(MAPS / f"{cfg}_confidence.tif", s["confidence"].astype(np.float32), grids[cfg])
        PL.write_like(MAPS / f"{cfg}_prob_higher_class.tif", ph, grids[cfg])
        for k in range(1, 5):
            PL.write_like(MAPS / f"{cfg}_prob_class{k}.tif", s["prob"][k].astype(np.float32), grids[cfg])

    # ---------- can the 30 m ensemble predict what lidar says?
    val = {}
    onehot_det = np.stack([(base["poor"] == k) for k in range(5)]).astype(np.float32)
    val["deterministic"] = {"accuracy_vs_lidar": float((base["poor"] == ref30).mean()),
                            "brier": brier(onehot_det, ref30),
                            "underrated_share": float((base["poor"] < ref30).mean()),
                            "overrated_share": float((base["poor"] > ref30).mean())}
    under = base["poor"] < ref30
    for name in ("inputs", "all"):
        s, ph = summ[("poor", name)]
        v = {"accuracy_of_most_likely_vs_lidar": float((s["mode"] == ref30).mean()),
             "brier": brier(s["prob"], ref30),
             "reliability": reliability(s["prob"], ref30),
             "lidar_class_has_prob_ge_0.1": float(np.take_along_axis(s["prob"], ref30[None].astype(np.int64), 0)[0].__ge__(0.1).mean()),
             "flagging": []}
        for thr in (0.1, 0.2, 0.3, 0.5):
            flag = ph > thr
            v["flagging"].append({"threshold": thr, "area_flagged": float(flag.mean()),
                                  "underrated_cells_caught": float(flag[under].mean()),
                                  "precision": float(under[flag].mean()) if flag.any() else None})
        conf = s["confidence"]
        agree = base["poor"] == ref30
        v["accuracy_by_confidence"] = []
        for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
            m = (conf >= lo) & (conf < hi)
            v["accuracy_by_confidence"].append({"confidence": f"{lo}-{min(hi, 1.0)}", "share": float(m.mean()),
                                                "deterministic_accuracy": float(agree[m].mean()) if m.any() else None})
        val[name] = v
    out["validation_30m_vs_lidar"] = val

    # slope-only null model: how much underrating would a naive area share catch?
    out["class_shares"] = {k: metrics[k] for k in metrics if k.startswith("class_share")}
    out["agreement"] = {k: metrics[k] for k in metrics if k.startswith("agree")}
    out["error_model"] = metrics["error_model"]
    out["seconds_per_run"] = {"10m": metrics["seconds_per_run_10m"], "30m": metrics["seconds_per_run_30m"]}
    (R / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf8")

    print(json.dumps({k: v for k, v in out.items() if k != "error_model"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
