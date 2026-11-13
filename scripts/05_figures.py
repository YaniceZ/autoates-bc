"""Publication figures (PNG at 400 dpi and vector PDF) in results/figures.

Map conventions: relief by multiplying colour with a hillshade (no grey veil),
200 m contours, scale bar, UTM 10N grid ticks in km, panel letters. ATES classes
keep the colours of the ATES v2 standard (green, blue, black, red).

The main product is the value-by-alpha map: the hue is the most likely class
and the colour fades to white as the ensemble agrees less, so one map carries
both the class and how far it can be trusted.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle

from autoates_bc import pipeline as PL
from autoates_bc import uncertainty as U

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed"
R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(exist_ok=True)

INK, MUTED, RULE = "#1a1a1a", "#5f6672", "#d5d9df"
ATES = {1: "#3f9b52", 2: "#2c6fbb", 3: "#26262b", 4: "#cf3a2b"}
NAMES = {1: "Simple (1)", 2: "Challenging (2)", 3: "Complex (3)", 4: "Extreme (4)"}
ORANGE = LinearSegmentedColormap.from_list("o", ["#ffffff", "#fde3c4", "#f6a45c", "#d95f0e", "#8c2d04"])
BLUE = LinearSegmentedColormap.from_list("b", ["#ffffff", "#d6e4f2", "#8fb5dc", "#3f7fbf", "#173f73"])
GREEN = LinearSegmentedColormap.from_list("g", ["#ffffff", "#dcebd6", "#9cc890", "#4f9a55", "#1d5a2e"])
DIVERGE = LinearSegmentedColormap.from_list("d", ["#8c2d04", "#e08a45", "#f3f3f1", "#6f9fd0", "#1b4a8a"])
ELEV = LinearSegmentedColormap.from_list("e", ["#3d6b4f", "#8fa97a", "#d8cfa5", "#b79a7a", "#e9e6e1", "#ffffff"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6.5, "axes.linewidth": 0.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 1.0,
})
MM = 1 / 25.4


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=400)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


class Base:
    """Shared map furniture for one grid."""

    def __init__(self, grid):
        self.g = grid
        cs = grid.cell_size
        h, w = grid.dem.shape
        x0, y1 = grid.transform[2] / 1000, grid.transform[5] / 1000
        self.extent = (x0, x0 + w * cs / 1000, y1 - h * cs / 1000, y1)
        ls = LightSource(azdeg=315, altdeg=42)
        hs = ls.hillshade(grid.dem.astype(float), vert_exag=1.3, dx=cs, dy=cs)
        self.shade = 0.38 + 0.62 * hs            # multiply factor, keeps colours clean

    def show(self, ax, rgb, letter=None, title=None, contours=True, bar=False, ticks="none"):
        img = np.clip(rgb * self.shade[..., None], 0, 1)
        ax.imshow(img, extent=self.extent, interpolation="nearest")
        if contours:
            h, w = self.g.dem.shape
            xs = np.linspace(self.extent[0], self.extent[1], w)
            ys = np.linspace(self.extent[3], self.extent[2], h)
            ax.contour(xs, ys, self.g.dem, levels=np.arange(200, 2800, 200), colors="k",
                       linewidths=0.18, alpha=0.35)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_aspect("equal")
        ax.set_xticks([512, 514, 516, 518, 520])
        ax.set_yticks([5512, 5514, 5516, 5518])
        ax.tick_params(direction="in", top=True, right=True, length=2,
                       labelbottom=ticks in ("x", "both"), labelleft=ticks in ("y", "both"))
        ax.set_yticklabels([f"{v}" for v in (5512, 5514, 5516, 5518)], rotation=90, va="center")
        if ticks in ("x", "both"):
            ax.set_xlabel("Easting (km, UTM 10N)", labelpad=1.5)
        if ticks in ("y", "both"):
            ax.set_ylabel("Northing (km)", labelpad=1.5)
        if letter:
            ax.text(0.0, 1.025, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="bottom", ha="left", color=INK)
        if title:
            ax.text(0.065, 1.03, title, transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left", color=INK)
        if bar:
            x, y = self.extent[1] - 2.75, self.extent[2] + 0.42
            ax.add_patch(Rectangle((x - 0.22, y - 0.2), 2.44, 0.78, fc="white", ec="none", alpha=0.82, zorder=5))
            ax.add_patch(Rectangle((x, y), 1, 0.11, fc=INK, ec=INK, lw=0.4, zorder=6))
            ax.add_patch(Rectangle((x + 1, y), 1, 0.11, fc="white", ec=INK, lw=0.4, zorder=6))
            for dx, lab in ((0, "0"), (1, "1"), (2, "2 km")):
                ax.text(x + dx, y + 0.19, lab, fontsize=5.5, ha="center", va="bottom", zorder=6, color=INK)
            ax.annotate("N", xy=(self.extent[0] + 0.55, self.extent[3] - 0.35), xytext=(self.extent[0] + 0.55, self.extent[3] - 1.3),
                        ha="center", va="center", fontsize=6, fontweight="bold", color=INK, zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, shrinkA=3, shrinkB=0),
                        bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="none", alpha=0.82))


def class_rgb(cls, alpha=None):
    rgb = np.ones(cls.shape + (3,))
    for k, c in ATES.items():
        rgb[cls == k] = to_rgb(c)
    if alpha is not None:
        rgb = 1 - (1 - rgb) * alpha[..., None]
    return rgb


def conf_alpha(conf):
    """Confidence 0.4 -> almost white, 1.0 -> full colour."""
    return np.clip((conf - 0.4) / 0.6, 0, 1) ** 1.2 * 0.92 + 0.08


def scalar_rgb(a, cmap, vmin, vmax):
    return cmap(np.clip((a - vmin) / (vmax - vmin), 0, 1))[..., :3]


def colorbar(fig, ax, cmap, vmin, vmax, label, ticks=None, fmt=None, y=-0.085):
    cax = ax.inset_axes([0.0, y, 0.42, 0.028])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, labelsize=5.5, pad=1)
    if fmt:
        cb.ax.set_xticklabels(fmt)
    cax.text(1.06, 0.5, label, transform=cax.transAxes, fontsize=6, va="center", ha="left", color=INK)


def class_legend(ax, loc=(0.0, -0.1), ncol=4):
    ax.legend(handles=[Patch(fc=ATES[k], ec="none", label=NAMES[k]) for k in ATES], loc="upper left",
              bbox_to_anchor=loc, ncol=ncol, frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.0, handletextpad=0.4, borderaxespad=0)


def vba_legend(ax, loc):
    """2-D legend: class down, ensemble agreement across."""
    lax = ax.inset_axes(loc)
    confs = np.array([0.45, 0.6, 0.75, 0.9, 1.0])
    img = np.ones((4, len(confs), 3))
    for r, k in enumerate((4, 3, 2, 1)):
        for c, cf in enumerate(confs):
            img[r, c] = 1 - (1 - np.array(to_rgb(ATES[k]))) * conf_alpha(np.array(cf))
    lax.imshow(img, aspect="auto", extent=(0, len(confs), 0, 4))
    lax.set_yticks([3.5, 2.5, 1.5, 0.5])
    lax.set_yticklabels(["Extreme", "Complex", "Challenging", "Simple"], fontsize=5.5)
    lax.set_xticks([0.5, 2.5, 4.5])
    lax.set_xticklabels(["45%", "75%", "100%"], fontsize=5.5)
    lax.tick_params(length=0, pad=1.5)
    lax.set_xlabel("runs agreeing on the class", fontsize=5.5, labelpad=1.5)
    for s in lax.spines.values():
        s.set_linewidth(0.4)


def load_mc(cfg, name):
    p = R / f"mc_{cfg}_{name}.npz"
    return np.load(p)["counts"] if p.exists() else None


def main():
    g10 = PL.load_dem(D / "dem_lidar_10m.tif")
    g30 = PL.load_dem(D / "dem_cop30_30m.tif")
    b10, b30 = Base(g10), Base(g30)
    det = np.load(R / "deterministic.npz")
    ref30 = np.load(R / "reference30.npy")
    summary = json.loads((R / "summary.json").read_text(encoding="utf8")) if (R / "summary.json").exists() else None
    transfer = json.loads((R / "transfer_test.json").read_text(encoding="utf8")) if (R / "transfer_test.json").exists() else None
    W2 = 180 * MM   # two-column width

    # ---------------- Fig 1: study area and inputs
    pcc_l = np.clip(PL.load_raster(D / "pcc_lidar_10m.tif"), 0, 100)
    pcc_v = np.clip(PL.load_raster(D / "pcc_vri_10m.tif"), 0, 100)
    err = g30.dem - g10.dem.reshape(280, 3, 350, 3).mean(axis=(1, 3))
    fig, axes = plt.subplots(2, 2, figsize=(W2, W2 * 0.9))
    b10.show(axes[0, 0], scalar_rgb(g10.dem, ELEV, 150, 2600), "a", "Lidar terrain model, 10 m", bar=True, ticks="y")
    colorbar(fig, axes[0, 0], ELEV, 150, 2600, "elevation (m)", [500, 1500, 2500])
    b10.show(axes[0, 1], scalar_rgb(pcc_l, GREEN, 0, 100), "b", "Canopy cover from lidar")
    colorbar(fig, axes[0, 1], GREEN, 0, 100, "cover (%)", [0, 50, 100])
    b10.show(axes[1, 0], scalar_rgb(pcc_v, GREEN, 0, 100), "c", "Crown closure, forest inventory (VRI)", ticks="both")
    colorbar(fig, axes[1, 0], GREEN, 0, 100, "closure (%)", [0, 50, 100], y=-0.2)
    b30.show(axes[1, 1], scalar_rgb(err, DIVERGE, -40, 40), "d", "Copernicus GLO-30 minus lidar", ticks="x")
    colorbar(fig, axes[1, 1], DIVERGE, -40, 40, "difference (m)", [-40, 0, 40], y=-0.2)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.96, bottom=0.12, wspace=0.07, hspace=0.24)
    save(fig, "fig1_study_area_inputs")

    # ---------------- Fig 2: lidar configuration, hard map vs probabilistic map
    c = load_mc("rich", "all")
    if c is not None:
        s = U.summarize(c)
        ph = U.prob_higher(c, det["lidar10_lidarpcc"])
        fig, axes = plt.subplots(1, 3, figsize=(W2, W2 * 0.375))
        b10.show(axes[0], class_rgb(det["lidar10_lidarpcc"]), "a", "AutoATES, single run", bar=True, ticks="both")
        class_legend(axes[0], loc=(0.0, -0.24), ncol=2)
        b10.show(axes[1], class_rgb(s["mode"], conf_alpha(s["confidence"])), "b", "Most likely class, faded by agreement", ticks="x")
        vba_legend(axes[1], [0.33, -0.47, 0.42, 0.22])
        b10.show(axes[2], scalar_rgb(ph, ORANGE, 0, 0.8), "c", "Probability of a higher class than (a)", ticks="x")
        colorbar(fig, axes[2], ORANGE, 0, 0.8, "probability", [0, 0.4, 0.8], y=-0.3)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.25, wspace=0.08)
        save(fig, "fig2_lidar_probabilistic")

    # ---------------- Fig 3: province-wide inputs against the lidar result
    c = load_mc("poor", "all")
    base = det["cop30_vripcc"]
    s = U.summarize(c)
    ph = U.prob_higher(c, base)
    fig, axes = plt.subplots(2, 2, figsize=(W2, W2 * 0.95))
    b30.show(axes[0, 0], class_rgb(ref30), "a", "Lidar DEM and lidar canopy (reference)", bar=True, ticks="y")
    b30.show(axes[0, 1], class_rgb(base), "b", "Copernicus 30 m and VRI, single run")
    class_legend(axes[0, 0], loc=(0.0, -0.045))
    diff = np.clip(base.astype(int) - ref30, -2, 2)
    drgb = np.ones(diff.shape + (3,))
    for v, col in zip((-2, -1, 0, 1, 2), ("#8c2d04", "#e58b3f", "#f1f1ef", "#86afd9", "#1b4a8a")):
        drgb[diff == v] = to_rgb(col)
    b30.show(axes[1, 0], drgb, "c", "(b) minus (a)", ticks="both")
    axes[1, 0].legend(handles=[Patch(fc=col, ec="none", label=lab) for col, lab in
                               (("#8c2d04", "2 classes too low"), ("#e58b3f", "1 too low"), ("#f1f1ef", "same"),
                                ("#86afd9", "1 too high"), ("#1b4a8a", "2 too high"))],
                      loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=5, frameon=False, handlelength=1.1,
                      handleheight=0.9, columnspacing=0.9, handletextpad=0.4, borderaxespad=0)
    b30.show(axes[1, 1], class_rgb(s["mode"], conf_alpha(s["confidence"])), "d",
             "30 m ensemble: most likely class, faded by agreement", ticks="x")
    vba_legend(axes[1, 1], [0.3, -0.36, 0.4, 0.15])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.96, bottom=0.17, wspace=0.07, hspace=0.24)
    save(fig, "fig3_30m_vs_lidar")

    # ---------------- Fig 4: is the uncertainty honest, and where does it come from
    if summary:
        val = summary["validation_30m_vs_lidar"]
        fig, axes = plt.subplots(1, 3, figsize=(W2, W2 * 0.36))
        ax = axes[0]
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, zorder=1)
        rows = val["inputs"]["reliability"]
        ax.plot([r["mean_predicted"] for r in rows], [r["observed"] for r in rows], "-o", color="#2c6fbb",
                ms=3.2, mec="white", mew=0.5, label="whole area, error model fitted in place", zorder=3)
        if transfer:
            rows = transfer["east_unseen"]["reliability"]
            ax.plot([r["mean_predicted"] for r in rows], [r["observed"] for r in rows], "-s", color="#d95f0e",
                    ms=3.2, mec="white", mew=0.5, label="eastern half, model fitted on west", zorder=3)
        ax.set_xlabel("Ensemble probability of a class")
        ax.set_ylabel("Share where lidar gives that class")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=5.8, handlelength=1.6, borderaxespad=0)

        ax = axes[1]
        rows = val["inputs"]["accuracy_by_confidence"]
        xs = np.arange(len(rows))
        acc = [r["deterministic_accuracy"] * 100 for r in rows]
        ax.bar(xs, acc, width=0.62, color="#2c6fbb", ec="none")
        for x, a, r in zip(xs, acc, rows):
            ax.text(x, a + 2, f"{a:.0f}%", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lab}\n({r['share'] * 100:.0f}%)" for lab, r in
                            zip(("<50%", "50-70%", "70-90%", ">90%"), rows)])
        ax.set_xlabel("Ensemble agreement on the class\n(share of area in brackets)")
        ax.set_ylabel("Single 30 m run matches lidar (%)")
        ax.set_ylim(0, 108)

        ax = axes[2]
        names = [("dem", "DEM error"), ("forest", "Forest layer"), ("params", "Thresholds"), ("all", "All together")]
        ys = np.arange(len(names))[::-1]
        for cfg, col, mk, lab, off in (("poor", "#d95f0e", "s", "Copernicus 30 m + VRI", 0.16),
                                       ("rich", "#2c6fbb", "o", "Lidar 10 m + lidar canopy", -0.16)):
            vals = [summary["experiments"].get(f"{cfg}/{n}", {}).get("share_conf_below_0.8") for n, _ in names]
            if any(v is None for v in vals):
                continue
            vals = [v * 100 for v in vals]
            ax.hlines(ys + off, 0, vals, color=col, lw=0.6, alpha=0.5)
            ax.plot(vals, ys + off, mk, color=col, ms=4, mec="white", mew=0.5, label=lab)
            for v, y in zip(vals, ys + off):
                ax.text(v + 3, y, f"{v:.0f}", va="center", fontsize=6, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([n for _, n in names])
        ax.set_ylim(-0.6, 3.9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Terrain with class agreement <80% (%)")
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=5.8, borderaxespad=0, handletextpad=0.3)
        ax.tick_params(axis="y", length=0)
        for ax, letter in zip(axes, "abc"):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.text(-0.3, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
        fig.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.25, wspace=0.5)
        save(fig, "fig4_validation_and_sources")


if __name__ == "__main__":
    main()
