"""Manuscript figures 1, 5 and 6 (PNG at 400 dpi and vector PDF).

Fig. 1  study areas and workflow
Fig. 5  skill and calibration across areas, with 95% block-bootstrap intervals
Fig. 6  sources of uncertainty and province-wide forest layers

Needs results/statistics.json (10_statistics.py), results/multi_area.json,
results/summary.json and data/ne/ne_10m admin-1 and lakes (Natural Earth).
Categorical colours follow Okabe and Ito (2008), which stay distinct under the
common forms of colour-vision deficiency.
"""
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import shapefile
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("figs", ROOT / "scripts" / "05_figures.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
plt, R, MM, INK, MUTED, RULE, save = F.plt, F.R, F.MM, F.INK, F.MUTED, F.RULE, F.save

from autoates_bc import uncertainty as U          # noqa: E402
from autoates_bc.sites import SITES               # noqa: E402

OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7", grey="#8A8A8A")
SHORT = {"mamquam": "Mamquam", "coast_4986": "Sunshine Coast", "coast_5036": "Powell River N1",
         "coast_5066": "Powell River N2", "marble_5122": "Marble Range"}
NUM = {"mamquam": 1, "coast_4986": 2, "coast_5036": 3, "coast_5066": 4, "marble_5122": 5}
CENTRES = {"mamquam": (-122.780, 49.786), "coast_4986": (-123.623, 49.859), "coast_5036": (-124.024, 50.364),
           "coast_5066": (-124.074, 50.657), "marble_5122": (-121.920, 51.219),
           "Connaught": (-117.556, 51.302), "BowSummit": (-116.492, 51.697)}
COAST = [k for k, v in SITES.items() if v[2] == "Coast Mountains"]
W2 = 183 * MM


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def fig_letter(fig, ax, s, dx=0.05, y=0.955):
    """Panel letter in figure coordinates, so all letters in a row line up."""
    fig.text(ax.get_position().x0 - dx, y, s, fontsize=9, fontweight="bold", va="top", ha="left", color=INK)


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------- Fig. 1
def read_ne(zpath, name):
    z = zipfile.ZipFile(zpath)
    return shapefile.Reader(shp=io.BytesIO(z.read(f"{name}.shp")), dbf=io.BytesIO(z.read(f"{name}.dbf")),
                            encoding="latin1")


def draw_shapes(ax, reader, keep, tr, **kw):
    fields = [f[0] for f in reader.fields[1:]]
    for sr in reader.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        if not keep(rec):
            continue
        pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = np.array(pts[a:b])
            if len(ring) < 3:
                continue
            x, y = tr.transform(ring[:, 0], ring[:, 1])
            ax.add_patch(Polygon(np.c_[x, y], closed=True, **kw))


def box(ax, xy, w, h, text, fc="#ffffff", ec=INK, fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=0.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, a, b, color=INK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=0.6, color=color,
                                 shrinkA=0, shrinkB=0))


def fig1():
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3005", always_xy=True)
    adm = read_ne(ROOT / "data" / "ne" / "ne10_admin1.zip", "ne_10m_admin_1_states_provinces")
    lakes = read_ne(ROOT / "data" / "ne" / "ne10_lakes.zip", "ne_10m_lakes")
    fig = plt.figure(figsize=(W2, W2 * 0.47))
    ax = fig.add_axes([0.045, 0.32, 0.46, 0.64])
    ax.set_facecolor("#e6eef4")
    near = lambda r: r.get("admin") in ("Canada", "United States of America")
    draw_shapes(ax, adm, near, tr, fc="#f4f3ef", ec="#b9b9b9", lw=0.35)
    draw_shapes(ax, adm, lambda r: r.get("name") in ("British Columbia",), tr, fc="#fbfaf7", ec="#8a8a8a", lw=0.5)
    draw_shapes(ax, lakes, lambda r: True, tr, fc="#e6eef4", ec="none")
    x0, y0 = tr.transform(-126.4, 48.95)
    x1, y1 = tr.transform(-115.6, 52.0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    # graticule
    for lon in range(-126, -115, 2):
        xs, ys = tr.transform(np.full(50, lon), np.linspace(48, 53, 50))
        ax.plot(xs, ys, color="#c9c9c9", lw=0.3, zorder=1)
        xb, _ = tr.transform(lon, 48.95)
        ax.text(xb, y0 - 6000, f"{-lon}°W", ha="center", va="top", fontsize=5.5, color=MUTED)
    for lat in (49, 50, 51, 52):
        xs, ys = tr.transform(np.linspace(-127, -115, 50), np.full(50, lat))
        ax.plot(xs, ys, color="#c9c9c9", lw=0.3, zorder=1)
        _, yb = tr.transform(-126.4, lat)
        ax.text(x0 - 6000, yb, f"{lat}°N", ha="right", va="center", fontsize=5.5, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    for name, lon, lat in (("Coast Mountains", -124.9, 50.95), ("Columbia\nMountains", -118.6, 50.55),
                           ("Rocky\nMountains", -116.75, 50.95)):
        x, y = tr.transform(lon, lat)
        ax.text(x, y, name, fontsize=6, style="italic", color="#6b6b6b", ha="center", va="center", rotation=0)
    for name, lon, lat in (("BRITISH\nCOLUMBIA", -120.4, 49.85), ("ALBERTA", -115.95, 51.9), ("USA", -119.5, 49.05)):
        x, y = tr.transform(lon, lat)
        ax.text(x, y, name, fontsize=5.5, color="#9a9a9a", ha="center", va="center", fontweight="bold")
    x, y = tr.transform(-123.12, 49.28)
    ax.plot(x, y, "o", ms=2.2, color="#555555", zorder=4)
    ax.text(x + 9000, y - 7000, "Vancouver", fontsize=5.5, color="#555555", va="top")
    offs = {"mamquam": (7, 0), "coast_4986": (0, 7), "coast_5036": (7, 0), "coast_5066": (7, 0),
            "marble_5122": (7, 0)}
    for k, (lon, lat) in CENTRES.items():
        x, y = tr.transform(lon, lat)
        if k in SITES:
            col = OI["blue"] if k in COAST else OI["green"]
            ax.plot(x, y, "s", ms=4.2, mfc=col, mec="white", mew=0.5, zorder=5)
            dx, dy = offs[k]
            ax.annotate(str(NUM[k]), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.3,
                        fontweight="bold", color=col, ha="center", va="center")
        else:
            ax.plot(x, y, "^", ms=5, mfc=OI["vermillion"], mec="white", mew=0.5, zorder=5)
            lab = "Connaught Creek" if k == "Connaught" else "Bow Summit"
            ax.annotate(lab, (x, y), xytext=(-6, 0), textcoords="offset points",
                        fontsize=5.8, color=OI["vermillion"], ha="right", va="center")
    # scale bar, 100 km
    sx, sy = x1 - 160000, y0 + 70000
    ax.add_patch(Rectangle((sx, sy), 100000, 4500, fc=INK, ec=INK, lw=0.4, zorder=6))
    ax.text(sx + 50000, sy + 9000, "100 km", fontsize=5.5, ha="center", va="bottom")
    h = [plt.Line2D([], [], marker="s", ls="", ms=4.2, mfc=OI["blue"], mec="white", label="Coast Mountains block, lidar reference"),
         plt.Line2D([], [], marker="s", ls="", ms=4.2, mfc=OI["green"], mec="white", label="Interior block, lidar reference"),
         plt.Line2D([], [], marker="^", ls="", ms=5, mfc=OI["vermillion"], mec="white", label="Expert benchmark (Sykes et al., 2024)")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.01, -0.09), frameon=False, fontsize=5.6,
              handletextpad=0.3, borderaxespad=0, ncol=1)
    ax.text(1.0, -0.12, "1 Mamquam    2 Sunshine Coast" + chr(10) + "3 Powell River N1    4 Powell River N2" + chr(10)
            + "5 Marble Range", transform=ax.transAxes, fontsize=5.6, color=INK, ha="right", va="top", linespacing=1.4)
    letter(ax, "a", x=-0.01, y=1.01)

    # ---- (b) workflow
    bx = fig.add_axes([0.53, 0.03, 0.465, 0.94])
    bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.axis("off")
    letter(bx, "b", x=0.0, y=0.955)
    bx.text(0.02, 0.955, "Inputs and their perturbation", fontsize=6.5, fontweight="bold", va="bottom")
    rows = [("Elevation model", "Correlated Gaussian error;|bias and spread by slope|band and forest cover"),
            ("Forest layer", "Correlated noise;|choice of forest product"),
            ("Thresholds", "Uniform over the range|of published calibrations")]
    for i, (a_, b_) in enumerate(rows):
        y = 0.79 - i * 0.135
        box(bx, (0.02, y), 0.2, 0.11, a_, fc="#f4f3ef", bold=True, fs=5.8)
        box(bx, (0.255, y), 0.38, 0.11, b_.replace("|", chr(10)), fc="#ffffff", fs=5.4)
        arrow(bx, (0.22, y + 0.055), (0.255, y + 0.055))
        arrow(bx, (0.635, y + 0.055), (0.7, y + 0.055))
    bx.add_patch(FancyBboxPatch((0.7, 0.47), 0.29, 0.46, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc="#eef3f8", ec=OI["blue"], lw=0.7))
    bx.text(0.845, 0.895, "AutoATES v2.0", ha="center", fontsize=6.3, fontweight="bold", color=OI["blue"])
    for j, t in enumerate(("Release areas|(fuzzy PRA)", "Runout (Flow-Py|with detrainment)", "Classifier")):
        yy = 0.77 - j * 0.1
        box(bx, (0.72, yy), 0.25, 0.075, t.replace("|", chr(10)), fs=5.3)
        if j:
            arrow(bx, (0.845, yy + 0.1), (0.845, yy + 0.075))
    bx.text(0.845, 0.49, "repeated 30-200 times", ha="center", fontsize=5.4, color=MUTED, style="italic")
    arrow(bx, (0.845, 0.47), (0.845, 0.39))
    box(bx, (0.53, 0.25), 0.46, 0.135, "Per cell: class probabilities, most likely|class, agreement, P(higher class)".replace("|", chr(10)),
        fc="#ffffff", ec=OI["blue"], fs=5.5)
    bx.text(0.02, 0.37, "Evaluation", fontsize=6.5, fontweight="bold", va="bottom")
    box(bx, (0.02, 0.2), 0.44, 0.13, "Five lidar-covered blocks, leave one|out: error model fitted on the other|four, scored against the lidar run".replace("|", chr(10)),
        fc="#f4f3ef", fs=5.4)
    box(bx, (0.02, 0.035), 0.44, 0.13, "Connaught Creek and Bow Summit:|scored against expert|benchmark maps".replace("|", chr(10)),
        fc="#f4f3ef", fs=5.4)
    box(bx, (0.53, 0.035), 0.46, 0.17, "1.5 km block bootstrap: accuracy;|Brier score against the single map|and a confusion-matrix baseline;|reliability; AUC of agreement".replace("|", chr(10)),
        fc="#ffffff", fs=5.4)
    arrow(bx, (0.76, 0.25), (0.76, 0.205))
    arrow(bx, (0.46, 0.265), (0.53, 0.15))
    arrow(bx, (0.46, 0.1), (0.53, 0.1))
    save(fig, "fig01_study_design")


# ---------------------------------------------------------------- Fig. 5
def pooled_reliability(items, bins=10):
    """items: list of (counts, ref, mask, merge0). One-vs-rest over classes 1-4."""
    P, O = [], []
    for counts, ref, mask, merge0 in items:
        c = counts.astype(float)
        r = ref.copy()
        if merge0:
            c[1] += c[0]; c[0] = 0
            r = np.where(r == 0, 1, r)
        p = c / c.sum(axis=0)
        for k in range(1, 5):
            P.append(p[k][mask]); O.append((r == k)[mask])
    P, O = np.concatenate(P), np.concatenate(O).astype(float)
    idx = np.clip((P * bins).astype(int), 0, bins - 1)
    n = np.bincount(idx, minlength=bins)
    mp = np.bincount(idx, weights=P, minlength=bins) / np.maximum(n, 1)
    ob = np.bincount(idx, weights=O, minlength=bins) / np.maximum(n, 1)
    # Wilson 95% interval on the observed frequency
    z = 1.96
    den = 1 + z ** 2 / np.maximum(n, 1)
    ctr = (ob + z ** 2 / (2 * np.maximum(n, 1))) / den
    half = z * np.sqrt(ob * (1 - ob) / np.maximum(n, 1) + z ** 2 / (4 * np.maximum(n, 1) ** 2)) / den
    return mp, ob, ctr - half, ctr + half, n


def agreement_curve(items, edges=(0, .5, .6, .7, .8, .9, 1.0001)):
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        num = den = 0
        for counts, det, ref, mask, merge0 in items:
            c = counts.astype(float)
            d, r = det, ref
            if merge0:
                c[1] += c[0]; c[0] = 0
                d, r = np.where(d == 0, 1, d), np.where(r == 0, 1, r)
            conf = (c / c.sum(axis=0)).max(axis=0)
            m = mask & (conf >= lo) & (conf < hi)
            num += (d == r)[m].sum(); den += m.sum()
        if den > 200:
            xs.append((lo + min(hi, 1)) / 2); ys.append(num / den)
    return np.array(xs), np.array(ys)


def fig5():
    st = json.loads((R / "statistics.json").read_text(encoding="utf8"))
    areas = [(f"{k}/vri", SHORT[k]) for k in SITES] + [("Connaught/all", "Connaught Creek"), ("BowSummit/all", "Bow Summit")]
    fig = plt.figure(figsize=(W2, W2 * 0.4))
    ax = fig.add_axes([0.12, 0.17, 0.25, 0.7])
    ys = np.arange(len(areas))[::-1].astype(float)
    ys[-2:] -= 0.5
    series = [("brier_single", "Single map", OI["grey"], "o", 0.2), ("brier_conf", "Confusion-matrix baseline", OI["sky"], "D", 0.0),
              ("brier_ens", "Ensemble", OI["blue"], "s", -0.2)]
    for key, lab, col, mk, off in series:
        for (k, _), y in zip(areas, ys):
            v = st[k][key]
            ax.plot([v["lo"], v["hi"]], [y + off] * 2, color=col, lw=0.9, solid_capstyle="butt")
            ax.plot(v["value"], y + off, mk, ms=3.2, color=col, mec="white", mew=0.3, label=lab if k == areas[0][0] else None)
    ax.set_yticks(ys); ax.set_yticklabels([a[1] for a in areas])
    ax.axhline(ys[-2] + 0.75, color=RULE, lw=0.6)
    ax.text(1.02, ys[-2] + 0.85, "vs lidar run", fontsize=5.3, color=MUTED, ha="right", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.text(1.02, ys[-2] + 0.62, "vs experts", fontsize=5.3, color=MUTED, ha="right", va="top",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.45, 1.0), ncol=3, fontsize=5.5,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0)
    clean(ax); fig_letter(fig, ax, "a", dx=0.11)

    # (b) reliability
    coast_items, cn, bw = [], [], []
    for k in COAST:
        counts = np.load(R / "sites" / f"{k}_mc_vri.npz")["counts"]
        ref = np.load(R / "sites" / f"{k}_ref.npz")["ref30"]
        coast_items.append((counts, ref, np.ones(ref.shape, bool), False))
    bench = {}
    for site in ("Connaught", "BowSummit"):
        z = np.load(R / f"benchmark_{site}.npz")
        mask = (z["valid"] >= 0) & (z["dem"] != -9999)
        bench[site] = (z["counts_all"], z["det"].astype(int), np.where(mask, z["valid"], 1).astype(int), mask)
    ax = fig.add_axes([0.46, 0.17, 0.22, 0.7])
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, zorder=1)
    curves = [("Coast Mountains, 4 blocks", coast_items, OI["blue"], "o"),
              ("Connaught Creek", [(bench["Connaught"][0], bench["Connaught"][2], bench["Connaught"][3], True)], OI["vermillion"], "^"),
              ("Bow Summit", [(bench["BowSummit"][0], bench["BowSummit"][2], bench["BowSummit"][3], True)], OI["orange"], "v")]
    for lab, items, col, mk in curves:
        mp, ob, lo, hi, n = pooled_reliability(items)
        ok = n > 100
        ax.fill_between(mp[ok], lo[ok], hi[ok], color=col, alpha=0.15, lw=0)
        ax.plot(mp[ok], ob[ok], "-" + mk, color=col, ms=2.8, lw=1.0, mec="white", mew=0.3, label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Ensemble probability of a class")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5, handlelength=1.4, borderaxespad=0.1)
    clean(ax); fig_letter(fig, ax, "b", dx=0.055)

    # (c) accuracy of the single map vs agreement
    ax = fig.add_axes([0.765, 0.17, 0.215, 0.7])
    lines = [("Coast Mountains", [(np.load(R / "sites" / f"{k}_mc_vri.npz")["counts"], np.load(R / "sites" / f"{k}_det.npz")["vri"],
                                   np.load(R / "sites" / f"{k}_ref.npz")["ref30"], np.ones((280, 350), bool), False) for k in COAST],
              OI["blue"], "o", "coast_pooled/vri"),
             ("Connaught Creek", [(bench["Connaught"][0], bench["Connaught"][1], bench["Connaught"][2], bench["Connaught"][3], True)],
              OI["vermillion"], "^", "Connaught/all"),
             ("Bow Summit", [(bench["BowSummit"][0], bench["BowSummit"][1], bench["BowSummit"][2], bench["BowSummit"][3], True)],
              OI["orange"], "v", "BowSummit/all")]
    for i, (lab, items, col, mk, key) in enumerate(lines):
        xs, ys_ = agreement_curve(items)
        ax.plot(xs, ys_ * 100, "-" + mk, color=col, ms=2.8, lw=1.0, mec="white", mew=0.3)
        a = st[key]["auc"]
        ax.text(0.03, 0.97 - i * 0.075, f"{lab}: AUC {a['value']:.2f} ({a['lo']:.2f}-{a['hi']:.2f})",
                transform=ax.transAxes, fontsize=5.4, color=col, va="top")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0, 100)
    ax.set_xlabel("Agreement (top-class probability)")
    ax.set_ylabel("Single map correct (%)")
    clean(ax); fig_letter(fig, ax, "c", dx=0.06)
    save(fig, "fig05_skill_and_calibration")


# ---------------------------------------------------------------- Fig. 6
def fig6():
    summ = json.loads((R / "summary.json").read_text(encoding="utf8"))
    ma = json.loads((R / "multi_area.json").read_text(encoding="utf8"))
    fig = plt.figure(figsize=(W2, W2 * 0.34))
    ax = fig.add_axes([0.1, 0.2, 0.32, 0.62])
    names = [("dem", "Elevation model"), ("forest", "Forest layer"), ("params", "Thresholds"), ("all", "All together")]
    y = np.arange(len(names))[::-1]
    for cfg, col, mk, lab, off in (("poor", OI["vermillion"], "s", "Copernicus 30 m + VRI", 0.15),
                                   ("rich", OI["blue"], "o", "Lidar 10 m + lidar canopy", -0.15)):
        v = [summ["experiments"][f"{cfg}/{n}"]["share_conf_below_0.8"] * 100 for n, _ in names]
        ax.hlines(y + off, 0, v, color=col, lw=0.7, alpha=0.6)
        ax.plot(v, y + off, mk, ms=3.6, color=col, mec="white", mew=0.3, label=lab)
        for vv, yy in zip(v, y + off):
            ax.text(vv + 2, yy, f"{vv:.0f}", va="center", fontsize=5.6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in names])
    ax.set_xlim(0, 100); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Terrain where < 80% of runs agree (%)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=5.6,
              handletextpad=0.2, columnspacing=0.9, borderaxespad=0)
    clean(ax); fig_letter(fig, ax, "a", dx=0.09)
    ax.text(1.0, -0.3, "Mamquam block, one factor perturbed at a time", transform=ax.transAxes, fontsize=5.4,
            color=MUTED, ha="right")

    ax = fig.add_axes([0.6, 0.2, 0.38, 0.62])
    layers = [("vri", "VRI", OI["grey"], "o"), ("scanfi", "SCANFI", OI["yellow"], "D"),
              ("vri_scaled", "VRI × k", OI["blue"], "s"), ("scanfi_scaled", "SCANFI × k", OI["orange"], "^"),
              ("lidar_cover", "Lidar canopy", "#000000", "P")]
    ids = COAST
    yy = np.arange(len(ids))[::-1]
    for j, (key, lab, col, mk) in enumerate(layers):
        v = [ma[k]["forest_layers"][key]["single_matches_lidar"] * 100 for k in ids]
        ax.plot(v, yy + (2 - j) * 0.12, mk, ms=3.4, color=col, mec="#333333" if key == "scanfi" else "white", mew=0.3,
                label=lab, ls="")
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[k] for k in ids])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(45, 80)
    ax.set_xlabel("Single 30 m map matches lidar run (%)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=5, fontsize=5.6,
              handletextpad=0.1, columnspacing=0.7, borderaxespad=0)
    clean(ax); fig_letter(fig, ax, "b", dx=0.1)
    ax.text(1.0, -0.3, "k fitted on the other blocks (1.63-1.73); lidar canopy is not available province-wide",
            transform=ax.transAxes, fontsize=5.4, color=MUTED, ha="right")
    save(fig, "fig06_uncertainty_sources")


if __name__ == "__main__":
    fig1()
    fig5()
    fig6()
