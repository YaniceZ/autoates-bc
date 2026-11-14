"""Score the ensemble against expert ATES benchmark maps.

Data: OSF project zxjw5 (Sykes et al. 2024, GPL-3.0), two study areas with a
26 m ALOS DEM, VRI basal area, and an expert benchmark map each:
  Connaught Creek (Glacier National Park, BC) and Bow Summit (Banff, AB).
Expected in data/osf/: {site}_ALOS30m_final.zip, Connaught_LiDAR5m_final.zip,
{site}_ALOS30m_autoATES_validation.RData.

The question: where the ensemble disagrees with itself, is that where AutoATES
disagrees with the experts?

DEM error model: fitted at Connaught Creek, where a 5 m lidar DEM exists, in
strata of ALOS slope band x forest / open. Bow Summit has no lidar, so the
Connaught statistics are transferred to it.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pyreadr
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter

from autoates_bc import classifier as C
from autoates_bc import pipeline as PL
from autoates_bc import uncertainty as U

ROOT = Path(__file__).resolve().parents[1]
OSF = ROOT / "data" / "osf"
R = ROOT / "results"
ND = -9999

# site-specific optima from each archive's inputpara.csv (Sykes et al. 2024, table 4)
SITES = {
    "Connaught": dict(SAT12=18, SAT23=29, SAT34=39, AAT3=34, TREE1=5, CC2=25, WIN_SIZE=3),
    "BowSummit": dict(SAT12=19, SAT23=28, SAT34=40, AAT3=33, TREE1=10, CC2=40, WIN_SIZE=1),
}
# spread of the published optima for basal-area forest input (tables 4 and 8)
PARAM_RANGES = {"SAT12": (18, 19), "SAT23": (28, 31), "SAT34": (37, 40), "AAT3": (33, 34),
                "CC2": (25, 40), "TREE1": (5, 10)}
N = {"dem": 100, "forest": 100, "params": 100, "inputs": 200, "all": 200}
FOREST_SD = 5.0   # m2/ha, assumed: no accuracy figure for VRI basal area was found
BANDS = [0, 15, 30, 45, 91]


def read_zip(zf, name):
    names = {n.lower(): n for n in zf.namelist()}
    with MemoryFile(zf.read(names[name.lower()])) as m, m.open() as s:
        return s.read(1), s.transform, s.crs, s.res


def load_site(site):
    zf = zipfile.ZipFile(OSF / f"{site}_ALOS30m_final.zip")
    g = lambda n: read_zip(zf, "ALOS30m_final/" + n)
    dem, transform, crs, res = g("inputdata/DEM_ATES.tif")
    d = dict(dem=dem.astype(np.float32), transform=transform, crs=crs, res=res,
             forest=g("inputdata/FOREST_ATES.tif")[0], forest_bi=g("inputdata/FOREST_BI_ATES.tif")[0],
             density=g("inputdata/forest_scaled.tif")[0], valid=g("valid_raster.tif")[0],
             published=g("ates_gen.tif")[0])
    d["inside"] = d["dem"] != ND
    ba = d["forest"].astype(float)
    ok = (ba > 0) & (d["density"] > 0)
    ratio = np.full(ba.shape, float(np.median(d["density"][ok] / ba[ok])))
    ratio[ok] = d["density"][ok] / ba[ok]
    d["ratio"] = ratio
    return d


def make_runner(d, clean_density=False):
    xres, yres = d["res"]

    def runner(dem, forest, p):
        dens = np.clip(np.where(forest < 0, 0, forest) * d["ratio"], 0, 1)
        if clean_density:
            dens = np.where(d["density"] == ND, 0.0, dens)
        else:
            # the published runs hand the -9999 cells of forest_scaled.tif to Flow-Py
            dens = np.where(d["density"] == ND, float(ND), dens)
        return PL.run(dem, forest.astype(np.int16), xres, p, forest_type="bav", forest_density=dens,
                      clip_density=False, nodata=ND, yres=yres)
    return runner


def fit_error_model(d):
    """ALOS minus lidar at Connaught Creek, per slope band and forest / open."""
    zf = zipfile.ZipFile(OSF / "Connaught_LiDAR5m_final.zip")
    lid, tr, crs, _ = read_zip(zf, "LiDAR5m_final/inputdata/DEM_ATES.tif")
    lid = lid.astype(np.float32)
    lid[lid == ND] = np.nan
    lid30 = np.full(d["dem"].shape, np.nan, np.float32)
    reproject(lid, lid30, src_transform=tr, src_crs=crs, dst_transform=d["transform"], dst_crs=d["crs"],
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    ok = np.isfinite(lid30) & d["inside"]
    xres, yres = d["res"]
    slope_a = C.horn_slope(d["dem"], xres, yres, nodata=ND)
    slope_l = C.horn_slope(np.where(ok, lid30, ND), xres, yres, nodata=ND)
    both = ok & (slope_a >= 0) & (slope_l >= 0)
    err = d["dem"] - lid30
    table = []
    for cover, fm in (("open", d["forest_bi"] != 1), ("forest", d["forest_bi"] == 1)):
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            m = both & fm & (slope_a >= lo) & (slope_a < hi)
            if m.sum() < 30:
                m = both & (slope_a >= lo) & (slope_a < hi)
            table.append(dict(cover=cover, lo=lo, hi=hi, n=int(m.sum()),
                              bias_m=float(err[m].mean()), sd_m=float(err[m].std())))
    target = float((slope_a - slope_l)[both].std())
    sd_map, bias_map = strata_maps(d, table)
    rng = np.random.default_rng(0)
    best = None
    for corr in np.arange(1.0, 4.01, 0.2):
        sims = []
        for _ in range(5):
            pert = np.where(d["inside"], d["dem"] + sd_map * U.random_field(err.shape, corr, rng), ND)
            sims.append((C.horn_slope(pert, xres, yres, nodata=ND) - slope_a)[both].std())
        if best is None or abs(np.mean(sims) - target) < best[0]:
            best = (abs(np.mean(sims) - target), float(corr), float(np.mean(sims)))
    return dict(strata=table, corr_px=best[1], slope_error_sd_deg=target,
                simulated_slope_error_sd_deg=best[2], lidar_overlap_cells=int(both.sum()),
                bias_m=float(err[both].mean()), sd_m=float(err[both].std()))


def strata_maps(d, table):
    xres, yres = d["res"]
    slope = C.horn_slope(d["dem"], xres, yres, nodata=ND)
    slope = np.where(slope < 0, 0, slope)
    sd = np.zeros(d["dem"].shape, np.float32)
    bias = np.zeros(d["dem"].shape, np.float32)
    for row in table:
        fm = (d["forest_bi"] == 1) if row["cover"] == "forest" else (d["forest_bi"] != 1)
        m = fm & (slope >= row["lo"]) & (slope < row["hi"])
        sd[m] = row["sd_m"]
        bias[m] = row["bias_m"]
    return gaussian_filter(sd, 3), gaussian_filter(bias, 3)


def no0(a):
    """Sykes et al. count non-avalanche terrain (0) as simple (1)."""
    return np.where(a == 0, 1, a)


def score(counts, det, d, site):
    v = d["valid"]
    k = (v >= 0) & d["inside"]
    ref = no0(v.astype(int))[k]
    det1 = no0(det.astype(int))[k]
    c = counts.astype(np.float64).copy()
    c[1] += c[0]
    c[0] = 0
    prob = (c / c.sum(axis=0))[:, k]
    mode = prob.argmax(axis=0)
    conf = prob.max(axis=0)
    onehot = np.stack([ref == j for j in range(5)]).astype(float)
    det1h = np.stack([det1 == j for j in range(5)]).astype(float)
    cum = np.cumsum(prob[::-1], axis=0)[::-1]
    p_higher = np.array([cum[min(j + 1, 4), i] if j < 4 else 0.0 for i, j in enumerate(det1)])
    correct = det1 == ref
    under = det1 < ref
    out = dict(cells=int(k.sum()),
               deterministic_accuracy=float(correct.mean()),
               most_likely_accuracy=float((mode == ref).mean()),
               brier_deterministic=float(((det1h - onehot) ** 2).sum(0).mean()),
               brier_ensemble=float(((prob - onehot) ** 2).sum(0).mean()),
               underrated_share=float(under.mean()), overrated_share=float((det1 > ref).mean()),
               expert_class_has_prob_ge_0_1=float((prob[ref, np.arange(ref.size)] >= 0.1).mean()))
    edges = np.linspace(0, 1, 6)
    p = prob[1:].ravel()
    o = onehot[1:].ravel()
    idx = np.clip(np.digitize(p, edges) - 1, 0, 4)
    out["reliability"] = [dict(bin=f"{edges[b]:.1f}-{edges[b + 1]:.1f}", n=int((idx == b).sum()),
                               mean_predicted=float(p[idx == b].mean()), observed=float(o[idx == b].mean()))
                          for b in range(5) if (idx == b).any()]
    out["accuracy_by_confidence"] = []
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        m = (conf >= lo) & (conf < hi)
        out["accuracy_by_confidence"].append(dict(confidence=f"{lo}-{min(hi, 1.0)}", share=float(m.mean()),
                                                  deterministic_accuracy=float(correct[m].mean()) if m.any() else None))
    out["flagging"] = []
    for thr in (0.1, 0.2, 0.3):
        flag = p_higher > thr
        out["flagging"].append(dict(threshold=thr, area_flagged=float(flag.mean()),
                                    underrated_caught=float(flag[under].mean()) if under.any() else None,
                                    precision=float(under[flag].mean()) if flag.any() else None))

    # do the three manual mappers disagree where the ensemble does?
    df = pyreadr.read_r(str(OSF / f"{site}_ALOS30m_autoATES_validation.RData"))["ATESdf"]
    inv = ~d["transform"]
    cols, rows = inv * (df["x"].to_numpy(), df["y"].to_numpy())
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    ok = (rows >= 0) & (rows < v.shape[0]) & (cols >= 0) & (cols < v.shape[1])
    man = np.stack([df[f"ManualATES{i}_no0"].to_numpy() for i in (1, 2, 3)])[:, ok].astype(float)
    conf_map = (c / c.sum(axis=0)).max(axis=0)
    cf = conf_map[rows[ok], cols[ok]]
    good = np.isfinite(man).all(axis=0) & d["inside"][rows[ok], cols[ok]]
    split = (man.max(axis=0) != man.min(axis=0))[good]
    cf = cf[good]
    out["manual_mappers"] = dict(cells=int(good.sum()), share_where_mappers_disagree=float(split.mean()),
                                 mean_confidence_mappers_agree=float(cf[~split].mean()),
                                 mean_confidence_mappers_disagree=float(cf[split].mean()),
                                 share_conf_below_0_8_mappers_agree=float((cf[~split] < 0.8).mean()),
                                 share_conf_below_0_8_mappers_disagree=float((cf[split] < 0.8).mean()))
    return out


def main():
    out = {}
    data = {s: load_site(s) for s in SITES}
    em = fit_error_model(data["Connaught"])
    out["error_model_fitted_at_connaught"] = em
    print(json.dumps({k: v for k, v in em.items() if k != "strata"}, indent=1), flush=True)

    for site, par in SITES.items():
        d = data[site]
        base = C.Params.for_forest("bav").with_(**par)
        runner = make_runner(d)
        det = runner(d["dem"], d["forest"], base)
        pub = np.where(d["published"] == ND, 0, d["published"])
        k = (d["valid"] >= 0) & d["inside"]
        res = dict(parameters=par,
                   port_equals_published_map=float((det[d["inside"]] == pub[d["inside"]]).mean()),
                   published_accuracy=float((no0(pub)[k] == no0(d["valid"].astype(int))[k]).mean()),
                   density_nodata_share_of_study_area=float((d["density"][d["inside"]] == ND).mean()))
        if res["density_nodata_share_of_study_area"] > 0:
            clean = make_runner(d, clean_density=True)(d["dem"], d["forest"], base)
            res["accuracy_with_density_nodata_as_zero"] = float(
                (no0(clean)[k] == no0(d["valid"].astype(int))[k]).mean())
        sd_map, bias_map = strata_maps(d, em["strata"])
        models = {"dem": U.ErrorModel(dem_sd=sd_map, dem_corr_px=em["corr_px"], dem_bias=bias_map),
                  "forest": U.ErrorModel(forest_sd=FOREST_SD),
                  "params": U.ErrorModel(params=True),
                  "inputs": U.ErrorModel(dem_sd=sd_map, dem_corr_px=em["corr_px"], dem_bias=bias_map,
                                         forest_sd=FOREST_SD),
                  "all": U.ErrorModel(dem_sd=sd_map, dem_corr_px=em["corr_px"], dem_bias=bias_map,
                                      forest_sd=FOREST_SD, params=True)}
        save = dict(dem=d["dem"], valid=d["valid"], det=det, published=pub.astype(np.int8),
                    res=np.array(d["res"]))
        res["experiments"] = {}
        for name, model in models.items():
            print(site, name, N[name], flush=True)
            counts = U.monte_carlo(d["dem"], d["forest"], d["res"][0], model, N[name],
                                   seed=sum(map(ord, site + name)), base=base, runner=runner,
                                   param_ranges=PARAM_RANGES, forest_max=200, nodata=ND)
            save[f"counts_{name}"] = counts
            res["experiments"][name] = score(counts, det, d, site)
        np.savez_compressed(R / f"benchmark_{site}.npz", **save)
        out[site] = res
        (R / "benchmark.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print("done")


if __name__ == "__main__":
    main()
