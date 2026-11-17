"""Skill and calibration scores with spatial block-bootstrap 95% intervals.

Neighbouring cells are not independent, so intervals come from resampling
1.5 km square blocks with replacement (Roberts et al. 2017), 2000 times.

Scores, for each area:
  acc_single     share of cells where the single deterministic map is right
  acc_mode       same for the ensemble's most likely class
  brier_single   multi-class Brier score of the single map (one-hot)
  brier_ens      multi-class Brier score of the ensemble probabilities
  bss            Brier skill score of the ensemble relative to the single map
  ece            top-label expected calibration error, 10 equal-width bins
  auc            how well ensemble agreement (top-class probability) ranks the
                 cells where the single map is right above those where it is
                 wrong; 0.5 = no skill
  recall_under   share of under-rated cells with P(higher class) > 0.2
  flagged        share of the area with P(higher class) > 0.2
  brier_conf     Brier score of a fairer baseline: the single map turned into
                 probabilities with a confusion matrix P(reference | single
                 class) estimated on the other areas only
  bss_conf       Brier skill score of the ensemble relative to that baseline

Reference: the lidar run for the five BC blocks (30 m ensemble, error model
fitted on the other blocks); the expert benchmark map at Connaught Creek and
Bow Summit (class 0 counted as 1, as in Sykes et al. 2024).
"""
import json
from pathlib import Path

import numpy as np

from autoates_bc.sites import SITES

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
B = 2000
NBIN = 10
HBIN = 100


def cell_terms(counts, det, ref, merge0=False, confusion=None):
    c = counts.astype(np.float64)
    if merge0:
        c[1] += c[0]
        c[0] = 0
        det = np.where(det == 0, 1, det)
        ref = np.where(ref == 0, 1, ref)
    prob = c / c.sum(axis=0)
    mode = prob.argmax(axis=0)
    conf = prob.max(axis=0)
    onehot = np.stack([ref == k for k in range(5)]).astype(np.float64)
    det1h = np.stack([det == k for k in range(5)]).astype(np.float64)
    cum = np.cumsum(prob[::-1], axis=0)[::-1]
    p_higher = np.zeros(det.shape)
    for k in range(4):
        m = det == k
        p_higher[m] = cum[k + 1][m]
    under = det < ref
    flag = p_higher > 0.2
    base = confusion[:, det] if confusion is not None else det1h   # (ref class, H, W)
    return dict(bs_conf=((base - onehot) ** 2).sum(axis=0), single=(det == ref).astype(float), mode=(mode == ref).astype(float),
                bs_single=((det1h - onehot) ** 2).sum(axis=0), bs_ens=((prob - onehot) ** 2).sum(axis=0),
                conf=conf, mode_ok=(mode == ref), under=under, under_flag=under & flag, flag=flag)


def block_sums(t, mask, block_px):
    h, w = mask.shape
    bid = (np.arange(h)[:, None] // block_px) * ((w + block_px - 1) // block_px) + np.arange(w)[None, :] // block_px
    bid = bid[mask]
    nb = int(bid.max()) + 1
    s = lambda v: np.bincount(bid, weights=v[mask].astype(float), minlength=nb)
    ebin = np.clip((t["conf"][mask] * NBIN).astype(int), 0, NBIN - 1)
    hbin = np.clip((t["conf"][mask] * HBIN).astype(int), 0, HBIN - 1)
    ece_n = np.zeros((nb, NBIN)); ece_c = np.zeros((nb, NBIN)); ece_a = np.zeros((nb, NBIN))
    np.add.at(ece_n, (bid, ebin), 1)
    np.add.at(ece_c, (bid, ebin), t["conf"][mask])
    np.add.at(ece_a, (bid, ebin), t["mode_ok"][mask])
    h_ok = np.zeros((nb, HBIN)); h_bad = np.zeros((nb, HBIN))
    ok = t["single"][mask] > 0
    np.add.at(h_ok, (bid[ok], hbin[ok]), 1)
    np.add.at(h_bad, (bid[~ok], hbin[~ok]), 1)
    keep = s(np.ones_like(t["conf"])) > 0
    out = dict(n=s(np.ones_like(t["conf"])), single=s(t["single"]), mode=s(t["mode"]), bs_single=s(t["bs_single"]),
               bs_ens=s(t["bs_ens"]), bs_conf=s(t["bs_conf"]), under=s(t["under"]), under_flag=s(t["under_flag"]), flag=s(t["flag"]),
               ece_n=ece_n, ece_c=ece_c, ece_a=ece_a, h_ok=h_ok, h_bad=h_bad)
    return {k: v[keep] for k, v in out.items()}


def scores(bs, w=None):
    """Scores from block sums; w = how many times each block is drawn."""
    if w is None:
        w = np.ones(len(bs["n"]))
    S = lambda k: (bs[k] * (w if bs[k].ndim == 1 else w[:, None])).sum(axis=0)
    n = S("n")
    out = dict(acc_single=S("single") / n, acc_mode=S("mode") / n, brier_single=S("bs_single") / n,
               brier_ens=S("bs_ens") / n, recall_under=S("under_flag") / max(S("under"), 1e-9),
               flagged=S("flag") / n)
    out["bss"] = 1 - out["brier_ens"] / out["brier_single"]
    out["brier_conf"] = S("bs_conf") / n
    out["bss_conf"] = 1 - out["brier_ens"] / out["brier_conf"]
    en, ec, ea = S("ece_n"), S("ece_c"), S("ece_a")
    nz = en > 0
    out["ece"] = float((np.abs(ec[nz] / en[nz] - ea[nz] / en[nz]) * en[nz]).sum() / en.sum())
    ho, hb = S("h_ok"), S("h_bad")
    # P(conf_ok > conf_bad) + 0.5 P(tie), from the two histograms
    below_bad = np.concatenate([[0], np.cumsum(hb)[:-1]])
    out["auc"] = float((ho * (below_bad + 0.5 * hb)).sum() / max(ho.sum() * hb.sum(), 1e-9))
    return {k: float(v) for k, v in out.items()}


def with_ci(parts, rng):
    """parts: list of block-sum dicts, resampled independently (stratified)."""
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    point = scores(merged)
    draws = {k: [] for k in point}
    for _ in range(B):
        w = np.concatenate([np.bincount(rng.integers(0, len(p["n"]), len(p["n"])), minlength=len(p["n"]))
                            for p in parts]).astype(float)
        for k, v in scores(merged, w).items():
            draws[k].append(v)
    return {k: dict(value=point[k], lo=float(np.percentile(draws[k], 2.5)), hi=float(np.percentile(draws[k], 97.5)),
                    blocks=int(len(merged["n"])))
            for k in point}


def confusion_from(pairs, merge0=False):
    """Column-normalised P(reference = j | single = i) from (det, ref) pairs."""
    m = np.ones((5, 5))  # add-one smoothing
    for det, ref in pairs:
        if merge0:
            det, ref = np.where(det == 0, 1, det), np.where(ref == 0, 1, ref)
        np.add.at(m, (ref.ravel(), det.ravel()), 1)
    return m / m.sum(axis=0, keepdims=True)


def main():
    rng = np.random.default_rng(20261117)
    out = {"bootstrap": dict(replicates=B, block_km=1.5)}
    parts = {}
    pairs = {(sid, layer): (np.load(R / "sites" / f"{sid}_det.npz")[layer], np.load(R / "sites" / f"{sid}_ref.npz")["ref30"])
             for sid in SITES for layer in ("vri", "scanfi") if (R / "sites" / f"{sid}_det.npz").exists()}
    for sid in SITES:
        for layer in ("vri", "scanfi"):
            f = R / "sites" / f"{sid}_mc_{layer}.npz"
            if not f.exists():
                continue
            counts = np.load(f)["counts"]
            det = np.load(R / "sites" / f"{sid}_det.npz")[layer]
            ref = np.load(R / "sites" / f"{sid}_ref.npz")["ref30"]
            conf = confusion_from([v for (o, l), v in pairs.items() if o != sid and l == layer])
            t = cell_terms(counts, det, ref, confusion=conf)
            bs = block_sums(t, np.ones(ref.shape, bool), 50)
            parts[(sid, layer)] = bs
            out[f"{sid}/{layer}"] = with_ci([bs], rng)
            print(sid, layer, {k: round(v["value"], 3) for k, v in out[f"{sid}/{layer}"].items()}, flush=True)
    coast = [s for s, v in SITES.items() if v[2] == "Coast Mountains"]
    for layer in ("vri", "scanfi"):
        ps = [parts[(s, layer)] for s in coast if (s, layer) in parts]
        if ps:
            out[f"coast_pooled/{layer}"] = with_ci(ps, rng)
            print("coast pooled", layer, {k: round(v["value"], 3) for k, v in out[f"coast_pooled/{layer}"].items()})
    bench = {}
    for site in ("Connaught", "BowSummit"):
        f = R / f"benchmark_{site}.npz"
        if f.exists():
            z = np.load(f)
            m = (z["valid"] >= 0) & (z["dem"] != -9999)
            bench[site] = (z["det"].astype(int)[m], z["valid"].astype(int)[m])
    for site in ("Connaught", "BowSummit"):
        f = R / f"benchmark_{site}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        mask = (z["valid"] >= 0) & (z["dem"] != -9999)
        ref = np.where(mask, z["valid"], 1).astype(int)
        for name in ("inputs", "all"):
            conf = confusion_from([v for o, v in bench.items() if o != site], merge0=True)
            t = cell_terms(z[f"counts_{name}"], z["det"].astype(int), ref, merge0=True, confusion=conf)
            out[f"{site}/{name}"] = with_ci([block_sums(t, mask, 58)], rng)
            print(site, name, {k: round(v["value"], 3) for k, v in out[f"{site}/{name}"].items()}, flush=True)
    (R / "statistics.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
