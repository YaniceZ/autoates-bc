"""Monte Carlo propagation of input and parameter uncertainty through AutoATES.

AutoATES returns one hard class per cell. Here the inputs (DEM, forest layer)
and the classifier thresholds are perturbed, the whole chain is rerun, and the
per-cell frequency of each class is kept. The result is a class probability
per cell instead of a single label.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from . import classifier as C
from . import pipeline as PL

# Spread of the classifier thresholds across the published calibrations:
# AutoATES v2.0 defaults (Toft et al. 2024, tables 1-2, and the released code)
# and the grid-search optima of Sykes et al. (2024, tables 4 and 8) for two
# study areas and four DEM products. Integer values, sampled uniformly.
PARAM_RANGES = {
    "SAT12": (18, 19),
    "SAT23": (28, 31),
    "SAT34": (37, 40),
    "AAT3": (33, 34),
    "CC2": (25, 40),      # OE23 in the papers
    "TREE1": (10, 20),    # canopy cover: code uses 10/50/65, the paper 20/55/75
    "TREE2": (50, 55),
    "TREE3": (65, 75),
}


def random_field(shape, corr_px, rng):
    """Zero-mean, unit-variance Gaussian field with Gaussian autocorrelation."""
    f = gaussian_filter(rng.standard_normal(shape), corr_px, mode="reflect")
    return (f - f.mean()) / f.std()


def corr_px_from_errors(err, max_lag=30):
    """Filter width (pixels) whose field has the same half-correlation lag."""
    e = err - np.nanmean(err)
    e = np.where(np.isfinite(e), e, 0.0)
    var = (e * e).mean()
    for lag in range(1, max_lag):
        rho = 0.5 * ((e[:, lag:] * e[:, :-lag]).mean() + (e[lag:] * e[:-lag]).mean()) / var
        if rho < 0.5:
            # for gaussian-filtered noise rho(h) = exp(-h^2 / (4 s^2))
            return lag / (2.0 * np.sqrt(np.log(2.0)))
    return max_lag / (2.0 * np.sqrt(np.log(2.0)))


@dataclass
class ErrorModel:
    """What is treated as uncertain in one experiment."""
    dem_sd: float | np.ndarray = 0.0     # metres; scalar or per-cell array
    dem_corr_px: float = 1.5
    dem_bias: float | np.ndarray = 0.0   # metres, subtracted from the DEM
    forest_sd: float = 0.0               # canopy cover points
    forest_corr_px: float = 2.0
    forest_alternatives: list = field(default_factory=list)  # other pcc rasters
    params: bool = False


def sample_params(rng, base: C.Params) -> C.Params:
    kw = {k: int(rng.integers(lo, hi + 1)) for k, (lo, hi) in PARAM_RANGES.items()}
    return base.with_(**kw)


def monte_carlo(dem, pcc, cell_size, model: ErrorModel, n, seed=0, base=None,
                progress=None):
    """Run ``n`` perturbed AutoATES runs. Returns counts with shape (5, H, W)."""
    rng = np.random.default_rng(seed)
    base = base or C.Params.for_forest("pcc")
    counts = np.zeros((5,) + dem.shape, np.uint16)
    perturb_dem = np.any(np.asarray(model.dem_sd) > 0)
    ws = None
    if not perturb_dem:
        from . import pra as P
        ws = P.windshelter(dem, cell_size, radius=max(1, int(round(60.0 / cell_size))))
    sources = [pcc] + list(model.forest_alternatives)
    for i in range(n):
        d = dem
        if perturb_dem:
            d = (dem - model.dem_bias
                 + model.dem_sd * random_field(dem.shape, model.dem_corr_px, rng)).astype(np.float32)
        f = sources[int(rng.integers(len(sources)))].astype(np.float32)
        if model.forest_sd > 0:
            f = f + model.forest_sd * random_field(dem.shape, model.forest_corr_px, rng)
        f = np.clip(np.round(f), 0, 100)
        p = sample_params(rng, base) if model.params else base
        ates = PL.run(d, f, cell_size, p, ws=ws)
        for k in range(5):
            counts[k] += ates == k
        if progress:
            progress(i + 1, n)
    return counts


def summarize(counts):
    """Per-cell probability, most likely class, confidence and entropy."""
    n = counts.sum(axis=0).astype(np.float32)
    prob = counts / n
    mode = prob.argmax(axis=0).astype(np.int8)
    conf = prob.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(prob > 0, prob * np.log(prob), 0)).sum(axis=0) / np.log(5)
    return {"prob": prob, "mode": mode, "confidence": conf, "entropy": ent.astype(np.float32)}


def prob_higher(counts, baseline):
    """Probability that the class is higher than the deterministic class."""
    n = counts.sum(axis=0).astype(np.float32)
    cum = np.cumsum(counts[::-1], axis=0)[::-1]   # cum[k] = runs with class >= k
    out = np.zeros(baseline.shape, np.float32)
    for k in range(4):
        m = baseline == k
        out[m] = cum[k + 1][m] / n[m]
    return out


def block_mode(a, k):
    """Aggregate a class raster by k x k majority; ties go to the higher class."""
    h, w = a.shape[0] // k, a.shape[1] // k
    blocks = a[: h * k, : w * k].reshape(h, k, w, k).transpose(0, 2, 1, 3).reshape(h, w, k * k)
    best = np.zeros((h, w), np.int8)
    best_n = np.zeros((h, w), np.int16)
    for c in range(5):
        cnt = (blocks == c).sum(axis=2)
        m = cnt >= best_n
        best[m] = c
        best_n[m] = cnt[m]
    return best
