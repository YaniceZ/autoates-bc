"""Flow-Py runout model with the forest detrainment module.

numba port of ``FlowPy_detrainment/flow_core.py::calculation_effect`` and
``flow_class.py::Cell`` as bundled with AutoATES v2.0 (Flow-Py: D'Amboise,
Neuhauser et al. 2022, GPL-3.0). The upstream code builds a Python object per
visited cell, which is why it is the slowest part of AutoATES. This port keeps
the same routing rules (order of visits, merging of pending cells, persistence,
flux redistribution, forest friction and detrainment) in flat arrays so that a
Monte Carlo ensemble is affordable on one desktop.

It is checked against the unmodified upstream functions in
``tests/test_port_matches_upstream.py``.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

SQRT2 = np.sqrt(2.0)
# 30 m/s expressed as an energy line height, as written upstream
NO_EFFECT_ZDELTA = 30.0 ** 2 / (np.sqrt(2.0) * 9.8)
MAX_FRICTION, MIN_FRICTION = 10.0, 2.0
MAX_DETRAIN, MIN_DETRAIN = 0.0003, 0.00001


@njit(cache=True)
def _grow_f(a):
    b = np.empty(a.size * 2, a.dtype)
    b[: a.size] = a
    return b


@njit(cache=True)
def _one_start(dem, valid, forest, cs, sr, sc, alpha, exp, flux_thr, max_z,
               pend, z_out, flux_out, cnt_out, zsum_out, ta_out, dis_out):
    h, w = dem.shape
    cap = 4096
    c_r = np.empty(cap, np.int32)
    c_c = np.empty(cap, np.int32)
    c_flux = np.empty(cap, np.float64)
    c_zd = np.empty(cap, np.float64)
    c_mind = np.zeros(cap, np.float64)
    c_gam = np.zeros(cap, np.float64)
    c_first = np.empty(cap, np.int32)   # first parent (list index), -1 for start
    c_head = np.empty(cap, np.int32)    # head of the parent linked list
    lcap = 8192
    l_par = np.empty(lcap, np.int32)
    l_next = np.empty(lcap, np.int32)
    nl = 0

    c_r[0] = sr
    c_c[0] = sc
    c_flux[0] = 1.0
    c_zd[0] = 0.0
    c_first[0] = -1
    c_head[0] = -1
    n = 1
    pend[sr, sc] = 0
    start_alt = dem[sr, sc]
    tan_a0 = 0.0

    zdn = np.empty((3, 3))
    pers = np.empty((3, 3))
    noflow = np.empty((3, 3))
    tb = np.empty((3, 3))
    rt = np.empty((3, 3))
    dist = np.empty((3, 3))
    s_r = np.empty(9, np.int32)
    s_c = np.empty(9, np.int32)
    s_f = np.empty(9, np.float64)
    s_z = np.empty(9, np.float64)

    idx = 0
    while idx < n:
        r = c_r[idx]
        c = c_c[idx]
        alt = dem[r, c]
        zd = c_zd[idx]
        fo = forest[r, c]
        is_start = idx == 0

        # --- z_delta of the neighbours (forest adds friction to alpha)
        if fo > 0:
            if zd < NO_EFFECT_ZDELTA:
                rest = MAX_FRICTION * fo
                k = (rest - MIN_FRICTION) / (0.0 - NO_EFFECT_ZDELTA)
                fr = max(MIN_FRICTION, k * zd + rest)
                a_calc = alpha + max(0.0, fr)
            else:
                a_calc = alpha + MIN_FRICTION
        else:
            a_calc = alpha
        tan_a = np.tan(np.deg2rad(a_calc))
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    ds = 0.0
                elif i != 1 and j != 1:
                    ds = SQRT2
                else:
                    ds = 1.0
                v = zd + (alt - dem[r - 1 + i, c - 1 + j]) - ds * cs * tan_a
                if v < 0:
                    v = 0.0
                if v > max_z:
                    v = max_z
                zdn[i, j] = v

        # --- persistence
        for i in range(3):
            for j in range(3):
                pers[i, j] = 0.0
                noflow[i, j] = 1.0
        if is_start or c_first[idx] == 0:
            for i in range(3):
                for j in range(3):
                    pers[i, j] = 1.0
        else:
            ln = c_head[idx]
            while ln != -1:
                p = l_par[ln]
                dx = c_c[p] - c
                dy = c_r[p] - r
                noflow[dy + 1, dx + 1] = 0.0
                mw = c_zd[p]
                if dx == -1:
                    if dy == -1:
                        pers[2, 2] += mw
                        pers[2, 1] += 0.707 * mw
                        pers[1, 2] += 0.707 * mw
                    if dy == 0:
                        pers[1, 2] += mw
                        pers[2, 2] += 0.707 * mw
                        pers[0, 2] += 0.707 * mw
                    if dy == 1:
                        pers[0, 2] += mw
                        pers[0, 1] += 0.707 * mw
                        pers[1, 2] += 0.707 * mw
                if dx == 0:
                    if dy == -1:
                        pers[2, 1] += mw
                        pers[2, 0] += 0.707 * mw
                        pers[2, 2] += 0.707 * mw
                    if dy == 1:
                        pers[0, 1] += mw
                        pers[0, 0] += 0.707 * mw
                        pers[0, 2] += 0.707 * mw
                if dx == 1:
                    if dy == -1:
                        pers[2, 0] += mw
                        pers[1, 0] += 0.707 * mw
                        pers[2, 1] += 0.707 * mw
                    if dy == 0:
                        pers[1, 0] += mw
                        pers[0, 0] += 0.707 * mw
                        pers[2, 0] += 0.707 * mw
                    if dy == 1:
                        pers[0, 0] += mw
                        pers[0, 1] += 0.707 * mw
                        pers[1, 0] += 0.707 * mw
                ln = l_next[ln]
        for i in range(3):
            for j in range(3):
                pers[i, j] *= noflow[i, j]

        # --- terrain routing
        tb_sum = 0.0
        for i in range(3):
            for j in range(3):
                if i != 1 and j != 1:
                    d = SQRT2 * cs
                else:
                    d = cs
                beta = np.arctan((alt - dem[r - 1 + i, c - 1 + j]) / d) + np.pi / 2
                v = np.tan(beta / 2)
                if zdn[i, j] <= 0 or pers[i, j] <= 0 or (i == 1 and j == 1):
                    v = 0.0
                tb[i, j] = v
                tb_sum += v
        rt_sum = 0.0
        if abs(tb_sum) > 0:
            te = 0.0
            for i in range(3):
                for j in range(3):
                    te += tb[i, j] ** exp
            for i in range(3):
                for j in range(3):
                    rt[i, j] = tb[i, j] ** exp / te
                    rt_sum += rt[i, j]
        else:
            for i in range(3):
                for j in range(3):
                    rt[i, j] = 0.0

        # --- forest detrainment
        rest = MAX_DETRAIN * fo
        k = (rest - MIN_DETRAIN) / (0.0 - NO_EFFECT_ZDELTA)
        detr = max(MIN_DETRAIN, k * zd + rest)

        # --- travel angle along the flow path
        if not is_start:
            md = 1e30
            ln = c_head[idx]
            while ln != -1:
                p = l_par[ln]
                dx = abs(c_c[p] - c)
                dy = abs(c_r[p] - r)
                dd = np.sqrt(float(dx * dx + dy * dy)) * cs + c_mind[p]
                if dd < md:
                    md = dd
                ln = l_next[ln]
            c_mind[idx] = md
            c_gam[idx] = np.rad2deg(np.arctan((start_alt - alt) / md))

        flux = max(0.0003, c_flux[idx] - detr)
        c_flux[idx] = flux

        # --- distribute flux
        for i in range(3):
            for j in range(3):
                dist[i, j] = 0.0
        if rt_sum > 0:
            pr = 0.0
            for i in range(3):
                for j in range(3):
                    pr += pers[i, j] * rt[i, j]
            for i in range(3):
                for j in range(3):
                    dist[i, j] = pers[i, j] * rt[i, j] / pr * flux
        count = 0
        mass = 0.0
        for i in range(3):
            for j in range(3):
                if 0 < dist[i, j] < flux_thr:
                    count += 1
                if dist[i, j] < flux_thr:
                    mass += dist[i, j]
        if mass > 0 and count > 0:
            for i in range(3):
                for j in range(3):
                    if dist[i, j] > flux_thr:
                        dist[i, j] += mass / count
                    if dist[i, j] < flux_thr:
                        dist[i, j] = 0.0
        tot = 0.0
        for i in range(3):
            for j in range(3):
                tot += dist[i, j]
        if tot < flux and count > 0:
            add = (flux - tot) / count
            for i in range(3):
                for j in range(3):
                    if dist[i, j] > flux_thr:
                        dist[i, j] += add

        m = 0
        for i in range(3):
            for j in range(3):
                if dist[i, j] > flux_thr:
                    s_r[m] = r - 1 + i
                    s_c[m] = c - 1 + j
                    s_f[m] = dist[i, j]
                    s_z[m] = zdn[i, j]
                    m += 1
        # sort ascending by (z_delta, flux, row, col), as upstream
        for a in range(1, m):
            b = a
            while b > 0:
                gt = False
                if s_z[b - 1] > s_z[b]:
                    gt = True
                elif s_z[b - 1] == s_z[b]:
                    if s_f[b - 1] > s_f[b]:
                        gt = True
                    elif s_f[b - 1] == s_f[b]:
                        if s_r[b - 1] > s_r[b]:
                            gt = True
                        elif s_r[b - 1] == s_r[b] and s_c[b - 1] > s_c[b]:
                            gt = True
                if not gt:
                    break
                tr, tc, tf, tz = s_r[b], s_c[b], s_f[b], s_z[b]
                s_r[b], s_c[b], s_f[b], s_z[b] = s_r[b - 1], s_c[b - 1], s_f[b - 1], s_z[b - 1]
                s_r[b - 1], s_c[b - 1], s_f[b - 1], s_z[b - 1] = tr, tc, tf, tz
                b -= 1

        for a in range(m):
            rr = s_r[a]
            cc = s_c[a]
            q = pend[rr, cc]
            if q > idx:
                # cell is already waiting in the list: merge
                c_flux[q] += s_f[a]
                if nl >= lcap:
                    l_par = _grow_f(l_par)
                    l_next = _grow_f(l_next)
                    lcap *= 2
                # append at the tail to keep upstream parent order
                l_par[nl] = idx
                l_next[nl] = -1
                t = c_head[q]
                while l_next[t] != -1:
                    t = l_next[t]
                l_next[t] = nl
                nl += 1
                if s_z[a] > c_zd[q]:
                    c_zd[q] = s_z[a]
                continue
            # new cell, needs a complete 3x3 neighbourhood
            if rr < 1 or cc < 1 or rr > h - 2 or cc > w - 2:
                continue
            ok = True
            for i in range(3):
                for j in range(3):
                    if not valid[rr - 1 + i, cc - 1 + j]:
                        ok = False
            if not ok:
                continue
            if n >= cap:
                c_r = _grow_f(c_r)
                c_c = _grow_f(c_c)
                c_flux = _grow_f(c_flux)
                c_zd = _grow_f(c_zd)
                c_mind = _grow_f(c_mind)
                c_gam = _grow_f(c_gam)
                c_first = _grow_f(c_first)
                c_head = _grow_f(c_head)
                cap *= 2
            if nl >= lcap:
                l_par = _grow_f(l_par)
                l_next = _grow_f(l_next)
                lcap *= 2
            c_r[n] = rr
            c_c[n] = cc
            c_flux[n] = s_f[a]
            c_zd[n] = s_z[a]
            c_mind[n] = 0.0
            c_gam[n] = 0.0
            c_first[n] = idx
            l_par[nl] = idx
            l_next[nl] = -1
            c_head[n] = nl
            nl += 1
            pend[rr, cc] = n
            n += 1
        idx += 1

    for q in range(n):
        r = c_r[q]
        c = c_c[q]
        pend[r, c] = -1
        if c_zd[q] > z_out[r, c]:
            z_out[r, c] = c_zd[q]
        if c_flux[q] > flux_out[r, c]:
            flux_out[r, c] = c_flux[q]
        cnt_out[r, c] += 1
        zsum_out[r, c] += c_zd[q]
        if c_gam[q] > ta_out[r, c]:
            ta_out[r, c] = c_gam[q]
        if c_mind[q] < dis_out[r, c]:
            dis_out[r, c] = c_mind[q]


@njit(parallel=True, cache=True)
def _run(dem, valid, forest, cs, rows, cols, alpha, exp, flux_thr, max_z, nchunk):
    h, w = dem.shape
    z = np.zeros((nchunk, h, w), np.float32)
    fl = np.zeros((nchunk, h, w), np.float32)
    cnt = np.zeros((nchunk, h, w), np.int32)
    zs = np.zeros((nchunk, h, w), np.float32)
    ta = np.zeros((nchunk, h, w), np.float32)
    di = np.full((nchunk, h, w), 10002.0, np.float32)
    ns = rows.size
    for k in prange(nchunk):
        pend = np.full((h, w), -1, np.int32)
        zk = np.zeros((h, w))
        fk = np.zeros((h, w))
        ck = np.zeros((h, w), np.int32)
        sk = np.zeros((h, w))
        tk = np.zeros((h, w))
        dk = np.full((h, w), 10002.0)
        for s in range(k, ns, nchunk):
            sr = rows[s]
            sc = cols[s]
            if sr < 1 or sc < 1 or sr > h - 2 or sc > w - 2:
                continue
            ok = True
            for i in range(3):
                for j in range(3):
                    if not valid[sr - 1 + i, sc - 1 + j]:
                        ok = False
            if not ok:
                continue
            _one_start(dem, valid, forest, cs, sr, sc, alpha, exp, flux_thr, max_z,
                       pend, zk, fk, ck, sk, tk, dk)
        z[k] = zk
        fl[k] = fk
        cnt[k] = ck
        zs[k] = sk
        ta[k] = tk
        di[k] = dk
    return z, fl, cnt, zs, ta, di


def flowpy(dem, release, cell_size, forest=None, alpha=24.0, exp=8, flux=0.003,
           max_z=270.0, nodata=-9999, nchunk=None):
    """Run Flow-Py from every release cell.

    Parameters
    ----------
    dem : 2-D float array
    release : 2-D array, cells > 0 are release cells
    forest : optional 2-D array of forest density scaled 0-1
    alpha, exp, flux, max_z : Flow-Py parameters. The defaults are the values
        Sykes et al. (2024) report for AutoATES (24 deg, 8, 0.003, 270 m).

    Returns a dict with ``z_delta``, ``flux``, ``cell_counts``, ``z_delta_sum``,
    ``fp_travel_angle`` and ``sl_travel_angle`` (the upstream output names).
    """
    import numba
    dem = np.ascontiguousarray(dem, dtype=np.float64)
    valid = np.ones(dem.shape, bool) if nodata is None else dem != nodata
    valid &= np.isfinite(dem)
    if forest is None:
        forest = np.zeros(dem.shape)
    forest = np.ascontiguousarray(forest, dtype=np.float64)
    rows, cols = np.where(np.asarray(release) > 0)
    if nchunk is None:
        nchunk = max(1, min(numba.get_num_threads(), rows.size))
    z, fl, cnt, zs, ta, di = _run(dem, valid, forest, float(cell_size),
                                  rows.astype(np.int32), cols.astype(np.int32),
                                  float(alpha), int(exp), float(flux), float(max_z),
                                  int(nchunk))
    return {
        "z_delta": z.max(axis=0),
        "flux": fl.max(axis=0),
        "cell_counts": cnt.sum(axis=0),
        "z_delta_sum": zs.sum(axis=0),
        "fp_travel_angle": ta.max(axis=0),
        "sl_travel_angle": di.min(axis=0),
    }
