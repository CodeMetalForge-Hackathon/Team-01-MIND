"""TDOA localisation: delay estimation, position solver, uncertainty, and the maps to show them.

One solver for the whole project (record_plot.py, live.py and session.py all call this).

1. Delays: GCC-PHAT (generalised cross-correlation with phase transform) between nodes, on the
   loudness envelope around the onsets, upsampled for sub-sample resolution.  Our nodes have very
   different front ends (nodes 1/2 clip the bottom half of the wave, node 3 has ~1/20 the gain), so
   raw waveforms don't look alike but their envelopes do.  PHAT whitens the spectrum so the peak is
   sharp and the slow rise/decay of the knock doesn't dominate.  On session1 (tape-measured truth)
   this cut the median error from 87 cm to 47 cm and the wild fixes (> 3 m off) from 8/26 to 0/26
   compared with threshold-crossing onsets.
2. Position: closed form first (Fang's method for 3 nodes, spherical-interpolation least squares for
   4+), then Gauss-Newton on (x, y, t0) from every closed-form root.  With exactly 3 nodes two
   hyperbola intersections can both fit perfectly; that case is flagged as ambiguous.
3. Uncertainty: covariance (c * sigma_t)^2 (J^T J)^-1 (the Cramer-Rao bound for equal timing error),
   95 % ellipse, and GDOP = sqrt(trace(cov)) / (c * sigma_t): how much the layout magnifies timing
   error at that spot (about 1.2 inside an equilateral triangle, much worse outside it).
4. Maps: likelihood heatmap (where the hyperbolas agree), the hyperbola of every node pair, and a
   GDOP map of the layout.

References: PySDR "TDOA" chapter (GCC-PHAT, upsampling, Fang closed form, Gauss-Newton, GDOP and
likelihood maps); dharmsen/tdoa-demo (pairwise hyperbola systems for 3 microphones).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

CHI2_95_2D = 5.991          # 95 % of a 2-D Gaussian lies inside sqrt(5.991) sigma
# Timing error per node, measured: on session1 the 95 % ellipses contain the tape-measured spot
# 26/26 times at 1.5 ms, 18/26 at 1.0 ms, 0/26 at the old 0.2 ms. Includes tape/position error.
SIGMA_T = 1.3e-3
MAX_GDOP = 9.0              # fixes worse than this are rejected (layout too poor at that spot)


def usable(sol, pos, margin=2.0, max_resid_m=0.05):
    """A fix worth showing: near the nodes, consistent (small residual), and geometry good enough."""
    if not sol:
        return False
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if not (min(xs) - margin <= sol["x"] <= max(xs) + margin and min(ys) - margin <= sol["y"] <= max(ys) + margin):
        return False
    return sol["resid_m"] <= max_resid_m and math.isfinite(sol["gdop"]) and sol["gdop"] <= MAX_GDOP


# ============================================================ delay estimation
def gcc_phat(a, b, fs, max_tau_s, upsample=16, phat=True, eps=1e-9):
    """Delay of a relative to b (seconds, > 0 means a heard it later) and the peak's sharpness.

    Cross-spectrum A * conj(B), divided by its magnitude (PHAT) so only phase is left, then inverse
    FFT zero-padded `upsample` times for sub-sample resolution.  Only lags within +-max_tau_s are
    searched (the sound can't take longer than the node spacing allows).  Quality = peak / RMS of
    the correlation: ~3 is noise, > 6 is a clean peak.
    """
    n = len(a) + len(b)
    X = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    if phat:
        X = X / (np.abs(X) + eps)
    cc = np.fft.irfft(X, n * upsample)
    m = max(int(max_tau_s * fs * upsample), 1)
    cc = np.concatenate([cc[-m:], cc[:m + 1]])
    k = int(np.argmax(cc))
    quality = float(cc[k] / (np.sqrt(np.mean(cc ** 2)) + 1e-12))
    return (k - m) / (fs * upsample), quality


def refine_times(group, signals, pos, c, pre_s=0.010, post_s=0.030, upsample=16, margin_s=0.001):
    """Arrival times (s) of one sound at each node, from GCC-PHAT delays between every node pair.

    group:   {node: event} with event["onset_us"] (from the onset detector; used to centre windows)
    signals: {node: (t_us, envelope)} on the shared clock
    Returns ({node: time_s}, {(i, j): (delay_s, quality)}).  The times keep the mean of the onsets,
    so they line up with the recording; only their differences matter for the position.
    """
    ids = sorted(n for n in group if n in signals and n in pos)
    if len(ids) < 2:
        return {n: group[n]["onset_us"] / 1e6 for n in group}, {}
    tc = float(np.mean([group[n]["onset_us"] for n in ids]))
    dt_us = min(float(np.median(np.diff(signals[n][0][:2000]))) for n in ids)
    fs = 1e6 / dt_us
    grid = tc + np.arange(-pre_s * 1e6, post_s * 1e6, dt_us)
    sig = {}
    for n in ids:
        t, env = signals[n]
        x = np.interp(grid, t, env)
        sig[n] = x - x.mean()
    pairs = {}
    for i, j in combinations(ids, 2):
        max_tau = math.dist(pos[i], pos[j]) / c + margin_s
        pairs[(i, j)] = gcc_phat(sig[i], sig[j], fs, max_tau, upsample)
    # least squares: t_i - t_j = delay_ij, weighted by peak quality; mean time pinned to tc
    A, rhs, w = [], [], []
    for (i, j), (d, q) in pairs.items():
        row = np.zeros(len(ids))
        row[ids.index(i)], row[ids.index(j)] = 1.0, -1.0
        A.append(row), rhs.append(d), w.append(max(q, 0.1))
    A.append(np.ones(len(ids)) / len(ids)), rhs.append(0.0), w.append(100.0)
    sw = np.sqrt(np.array(w))
    t_rel = np.linalg.lstsq(np.array(A) * sw[:, None], np.array(rhs) * sw, rcond=None)[0]
    times = {n: tc / 1e6 + float(t_rel[k]) for k, n in enumerate(ids)}
    for n in group:                                  # nodes without a position keep their onset
        times.setdefault(n, group[n]["onset_us"] / 1e6)
    return times, pairs


# ============================================================ position solver
def closed_form(P, t, c):
    """Candidate positions from the linearised range equations (no initial guess needed).

    With node 0 as reference, r_i = r_0 + d_i (d_i = c (t_i - t_0)).  Squaring and subtracting
    |u - s_0|^2 = r_0^2 gives, for i >= 1:  -2 (s_i - s_0) . u' = 2 r_0 d_i + d_i^2 - |s_i - s_0|^2.
    3 nodes (Fang): u' = p + r_0 q, and |u'| = r_0 gives a quadratic in r_0 -> up to 2 roots.
    4+ nodes: linear least squares in (u', r_0) (spherical interpolation).
    """
    P = np.asarray(P, float)
    t = np.asarray(t, float)
    S = P[1:] - P[0]
    d = c * (t[1:] - t[0])
    A = -2.0 * S
    b = d ** 2 - np.sum(S ** 2, axis=1)
    out = []
    if len(P) == 3:
        try:
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return out
        p, q = Ai @ b, Ai @ (2.0 * d)
        a2, a1, a0 = q @ q - 1.0, 2.0 * (p @ q), p @ p
        roots = [-a0 / a1] if abs(a2) < 1e-12 and abs(a1) > 1e-12 else np.roots([a2, a1, a0])
        for r0 in roots:
            if abs(np.imag(r0)) < 1e-9 and np.real(r0) >= 0:
                out.append(P[0] + p + float(np.real(r0)) * q)
    elif len(P) >= 4:
        M = np.column_stack([A, -2.0 * d])
        sol = np.linalg.lstsq(M, b, rcond=None)[0]
        out.append(P[0] + sol[:2])
    return out


def _gauss_newton(P, t, c, start, iters=40):
    x = np.array([start[0], start[1], float(np.min(c * t - np.hypot(*(P - start).T)))])
    for _ in range(iters):
        dx, dy = x[0] - P[:, 0], x[1] - P[:, 1]
        r = np.maximum(np.hypot(dx, dy), 1e-9)
        res = c * t - x[2] - r
        J = np.column_stack([-dx / r, -dy / r, -np.ones_like(r)])
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        x = x + step
        if np.linalg.norm(step[:2]) < 1e-7:
            break
    r = np.hypot(x[0] - P[:, 0], x[1] - P[:, 1])
    return x, float(np.sum((c * t - x[2] - r) ** 2))


def covariance(P, xy, c, sigma_t):
    """2x2 position covariance (m^2) at xy for timing error sigma_t, or None if singular."""
    P = np.asarray(P, float)
    dx, dy = xy[0] - P[:, 0], xy[1] - P[:, 1]
    r = np.maximum(np.hypot(dx, dy), 1e-9)
    J = np.column_stack([dx / r, dy / r, np.ones_like(r)])
    try:
        return (c * sigma_t) ** 2 * np.linalg.inv(J.T @ J)[:2, :2]
    except np.linalg.LinAlgError:
        return None


def solve(positions, times_s, c, sigma_t=SIGMA_T, ambiguity_m=0.05, max_range_x=20.0):
    """Source (x, y) and emission time t0 from arrival times at 3+ nodes.

    Returns dict: x, y, t0_s, dist (range to each node), resid_m (RMS range residual; 0 with exactly
    3 nodes), radius95 (long radius of the 95 % ellipse), cov (2x2), gdop, ambiguous (two
    intersections fit equally well), alt (the other one, or None), method.
    None if < 3 nodes, or if no position within max_range_x node spacings fits (impossible delays).
    """
    P = np.asarray(positions, float)
    t_abs = np.asarray(times_s, float)
    if len(t_abs) < 3:
        return None
    tref = t_abs.min()
    t = t_abs - tref
    centroid = P.mean(axis=0)
    span = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), 1.0)
    starts = [(s, "closed-form") for s in closed_form(P, t, c)]
    starts += [(centroid + span * np.array(v), "multi-start") for v in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
    cands = []
    for s, how in starts:
        x, cost = _gauss_newton(P, t, c, np.asarray(s, float))
        far = float(np.hypot(*(x[:2] - centroid)))
        # delays bigger than the spacing allows have no intersection: Gauss-Newton runs off to infinity
        if np.all(np.isfinite(x)) and far <= max_range_x * span:
            cands.append((cost, far, x, how))
    if not cands:
        return None                                   # no consistent position for these delays
    best_cost = min(cd[0] for cd in cands)
    tol = max(1e-6, 4 * best_cost) + (0.01 * c * sigma_t) ** 2
    good = [cd for cd in cands if cd[0] <= tol]
    good.sort(key=lambda cd: cd[1])                  # equally good: prefer the one nearer the nodes
    cost, _, x, how = good[0]
    alt = next((cd[2][:2] for cd in good[1:] if np.hypot(*(cd[2][:2] - x[:2])) > ambiguity_m), None)
    r = np.hypot(x[0] - P[:, 0], x[1] - P[:, 1])
    cov = covariance(P, x[:2], c, sigma_t)
    if cov is None:
        radius, gdop = float("inf"), float("inf")
    else:
        radius = float(math.sqrt(CHI2_95_2D * max(np.linalg.eigvalsh(cov).max(), 0.0)))
        gdop = float(math.sqrt(max(np.trace(cov), 0.0)) / (c * sigma_t))
    t0 = tref + x[2] / c
    resid = c * (t_abs - t0) - r
    return dict(x=float(x[0]), y=float(x[1]), t0_s=float(t0), dist=r.tolist(),
                resid_m=float(np.sqrt(np.mean(resid ** 2))), radius95=radius,
                cov=None if cov is None else cov.tolist(), gdop=gdop,
                ambiguous=alt is not None, alt=None if alt is None else [float(alt[0]), float(alt[1])],
                method=how)


# ============================================================ maps
def map_grid(pos, pad=1.0, n=160):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    gx = np.linspace(min(xs) - pad, max(xs) + pad, n)
    gy = np.linspace(min(ys) - pad, max(ys) + pad, n)
    return np.meshgrid(gx, gy)


def hyperbolas(positions, times_s, c, X, Y):
    """For each node pair (i, j): field |u - s_i| - |u - s_j| - c (t_i - t_j); its zero contour is
    the hyperbola the source must lie on."""
    P = np.asarray(positions, float)
    R = [np.hypot(X - p[0], Y - p[1]) for p in P]
    return [(i, j, R[i] - R[j] - c * (times_s[i] - times_s[j])) for i, j in combinations(range(len(P)), 2)]


def log_likelihood(positions, times_s, c, X, Y, sigma_t=SIGMA_T):
    """Gaussian log-likelihood of the source being at each grid point (0 at the best point).
    Bright ridges follow the hyperbolas; where they cross is the fix."""
    cost = sum(F ** 2 for _, _, F in hyperbolas(positions, times_s, c, X, Y))
    ll = -cost / (2 * len(positions) * (c * sigma_t) ** 2)
    return ll - ll.max()


def gdop_map(positions, X, Y):
    """GDOP at every grid point for this layout (independent of any sound): position error per unit
    of range error.  Low inside the triangle, rising fast outside and along the extended sides."""
    P = np.asarray(positions, float)
    U = [(X - p[0], Y - p[1]) for p in P]
    R = [np.maximum(np.hypot(ux, uy), 1e-6) for ux, uy in U]
    gx = [ux / r for (ux, _), r in zip(U, R)]
    gy = [uy / r for (_, uy), r in zip(U, R)]
    JtJ = np.empty(X.shape + (3, 3))
    JtJ[..., 0, 0] = sum(a * a for a in gx)
    JtJ[..., 0, 1] = JtJ[..., 1, 0] = sum(a * b for a, b in zip(gx, gy))
    JtJ[..., 1, 1] = sum(b * b for b in gy)
    JtJ[..., 0, 2] = JtJ[..., 2, 0] = sum(gx)
    JtJ[..., 1, 2] = JtJ[..., 2, 1] = sum(gy)
    JtJ[..., 2, 2] = len(P)
    with np.errstate(all="ignore"):
        det = np.linalg.det(JtJ)
        ok = np.abs(det) > 1e-9
        inv = np.zeros_like(JtJ)
        inv[ok] = np.linalg.inv(JtJ[ok])
        g = np.sqrt(np.clip(inv[..., 0, 0] + inv[..., 1, 1], 0, None))
    g[~ok] = np.inf
    return g


def ellipse_xy(sol, k=math.sqrt(CHI2_95_2D), n=72):
    """Outline (xs, ys) of the 95 % error ellipse of a solve() result, or None."""
    if not sol or sol.get("cov") is None:
        return None
    w, V = np.linalg.eigh(np.asarray(sol["cov"]))
    a = np.linspace(0, 2 * np.pi, n)
    pts = V @ (np.sqrt(np.maximum(w, 0))[:, None] * k * np.vstack([np.cos(a), np.sin(a)]))
    return sol["x"] + pts[0], sol["y"] + pts[1]


def draw_layout(ax, pos, gdop_levels=(1.5, 2, 3, 5), with_gdop=True, labels=True, pad=1.0):
    """Nodes, the triangle(s) between them (matplotlib.tri Delaunay), and GDOP contours behind."""
    import matplotlib.tri as mtri
    ids = sorted(pos)
    P = np.array([pos[n] for n in ids], float)
    if with_gdop and len(P) >= 3:
        X, Y = map_grid(pos, pad=pad, n=140)
        G = gdop_map(P, X, Y)
        cs = ax.contourf(X, Y, np.clip(G, 0, 8), levels=[0, *gdop_levels, 8], cmap="Greys", alpha=0.25)
        cl = ax.contour(X, Y, G, levels=list(gdop_levels), colors="0.45", linewidths=0.6)
        ax.clabel(cl, fmt=lambda v: f"GDOP {v:g}", fontsize=7)
    if len(P) >= 3:
        tri = mtri.Triangulation(P[:, 0], P[:, 1])
        ax.triplot(tri, color="0.35", lw=1.0, ls="--")
    ax.plot(P[:, 0], P[:, 1], "s", ms=12, color="0.25", zorder=5)
    if labels:
        for n, (x, y) in zip(ids, P):
            ax.annotate(f"node {n}", (x, y), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=9)
