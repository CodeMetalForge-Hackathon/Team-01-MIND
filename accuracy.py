"""Expected position accuracy across the floor (Plot 3) and error vs timing noise (Plot 10).

Fast method: at each grid point the TDOA Jacobian J (rows [(x-sx)/d, (y-sy)/d, 1]) gives
    cov = sigma_t^2 * c^2 * (J^T J)^-1      error = sqrt(trace of the x,y block)  (RMS radial)
Check: simulate sounds with timing noise at coarse grid points, solve, take the median error.
"""
from __future__ import annotations

import numpy as np

import style
from localize import solve_position

LOST_M = 10.0   # errors beyond this count as "lost" (solver diverged / wrong branch); caps the median


def grid_for(sensors, margin=1.5, step=0.05):
    s = np.asarray(sensors, float)
    lo, hi = s.min(0) - margin, s.max(0) + margin
    xs = np.arange(lo[0], hi[0] + step / 2, step)
    ys = np.arange(lo[1], hi[1] + step / 2, step)
    return xs, ys


def crlb_map(sensors, xs, ys, c, sigma_t):
    """RMS radial error (m) predicted by the Jacobian at every grid point. Shape [len(ys), len(xs)]."""
    s = np.asarray(sensors, float)
    X, Y = np.meshgrid(xs, ys)
    dx = X[..., None] - s[:, 0]
    dy = Y[..., None] - s[:, 1]
    d = np.sqrt(dx ** 2 + dy ** 2).clip(1e-6)
    J = np.stack([dx / d, dy / d, np.ones_like(d)], axis=-1)          # [ny, nx, k, 3]
    JtJ = np.einsum("...ki,...kj->...ij", J, J)
    det = np.linalg.det(JtJ)
    out = np.full(X.shape, np.inf)
    ok = np.abs(det) > 1e-10
    inv = np.linalg.inv(JtJ[ok])
    out[ok] = sigma_t * c * np.sqrt(inv[:, 0, 0] + inv[:, 1, 1])
    return out


def median_from_cov_factor():
    """Median radial error / RMS radial error for an isotropic 2-D Gaussian (= sqrt(ln 4)/sqrt(2))."""
    return np.sqrt(np.log(4)) / np.sqrt(2)          # 0.833


def mc_map(sensors, xs, ys, c, sigma_t, n=50, seed=0):
    """Median error of n simulated sounds per grid point (solver + timing noise)."""
    rng = np.random.default_rng(seed)
    s = np.asarray(sensors, float)
    out = np.zeros((len(ys), len(xs)))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            tau = np.hypot(s[:, 0] - x, s[:, 1] - y) / c
            errs = []
            for _ in range(n):
                sol = solve_position(s, tau + rng.normal(0, sigma_t, len(s)), c, sigma_t)
                errs.append(min(np.hypot(sol["x"] - x, sol["y"] - y), LOST_M))
            out[iy, ix] = np.median(errs)
    return out


def inside_hull(sensors, X, Y):
    from matplotlib.path import Path as MPath
    from scipy.spatial import ConvexHull
    s = np.asarray(sensors, float)
    hull = MPath(s[ConvexHull(s).vertices])
    return hull.contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)


def summarize_map(err, xs, ys, sensors, target_m=0.15):
    """Numbers for titles and the recommendation."""
    X, Y = np.meshgrid(xs, ys)
    med = err * median_from_cov_factor()
    ins = inside_hull(sensors, X, Y)
    worst = np.unravel_index(np.argmax(np.where(np.isfinite(med), med, -1)), med.shape)
    out_ok = np.isfinite(med) & ~ins
    step = float(xs[1] - xs[0])
    return dict(area_under_target_m2=float(np.sum(med <= target_m) * step * step),
                inside_median_m=float(np.median(med[ins])), inside_max_m=float(med[ins].max()),
                frac_area_over_target=float(np.mean(~(med <= target_m))),
                frac_inside_over_target=float(np.mean(~(med[ins] <= target_m))),
                outside_median_m=float(np.median(med[out_ok])) if out_ok.any() else np.nan,
                worst_xy=(float(xs[worst[1]]), float(ys[worst[0]])), target_m=target_m)


def error_vs_jitter(sensors, c, jitters_s, n=200, seed=1):
    """Median position error for random sounds inside the sensor hull, per timing noise level."""
    rng = np.random.default_rng(seed)
    s = np.asarray(sensors, float)
    lo, hi = s.min(0), s.max(0)
    pts = []
    while len(pts) < n:
        p = rng.uniform(lo, hi)
        if inside_hull(s, np.array([[p[0]]]), np.array([[p[1]]]))[0, 0]:
            pts.append(p)
    med = []
    for sj in jitters_s:
        errs = []
        for p in pts:
            tau = np.hypot(s[:, 0] - p[0], s[:, 1] - p[1]) / c
            sol = solve_position(s, tau + rng.normal(0, sj, len(s)), c, sj)
            errs.append(min(np.hypot(sol["x"] - p[0], sol["y"] - p[1]), LOST_M))
        med.append(np.median(errs))
    return np.array(med)


# --------------------------------------------------------------------------- plots
def plot_heatmaps(maps, sensors, names, xs, ys, titles, suptitle, vmax=1.0, target_m=0.15):
    """maps: list of median-error maps (m). Log color scale, target contour, equal aspect."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    s = np.asarray(sensors, float)
    fig, axes = plt.subplots(1, len(maps), figsize=(6.4 * len(maps), 5.6), squeeze=False)
    for ax, m, t in zip(axes[0], maps, titles):
        im = ax.pcolormesh(xs, ys, np.clip(m * 100, 0.5, vmax * 100), norm=LogNorm(1, vmax * 100),
                           cmap="viridis_r", shading="auto")
        cs = ax.contour(xs, ys, m * 100, levels=[target_m * 100], colors="white", linewidths=1.5)
        ax.clabel(cs, fmt=f"{target_m * 100:.0f} cm target", fontsize=8)
        ax.scatter(s[:, 0], s[:, 1], marker="s", s=90, color=style.SENSOR, edgecolors="white", zorder=3)
        for n, (x, y) in zip(names, s):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8,
                        color="white", fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
        ax.set_title(t, fontsize=10)
        fig.colorbar(im, ax=ax, label="median position error (cm, log)", shrink=0.85)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def plot_error_vs_jitter(jitters_s, curves, sample_period_s, title, target_m=0.15):
    """curves: {name: (median errors, color)}"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, (med, col) in curves.items():
        ax.loglog(np.asarray(jitters_s) * 1e6, med * 100, "o-", color=col, label=name, lw=2, ms=5)
    ax.axhline(target_m * 100, color="#666", ls="--", lw=1, label=f"{target_m * 100:.0f} cm target")
    ax.axvline(sample_period_s * 1e6, color="#666", ls=":", lw=1, label=f"1 sample at {1 / sample_period_s / 1000:.0f} kHz")
    ax.set_xlabel("timing noise per sensor, σ_t (µs)")
    ax.set_ylabel("median position error (cm)")
    ax.grid(True, which="both", color="#eee")
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
