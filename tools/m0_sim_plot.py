"""M0 simulation plot: how well the bearing and position solvers do on simulated knocks.

    python tools/m0_sim_plot.py                       # writes data/figures/m0_simulation.png
    python tools/m0_sim_plot.py --noise-us 50 --out m0.png

Four panels:
  A. Direction mode: example sources, solved arrows and +/-2 sigma wedges.
  B. Bearing error vs true angle, against the 5 deg target.
  C. Position mode: solved knocks and 95 % ellipses inside a 2 m triangle.
  D. Median position error vs timing noise, air vs water (same sampling).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dropnlisten.bearing import angle_diff_deg, bearing_from_delay  # noqa: E402
from dropnlisten.config import REPO_ROOT, Conditions  # noqa: E402
from dropnlisten.geo import compass_deg_to_local_dir, local_dir_to_compass_deg  # noqa: E402
from dropnlisten.sources.sim_source import simulate_arrivals  # noqa: E402
from dropnlisten.speed_of_sound import speed_of_sound  # noqa: E402
from dropnlisten.tdoa import ellipse_95, solve_position  # noqa: E402

# Okabe-Ito (colorblind-safe), per docs style rules.
BLUE, ORANGE, VERMILLION, SKY = "#0072B2", "#E69F00", "#D55E00", "#56B4E9"
SENSOR, INK, MUTED, GRID = "#333333", "#222222", "#777777", "#E5E5E5"

D_M = 0.5
PAIR = np.array([[-D_M / 2, 0.0], [D_M / 2, 0.0]])
TRI = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]])


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def bearing_trials(c, noise_s, n, rng):
    """True angle (deg from +x), bearing error (deg), reported sigma (deg)."""
    ang, err, sig = [], [], []
    for _ in range(n):
        r, a = rng.uniform(1, 3), rng.uniform(0, 180)
        src = r * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        t = simulate_arrivals(src, PAIR, c, noise_s, rng)
        res = bearing_from_delay(t[1] - t[0], PAIR[0], PAIR[1], c, timing_sigma_s=noise_s)
        true = local_dir_to_compass_deg(src[0], src[1], 90.0)
        ang.append(a)
        err.append(float(angle_diff_deg(res.bearing_deg, true)))
        sig.append(res.bearing_sigma_deg)
    return np.array(ang), np.array(err), np.array(sig)


def panel_bearing_map(ax, c, noise_s, rng):
    for k, a in enumerate(np.linspace(35, 145, 5)):
        r = 1.5 + 1.5 * ((k * 0.37) % 1)
        src = r * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        t = simulate_arrivals(src, PAIR, c, noise_s, rng)
        res = bearing_from_delay(t[1] - t[0], PAIR[0], PAIR[1], c, timing_sigma_s=noise_s)
        # +/-2 sigma wedge, drawn out to the source's range so arrow and X pair up
        th = np.linspace(res.bearing_deg - 2 * res.bearing_sigma_deg,
                         res.bearing_deg + 2 * res.bearing_sigma_deg, 30)
        ux, uy = compass_deg_to_local_dir(th, 90.0)
        ax.fill(np.r_[0, r * ux, 0], np.r_[0, r * uy, 0], color=BLUE, alpha=0.12, lw=0)
        ux, uy = compass_deg_to_local_dir(res.bearing_deg, 90.0)
        ax.annotate("", xy=(r * ux, r * uy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
        ax.plot(*src, marker="x", color=INK, ms=9, mew=2, ls="none")
    ax.plot(PAIR[:, 0], PAIR[:, 1], "s", color=SENSOR, ms=10)
    for i, (x, y) in enumerate(PAIR):
        ax.annotate(f"DNL-S{i + 1}", (x, y), xytext=(-6 if i == 0 else 6, -16),
                    textcoords="offset points", ha="right" if i == 0 else "left",
                    color=INK, fontsize=9)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.text(3.3, -0.25, "assumes sounds from the front (+y) side", ha="right",
            color=MUTED, fontsize=8)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-0.5, 3.4)
    ax.set_aspect("equal")
    style(ax, "x (m)", "y (m)")
    ax.set_title("A. Direction mode: arrow (solved) vs X (true);\n"
                 "the ±2σ wedge widens toward the sensor axis", loc="left", fontsize=10)


def panel_bearing_error(ax, c, rng, noise_list_s):
    colors = [VERMILLION, BLUE]
    angs = np.linspace(1, 179, 300)
    for noise_s, col in zip(noise_list_s, colors, strict=True):
        a, e, _ = bearing_trials(c, noise_s, 1500, rng)
        front = (a >= 30) & (a <= 150)
        hit_all = np.mean(np.abs(e) <= 5)
        hit_front = np.mean(np.abs(e[front]) <= 5)
        ax.plot(a, e, ".", color=col, ms=3, alpha=0.35)
        pred = np.degrees(c * np.sqrt(2) * noise_s / (D_M * np.sin(np.radians(angs))))
        ax.plot(angs, 2 * pred, color=col, lw=2)
        ax.plot(angs, -2 * pred, color=col, lw=2,
                label=f"{noise_s * 1e6:.0f} µs: {hit_all:.0%} within 5° "
                      f"({hit_front:.0%} at 30–150°)")
    ax.axhspan(-5, 5, color="#000000", alpha=0.06, lw=0)
    ax.text(178, 5.6, "±5° target", ha="right", color=MUTED, fontsize=8)
    ax.set_ylim(-30, 30)
    ax.set_xlim(0, 180)
    style(ax, "true angle from sensor axis (deg; 90 = straight ahead)", "bearing error (deg)")
    ax.legend(loc="lower center", fontsize=8, frameon=False, title="per-channel jitter "
              "(dots: trials, lines: predicted ±2σ)", title_fontsize=8)
    ax.set_title("B. 100 µs jitter misses the 5°/90% target; ≈40 µs meets it in front",
                 loc="left", fontsize=10)


def panel_position(ax, c, noise_s, rng):
    spots = [(1.0, 0.58), (0.6, 0.35), (1.4, 0.35), (1.0, 1.1), (0.45, 0.6), (1.55, 0.6)]
    errs = []
    for sx, sy in spots:
        pts, covs = [], []
        for _ in range(40):
            t = simulate_arrivals((sx, sy), TRI, c, noise_s, rng)
            res = solve_position(t, TRI, c, timing_sigma_s=noise_s)
            pts.append((res.x_m, res.y_m))
            covs.append(res.cov_m2)
        pts = np.array(pts)
        errs += list(np.hypot(pts[:, 0] - sx, pts[:, 1] - sy))
        ell = ellipse_95(np.mean(covs, axis=0)) + (sx, sy)
        ax.fill(ell[:, 0], ell[:, 1], color=BLUE, alpha=0.15, lw=0)
        ax.plot(ell[:, 0], ell[:, 1], color=BLUE, lw=1)
        ax.plot(pts[:, 0], pts[:, 1], "o", color=BLUE, ms=3, alpha=0.6)
        ax.plot(sx, sy, marker="x", color=INK, ms=9, mew=2, ls="none")
    closed = np.vstack([TRI, TRI[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=MUTED, lw=0.8, ls=":")
    ax.plot(TRI[:, 0], TRI[:, 1], "s", color=SENSOR, ms=10)
    for i, (x, y) in enumerate(TRI):
        ax.annotate(f"DNL-S{i + 1}", (x, y), xytext=(0, 9 if y > 0 else -16),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 2.0)
    ax.set_aspect("equal")
    style(ax, "x (m)", "y (m)")
    ax.set_title(f"C. Position mode: median error {np.median(errs) * 100:.1f} cm "
                 f"(target < 15 cm)\n{noise_s * 1e6:.0f} µs jitter; dots = solves, "
                 "ellipse = reported 95 %", loc="left", fontsize=10)


def panel_noise_sweep(ax, rng):
    noise = np.logspace(-5, -3, 9)
    cond = Conditions(temperature_c=20.0, salinity_ppt=0.0)
    lines = [("air", speed_of_sound("air", cond), BLUE), ("water", speed_of_sound("water", cond),
                                                             ORANGE)]
    for name, c, col in lines:
        med = []
        for ns in noise:
            e = []
            for _ in range(200):
                p = rng.uniform([0.3, 0.2], [1.7, 1.0])
                t = simulate_arrivals(p, TRI, c, ns, rng)
                r = solve_position(t, TRI, c, timing_sigma_s=ns)
                e.append(np.hypot(r.x_m - p[0], r.y_m - p[1]))
            med.append(np.median(e))
        ax.loglog(noise * 1e6, med, "-o", color=col, lw=2, ms=6)
        ax.annotate(f"{name} ({c:.0f} m/s)", (noise[-1] * 1e6, med[-1]), xytext=(-6, 6),
                    textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.axhline(0.15, color=MUTED, lw=1, ls="--")
    ax.text(11, 0.165, "15 cm target", color=MUTED, fontsize=8)
    ax.axvline(100, color=MUTED, lw=0.8, ls=":")
    ax.text(105, 0.004, "1 sample\n@10 kHz", color=MUTED, fontsize=8)
    style(ax, "per-channel timing jitter (µs)", "median position error (m)")
    ax.set_title("D. Water needs ~4× tighter timing for the same accuracy",
                 loc="left", fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noise-us", type=float, default=100.0, help="per-channel jitter, µs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "figures" / "m0_simulation.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    noise_s = args.noise_us * 1e-6
    c = speed_of_sound("air", Conditions(temperature_c=20.0, humidity_pct=50.0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    panel_bearing_map(axes[0, 0], c, noise_s, rng)
    panel_bearing_error(axes[0, 1], c, rng, [noise_s, 40e-6])
    panel_position(axes[1, 0], c, noise_s, rng)
    panel_noise_sweep(axes[1, 1], rng)
    fig.suptitle(f"Drop-and-Listen M0 simulation  ·  air c = {c:.1f} m/s  ·  "
                 f"pair spacing {D_M} m, 2 m triangle", fontsize=12, color=INK)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
