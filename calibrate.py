"""Calibration (Plot 9) and repeatability (Plot 4).

Protocol (one recording per taped spot, every knock from that spot):
    calibration  : 5-10 knocks at each of 1-5 taped spots  -> per-sensor delays
    jitter       : 20 knocks at one spot                   -> timing.sigma_s
    repeatability: 10+ knocks at one spot NOT used to fit  -> is the ellipse/wedge honest?
Each recording goes through pipeline.py with --true-spot, so its detections.csv
carries the spot. Delay of sensor k (sensor 0 = reference, 0 by definition):
    g_k = (t_k - t_0) - (tau_k - tau_0)    measured minus expected gap, per knock
    d_k = mean(g_k)                          standard error = std(g_k) / sqrt(n)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binom

import pipeline as P
import style
from arraycfg import ROOT, front_vector, load_array, local_angle_to_compass

CHI2_95 = style.CHI2_95_2D


# --------------------------------------------------------------------------- sim sessions
def make_spot_session(array, run, spot, n_knocks, seed=0, jitter_s=100e-6, delays_s=(0.0, 120e-6, -80e-6),
                      spacing_s=(0.9, 1.5)):
    """Synthesize a recording of n knocks at one taped spot (with per-sensor timing jitter
    and fixed delays baked in), then run it through pipeline.py WITHOUT calibration."""
    import sim
    rng = np.random.default_rng(seed)
    cfg = load_array(array)
    xy, c, fs = cfg["_xy"], cfg["_c_mps"], int(cfg["sampling"]["fs_hz"])
    sx, sy = P.find_spot(spot)
    times = 1.0 + np.cumsum(rng.uniform(*spacing_s, n_knocks))
    n = int((times[-1] + 1.5) * fs)
    t = np.arange(n) / fs
    x = np.stack([rng.normal(0, 5.0, n) + 3 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 6)) for _ in xy], axis=1)
    for t0 in times:
        src = sim.knock(rng, fs)
        for ch, s in enumerate(xy):
            d = np.hypot(s[0] - sx, s[1] - sy)
            arr = t0 + d / c + delays_s[ch] + rng.normal(0, jitter_s)
            sim.add_delayed(x[:, ch], src, arr, 450.0 / max(d, 0.3), fs)
    samples = np.clip(np.round(x + 2048), 0, 4095).astype(np.int16)
    (ROOT / "data/recordings").mkdir(parents=True, exist_ok=True)
    rec = f"data/recordings/{run}.npz"
    np.savez_compressed(ROOT / rec, samples=samples, fs_hz=fs)
    P.run_pipeline(rec, run, array, None, "__no_delays__", true_spot=spot, true_label="knock_tap",
                   start_time="2026-11-07T23:00:00+00:00")
    meta_p = ROOT / "data/runs" / run / "run.yaml"
    meta = yaml.safe_load(meta_p.read_text())
    meta.update(sim_truth=dict(jitter_s=jitter_s, delays_s=list(delays_s[:len(xy)])))
    meta_p.write_text(yaml.safe_dump(meta, sort_keys=False))
    return rec


# --------------------------------------------------------------------------- loading
def load_run(run):
    d = ROOT / "data/runs" / run
    meta = yaml.safe_load((d / "run.yaml").read_text())
    if not meta.get("true_spot"):
        raise SystemExit(f"{run}: no true_spot in run.yaml (run pipeline.py with --true-spot)")
    det = pd.read_csv(d / "detections.csv")
    return meta, det


def gap_table(runs, array):
    """One row per knock heard on every sensor: measured minus expected gap to sensor 0, per sensor."""
    cfg = load_array(array)
    xy, c = cfg["_xy"], cfg["_c_mps"]
    rows = []
    for run in runs:
        meta, det = load_run(run)
        sx, sy = P.find_spot(meta["true_spot"])
        tau = np.hypot(xy[:, 0] - sx, xy[:, 1] - sy) / c
        for sid, g in det.groupby("sound_id"):
            if g.ch.nunique() < len(xy):
                continue
            t = g.drop_duplicates("ch").set_index("ch").t_s.sort_index().to_numpy()
            gk = (t - t[0]) - (tau - tau[0])
            rows.append(dict(run=run, spot=meta["true_spot"], sound_id=sid,
                             **{f"g{k}": gk[k] for k in range(1, len(xy))}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fits
def fit_delays(gaps, n_ch):
    cols = [f"g{k}" for k in range(1, n_ch)]
    d = np.r_[0.0, gaps[cols].mean().to_numpy()]
    se = np.r_[0.0, (gaps[cols].std() / np.sqrt(len(gaps))).to_numpy()]
    before = gaps[cols].to_numpy()
    after = before - d[1:]
    return dict(delays_s=d, se_s=se, n=len(gaps), before=before, after=after,
                rms_before_us=float(np.sqrt(np.mean(before ** 2)) * 1e6),
                rms_after_us=float(np.sqrt(np.mean(after ** 2)) * 1e6))


def measure_jitter(gaps, n_ch):
    """Per-sensor jitter from knocks at one spot: spread of each gap / sqrt(2), averaged over pairs."""
    cols = [f"g{k}" for k in range(1, n_ch)]
    per_pair = gaps[cols].std().to_numpy() / np.sqrt(2)
    return dict(per_pair_s=per_pair, sigma_s=float(np.sqrt(np.mean(per_pair ** 2))), n=len(gaps))


def write_delays(fit, array, runs, path="config/channel_delays.yaml"):
    doc = dict(delays_s=[float(v) for v in fit["delays_s"]], se_s=[float(v) for v in fit["se_s"]],
               reference="sensor ch0 = 0", array=str(array), fitted_from=list(runs), n_knocks=int(fit["n"]),
               note="Refit after moving, rewiring or reflashing any sensor.")
    (ROOT / path).write_text(yaml.safe_dump(doc, sort_keys=False))
    return ROOT / path


# --------------------------------------------------------------------------- repeatability
def repeatability(run, array, delays_file):
    """Re-solve a held-out spot run with the fitted delays. Returns fixes + honesty stats."""
    meta, _ = load_run(run)
    _, fixes, _ = P.run_pipeline(meta["recording"], run + "-cal", array, None, delays_file,
                                 true_spot=meta["true_spot"], true_label="knock_tap",
                                 start_time=meta.get("start_time"))
    _, raw, _ = P.run_pipeline(meta["recording"], run + "-uncal", array, None, "__no_delays__",
                               true_spot=meta["true_spot"], true_label="knock_tap", start_time=meta.get("start_time"))
    return fixes, raw, honesty(fixes, array)


def honesty(fixes, array):
    """Fraction of knocks whose OWN 95% region contains the true spot, and a verdict."""
    cfg = load_array(array)
    n = len(fixes)
    if fixes.kind.iloc[0] == "position":
        d2 = []
        for r in fixes.itertuples():
            cov = np.array([[r.cov_xx, r.cov_xy], [r.cov_xy, r.cov_yy]])
            e = np.array([r.x_m - r.true_x_m, r.y_m - r.true_y_m])
            d2.append(float(e @ np.linalg.solve(cov, e)))
        d2 = np.array(d2)
        inside = d2 <= CHI2_95
        scale = float(np.sqrt(np.mean(d2) / 2.0))           # 1.0 = honest; >1 overconfident; <1 conservative
        err = np.hypot(fixes.x_m - fixes.true_x_m, fixes.y_m - fixes.true_y_m).to_numpy()
    else:
        mid = cfg["_xy"][:2].mean(axis=0)
        tb = local_angle_to_compass(np.degrees(np.arctan2(fixes.true_y_m - mid[1], fixes.true_x_m - mid[0])),
                                    cfg["geo"]["x_axis_bearing_deg"])
        err = ((fixes.bearing_deg - tb + 180) % 360 - 180).to_numpy()
        z = err / fixes.bearing_sigma_deg.to_numpy()
        inside = np.abs(z) <= 1.96
        scale = float(np.sqrt(np.mean(z ** 2)))
    k = int(inside.sum())
    k_min = int(binom.ppf(0.01, n, 0.95))                    # fewer inside than this: <1% chance if honest
    if k < k_min:
        verdict = "uncertainty is overconfident"
    elif scale < 0.5:
        verdict = "uncertainty is honest (but conservative: regions are about 2x too big)"
    else:
        verdict = "uncertainty is honest"
    return dict(n=n, inside=k, k_min=k_min, scale=scale, verdict=verdict, err=err,
                bias=float(np.mean(err)) if fixes.kind.iloc[0] == "bearing" else None)


# --------------------------------------------------------------------------- plots
def plot_calibration(fit, names, truth_delays=None, title=None):
    import matplotlib.pyplot as plt
    n_ch = len(names)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    rng = np.random.default_rng(0)
    for k in range(1, n_ch):
        for j, (vals, col, lab) in enumerate([(fit["before"][:, k - 1], style.WARNING, "before"),
                                              (fit["after"][:, k - 1], style.LABEL_COLOR["knock_tap"], "after")]):
            xpos = (k - 1) * 3 + j
            a1.scatter(xpos + rng.uniform(-0.18, 0.18, len(vals)), vals * 1e6, s=16, color=col, alpha=0.7,
                       label=f"{lab} subtracting delays" if k == 1 else None)
            a1.hlines(np.mean(vals) * 1e6, xpos - 0.3, xpos + 0.3, color="black", lw=1.5)
    a1.axhline(0, color="#999", lw=0.8)
    a1.set_xticks([(k - 1) * 3 + 0.5 for k in range(1, n_ch)])
    a1.set_xticklabels([f"{names[k]} vs {names[0]}" for k in range(1, n_ch)])
    a1.set_ylabel("measured − expected time gap (µs)")
    a1.set_title(f"Timing residuals: {fit['rms_before_us']:.0f} µs RMS before → {fit['rms_after_us']:.0f} µs after",
                 fontsize=10)
    a1.legend(fontsize=8, loc="best")
    x = np.arange(n_ch)
    a2.bar(x, fit["delays_s"] * 1e6, color=style.SENSOR, alpha=0.75, yerr=2 * fit["se_s"] * 1e6, capsize=5,
           label="fitted delay ± 2 SE")
    if truth_delays is not None:
        a2.scatter(x, np.asarray(truth_delays) * 1e6, marker="x", color="black", s=60, zorder=3, label="true (sim)")
    a2.axhline(0, color="#999", lw=0.8)
    a2.set_xticks(x)
    a2.set_xticklabels([f"{n}\n(reference)" if i == 0 else n for i, n in enumerate(names)])
    a2.set_ylabel("fitted delay (µs)")
    a2.set_title(f"Per-sensor delays from {fit['n']} calibration knocks", fontsize=10)
    a2.legend(fontsize=8)
    fig.suptitle(title or "", fontsize=12)
    fig.tight_layout()
    return fig


def plot_repeatability(fixes, raw, stats, array, title=None):
    import matplotlib.pyplot as plt
    from localize import ellipse_points
    cfg = load_array(array)
    names, xy = cfg["_names"], cfg["_xy"]
    kc = style.LABEL_COLOR["knock_tap"]
    if fixes.kind.iloc[0] == "position":
        fig, ax = plt.subplots(figsize=(7.5, 6.8))
        ax.scatter(xy[:, 0], xy[:, 1], marker="s", s=90, color=style.SENSOR, zorder=4)
        for n, (x, y) in zip(names, xy):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
        tx, ty = fixes.true_x_m.iloc[0], fixes.true_y_m.iloc[0]
        cov = fixes[["cov_xx", "cov_xy", "cov_yy"]].mean()
        e = ellipse_points(tx, ty, [[cov.cov_xx, cov.cov_xy], [cov.cov_xy, cov.cov_yy]])
        ax.fill(e[:, 0], e[:, 1], color=kc, alpha=style.UNCERTAINTY_FILL_ALPHA, label="average 95% ellipse (at true spot)")
        ax.plot(e[:, 0], e[:, 1], color=kc, lw=1)
        ax.scatter(raw.x_m, raw.y_m, s=28, facecolors="none", edgecolors="#999", label="solved, no calibration")
        ax.scatter(fixes.x_m, fixes.y_m, s=28, color=kc, label="solved, calibrated", zorder=3)
        ax.scatter([tx], [ty], marker="x", s=120, color="black", lw=2.5, label="true spot", zorder=5)
        ax.set_aspect("equal")
        pad = max(0.25, 3 * np.sqrt(max(cov.cov_xx, cov.cov_yy)))
        allx = np.r_[fixes.x_m, raw.x_m, tx]
        ally = np.r_[fixes.y_m, raw.y_m, ty]
        ax.set_xlim(allx.min() - pad, allx.max() + pad)
        ax.set_ylim(ally.min() - pad, ally.max() + pad)
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(color="#eee")
    else:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        err = stats["err"]
        mid = xy[:2].mean(axis=0)
        tb = local_angle_to_compass(np.degrees(np.arctan2(raw.true_y_m - mid[1], raw.true_x_m - mid[0])),
                                    cfg["geo"]["x_axis_bearing_deg"])
        err_raw = ((raw.bearing_deg - tb + 180) % 360 - 180).to_numpy()
        s2 = 2 * fixes.bearing_sigma_deg.mean()
        bins = np.linspace(min(err.min(), err_raw.min(), -s2) - 2, max(err.max(), err_raw.max(), s2) + 2, 25)
        ax.hist(err_raw, bins=bins, color="#BBBBBB", alpha=0.8, label="no calibration")
        ax.hist(err, bins=bins, color=kc, alpha=0.75, label="calibrated")
        ax.axvspan(-s2, s2, color=kc, alpha=style.UNCERTAINTY_FILL_ALPHA, label=f"average ±2σ wedge (±{s2:.1f}°)")
        ax.axvline(0, color="black", lw=2, label="true bearing")
        ax.set_xlabel("bearing error (degrees, solved − true)")
        ax.set_ylabel("knocks")
        ax.legend(fontsize=8)
    ax.set_title(title or "", fontsize=11)
    fig.tight_layout()
    return fig
