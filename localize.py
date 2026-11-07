"""TDOA localization: position fix (>=3 sensors) and bearing (2 sensors).

STAND-IN for the teammate's tdoa.py and bearing.py, written to the same math as
the agreed doc. Swap to her modules once her package is in this repo.

Covariance: sigma_t^2 * c^2 * (J^T J)^-1, J the Jacobian of the range
residuals with respect to (x, y, u), u = c * t_emit.
"""
from __future__ import annotations

import numpy as np


def _gauss_newton(s, t, c, theta, iters=40):
    for _ in range(iters):
        d = np.linalg.norm(s - theta[:2], axis=1).clip(1e-6)
        r = d - (c * t - theta[2])
        J = np.column_stack([(theta[0] - s[:, 0]) / d, (theta[1] - s[:, 1]) / d, np.ones_like(d)])
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        theta = theta + step
        if np.linalg.norm(step[:2]) < 1e-8:
            break
    d = np.linalg.norm(s - theta[:2], axis=1).clip(1e-6)
    r = d - (c * t - theta[2])
    J = np.column_stack([(theta[0] - s[:, 0]) / d, (theta[1] - s[:, 1]) / d, np.ones_like(d)])
    return theta, r, J


def solve_position(sensors_xy, t_s, c_mps, sigma_t_s):
    """Least-squares TDOA solve with several starting guesses.

    With exactly 3 sensors a sound outside the triangle can have two valid
    answers; like the M0 solver we keep the best fit, ties broken toward the
    middle of the array. Returns dict(x, y, cov [2x2] m^2, residual_m).
    """
    s = np.asarray(sensors_xy, float)
    t = np.asarray(t_s, float)
    if len(t) < 3:
        raise ValueError("position fix needs >= 3 sensors")
    mid = s.mean(axis=0)
    span = max(np.ptp(s[:, 0]), np.ptp(s[:, 1]), 0.5)
    starts = [mid] + [mid + span * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    best = None
    for p0 in starts:
        u0 = np.mean(c_mps * t - np.linalg.norm(s - p0, axis=1))
        theta, r, J = _gauss_newton(s, t, c_mps, np.array([p0[0], p0[1], u0]))
        score = (round(float(np.sqrt(np.mean(r ** 2))), 6), float(np.linalg.norm(theta[:2] - mid)))
        if best is None or score < best[0]:
            best = (score, theta, r, J)
    _, theta, r, J = best
    cov3 = (sigma_t_s * c_mps) ** 2 * np.linalg.pinv(J.T @ J)
    return {"x": float(theta[0]), "y": float(theta[1]), "cov": cov3[:2, :2],
            "residual_m": float(np.sqrt(np.mean(r ** 2)))}


def solve_bearing(a_xy, b_xy, t_a, t_b, c_mps, sigma_t_s, front_xy, model_sigma_deg=0.0):
    """Far-field direction from one sensor pair, resolved to the front side.

    cos(phi) = -c * (t_b - t_a) / d, phi measured from the A->B line.
    sigma_phi = c * sqrt(2) * sigma_t / (d * sin(phi))   (same as the M0 doc)
    Returns dict(theta_deg  local angle CCW from +x at the pair midpoint,
                 phi_deg    angle from the A->B line, 0..180,
                 sigma_deg, endfire_clipped).
    endfire_clipped: noise made the gap larger than the spacing allows, so the
    answer was pinned to the axis. Not the same thing as ADC clipping.
    """
    a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
    base = np.linalg.norm(b - a)
    e = (b - a) / base
    n = np.asarray(front_xy, float)
    raw = -c_mps * (t_b - t_a) / base
    clipped = bool(abs(raw) > 1.0)
    cos_phi = float(np.clip(raw, -1.0, 1.0))
    sin_phi = np.sqrt(1.0 - cos_phi ** 2)
    w = cos_phi * e + sin_phi * n
    theta = np.degrees(np.arctan2(w[1], w[0])) % 360.0
    sig_t = np.degrees(c_mps * np.sqrt(2) * sigma_t_s / (base * max(sin_phi, 1e-3)))
    return {"theta_deg": float(theta), "phi_deg": float(np.degrees(np.arccos(cos_phi))),
            "sigma_deg": float(min(np.hypot(sig_t, model_sigma_deg), 90.0)), "endfire_clipped": clipped}


def ellipse_points(x, y, cov, chi2=5.991, n=72):
    """Outline of the confidence ellipse (default 95 %) as [n, 2]."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, float))
    vals = np.clip(vals, 0, None)
    ang = np.linspace(0, 2 * np.pi, n)
    pts = vecs @ (np.sqrt(chi2 * vals)[:, None] * np.stack([np.cos(ang), np.sin(ang)]))
    return pts.T + np.array([x, y])
