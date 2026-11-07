"""Reads config/array.yaml (schema from the agreed M0 doc).

STAND-IN: mirrors the teammate's base/dropnlisten/config.py and
speed_of_sound.py. Once her package is in this repo, import from it instead and
delete this file, so there is only one source of truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def speed_of_sound(medium="air", temperature_c=20.0, humidity_pct=50.0, salinity_ppt=0.0, depth_m=0.0):
    """Air: 331.4 + 0.6 T + 0.0124 RH.  Water: Medwin (salinity 0 = fresh water)."""
    T = temperature_c
    if medium == "air":
        return 331.4 + 0.6 * T + 0.0124 * humidity_pct
    if medium == "water":
        return (1449.2 + 4.6 * T - 0.055 * T ** 2 + 0.00029 * T ** 3
                + (1.34 - 0.01 * T) * (salinity_ppt - 35) + 0.016 * depth_m)
    raise ValueError(f"unknown medium {medium!r}")


def load_array(path="config/array.yaml"):
    p = Path(path)
    cfg = yaml.safe_load((p if p.is_absolute() else ROOT / p).read_text())
    sensors = sorted(cfg["sensors"], key=lambda s: s["ch"])
    cond = cfg.get("conditions", {})
    cfg["_xy"] = np.array([[s["x_m"], s["y_m"]] for s in sensors], float)
    cfg["_names"] = [s["name"] for s in sensors]
    cfg["_c_mps"] = speed_of_sound(cfg.get("medium", "air"), cond.get("temperature_c", 20), cond.get("humidity_pct", 50),
                                   cond.get("salinity_ppt", 0.0))
    cfg["_mode"] = "direction" if len(sensors) == 2 else "position"
    cfg["_path"] = str(path)
    return cfg


def front_vector(a_xy, b_xy, front="left"):
    """Unit normal to the A->B line on the configured side ('left' or 'right')."""
    e = (np.asarray(b_xy, float) - np.asarray(a_xy, float))
    e /= np.linalg.norm(e)
    left = np.array([-e[1], e[0]])
    return left if front == "left" else -left


def local_angle_to_compass(theta_deg, x_axis_bearing_deg=90.0):
    """Local CCW-from-+x angle -> compass bearing (clockwise from north)."""
    return (x_axis_bearing_deg - theta_deg) % 360.0


def compass_to_local_angle(bearing_deg, x_axis_bearing_deg=90.0):
    return (x_axis_bearing_deg - bearing_deg) % 360.0
