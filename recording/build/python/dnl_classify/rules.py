"""The handbook's rule classifier (section 11.2), on a feature dict from features.py."""
from __future__ import annotations

import math

PERSON = {"knock_tap", "voice", "cough", "breath"}


def classify_rules(f: dict, th: dict, rise_key: str = "rise_ms") -> tuple[str, float, str]:
    """rise_key="rise1_ms" uses the first burst's rise (handles merged knock series)."""
    d, r, m = f["dur_ms"], f[rise_key], f["mod_hz"]
    conf = round(0.5 + 0.45 * min(1.0, math.log10(max(f["snr"], 1.0)) / 2.0), 2)
    if d >= th["steady_min_dur_ms"] and m < th["steady_max_mod_hz"]:
        return "steady", conf, f"long ({d:.0f} ms) and steady"
    if r <= th["knock_max_rise_ms"] and d <= th["knock_max_dur_ms"]:
        return "knock_tap", conf, f"sharp start ({r:.1f} ms) and short ({d:.0f} ms)"
    if r <= th["knock_max_rise_ms"]:
        return "impact", conf, f"sharp start but long ({d:.0f} ms)"
    if th["voice_min_dur_ms"] <= d <= th["voice_max_dur_ms"] and th["voice_min_mod_hz"] <= m <= th["voice_max_mod_hz"]:
        return "voice", conf, f"{d:.0f} ms with a {m:.1f} Hz rhythm"
    if th["cough_min_dur_ms"] <= d <= th["cough_max_dur_ms"] and m < th["voice_min_mod_hz"]:
        return "cough", round(conf * 0.8, 2), f"short burst ({d:.0f} ms), slower start"
    return "other", round(conf * 0.6, 2), "no known pattern"
