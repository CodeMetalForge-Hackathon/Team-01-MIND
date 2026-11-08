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


SOUND_CLASSES = ["knock", "thud", "cough", "speaking", "other"]


def _per_node(v, node):
    """A threshold that is either one number or a {node: value, default: value} map."""
    if isinstance(v, dict):
        return v.get(node, v.get(str(node), v["default"]))
    return v


def classify_sound(f: dict, th: dict, node: int | None = None) -> tuple[str, float, str]:
    """Four classes for record_plot: knock, thud, cough, speaking (else "other").

    Builds on classify_rules (first-burst rise, so knock series count as knocks), then:
      * every sharp-onset sound is a knock or a thud, whatever its length (a fast knock series
        merges into one long sound): a knock on wood/door is bright, a thud (fist on a wall,
        a book dropped on the floor, a stamp) is dull and low-pitched. Brightness depends on
        the sensor (the S3 node reads ~2x brighter), so the cutoff can be set per node.
      * a short sound (under cough_min_dur_ms) with a fairly quick rise is percussive too:
        far nodes see a knock's start smeared out, but speech and coughs last longer.
      * voice -> speaking.
    """
    label, conf, why = classify_rules(f, th, rise_key="rise1_ms")
    sharp = label in ("knock_tap", "impact")
    short_hit = (label == "other" and f["dur_ms"] < th["cough_min_dur_ms"]
                 and f["rise1_ms"] <= th["hit_max_rise_ms"])
    if sharp or short_hit:
        if short_hit:
            conf, why = round(conf * 0.7, 2), f"short ({f['dur_ms']:.0f} ms), rise {f['rise1_ms']:.1f} ms"
        cut = _per_node(th["thud_max_centroid_hz"], node)
        if f["centroid"] < cut:
            return "thud", conf, f"sharp start, dull ({f['centroid']:.0f} Hz < {cut:.0f})"
        series = f" series of {f['n_bursts']}" if f["dur_ms"] > th["knock_max_dur_ms"] else ""
        return "knock", conf, f"{why},{series} bright ({f['centroid']:.0f} Hz)"
    if label == "voice":
        return "speaking", conf, why
    if label == "cough":
        return "cough", conf, why
    return "other", conf, why
