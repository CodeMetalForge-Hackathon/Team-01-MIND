"""Drop-and-Listen mission control: one screen for responders, tabs for everything else.

    python dashboard.py --source sim                          # sim-direction (2 sensors)
    python dashboard.py --source sim --array config/array_triangle.yaml
    python dashboard.py --source replay --run <run>           # any data/runs/<run>/
    python dashboard.py --source live --run live              # follows live.py while the node streams
then open http://127.0.0.1:8050  (works on a phone/tablet on the same network with --host 0.0.0.0)

Live tab     : status bar, map (close-up meters / street map / satellite), SOUND SOURCES (human-like
               sounds grouped by location) with plain-language location, Acknowledge / Mark rescued, timeline.
ATAK         : Setup tab sends sound sources + sensors to ATAK phones (see atak.py).
Sounds log   : every sound, newest first.
Sensors      : per-sensor health, calibration and timing settings.
Evidence     : every figure in docs/figures (why the ellipse can be trusted).
Setup        : switch data source, process a new recording, display options.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

import style
from arraycfg import ROOT, compass_to_local_angle, front_vector, load_array
from localize import ellipse_points

HISTORY_S = 60.0
LOW_SNR_DB = 15.0
MAX_RESIDUAL_M = 0.10
WIDE_BEARING_DEG = 20.0
TICK_S = 0.25
PLAIN = {"knock_tap": "Knocking", "voice": "Voice / calling", "impact": "Impact (debris?)",
         "other": "Background", "structure_creak": "Structure creaking", "machinery": "Machinery", "nature": "Nature"}
LOG_COLS = ["time (s)", "sound", "confidence", "person?", "where", "accuracy", "repeated", "warnings"]
COMPASS8 = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


# =========================================================================== sources
class Source:
    """What every source provides. The views only call snapshot() and read the attributes."""
    mode: str
    cfg: dict
    c_mps: float
    medium: str
    duration_s: float | None
    meta: dict

    def snapshot(self, now_s: float):
        raise NotImplementedError


class PlaybackSource(Source):
    """Replays a run folder (fixes.csv + detections.csv + run.yaml) on a clock."""

    def __init__(self, run_dir: Path, array_path: str | None = None):
        self.run_dir = run_dir
        meta_p = run_dir / "run.yaml"
        self.meta = yaml.safe_load(meta_p.read_text()) if meta_p.exists() else {}
        self.cfg = load_array(array_path or self.meta.get("array", "config/array.yaml"))
        self.mode = self.cfg["_mode"]
        self.c_mps = float(self.meta.get("c_mps", self.cfg["_c_mps"]))
        self.medium = self.meta.get("medium", self.cfg.get("medium", "air"))
        self.fixes = pd.read_csv(run_dir / "fixes.csv")
        for col in ("human_likely", "repeated"):
            self.fixes[col] = self.fixes[col].astype(str).str.lower().eq("true")
        det_p = run_dir / "detections.csv"
        self.det = pd.read_csv(det_p) if det_p.exists() else pd.DataFrame(
            columns=["sound_id", "ch", "t_s", "sample_index", "peak", "snr", "clipped"])
        if len(self.det):
            self.det["clipped"] = self.det["clipped"].astype(str).str.lower().eq("true")
        t_det = self.det.groupby("sound_id").t_s.min()
        wall = pd.to_datetime(self.fixes.wall_time, utc=True, errors="coerce")
        t_wall = (wall - wall.min()).dt.total_seconds() + 1.0
        self.fixes["t_event"] = self.fixes.sound_id.map(t_det).fillna(t_wall)
        self.fixes = self.fixes.sort_values("t_event").reset_index(drop=True)
        last = float(self.fixes.t_event.max()) if len(self.fixes) else 0.0
        self.duration_s = float(max(self.meta.get("duration_s", 0) or 0, last + 3))

    def snapshot(self, now_s):
        f = self.fixes[self.fixes.t_event <= now_s]
        return f, self.det[self.det.sound_id.isin(f.sound_id)]


class LiveSource(PlaybackSource):
    """A run folder that live.py keeps rewriting. Reloads when its files change; the clock is
    wall time since the stream started, so everything heard so far is on screen."""

    def __init__(self, run_dir: Path, array_path: str | None = None):
        self._array_path, self._mtime = array_path, None
        self.run_dir = run_dir
        self.refresh()

    def refresh(self):
        import os
        m = os.stat(self.run_dir / "run.yaml").st_mtime
        if m != self._mtime:
            for _ in range(3):   # live.py replaces files atomically; retry if caught between two of them
                try:
                    PlaybackSource.__init__(self, self.run_dir, self._array_path)
                    break
                except (pd.errors.EmptyDataError, PermissionError, KeyError):
                    __import__("time").sleep(0.05)
            self._mtime = m
            self._t0 = pd.Timestamp(self.meta["start_time"]).timestamp()
        self.duration_s = 1e9

    def now(self):
        import time   # stream time; never behind what live.py has processed (matters for sped-up --fake)
        return max(time.time() - self._t0, float(self.meta.get("duration_s", 0) or 0))


def list_runs():
    return sorted(p.parent.name for p in (ROOT / "data/runs").glob("*/fixes.csv"))


def list_recordings():
    """Recordings from record.py (data/recordings) and the ones committed in recording/."""
    found = list((ROOT / "data/recordings").glob("*.npz")) + list((ROOT / "recording").rglob("*.npz"))
    return sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in found)


# =========================================================================== helpers
def rgba(label, alpha):
    return style.hex_to_rgba(style.LABEL_COLOR.get(label, "#BBBBBB"), alpha)


def cov_of(r):
    return np.array([[r.cov_xx, r.cov_xy], [r.cov_xy, r.cov_yy]])


def r95_cov(cov):
    return float(np.sqrt(style.CHI2_95_2D * np.linalg.eigvalsh(cov).max()))


def compass_word(bearing_deg):
    return COMPASS8[int(round((bearing_deg % 360) / 45)) % 8]


def pair_geometry(src):
    a, b = src.cfg["_xy"][0], src.cfg["_xy"][1]
    return (a + b) / 2, front_vector(a, b, src.cfg["bearing"]["front"])


def repeated_groups(fixes, max_gap_s=3.0):
    groups, cur = [], []
    for _, r in fixes.sort_values("t_event").iterrows():
        if bool(r.repeated) and (not cur or r.t_event - cur[-1].t_event <= max_gap_s):
            cur.append(r)
        else:
            if len(cur) >= 2:
                groups.append(pd.DataFrame(cur))
            cur = [r] if bool(r.repeated) else []
    if len(cur) >= 2:
        groups.append(pd.DataFrame(cur))
    return groups


def describe_group(g):
    gaps = np.diff(g.t_event.to_numpy())
    even = len(gaps) and gaps.std() / gaps.mean() < 0.15
    return (f"{len(g)} knocks, evenly spaced ({gaps.mean():.2f} s apart): very likely a person" if even
            else f"{len(g)} knocks repeated: likely a person")


def where_text(src, c):
    """Plain-language location of a contact."""
    xb = src.cfg["geo"]["x_axis_bearing_deg"]
    if c["kind"] == "position":
        xy = src.cfg["_xy"]
        if len(xy) == 1:
            return f"Near {src.cfg['_names'][0]} (one mic: no direction)", f"~{c['r95']:.0f} m (guessed range)"
        d = np.hypot(xy[:, 0] - c["x"], xy[:, 1] - c["y"])
        k = int(d.argmin())
        ang = np.degrees(np.arctan2(c["y"] - xy[k, 1], c["x"] - xy[k, 0]))
        return f"{d[k]:.1f} m {compass_word(xb - ang)} of {src.cfg['_names'][k]}", f"± {c['r95'] * 100:.0f} cm"
    return f"Towards the {compass_word(c['bearing'])} ({c['bearing']:.0f}°) from the sensor pair", f"± {2 * c['sigma']:.0f}°"


# =========================================================================== contacts
def build_contacts(fixes):
    """Group human-likely sounds that come from the same place into one sound source.
    Same place = statistically consistent given both uncertainties (99% gate).
    Position: inverse-covariance average, so more knocks -> tighter pin. Bearing: 1/sigma^2 average."""
    contacts = []
    for _, r in fixes[fixes.human_likely].iterrows():
        match = None
        for c in contacts:
            # same place = statistically consistent given BOTH uncertainties (99% gate), plus a few cm of slack
            if r.kind == "position" and c["kind"] == "position" and not np.isnan(r.x_m):
                d = np.array([r.x_m - c["x"], r.y_m - c["y"]])
                S = cov_of(r) + c["cov"] + np.eye(2) * 0.03 ** 2
                if d @ np.linalg.solve(S, d) <= 9.21:
                    match = c
                    break
            elif r.kind == "bearing" and c["kind"] == "bearing" and not np.isnan(r.bearing_deg):
                dd = abs((r.bearing_deg - c["bearing"] + 180) % 360 - 180)
                if dd <= 3 * np.hypot(r.bearing_sigma_deg, c["sigma"]) + 2:
                    match = c
                    break
        if match is None:
            match = dict(cid=f"c{int(r.sound_id)}", kind=r.kind, sounds=[], labels=Counter(), first_t=r.t_event,
                         info=[], conf=0.0, repeated=False, true=None)
            contacts.append(match)
        c = match
        c["sounds"].append(int(r.sound_id))
        c["labels"][r.label] += 1
        c["last_t"] = r.t_event
        c["conf"] = max(c["conf"], float(r.confidence))
        c["repeated"] |= bool(r.repeated)
        if not np.isnan(r.get("true_x_m", np.nan)):
            c["true"] = (r.true_x_m, r.true_y_m)
        if r.kind == "position":
            c["info"].append((np.array([r.x_m, r.y_m]), np.linalg.inv(cov_of(r))))
            info = sum(w for _, w in c["info"])
            cov = np.linalg.inv(info)
            x = cov @ sum(w @ p for p, w in c["info"])
            if all(np.array_equal(p, c["info"][0][0]) for p, _ in c["info"]):
                cov = np.linalg.inv(c["info"][0][1])   # one-mic fixes all sit on the node: repeats add no position info
            c.update(x=float(x[0]), y=float(x[1]), cov=cov, r95=r95_cov(cov))
        else:
            c["info"].append((r.bearing_deg, 1 / r.bearing_sigma_deg ** 2))
            ref = c["info"][0][0]
            num = sum(w * (ref + ((b - ref + 180) % 360 - 180)) for b, w in c["info"])
            den = sum(w for _, w in c["info"])
            c.update(bearing=float((num / den) % 360), sigma=float(1 / np.sqrt(den)))
    for i, c in enumerate(sorted(contacts, key=lambda c: c["first_t"]), start=1):
        c["num"] = i
    return contacts


def headline(c):
    """Honest wording: a contact is a sound SOURCE location; one person or several close together."""
    labs = set(c["labels"])
    if labs == {"knock_tap"}:
        return "KNOCKING · likely person"
    if labs == {"voice"}:
        return "CALLING · likely person"
    return "KNOCKS + VOICE · likely person"


def priority(c, acks):
    state = acks.get(c["cid"], "")
    return (state == "resolved", state == "ack", not c["repeated"], -len(c["sounds"]), -c["last_t"])


# =========================================================================== figures
def map_bounds(src):
    pts = [src.cfg["_xy"]]
    f = src.fixes
    if src.mode == "position":
        pts.append(f[["x_m", "y_m"]].dropna().to_numpy())
    pts.append(f[["true_x_m", "true_y_m"]].dropna().to_numpy())
    reach = None
    if src.mode == "direction":
        mid, front = pair_geometry(src)
        reach = max([3.0] + [float(np.linalg.norm(p - mid)) for p in pts[-1]])
        e = np.array([front[1], -front[0]])
        pts.append(np.array([mid + reach * front, mid + reach * e, mid - reach * e, mid - 0.6 * front]))
    p = np.vstack([q for q in pts if len(q)])
    return p.min(axis=0) - 0.5, p.max(axis=0) + 0.5, reach


def map_figure(src, fixes, contacts, acks, selected, now_s, show_truth, show_all, bounds):
    fig = go.Figure()
    lo, hi, reach = bounds
    xb = src.cfg["geo"]["x_axis_bearing_deg"]
    recent = fixes[now_s - fixes.t_event <= HISTORY_S]

    # faint history: every individual sound (non-human only when "show all sounds" is on)
    for _, r in recent.iterrows():
        if not r.human_likely and (not show_all or r.label in style.HIDDEN_BY_DEFAULT):
            continue
        a = 0.35 if r.human_likely else 0.25
        if r.kind == "position" and not np.isnan(r.x_m):
            fig.add_trace(go.Scatter(x=[r.x_m], y=[r.y_m], mode="markers", showlegend=False,
                                     marker=dict(size=7, color=rgba(r.label, a)), hoverinfo="text",
                                     hovertext=f"{PLAIN.get(r.label, r.label)} @ {r.t_event:.1f} s"))
        elif r.kind == "bearing" and not np.isnan(r.bearing_deg):
            mid, _ = pair_geometry(src)
            th = np.radians(compass_to_local_angle(r.bearing_deg, xb))
            tip = mid + reach * 0.6 * np.array([np.cos(th), np.sin(th)])
            fig.add_trace(go.Scatter(x=[mid[0], tip[0]], y=[mid[1], tip[1]], mode="lines", showlegend=False,
                                     line=dict(color=rgba(r.label, a), width=1.5), hoverinfo="skip"))

    # sound sources: big numbered pins with honest uncertainty
    for c in contacts:
        state = acks.get(c["cid"], "")
        lab = c["labels"].most_common(1)[0][0]
        alpha = 0.35 if state == "resolved" else 1.0
        col = rgba(lab, alpha)
        sel = c["cid"] == selected
        if c["kind"] == "position":
            e = ellipse_points(c["x"], c["y"], c["cov"])
            fig.add_trace(go.Scatter(x=e[:, 0], y=e[:, 1], fill="toself", mode="lines", hoverinfo="skip",
                                     fillcolor=rgba(lab, 0.18 * alpha), line=dict(color=col, width=1), showlegend=False))
            px, py = c["x"], c["y"]
        else:
            mid, _ = pair_geometry(src)
            th = np.radians(compass_to_local_angle(c["bearing"], xb))
            s2 = np.radians(min(2 * c["sigma"], 89.0))
            angs = np.linspace(th - s2, th + s2, 40)
            R = reach * 0.95
            fig.add_trace(go.Scatter(x=np.r_[mid[0], mid[0] + R * np.cos(angs), mid[0]],
                                     y=np.r_[mid[1], mid[1] + R * np.sin(angs), mid[1]], fill="toself", mode="lines",
                                     fillcolor=rgba(lab, 0.18 * alpha), line=dict(color=rgba(lab, 0.5 * alpha), width=1),
                                     hoverinfo="skip", showlegend=False))
            px, py = mid + reach * 0.82 * np.array([np.cos(th), np.sin(th)])
            fig.add_annotation(x=px, y=py, ax=mid[0], ay=mid[1], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=4, arrowcolor=col, text="")
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers+text", text=[f"<b>{c['num']}</b>"], textfont=dict(color="white", size=15),
            marker=dict(size=32 if sel else 26, color=col, line=dict(color="black" if sel else "white", width=4 if sel else 2)),
            hovertext=f"#{c['num']} {headline(c)}<br>{len(c['sounds'])} sounds", hoverinfo="text", showlegend=False))
        if show_truth and c["true"] is not None:
            fig.add_trace(go.Scatter(x=[c["true"][0]], y=[c["true"][1]], mode="markers", hoverinfo="text",
                                     hovertext="true spot (test)", showlegend=False,
                                     marker=dict(symbol="x-thin", size=13, line=dict(color="black", width=2.5))))

    xy = src.cfg["_xy"]
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=src.cfg["_names"],
                             textposition="bottom center", textfont=dict(size=12, color=style.SENSOR),
                             marker=dict(symbol="square", size=15, color=style.SENSOR), hoverinfo="text",
                             hovertext=[f"{n} ({x:.2f}, {y:.2f}) m" for n, (x, y) in zip(src.cfg["_names"], xy)],
                             showlegend=False))
    if src.mode == "direction":
        mid, front = pair_geometry(src)
        p = mid - 0.4 * front
        fig.add_annotation(x=p[0], y=p[1], text="<i>assumes sounds from the front side.</i>", showarrow=False,
                           font=dict(size=12, color="#555"))
    # compass rose
    fig.add_annotation(x=hi[0] - 0.25, y=hi[1] - 0.15, text="<b>N ↑</b>", showarrow=False, font=dict(size=14))
    fig.update_layout(margin=dict(l=50, r=10, t=10, b=40), plot_bgcolor="white", paper_bgcolor="white",
                      height=600, dragmode="pan",
                      xaxis=dict(title="east (m)", range=[lo[0], hi[0]], gridcolor="#eee", zeroline=False),
                      yaxis=dict(title="north (m)", range=[lo[1], hi[1]], gridcolor="#eee", zeroline=False,
                                 scaleanchor="x", scaleratio=1))
    return fig


def to_latlon(src, x, y):
    """Local meters -> lat/lon (same formula as the M0 geo.py): b = compass direction of the x axis."""
    g = src.cfg["geo"]
    b = np.radians(g["x_axis_bearing_deg"])
    x, y = np.asarray(x, float), np.asarray(y, float)
    east = x * np.sin(b) - y * np.cos(b)
    north = x * np.cos(b) + y * np.sin(b)
    lat = g["lat0"] + north / 111320.0
    lon = g["lon0"] + east / (111320.0 * np.cos(np.radians(g["lat0"])))
    return lat, lon

def node_sensors_panel(array_path="config/array.yaml"):
    """Node sensors: microphones (live), humidity/temperature (sets the speed of sound), thermal (status)."""
    from dash import html
    try:
        cfg = load_array(array_path)
        cond = cfg.get("conditions", {})
        t, rh, c = cond.get("temperature_c", 20), cond.get("humidity_pct", 50), cfg["_c_mps"]
        n = len(cfg["_xy"])
        env = f"{t} °C · {rh}% RH → speed of sound {c:.1f} m/s (used for every location)"
    except Exception as e:  # never let this panel break the dashboard
        n, env = "?", f"could not read {array_path}: {e}"
    row = lambda name, state, text: html.Div([html.B(name), html.Span(f"  [{state}]  "), html.Span(text)],
                                             style={"margin": "6px 0", "fontSize": 15})
    return html.Div(className="panel", children=[
        html.H2("Node sensors"),
        row("Microphones", "live", f"{n} sensors, timing → position"),
        row("Humidity + temperature", "config", env),
        html.Div("Set temperature_c / humidity_pct in config/array.yaml from the node's reading.", className="msg"),
        row("Thermal camera", "not connected", "confirms a warm body near a contact; integration in progress"),
    ])

ESRI_SAT = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def geo_figure(src, fixes, contacts, acks, selected, now_s, show_truth, basemap, bounds):
    """Same sound sources drawn on real map tiles: street map or satellite (needs internet for the tiles)."""
    fig = go.Figure()
    lo, hi, reach = bounds
    xb = src.cfg["geo"]["x_axis_bearing_deg"]
    for c in contacts:
        state = acks.get(c["cid"], "")
        lab = c["labels"].most_common(1)[0][0]
        alpha = 0.35 if state == "resolved" else 1.0
        if c["kind"] == "position":
            e = ellipse_points(c["x"], c["y"], c["cov"])
            px, py = c["x"], c["y"]
        else:
            mid, _ = pair_geometry(src)
            th = np.radians(compass_to_local_angle(c["bearing"], xb))
            s2 = np.radians(min(2 * c["sigma"], 89.0))
            angs = np.linspace(th - s2, th + s2, 40)
            R = reach * 0.95
            e = np.c_[np.r_[mid[0], mid[0] + R * np.cos(angs), mid[0]], np.r_[mid[1], mid[1] + R * np.sin(angs), mid[1]]]
            px, py = mid + reach * 0.82 * np.array([np.cos(th), np.sin(th)])
            la, lo_ = to_latlon(src, [mid[0], px], [mid[1], py])
            fig.add_trace(go.Scattermap(lat=la, lon=lo_, mode="lines", line=dict(color=rgba(lab, alpha), width=4),
                                        hoverinfo="skip", showlegend=False))
        la, lo_ = to_latlon(src, e[:, 0], e[:, 1])
        fig.add_trace(go.Scattermap(lat=la, lon=lo_, mode="lines", fill="toself", fillcolor=rgba(lab, 0.25 * alpha),
                                    line=dict(color=rgba(lab, alpha), width=2), hoverinfo="skip", showlegend=False))
        la, lo_ = to_latlon(src, [px], [py])
        sel = c["cid"] == selected
        fig.add_trace(go.Scattermap(lat=la, lon=lo_, mode="markers+text", text=[f"{c['num']}"],
                                    textfont=dict(color="white", size=14), textposition="middle center",
                                    marker=dict(size=30 if sel else 24, color=rgba(lab, alpha)),
                                    hovertext=f"#{c['num']} {headline(c)}", hoverinfo="text", showlegend=False))
        if show_truth and c["true"] is not None:
            la, lo_ = to_latlon(src, [c["true"][0]], [c["true"][1]])
            fig.add_trace(go.Scattermap(lat=la, lon=lo_, mode="markers", marker=dict(size=9, color="black"),
                                        hovertext="true spot (test)", hoverinfo="text", showlegend=False))
    xy = src.cfg["_xy"]
    la, lo_ = to_latlon(src, xy[:, 0], xy[:, 1])
    fig.add_trace(go.Scattermap(lat=la, lon=lo_, mode="markers+text", text=src.cfg["_names"], textposition="bottom right",
                                textfont=dict(size=12, color="white" if basemap == "satellite" else "#222"),
                                marker=dict(size=14, color=style.SENSOR), hoverinfo="text",
                                hovertext=src.cfg["_names"], showlegend=False))
    la, lo_ = to_latlon(src, [lo[0], hi[0]], [lo[1], hi[1]])
    span_m = float(max(hi - lo))
    # web-map zoom so the whole area (span_m) fits ~700 px; tiles get blurry past their native ~zoom 19
    zoom = float(np.clip(np.log2(156543.03 * np.cos(np.radians(la.mean())) * 700 / span_m), 3, 19))
    layers = [dict(sourcetype="raster", source=[ESRI_SAT], below="traces")] if basemap == "satellite" else []
    fig.update_layout(map=dict(style="white-bg" if basemap == "satellite" else "open-street-map", layers=layers,
                               center=dict(lat=float(la.mean()), lon=float(lo_.mean())), zoom=zoom),
                      margin=dict(l=0, r=0, t=0, b=0), height=600, uirevision=f"{basemap}-{src.run_dir.name}")
    return fig


def timeline_figure(src, fixes, now_s):
    fig = go.Figure()
    for _, r in fixes.iterrows():
        hidden = r.label in style.HIDDEN_BY_DEFAULT
        col = rgba(r.label, 0.45 if hidden else max(style.LABEL_ALPHA.get(r.label, 0.4), 0.5))
        fig.add_trace(go.Scatter(x=[r.t_event], y=[0], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=f"{PLAIN.get(r.label, r.label)} {r.confidence:.2f} @ {r.t_event:.1f} s",
                                 marker=dict(symbol="line-ns-open", size=34 if r.human_likely else 22, color=col,
                                             line=dict(color=col, width=4 if r.human_likely else 2))))
    for g in repeated_groups(fixes):
        a, b = g.t_event.min() - 0.15, g.t_event.max() + 0.15
        fig.add_shape(type="path", path=f"M {a},0.45 L {a},0.65 L {b},0.65 L {b},0.45",
                      line=dict(color=style.LABEL_COLOR["knock_tap"], width=2))
        fig.add_annotation(x=(a + b) / 2, y=0.7, text=f"<b>{describe_group(g)}</b>", showarrow=False,
                           yanchor="bottom", font=dict(color=style.LABEL_COLOR["knock_tap"], size=13))
    for lab in style.LABEL_ORDER:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=PLAIN[lab],
                                 marker=dict(symbol="square", size=10, color=style.LABEL_COLOR[lab])))
    fig.add_vline(x=now_s, line=dict(color="#999", dash="dot", width=1))
    fig.update_layout(height=170, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=50, r=10, t=10, b=35),
                      xaxis=dict(title="time since sensors were placed (s)",
                                 range=[0, src.duration_s or max(now_s, 30)], gridcolor="#eee"),
                      yaxis=dict(range=[-0.8, 1.35], visible=False),
                      legend=dict(orientation="h", yanchor="top", y=1.0, x=0, xanchor="left", font=dict(size=11)))
    return fig


# =========================================================================== panels
def status_bar(src, fixes, contacts, acks, now_s):
    open_c = [c for c in contacts if acks.get(c["cid"]) != "resolved"]
    last = f"last sound {now_s - fixes.t_event.iloc[-1]:.0f} s ago" if len(fixes) else "no sounds yet"
    if not contacts:
        cls, state, sub = "st-listening", "LISTENING", f"No person detected yet · {len(fixes)} sounds heard · {last}"
    elif not open_c:
        cls, state, sub = "st-resolved", "ALL CONTACTS RESOLVED", f"{len(contacts)} marked rescued · {last}"
    else:
        likely = any(c["repeated"] or len(c["sounds"]) >= 2 for c in open_c)
        cls = "st-likely" if likely else "st-possible"
        state = "PERSON LIKELY" if likely else "POSSIBLE PERSON"
        new = sum(1 for c in open_c if not acks.get(c["cid"]))
        sub = (f"{len(open_c)} open sound source{'s' if len(open_c) > 1 else ''}"
               + (f" · {new} not yet acknowledged" if new else "") + f" · {last}")
    return cls, state, sub


def contact_cards(src, contacts, acks, selected, now_s, html):
    if not contacts:
        return [html.H2("Sound sources"),
                html.Div(["No person detected yet.", html.Br(),
                          "Knocks and voices that sound human will appear here as numbered sound sources."],
                         className="empty")]
    cards = [html.H2(f"Sound sources ({len(contacts)})"),
             html.Div("Each card is one place human-like sounds come from. Two people closer together than "
                      "the accuracy shown appear as one source.", className="meta", style={"margin": "0 4px 10px"})]
    for c in sorted(contacts, key=lambda c: priority(c, acks)):
        state = acks.get(c["cid"], "")
        lab = c["labels"].most_common(1)[0][0]
        where, acc = where_text(src, c)
        n = len(c["sounds"])
        pattern = " in a regular pattern" if c["repeated"] else ""
        what = (f"Knocked {c['labels']['knock_tap']}×{pattern}" if c["labels"].get("knock_tap") else "") + \
               (" · " if c["labels"].get("knock_tap") and c["labels"].get("voice") else "") + \
               (f"Voice heard {c['labels']['voice']}×" if c["labels"].get("voice") else "")
        ago = max(0, int(now_s - c["last_t"]))
        cls = "card " + ("resolved" if state == "resolved" else "ack" if state == "ack" else "new") + \
              (" selected" if c["cid"] == selected else "")
        status = {"resolved": "✔ Marked rescued", "ack": "✔ Acknowledged"}.get(state, "NEW")
        cards.append(html.Div(className=cls, style={"borderLeftColor": style.LABEL_COLOR.get(lab, "#0072B2")}, children=[
            html.Div(f"#{c['num']}  {headline(c)}", className="head"),
            html.Div(where, className="where"),
            html.Div(f"{what} · last heard {ago} s ago · accuracy {acc} · confidence {c['conf']:.0%}", className="meta"),
            html.Div(status, className="meta", style={"fontWeight": 700, "marginTop": "4px"}),
            html.Div(className="btns", children=[
                html.Button("Locate on map", id={"type": "locate", "cid": c["cid"]}),
                html.Button("Acknowledge", id={"type": "ack", "cid": c["cid"]}, disabled=bool(state)),
                html.Button("Undo" if state == "resolved" else "Mark rescued",
                            id={"type": "resolve", "cid": c["cid"]}),
            ]),
        ]))
    return cards


def sounds_log(src, fixes, det):
    rows = []
    for _, r in fixes.iloc[::-1].iterrows():
        d = det[det.sound_id == r.sound_id]
        warns = []
        if len(d) and (d.snr < LOW_SNR_DB).any():
            warns.append("low signal")
        if len(d) and d.clipped.any():
            warns.append("clipped")
        if r.kind == "position" and not np.isnan(r.residual_m) and r.residual_m > MAX_RESIDUAL_M:
            warns.append("sensors disagree")
        if r.kind == "bearing" and r.bearing_sigma_deg > WIDE_BEARING_DEG:
            warns.append("near sensor axis")
        if r.kind == "position" and not np.isnan(r.x_m):
            c = dict(kind="position", x=r.x_m, y=r.y_m, r95=r95_cov(cov_of(r)))
        else:
            c = dict(kind="bearing", bearing=r.bearing_deg, sigma=r.bearing_sigma_deg)
        where, acc = where_text(src, c)
        rows.append({"time (s)": round(r.t_event, 2), "sound": PLAIN.get(r.label, r.label),
                     "confidence": f"{r.confidence:.0%}", "person?": "yes" if r.human_likely else "",
                     "where": where, "accuracy": acc, "repeated": "yes" if r.repeated else "",
                     "warnings": ", ".join(warns)})
    return rows


def sensors_panel(src, det, html):
    names = src.cfg["_names"]
    cards = []
    for k, n in enumerate(names):
        d = det[det.ch == k] if len(det) else det
        snr = f"{d.snr.median():.0f} dB median" if len(d) else "n/a"
        clip = int(d.clipped.sum()) if len(d) else 0
        lastt = f"{d.t_s.max():.1f} s" if len(d) else "not yet"
        x, y = src.cfg["_xy"][k]
        cards.append(html.Div(className="panel sensor", children=[
            html.B(n), html.Div(f"position ({x:.2f}, {y:.2f}) m"),
            html.Div(f"heard {len(d)} sounds, last at {lastt}"),
            html.Div(f"signal: {snr}", className="warn" if len(d) and d.snr.median() < LOW_SNR_DB else ""),
            html.Div(f"clipped sounds: {clip}", className="warn" if clip else ""),
        ]))
    delays = ROOT / "config/channel_delays.yaml"
    cal = yaml.safe_load(delays.read_text()) if delays.exists() else None
    meta = src.meta
    cards.append(html.Div(className="panel sensor", children=[
        html.B("Setup"),
        html.Div(f"mode: {src.mode} ({len(names)} sensors)"),
        html.Div(f"medium: {src.medium}, speed of sound {src.c_mps:.1f} m/s"),
        html.Div(f"timing noise assumed: {src.cfg['timing']['sigma_s'] * 1e6:.0f} µs"),
        html.Div(f"sample rate: {meta.get('fs_hz', src.cfg['sampling']['fs_hz'])} Hz"),
        html.Div(f"map origin: {src.cfg['geo']['lat0']:.5f}, {src.cfg['geo']['lon0']:.5f}"),
        html.Div(("calibration: " + ", ".join(f"{v * 1e6:+.0f} µs" for v in cal["delays_s"])
                  + f" from {cal.get('n_knocks', '?')} knocks") if cal else "calibration: none yet (run 02_calibration)",
                 className="" if cal else "warn"),
        html.Div(f"labels by: {meta.get('labeler', 'sim truth')}"),
    ]))
    return cards


def evidence_panel(html):
    out = []
    captions = {
        "plot1": "Detection tuning: where each knock's onset is placed.",
        "plot3": "Accuracy across the floor: pins are trustworthy inside the sensor triangle.",
        "plot4": "Repeatability: do knocks land inside their own 95% region? (honest uncertainty)",
        "plot5": "Classifier on sessions it never saw: how often a person is missed.",
        "plot6": "What each sound looks like.",
        "plot7": "Two features that separate voices from knocks.",
        "plot9": "Calibration: fitted per-sensor delays shrink timing errors.",
        "plot10": "Error vs timing noise, air vs water.",
    }
    for p in sorted((ROOT / "docs/figures").glob("*.png")):
        b64 = base64.b64encode(p.read_bytes()).decode()
        cap = next((v for k, v in captions.items() if p.stem.startswith(k + "_") or p.stem == k), "")
        out.append(html.Div(className="panel", children=[
            html.Img(src=f"data:image/png;base64,{b64}"),
            html.Div([html.B(p.stem), " · ", cap], className="cap")]))
    return out or [html.Div("No figures yet: run the notebooks to create docs/figures/*.png", className="empty")]


# =========================================================================== app
def build_app(initial_run: str, array_override: str | None = None, start_tab: str = "live"):
    from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

    cache: dict[str, Source] = {}
    atak_state: dict = {}

    def get_src(run):
        if run not in cache:
            d = ROOT / "data/runs" / run
            meta = yaml.safe_load((d / "run.yaml").read_text()) if (d / "run.yaml").exists() else {}
            cls = LiveSource if (meta or {}).get("source") == "live" else PlaybackSource
            cache[run] = cls(d, array_override if run == initial_run else None)
        src = cache[run]
        if isinstance(src, LiveSource):
            src.refresh()
        return src

    app = Dash(__name__, title="Drop-and-Listen", assets_folder=str(ROOT / "assets"),
               suppress_callback_exceptions=True)
    app.layout = html.Div([
        html.Div(id="statusbar", className="statusbar st-listening", children=[
            html.Div("LISTENING", id="state", className="state"), html.Div(id="sub", className="sub"),
            html.Div(className="right ctl", children=[
                html.Span(id="source-chip", className="chip"),
                html.Span(id="clock", className="chip"),
                html.Button("Pause", id="play"), html.Button("Restart", id="restart"),
                html.Button("Show all", id="all"),
                dcc.RadioItems(id="speed", options=[{"label": f" {s}×", "value": s} for s in (1, 5, 10)], value=1,
                               inline=True, inputStyle={"marginLeft": "8px"}, labelStyle={"color": "inherit"}),
            ]),
        ]),
        html.Div(id="hazard"),
        dcc.Tabs(id="tabs", value=start_tab, className="tabs", children=[
            dcc.Tab(label="Live", value="live", className="tab", children=[
                html.Div(className="main", children=[
                    html.Div(className="panel mapwrap", children=[
                        dcc.RadioItems(id="basemap", value="grid", inline=True, className="basemap",
                                       inputStyle={"marginRight": "5px", "marginLeft": "14px"},
                                       options=[{"label": "Close-up (meters)", "value": "grid"},
                                                {"label": "Street map", "value": "street"},
                                                {"label": "Satellite", "value": "satellite"}]),
                        dcc.Graph(id="map", config={"displayModeBar": False, "scrollZoom": True})]),
                    html.Div(id="contacts", className="panel contacts"),
                ]),
                html.Div(className="main", children=[html.Div(className="panel", style={"flex": "1"}, children=[
                    dcc.Graph(id="timeline", config={"displayModeBar": False})])]),
            ]),
            dcc.Tab(label="Sounds log", value="log", className="tab", children=[
                html.Div(className="main", children=[html.Div(className="panel", style={"flex": "1"}, children=[
                    dash_table.DataTable(id="log", page_size=15, data=[],
                                         columns=[{"name": c, "id": c} for c in LOG_COLS], sort_action="native", filter_action="native",
                                         style_cell={"fontSize": 15, "padding": "8px", "textAlign": "left"},
                                         style_header={"fontWeight": 700},
                                         style_data_conditional=[{"if": {"filter_query": '{person?} = "yes"'},
                                                                  "backgroundColor": "#e6f2fa"},
                                                                 {"if": {"column_id": "warnings"}, "color": "#b45309",
                                                                  "fontWeight": 700}])])])]),
            dcc.Tab(label="Sensors", value="sensors", className="tab", children=[html.Div(id="sensors", className="grid")]),
            dcc.Tab(label="Evidence", value="evidence", className="tab", children=[
                html.Div(id="evidence", className="evidence")]),
            dcc.Tab(label="Setup", value="setup", className="tab", children=[
                html.Div(className="setup", children=[
                    html.Div(className="panel", children=[
                        html.H2("Data source"),
                        html.Label("Run to show"),
                        dcc.Dropdown(id="run", options=list_runs(), value=initial_run, clearable=False),
                        html.Button("Refresh list", id="refresh", style={"marginTop": "8px"}),
                    ]),
                    html.Div(className="panel", children=[
                        html.H2("Process a new recording"),
                        html.Label("Recording (data/recordings or recording/)"),
                        dcc.Dropdown(id="rec", options=list_recordings()),
                        html.Label("Sensor layout"),
                        dcc.Dropdown(id="rec-array", value="auto", clearable=False, options=[
                            {"label": "auto (by channel count)", "value": "auto"},
                            {"label": "config/array.yaml (2-sensor pair)", "value": "config/array.yaml"},
                            {"label": "config/array_triangle.yaml (3 sensors)", "value": "config/array_triangle.yaml"}]),
                        html.Label("Taped spot (tests only, optional)"),
                        dcc.Input(id="rec-spot", placeholder="e.g. P45", style={"width": "100%", "fontSize": 15}),
                        html.Button("Process and open", id="process", style={"marginTop": "10px"}),
                        dcc.Loading(html.Div(id="process-msg", className="msg")),
                    ]),
                    html.Div(className="panel", children=[
                        node_sensors_panel(),
                        html.H2("ATAK link"),
                        html.Div("Sends each sound source (one marker per place, with its accuracy circle and "
                                 "area) and the sensors to ATAK phones on the same Wi-Fi / hotspot. Only "
                                 "human-likely sources are sent.", className="msg"),
                        dcc.Checklist(id="atak-on", value=[], inputStyle={"marginRight": "6px"},
                                      labelStyle={"display": "block", "fontSize": 16, "margin": "8px 0"},
                                      options=[{"label": "Send to ATAK while the Live tab runs", "value": "on"}]),
                        html.Label("Send to"),
                        dcc.Input(id="atak-to", value="multicast", style={"width": "100%", "fontSize": 15}),
                        html.Div("multicast = every ATAK on the network (default). Or a phone: 192.168.43.17:4242. "
                                 "Several: separate with commas.", className="msg"),
                        html.Button("Send test marker", id="atak-test", style={"marginTop": "8px"}),
                        html.Div(id="atak-msg", className="msg"),
                        html.Div(id="atak-status", className="msg"),
                    ]),
                    html.Div(className="panel", children=[
                        html.H2("Display"),
                        dcc.Checklist(id="opts", value=["truth"], inputStyle={"marginRight": "6px"},
                                      labelStyle={"display": "block", "fontSize": 16, "margin": "6px 0"},
                                      options=[{"label": "Show true spots (test recordings)", "value": "truth"},
                                               {"label": "Show non-human sounds on the map", "value": "all"}]),
                    ]),
                ])]),
        ]),
        dcc.Interval(id="tick", interval=int(TICK_S * 1000)),
        dcc.Store(id="play-state", data={"t": 0.0, "playing": True}),
        dcc.Store(id="acks", data={}),
        dcc.Store(id="selected", data=None),
        dcc.Store(id="cards-sig", data=None),
    ])

    @app.callback(Output("play-state", "data"), Output("play", "children"),
                  Input("tick", "n_intervals"), Input("play", "n_clicks"), Input("restart", "n_clicks"),
                  Input("all", "n_clicks"), Input("run", "value"),
                  State("play-state", "data"), State("speed", "value"))
    def advance(_t, _p, _r, _a, run, st, speed):
        src = get_src(run)
        trig = ctx.triggered_id
        if trig == "play":
            st["playing"] = not st["playing"]
        elif trig in ("restart", "run"):
            st = {"t": 0.0, "playing": True}
        elif trig == "all":
            st = {"t": src.duration_s, "playing": False}
        elif isinstance(src, LiveSource):
            st = {"t": src.now(), "playing": True}
        elif st["playing"]:
            st["t"] = min(st["t"] + TICK_S * speed, src.duration_s)
            if st["t"] >= src.duration_s:
                st["playing"] = False
        return st, ("Pause" if st["playing"] else "Play")

    @app.callback(Output("acks", "data"), Output("selected", "data"),
                  Input({"type": "ack", "cid": ALL}, "n_clicks"), Input({"type": "resolve", "cid": ALL}, "n_clicks"),
                  Input({"type": "locate", "cid": ALL}, "n_clicks"), Input("run", "value"),
                  State("acks", "data"), State("selected", "data"), prevent_initial_call=True)
    def on_card(_a, _r, _l, run, acks, selected):
        trig = ctx.triggered_id
        if trig == "run":
            return {}, None
        if not isinstance(trig, dict) or not ctx.triggered or not ctx.triggered[0]["value"]:
            return no_update, no_update
        cid = trig["cid"]
        if trig["type"] == "ack":
            acks[cid] = "ack"
        elif trig["type"] == "resolve":
            acks[cid] = "ack" if acks.get(cid) == "resolved" else "resolved"
        elif trig["type"] == "locate":
            selected = None if selected == cid else cid
        return acks, selected

    @app.callback(Output("statusbar", "className"), Output("state", "children"), Output("sub", "children"),
                  Output("hazard", "children"), Output("map", "figure"), Output("timeline", "figure"),
                  Output("contacts", "children"), Output("cards-sig", "data"), Output("log", "data"),
                  Output("sensors", "children"), Output("clock", "children"), Output("source-chip", "children"),
                  Output("atak-status", "children"),
                  Input("play-state", "data"), Input("run", "value"), Input("opts", "value"),
                  Input("acks", "data"), Input("selected", "data"), Input("basemap", "value"),
                  State("cards-sig", "data"), State("tabs", "value"), State("atak-on", "value"),
                  State("atak-to", "value"))
    def render(st, run, opts, acks, selected, basemap, sig_prev, tab, atak_on, atak_to):
        src = get_src(run)
        now = st["t"]
        fixes, det = src.snapshot(now)
        contacts = build_contacts(fixes)
        cls, state, sub = status_bar(src, fixes, contacts, acks, now)
        hazard = None
        if fixes[now - fixes.t_event <= HISTORY_S].label.isin(style.HAZARD_LABELS).any():
            hazard = html.Div("⚠ HAZARD: structure creaking nearby. Keep responders clear.", className="hazard")
        # re-render cards only when something visible changes (keeps buttons clickable)
        sig = [(c["cid"], len(c["sounds"]), acks.get(c["cid"], ""), int((now - c["last_t"]) // 5)) for c in contacts]
        sig = [run, selected, sig]
        cards = contact_cards(src, contacts, acks, selected, now, html) if sig != sig_prev else no_update
        if basemap in ("street", "satellite"):
            fig_map = geo_figure(src, fixes, contacts, acks, selected, now, "truth" in (opts or []), basemap,
                                 map_bounds(src))
        else:
            fig_map = map_figure(src, fixes, contacts, acks, selected, now, "truth" in (opts or []),
                                 "all" in (opts or []), map_bounds(src))
        log = sounds_log(src, fixes, det) if tab == "log" else no_update
        sens = sensors_panel(src, det, html) if tab == "sensors" else no_update
        atak_status = atak_push(src, contacts, acks, run, "on" in (atak_on or []), atak_to)
        return (f"statusbar {cls}", state, sub, hazard, fig_map, timeline_figure(src, fixes, now), cards, sig,
                log, sens, f"t = {now:5.1f} s", f"{src.meta.get('source', 'replay')} · {run}", atak_status)

    def atak_push(src, contacts, acks, run, on, targets):
        """Send to ATAK when a source appears/changes, and refresh every 10 s so markers don't go stale."""
        import time
        import atak as A
        if not on:
            return no_update if not atak_state.get("was_on") else "ATAK link off."
        atak_state["was_on"] = True
        sig = [run, targets, [(c["cid"], len(c["sounds"]), acks.get(c["cid"], "")) for c in contacts]]
        if sig == atak_state.get("sig") and time.monotonic() - atak_state.get("t", 0) < 10:
            return no_update
        sender = A.CotSender([t for t in (targets or "multicast").split(",")])
        xmls = A.sensor_events(src, run) + A.source_events(src, contacts, acks, run)
        err = sender.send(xmls)
        atak_state.update(sig=sig, t=time.monotonic())
        stamp = __import__("datetime").datetime.now().strftime("%H:%M:%S")
        return (f"⚠ ATAK send failed: {'; '.join(err)}" if err else
                f"✔ {stamp}: sent {len(contacts)} sound source(s) + {len(src.cfg['_names'])} sensors to {targets}")

    @app.callback(Output("atak-msg", "children"), Input("atak-test", "n_clicks"), State("atak-to", "value"),
                  State("run", "value"), prevent_initial_call=True)
    def atak_test(_n, targets, run):
        import atak as A
        g = get_src(run).cfg["geo"]
        err = A.CotSender([t for t in (targets or "multicast").split(",")]).send([A.test_event(g["lat0"], g["lon0"])])
        return (f"⚠ {'; '.join(err)}" if err else
                f"✔ Test marker sent to {targets} at {g['lat0']:.5f}, {g['lon0']:.5f}. "
                "It should appear as 'DNL TEST MARKER' in ATAK within a few seconds.")

    @app.callback(Output("evidence", "children"), Input("tabs", "value"))
    def evidence(tab):
        return evidence_panel(html) if tab == "evidence" else no_update

    @app.callback(Output("run", "options"), Output("run", "value"), Output("process-msg", "children"),
                  Output("rec", "options"),
                  Input("process", "n_clicks"), Input("refresh", "n_clicks"),
                  State("rec", "value"), State("rec-array", "value"), State("rec-spot", "value"), State("run", "value"),
                  prevent_initial_call=True)
    def process(_p, _r, rec, array, spot, current):
        if ctx.triggered_id == "refresh" or not rec:
            msg = "" if ctx.triggered_id == "refresh" else "Pick a recording first."
            return list_runs(), current, msg, list_recordings()
        import pipeline as P
        smp = np.load(ROOT / rec)["samples"]
        n_ch = 1 if smp.ndim == 1 else smp.shape[1]
        if array == "auto":
            array = {2: "config/array.yaml", 3: "config/array_triangle.yaml"}.get(n_ch)
        if array is None:
            return no_update, no_update, (f"{rec} has {n_ch} channel(s): a location needs 2 or 3. "
                                          "Open notebooks/01_tune_detection.ipynb to check the detector."), no_update
        run = Path(rec).stem
        try:
            det, fixes, meta = P.run_pipeline(rec, run, array, None, true_spot=spot or None)
        except SystemExit as e:
            return no_update, no_update, f"Could not process: {e}", no_update
        cache.pop(run, None)
        msg = (f"✔ {run}: {det.sound_id.nunique() if len(det) else 0} sounds detected, {len(fixes)} located, "
               f"{len(meta['skipped_sounds'])} skipped. Delays {'calibrated' if meta['calibrated'] else 'NOT calibrated'}. "
               "Now showing it on the Live tab.")
        return list_runs(), run, msg, list_recordings()

    return app


def initial_run_for(args):
    if args.source == "sim":
        array = args.array or "config/array.yaml"
        run = args.run or ("sim-direction" if load_array(array)["_mode"] == "direction" else "sim-position")
        if not (ROOT / "data/runs" / run / "fixes.csv").exists() or args.regen:
            import sim
            sim.simulate_run(array, run, args.seed)
        return run, array
    if args.source == "replay":
        if not args.run:
            raise SystemExit("--source replay needs --run <name> (a folder under data/runs/)")
        if not (ROOT / "data/runs" / args.run / "fixes.csv").exists():
            raise SystemExit(f"data/runs/{args.run}/fixes.csv not found")
        return args.run, args.array
    run = args.run or "live"
    meta_p = ROOT / "data/runs" / run / "run.yaml"
    if not meta_p.exists() or (yaml.safe_load(meta_p.read_text()) or {}).get("source") != "live":
        raise SystemExit(f"data/runs/{run} is not a live run yet: start `python live.py --port COM5 --run {run}` first")
    return run, args.array


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sim", "replay", "live"], default="sim")
    ap.add_argument("--run", help="run folder under data/runs/")
    ap.add_argument("--array", help="array file (default: from run.yaml)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it from a phone/tablet on the same network")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--tab", default="live", choices=["live", "log", "sensors", "evidence", "setup"])
    args = ap.parse_args()
    run, array = initial_run_for(args)
    print(f"Drop-and-Listen mission control: {run}  ->  http://{args.host}:{args.port}")
    build_app(run, array, args.tab).run(debug=False, host=args.host, port=args.port)
