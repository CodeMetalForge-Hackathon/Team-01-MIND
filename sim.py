"""Synthetic data generator for Drop-and-Listen (obeys CLAUDE.md contracts).

    python sim.py                                   # both demo runs + clips
    python sim.py --array config/array.yaml --run sim-direction --no-clips
    python sim.py --array config/array_triangle.yaml --run sim-position --no-clips

Layout, sample rate, medium, speed of sound and timing sigma all come from the
array file (agreed M0 schema). Writes
    data/recordings/<run>.npz
    data/runs/<run>/{detections.csv, fixes.csv, run.yaml, snippets/<id>.npz}
    data/clips/<label>/<session>_<n>.npz    (unless --no-clips)

Every sound has a known true position and label. The timing noise injected per
label is the same sigma used for the uncertainty, so wedges/ellipses should be
honest -- the same check the M0 simulation makes.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from arraycfg import front_vector, load_array, local_angle_to_compass
from localize import solve_bearing, solve_position

ROOT = Path(__file__).resolve().parent
FS = 10_000            # default only; runs use sampling.fs_hz from the array file
ADC_BITS, ADC_MID, ADC_MAX = 12, 2048, 4095   # ESP32-S2 ADC1, 12-bit
RUN_START = dt.datetime(2026, 11, 7, 18, 0, 0, tzinfo=dt.timezone.utc)

# Onset timing noise per label, as a multiple of timing.sigma_s in the array
# file: knocks are sharp, voices have soft attacks.
SIGMA_MULT = {"knock_tap": 1.0, "impact": 1.6, "voice": 5.0, "other": 6.0}
EVENT_WIN_S = {"knock_tap": 0.06, "impact": 0.15, "voice": 0.8, "other": 0.5}
AMP_1M = {"knock_tap": 450.0, "impact": 3000.0, "voice": 250.0, "other": 120.0}  # ADC counts at 1 m
CONFIDENCE = {"knock_tap": (0.85, 0.97), "voice": (0.75, 0.90), "impact": (0.70, 0.90), "other": (0.60, 0.80)}
HUMAN = {"knock_tap", "voice"}


# --------------------------------------------------------------------------- sources
def _env(n, attack, release, fs):
    e = np.ones(n)
    a, r = max(int(attack * fs), 1), max(int(release * fs), 1)
    e[:a] = np.linspace(0, 1, a)
    e[-r:] *= np.linspace(1, 0, r)
    return e


def knock(rng, fs=FS, f_res=(500, 1200), tau=(0.008, 0.015)):
    """Sharp onset, ringing damped modes: a short vertical stripe in a spectrogram."""
    n = int(0.08 * fs)
    t = np.arange(n) / fs
    y = np.zeros(n)
    for _ in range(3):
        f = rng.uniform(*f_res) * rng.uniform(0.8, 1.6)
        y += rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3)) * np.exp(-t / rng.uniform(*tau))
    click = np.zeros(n)
    k = int(0.001 * fs)
    click[:k] = rng.normal(0, 1, k) * np.linspace(1, 0, k)
    y = y + 0.8 * click
    return y / np.abs(y).max()


def voice(rng, fs=FS, f0=180.0, dur=0.6):
    """Harmonic stack with vowel formants, vibrato, syllable rhythm: horizontal bands."""
    n = int(dur * fs)
    t = np.arange(n) / fs
    f0_t = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t) + 0.04 * (t / dur - 0.5) * rng.choice([-1, 1]))
    phase = 2 * np.pi * np.cumsum(f0_t) / fs
    vowels = [(730, 1090, 2440), (530, 1840, 2480), (570, 840, 2410), (270, 2290, 3010)]
    F = vowels[rng.integers(len(vowels))]
    y = np.zeros(n)
    for h in range(1, int(3500 / f0)):
        fh = h * f0
        amp = sum(np.exp(-0.5 * ((fh - Fk) / 120) ** 2) for Fk in F) + 0.05
        y += amp / h ** 0.5 * np.sin(h * phase)
    syll = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(3, 5) * t) ** 2
    y = y * syll * _env(n, 0.04, 0.08, fs) + 0.03 * rng.normal(0, 1, n)
    return y / np.abs(y).max()


def impact(rng, fs=FS):
    """Single, short, broadband burst plus a low thump."""
    n = int(0.25 * fs)
    t = np.arange(n) / fs
    noise = rng.normal(0, 1, n) * np.exp(-t / rng.uniform(0.03, 0.06))
    thump = np.sin(2 * np.pi * rng.uniform(60, 120) * t) * np.exp(-t / 0.05)
    y = noise + 1.5 * thump
    return y / np.abs(y).max()


def _lowpass_noise(rng, n, alpha):
    x = rng.normal(0, 1, n)
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = alpha * acc + (1 - alpha) * x[i]
        y[i] = acc
    return y


def _pink(rng, n):
    X = np.fft.rfft(rng.normal(0, 1, n))
    f = np.arange(len(X))
    X[1:] /= np.sqrt(f[1:])
    X[0] = 0
    y = np.fft.irfft(X, n)
    return y / (y.std() + 1e-12)


def other(rng, fs=FS, n=None, kind=None):
    """Non-human, non-impact sounds: fan hum, rustle, rumble, pink noise."""
    n = n or int(0.6 * fs)
    t = np.arange(n) / fs
    kind = kind or rng.choice(["fan", "rustle", "rumble", "pink"])
    if kind == "fan":
        f = rng.uniform(25, 60)
        y = sum(np.sin(2 * np.pi * k * f * t + rng.uniform(0, 6)) / k for k in range(1, 6))
        y = y + 0.5 * _pink(rng, n)
    elif kind == "rustle":
        y = rng.normal(0, 1, n) * (0.3 + np.abs(_lowpass_noise(rng, n, 0.999)) * 8)
    elif kind == "rumble":
        y = _lowpass_noise(rng, n, 0.98)
    else:
        y = _pink(rng, n)
    y = y * _env(n, 0.05, 0.05, fs)
    return y / (np.abs(y).max() + 1e-12)


SOURCES = {"knock_tap": knock, "voice": voice, "impact": impact, "other": other}


# --------------------------------------------------------------------------- helpers
def add_delayed(dst, src, start_s, gain, fs=FS):
    """Add src into dst starting at start_s with a fractional-sample delay."""
    pos = start_s * fs
    i0 = int(np.floor(pos))
    frac = pos - i0
    m = len(src) + 64
    f = np.fft.rfftfreq(m)
    y = np.fft.irfft(np.fft.rfft(src, m) * np.exp(-2j * np.pi * f * frac), m) * gain
    lo, hi = max(i0, 0), min(i0 + m, len(dst))
    if hi > lo:
        dst[lo:hi] += y[lo - i0:hi - i0]


def load_spots(group):
    spots = yaml.safe_load((ROOT / "config/cal_spots.yaml").read_text())[group]
    return {k: (v["x"], v["y"]) for k, v in spots.items()}


# --------------------------------------------------------------------------- runs
def demo_events(mode):
    """Demo story. Times are emission times (s since recording start)."""
    if mode == "direction":       # pair at +/-0.25 m on the x axis, front = +y
        P45, P90 = (load_spots("direction")[k] for k in ("P45", "P90"))
        return [
            dict(label="knock_tap", xy=P90, t=2.0, spot="P90"),
            dict(label="knock_tap", xy=(-1.2, 1.0), t=4.5),
            dict(label="impact", xy=(1.6, 0.4), t=7.0, note="near the sensor axis: wide wedge; clips DNL-S2"),
            dict(label="voice", xy=(-0.5, 2.5), t=10.0, dur=0.8),
            dict(label="knock_tap", xy=P45, t=13.0, spot="P45", repeated=True),
            dict(label="knock_tap", xy=P45, t=13.9, spot="P45", repeated=True),
            dict(label="knock_tap", xy=P45, t=14.8, spot="P45", repeated=True),
            dict(label="other", xy=(-1.5, 2.0), t=18.0, note="rustle"),
            dict(label="knock_tap", xy=(0.8, 2.6), t=21.0),
            dict(label="knock_tap", xy=P45, t=24.0, spot="P45"),
        ]
    T = load_spots("triangle")
    return [
        dict(label="knock_tap", xy=T["T1"], t=2.0, spot="T1"),
        dict(label="knock_tap", xy=T["T2"], t=4.5, spot="T2"),
        dict(label="impact", xy=(1.8, 0.15), t=7.0, note="debris fall next to DNL-S2, clips it"),
        dict(label="voice", xy=(0.7, 0.9), t=10.0, dur=0.8),
        dict(label="knock_tap", xy=T["T4"], t=13.0, spot="T4", repeated=True),
        dict(label="knock_tap", xy=T["T4"], t=13.9, spot="T4", repeated=True),
        dict(label="knock_tap", xy=T["T4"], t=14.8, spot="T4", repeated=True),
        dict(label="other", xy=(1.3, 0.9), t=18.0, note="rustle"),
        dict(label="knock_tap", xy=T["T3"], t=21.0, spot="T3"),
        dict(label="knock_tap", xy=T["T1"], t=24.0, spot="T1"),
    ]


def simulate_run(array="config/array.yaml", run="sim-direction", seed=0, duration_s=30.0,
                 channel_delay_s=(0.0, 120e-6, -80e-6, 60e-6), events=None):
    rng = np.random.default_rng(seed)
    cfg = load_array(array)
    nodes, c_mps, fs, mode = cfg["_xy"], cfg["_c_mps"], int(cfg["sampling"]["fs_hz"]), cfg["_mode"]
    sigma0 = float(cfg["timing"]["sigma_s"])
    xb = float(cfg["geo"]["x_axis_bearing_deg"])
    n_ch = len(nodes)
    delays = np.asarray(channel_delay_s[:n_ch], float)
    events = events or demo_events(mode)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # Background: white noise + mains hum, independent per channel.
    x = np.empty((n, n_ch))
    for ch in range(n_ch):
        x[:, ch] = rng.normal(0, 5.0, n) + 3.0 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 6.3))

    for ev in events:
        src_fn = SOURCES[ev["label"]]
        src = src_fn(rng, fs, dur=ev.get("dur", 0.6)) if ev["label"] == "voice" else src_fn(rng, fs)
        ev["_arrival"] = {}
        for ch in range(n_ch):
            d = np.linalg.norm(nodes[ch] - np.asarray(ev["xy"]))
            ev["_arrival"][ch] = ev["t"] + d / c_mps + delays[ch]
            add_delayed(x[:, ch], src, ev["_arrival"][ch], AMP_1M[ev["label"]] / max(d, 0.3), fs)

    samples = np.clip(np.round(x + ADC_MID), 0, ADC_MAX).astype(np.int16)

    det_rows, fix_rows = [], []
    for sid, ev in enumerate(events, start=1):
        lab = ev["label"]
        sig = sigma0 * SIGMA_MULT[lab]
        t_meas = {}
        for ch in range(n_ch):
            ts = ev["_arrival"][ch] + rng.normal(0, sig)
            t_meas[ch] = ts
            i0 = int(round(ts * fs))
            win = samples[i0:i0 + int(EVENT_WIN_S[lab] * fs), ch].astype(float)
            pre = samples[max(i0 - int(0.2 * fs), 0):i0 - int(0.02 * fs), ch].astype(float)
            peak = np.abs(win - ADC_MID).max()
            det_rows.append(dict(sound_id=sid, ch=ch, t_s=round(ts, 7), sample_index=i0, peak=int(peak),
                                 snr=round(20 * np.log10(peak / (pre.std() + 1e-9)), 2),
                                 clipped=bool((win <= 0).any() or (win >= ADC_MAX).any())))

        # Fix (sim stands in for calibration: true channel delays are removed).
        tc = np.array([t_meas[ch] - delays[ch] for ch in range(n_ch)])
        lo, hi = CONFIDENCE[lab]
        row = dict(sound_id=sid, wall_time=(RUN_START + dt.timedelta(seconds=max(tc) + 0.35)).isoformat(timespec="milliseconds"),
                   bearing_deg=np.nan, bearing_sigma_deg=np.nan, x_m=np.nan, y_m=np.nan, cov_xx=np.nan,
                   cov_xy=np.nan, cov_yy=np.nan, residual_m=np.nan, label=lab,
                   confidence=round(rng.uniform(lo, hi), 3), human_likely=lab in HUMAN,
                   repeated=bool(ev.get("repeated", False)),
                   true_x_m=ev["xy"][0], true_y_m=ev["xy"][1], true_label=lab)
        if mode == "position":
            sol = solve_position(nodes, tc, c_mps, sig)
            row.update(kind="position", x_m=round(sol["x"], 4), y_m=round(sol["y"], 4),
                       cov_xx=sol["cov"][0, 0], cov_xy=sol["cov"][0, 1], cov_yy=sol["cov"][1, 1],
                       residual_m=round(sol["residual_m"], 4))
        else:
            b = solve_bearing(nodes[0], nodes[1], tc[0], tc[1], c_mps, sig,
                              front_vector(nodes[0], nodes[1], cfg["bearing"]["front"]))
            row.update(kind="bearing", bearing_deg=round(local_angle_to_compass(b["theta_deg"], xb), 2),
                       bearing_sigma_deg=round(b["sigma_deg"], 2))
            ev["_endfire"] = b["endfire_clipped"]
        fix_rows.append(row)

    cols_det = ["sound_id", "ch", "t_s", "sample_index", "peak", "snr", "clipped"]
    cols_fix = ["sound_id", "kind", "wall_time", "bearing_deg", "bearing_sigma_deg", "x_m", "y_m", "cov_xx",
                "cov_xy", "cov_yy", "residual_m", "label", "confidence", "human_likely", "repeated",
                "true_x_m", "true_y_m", "true_label"]
    det = pd.DataFrame(det_rows)[cols_det]
    fix = pd.DataFrame(fix_rows)[cols_fix]

    rec_dir = ROOT / "data/recordings"
    run_dir = ROOT / "data/runs" / run
    snip_dir = run_dir / "snippets"
    for d in (rec_dir, snip_dir):
        d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rec_dir / f"{run}.npz", samples=samples, fs_hz=fs, adc_bits=ADC_BITS, adc_mid=ADC_MID,
                        sensor_names=np.array(cfg["_names"]))
    det.to_csv(run_dir / "detections.csv", index=False)
    fix.to_csv(run_dir / "fixes.csv", index=False)
    for sid, g in det.groupby("sound_id"):
        i0 = int(round(max(g.t_s.min() - 0.25, 0.0) * fs))
        np.savez_compressed(snip_dir / f"{sid}.npz", samples=samples[i0:i0 + fs], fs_hz=fs, sound_id=sid, t0_s=i0 / fs)

    meta = dict(run=run, source="sim", seed=seed, mode=mode, array=str(array), medium=cfg["medium"],
                c_mps=round(c_mps, 2), fs_hz=fs, sigma_t_s=sigma0, recording=f"data/recordings/{run}.npz",
                start_time=RUN_START.isoformat(), duration_s=duration_s,
                true_channel_delays_s=[float(v) for v in delays],
                fixes_delay_correction="true channel delays subtracted before solving (sim stands in for calibration)",
                events=[dict(sound_id=i, label=e["label"], t_emit_s=e["t"], x_m=float(e["xy"][0]), y_m=float(e["xy"][1]),
                             **{k: e[k] for k in ("spot", "note", "repeated") if k in e},
                             **({"endfire_clipped": e["_endfire"]} if "_endfire" in e else {}))
                        for i, e in enumerate(events, start=1)])
    (run_dir / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return det, fix


# --------------------------------------------------------------------------- clips
SESSIONS = [  # session id (no underscores), environment, voice f0, knock resonance, knock decay, background
    dict(session="sim-s1", environment="table", f0=210, f_res=(500, 1100), tau=(0.008, 0.015), bg="pink"),
    dict(session="sim-s2", environment="floor", f0=120, f_res=(180, 450), tau=(0.006, 0.012), bg="fan"),
    dict(session="sim-s3", environment="metal", f0=200, f_res=(1500, 2800), tau=(0.03, 0.06), bg="rumble"),
    dict(session="sim-s4", environment="floor", f0=135, f_res=(200, 500), tau=(0.006, 0.012), bg="fan"),
]


def make_clip(rng, label, s, fs=FS):
    n = fs
    bg = other(rng, fs, n=n, kind=s["bg"])
    bg = bg / (bg.std() + 1e-12)
    ev = np.zeros(n)
    if label == "knock_tap":
        t0 = rng.uniform(0.05, 0.3)
        for _ in range(rng.integers(1, 4)):
            add_delayed(ev, knock(rng, fs, s["f_res"], s["tau"]), t0, rng.uniform(0.7, 1.0), fs)
            t0 += rng.uniform(0.15, 0.3)
    elif label == "voice":
        dur = rng.uniform(0.3, 0.8)
        add_delayed(ev, voice(rng, fs, s["f0"] * rng.uniform(0.85, 1.15), dur), rng.uniform(0.05, 0.95 - dur), 1.0, fs)
    elif label == "impact":
        add_delayed(ev, impact(rng, fs), rng.uniform(0.05, 0.5), 1.0, fs)
    else:  # other: a different non-human sound over the session background
        add_delayed(ev, other(rng, fs, n=int(rng.uniform(0.4, 0.9) * fs)), rng.uniform(0.0, 0.1), 1.0, fs)
    snr_db = rng.uniform(6, 30)  # distance stand-in: peak-over-background-RMS
    y = ev * 10 ** (snr_db / 20) / (np.abs(ev).max() + 1e-12) + bg
    return (0.9 * y / np.abs(y).max()).astype(np.float32), snr_db


def simulate_clips(per_label=40, seed=1, labels=("knock_tap", "voice", "impact", "other"), fs=FS):
    rng = np.random.default_rng(seed)
    root = ROOT / "data/clips"
    count = 0
    for label in labels:
        (root / label).mkdir(parents=True, exist_ok=True)
        for i in range(per_label):
            s = SESSIONS[i % len(SESSIONS)]
            audio, snr_db = make_clip(rng, label, s, fs)
            k = i // len(SESSIONS)
            np.savez_compressed(root / label / f"{s['session']}_{k:03d}.npz", audio=audio, fs_hz=fs, label=label,
                                session=s["session"], environment=s["environment"], sim_snr_db=round(snr_db, 1))
            count += 1
    return count


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", help="array file; default: both demo runs")
    ap.add_argument("--run", help="run name (required with --array)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--clips-per-label", type=int, default=40)
    args = ap.parse_args()
    jobs = [(args.array, args.run or "sim-custom")] if args.array else [
        ("config/array.yaml", "sim-direction"), ("config/array_triangle.yaml", "sim-position")]
    for array, run in jobs:
        det, fix = simulate_run(array, run, args.seed)
        print(f"run {run} ({array}): {fix.sound_id.nunique()} sounds, {len(det)} detections -> data/runs/{run}/")
    if not args.no_clips:
        fs = int(load_array("config/array.yaml")["sampling"]["fs_hz"])
        print(f"clips: {simulate_clips(args.clips_per_label, fs=fs)} -> data/clips/ at {fs} Hz")
