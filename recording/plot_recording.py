"""Plot recordings: signal check, millisecond zoom, direction, and position from two nodes.

One recording, 1 channel (Stage 1 check):
    python plot_recording.py data/recordings/stage1_knocks.npz

One recording, 2 channels = one direction node (Stage 2):
    python plot_recording.py data/recordings/node1_SLR.npz --spacing 0.3

Two recordings = two direction nodes; their directions are crossed to get positions:
    python plot_recording.py node1.npz node2.npz --spacing 0.3 0.3 --node-gap 1.0

Options: --knock N (which knock to zoom on), --thr 8 (detector strictness), --hp 1000
(high-pass, Hz; default 1000 for MEMS, 300 for Grove), --temp 20, --save out.png.

Delay between mic A and mic B: MEMS recordings use waveform matching (GCC-PHAT, 16x
interpolated); Grove recordings use the sharp rise of each channel, because the Grove's
one-sided loudness signal doesn't match up well between mics.

Geometry (top-down, meters): each node's mic A (channel 0) is on the left at x = -spacing/2,
mic B (channel 1) on the right at x = +spacing/2, both facing the front (+y). Node 1 is centred
at (0, 0) and node 2 at (node_gap, 0). Angles: 0 deg = straight ahead, + = toward mic B's side.
Record both nodes at the same time in two terminals; record.py saves the start time used to
pair their knocks.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

PAIR_WINDOW_S = 0.05  # knocks on the two nodes within 50 ms are the same knock


# ---------------------------------------------------------------- loading and stats
def speed_of_sound_air(temp_c: float, humidity_pct: float = 50.0) -> float:
    """m/s; same formula as base/dropnlisten/speed_of_sound.py."""
    return 331.4 + 0.6 * temp_c + 0.0124 * humidity_pct


def load(path: str) -> dict:
    rec = np.load(path)
    raw = rec["samples"].astype(float)
    fs = float(rec["fs_hz"])
    mems = np.median(raw) > 16384                       # MEMS = offset binary around 32768
    centered = raw - (32768.0 if mems else np.median(raw, axis=0))
    top = 65535 if mems else 4095
    clipped = np.mean((raw <= 1) | (raw >= top - 1), axis=0)
    t0 = float(rec["t_start_unix"]) if "t_start_unix" in rec.files and rec["t_start_unix"].ndim == 0 \
        and rec["t_start_unix"].dtype != object else None
    return dict(path=path, raw=raw, x=centered, fs=fs, kind="MEMS" if mems else "Grove",
                clipped=clipped, t0=t0, dropped=int(rec["dropped_packets"]),
                overflows=int(rec["adc_overflows"]))


def print_stats(r: dict) -> None:
    print(f"\n== {r['path']} ({r['kind']}, {r['fs']:.0f} Hz, {r['x'].shape[1]} ch, "
          f"{len(r['x']) / r['fs']:.1f} s)")
    for k in range(r["x"].shape[1]):
        sig = r["x"][:, k]
        noise = np.std(sig[: int(r["fs"])])
        peak = np.max(np.abs(sig))
        warn = "  CLIPPING" if r["clipped"][k] > 0.0005 else ""
        print(f"  ch{k}: noise {noise:.1f}, biggest swing {peak:.0f} (peak/noise {peak / max(noise, 1e-9):.0f}x){warn}")
    print(f"  dropped packets: {r['dropped']}, ADC overflows: {r['overflows']}")


# ---------------------------------------------------------------- detection and delay
def moving_avg(v: np.ndarray, n: float) -> np.ndarray:
    n = max(int(n), 1)
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    out[: n - 1] = c[1:n] / np.arange(1, n)
    return out


def detect_knocks(hp: np.ndarray, fs: float, thr: float, refractory_s: float = 0.3) -> np.ndarray:
    """Onset indices: 2 ms energy compared with the 50 ms background just before it."""
    e = np.sum(hp ** 2, axis=1)
    short = moving_avg(e, 0.002 * fs)
    long = moving_avg(e, 0.050 * fs)
    gap = int(0.005 * fs)
    background = np.concatenate([np.full(gap, long[0]), long[:-gap]])
    floor = np.median(short) + 1e-12
    ratio = short / (background + 0.1 * floor)
    peaks, _ = find_peaks(ratio, height=thr, distance=int(refractory_s * fs))
    peaks = peaks[(peaks > 0.06 * fs) & (peaks < len(e) - 0.03 * fs)]  # skip filter start-up and the end
    return peaks[short[peaks] > 3 * floor]


def onset_delay(x2: np.ndarray, fs: float, frac: float = 0.3) -> float:
    """t_B - t_A from where each channel's smoothed level first reaches frac of its own peak.

    Used for Grove sensors: their one-sided loudness signal doesn't cross-correlate well,
    but its sharp rise is a good timing mark. Interpolates between samples.
    """
    times = []
    for ch in range(2):
        env = moving_avg(np.abs(x2[:, ch]), 0.0003 * fs)
        env = env - np.median(env[: max(int(0.004 * fs), 1)])   # level before the knock
        level = frac * env.max()
        j = int(np.argmax(env >= level))
        if j == 0:
            times.append(0.0)
            continue
        a, b = env[j - 1], env[j]
        times.append((j - 1 + (level - a) / (b - a + 1e-12)) / fs)
    return times[1] - times[0]


def gcc_phat(a: np.ndarray, b: np.ndarray, fs: float, max_tau: float, interp: int = 16):
    """tau = t_a - t_b in seconds (positive: a hears it later), plus the match curve."""
    n = len(a) + len(b)
    R = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n * interp)
    m = min(int(interp * fs * max_tau), n * interp // 2)
    cc = np.concatenate((cc[-m:], cc[: m + 1]))
    lags = np.arange(-m, m + 1) / (interp * fs)
    return lags[int(np.argmax(np.abs(cc)))], lags, cc


def analyze(r: dict, spacing: float | None, c: float, thr: float, hp_hz: float | None) -> None:
    """Adds r['hp'] and r['knocks'] (with direction if 2 channels and spacing given)."""
    fs = r["fs"]
    hp_hz = hp_hz or (1000.0 if r["kind"] == "MEMS" else 300.0)
    sos = butter(4, hp_hz, btype="highpass", fs=fs, output="sos")
    r["hp"] = sosfiltfilt(sos, r["x"], axis=0)
    r["hp_hz"] = hp_hz
    knocks = []
    for i in detect_knocks(r["hp"], fs, thr):
        lo, hi = max(i - int(0.005 * fs), 0), min(i + int(0.020 * fs), len(r["hp"]))
        k = dict(i=int(i), t=i / fs, peak=float(np.max(np.abs(r["hp"][lo:hi]))))
        if r["x"].shape[1] >= 2 and spacing:
            tau_ab, lags, cc = gcc_phat(r["hp"][lo:hi, 0], r["hp"][lo:hi, 1], fs, spacing / c * 1.2)
            delay = -tau_ab                                  # t_B - t_A, waveform match (MEMS)
            if r["kind"] == "Grove":
                delay = onset_delay(r["x"][lo:hi, :2], fs)   # sharp-rise timing (Grove)
            s = -c * delay / spacing                         # sin(angle), + toward mic B
            k.update(delay_us=delay * 1e6, angle=float(np.degrees(np.arcsin(np.clip(s, -1, 1)))),
                     edge=abs(s) > 0.98, lags=lags, cc=cc)
        knocks.append(k)
    r["knocks"] = knocks


def print_knocks(r: dict, name: str) -> None:
    ks = r["knocks"]
    if not ks:
        print(f"  {name}: no knocks found (try --thr 5, or knock louder)")
        return
    has_dir = "angle" in ks[0]
    print(f"  {name}: {len(ks)} knocks" + ("" if has_dir else " (add --spacing for direction)"))
    print(f"  {'#':>3} {'time s':>7}" + (f" {'delay us':>9} {'angle':>7}" if has_dir else ""))
    for n, k in enumerate(ks):
        line = f"  {n:>3} {k['t']:>7.2f}"
        if has_dir:
            line += f" {k['delay_us']:>9.0f} {k['angle']:>+6.1f}°"
            if k["edge"]:
                line += "  (at the limit: along the mic line, or a bad match)"
        print(line)


# ---------------------------------------------------------------- geometry
def direction_vec(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([np.sin(a), np.cos(a)])


def cross_bearings(p1, u1, p2, u2):
    """Point where two direction lines meet, or None if nearly parallel or behind a node."""
    A = np.column_stack([u1, -u2])
    if abs(np.linalg.det(A)) < 0.1:
        return None
    s, t = np.linalg.solve(A, p2 - p1)
    if s <= 0 or t <= 0:
        return None
    return p1 + s * u1


# ---------------------------------------------------------------- plotting helpers
def plot_overview(ax, r: dict, title: str) -> None:
    t = np.arange(len(r["x"])) / r["fs"]
    for k in range(r["x"].shape[1]):
        ax.plot(t, r["x"][:, k], lw=0.4, alpha=1 if k == 0 else 0.7,
                label=f"mic {'AB'[k] if k < 2 else k} (ch{k})")
    top = np.max(np.abs(r["x"])) or 1
    for n, k in enumerate(r["knocks"]):
        ax.axvline(k["t"], color="k", lw=0.6, alpha=0.5)
        label = f"#{n}" + (f"\n{k['angle']:+.0f}°" if "angle" in k else "")
        ax.text(k["t"], top * 1.02, label, ha="center", va="bottom", fontsize=7)
    ax.set_ylim(-top * 1.05, top * 1.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal (from rest)")
    ax.legend(loc="lower right", fontsize=7)


def plot_zoom(ax, r: dict, sel: int, name: str) -> None:
    k = r["knocks"][sel]
    fs = r["fs"]
    w0, w1 = max(k["i"] - int(0.005 * fs), 0), min(k["i"] + int(0.010 * fs), len(r["x"]))
    tm = (np.arange(w0, w1) - k["i"]) / fs * 1000
    for ch in range(min(r["x"].shape[1], 2)):
        ax.plot(tm, r["hp"][w0:w1, ch], lw=1, alpha=1 if ch == 0 else 0.8, label=f"mic {'AB'[ch]}")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    if "angle" in k:
        first = "B" if k["delay_us"] < 0 else "A"
        ax.set_title(f"{name} knock #{sel}: mic {first} heard it {abs(k['delay_us']):.0f} µs earlier "
                     f"→ {k['angle']:+.1f}°", fontsize=10)
        inset = ax.inset_axes([0.62, 0.08, 0.35, 0.3])
        inset.plot(k["lags"] * 1e6, k["cc"], lw=0.8, color="gray")
        inset.axvline(-k["delay_us"], color="red", lw=0.8)
        inset.set_title("match vs delay (µs)", fontsize=7)
        inset.tick_params(labelsize=6)
    else:
        ax.set_title(f"{name} knock #{sel} zoom", fontsize=10)
    ax.set_xlabel(f"time from detected onset (ms), high-passed above {r['hp_hz']:.0f} Hz")
    ax.set_ylabel("signal")
    ax.legend(fontsize=7, loc="upper right")


def plot_node(ax, center, spacing, label):
    ax.plot([center[0] - spacing / 2, center[0] + spacing / 2], [center[1]] * 2, "s", color="0.2", ms=8)
    ax.text(center[0], center[1] - 0.06, label, ha="center", va="top", fontsize=8)


def plot_directions(ax, r: dict, sel: int, spacing: float) -> None:
    L = max(1.0, 3 * spacing)
    ax.axhspan(0, L * 1.15, color="tab:blue", alpha=0.05)
    for n, k in enumerate(r["knocks"]):
        u = direction_vec(k["angle"]) * L
        bold = n == sel
        ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(
            arrowstyle="-|>", lw=2.2 if bold else 0.8, color="tab:red" if bold else "tab:blue",
            alpha=1 if bold else 0.45))
        ax.text(*(u * 1.08), f"#{n}", fontsize=7, ha="center", va="center",
                color="tab:red" if bold else "tab:blue")
    plot_node(ax, np.array([0.0, 0.0]), spacing, "mic A   mic B")
    ax.set_xlim(-L * 1.25, L * 1.25)
    ax.set_ylim(-0.3 * L, L * 1.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Directions, top-down (red = knock #{sel})", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m), front")


def pick(r: dict, requested: int | None) -> int:
    if requested is not None:
        return min(max(requested, 0), len(r["knocks"]) - 1)
    return int(np.argmax([k["peak"] for k in r["knocks"]]))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="one recording, or two (node 1, node 2)")
    ap.add_argument("--spacing", type=float, nargs="+", default=None,
                    help="mic A to mic B distance per node, m (one value, or one per recording)")
    ap.add_argument("--node-gap", type=float, default=1.0, help="distance between node centres, m")
    ap.add_argument("--temp", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=8.0)
    ap.add_argument("--hp", type=float, default=None)
    ap.add_argument("--knock", type=int, default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if len(args.paths) > 2:
        raise SystemExit("Give one recording, or two (node 1 and node 2).")
    recs = [load(p) for p in args.paths]
    spacings = args.spacing or [None]
    spacings = (spacings * len(recs))[: len(recs)] if len(spacings) == 1 else spacings
    c = speed_of_sound_air(args.temp)
    for r, sp in zip(recs, spacings):
        print_stats(r)
        analyze(r, sp, c, args.thr, args.hp)
        print_knocks(r, "detections")

    if len(recs) == 1:
        r, sp = recs[0], spacings[0]
        has_dir = bool(r["knocks"]) and "angle" in r["knocks"][0]
        fig = plt.figure(figsize=(13, 8.5))
        gs = fig.add_gridspec(2, 2 if has_dir else 1, height_ratios=[1, 1.15])
        plot_overview(fig.add_subplot(gs[0, :]), r,
                      f"{len(r['knocks'])} knocks found" + ("; label = angle (+ toward mic B)" if has_dir else ""))
        if r["knocks"]:
            sel = pick(r, args.knock)
            plot_zoom(fig.add_subplot(gs[1, 0]), r, sel, "")
            if has_dir:
                plot_directions(fig.add_subplot(gs[1, 1]), r, sel, sp)
    else:
        r1, r2 = recs
        if not all("angle" in r["knocks"][0] for r in recs if r["knocks"]):
            raise SystemExit("Two-node mode needs 2-channel recordings and --spacing for both.")
        if r1["t0"] is None or r2["t0"] is None:
            raise SystemExit("These recordings have no start time. Re-record with the updated record.py.")
        p1, p2 = np.array([0.0, 0.0]), np.array([args.node_gap, 0.0])
        pairs = []
        for n1, k1 in enumerate(r1["knocks"]):
            tw1 = r1["t0"] + k1["t"]
            best = min(((abs(r2["t0"] + k2["t"] - tw1), n2) for n2, k2 in enumerate(r2["knocks"])),
                       default=(None, None))
            if best[0] is not None and best[0] < PAIR_WINDOW_S:
                k2 = r2["knocks"][best[1]]
                pos = cross_bearings(p1, direction_vec(k1["angle"]), p2, direction_vec(k2["angle"]))
                pairs.append(dict(n1=n1, n2=best[1], dt_ms=best[0] * 1000, pos=pos,
                                  a1=k1["angle"], a2=k2["angle"]))
        print(f"\n== Paired knocks (node 1 at (0, 0), node 2 at ({args.node_gap}, 0))")
        print(f"  {'n1':>3} {'n2':>3} {'gap ms':>7} {'angle1':>7} {'angle2':>7} {'x m':>6} {'y m':>6}")
        for pr in pairs:
            xy = "   no crossing (lines nearly parallel)" if pr["pos"] is None else \
                f" {pr['pos'][0]:>6.2f} {pr['pos'][1]:>6.2f}"
            print(f"  {pr['n1']:>3} {pr['n2']:>3} {pr['dt_ms']:>7.1f} {pr['a1']:>+6.1f}° {pr['a2']:>+6.1f}°{xy}")

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])
        plot_overview(fig.add_subplot(gs[0, :]), r1, f"Node 1 ({r1['kind']}): label = angle")
        plot_overview(fig.add_subplot(gs[1, :]), r2, f"Node 2 ({r2['kind']}): label = angle")
        if r1["knocks"]:
            plot_zoom(fig.add_subplot(gs[2, 0]), r1, pick(r1, args.knock), "Node 1")
        ax = fig.add_subplot(gs[2, 1])
        reach = max(1.5, args.node_gap * 1.5)
        for pr in pairs:
            for p, a in ((p1, pr["a1"]), (p2, pr["a2"])):
                end = p + direction_vec(a) * reach
                ax.plot([p[0], end[0]], [p[1], end[1]], color="tab:blue", lw=0.6, alpha=0.35)
            if pr["pos"] is not None:
                ax.plot(*pr["pos"], "o", color="tab:red", ms=6)
                ax.text(pr["pos"][0], pr["pos"][1] + 0.04, f"#{pr['n1']}", fontsize=7, ha="center")
        plot_node(ax, p1, spacings[0], "node 1")
        plot_node(ax, p2, spacings[1], "node 2")
        ax.set_xlim(-reach * 0.6, args.node_gap + reach * 0.6)
        ax.set_ylim(-0.25, reach)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title("Positions: red dot = where the two nodes' directions cross", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m), front")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"\nsaved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()