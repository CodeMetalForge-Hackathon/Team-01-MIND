"""Knock / voice onset detection, and the detection tuning view (Plot 1).

Pipeline per channel:
    raw ADC -> band-pass (filter band) -> envelope -> trailing noise floor
    trigger  = first rise of envelope above  k_ratio * noise floor
    onset    = walk back from the trigger to where the envelope crossed
               floor + low_frac * (peak - floor), interpolated between samples
Then onsets on different channels within the physical max time gap are
grouped into one sound and written in the detections.csv contract.

All tuning knobs live in DetectParams; the notebook sets them at the top.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

import style
from arraycfg import ROOT, load_array

ADC_MID, ADC_MAX = 2048, 4095   # ADC_MID only as a fallback: the real quiet level is measured per channel


@dataclass
class DetectParams:
    k_ratio: float = 8.0            # trigger when envelope > k_ratio x noise floor
    low_frac: float = 0.2           # onset = where envelope crosses floor + low_frac x (peak - floor)
    band_hz: tuple = (300.0, 3000.0)  # band-pass edges
    env_smooth_ms: float = 1.0      # envelope smoothing (centered, so no timing bias)
    noise_win_s: float = 2.0        # trailing window for the noise floor (median)
    peak_win_ms: float = 20.0       # where to look for the peak after the trigger
    max_walkback_ms: float = 30.0   # never walk back further than this
    release_s: float = 0.15         # envelope must stay below threshold this long to re-arm
    min_gap_s: float = 0.25         # minimum time between two sounds on one channel
    snr_min_db: float = 15.0        # below this, flag as low SNR (warning only)


# --------------------------------------------------------------------------- signal chain
def bandpass(x, fs, band_hz, order=4):
    lo, hi = band_hz
    hi = min(hi, 0.45 * fs)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)  # zero-phase: filtering does not shift onset times


def envelope(xf, fs, smooth_ms):
    env = np.abs(hilbert(xf))
    k = max(int(smooth_ms * 1e-3 * fs), 1)
    return uniform_filter1d(env, k) if k > 1 else env


def noise_floor(env, fs, win_s, block_s=0.005):
    """Trailing median of the envelope (past only, so a sound can't raise its own floor)."""
    b = max(int(block_s * fs), 1)
    nb = len(env) // b + 1
    blocks = np.array([env[i * b:(i + 1) * b].mean() if i * b < len(env) else env[-1] for i in range(nb)])
    med = pd.Series(blocks).rolling(max(int(win_s / block_s), 1), min_periods=1).median().to_numpy()
    med = np.concatenate([[med[0]], med[:-1]])  # shift one block: strictly past
    return np.repeat(med, b)[:len(env)]


def channel_chain(raw, fs, p: DetectParams):
    bias = float(np.median(raw))            # quiet level of THIS sensor (real hardware: not always 2048)
    x = raw.astype(float) - bias
    xf = bandpass(x, fs, p.band_hz)
    env = envelope(xf, fs, p.env_smooth_ms)
    floor = noise_floor(env, fs, p.noise_win_s)
    return dict(x=x, xf=xf, env=env, floor=floor, thr=p.k_ratio * floor, bias=bias)


# --------------------------------------------------------------------------- detection
def find_onsets(env, floor, thr, fs, p: DetectParams):
    """Return list of dicts: trigger index, onset (fractional sample), low level, end index."""
    above = env > thr
    rises = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    rel = max(int(p.release_s * fs), 1)
    out, rearm = [], 0
    for i in rises:
        if i < rearm:
            continue
        w = int(p.peak_win_ms * 1e-3 * fs)
        pk = env[i:i + w].max()
        level = floor[i] + p.low_frac * (pk - floor[i])
        j = i
        jmin = max(i - int(p.max_walkback_ms * 1e-3 * fs), 1)
        if env[j] > level:                      # walk back (usual case)
            while j > jmin and env[j - 1] > level:
                j -= 1
            j -= 1                               # env[j] <= level < env[j+1]
        else:                                    # level above trigger: walk forward
            while j < len(env) - 2 and env[j + 1] <= level:
                j += 1
        d = env[j + 1] - env[j]
        frac = np.clip((level - env[j]) / d, 0, 1) if d > 0 else 0.0
        # end of event: envelope below threshold for `rel` samples
        k = i + 1
        while k < len(env) and above[k:k + rel].any():
            k += rel
        out.append(dict(trigger=int(i), onset=float(j + frac), level=float(level), peak_env=float(pk), end=int(k)))
        rearm = max(k, i + int(p.min_gap_s * fs))
    return out


def detect_recording(samples, fs, p: DetectParams, sensors_xy=None, c_mps=344.0):
    """Run the chain on every channel. Returns (detections DataFrame, chains, onsets_by_ch)."""
    samples = np.atleast_2d(samples.T).T if samples.ndim == 1 else samples
    n_ch = samples.shape[1]
    chains, rows = [], []
    for ch in range(n_ch):
        c = channel_chain(samples[:, ch], fs, p)
        chains.append(c)
        for o in find_onsets(c["env"], c["floor"], c["thr"], fs, p):
            i0 = int(round(o["onset"]))
            win = samples[i0:max(o["end"], i0 + int(0.06 * fs)), ch].astype(float)[: fs]
            pre = samples[max(i0 - int(0.2 * fs), 0):max(i0 - int(0.02 * fs), 1), ch].astype(float)
            peak = np.abs(win - c["bias"]).max()
            rows.append(dict(ch=ch, t_s=o["onset"] / fs, sample_index=i0, peak=int(peak),
                             snr=round(20 * np.log10(peak / (pre.std() + 1e-9)), 2),
                             clipped=bool((win <= 0).any() or (win >= ADC_MAX).any()), **o))
    det = pd.DataFrame(rows)
    if det.empty:
        return pd.DataFrame(columns=["sound_id", "ch", "t_s", "sample_index", "peak", "snr", "clipped"]), chains, det
    # group channels into sounds: onsets closer than the largest physical gap
    if sensors_xy is not None and len(sensors_xy) > 1:
        s = np.asarray(sensors_xy)
        max_lag = max(np.linalg.norm(a - b) for a in s for b in s) / c_mps + 0.005
    else:
        max_lag = 0.005
    det = det.sort_values("t_s").reset_index(drop=True)
    # Cluster onsets that sit within one travel-time window of each other, then keep the CLEAREST
    # onset per channel (highest SNR) as the sound. Stray noise triggers in the same window become
    # their own one-channel "sounds" (later skipped) instead of splitting the real sound in two.
    ids = np.zeros(len(det), dtype=int)
    sid, i, n = 0, 0, len(det)
    while i < n:
        j = i + 1
        while j < n and det.t_s.iat[j] - det.t_s.iat[i] <= max_lag:
            j += 1
        block = det.iloc[i:j]
        sid += 1
        best = block.sort_values("snr", ascending=False).drop_duplicates("ch").index
        ids[best] = sid
        for k in block.index.difference(best):
            sid += 1
            ids[k] = sid
        i = j
    det["sound_id"] = ids
    if sensors_xy is not None and n_ch >= 3:
        det = guided_fill(det, chains, fs, np.asarray(sensors_xy, float), c_mps)
    det["t_s"] = det["t_s"].round(7)
    return det, chains, det.copy()


def guided_onset(env, fs, t_center, half_win, k_local=3.0):
    """Onset of a sound we KNOW happened (other sensors heard it) near t_center, judged against the
    local background just before the window. Returns (onset_sample, peak) or None."""
    i0, i1 = int((t_center - half_win) * fs), int((t_center + half_win) * fs)
    b0 = i0 - int(0.1 * fs)
    if b0 < 0 or i1 >= len(env) or i1 - i0 < 5:
        return None
    bg = float(np.median(env[b0:i0]))
    seg = env[i0:i1]
    peak = float(seg.max())
    if peak < k_local * max(bg, 1e-9):
        return None
    s_ = i0 + int(np.argmax(seg > bg + 0.5 * (peak - bg)))
    low = bg + 0.25 * (env[s_] - bg)
    j = s_
    while j > max(s_ - int(0.010 * fs), 1) and env[j - 1] >= low:
        j -= 1
    j0 = max(j - 1, 0)
    d = env[j] - env[j0]
    frac = float(np.clip((low - env[j0]) / d, 0, 1)) if d > 0 else 0.0
    return j0 + frac, peak


def guided_fill(det, chains, fs, xy, c_mps, margin_s=0.005):
    """A sound heard clearly on all but one sensor did happen: look for it on the missing sensor
    (it may have been deaf for a moment after a noise trigger). Same idea as record_plot.py."""
    n_ch = len(chains)
    add = []
    for sid, g in det.groupby("sound_id"):
        if len(g) != n_ch - 1:
            continue
        (m,) = set(range(n_ch)) - set(g.ch)
        t_mid = float(g.t_s.mean())
        half = max(np.linalg.norm(xy[m] - xy[int(k)]) for k in g.ch) / c_mps + margin_s
        hit = guided_onset(chains[m]["env"], fs, t_mid, half)
        if hit is None:
            continue
        on, pk = hit
        t_on = on / fs
        if all(abs(t_on - tk) <= np.linalg.norm(xy[m] - xy[int(k)]) / c_mps + margin_s for k, tk in zip(g.ch, g.t_s)):
            add.append(dict(ch=m, t_s=t_on, sample_index=int(round(on)), peak=int(pk), snr=np.nan, clipped=False,
                            sound_id=sid, guided=True))
    if add:
        det = pd.concat([det, pd.DataFrame(add)], ignore_index=True).sort_values(["sound_id", "ch"]).reset_index(drop=True)
    return det


def to_contract(det):
    return det[["sound_id", "ch", "t_s", "sample_index", "peak", "snr", "clipped"]].sort_values(["sound_id", "ch"])


# --------------------------------------------------------------------------- comparison
def true_arrivals(run_dir):
    """Per-channel true arrival times for sim runs (from run.yaml), else None."""
    meta_p = run_dir / "run.yaml"
    if not meta_p.exists():
        return None
    meta = yaml.safe_load(meta_p.read_text())
    if "events" not in meta:
        return None
    cfg = load_array(meta["array"])
    delays = meta.get("true_channel_delays_s", [0.0] * len(cfg["_xy"]))
    rows = []
    for e in meta["events"]:
        for ch, xy in enumerate(cfg["_xy"]):
            d = np.hypot(xy[0] - e["x_m"], xy[1] - e["y_m"])
            rows.append(dict(sound_id=e["sound_id"], ch=ch, label=e["label"],
                             t_true=e["t_emit_s"] + d / meta["c_mps"] + delays[ch]))
    return pd.DataFrame(rows)


def compare(det, ref, tol_s=0.05):
    """Match our onsets to reference onsets per channel. ref needs columns ch, t_ref (+ sound_id)."""
    rows = []
    used = set()
    for _, r in ref.iterrows():
        cand = det[(det.ch == r.ch)]
        cand = cand[~cand.index.isin(used)]
        dt = (cand.t_s - r.t_ref).abs() if len(cand) else pd.Series(dtype=float)
        if len(dt) and dt.min() <= tol_s:
            k = dt.idxmin()
            used.add(k)
            rows.append(dict(ref_sound=r.sound_id, ch=r.ch, status="found", err_us=(det.at[k, "t_s"] - r.t_ref) * 1e6,
                             our_sound=det.at[k, "sound_id"]))
        else:
            rows.append(dict(ref_sound=r.sound_id, ch=r.ch, status="missed", err_us=np.nan, our_sound=np.nan))
    for k in det.index.difference(list(used)):
        rows.append(dict(ref_sound=np.nan, ch=det.at[k, "ch"], status="false", err_us=np.nan,
                         our_sound=det.at[k, "sound_id"], t_s=det.at[k, "t_s"]))
    return pd.DataFrame(rows)


def gap_errors_us(cmp):
    """Error of each channel's time gap to the first channel, per sound (us).
    This is what localization runs on: a constant early/late offset cancels here."""
    f = cmp[cmp.status == "found"]
    piv = f.pivot_table(index="ref_sound", columns="ch", values="err_us")
    if piv.shape[1] < 2:
        return np.array([])
    first = piv.columns[0]
    return piv.drop(columns=first).sub(piv[first], axis=0).to_numpy().ravel()


def summarize(cmp):
    n = cmp.status.value_counts()
    err = cmp.loc[cmp.status == "found", "err_us"]
    g = gap_errors_us(cmp)
    g = g[~np.isnan(g)]
    return dict(found=int(n.get("found", 0)), missed=int(n.get("missed", 0)), false=int(n.get("false", 0)),
                median_err_us=float(err.median()) if len(err) else np.nan,
                spread_us=float(err.std()) if len(err) > 1 else np.nan,
                # robust: typical |gap error| and a count of outliers (> 500 us, e.g. soft voice onsets)
                gap_typical_us=float(np.median(np.abs(g))) if len(g) else np.nan,
                gap_outliers=int((np.abs(g) > 500).sum()),
                sigma_t_implied_s=float(1.4826 * np.median(np.abs(g)) / np.sqrt(2) * 1e-6) if len(g) else np.nan)


# --------------------------------------------------------------------------- Plot 1
def plot_tuning(samples, fs, chains, det, p: DetectParams, ref=None, truth=None, names=None,
                zoom_sound=None, zoom_ch=0, title=None, zoom_ms=(-12, 25)):
    """Detection tuning view. ref: detections.csv (gray dashed). truth: true arrivals (black dotted)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_ch = len(chains)
    names = names or [f"ch{c}" for c in range(n_ch)]
    t = np.arange(samples.shape[0]) / fs
    heights = [1, 1, 1.6] * n_ch + [2.6]
    fig = plt.figure(figsize=(14, 1.15 * sum(heights) + 1.2))
    gs = GridSpec(len(heights), 1, height_ratios=heights, hspace=0.12, figure=fig)
    axes = []
    for ch, c in enumerate(chains):
        a_raw = fig.add_subplot(gs[3 * ch], sharex=axes[0] if axes else None)
        a_flt = fig.add_subplot(gs[3 * ch + 1], sharex=a_raw)
        a_env = fig.add_subplot(gs[3 * ch + 2], sharex=a_raw)
        axes += [a_raw, a_flt, a_env]
        a_raw.plot(t, samples[:, ch], color=style.SIGNAL_RAW, lw=0.4)
        clip = (samples[:, ch] <= 0) | (samples[:, ch] >= ADC_MAX)
        if clip.any():
            a_raw.plot(t[clip], samples[clip, ch], ".", color=style.WARNING, ms=2, label="ADC clipped")
            a_raw.legend(loc="upper right", fontsize=7, labelcolor=style.WARNING)
        lim = np.percentile(np.abs(samples[:, ch].astype(float) - c["bias"]), 99.97) * 1.5 + 5
        a_raw.set_ylim(c["bias"] - lim, c["bias"] + lim)
        a_raw.axhline(c["bias"], color=style.NOISE_FLOOR, lw=0.6, ls="--")
        a_raw.set_ylabel(f"{names[ch]}\nraw\n(counts)", fontsize=8)
        a_flt.plot(t, c["xf"], color=style.SIGNAL_FILT, lw=0.4)
        limf = np.percentile(np.abs(c["xf"]), 99.97) * 1.5 + 1
        a_flt.set_ylim(-limf, limf)
        a_flt.set_ylabel(f"filtered\n{p.band_hz[0]:.0f}-{p.band_hz[1]:.0f} Hz\n(counts)", fontsize=8)
        a_env.semilogy(t, np.maximum(c["env"], 1e-1), color=style.SIGNAL_ENV, lw=0.6, label="envelope")
        a_env.semilogy(t, c["floor"], color=style.NOISE_FLOOR, lw=1.0, ls="--", label="noise floor")
        a_env.semilogy(t, c["thr"], color=style.THRESHOLD, lw=1.0, label=f"threshold (k_ratio={p.k_ratio:g})")
        a_env.set_ylim(max(np.median(c["floor"]) / 4, 0.1), c["env"].max() * 2)
        a_env.set_ylabel("envelope\n(counts, log)", fontsize=8)
        for a in (a_raw, a_flt, a_env):
            if ref is not None:
                for ts in ref.loc[ref.ch == ch, "t_s"]:
                    a.axvline(ts, color=style.REF_ONSET, ls="--", lw=0.8)
            for ts in det.loc[det.ch == ch, "t_s"]:
                a.axvline(ts, color=style.OUR_ONSET, lw=0.9, alpha=0.8)
            a.tick_params(labelsize=7, labelbottom=False)
        if ch == 0:
            from matplotlib.lines import Line2D
            h, l = a_env.get_legend_handles_labels()
            h += [Line2D([], [], color=style.OUR_ONSET, lw=1), Line2D([], [], color=style.REF_ONSET, ls="--")]
            l += ["onset, this tuning", "onset, detections.csv"]
            a_env.legend(h, l, loc="upper right", fontsize=7, ncol=5, framealpha=0.9)
    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel("time since recording start (s)", fontsize=8)

    # ---- zoom on one sound: trigger -> walk-back -> onset
    az = fig.add_subplot(gs[-1])
    c = chains[zoom_ch]
    mine = det[det.ch == zoom_ch]
    if len(mine):
        if zoom_sound is not None and (mine.sound_id == zoom_sound).any():
            o = mine[mine.sound_id == zoom_sound].iloc[0]
        else:
            cand = mine[~mine.clipped]
            if truth is not None and "label" in truth:
                kt = truth[(truth.ch == zoom_ch) & (truth.label == "knock_tap")].t_true.to_numpy()
                is_knock = cand.t_s.apply(lambda v: len(kt) > 0 and np.abs(kt - v).min() < 0.01)
                cand = cand[is_knock] if is_knock.any() else cand
            cand = cand if len(cand) else mine
            o = cand.iloc[(cand.snr - cand.snr.median()).abs().argmin()]
        t0 = o.t_s
        i_lo = max(int((t0 + zoom_ms[0] * 1e-3) * fs), 0)
        i_hi = min(int((t0 + zoom_ms[1] * 1e-3) * fs), len(t))
        tt = (t[i_lo:i_hi] - t0) * 1e3
        az.plot(tt, c["env"][i_lo:i_hi], ".-", color=style.SIGNAL_ENV, lw=0.8, ms=3, label="envelope (dots = samples)")
        az.axhline(c["floor"][o.trigger], color=style.NOISE_FLOOR, ls="--", lw=1, label="noise floor")
        az.axhline(c["thr"][o.trigger], color=style.THRESHOLD, lw=1, label="threshold")
        az.axhline(o.level, color=style.OUR_ONSET, ls=":", lw=1, label=f"onset level (low_frac={p.low_frac:g})")
        tr_ms = (o.trigger / fs - t0) * 1e3
        az.plot(tr_ms, c["env"][o.trigger], "v", color=style.THRESHOLD, ms=9, label=f"trigger fired ({tr_ms:+.2f} ms)")
        az.plot(0, o.level, "o", color=style.OUR_ONSET, ms=8, label="onset (sub-sample)")
        az.annotate("", xy=(0, o.level), xytext=(tr_ms, c["env"][o.trigger]),
                    arrowprops=dict(arrowstyle="->", color=style.OUR_ONSET, lw=1.5))
        walk = "walk-back" if tr_ms >= 0 else "walk-forward (onset level above threshold)"
        az.annotate(f"{walk} {abs(tr_ms):.2f} ms\n+ interpolated between samples", xy=(min(tr_ms, 0) - 0.1, o.level),
                    xytext=(-11, o.peak_env * 0.55), color=style.OUR_ONSET, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=style.OUR_ONSET, lw=0.6, ls=":"))
        info = []
        if ref is not None and len(ref[ref.ch == zoom_ch]):
            rr = ref[ref.ch == zoom_ch]
            k = (rr.t_s - t0).abs().idxmin()
            if abs(rr.at[k, "t_s"] - t0) < 0.01:
                az.axvline((rr.at[k, "t_s"] - t0) * 1e3, color=style.REF_ONSET, ls="--", lw=1, label="detections.csv onset")
        if truth is not None and len(truth[truth.ch == zoom_ch]):
            tr = truth[truth.ch == zoom_ch]
            k = (tr.t_true - t0).abs().idxmin()
            if abs(tr.at[k, "t_true"] - t0) < 0.01:
                e_us = (t0 - tr.at[k, "t_true"]) * 1e6
                az.axvline((tr.at[k, "t_true"] - t0) * 1e3, color=style.TRUE_SPOT, ls=":", lw=1.2, label="true arrival")
                info.append(f"onset {e_us:+.0f} µs vs true arrival ({'late' if e_us > 0 else 'early'})")
                # time-gap error of this sound vs the other channels
                gaps = []
                for och in sorted(set(det.ch) - {zoom_ch}):
                    od = det[(det.ch == och) & (det.sound_id == o.sound_id)]
                    ot = truth[truth.ch == och]
                    if len(od) and len(ot):
                        tt_o = ot.t_true.iloc[(ot.t_true - od.t_s.iloc[0]).abs().argmin()]
                        gaps.append(((od.t_s.iloc[0] - t0) - (tt_o - tr.at[k, "t_true"])) * 1e6)
                if gaps:
                    info.append("but time-gap error to other channel(s): " + ", ".join(f"{g:+.0f} µs" for g in gaps)
                                + "  (the shared offset cancels; only the gap matters)")
        az.set_xlim(*zoom_ms)
        az.set_xlabel(f"time from detected onset (ms) — {names[zoom_ch]}, sound {int(o.sound_id)}", fontsize=8)
        az.set_ylabel("envelope (counts)", fontsize=8)
        az.legend(loc="upper right", fontsize=7, ncol=2)
        az.tick_params(labelsize=7)
        if info:
            az.text(0.01, 0.95, "\n".join(info), transform=az.transAxes, fontsize=9, va="top")
    else:
        az.text(0.5, 0.5, f"No onsets on {names[zoom_ch]} with these parameters", ha="center", transform=az.transAxes,
                color=style.WARNING)
    if title:
        fig.suptitle(title, fontsize=12, y=0.995, ha="center")
    fig.subplots_adjust(top=1 - 0.55 / fig.get_figheight(), bottom=0.05, left=0.07, right=0.99)
    return fig


def finding_title(det, summary=None, names=None, truth_based=False):
    n_sounds = det.sound_id.nunique() if len(det) else 0
    parts = [f"{n_sounds} sounds, {len(det)} onsets found"]
    if summary:
        parts.append(f"{summary['missed']} missed, {summary['false']} false")
        if not np.isnan(summary.get("gap_typical_us", np.nan)):
            out = f", {summary['gap_outliers']} outlier(s) > 0.5 ms" if summary["gap_outliers"] else ""
            parts.append(f"time-gap error typically {summary['gap_typical_us']:.0f} µs vs "
                         f"{'truth' if truth_based else 'detections.csv'}{out}")
    clipped = det[det.clipped]
    if len(clipped):
        names = names or {}
        chs = sorted(set(clipped.ch))
        parts.append("ADC clipped on " + ", ".join(names[c] if names else f"ch{c}" for c in chs))
    else:
        parts.append("no channel clipped")
    return " · ".join(parts)


def measure_jitter(det, ch_a=0, ch_b=1):
    """Stage 2: knock N times from one taped spot. Per-sensor jitter = std(time gap) / sqrt(2)."""
    g = det.pivot_table(index="sound_id", columns="ch", values="t_s").dropna(subset=[ch_a, ch_b])
    gap = g[ch_b] - g[ch_a]
    return dict(n=len(gap), gap_mean_us=gap.mean() * 1e6, gap_std_us=gap.std() * 1e6,
                sigma_s=gap.std() / np.sqrt(2))