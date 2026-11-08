"""Features of one sound, measured on raw Grove ADC samples at the node rate.

The core features copy the firmware's SoundDetector (firmware/mesh/node/detector.h) so the
rules and a trained model mean the same thing on the laptop and on a node:
  rise_ms, dur_ms, mod_hz, peak, floor, snr, energy
Extra features (laptop only) help a learned model:
  decay_ms   peak -> envelope back below 1/e of peak
  frac_half  share of the sound's duration spent above half its peak (blocky vs spiky)
  n_bursts   separate bursts in the 2 s after onset (knock series, syllables, alarm beeps)
  ioi_cv     regularity of those bursts (0 = metronomic: alarms, knocks in a series)
  zero_frac  share of samples pinned at 0 during the sound (how hard the negative half clips)
  centroid   spectral centroid of the raw signal during the sound, Hz (crude brightness)
  slow_kurt  peakiness of the 15 ms envelope
"""
from __future__ import annotations

import numpy as np
from scipy import signal

FEATURES = ["rise_ms", "rise1_ms", "dur_ms", "mod_hz", "log_snr", "log_energy_per_peak", "decay_ms",
            "frac_half", "n_bursts", "ioi_cv", "zero_frac", "centroid", "slow_kurt"]


def _ema(x, fs, tau_s):
    a = 1.0 - np.exp(-1.0 / (fs * tau_s))
    return signal.lfilter([a], [1, -(1 - a)], x)


def envelopes(adc: np.ndarray, fs: float):
    """Same smoothers as detector.h: resting level 1 s, envelope 0.7 ms, slow envelope 15 ms."""
    x = adc.astype(float)
    base = _ema(x, fs, 1.0)
    base[: int(fs * 0.2)] = np.median(x[: int(fs * 1.0)])     # EMA start-up
    env = _ema(np.abs(x - base), fs, 0.0007)
    slow = _ema(env, fs, 0.015)
    return env, slow


def find_events(adc, fs, k=5.0, abs_min=15.0, refractory_s=0.12, end_frac=0.2):
    """Onsets and ends like the firmware: trigger at k x floor, walk back to 30 %, end after
    150 ms of quiet (slow envelope < 1.8 x floor) or 2 s. Returns list of (onset, end, floor)."""
    env, slow = envelopes(adc, fs)
    n = len(env)
    # noise floor: 1 s rolling 30th percentile of env, sampled every 50 ms (robust to the event itself)
    hop = int(0.05 * fs)
    ref = env[::hop]
    w = max(int(1.0 * fs / hop), 3)
    floors = np.array([np.median(ref[max(0, i - w):i + 1]) for i in range(len(ref))])
    floor = np.maximum(np.repeat(floors, hop)[:n], 1.0)
    thr = np.maximum(k * floor, abs_min)
    above = env > thr
    events, i = [], int(0.3 * fs)                       # skip filter start-up
    quiet_n, max_n = int(0.15 * fs), int(2.0 * fs)
    while i < n:
        hits = np.flatnonzero(above[i:])
        if not len(hits):
            break
        s = i + hits[0]
        f0, t0 = floor[s], thr[s]
        on = s
        while on > max(s - int(0.0064 * fs), 1) and env[on - 1] >= 0.3 * t0:
            on -= 1
        seg = slow[s:min(s + max_n, n)]
        # Firmware: loud = slow > 1.8 x floor. On nodes 1/2 room chatter keeps that true for
        # seconds after a knock (measured on quiet_spot6/7), so here a sound also has to stay
        # above end_frac of its own peak so far. Deliberate laptop-side deviation.
        loud = seg > np.maximum(1.8 * f0, end_frac * np.maximum.accumulate(seg))
        # end = last loud sample before the first 150 ms run of quiet
        last = s
        run = 0
        for j, l in enumerate(loud):
            if l:
                last, run = s + j, 0
            else:
                run += 1
                if run > quiet_n:
                    break
        events.append((on, max(last, s + 1), float(f0)))
        i = max(last, s) + int(refractory_s * fs) + quiet_n
    return events, env, slow


def _count_peaks(slow_dec):
    """Copy of SoundDetector::countPeaks_ (25 % hysteresis, peaks above 35 % of max)."""
    if len(slow_dec) < 3:
        return 0, []
    mx = slow_dec.max()
    if mx <= 0:
        return 0, []
    hyst, ext, rising, peaks, where = 0.25 * mx, slow_dec[0], True, 0, []
    ext_i = 0
    for i in range(1, len(slow_dec)):
        v = slow_dec[i]
        if rising:
            if v > ext:
                ext, ext_i = v, i
            elif ext - v > hyst:
                if ext > 0.35 * mx:
                    peaks += 1
                    where.append(ext_i)
                rising, ext = False, v
        else:
            if v < ext:
                ext = v
            elif v - ext > hyst:
                rising, ext, ext_i = True, v, i
    return peaks, where


def event_features(adc, fs, on, end, floor, env, slow) -> dict:
    seg_env = env[on:end + 1]
    pk_i = int(np.argmax(seg_env))
    peak = float(seg_env[pk_i])
    dur_ms = (end - on) / fs * 1000
    rise_ms = pk_i / fs * 1000
    # rise of the FIRST burst: fast knock series merge into one sound, and rise_ms then measures
    # the time to the loudest knock (hundreds of ms). Onset -> highest point in the first 50 ms.
    rise1_ms = float(np.argmax(seg_env[: int(0.05 * fs)])) / fs * 1000
    dec = slow[on:end + 1:int(fs / 100)]               # slow envelope every 10 ms, like the node
    peaks, _ = _count_peaks(dec)
    mod_hz = peaks / (dur_ms / 1000) if (dur_ms > 250 and peaks >= 2) else 0.0
    energy = float(seg_env.sum() / fs * 1000)

    after = seg_env[pk_i:]
    below = np.flatnonzero(after < peak / np.e)
    decay_ms = (below[0] if len(below) else len(after)) / fs * 1000
    frac_half = float((seg_env > 0.5 * peak).mean())

    # bursts in the 2 s window from onset (context the classifier is allowed to see)
    w2 = slow[on:min(on + int(2 * fs), len(slow)):int(fs / 100)]
    nb, where = _count_peaks(w2)
    ioi = np.diff(where) if len(where) >= 3 else np.array([])
    ioi_cv = float(ioi.std() / ioi.mean()) if len(ioi) and ioi.mean() > 0 else 1.0

    raw = adc[on:end + 1].astype(float)
    zero_frac = float((raw <= 1).mean())
    r = raw - raw.mean()
    if len(r) >= 64:
        f, P = signal.welch(r, fs=fs, nperseg=min(256, len(r)))
        centroid = float((f * P).sum() / (P.sum() + 1e-12))
    else:
        centroid = 0.0
    sd = slow[on:end + 1]
    slow_kurt = float(((sd - sd.mean()) ** 4).mean() / (sd.var() ** 2 + 1e-12)) if len(sd) > 4 else 0.0

    return dict(rise_ms=rise_ms, rise1_ms=rise1_ms, dur_ms=dur_ms, mod_hz=mod_hz, peak=peak, floor=floor,
                snr=peak / max(floor, 1.0), energy=energy,
                log_snr=float(np.log10(peak / max(floor, 1.0))),
                log_energy_per_peak=float(np.log10(energy / max(peak, 1e-6) + 1e-6)),
                decay_ms=decay_ms, frac_half=frac_half, n_bursts=nb, ioi_cv=ioi_cv,
                zero_frac=zero_frac, centroid=centroid, slow_kurt=slow_kurt)


def extract(adc, fs, k=5.0, abs_min=15.0, end_frac=0.2):
    """All sounds in a recording -> list of feature dicts (with onset/end sample indices)."""
    events, env, slow = find_events(adc, fs, k=k, abs_min=abs_min, end_frac=end_frac)
    out = []
    for on, end, fl in events:
        d = event_features(adc, fs, on, end, fl, env, slow)
        d.update(on=on, end=end)
        out.append(d)
    return out
