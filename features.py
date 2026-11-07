"""Per-clip features for the classifier (1.0 s mono clips, any sample rate).

All features are loudness-independent (the clip is peak-normalized first), so a
knock 0.5 m away and one 2 m away look the same apart from noise.
MFCC means are added only if librosa is installed.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert

try:
    import librosa  # optional
except ImportError:  # degrade gracefully
    librosa = None

BASE_FEATURES = ["duration_s", "rise_time_ms", "n_bursts", "spectral_centroid_hz", "spectral_rolloff_hz",
                 "spectral_flatness", "zero_crossing_rate", "pitch_strength", "voiced_frac",
                 "band_low", "band_mid", "band_high", "crest_factor"]
N_MFCC = 13


def feature_names():
    return BASE_FEATURES + ([f"mfcc{i}" for i in range(N_MFCC)] if librosa is not None else [])


def _pitch_strength(frame, fs, fmin=80, fmax=400):
    """Max normalized autocorrelation in the voice pitch range (0 = noise/click, ~1 = steady pitch)."""
    f = frame - frame.mean()
    e = np.dot(f, f)
    if e <= 0:
        return 0.0
    ac = np.correlate(f, f, "full")[len(f) - 1:] / e
    lo, hi = int(fs / fmax), min(int(fs / fmin), len(ac) - 1)
    return float(ac[lo:hi].max()) if hi > lo else 0.0


def extract(audio, fs_hz):
    """Feature dict for one clip."""
    x = np.asarray(audio, float)
    x = x - x.mean()
    x = x / (np.abs(x).max() + 1e-12)
    fs = int(fs_hz)
    env = uniform_filter1d(np.abs(hilbert(x)), max(int(0.005 * fs), 1))
    floor, peak = np.median(env), env.max()       # median = background level (events are short)
    thr = floor + 0.3 * (peak - floor)
    active = env > thr
    duration = active.sum() / fs
    ipk = int(env.argmax())
    lo10 = floor + 0.1 * (peak - floor)
    lo90 = floor + 0.9 * (peak - floor)
    before = env[:ipk + 1]
    i10 = np.flatnonzero(before < lo10)
    i90 = np.flatnonzero(before < lo90)
    rise = ((i90[-1] if len(i90) else ipk) - (i10[-1] if len(i10) else 0)) / fs
    bursts, _ = find_peaks(env, height=floor + 0.4 * (peak - floor), distance=int(0.08 * fs))

    seg = x[active] if active.sum() > 0.01 * fs else x
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / fs)
    p = spec / (spec.sum() + 1e-20)
    centroid = float((freqs * p).sum())
    rolloff = float(freqs[min(np.searchsorted(np.cumsum(p), 0.85), len(freqs) - 1)])
    flatness = float(np.exp(np.mean(np.log(spec + 1e-20))) / (np.mean(spec) + 1e-20))
    zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
    low = p[freqs < 300].sum()
    mid = p[(freqs >= 300) & (freqs < 1500)].sum()
    high = p[freqs >= 1500].sum()

    flen = int(0.04 * fs)
    starts = [s for s in range(0, len(x) - flen, flen // 2) if active[s:s + flen].mean() > 0.5]
    ps = [_pitch_strength(x[s:s + flen], fs) for s in starts]
    pitch = float(np.median(ps)) if ps else 0.0
    voiced = float(np.mean(np.array(ps) > 0.6)) * len(ps) * (flen / 2) / fs if ps else 0.0  # seconds voiced

    f = dict(duration_s=duration, rise_time_ms=rise * 1e3, n_bursts=len(bursts), spectral_centroid_hz=centroid,
             spectral_rolloff_hz=rolloff, spectral_flatness=flatness, zero_crossing_rate=zcr,
             pitch_strength=pitch, voiced_frac=voiced, band_low=low, band_mid=mid, band_high=high,
             crest_factor=float(1.0 / (np.sqrt(np.mean(x ** 2)) + 1e-12)))
    if librosa is not None:
        m = librosa.feature.mfcc(y=x.astype(np.float32), sr=fs, n_mfcc=N_MFCC, n_fft=512, hop_length=128)
        f.update({f"mfcc{i}": float(v) for i, v in enumerate(m.mean(axis=1))})
    return f
