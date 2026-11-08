"""Turn ordinary audio into what our Grove Sound Sensor + ESP32 ADC would have recorded.

Why: public datasets are full-band microphone audio (16-48 kHz, signed). Our nodes record
something very different, measured on quiet_spot6.npz:

  node 1/2 (S2, C5): signal rides a small positive bias, the negative half is clipped at 0
                     (~25 % of samples read exactly 0), loud knocks hit the rail
                     (~2930 counts on the S2, ~1800 on the C5), most energy < 1 kHz.
  node 3 (S3):       much lower gain: resting level ~10 counts, knocks ~100-140 counts,
                     ADC noise of a few counts dominates the quiet parts.

So we model:  y = clip(bias + gain * bandpass(audio), 0, rail) + adc_noise, quantised, at 10 kHz.
Every clip is rendered several times with random gain/bias/filter/noise (domain randomisation),
so the classifier learns sound *shape*, not one sensor's quirks.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

FS_NODE = 10_000

# Measured on quiet_spot6 (see module docstring). "hi" = nodes 1/2, "lo" = node 3.
PROFILES = {
    "hi": dict(rail=(1800, 2950), bias=(20, 250), peak_frac=(0.15, 1.6), adc_noise=(2, 12)),
    "lo": dict(rail=(4095, 4095), bias=(5, 15), peak_frac=(0.008, 0.08), adc_noise=(2, 5)),
}


def to_node_rate(x: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 10 kHz (polyphase, anti-aliased)."""
    if sr == FS_NODE:
        return x.astype(float)
    g = np.gcd(FS_NODE, sr)
    return signal.resample_poly(x.astype(float), FS_NODE // g, sr // g)


def pink_noise(n: int, rng) -> np.ndarray:
    """Room-like background: 1/f noise with extra rumble, unit RMS."""
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / FS_NODE)
    shape = 1 / np.sqrt(np.maximum(f, 20)) * (1 + 3 * (f < 300))
    p = np.fft.irfft(np.fft.rfft(w) * shape, n)
    return p / (p.std() + 1e-12)


def emulate(audio: np.ndarray, sr: int, rng: np.random.Generator, profile: str | None = None,
            background: np.ndarray | None = None, snr_db: float | None = None) -> tuple[np.ndarray, dict]:
    """Render one clip through a randomised Grove sensor. Returns (adc counts as float, params).

    background: optional extra scene sound (already at 10 kHz) mixed in before the sensor.
    snr_db:     level of the event above the background (random 0-35 dB if None).
    """
    profile = profile or ("hi" if rng.random() < 0.67 else "lo")
    P = PROFILES[profile]
    x = to_node_rate(audio, sr)
    x = x - x.mean()
    # electret coupling cap + op-amp bandwidth
    hp = rng.uniform(20, 120)
    lp = rng.uniform(1500, 4500)
    sos = signal.butter(2, [hp, lp], btype="band", fs=FS_NODE, output="sos")
    x = signal.sosfilt(sos, x)
    peak = np.percentile(np.abs(x), 99.9) + 1e-9
    x = x / peak                                   # event peak ~1

    # background before the sensor (the room is always there)
    snr_db = rng.uniform(0, 35) if snr_db is None else snr_db
    bg = pink_noise(len(x), rng)
    if background is not None and len(background):
        b = np.resize(background, len(x))
        b = signal.sosfilt(sos, b - b.mean())
        bg = 0.5 * bg + b / (b.std() + 1e-12)
        bg = bg / (bg.std() + 1e-12)
    x = x + bg * 10 ** (-snr_db / 20) / 3          # /3: peak-vs-rms of the event

    rail = rng.uniform(*P["rail"])
    bias = rng.uniform(*P["bias"])
    gain = rng.uniform(*P["peak_frac"]) * rail
    y = np.clip(bias + gain * x, 0, rail)
    y = y + rng.normal(0, rng.uniform(*P["adc_noise"]), len(y))
    y = np.clip(np.round(y), 0, rail)
    return y, dict(profile=profile, hp=hp, lp=lp, snr_db=snr_db, rail=rail, bias=bias, gain=gain)
