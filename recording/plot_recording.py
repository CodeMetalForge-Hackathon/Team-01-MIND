"""Plot a recording and report whether knocks stand out from the noise.

Usage:
    python tools/plot_recording.py data/recordings/stage1_knocks.npz
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np


def main(path: str) -> None:
    rec = np.load(path)
    x = rec["samples"].astype(float)
    fs = float(rec["fs_hz"])
    t = np.arange(len(x)) / fs
    n_ch = x.shape[1]

    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(11, 3 * (n_ch + 1)), squeeze=False)
    for k in range(n_ch):
        sig = x[:, k]
        bias = np.median(sig)
        noise = np.std(sig[: int(fs)])  # first second, assumed quiet
        peak = np.max(np.abs(sig - bias))
        ratio = peak / max(noise, 1e-9)
        clipped = np.mean((sig <= 1) | (sig >= sig.max())) > 0.001 and sig.max() >= 4000
        print(f"ch{k}: bias {bias:.0f} counts, noise {noise:.1f}, biggest swing {peak:.0f} "
              f"(peak/noise {ratio:.0f}x){'  CLIPPING' if clipped else ''}")
        ax = axes[k, 0]
        ax.plot(t, sig, lw=0.5)
        ax.axhline(bias, color="gray", lw=0.8, ls="--")
        ax.set_title(f"Channel {k}: peak/noise {ratio:.0f}x (want > 10x)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ADC counts")

    # Zoom on the loudest moment of channel 0, +-50 ms
    i = int(np.argmax(np.abs(x[:, 0] - np.median(x[:, 0]))))
    w = int(0.05 * fs)
    sl = slice(max(i - w, 0), min(i + w, len(x)))
    ax = axes[n_ch, 0]
    for k in range(n_ch):
        ax.plot((t[sl] - t[i]) * 1000, x[sl, k], lw=1, label=f"ch{k}")
    ax.set_title("Zoom on the loudest knock (0 ms = peak on ch0)")
    ax.set_xlabel("time from peak (ms)")
    ax.set_ylabel("ADC counts")
    ax.legend()

    print(f"dropped packets: {int(rec['dropped_packets'])}, ADC overflows: {int(rec['adc_overflows'])}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
