"""Cut real classifier clips out of Siddhi's recordings (no synthetic data).

Every detected onset becomes a 1.0 s clip of LABEL; quiet stretches between sounds become
"other" (background) clips. Each recording is its own session, so classify.py tests on
recordings it never trained on.

    python make_clips.py recording/mockup/*.npz --label knock_tap --clear
    python make_clips.py data/recordings/voices1.npz --label voice
    python make_clips.py data/recordings/drops1.npz --label impact --k-ratio 6
    python classify.py

Record one label per file (e.g. 20 s of only knocks, 20 s of only calling out), and say which
label it is here. --clear deletes ALL existing clips first (e.g. the synthetic ones).
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np

import detect as D
from arraycfg import ROOT


def session_name(path):
    return Path(path).stem.replace("_", "-")          # '_' is reserved as the clip-number separator


def cut(recording, label, p, bg_label="other", max_bg=None, environment=""):
    z = np.load(ROOT / recording if not Path(recording).is_absolute() else recording)
    s = z["samples"]
    s = s[:, None] if s.ndim == 1 else s
    fs = int(z["fs_hz"])
    ch = int(np.argmax(s.astype(float).std(axis=0)))   # loudest channel
    x = s[:, ch].astype(float)
    x -= np.median(x)
    det, _, _ = D.detect_recording(s[:, [ch]], fs, p, None, 344.0)
    onsets = sorted(det.t_s.tolist())
    sess = session_name(recording)
    pre, n = int(0.25 * fs), fs
    out = {label: [], bg_label: []}
    for t in onsets:
        i0 = int(round(t * fs)) - pre
        if 0 <= i0 and i0 + n <= len(x):
            out[label].append(x[i0:i0 + n])
    # background: 1 s windows at least 0.5 s away from every detected sound
    busy = np.zeros(len(x), bool)
    for t in onsets:
        busy[max(int((t - 0.5) * fs), 0):int((t + 1.0) * fs)] = True
    i = 0
    while i + n <= len(x) and len(out[bg_label]) < (max_bg or len(out[label]) or 10):
        if not busy[i:i + n].any():
            out[bg_label].append(x[i:i + n])
            i += n
        else:
            i += fs // 10
    written = {}
    for lab, clips in out.items():
        d = ROOT / "data/clips" / lab
        d.mkdir(parents=True, exist_ok=True)
        for k, a in enumerate(clips):
            a = (0.9 * a / (np.abs(a).max() + 1e-12)).astype(np.float32)
            np.savez_compressed(d / f"{sess}_{k:03d}.npz", audio=a, fs_hz=fs, label=lab, session=sess,
                                environment=environment or sess, source=str(recording))
        written[lab] = len(clips)
    return written, len(onsets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--label", required=True, choices=["knock_tap", "voice", "impact", "other"],
                    help="what the sounds in these recordings are")
    ap.add_argument("--k-ratio", type=float, default=4.0, help="detection threshold (tune in notebook 01)")
    ap.add_argument("--band", type=float, nargs=2, default=(300.0, 3000.0))
    ap.add_argument("--environment", default="", help="e.g. table, floor, metal")
    ap.add_argument("--no-background", action="store_true", help="don't cut 'other' clips from quiet parts")
    ap.add_argument("--clear", action="store_true", help="delete ALL existing clips first (removes synthetic ones)")
    args = ap.parse_args()
    if args.clear and (ROOT / "data/clips").exists():
        shutil.rmtree(ROOT / "data/clips")
        print("cleared data/clips")
    p = D.DetectParams(k_ratio=args.k_ratio, band_hz=tuple(args.band))
    files = [f for pat in args.recordings for f in (glob.glob(pat) or [pat])]
    for f in files:
        w, n = cut(f, args.label, p, max_bg=0 if args.no_background else None, environment=args.environment)
        print(f"{f}: {n} sounds detected -> " + ", ".join(f"{v} {k}" for k, v in w.items()) + f" (session {session_name(f)})")
    print("next: python classify.py")