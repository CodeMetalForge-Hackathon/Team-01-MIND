"""Render public datasets through the Grove emulator and extract features -> CSV.

  python -m dnl_classify.build --root C:/Users/siddh/Desktop/dnl_datasets --renders 3 --out data/features_public.csv

Expected layout under --root (each is optional):
  esc50/ESC-50-master/{meta/esc50.csv,audio/*.wav}
  nonspeech7k/{test.csv,test/*.wav}
  librispeech/LibriSpeech/dev-clean/<spk>/<chapter>/*.flac   (needs soundfile; skipped otherwise)

Each clip gets 1.5 s of room noise before it (so the noise floor is known when it starts),
is rendered --renders times with random sensor settings, and the loudest sound per render is kept.
Fold = ESC-50 fold, or a hash of the speaker/source so one source never sits in train and test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from .features import FEATURES, extract
from .grove_emu import FS_NODE, emulate

HERE = Path(__file__).parent


def _fold(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 + 1


def list_clips(root: Path, tax: dict, max_speech: int):
    clips = []                                   # (path, dataset, fine label, coarse, fold)
    esc = root / "esc50/ESC-50-master"
    if (esc / "meta/esc50.csv").exists():
        for r in csv.DictReader(open(esc / "meta/esc50.csv")):
            clips.append((str(esc / "audio" / r["filename"]), "esc50", r["category"],
                          tax["esc50"].get(r["category"], "other"), int(r["fold"])))
    ns = root / "nonspeech7k"
    meta = next(iter(ns.glob("*.csv")), None) if ns.exists() else None
    if meta:
        wavs = {p.name: p for p in ns.rglob("*.wav")}
        rows = list(csv.DictReader(open(meta, encoding="utf-8-sig")))
        for r in rows:                         # Filename, File_ID (source recording), Classname
            p = wavs.get(r["Filename"])
            lab = r["Classname"].strip().lower()
            if p and lab in tax["nonspeech7k"]:
                # several clips are cut from one Freesound recording: fold by source, not clip
                clips.append((str(p), "nonspeech7k", lab, tax["nonspeech7k"][lab], _fold("fs" + r["File_ID"])))
    ls = root / "librispeech/LibriSpeech/dev-clean"
    if ls.exists():
        try:
            import soundfile  # noqa: F401
            flacs = sorted(ls.rglob("*.flac"))
            rng = np.random.default_rng(0)
            for p in rng.permutation(flacs)[:max_speech]:
                spk = Path(p).parts[-3]
                clips.append((str(p), "librispeech", "speech", tax["librispeech"], _fold("spk" + spk)))
        except ImportError:
            print("soundfile not installed: skipping LibriSpeech (pip install soundfile)", file=sys.stderr)
    return clips


def load_audio(path: str):
    if path.endswith(".flac"):
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32")
    else:
        sr, x = wavfile.read(path)
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x[: int(sr * 6.0)], sr               # long utterances: first 6 s is enough


def render_clip(args):
    path, ds, fine, coarse, fold, renders, seed, det = args
    rng = np.random.default_rng(seed)
    try:
        x, sr = load_audio(path)
    except Exception as e:                      # noqa: BLE001
        return [], f"{path}: {e}"
    if np.abs(x).max() == 0:
        return [], None
    x = np.concatenate([np.zeros(int(1.5 * sr), np.float32), x, np.zeros(int(0.5 * sr), np.float32)])
    rows = []
    for k in range(renders):
        y, p = emulate(x, sr, rng)
        evs = [e for e in extract(y, FS_NODE, **det) if e["on"] > int(1.2 * FS_NODE)]
        if not evs:
            continue
        e = max(evs, key=lambda e: e["snr"])
        row = {f: e[f] for f in FEATURES}
        row.update(snr=e["snr"], file=os.path.basename(path), dataset=ds, fine=fine, label=coarse,
                   fold=fold, render=k, profile=p["profile"], sim_snr_db=round(p["snr_db"], 1),
                   n_events=len(evs))
        rows.append(row)
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/Users/siddh/Desktop/dnl_datasets")
    ap.add_argument("--out", default="data/features_public.csv")
    ap.add_argument("--renders", type=int, default=3)
    ap.add_argument("--max-speech", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    a = ap.parse_args()
    tax = yaml.safe_load(open(HERE / "taxonomy.yaml"))
    det = yaml.safe_load(open(HERE / "classify.yaml"))["detection"]
    clips = list_clips(Path(a.root), tax, a.max_speech)
    print(f"{len(clips)} clips: " + ", ".join(f"{d}={sum(c[1] == d for c in clips)}"
                                               for d in ("esc50", "nonspeech7k", "librispeech")))
    jobs = [(*c, a.renders, i, det) for i, c in enumerate(clips)]
    rows, errs = [], 0
    ex = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
    it = ex.map(render_clip, jobs, chunksize=8) if ex else map(render_clip, jobs)
    for i, (r, err) in enumerate(it):
        rows += r
        if err:
            errs += 1
            print("  skip", err)
        if i % 500 == 0:
            print(f"  {i}/{len(jobs)} clips, {len(rows)} sounds", flush=True)
    if ex:
        ex.shutdown()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} sounds to {a.out} ({errs} unreadable clips)")


if __name__ == "__main__":
    main()
