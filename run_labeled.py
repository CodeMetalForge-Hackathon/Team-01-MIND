"""Run a labelled 3-node session (Siddhi's sessionN.npz + sessionN.labels.json) through our pipeline.

    python run_labeled.py datasets/2026-11-08/session1.npz
    python dashboard.py --source replay --run session1

Steps: line the nodes up on one clock (nodes_to_run.py, node positions taken from the labels file),
detect + locate + classify (pipeline.py), then mark each located sound with the true spot of the
block it fell in (knock_A, voice_A ...), so the dashboard can show truth vs. estimate.
Prints the real accuracy: median / best error over located sounds that have a true spot.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="3-node recording from Siddhi's recorder")
    ap.add_argument("--labels", help="default: <npz without .npz>.labels.json")
    ap.add_argument("--run", help="run name under data/runs/ (default: file stem)")
    ap.add_argument("--k", type=float, default=5.0, help="detector trigger, x noise floor (lower = more sensitive)")
    ap.add_argument("--sigma", type=float, default=1.5e-3,
                    help="timing error per node, s. Real ESP-NOW-synced Grove nodes: ~1-2 ms (sets the consistency "
                         "check and the size of the accuracy circles)")
    a = ap.parse_args()

    npz = Path(a.npz)
    labels_p = Path(a.labels) if a.labels else npz.with_name(npz.stem + ".labels.json")
    run = a.run or npz.stem
    lab = json.loads(labels_p.read_text())
    pos = {int(k): v for k, v in lab["pos"].items()}
    blocks = lab["blocks"]
    print(f"{run}: nodes at " + ", ".join(f"{n}=({x},{y})" for n, (x, y) in sorted(pos.items())))

    # 1. one shared clock -> normal 3-channel recording + array config with these node positions
    merged = npz.with_name(npz.stem + "_merged.npz")
    subprocess.run([sys.executable, str(ROOT / "nodes_to_run.py"), str(npz), "--out", str(merged),
                    "--sigma", str(a.sigma),
                    "--pos", *[f"{n}={x},{y}" for n, (x, y) in sorted(pos.items())]], check=True)
    (ROOT / "config/params_nodes.yaml").write_text(f"k_ratio: {a.k}\nband_hz: [200.0, 2000.0]\n")

    # 2. detect, locate, classify
    sys.path.insert(0, str(ROOT))
    import pipeline as P
    det, fixes, meta = P.run_pipeline(str(merged.resolve()), run, array="config/array_nodes.yaml",
                                      params=P.load_params("config/params_nodes.yaml"))
    print(f"{len(det.sound_id.unique()) if len(det) else 0} sounds detected, {len(fixes)} located")
    why = pd.Series([("heard on <3 nodes" if str(s["reason"]).startswith("heard on") else
                      "timing inconsistent" if "inconsistent" in str(s["reason"]) else str(s["reason"])[:40])
                     for s in meta.get("skipped_sounds", [])], dtype=object)
    if len(why):
        print("not located: " + ", ".join(f"{k} {v}" for k, v in why.value_counts().items()))

    # 3. ground truth from the labels: which block was each sound in?
    raw = np.load(npz)
    first = min(float(raw[f"n{n}_t_us"][0]) for n in raw["nodes"])
    t0 = float(np.load(merged)["t0_us"])
    offset = (t0 - first) / 1e6            # labels count from the earliest node's first sample
    t_sound = det.groupby("sound_id").t_s.min() + offset if len(det) else pd.Series(dtype=float)

    def block_at(t):
        for b in blocks:
            if b.get("truth") and b["start_s"] <= t <= b["end_s"] + 1.0:
                return b
        return None

    out = ROOT / "data/runs" / run
    rows = []
    fixes["true_label"] = fixes["true_label"].astype(object)
    for i, r in fixes.iterrows():
        b = block_at(float(t_sound.get(r.sound_id, np.nan)))
        if b is None:
            continue
        fixes.loc[i, ["true_x_m", "true_y_m"]] = b["truth"]
        fixes.loc[i, "true_label"] = b["label"]
        if r.kind == "position" and np.isfinite(r.x_m):
            err = float(np.hypot(r.x_m - b["truth"][0], r.y_m - b["truth"][1]))
            rows.append(dict(block=b["label"], kind=b["kind"], err_cm=100 * err))
    fixes.to_csv(out / "fixes.csv", index=False)

    if rows:
        e = pd.DataFrame(rows)
        e.to_csv(out / "accuracy.csv", index=False)
        print("\nREAL ACCURACY (located sounds inside a labelled block, error to the true spot)")
        print(e.groupby("block").err_cm.agg(n="count", median="median", best="min").round(1).to_string())
        print(f"\nall: n={len(e)}  median {e.err_cm.median():.0f} cm  best {e.err_cm.min():.0f} cm  "
              f"80% within {e.err_cm.quantile(0.8):.0f} cm")
    else:
        print("\nno located sound fell inside a labelled block: try a lower --k (e.g. 4) and paste me the output")
    print(f"\nview it:  python dashboard.py --source replay --run {run}")


if __name__ == "__main__":
    main()