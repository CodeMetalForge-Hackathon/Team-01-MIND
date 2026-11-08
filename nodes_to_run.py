"""Turn a 3-node recording from recording/record_plot.py into our pipeline's format.

Siddhi's nodes each stream their own samples stamped with a SHARED clock (ESP-NOW sync).
This lines the nodes up on one common time grid, writes a normal multi-channel recording
(samples [N, n_nodes], fs_hz) and an array config with the nodes where you placed them.
Then the usual pipeline and dashboard work unchanged (position mode, 3 sensors).

    python nodes_to_run.py data/run1.npz --pos 1=0,0 2=2.5,0 3=1.25,2
    python pipeline.py data/run1_merged.npz --run real1 --array config/array_nodes.yaml --params config/params_nodes.yaml
    python dashboard.py --source replay --run real1

--pos must match how the nodes were laid out (metres, node id = x,y), same as record_plot.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def fix_timing(t, fs, pkt):
    """Real sample spacing from packet start times (same idea as record_plot.fix_timing)."""
    if len(t) < 3 * pkt:
        return t, fs
    first = t[: (len(t) // pkt) * pkt: pkt]
    steps = np.diff(first)
    steps = steps[(steps > 0) & (steps < 10 * pkt * 1e6 / fs)]
    if len(steps) == 0:
        return t, fs
    dt = float(np.median(steps)) / pkt
    idx = np.arange(len(t))
    return np.repeat(t[::pkt], pkt)[: len(t)] + (idx % pkt) * dt, 1e6 / dt


def parse_pos(items):
    pos = {}
    for it in items:
        nid, xy = it.split("=")
        x, y = xy.split(",")
        pos[int(nid)] = (float(x), float(y))
    return pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="npz from record_plot.py record")
    ap.add_argument("--pos", nargs="+", default=["1=0,0", "2=2.5,0", "3=1.25,2"])
    ap.add_argument("--out", help="default: <path>_merged.npz")
    ap.add_argument("--array-out", default="config/array_nodes.yaml")
    ap.add_argument("--sigma", type=float, default=2e-4, help="timing error per node, s (ESP-NOW sync ~0.2 ms)")
    a = ap.parse_args()

    rec = np.load(a.path)
    nodes = [int(n) for n in rec["nodes"]]
    pos = parse_pos(a.pos)
    missing = [n for n in nodes if n not in pos]
    if missing:
        raise SystemExit(f"no --pos for node(s) {missing}")

    series = {}
    for n in nodes:
        t = rec[f"n{n}_t_us"].astype(float)
        adc = rec[f"n{n}_adc"].astype(float)
        fs = float(rec[f"n{n}_fs"])
        pkt = int(rec[f"n{n}_pkt"]) if f"n{n}_pkt" in rec.files else 250
        t, fs = fix_timing(t, fs, pkt)
        o = np.argsort(t, kind="stable")
        series[n] = (t[o], adc[o], fs)
        print(f"node {n}: {len(adc)} samples at {fs:.0f} Hz, {(t[o][-1] - t[o][0]) / 1e6:.1f} s")

    t0 = max(s[0][0] for s in series.values())
    t1 = min(s[0][-1] for s in series.values())
    if t1 <= t0:
        raise SystemExit("the nodes' recordings don't overlap in time: was node 1 (time master) running?")
    fs_out = int(round(min(s[2] for s in series.values())))
    grid = t0 + np.arange(int((t1 - t0) / 1e6 * fs_out)) * (1e6 / fs_out)
    cols = []
    for n in nodes:
        t, adc, _ = series[n]
        # gaps (dropped packets) become the node's median level, not a ramp across the gap
        v = np.interp(grid, t, adc)
        k = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        gap = np.minimum(np.abs(t[k] - grid), np.abs(t[k - 1] - grid)) > 3e6 / fs_out
        v[gap] = np.median(adc)
        if gap.any():
            print(f"node {n}: {gap.mean() * 100:.1f}% of the shared span is a gap (dropped packets)")
        cols.append(v)
    samples = np.column_stack(cols).astype(np.float32)
    out = a.out or str(Path(a.path).with_name(Path(a.path).stem + "_merged.npz"))
    np.savez_compressed(out, samples=samples, fs_hz=fs_out, nodes=np.array(nodes), t0_us=t0)
    print(f"saved {out}: {samples.shape[0] / fs_out:.1f} s shared, {len(nodes)} channels at {fs_out} Hz")

    base = yaml.safe_load((ROOT / "config/array_triangle.yaml").read_text())
    base["sampling"]["fs_hz"] = fs_out
    base["timing"]["sigma_s"] = a.sigma
    base["sensors"] = [{"ch": i, "name": f"DNL-N{n}", "x_m": pos[n][0], "y_m": pos[n][1]} for i, n in enumerate(nodes)]
    p = ROOT / a.array_out
    p.write_text("# Written by nodes_to_run.py from --pos. Channel order = node order.\n"
                 + yaml.safe_dump(base, sort_keys=False))
    print(f"saved {a.array_out}")
    params = ROOT / "config/params_nodes.yaml"
    if not params.exists():   # Grove analog mic at ~10 kHz: keep the band under Nyquist, gentler trigger
        params.write_text(yaml.safe_dump({"k_ratio": 5.0, "band_hz": [200.0, 2000.0]}, sort_keys=False))
        print("saved config/params_nodes.yaml (detector settings for the Grove nodes)")


if __name__ == "__main__":
    main()