"""Record all three nodes at once, find knocks, triangulate, and plot each node.

Record (one USB port per node; any order, the node ID comes from the data):
    python record_plot.py record --ports COM9 COM10 COM11 --seconds 20 --out data/run1.npz

Analyse and plot a recording:
    python record_plot.py plot data/run1.npz --pos 1=0,0 2=2.5,0 3=1.25,2

Both in one go:
    python record_plot.py both --ports COM9 COM10 COM11 --seconds 20 --pos 1=0,0 2=2.5,0 3=1.25,2

Per node it makes one figure with three panels:
  1. ADC vs time (ms), whole recording, detected knocks marked
  2. ADC vs time (ms), zoomed on the chosen knock, onset marked
  3. ADC vs distance (m) travelled by the sound since it was made (after triangulation),
     with the node's computed distance to the source marked; the knock should line up with it.

Triangulation (3 nodes): the knock happened at unknown (x, y) at unknown time t0.
For node i at p_i, heard at t_i:   |(x, y) - p_i| = c * (t_i - t0).
Three equations, three unknowns, solved by Gauss-Newton least squares.
"""
from __future__ import annotations

import argparse
import math
import struct
import threading
import time
from pathlib import Path

import numpy as np

HDR = struct.Struct("<HBBHIIqHBB")   # magic node synced seq fs first_index first_t_us n overflows reserved
MAGIC = b"\xa6\x5a"                  # 0x5AA6, little-endian


# ============================================================ recording
def read_packets(port):
    """Yield (node, synced, seq, fs, first_t_us, overflows, samples) from one serial port."""
    buf = bytearray()
    while True:
        chunk = port.read(4096)
        if chunk:
            buf += chunk
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                del buf[:-1]
                break
            if len(buf) - i < HDR.size:
                break
            _, node, synced, seq, fs, _first_idx, first_t, n, ov, _ = HDR.unpack_from(buf, i)
            if not (1 <= node <= 16 and 0 < n <= 2000 and 1000 <= fs <= 100000):
                del buf[: i + 1]
                continue
            total = HDR.size + 2 * n + 2
            if len(buf) - i < total:
                break
            body = bytes(buf[i : i + total - 2])
            (chk,) = struct.unpack_from("<H", buf, i + total - 2)
            if sum(body) & 0xFFFF != chk:
                del buf[: i + 1]
                continue
            samples = np.frombuffer(body, dtype="<u2", offset=HDR.size).copy()
            del buf[: i + total]
            yield node, synced, seq, fs, first_t, ov, samples


def record(ports, seconds, out, settle=3.0):
    import serial
    data = {}
    warned = set()
    port_bytes = {}
    lock = threading.Lock()
    start = time.time() + settle
    stop = start + seconds

    class Counting:
        """Wraps a serial port and counts raw bytes, to tell 'silent' from 'wrong data'."""
        def __init__(self, port, name):
            self.port, self.name = port, name

        def read(self, n):
            b = self.port.read(n)
            with lock:
                port_bytes[self.name] = port_bytes.get(self.name, 0) + len(b)
            return b

    def worker(name):
        try:
            s = serial.Serial(name, 921600, timeout=0.2)   # normal open: native-USB boards need DTR on
        except serial.SerialException as e:
            print(f"!! could not open {name}: {e}")
            return
        with s:
            s.reset_input_buffer()
            last_seq = None
            for node, synced, seq, fs, first_t, ov, samples in read_packets(Counting(s, name)):
                now = time.time()
                if now > stop:
                    return
                if now < start:          # settle time: let the clocks sync before recording
                    continue
                with lock:
                    d = data.setdefault(node, dict(port=name, fs=fs, t=[], adc=[], synced=0, packets=0,
                                                   unsynced=0, dropped=0, overflows=0))
                    if d["port"] != name:    # two boards claim the same node ID
                        if name not in warned:
                            warned.add(name)
                            print(f"!! {name} and {d['port']} both say they are node {node}. "
                                  f"Ignoring {name}: give each board its own ID (FORCE_NODE_ID) and reflash.")
                        continue
                    if last_seq is not None and seq != (last_seq + 1) % 65536:
                        d["dropped"] += (seq - last_seq - 1) % 65536
                    last_seq = seq
                    d["packets"] += 1
                    d["overflows"] += ov
                    if not synced:       # timestamps not on the shared clock yet: unusable
                        d["unsynced"] += 1
                        continue
                    d["synced"] += 1
                    d["pkt"] = len(samples)
                    d["t"].append(first_t + np.arange(len(samples)) * (1e6 / fs))
                    d["adc"].append(samples)

    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in ports]
    print(f"Waiting {settle:.0f} s for the clocks to sync, then recording {seconds:.0f} s from {', '.join(ports)}.")
    for th in threads:
        th.start()
    time.sleep(settle)
    print("RECORDING NOW: stay quiet 2 s, then knock between the nodes...")
    for th in threads:
        th.join(seconds + 5)

    seen_ports = {d["port"] for d in data.values()}
    for name in ports:
        if name not in seen_ports and name not in warned:
            nb = port_bytes.get(name, 0)
            if nb == 0:
                print(f"!! {name}: no bytes at all. Board not running the sketch: press its RESET, "
                      "check the cable, or re-upload dnl_stream_node.")
            else:
                print(f"!! {name}: received {nb} bytes but no valid packets. It's running a different "
                      "sketch (e.g. a debug sketch): upload dnl_stream_node.")
    arrays = {}
    for node, d in sorted(data.items()):
        if not d["t"]:
            print(f"node {node} on {d['port']}: no synced data ({d['unsynced']} unsynced packets). "
                  "Is node 1 (the S2, time master) running?")
            continue
        t = np.concatenate(d["t"]).astype(np.int64)
        adc = np.concatenate(d["adc"])
        arrays[f"n{node}_t_us"] = t
        arrays[f"n{node}_adc"] = adc
        arrays[f"n{node}_fs"] = d["fs"]
        arrays[f"n{node}_pkt"] = d.get("pkt", 250)
        pk = d.get("pkt", 250)
        backwards = int(np.sum(np.diff(t[::pk]) <= 0))
        if backwards:
            print(f"!! node {node}: timestamps jumped backwards {backwards} times: its timing is unstable. "
                  "Re-upload the latest dnl_stream_node.")
        sf = d["synced"] / max(d["packets"], 1)
        print(f"node {node} on {d['port']}: {len(adc)} samples ({len(adc) / d['fs']:.1f} s), "
              f"synced {sf * 100:.0f}% (skipped {d['unsynced']} unsynced packets), "
              f"dropped {d['dropped']}, overflows {d['overflows']}"
              + ("" if sf > 0.95 and d["dropped"] == 0 and d["overflows"] == 0 else "   <- check"))
    arrays_nodes = sorted(int(k[1:].split("_")[0]) for k in arrays if k.endswith("_adc"))
    if not arrays:
        raise SystemExit("No data received. Check the ports and that dnl_stream_node is flashed.")
    arrays["nodes"] = np.array(arrays_nodes)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"saved {out}")


# ============================================================ analysis
def speed_of_sound(temp_c, rh=50.0):
    return 331.4 + 0.6 * temp_c + 0.0124 * rh


def moving_avg(v, n):
    n = max(int(n), 1)
    c = np.cumsum(np.insert(v.astype(float), 0, 0.0))
    out = np.empty(len(v))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    out[: n - 1] = c[1:n] / np.arange(1, n)
    return out


def envelope(adc, fs):
    """Smoothed loudness: distance from the slowly moving resting level, averaged over ~0.7 ms."""
    x = adc.astype(float)
    base = moving_avg(x, fs * 1.0)
    return moving_avg(np.abs(x - base), max(fs * 0.0007, 1))


def detect(t_us, env, fs, k=5.0, abs_min=15.0, refractory_s=0.3):
    """Knock onsets with sub-sample timing (walk back to 30 % of the threshold, interpolate)."""
    floor = max(float(np.median(env)), 1.0)
    thr = max(k * floor, abs_min)
    above = env > thr
    starts = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    events, last = [], -10**9
    look_back = int(0.010 * fs)
    for s in starts:
        if s - last < refractory_s * fs:
            continue
        last = s
        low = 0.3 * thr
        j = s
        while j > max(s - look_back, 1) and env[j - 1] >= low:
            j -= 1
        j0 = max(j - 1, 0)
        e0, e1 = env[j0], env[j]
        frac = (low - e0) / (e1 - e0) if e1 > e0 else 0.0
        onset = t_us[j0] + float(np.clip(frac, 0, 1)) * (t_us[j] - t_us[j0])
        w = slice(s, min(s + int(0.05 * fs), len(env)))
        events.append(dict(idx=int(s), onset_us=float(onset), peak=float(env[w].max()),
                           snr=float(env[w].max() / floor)))
    return events, floor, thr


def detect_sharp(t_us, env, fs, ratio=6.0, abs_min=15.0, refractory_s=0.3):
    """Knocks = sudden jumps above the sound level of the moment just before.

    Compares the last 2 ms of loudness with the 50 ms before it. Steady chatter raises both
    and doesn't trigger; a knock jumps far above its own background within a millisecond.
    Works in a noisy room where a fixed threshold either misses knocks or fires on chatter.
    """
    short = moving_avg(env, fs * 0.002)
    longw = moving_avg(env, fs * 0.050)
    gap = int(fs * 0.005)
    bg = np.concatenate([np.full(gap, longw[0]), longw[:-gap]])
    floor = max(float(np.median(short)), 1.0)
    r = short / (bg + 0.1 * floor)
    above = (r > ratio) & (short > abs_min)
    starts = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    events, last = [], -10**9
    look_back = int(0.010 * fs)
    for s in starts:
        if s - last < refractory_s * fs:
            continue
        last = s
        base = bg[s]
        w = slice(s, min(s + int(0.05 * fs), len(env)))
        peak = float(env[w].max())
        low = base + 0.25 * (env[s] - base)      # walk back to a quarter of the way up the jump
        j = s
        while j > max(s - look_back, 1) and env[j - 1] >= low:
            j -= 1
        j0 = max(j - 1, 0)
        e0, e1 = env[j0], env[j]
        frac = (low - e0) / (e1 - e0) if e1 > e0 else 0.0
        onset = t_us[j0] + float(np.clip(frac, 0, 1)) * (t_us[j] - t_us[j0])
        events.append(dict(idx=int(s), onset_us=float(onset), peak=peak, snr=float(peak / max(base, 1.0))))
    return events, floor, ratio


def guided_onset(t_us, env, fs, t_center_us, win_us, k_local=3.0):
    """Look for a knock's onset near a known time (another node heard it clearly).

    Uses the local background just before the window, so a node in a noisy spot can still
    pin down the onset of a sound we already know happened. Returns an event dict or None.
    """
    i0 = int(np.searchsorted(t_us, t_center_us - win_us))
    i1 = int(np.searchsorted(t_us, t_center_us + win_us))
    b0 = int(np.searchsorted(t_us, t_center_us - win_us - 100_000))
    if i1 - i0 < 5 or i0 - b0 < 50:
        return None
    bg = float(np.median(env[b0:i0]))
    seg = env[i0:i1]
    peak = float(seg.max())
    if peak < k_local * max(bg, 1.0) or peak < 15:
        return None
    thr = bg + 0.5 * (peak - bg)
    s = i0 + int(np.argmax(seg > thr))
    low = bg + 0.25 * (env[s] - bg)
    j = s
    while j > max(s - int(0.010 * fs), 1) and env[j - 1] >= low:
        j -= 1
    j0 = max(j - 1, 0)
    e0, e1 = env[j0], env[j]
    frac = (low - e0) / (e1 - e0) if e1 > e0 else 0.0
    onset = t_us[j0] + float(np.clip(frac, 0, 1)) * (t_us[j] - t_us[j0])
    return dict(idx=int(s), onset_us=float(onset), peak=peak, snr=peak / max(bg, 1.0), guided=True)


def triangulate(positions, times_s, c, sigma_t=2e-4):
    """(x, y, t0) from 3+ nodes by Gauss-Newton, several starts; returns dict or None."""
    P = np.asarray(positions, float)
    t = np.asarray(times_s, float)
    if len(t) < 3:
        return None
    tref = t.min()
    t = t - tref
    centroid = P.mean(axis=0)
    span = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), 1.0)
    best = None
    for start in [centroid] + [centroid + span * np.array(v) for v in ((1, 0), (-1, 0), (0, 1), (0, -1))]:
        x = np.array([start[0], start[1], float(np.min(c * t - np.hypot(*(P - start).T)))])
        for _ in range(40):
            dx, dy = x[0] - P[:, 0], x[1] - P[:, 1]
            r = np.maximum(np.hypot(dx, dy), 1e-9)
            res = c * t - x[2] - r
            J = np.column_stack([-dx / r, -dy / r, -np.ones_like(r)])
            step = np.linalg.lstsq(J, -res, rcond=None)[0]
            x = x + step
            if np.linalg.norm(step[:2]) < 1e-6:
                break
        r = np.hypot(x[0] - P[:, 0], x[1] - P[:, 1])
        cost = float(np.sum((c * t - x[2] - r) ** 2))
        key = (round(cost, 9), float(np.hypot(*(x[:2] - centroid))))
        if best is None or key < best[0]:
            best = (key, x.copy())
    x = best[1]
    dx, dy = x[0] - P[:, 0], x[1] - P[:, 1]
    r = np.maximum(np.hypot(dx, dy), 1e-9)
    J = np.column_stack([-dx / r, -dy / r, -np.ones_like(r)])
    try:
        cov = (c * sigma_t) ** 2 * np.linalg.inv(J.T @ J)[:2, :2]
        radius = float(math.sqrt(5.991 * max(np.linalg.eigvalsh(cov).max(), 0.0)))
    except np.linalg.LinAlgError:
        radius = float("inf")
    t0 = tref + x[2] / c
    resid = c * (np.asarray(times_s) - t0) - r
    return dict(x=float(x[0]), y=float(x[1]), t0_s=float(t0), dist=r.tolist(),
                resid_m=float(np.sqrt(np.mean(resid ** 2))), radius95=radius)


def group_events(per_node, pos, c, margin_s=0.005):
    """Group onsets from different nodes that can be the same sound (travel-time window)."""
    ids = sorted(per_node)
    dmax = max((math.dist(pos[a], pos[b]) for a in ids for b in ids if a < b and a in pos and b in pos), default=0)
    win = dmax / c + margin_s
    allev = sorted(((e["onset_us"] / 1e6, n, e) for n in ids for e in per_node[n]), key=lambda z: z[0])
    used, groups = set(), []
    for k, (t, n, e) in enumerate(allev):
        if id(e) in used:
            continue
        members = {n: e}
        best = {}
        for t2, n2, e2 in allev[k + 1:]:
            if t2 - t > win:
                break
            if n2 == n or id(e2) in used:
                continue
            d = math.dist(pos.get(n, (0, 0)), pos.get(n2, (0, 0)))
            if abs(t2 - t) <= d / c + margin_s:
                score = e2["snr"]                       # prefer the clearest candidate per node
                if n2 not in best or score > best[n2][0]:
                    best[n2] = (score, e2)
        for n2, (_, e2) in best.items():
            members[n2] = e2
        for e2 in members.values():
            used.add(id(e2))
        groups.append(members)
    return groups


def fix_timing(t, fs, pkt, node):
    """Use the real sample rate measured from packet timestamps, not the one in the header.

    Each packet carries the shared-clock time of its first sample. If the board actually samples
    faster or slower than it claims, the spacing between packet start times shows it.
    """
    if len(t) < 3 * pkt:
        return t, fs
    n_full = len(t) // pkt
    first = t[: n_full * pkt : pkt]
    steps = np.diff(first)
    steps = steps[(steps > 0) & (steps < 10 * pkt * 1e6 / fs)]
    if len(steps) == 0:
        return t, fs
    dt = float(np.median(steps)) / pkt
    fs_real = 1e6 / dt
    if abs(fs_real / fs - 1) > 0.03:
        print(f"node {node}: header says {fs:.0f} Hz but packets show {fs_real:.0f} Hz -> using {fs_real:.0f} Hz")
    idx = np.arange(len(t))
    starts = np.repeat(t[::pkt], pkt)[: len(t)]
    return starts + (idx % pkt) * dt, fs_real


def analyse(path, pos, temp_c, k, detector="level", ratio=6.0, guided=True):
    rec = np.load(path)
    nodes = [int(n) for n in rec["nodes"]]
    c = speed_of_sound(temp_c)
    data = {}
    for n in nodes:
        t = rec[f"n{n}_t_us"].astype(float)
        adc = rec[f"n{n}_adc"].astype(float)
        fs = float(rec[f"n{n}_fs"])
        pkt = int(rec[f"n{n}_pkt"]) if f"n{n}_pkt" in rec.files else 250
        t, fs = fix_timing(t, fs, pkt, n)
        order = np.argsort(t, kind="stable")
        t, adc = t[order], adc[order]
        env = envelope(adc, fs)
        if detector == "sharp":
            events, floor, thr = detect_sharp(t, env, fs, ratio=ratio)
            how = f"jump ratio > {thr:.1f}"
        else:
            events, floor, thr = detect(t, env, fs, k=k)
            how = f"threshold {thr:.1f}"
        data[n] = dict(t=t, adc=adc, fs=fs, env=env, events=events, floor=floor, thr=thr)
        print(f"node {n}: {len(events)} sounds, noise floor {floor:.1f}, {how}")
    groups = group_events({n: d["events"] for n, d in data.items()}, pos, c)
    if guided and len(nodes) >= 3:
        # A sound heard clearly by 2 nodes happened for sure: look for it at the missing node.
        added = 0
        for g in groups:
            if len(g) == 2:
                missing = [n for n in nodes if n not in g]
                tmid = float(np.mean([e["onset_us"] for e in g.values()]))
                for m in missing:
                    if m not in pos:
                        continue
                    dmax = max(math.dist(pos[m], pos[n]) for n in g if n in pos)
                    ev = guided_onset(data[m]["t"], data[m]["env"], data[m]["fs"], tmid,
                                      (dmax / c + 0.005) * 1e6)
                    if ev and all(abs(ev["onset_us"] - e["onset_us"]) / 1e6
                                  <= math.dist(pos[m], pos[n]) / c + 0.005 for n, e in g.items()):
                        g[m] = ev
                        added += 1
        if added:
            print(f"guided search found {added} missed onsets (sounds 2 nodes heard clearly), marked with *")
    results = []
    print(f"\nspeed of sound {c:.1f} m/s\n{'#':>3}  nodes   " + "  ".join(f"t{n} (ms)" for n in nodes) + "   position (m)")
    t_start = min(d["t"][0] for d in data.values())
    for gi, g in enumerate(groups):
        ids = sorted(g)
        sol = None
        if len(ids) >= 3 and all(n in pos for n in ids):
            sol = triangulate([pos[n] for n in ids], [g[n]["onset_us"] / 1e6 for n in ids], c)
        times = "  ".join((f"{(g[n]['onset_us'] - t_start) / 1000:8.2f}" + ("*" if g[n].get("guided") else " "))
                          if n in g else "       - " for n in nodes)
        where = (f"({sol['x']:.2f}, {sol['y']:.2f}) ±{sol['radius95']:.2f}, fit error {sol['resid_m'] * 100:.1f} cm"
                 if sol else ("needs all 3 nodes" if len(ids) < 3 else "missing node positions"))
        print(f"{gi:>3}  {','.join(map(str, ids)):<6}  {times}   {where}")
        results.append(dict(group=g, sol=sol))
    return data, results, c, t_start


# ============================================================ plotting
def plot(data, results, c, t_start, pos, choose, outdir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solved = [i for i, r in enumerate(results) if r["sol"]]
    if choose is None:
        pool = solved or list(range(len(results)))
        choose = max(pool, key=lambda i: min(e["snr"] for e in results[i]["group"].values())) if pool else None
    sel = results[choose] if choose is not None and choose < len(results) else None
    if sel:
        print(f"\nplotting sound #{choose}" + (" (triangulated)" if sel["sol"] else " (not triangulated)"))
    outdir.mkdir(parents=True, exist_ok=True)

    for n, d in sorted(data.items()):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        tms = (d["t"] - t_start) / 1000.0

        ax = axes[0]
        gaps = np.flatnonzero(np.diff(tms) > 5.0) + 1      # don't draw lines across gaps in the data
        ax.plot(np.insert(tms, gaps, np.nan), np.insert(d["adc"], gaps, np.nan), lw=0.4)
        for e in d["events"]:
            ax.axvline((e["onset_us"] - t_start) / 1000, color="tab:red", lw=0.7, alpha=0.6)
        ax.set_title(f"Node {n}: ADC vs time, whole recording ({len(d['events'])} sounds, red = detected onset)")
        ax.set_xlabel("time (ms, shared clock)")
        ax.set_ylabel("ADC counts")

        ax = axes[1]
        ax2 = axes[2]
        if sel and n in sel["group"]:
            e = sel["group"][n]
            on_ms = (e["onset_us"] - t_start) / 1000
            w = (tms > on_ms - 10) & (tms < on_ms + 60)
            ax.plot(tms[w], d["adc"][w], lw=0.9, label="ADC")
            ax.plot(tms[w], d["env"][w] + np.median(d["adc"]), lw=1.2, alpha=0.8, label="envelope (offset)")
            ax.axvline(on_ms, color="tab:red", ls="--", label=f"onset {on_ms:.3f} ms")
            ax.legend(fontsize=8)
            ax.set_title(f"Node {n}: ADC vs time, zoom on sound #{choose}")

            sol = sel["sol"]
            if sol:
                t0_us = sol["t0_s"] * 1e6
                xm = c * (d["t"] - t0_us) / 1e6
                dist_n = math.dist((sol["x"], sol["y"]), pos[n])
                w2 = (xm > -0.5) & (xm < dist_n + 8.0)
                ax2.plot(xm[w2], d["adc"][w2], lw=0.9)
                ax2.axvline(dist_n, color="tab:green", lw=2, alpha=0.7,
                            label=f"computed distance to source {dist_n:.2f} m")
                ax2.axvline(c * (e["onset_us"] - t0_us) / 1e6, color="tab:red", ls="--",
                            label="this node's detected onset")
                ax2.set_xlabel("distance travelled by the sound since it was made (m)")
                ax2.set_title(f"Node {n}: ADC vs distance (source at ({sol['x']:.2f}, {sol['y']:.2f}) m)")
            else:
                first = min(ev["onset_us"] for ev in sel["group"].values())
                xm = c * (d["t"] - first) / 1e6
                w2 = (xm > -1.0) & (xm < 8.0)
                ax2.plot(xm[w2], d["adc"][w2], lw=0.9)
                ax2.axvline(c * (e["onset_us"] - first) / 1e6, color="tab:red", ls="--",
                            label="this node's detected onset")
                ax2.set_xlabel("extra distance travelled compared with the nearest node (m)")
                ax2.set_title(f"Node {n}: ADC vs distance (not triangulated: needs all 3 nodes)")
            ax2.set_ylabel("ADC counts")
            ax2.legend(fontsize=8)
        else:
            ax.set_title(f"Node {n}: did not hear the chosen sound")
            ax2.set_title(f"Node {n}: no distance plot")
        ax.set_xlabel("time (ms, shared clock)")
        ax.set_ylabel("ADC counts")
        fig.tight_layout()
        f = outdir / f"node{n}.png"
        fig.savefig(f, dpi=110)
        print(f"saved {f}")
    # Overview: every node on one shared time axis, plus a map of nodes and located sounds
    nodes = sorted(data)
    fig = plt.figure(figsize=(14, 3 + 2.2 * len(nodes)))
    gs = fig.add_gridspec(len(nodes), 2, width_ratios=[2.2, 1])
    first_ax = None
    for row, n in enumerate(nodes):
        d = data[n]
        ax = fig.add_subplot(gs[row, 0], sharex=first_ax)
        first_ax = first_ax or ax
        tms = (d["t"] - t_start) / 1000.0
        gaps = np.flatnonzero(np.diff(tms) > 5.0) + 1
        ax.plot(np.insert(tms, gaps, np.nan), np.insert(d["adc"], gaps, np.nan), lw=0.4)
        for e in d["events"]:
            ax.axvline((e["onset_us"] - t_start) / 1000, color="tab:red", lw=0.7, alpha=0.6)
        ax.set_ylabel(f"node {n}\nADC")
        if row == len(nodes) - 1:
            ax.set_xlabel("time (ms, shared clock): the same knock should line up across nodes")
        if row == 0:
            ax.set_title("All nodes on one time axis (red = detected onsets)")
    axm = fig.add_subplot(gs[:, 1])
    for n, (x, y) in pos.items():
        axm.plot(x, y, "s", ms=12, color="0.3")
        axm.annotate(f"node {n}", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9)
    for i, r in enumerate(results):
        if r["sol"]:
            sx, sy = r["sol"]["x"], r["sol"]["y"]
            axm.plot(sx, sy, "*", ms=14, color="tab:red" if i == choose else "tab:orange")
            axm.annotate(f"#{i}", (sx, sy), textcoords="offset points", xytext=(6, 6), fontsize=8)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    axm.set_xlim(min(xs) - 1, max(xs) + 1)
    axm.set_ylim(min(ys) - 1, max(ys) + 1)
    axm.set_aspect("equal")
    axm.grid(alpha=0.3)
    axm.set_title("Map: nodes and triangulated sounds")
    axm.set_xlabel("x (m)")
    axm.set_ylabel("y (m)")
    fig.tight_layout()
    f = outdir / "overview.png"
    fig.savefig(f, dpi=110)
    print(f"saved {f}")
    print("Figures open as separate windows (one per node + overview); they may be stacked on top of each other.")
    if show:
        plt.show()


def parse_pos(items):
    pos = {}
    for it in items or []:
        nid, xy = it.split("=")
        x, y = xy.split(",")
        pos[int(nid)] = (float(x), float(y))
    return pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--ports", nargs="+", required=True)
    r.add_argument("--seconds", type=float, default=20)
    r.add_argument("--out", default="data/run.npz")
    r.add_argument("--settle", type=float, default=3.0, help="seconds to wait for clock sync before recording")
    for name in ("plot", "both"):
        p = sub.add_parser(name)
        if name == "plot":
            p.add_argument("path")
        else:
            p.add_argument("--ports", nargs="+", required=True)
            p.add_argument("--seconds", type=float, default=20)
            p.add_argument("--out", default="data/run.npz")
            p.add_argument("--settle", type=float, default=3.0)
        p.add_argument("--pos", nargs="+", default=["1=0,0", "2=2.5,0", "3=1.25,2"],
                       help="node positions in meters, e.g. 1=0,0 2=2.5,0 3=1.25,2")
        p.add_argument("--temp", type=float, default=20.0, help="air temperature, C")
        p.add_argument("--detector", choices=["level", "sharp"], default="level",
                       help="sharp: sudden jumps (best in noisy rooms); level: fixed threshold x noise floor")
        p.add_argument("--ratio", type=float, default=6.0, help="sharp detector: jump size needed (higher = stricter)")
        p.add_argument("--k", type=float, default=5.0, help="level detector: threshold, x noise floor")
        p.add_argument("--no-guided", action="store_true", help="don't search for onsets a node missed")
        p.add_argument("--sound", type=int, default=None, help="which sound # to plot (default: clearest)")
        p.add_argument("--save-only", action="store_true", help="save PNGs without opening windows")
    a = ap.parse_args()

    if a.cmd == "record":
        record(a.ports, a.seconds, a.out, a.settle)
        return
    if a.cmd == "both":
        record(a.ports, a.seconds, a.out, a.settle)
        path = a.out
    else:
        path = a.path
    pos = parse_pos(a.pos)
    data, results, c, t_start = analyse(path, pos, a.temp, a.k, a.detector, a.ratio, not a.no_guided)
    plot(data, results, c, t_start, pos, a.sound, Path(path).with_suffix("") / "plots", not a.save_only)


if __name__ == "__main__":
    main()