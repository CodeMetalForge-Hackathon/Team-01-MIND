"""Record all three nodes at once, find knocks, triangulate, and plot each node.

Record (one USB port per node; any order, the node ID comes from the data):
    python record_plot.py record --ports COM9 COM10 COM11 --seconds 20 --out data/run1.npz

Analyse and plot a recording:
    python record_plot.py plot data/run1.npz --layout data/layout.json     (or --pos 1=x,y 2=x,y 3=x,y)

Both in one go:
    python record_plot.py both --ports COM9 COM10 COM11 --seconds 20 --layout data/layout.json

Per node it makes one figure with three panels:
  1. ADC vs time (ms), whole recording, detected knocks marked
  2. ADC vs time (ms), zoomed on the chosen knock, onset marked
  3. ADC vs distance (m) travelled by the sound since it was made (after triangulation),
     with the node's computed distance to the source marked; the knock should line up with it.

Node 1 can also carry a vibration sensor (dnl_stream_node, GPIO8 on the S2). Its samples are
saved as n1_vib_* in the .npz, shown as a 4th panel and an extra overview row, and the table's
"vib" column says whether the floor shook at the same moment as each sound.

Each sound is also classified as knock, thud, cough or speaking ("other" if none fits), using the
rule classifier in dnl_classify (thresholds in dnl_classify/classify.yaml). The label comes from
a vote of the nodes that heard it, weighted by how clearly each heard it.

Localisation (tdoa.py): the knock happened at unknown (x, y) at unknown time t0.
For node i at p_i, heard at t_i:   |(x, y) - p_i| = c * (t_i - t0).
  - t_i: GCC-PHAT delays between the nodes' loudness envelopes (sub-sample, upsampled 16x),
    not threshold crossings: on session1 this halved the error and removed the wild fixes.
  - solved in closed form (Fang), refined by Gauss-Newton; two equally good intersections are
    flagged AMBIGUOUS; each fix gets a 95 % ellipse and its GDOP.
  - tdoa.png shows the chosen sound: envelopes before/after alignment, and the hyperbola of each
    node pair over the likelihood map. The overview map shades GDOP (where the layout is accurate).
  - layers.png stacks one map per sensor, bottom to top: mesh (nodes, links, GDOP), sound level,
    located sounds, vibration, thermal (empty until the camera is recorded). Right half: all layers
    on one map; in the window, tick boxes switch each layer on/off.
"""
from __future__ import annotations

import argparse
import math
import struct
import threading
import time
from pathlib import Path

import numpy as np

import tdoa

HDR = struct.Struct("<HBBHIIqHBB")   # magic node synced seq fs first_index first_t_us n overflows reserved
MAGIC = b"\xa6\x5a"                  # 0x5AA6, little-endian


# ============================================================ recording
CH_SOUND, CH_VIB = 0, 1               # header's last byte: which sensor (older firmware always sends 0)


def read_packets(port, with_channel=False):
    """Yield (node, synced, seq, fs, first_t_us, overflows, samples) from one serial port.

    Only sound packets by default (so callers that know nothing about the vibration sensor keep
    working); with_channel=True yields every packet with its channel appended as an 8th item.
    """
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
            _, node, synced, seq, fs, _first_idx, first_t, n, ov, ch = HDR.unpack_from(buf, i)
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
            if with_channel:
                yield node, synced, seq, fs, first_t, ov, samples, ch
            elif ch == CH_SOUND:
                yield node, synced, seq, fs, first_t, ov, samples


def record(ports, seconds, out, settle=0.0):
    import serial
    data = {}
    warned = set()
    port_bytes = {}
    port_ids = {}                    # port -> node ID its board reports
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
            last_seq = {}                # per channel: sound and vibration count packets separately
            for node, synced, seq, fs, first_t, ov, samples, ch in read_packets(Counting(s, name), True):
                now = time.time()
                if now > stop:
                    return
                if now < start:          # settle time: let the clocks sync before recording
                    continue
                with lock:
                    port_ids[name] = node
                    d = data.setdefault((node, ch), dict(port=name, fs=fs, t=[], adc=[], synced=0, packets=0,
                                                         unsynced=0, dropped=0, overflows=0))
                    if d["port"] != name:    # two boards claim the same node ID
                        warned.add(name)
                        continue
                    prev = last_seq.get(ch)
                    if prev is not None and seq != (prev + 1) % 65536:
                        d["dropped"] += (seq - prev - 1) % 65536
                    last_seq[ch] = seq
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
    print(f"Recording {seconds:.0f} s from {', '.join(ports)}.")
    for th in threads:
        th.start()
    for th in threads:
        th.join(settle + seconds + 5)

    if warned:                       # two or more boards claim the same node ID
        ids = ", ".join(f"{p} = node {port_ids[p]}" for p in ports if p in port_ids)
        print(f"!! Node ID clash: {ids}. Only one board per ID is kept; ignored: {', '.join(sorted(warned))}.")
        if 1 not in port_ids.values():
            print("!! No board is node 1, so there is no time master and nothing can sync.")
        print("   Fix: in firmware/dnl_stream_node/dnl_stream_node.ino set  #define FORCE_NODE_ID 0  "
              "(IDs then follow the chip: S2 = 1, C5 = 2, S3 = 3) and re-upload it to every board.")
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
    sound_nodes, vib_nodes = [], []
    for (node, ch), d in sorted(data.items()):
        what = f"node {node}" + (" vibration" if ch == CH_VIB else "")
        if not d["t"]:
            print(f"{what} on {d['port']}: no synced data ({d['unsynced']} unsynced packets). "
                  "Is node 1 (the S2, time master) running?")
            continue
        key = f"n{node}" if ch == CH_SOUND else f"n{node}_vib"
        (sound_nodes if ch == CH_SOUND else vib_nodes).append(node)
        t = np.concatenate(d["t"]).astype(np.int64)
        adc = np.concatenate(d["adc"])
        arrays[f"{key}_t_us"] = t
        arrays[f"{key}_adc"] = adc
        arrays[f"{key}_fs"] = d["fs"]
        arrays[f"{key}_pkt"] = d.get("pkt", 250)
        pk = d.get("pkt", 250)
        backwards = int(np.sum(np.diff(t[::pk]) <= 0))
        if backwards:
            print(f"!! {what}: timestamps jumped backwards {backwards} times: its timing is unstable. "
                  "Re-upload the latest dnl_stream_node.")
        sf = d["synced"] / max(d["packets"], 1)
        print(f"{what} on {d['port']}: {len(adc)} samples ({len(adc) / d['fs']:.1f} s), "
              f"synced {sf * 100:.0f}% (skipped {d['unsynced']} unsynced packets), "
              f"dropped {d['dropped']}, overflows {d['overflows']}"
              + ("" if sf > 0.95 and d["dropped"] == 0 and d["overflows"] == 0 else "   <- check"))
    if not sound_nodes:
        raise SystemExit("No data received. Check the ports and that dnl_stream_node is flashed.")
    arrays["nodes"] = np.array(sound_nodes)
    arrays["vib_nodes"] = np.array(vib_nodes, dtype=int)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"saved {out}")


# ============================================================ classification
CLASS_COLORS = {"knock": "tab:red", "thud": "tab:brown", "cough": "tab:purple",
                "speaking": "tab:green", "other": "0.5"}
_RULES = None


def _rules():
    global _RULES
    if _RULES is None:
        import yaml
        _RULES = yaml.safe_load((Path(__file__).parent / "dnl_classify" / "classify.yaml").read_text())["rules"]
    return _RULES


def classify_events(d, events):
    """Add cls / cls_conf / cls_why to each event of one node (skips ones already done)."""
    from dnl_classify.features import envelopes, event_end, event_features
    from dnl_classify.rules import classify_sound
    todo = [e for e in events if "cls" not in e]
    if not todo:
        return
    if "cenv" not in d:
        d["cenv"], d["cslow"] = envelopes(d["adc"], d["fs"])
        d["cfloor"] = max(float(np.median(d["cenv"])), 1.0)
    env, slow, floor, fs = d["cenv"], d["cslow"], d["cfloor"], d["fs"]
    for e in todo:
        on = int(np.clip(np.searchsorted(d["t"], e["onset_us"]), 0, len(env) - 2))
        s = max(e["idx"], on)
        # at least 30 ms, or there are too few samples to measure the pitch of a clipped click
        end = max(event_end(slow, s, floor, fs), s + 1, on + int(0.03 * fs))
        f = event_features(d["adc"], fs, on, min(end, len(env) - 1), floor, env, slow)
        e["cls"], e["cls_conf"], e["cls_why"] = classify_sound(f, _rules(), d.get("node"))
        e["feat"] = f


def group_class(g):
    """Label of a sound heard by several nodes.

    First the kind of sound (hit / cough / speaking / other) by a vote weighted by how clearly each
    node heard it; then, for a hit, knock vs thud from the node that heard it most clearly, since
    pitch is only trustworthy at a good SNR (nodes 1/2 read a single knock anywhere in 50-1300 Hz).
    """
    family = lambda c: "hit" if c in ("knock", "thud") else c
    score = {}
    for e in g.values():
        k = family(e["cls"])
        score[k] = score.get(k, 0.0) + e["cls_conf"] * math.log10(max(e["snr"], 1.0) + 1)
    best = max(score, key=score.get)
    if best != "hit":
        return best
    return max((e for e in g.values() if family(e["cls"]) == "hit"), key=lambda e: e["snr"])["cls"]


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


def triangulate(positions, times_s, c, sigma_t=tdoa.SIGMA_T):
    """(x, y, t0) from 3+ nodes: closed form + Gauss-Newton, with covariance, GDOP and an ambiguity
    flag. The one solver lives in tdoa.py; this name is kept for live.py / session.py."""
    return tdoa.solve(positions, times_s, c, sigma_t)


def tdoa_times(group, data, pos, c):
    """Arrival time (s) per node for one sound: GCC-PHAT delays between the nodes' envelopes when
    3+ nodes heard it, else the detected onsets. Also stores them as event["tdoa_us"]."""
    if len(group) >= 3 and all(n in pos for n in group):
        times, pairs = tdoa.refine_times(group, {n: (data[n]["t"], data[n]["env"]) for n in group}, pos, c)
    else:
        times, pairs = {n: e["onset_us"] / 1e6 for n, e in group.items()}, {}
    for n, e in group.items():
        e["tdoa_us"] = times[n] * 1e6
    return times, pairs


def group_events(per_node, pos, c, margin_s=0.005, unknown_spacing_m=10.0):
    """Group onsets from different nodes that can be the same sound (travel-time window).
    Nodes without a known position are assumed to be up to unknown_spacing_m apart."""
    ids = sorted(per_node)
    def spacing(a, b):
        return math.dist(pos[a], pos[b]) if a in pos and b in pos else unknown_spacing_m
    dmax = max((spacing(a, b) for a in ids for b in ids if a < b), default=0)
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
            d = spacing(n, n2)
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
        data[n] = dict(node=n, t=t, adc=adc, fs=fs, env=env, events=events, floor=floor, thr=thr)
        classify_events(data[n], events)
        counts = {k: sum(e["cls"] == k for e in events) for k in CLASS_COLORS}
        print(f"node {n}: {len(events)} sounds, noise floor {floor:.1f}, {how}   "
              + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    vib_nodes = [int(v) for v in rec["vib_nodes"]] if "vib_nodes" in rec.files else []
    for n in vib_nodes:
        if n not in data:
            continue
        key = f"n{n}_vib"
        t = rec[f"{key}_t_us"].astype(float)
        adc = rec[f"{key}_adc"].astype(float)
        fs = float(rec[f"{key}_fs"])
        t, fs = fix_timing(t, fs, int(rec[f"{key}_pkt"]), f"{n} vibration")
        order = np.argsort(t, kind="stable")
        t, adc = t[order], adc[order]
        env = envelope(adc, fs)
        events, floor, thr = detect(t, env, fs, k=k)
        data[n]["vib"] = dict(t=t, adc=adc, fs=fs, env=env, events=events, floor=floor, thr=thr)
        print(f"node {n} vibration: {len(events)} hits, noise floor {floor:.1f}, threshold {thr:.1f}")
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
    for n, d in data.items():
        classify_events(d, [g[n] for g in groups if n in g])     # guided onsets
    vib_onsets = np.array(sorted(e["onset_us"] for d in data.values() if "vib" in d for e in d["vib"]["events"]))
    has_vib = any("vib" in d for d in data.values())
    results = []
    print(f"\nspeed of sound {c:.1f} m/s   arrival times: GCC-PHAT delays when 3 nodes heard it, else onsets"
          f"\n{'#':>3}  {'sound':<9} " + ("vib  " if has_vib else "") + "nodes   "
          + "  ".join(f"t{n} (ms)" for n in nodes) + "   position (m), 95% radius, GDOP")
    t_start = min(d["t"][0] for d in data.values())
    for gi, g in enumerate(groups):
        ids = sorted(g)
        sol = None
        arr, pairs = tdoa_times(g, data, pos, c)
        if len(ids) >= 3 and all(n in pos for n in ids):
            sol = triangulate([pos[n] for n in ids], [arr[n] for n in ids], c)
        times = "  ".join((f"{(arr[n] * 1e6 - t_start) / 1000:8.2f}" + ("*" if g[n].get("guided") else " "))
                          if n in g else "       - " for n in nodes)
        where = (f"({sol['x']:.2f}, {sol['y']:.2f}) ±{sol['radius95']:.2f} m, GDOP {sol['gdop']:.1f}"
                 + ("  AMBIGUOUS (2nd fit at ({:.2f}, {:.2f}))".format(*sol["alt"]) if sol["ambiguous"] else "")
                 if sol else ("needs all 3 nodes" if len(ids) < 3 else
                              "missing node positions" if not all(n in pos for n in ids) else
                              "no consistent position (delays exceed the node spacing)"))
        cls = group_class(g)
        # felt through the floor too? a vibration hit from 50 ms before the first ear to 50 ms after the last
        on = [e["onset_us"] for e in g.values()]
        vib = bool(len(vib_onsets)) and bool(np.any((vib_onsets > min(on) - 50_000) & (vib_onsets < max(on) + 50_000)))
        vcol = ("yes  " if vib else "-    ") if has_vib else ""
        print(f"{gi:>3}  {cls:<9} {vcol}{','.join(map(str, ids)):<6}  {times}   {where}")
        results.append(dict(group=g, sol=sol, cls=cls, vib=vib, times=arr, pairs=pairs))
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
        print(f"\nplotting sound #{choose} ({sel['cls']})" + (" (triangulated)" if sel["sol"] else " (not triangulated)"))
    outdir.mkdir(parents=True, exist_ok=True)

    for n, d in sorted(data.items()):
        v = d.get("vib")
        fig, axes = plt.subplots(4 if v else 3, 1, figsize=(12, 13 if v else 10))
        tms = (d["t"] - t_start) / 1000.0
        if v:                                             # panel 4: vibration sensor on this node
            axv = axes[3]
            vms = (v["t"] - t_start) / 1000.0
            vg = np.flatnonzero(np.diff(vms) > 5.0) + 1
            axv.plot(np.insert(vms, vg, np.nan), np.insert(v["adc"], vg, np.nan), lw=0.4, color="tab:orange")
            for e in v["events"]:
                axv.axvline((e["onset_us"] - t_start) / 1000, color="k", lw=0.7, alpha=0.6)
            if sel and n in sel["group"]:
                axv.axvline((sel["group"][n]["onset_us"] - t_start) / 1000, color="tab:red", ls="--",
                            label=f"sound #{choose} heard here")
                axv.legend(fontsize=8, loc="upper right")
            axv.set_title(f"Node {n}: vibration sensor, whole recording ({len(v['events'])} hits, black = detected)")
            axv.set_xlabel("time (ms, shared clock)")
            axv.set_ylabel("ADC counts")
            axv.sharex(axes[0])

        ax = axes[0]
        gaps = np.flatnonzero(np.diff(tms) > 5.0) + 1      # don't draw lines across gaps in the data
        ax.plot(np.insert(tms, gaps, np.nan), np.insert(d["adc"], gaps, np.nan), lw=0.4)
        for e in d["events"]:
            ax.axvline((e["onset_us"] - t_start) / 1000, color=CLASS_COLORS[e["cls"]], lw=0.7, alpha=0.7)
        _class_legend(ax)
        ax.set_title(f"Node {n}: ADC vs time, whole recording ({len(d['events'])} sounds, lines = onsets by class)")
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
            ax.set_title(f"Node {n}: ADC vs time, zoom on sound #{choose}: {sel['cls']} "
                         f"(this node: {e['cls']}, {e['cls_why']})", fontsize=9)

            sol = sel["sol"]
            if sol:
                t0_us = sol["t0_s"] * 1e6
                xm = c * (d["t"] - t0_us) / 1e6
                dist_n = math.dist((sol["x"], sol["y"]), pos[n])
                w2 = (xm > -0.5) & (xm < dist_n + 8.0)
                ax2.plot(xm[w2], d["adc"][w2], lw=0.9)
                ax2.axvline(dist_n, color="tab:green", lw=2, alpha=0.7,
                            label=f"computed distance to source {dist_n:.2f} m")
                ax2.axvline(c * (e.get("tdoa_us", e["onset_us"]) - t0_us) / 1e6, color="tab:red", ls="--",
                            label="this node's arrival (GCC-PHAT)")
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
    rows = [(n, False) for n in nodes] + [(n, True) for n in nodes if "vib" in data[n]]
    fig = plt.figure(figsize=(14 if pos else 11, 3 + 2.2 * len(rows)))
    gs = fig.add_gridspec(len(rows), 2 if pos else 1, width_ratios=[2.2, 1] if pos else [1])
    first_ax = None
    for row, (n, is_vib) in enumerate(rows):
        d = data[n]["vib"] if is_vib else data[n]
        ax = fig.add_subplot(gs[row, 0], sharex=first_ax)
        first_ax = first_ax or ax
        tms = (d["t"] - t_start) / 1000.0
        gaps = np.flatnonzero(np.diff(tms) > 5.0) + 1
        ax.plot(np.insert(tms, gaps, np.nan), np.insert(d["adc"], gaps, np.nan), lw=0.4,
                color="tab:orange" if is_vib else None)
        for e in d["events"]:
            ax.axvline((e["onset_us"] - t_start) / 1000, color="k" if is_vib else CLASS_COLORS[e["cls"]],
                       lw=0.7, alpha=0.7)
        if not is_vib:
            _class_legend(ax)
        ax.set_ylabel(f"node {n}\nvibration" if is_vib else f"node {n}\nADC")
        if row == len(rows) - 1:
            ax.set_xlabel("time (ms, shared clock): the same knock should line up across nodes")
        if row == 0:
            ax.set_title("All nodes on one time axis (lines = detected onsets, coloured by class)")
    if not pos:                                           # no node positions: time plots only
        fig.tight_layout()
        f = outdir / "overview.png"
        fig.savefig(f, dpi=110)
        print(f"saved {f}")
        print("no node positions: skipped the maps and layers.png (add --pos 1=x,y 2=x,y 3=x,y or --layout)")
        if show:
            plt.show()
        return
    axm = fig.add_subplot(gs[:, 1])
    tdoa.draw_layout(axm, pos)
    for i, r in enumerate(results):
        if r["sol"]:
            sx, sy = r["sol"]["x"], r["sol"]["y"]
            el = tdoa.ellipse_xy(r["sol"])
            if el is not None:
                axm.fill(*el, color=CLASS_COLORS[r["cls"]], alpha=0.12, lw=0)
            axm.plot(sx, sy, "*", ms=18 if i == choose else 13, color=CLASS_COLORS[r["cls"]],
                     mec="k" if i == choose else "none")
            axm.annotate(f"#{i} {r['cls']}" + (" +vib" if r.get("vib") else ""), (sx, sy),
                         textcoords="offset points", xytext=(6, 6), fontsize=8)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    axm.set_xlim(min(xs) - 1, max(xs) + 1)
    axm.set_ylim(min(ys) - 1, max(ys) + 1)
    axm.set_aspect("equal")
    axm.grid(alpha=0.3)
    axm.set_title("Map: located sounds with 95% ellipses\n(grey = GDOP: accuracy falls off outside the triangle)",
                  fontsize=10)
    axm.set_xlabel("x (m)")
    axm.set_ylabel("y (m)")
    fig.tight_layout()
    f = outdir / "overview.png"
    fig.savefig(f, dpi=110)
    print(f"saved {f}")
    if sel and sel["sol"]:
        f = plot_tdoa(data, sel, choose, c, t_start, pos, outdir, plt)
        print(f"saved {f}")
    f = plot_layers(data, results, pos, outdir, plt, show)
    print(f"saved {f}")
    if show:
        plt.show()


def plot_tdoa(data, sel, choose, c, t_start, pos, outdir, plt):
    """How one sound was located: aligned envelopes, GCC-PHAT correlations, hyperbolas on the
    likelihood map."""
    g, sol, arr = sel["group"], sel["sol"], sel["times"]
    ids = sorted(g)
    P = [pos[n] for n in ids]
    T = [arr[n] for n in ids]
    colors = {n: f"C{k}" for k, n in enumerate(sorted(data))}
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])

    # (a) envelopes around the sound: raw, then shifted by the GCC-PHAT arrival times
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    tc = np.mean([g[n]["onset_us"] for n in ids])
    for n in ids:
        d = data[n]
        w = (d["t"] > tc - 15_000) & (d["t"] < tc + 45_000)
        env = d["env"][w] / (d["env"][w].max() + 1e-9)
        tms = (d["t"][w] - tc) / 1000
        ax.plot(tms, env, color=colors[n], lw=1.1, label=f"node {n}")
        ax.axvline((g[n]["onset_us"] - tc) / 1000, color=colors[n], ls=":", lw=1)
        axb.plot(tms - (arr[n] * 1e6 - tc) / 1000, env, color=colors[n], lw=1.1, label=f"node {n}")
    ax.set_title(f"Sound #{choose} ({sel['cls']}): loudness envelope at each node, dotted = detected onset", fontsize=10)
    ax.set_ylabel("envelope (normalised)")
    ax.legend(fontsize=8, loc="upper right")
    axb.axvline(0, color="k", lw=0.8)
    axb.set_title("Same envelopes shifted by each node's GCC-PHAT arrival time: they should line up at 0", fontsize=10)
    axb.set_xlabel("time (ms)")
    axb.set_ylabel("envelope (normalised)")

    # (b) likelihood map with one hyperbola per node pair
    axm = fig.add_subplot(gs[:, 1])
    X, Y = tdoa.map_grid(pos, pad=1.0, n=220)
    ll = tdoa.log_likelihood(P, T, c, X, Y)
    im = axm.pcolormesh(X, Y, np.clip(ll, -12, 0), cmap="magma", shading="auto", alpha=0.85)
    fig.colorbar(im, ax=axm, shrink=0.8, label="log-likelihood (0 = best)")
    styles = ["-", "--", ":", "-."]
    for k, (i, j, F) in enumerate(tdoa.hyperbolas(P, T, c, X, Y)):
        cs = axm.contour(X, Y, F, levels=[0], colors="cyan", linestyles=styles[k % 4], linewidths=1.3)
        dr = c * (T[i] - T[j]) * 100
        axm.plot([], [], color="cyan", ls=styles[k % 4], label=f"nodes {ids[i]}-{ids[j]}: Δr = {dr:+.0f} cm")
    tdoa.draw_layout(axm, pos, with_gdop=False)
    el = tdoa.ellipse_xy(sol)
    if el is not None:
        axm.plot(*el, color="white", lw=1.2, label=f"95% ellipse (±{sol['radius95']:.2f} m)")
    axm.plot(sol["x"], sol["y"], "*", ms=18, color=CLASS_COLORS[sel["cls"]], mec="white", label="fix")
    if sol["ambiguous"]:
        axm.plot(*sol["alt"], "X", ms=12, color="orange", label="2nd intersection (ambiguous)")
    axm.set_aspect("equal")
    axm.legend(fontsize=7, loc="upper right", framealpha=0.8)
    axm.set_title(f"Hyperbolas meet at ({sol['x']:.2f}, {sol['y']:.2f}) m, GDOP {sol['gdop']:.1f}", fontsize=10)
    axm.set_xlabel("x (m)")
    axm.set_ylabel("y (m)")
    fig.tight_layout()
    f = outdir / "tdoa.png"
    fig.savefig(f, dpi=110)
    return f


# ============================================================ sensor layers (stacked map)
def _layer_fields(data, results, pos, X, Y):
    """One (x, y) field per sensor layer on the map grid, each scaled 0..1, plus point marks.

    Returns a list of dicts, bottom to top: name, title, cmap, field (or None = no data), points.
    """
    ids = sorted(pos)
    P = np.array([pos[n] for n in ids], float)
    layers = []

    # 0. mesh: node-to-node links over the GDOP field (how well the mesh can locate a sound there)
    G = np.clip(tdoa.gdop_map(P, X, Y), 0, 8) if len(P) >= 3 else None
    layers.append(dict(name="mesh", title="mesh: nodes, links, GDOP", cmap="Greys",
                       field=None if G is None else 1 - (G - G.min()) / (np.ptp(G) + 1e-9), points=[]))

    # 1. sound level: how loud each node heard its sounds (median of its 5 loudest, dB over noise),
    #    spread between nodes by inverse-distance weighting
    lvl = {n: 20 * np.log10(max(np.median(sorted(e["snr"] for e in data[n]["events"])[-5:]), 1.0))
           for n in ids if n in data and data[n]["events"]}
    if lvl:
        w = np.array([1 / (np.hypot(X - pos[n][0], Y - pos[n][1]) ** 2 + 0.05) for n in lvl])
        F = (w * np.array(list(lvl.values()))[:, None, None]).sum(0) / w.sum(0)
        layers.append(dict(name="sound level", title=f"sound level ({min(lvl.values()):.0f}-{max(lvl.values()):.0f} dB over noise)",
                           cmap="YlOrRd", field=(F - F.min()) / (np.ptp(F) + 1e-9), raw=F,
                           points=[(pos[n][0], pos[n][1], "0.25", "s", f"{v:.0f} dB") for n, v in lvl.items()]))
    else:
        layers.append(dict(name="sound level", title="sound level: no sounds", cmap="YlOrRd", field=None, points=[]))

    # 2. located sounds: density of the fixes (each a blob as wide as its 95% radius) + one mark per sound
    fixes = [r for r in results if r["sol"] and tdoa.usable(r["sol"], pos)]
    if fixes:
        D = np.zeros_like(X)
        for r in fixes:
            s = max(min(r["sol"]["radius95"], 1.0), 0.1)
            D += np.exp(-((X - r["sol"]["x"]) ** 2 + (Y - r["sol"]["y"]) ** 2) / (2 * s * s))
        layers.append(dict(name="located sounds", title=f"located sounds ({len(fixes)} fixes, colour = class)",
                           cmap="Blues", field=D / D.max(),
                           points=[(r["sol"]["x"], r["sol"]["y"], CLASS_COLORS[r["cls"]], "*", r["cls"]) for r in fixes]))
    else:
        layers.append(dict(name="located sounds", title="located sounds: none usable", cmap="Blues", field=None, points=[]))

    # 3. vibration: hits per vibration sensor around its node, and the located sounds also felt in the floor
    vib = {n: len(data[n]["vib"]["events"]) for n in ids if n in data and "vib" in data[n]}
    if vib:
        V = np.zeros_like(X)
        for n, k in vib.items():
            V += k * np.exp(-((X - pos[n][0]) ** 2 + (Y - pos[n][1]) ** 2) / (2 * 0.6 ** 2))
        pts = [(pos[n][0], pos[n][1], "0.25", "s", f"{k} hits") for n, k in vib.items()]
        pts += [(r["sol"]["x"], r["sol"]["y"], "tab:orange", "o", "felt") for r in fixes if r.get("vib")]
        layers.append(dict(name="vibration", title=f"vibration ({sum(vib.values())} floor hits)",
                           cmap="Oranges", field=V / max(V.max(), 1e-9), points=pts))
    else:
        layers.append(dict(name="vibration", title="vibration: no sensor in this recording",
                           cmap="Oranges", field=None, points=[]))

    # 4. thermal: slot for the thermal camera, empty until its frames are recorded
    layers.append(dict(name="thermal", title="thermal: no camera data yet", cmap="inferno", field=None, points=[]))
    return layers


def plot_layers(data, results, pos, outdir, plt, show):
    """Sensor layers stacked like a GIS map: 3D stack on the left, all layers flattened on the right
    (tick boxes switch layers on/off in the window)."""
    import matplotlib
    from matplotlib.widgets import CheckButtons
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    X, Y = tdoa.map_grid(pos, pad=1.0, n=60)
    layers = _layer_fields(data, results, pos, X, Y)
    ids = sorted(pos)
    P = np.array([pos[n] for n in ids], float)
    links = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # left: the stack
    ax3 = fig.add_subplot(gs[0, 0], projection="3d", computed_zorder=False)
    for z, L in enumerate(layers):                       # bottom to top, each layer drawn over the one below
        if L["field"] is not None:
            if L["name"] == "mesh":                         # pale base: dark = poor geometry (high GDOP)
                rgba = matplotlib.colormaps["Greys"](0.05 + 0.35 * (1 - L["field"]))
                rgba[..., 3] = 0.5
            else:
                rgba = matplotlib.colormaps[L["cmap"]](0.05 + 0.85 * L["field"])
                rgba[..., 3] = 0.15 + 0.6 * L["field"]       # quiet parts see-through, hot parts solid
        else:
            rgba = np.zeros(X.shape + (4,))
            rgba[...] = (0.8, 0.8, 0.8, 0.12)
        ax3.plot_surface(X, Y, np.full_like(X, z), facecolors=rgba, rstride=1, cstride=1,
                         linewidth=0, antialiased=False, shade=False, zorder=10 * z)
        ex, ey = [X.min(), X.max(), X.max(), X.min(), X.min()], [Y.min(), Y.min(), Y.max(), Y.max(), Y.min()]
        ax3.plot(ex, ey, z, color="0.55", lw=0.7, zorder=10 * z + 1)         # plane outline
        ax3.plot(P[:, 0], P[:, 1], z, "s", ms=5, color="0.25", zorder=10 * z + 2)   # nodes on every layer
        if z == 0:
            for a, b in links:                                                  # mesh links on the base
                ax3.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], 0, color="0.3", ls="--", lw=1,
                         zorder=2)
        for x, y, col, mk, _ in L["points"]:
            if mk != "s":
                ax3.scatter([x], [y], [z], color=col, marker=mk, s=80 if mk == "*" else 35,
                            edgecolors="k", linewidths=0.4, depthshade=False, zorder=10 * z + 3)
        if L["field"] is None:
            ax3.text(np.mean(X), np.mean(Y), z, "no data", fontsize=8, color="0.5", ha="center",
                     zorder=10 * z + 4)
    top = len(layers) - 1
    for n in ids:                                                               # each node is a pillar
        ax3.plot([pos[n][0]] * 2, [pos[n][1]] * 2, [0, top], color="0.4", lw=0.8, alpha=0.6, zorder=10 * top + 5)
        ax3.text(pos[n][0], pos[n][1], top + 0.35, f"node {n}", fontsize=8, ha="center", zorder=10 * top + 6)
    ax3.set_zlim(0, top + 0.3)
    ax3.set_zticks(range(len(layers)))
    ax3.set_zticklabels([L["name"] for L in layers], fontsize=8)
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    span = max(np.ptp(X), np.ptp(Y))
    ax3.set_box_aspect((np.ptp(X) / span, np.ptp(Y) / span, 1.25))
    ax3.view_init(elev=18, azim=-62)
    ax3.set_title("Sensor layers stacked over the node map (bottom: mesh, top: thermal)", fontsize=10)
    ax3.text2D(0.02, 0.02, "\n".join(L["title"] for L in layers), transform=ax3.transAxes, fontsize=8,
               color="0.25", va="bottom")

    # right: the same layers flattened onto one map
    ax = fig.add_subplot(gs[0, 1])
    artists = {}
    for L in layers:
        arts = []
        if L["name"] == "sound level" and L["field"] is not None:
            cs = ax.contour(X, Y, L["raw"], levels=6, cmap="YlOrRd", linewidths=1.0)
            arts += [cs, *ax.clabel(cs, fmt="%.0f dB", fontsize=7)]
        elif L["field"] is not None and L["name"] != "mesh":
            m = np.ma.masked_less(L["field"], 0.08)
            arts.append(ax.pcolormesh(X, Y, m, cmap=L["cmap"], alpha=0.45, shading="auto", vmin=0, vmax=1))
        if L["name"] == "mesh":
            if L["field"] is not None:
                arts.append(ax.contourf(X, Y, L["field"], levels=8, cmap="Greys_r", alpha=0.18))
            for a, b in links:
                arts += ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="0.3", ls="--", lw=1)
                mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
                arts.append(ax.annotate(f"{math.dist(pos[a], pos[b]):.2f} m", (mx, my), fontsize=7, color="0.3",
                                        ha="center", va="bottom"))
            arts += ax.plot(P[:, 0], P[:, 1], "s", ms=12, color="0.25", zorder=5)
            for n in ids:
                arts.append(ax.annotate(f"node {n}", pos[n], textcoords="offset points", xytext=(0, -18),
                                        ha="center", fontsize=9))
        for x, y, col, mk, lab in L["points"]:
            if mk == "s":                                 # node readings: label only, the node is drawn by mesh
                arts.append(ax.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 6), fontsize=7,
                                        color=matplotlib.colormaps[L["cmap"]](0.8)))
            else:
                arts += ax.plot(x, y, mk, ms=13 if mk == "*" else 9, color=col, mec="k", mew=0.4, zorder=6,
                                mfc=col if mk == "*" else "none")
        artists[L["name"]] = arts
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("All layers on one map")
    _class_legend(ax)

    fig.subplots_adjust(left=0.0, right=0.9, top=0.9, bottom=0.07, wspace=0.05)
    f = outdir / "layers.png"
    fig.savefig(f, dpi=110)
    if show:                                              # tick boxes to switch layers on/off in the window
        names = [L["name"] for L in layers]
        bax = fig.add_axes([0.905, 0.72, 0.09, 0.2])
        cb = CheckButtons(bax, names, [True] * len(names))

        def toggle(name):
            for a in artists[name]:
                for part in getattr(a, "collections", None) or [a]:
                    part.set_visible(not part.get_visible())
            fig.canvas.draw_idle()
        cb.on_clicked(toggle)
        fig._layer_buttons = cb                            # keep a reference or the widget stops working
    return f


def _class_legend(ax):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=c, lw=2, label=k) for k, c in CLASS_COLORS.items()],
              fontsize=7, loc="upper right", ncol=len(CLASS_COLORS))


def load_positions(items, layout=None):
    """Node positions from --pos, or from a layout file made by live.py calibration. No defaults."""
    if layout:
        import json
        if not Path(layout).exists():
            raise SystemExit(f"Layout file {layout} not found. Make one by tap calibration "
                             f"(python live.py --ports ... --layout-file {layout}) or use --pos 1=x,y 2=x,y 3=x,y.")
        data = json.loads(Path(layout).read_text())
        return {int(k): (float(v[0]), float(v[1])) for k, v in data["nodes"].items()}
    if items:
        return parse_pos(items)
    raise SystemExit("Node positions are needed: --pos 1=x,y 2=x,y 3=x,y  or  --layout layout.json "
                     "(live.py writes one after you tap each node).")


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
    r.add_argument("--settle", type=float, default=0.0, help="optional seconds to skip at the start (default: none)")
    for name in ("plot", "both"):
        p = sub.add_parser(name)
        if name == "plot":
            p.add_argument("path")
        else:
            p.add_argument("--ports", nargs="+", required=True)
            p.add_argument("--seconds", type=float, default=20)
            p.add_argument("--out", default="data/run.npz")
            p.add_argument("--settle", type=float, default=0.0)
        p.add_argument("--pos", nargs="+", help="node positions in meters, e.g. 1=x,y 2=x,y 3=x,y (no default)")
        p.add_argument("--layout", help="layout.json written by live.py calibration (instead of --pos)")
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
    if a.layout and not Path(a.layout).exists():
        print(f"!! layout file {a.layout} not found. Make one by tap calibration:  python live.py --ports "
              f"COM11 COM12 COM13 --layout-file {a.layout}   or give --pos 1=x,y 2=x,y 3=x,y.")
        print("   Plotting without node positions for now (sounds and classes, no maps).")
        pos = parse_pos(a.pos)
    elif a.pos or a.layout:
        pos = load_positions(a.pos, a.layout)
    else:
        print("no --pos or --layout: plotting without node positions (sounds and classes, no maps).")
        pos = {}
    data, results, c, t_start = analyse(path, pos, a.temp, a.k, a.detector, a.ratio, not a.no_guided)
    plot(data, results, c, t_start, pos, a.sound, Path(path).with_suffix("") / "plots", not a.save_only)


if __name__ == "__main__":
    main()
