"""Live mode: knock -> position on the map (and on ATAK) within about a second.
No node positions, origin or compass direction are built in.

FIELD WORKFLOW (nodes dropped anywhere, layout unknown):
    python live.py --ports COM11 COM12 COM13
  1. It asks you to tap each node (3 taps per node, directly on the sensor): the travel time of each tap to the
     other nodes gives the distances between nodes, and from those it builds the layout.
  2. It saves the layout to a file, then goes live: knock anywhere and it shows the position.
  The map is relative: node 1 is the origin, node 2 sits on the +x axis, node 3 on the +y side.

REUSE A LAYOUT (nodes not moved):          python live.py --ports ... --layout-file data/layout.json
  (if the file does not exist yet it is created by the tap calibration)
MEASURED POSITIONS INSTEAD OF TAPS:         python live.py --ports ... --pos 1=x,y 2=x,y 3=x,y
NO HARDWARE (replay a recording):           python live.py --replay data/run.npz --layout-file data/layout.json

ATAK (nothing is assumed about where the nodes are): give two GPS readings from the phone's own position,
standing at node 1 and at node 2, and which side node 3 is on when you face node 2 from node 1:
    python live.py --ports ... --atak --gps1 LAT,LON --gps2 LAT,LON --node3-side left
  (or an explicit frame: --lat0 LAT --lon0 LON --bearing DEG, the compass bearing of the +x axis)
  If markers don't show up: --atak-host <phone IP> --atak-port 4242

What you see: orange dot = located knock; red star = LIKELY PERSON (2+ knocks in the same place within 6 s,
the only thing sent to ATAK unless --atak-all). Ctrl+C or closing the window stops it; everything heard is saved
to data/live/<time>.npz.

Check the calibration against a tape measure (optional): --tape 12=2.5 13=2.3 23=2.1
Other options: --taps 3 --nodes 1 2 3 --max-spacing 10 --temp 20 --k 5 --same-place 0.8 --no-guided
               --min-snr 3 --no-dash --snapshot out.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import record_plot as rp

WINDOW_S = 5.0       # seconds of signal analysed each cycle
CYCLE_S = 0.25       # how often
LATENCY_S = 0.30     # wait this long after a knock so the other nodes' data has arrived
M_PER_DEG_LAT = 111_320.0


# ------------------------------------------------------------------ helpers
def valid_fix(sol, pos, margin=2.0):
    if not sol:
        return False
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if not (min(xs) - margin <= sol["x"] <= max(xs) + margin and min(ys) - margin <= sol["y"] <= max(ys) + margin):
        return False
    return sol["resid_m"] <= 0.05 and math.isfinite(sol["radius95"]) and sol["radius95"] <= 1.5


def gps_pair(text):
    lat, lon = (float(v) for v in text.split(","))
    return lat, lon


class Atak:
    """Cursor-on-Target markers over UDP. The frame (origin, compass direction) is never assumed:
    it comes from --gps1/--gps2 (+ --node3-side) or from explicit --lat0/--lon0/--bearing."""

    def __init__(self, a):
        self.on = a.atak
        self.all = a.atak_all
        self.host, self.port = a.atak_host, a.atak_port
        self.gps1 = gps_pair(a.gps1) if a.gps1 else None
        self.gps2 = gps_pair(a.gps2) if a.gps2 else None
        self.side = a.node3_side
        self.lat0, self.lon0, self.bearing, self.mirror = a.lat0, a.lon0, a.bearing, False
        self.sock = None
        if self.on:
            if not ((self.gps1 and self.gps2) or (a.lat0 is not None and a.lon0 is not None and a.bearing is not None)):
                raise SystemExit("--atak needs to know where the nodes are: give --gps1 LAT,LON --gps2 LAT,LON "
                                 "(phone GPS standing at node 1 and node 2), or --lat0 --lon0 --bearing.")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    @property
    def ready(self):
        return self.lat0 is not None and self.lon0 is not None and self.bearing is not None

    def set_layout(self, pos):
        """Work out origin and compass direction of the local frame from the two GPS anchors."""
        if not (self.gps1 and self.gps2) or 1 not in pos or 2 not in pos:
            return
        (la1, lo1), (la2, lo2) = self.gps1, self.gps2
        d_n = (la2 - la1) * M_PER_DEG_LAT
        d_e = (lo2 - lo1) * M_PER_DEG_LAT * math.cos(math.radians(la1))
        beta12 = math.degrees(math.atan2(d_e, d_n)) % 360          # compass bearing node 1 -> node 2
        vx, vy = pos[2][0] - pos[1][0], pos[2][1] - pos[1][1]
        phi = math.degrees(math.atan2(vy, vx))                      # direction of that vector in the local frame
        self.bearing = (beta12 + phi) % 360                         # compass bearing of the local +x axis
        self.lat0, self.lon0 = la1 - 0.0, lo1
        # which side of the 1->2 line is node 3 in the local frame, versus where it really is
        if 3 in pos:
            local_left = (vx * (pos[3][1] - pos[1][1]) - vy * (pos[3][0] - pos[1][0])) > 0
            self.mirror = local_left != (self.side == "left")

    def latlon(self, x, y):
        if self.mirror:
            y = -y
        b = math.radians(self.bearing)
        east = x * math.sin(b) - y * math.cos(b)
        north = x * math.cos(b) + y * math.sin(b)
        return (self.lat0 + north / M_PER_DEG_LAT,
                self.lon0 + east / (M_PER_DEG_LAT * math.cos(math.radians(self.lat0))))

    def send(self, uid, x, y, ce, callsign, remarks, cot_type="a-u-G", stale_s=600):
        if not self.sock or not self.ready:
            return
        lat, lon = self.latlon(x, y)
        now = dt.datetime.now(dt.timezone.utc)
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        t, stale = now.strftime(fmt), (now + dt.timedelta(seconds=stale_s)).strftime(fmt)
        remarks = remarks.replace("&", "and").replace("<", "").replace(">", "").replace('"', "'")
        xml = (f'<event version="2.0" uid="{uid}" type="{cot_type}" how="m-g" time="{t}" start="{t}" '
               f'stale="{stale}"><point lat="{lat:.7f}" lon="{lon:.7f}" hae="0" ce="{min(ce, 9999):.1f}" '
               f'le="9999999"/><detail><contact callsign="{callsign}"/><remarks>{remarks}</remarks></detail></event>')
        try:
            self.sock.sendto(xml.encode(), (self.host, self.port))
        except OSError as e:
            print(f"[atak] send failed: {e}")

    def send_nodes(self, pos):
        for n, (x, y) in pos.items():
            self.send(f"dnl-node-{n}", x, y, 1.0, f"DNL-{n}", "Drop-and-Listen sensor node",
                      cot_type="a-f-G", stale_s=180)


# ------------------------------------------------------------------ layout from taps
def mds_layout(nodes, D):
    """2-D positions from pairwise distances (classical MDS). Node 1 at the origin, node 2 on +x, node 3 on +y."""
    n = len(nodes)
    M = np.zeros((n, n))
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i < j:
                M[i, j] = M[j, i] = D[(a, b)]
    J = np.eye(n) - np.ones((n, n)) / n
    w, V = np.linalg.eigh(-0.5 * J @ (M ** 2) @ J)
    idx = np.argsort(w)[::-1][:2]
    X = V[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    X = X - X[0]
    ang = math.atan2(X[1, 1], X[1, 0])
    R = np.array([[math.cos(-ang), -math.sin(-ang)], [math.sin(-ang), math.cos(-ang)]])
    X = X @ R.T
    if n >= 3 and X[2, 1] < 0:
        X[:, 1] *= -1
    return {nodes[i]: (float(X[i, 0]), float(X[i, 1])) for i in range(n)}, M


class Calibrator:
    """Tap each node (directly on its sensor) a few times; arrival-time differences give the distances."""

    def __init__(self, nodes, taps, c):
        self.nodes, self.c, self.taps = list(nodes), c, taps
        self.order = [n for n in nodes for _ in range(taps)]
        self.i = 0
        self.d = {}

    @property
    def done(self):
        return self.i >= len(self.order)

    @property
    def target(self):
        return None if self.done else self.order[self.i]

    def taps_done(self, node):
        return min(self.taps, max(0, self.i - self.order.index(node))) if node in self.order else 0

    def distances(self):
        out = {}
        for a in self.nodes:
            for b in self.nodes:
                if a < b:
                    est = self.d.get((a, b), []) + self.d.get((b, a), [])
                    if est:
                        out[(a, b)] = float(np.median(est))
        return out

    def add(self, g):
        """Returns (accepted, message)."""
        src = self.target
        if len(g) < len(self.nodes):
            return False, f"tap heard by nodes {sorted(g)} only: tap harder, directly on node {src}"
        first = min(g, key=lambda n: g[n]["onset_us"])
        if first != src:
            return False, f"that sound reached node {first} first: tap directly ON node {src}'s sensor"
        for j in self.nodes:
            if j != src and (g[j]["onset_us"] - g[src]["onset_us"]) / 1e6 < 0.002:
                return False, f"node {j} heard it less than 2 ms after node {src}: nodes too close, or not a tap on node {src}"
        for j in self.nodes:
            if j != src:
                self.d.setdefault((src, j), []).append(self.c * (g[j]["onset_us"] - g[src]["onset_us"]) / 1e6)
        self.i += 1
        return True, "ok"


# ------------------------------------------------------------------ data in
class Buffers:
    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.lock = threading.Lock()
        self.pk = {n: deque(maxlen=int(WINDOW_S / 0.02)) for n in nodes}
        self.saved = {n: [] for n in nodes}
        self.packets = {n: 0 for n in nodes}
        self.unsynced = {n: 0 for n in nodes}
        self.last_wall = {n: 0.0 for n in nodes}
        self.rate_hist = {n: deque(maxlen=40) for n in nodes}
        self.unknown = set()

    def add(self, node, synced, fs, first_t, samples):
        if node not in self.pk:
            if node not in self.unknown:
                self.unknown.add(node)
                print(f"[live] data from node {node}, which is not in --pos: ignored")
            return
        with self.lock:
            self.packets[node] += 1
            if not synced:
                self.unsynced[node] += 1
                return
            self.pk[node].append((first_t, fs, samples))
            self.saved[node].append((first_t, fs, samples))
            now = time.time()
            self.last_wall[node] = now
            self.rate_hist[node].append(now)

    def window(self, node):
        with self.lock:
            pk = list(self.pk[node])
        if not pk:
            return None, None, None
        t = np.concatenate([f + np.arange(len(s)) * (1e6 / fs) for f, fs, s in pk])
        adc = np.concatenate([s for _, _, s in pk]).astype(float)
        order = np.argsort(t, kind="stable")
        fs = float(np.median([p[1] for p in pk]))
        return t[order], adc[order], fs

    def status(self):
        now = time.time()
        out = {}
        with self.lock:
            for n in self.nodes:
                h = self.rate_hist[n]
                rate = (len(h) - 1) / (h[-1] - h[0]) if len(h) > 5 and h[-1] > h[0] else 0.0
                out[n] = dict(online=now - self.last_wall[n] < 2.0, pps=rate, unsynced=self.unsynced[n])
        return out

    def save(self, folder="data/live"):
        arrays = {}
        for n in self.nodes:
            if not self.saved[n]:
                continue
            arrays[f"n{n}_t_us"] = np.concatenate([f + np.arange(len(s)) * (1e6 / fs)
                                                   for f, fs, s in self.saved[n]]).astype(np.int64)
            arrays[f"n{n}_adc"] = np.concatenate([s for _, _, s in self.saved[n]]).astype(np.uint16)
            arrays[f"n{n}_fs"] = float(np.median([p[1] for p in self.saved[n]]))
            arrays[f"n{n}_pkt"] = len(self.saved[n][0][2])
        if not arrays:
            return None
        arrays["nodes"] = np.array([n for n in self.nodes if f"n{n}_adc" in arrays])
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(folder) / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ".npz")
        np.savez(path, **arrays)
        return path


def serial_source(port, buf, stop):
    import serial
    try:
        s = serial.Serial(port, 921600, timeout=0.2)     # normal open (DTR on): the S2 needs it
    except serial.SerialException as e:
        print(f"!! could not open {port}: {e}")
        return
    with s:
        s.reset_input_buffer()
        for node, synced, seq, fs, first_t, ov, samples in rp.read_packets(s):
            if stop.is_set():
                return
            buf.add(node, synced, fs, first_t, samples)


class Replay(threading.Thread):
    def __init__(self, path, buf, speed, stop):
        super().__init__(daemon=True)
        rec = np.load(path)
        self.chunks = []
        t_ref = None
        for n in [int(x) for x in rec["nodes"]]:
            t = rec[f"n{n}_t_us"].astype(float)
            adc = rec[f"n{n}_adc"]
            fs = float(rec[f"n{n}_fs"])
            pkt = int(rec[f"n{n}_pkt"]) if f"n{n}_pkt" in rec.files else 250
            t, fs = rp.fix_timing(t, fs, pkt, n)
            order = np.argsort(t, kind="stable")
            t, adc = t[order], adc[order]
            t_ref = t[0] if t_ref is None else min(t_ref, t[0])
            for i in range(0, len(t) - pkt + 1, pkt):
                self.chunks.append((t[i], n, fs, adc[i:i + pkt]))
        self.chunks.sort(key=lambda c: c[0])
        self.t_ref, self.buf, self.speed, self.stop = t_ref, buf, speed, stop
        self.duration = (self.chunks[-1][0] - t_ref) / 1e6 if self.chunks else 0.0
        self.done = False

    def run(self):
        start = time.time()
        for t, n, fs, samples in self.chunks:
            while (time.time() - start) * self.speed < (t - self.t_ref) / 1e6:
                if self.stop.is_set():
                    return
                time.sleep(0.005)
            self.buf.add(n, 1, fs, t, samples)
        self.done = True


# ------------------------------------------------------------------ processing
class Processor:
    def __init__(self, a, pos, nodes, buf, atak, layout_path):
        self.a, self.buf, self.atak, self.layout_path = a, buf, atak, layout_path
        self.nodes = sorted(nodes)
        self.pos = pos                                  # None until the layout is known
        self.c = rp.speed_of_sound(a.temp)
        self.cal = None if pos else Calibrator(self.nodes, a.taps, self.c)
        self.lock = threading.Lock()
        self.fixes = []       # dict(id, wall, x, y, radius, nodes, cluster)
        self.clusters = []    # dict(id, fixes=[(x,y)], last_wall, level)
        self.log = deque(maxlen=14)
        self.cal_note = ""
        self.processed_until = None
        self.n_fix = self.n_two = self.n_rejected = 0
        self.t0 = time.time()

    def grouping_pos(self):
        """Positions used only to size the 'same sound' time window. Before the layout is known, every
        pair is assumed to be --max-spacing apart (a physical limit, not a layout)."""
        if self.pos:
            return self.pos
        S = self.a.max_spacing
        base = [(0.0, 0.0), (S, 0.0), (S / 2, S * 0.866)]
        return {n: base[i % 3] if i < 3 else (S * i, S) for i, n in enumerate(self.nodes)}

    def say(self, text, short=None):
        print(text)
        with self.lock:
            self.log.appendleft(short or text)

    def cycle(self):
        data = {}
        for n in self.nodes:
            t, adc, fs = self.buf.window(n)
            if t is None or len(t) < int(2 * fs):
                continue
            env = rp.envelope(adc, fs)
            events, floor, _ = rp.detect(t, env, fs, k=self.a.k)
            data[n] = dict(t=t, env=env, fs=fs, events=events, floor=floor)
        if len(data) < 2:
            return
        gp = self.grouping_pos()
        window_end = min(d["t"][-1] for d in data.values())
        boundary = window_end - LATENCY_S * 1e6
        if self.processed_until is None:
            self.processed_until = max(d["t"][0] for d in data.values()) + 1.0e6   # let the baselines settle
        groups = rp.group_events({n: d["events"] for n, d in data.items()}, gp, self.c)
        if not self.a.no_guided and len(data) >= 3:
            for g in groups:
                if len(g) != 2:
                    continue
                tmid = float(np.mean([e["onset_us"] for e in g.values()]))
                for m in [n for n in data if n not in g]:
                    dmax = max(math.dist(gp[m], gp[n]) for n in g)
                    ev = rp.guided_onset(data[m]["t"], data[m]["env"], data[m]["fs"], tmid, (dmax / self.c + 0.005) * 1e6)
                    if ev and all(abs(ev["onset_us"] - e["onset_us"]) / 1e6
                                  <= math.dist(gp[m], gp[n]) / self.c + 0.005 for n, e in g.items()):
                        g[m] = ev
        lo = self.processed_until
        self.processed_until = max(lo, boundary)
        for g in sorted(groups, key=lambda g: min(e["onset_us"] for e in g.values())):
            anchor = min(e["onset_us"] for e in g.values())
            if lo < anchor <= boundary:
                self.handle_calibration(g) if self.cal else self.handle(g)

    # ---------------------------------------------------------- calibration
    def handle_calibration(self, g):
        src = self.cal.target
        ok, msg = self.cal.add(g)
        if not ok:
            self.say(f"[calibration] {msg}", short=f"x {msg[:44]}")
            return
        k = self.cal.taps_done(src) or self.cal.taps
        self.say(f"[calibration] tap {k}/{self.cal.taps} on node {src} accepted", short=f"tap on node {src} ok")
        if not self.cal.done:
            nxt = self.cal.target
            if nxt != src:
                self.say(f"[calibration] now tap node {nxt} (directly on its sensor), {self.cal.taps} times")
            return
        D = self.cal.distances()
        try:
            self.pos, M = mds_layout(self.nodes, D)
        except Exception as e:
            self.say(f"[calibration] failed ({e}); starting over")
            self.cal = Calibrator(self.nodes, self.a.taps, self.c)
            self.pos = None
            return
        out = {}
        for (a, b), d in D.items():
            out[f"{a}-{b}"] = round(d, 3)
        Path(self.layout_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.layout_path).write_text(json.dumps(dict(
            nodes={str(n): list(v) for n, v in self.pos.items()}, distances_m=out, speed_of_sound=self.c,
            note="estimated from tap calibration; node 1 at origin, node 2 on +x, node 3 on +y"), indent=1))
        self.say("\n[calibration] done. Layout (m): " + "  ".join(f"node {n} ({x:.2f}, {y:.2f})" for n, (x, y) in self.pos.items()))
        self.say("[calibration] node-to-node distances: " + "  ".join(f"{k} = {v:.2f} m" for k, v in out.items()))
        if self.a.tape:
            truth = {}
            for item in self.a.tape:
                key, val = item.split("=")
                truth[(int(key[0]), int(key[1]))] = float(val)
            for (a, b), t in sorted(truth.items()):
                if (a, b) in D:
                    self.say(f"[calibration] tape {a}-{b}: {t:.2f} m, estimated {D[(a, b)]:.2f} m, off by {100 * (D[(a, b)] - t):+.0f} cm")
        self.say(f"[calibration] layout saved: {self.layout_path}. Live: knock anywhere.")
        self.atak.set_layout(self.pos)
        if self.atak.on and self.atak.ready:
            self.atak.send_nodes(self.pos)
        self.cal = None

    # ---------------------------------------------------------- localisation
    def handle(self, g):
        ids = sorted(g)
        t_data = min(e["onset_us"] for e in g.values()) / 1e6      # data time: same behaviour at any replay speed
        if min(e["snr"] for e in g.values()) < self.a.min_snr:
            return
        if len(ids) < 3:
            self.n_two += 1
            return
        sol = rp.triangulate([self.pos[n] for n in ids], [g[n]["onset_us"] / 1e6 for n in ids], self.c)
        if not valid_fix(sol, self.pos):
            self.n_rejected += 1
            return
        now = time.time()
        with self.lock:
            cl = None
            for c in self.clusters:
                mx, my = np.median([p[0] for p in c["fixes"]]), np.median([p[1] for p in c["fixes"]])
                if t_data - c["last_t"] <= 6.0 and math.hypot(sol["x"] - mx, sol["y"] - my) <= self.a.same_place:
                    cl = c
                    break
            if cl is None:
                cl = dict(id=len(self.clusters) + 1, fixes=[], last_wall=now, last_t=t_data)
                self.clusters.append(cl)
            cl["fixes"].append((sol["x"], sol["y"]))
            cl["last_wall"] = now
            cl["last_t"] = t_data
            recent = cl["fixes"][-8:]
            mx, my = float(np.median([p[0] for p in recent])), float(np.median([p[1] for p in recent]))
            level = "likely_person" if len(cl["fixes"]) >= 2 else "confirmed"
            cl["level"], cl["pos"] = level, (mx, my)
            self.n_fix += 1
            fx = dict(id=self.n_fix, wall=now, x=sol["x"], y=sol["y"], radius=sol["radius95"],
                      nodes=ids, cluster=cl["id"], level=level, guided=any(e.get("guided") for e in g.values()))
            self.fixes.append(fx)
            line = (f"[{now - self.t0:6.1f}s] FIX #{fx['id']:<3} ({sol['x']:5.2f}, {sol['y']:5.2f}) "
                    f"±{sol['radius95']:.2f} m  {level.upper()}" + (f"  (cluster {cl['id']}, {len(cl['fixes'])} knocks)"
                                                                    if level == "likely_person" else ""))
            self.log.appendleft(f"#{fx['id']:<3}({sol['x']:.2f},{sol['y']:.2f}) "
                                + (f"PERSON x{len(cl['fixes'])}" if level == "likely_person" else "knock"))
        sent = ""
        if level == "likely_person" or self.atak.all:
            px, py = (mx, my) if level == "likely_person" else (sol["x"], sol["y"])
            self.atak.send(f"dnl-sound-{cl['id']}", px, py, max(sol["radius95"], 0.5),
                           f"KNOCK person-likely x{len(cl['fixes'])}",
                           f"Drop-and-Listen: heard by {len(ids)} nodes, {len(cl['fixes'])} knocks in this place")
            sent = "  -> ATAK" if (self.atak.on and self.atak.ready) else ""
        print(line + sent)


# ------------------------------------------------------------------ display
class Dashboard:
    def __init__(self, proc, buf, show=True):
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt, self.proc, self.buf = plt, proc, buf
        self.fig = plt.figure(figsize=(13, 7))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.7, 1])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.tx = self.fig.add_subplot(gs[0, 1])

    def draw(self):
        ax, tx, proc = self.ax, self.tx, self.proc
        ax.clear()
        tx.clear()
        tx.axis("off")
        status = self.buf.status()
        pos = proc.pos
        now = time.time()
        with proc.lock:
            fixes, clusters = list(proc.fixes), [dict(c) for c in proc.clusters]
            log = list(proc.log)
        lines = [f"speed of sound {proc.c:.1f} m/s", "", "NODES"]
        for n, s in status.items():
            lines.append(f" node {n}: {'online' if s['online'] else 'OFFLINE'}  {s['pps']:.0f} pkt/s")
        if pos:
            for n, (x, y) in pos.items():
                ax.plot(x, y, "s", ms=14, color="#2ca02c" if status.get(n, {}).get("online") else "#777777")
                ax.annotate(f"node {n}", (x, y), textcoords="offset points", xytext=(0, -20), ha="center", fontsize=10)
            for f in fixes:
                age = now - f["wall"]
                if age < 90:
                    ax.plot(f["x"], f["y"], "o", ms=7, color="#ff7f0e", alpha=max(0.25, 1 - age / 90))
            for c in clusters:
                if c["level"] == "likely_person" and now - c["last_wall"] < 90:
                    a = max(0.4, 1 - (now - c["last_wall"]) / 90)
                    ax.plot(*c["pos"], "*", ms=26, color="#d62728", alpha=a)
                    ax.annotate(f"LIKELY PERSON ({len(c['fixes'])} knocks)", c["pos"], textcoords="offset points",
                                xytext=(12, 12), color="#d62728", fontsize=10, alpha=a)
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
            ax.set_ylim(min(ys) - 0.8, max(ys) + 1.0)
            ax.set_title("Drop-and-Listen live   orange: located knock   red star: likely person\n"
                         "(relative map: node 1 = origin, node 2 on +x)", fontsize=10)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        else:
            cal = proc.cal
            tgt = cal.target if cal else None
            ax.text(0.5, 0.78, "CALIBRATING", ha="center", va="center", fontsize=26, color="#1f77b4", transform=ax.transAxes)
            if tgt:
                ax.text(0.5, 0.55, f"Tap node {tgt}\ndirectly on its sensor\n({cal.taps_done(tgt) + 1} of {cal.taps})",
                        ha="center", va="center", fontsize=20, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("No layout yet: tap each node so the system can measure the distances", fontsize=10)
        ax.set_aspect("equal") if pos else None
        ax.grid(alpha=0.3)
        lines += ["", f"fixes {proc.n_fix}  2-node only {proc.n_two}  rejected {proc.n_rejected}", "", "LATEST"]
        lines += [" " + x[:42] for x in log[:12]]
        tx.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)

    def run(self, stop):
        from matplotlib.animation import FuncAnimation
        self._anim = FuncAnimation(self.fig, lambda _: self.draw(), interval=300, cache_frame_data=False)
        self.fig.canvas.mpl_connect("close_event", lambda _: stop.set())
        self.plt.tight_layout()
        self.plt.show()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", nargs="+")
    ap.add_argument("--replay")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    ap.add_argument("--pos", nargs="+", help="measured node positions (optional), e.g. 1=x,y 2=x,y 3=x,y")
    ap.add_argument("--layout-file", help="load this layout if it exists; otherwise calibrate by tapping and save here")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1, 2, 3], help="node IDs in the system")
    ap.add_argument("--taps", type=int, default=3, help="calibration taps per node")
    ap.add_argument("--max-spacing", type=float, default=10.0, help="largest possible distance between nodes (m)")
    ap.add_argument("--tape", nargs="+", help="true distances to compare the calibration with, e.g. 12=2.5 13=2.3 23=2.1")
    ap.add_argument("--temp", type=float, default=20.0)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--min-snr", type=float, default=3.0)
    ap.add_argument("--same-place", type=float, default=0.8, help="m: knocks this close count as the same place")
    ap.add_argument("--no-guided", action="store_true")
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--snapshot", help="save the final dashboard image here")
    ap.add_argument("--atak", action="store_true")
    ap.add_argument("--atak-all", action="store_true", help="also send single (unconfirmed) fixes")
    ap.add_argument("--atak-host", default="239.2.3.1")
    ap.add_argument("--atak-port", type=int, default=6969)
    ap.add_argument("--gps1", help="LAT,LON of node 1 (read from the phone standing at node 1)")
    ap.add_argument("--gps2", help="LAT,LON of node 2")
    ap.add_argument("--node3-side", choices=["left", "right"], default="left",
                    help="which side of the node 1 -> node 2 line node 3 really is on")
    ap.add_argument("--lat0", type=float, help="explicit frame: latitude of the origin (node 1)")
    ap.add_argument("--lon0", type=float)
    ap.add_argument("--bearing", type=float, help="explicit frame: compass bearing of the +x axis, degrees")
    a = ap.parse_args()
    if not a.ports and not a.replay:
        raise SystemExit("Give --ports COM11 COM12 COM13, or --replay a recording.")

    layout_path = a.layout_file or f"data/layout_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    pos = None
    if a.pos:
        pos = rp.parse_pos(a.pos)
        print("Using the node positions you gave with --pos.")
    elif a.layout_file and Path(a.layout_file).exists():
        pos = rp.load_positions(None, a.layout_file)
        print(f"Loaded layout from {a.layout_file}.")
    nodes = sorted(pos) if pos else sorted(a.nodes)
    if len(nodes) < 3:
        raise SystemExit("Need at least 3 nodes.")

    buf = Buffers(nodes)
    stop = threading.Event()
    atak = Atak(a)
    if pos:
        atak.set_layout(pos)
    proc = Processor(a, pos, nodes, buf, atak, layout_path)
    replay = None
    if a.replay:
        replay = Replay(a.replay, buf, a.speed, stop)
        print(f"Replaying {a.replay} ({replay.duration:.0f} s at {a.speed}x)")
        replay.start()
    else:
        for p in a.ports:
            threading.Thread(target=serial_source, args=(p, buf, stop), daemon=True).start()
        print("Listening on", ", ".join(a.ports))
    if proc.cal:
        print(f"\nCALIBRATION: tap node {proc.cal.target} directly on its sensor, {a.taps} times "
              f"(then the other nodes in turn).")
    else:
        print("Layout known. Knock inside the triangle.")
        if atak.on and atak.ready:
            atak.send_nodes(pos)

    def worker():
        last_nodes = time.time()
        while not stop.is_set():
            t = time.time()
            try:
                proc.cycle()
            except Exception as e:                      # never let one bad window stop the demo
                print(f"[live] cycle error: {e}")
            if atak.on and atak.ready and proc.pos and time.time() - last_nodes > 60:
                last_nodes = time.time()
                atak.send_nodes(proc.pos)
            time.sleep(max(0.0, CYCLE_S - (time.time() - t)))

    threading.Thread(target=worker, daemon=True).start()
    dash = Dashboard(proc, buf, show=not a.no_dash)
    try:
        if a.no_dash:
            while not stop.is_set():
                time.sleep(0.2)
                if replay and replay.done:
                    time.sleep(2.0)
                    break
        else:
            dash.run(stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if a.snapshot:
            dash.draw()
            dash.fig.savefig(a.snapshot, dpi=110)
            print(f"snapshot saved: {a.snapshot}")
        saved = buf.save()
        print(f"\nfixes {proc.n_fix}, 2-node only {proc.n_two}, rejected {proc.n_rejected}"
              + (f"; saved {saved}" if saved else ""))
        if proc.cal:
            print("Calibration was not finished: tap each node the number of times it asked for.")


if __name__ == "__main__":
    main()
