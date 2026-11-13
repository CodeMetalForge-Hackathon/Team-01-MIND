"""Live: ESP32 node -> sounds on the map (ATAK + dashboard) within ~1-2 s, while you knock.

    python live.py --port COM5                                 # USB node -> ATAK multicast
    python live.py --port COM5 --atak 127.0.0.1:4242           # WinTAK on this laptop
    python live.py --fake data/recordings/sim-position.npz     # no hardware: stream a recording in real time
    python dashboard.py --source live --run live               # watch it (second terminal)

Every --every seconds it takes the last --window seconds of audio, runs the SAME detect -> locate
-> label steps as pipeline.py (pipeline.locate_sounds), keeps the sounds it has not seen before,
rewrites data/runs/<run>/ (fixes.csv, detections.csv, run.yaml) and pushes the sound sources to ATAK.
Channel count picks the layout: 1 mic -> config/array_single.yaml (no direction: marker on the node),
2 -> config/array.yaml (direction), 3 -> config/array_triangle.yaml (position). Override with --array.

Ctrl+C (or --seconds) stops it and saves the whole stream to data/recordings/<run>.npz, so
`python pipeline.py data/recordings/<run>.npz --run <run>-full` can reprocess it offline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import detect as D
import pipeline as P
from arraycfg import ROOT, load_array

AUTO_ARRAY = {1: "config/array_single.yaml", 2: "config/array.yaml", 3: "config/array_triangle.yaml"}
QUIET = 2048          # firmware maps silence to mid-scale; used to fill dropped packets


class StreamBuffer:
    """Growing [n, n_ch] int16 buffer shared by the reader thread and the processing loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        self.n = 0
        self.fs = None
        self.t_start = None        # unix time of sample 0
        self.dropped = 0
        self.overflows = 0
        self.done = False
        self.error = None

    def append(self, chunk, fs):
        with self.lock:
            if self.data is None:
                self.fs = int(fs)
                self.t_start = time.time() - chunk.shape[0] / self.fs
                self.data = np.empty((self.fs * 60, chunk.shape[1]), np.int16)
            if chunk.shape[1] != self.data.shape[1]:
                raise RuntimeError(f"channel count changed mid-stream ({self.data.shape[1]} -> {chunk.shape[1]})")
            if self.n + len(chunk) > len(self.data):
                grown = np.empty((len(self.data) * 2, self.data.shape[1]), np.int16)
                grown[: self.n] = self.data[: self.n]
                self.data = grown
            self.data[self.n: self.n + len(chunk)] = chunk
            self.n += len(chunk)

    def window(self, i0, i1):
        with self.lock:
            return self.data[i0:i1].copy()


def serial_reader(buf, port_name):
    import serial
    spec = importlib.util.spec_from_file_location("dnl_record", ROOT / "recording/record.py")
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    last_seq = None
    try:
        with serial.Serial(port_name, 921600, timeout=0.2) as port:
            port.reset_input_buffer()
            for seq, fs, _first, ov, samples in rec.read_packets(port):
                if last_seq is not None and seq != (last_seq + 1) % 65536:
                    gap = (seq - last_seq - 1) % 65536
                    buf.dropped += gap      # keep the timeline honest: fill the gap with silence
                    buf.append(np.full((gap * samples.shape[0], samples.shape[1]), QUIET, np.int16), fs)
                last_seq = seq
                buf.overflows += ov
                buf.append(samples.astype(np.int16), fs)
                if buf.done:
                    return
    except Exception as e:   # surfaced by the main loop
        buf.error = f"{type(e).__name__}: {e}"


def fake_reader(buf, path, speed):
    rec = np.load(path)
    s, fs = rec["samples"], int(rec["fs_hz"])
    s = (s[:, None] if s.ndim == 1 else s).astype(np.int16)
    step = 256
    t0 = time.monotonic()
    for i in range(0, len(s), step):
        lag = t0 + (i / fs) / speed - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        buf.append(s[i: i + step], fs)
        if buf.done:
            return
    buf.done = True


def write_atomic(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="serial port of the ESP32 node, e.g. COM5")
    src.add_argument("--fake", help="npz recording to stream in real time instead of hardware")
    ap.add_argument("--speed", type=float, default=1.0, help="--fake only: playback speed")
    ap.add_argument("--run", default="live", help="output folder under data/runs/ (overwritten)")
    ap.add_argument("--array", help="array file (default: by channel count)")
    ap.add_argument("--params", help="detect params yaml (from the tuning notebook)")
    ap.add_argument("--delays", default="config/channel_delays.yaml")
    ap.add_argument("--atak", action="append", help="multicast (default), <ip>:<port>, tcp:<ip>:<port>; repeatable")
    ap.add_argument("--no-atak", action="store_true", help="dashboard only")
    ap.add_argument("--iface", help="laptop IP on the hotspot, if multicast leaves on the wrong adapter")
    ap.add_argument("--radius", type=float, default=3.0, help="1 mic only: guessed hearing radius (m)")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between processing passes")
    ap.add_argument("--window", type=float, default=12.0, help="seconds of audio looked at per pass")
    ap.add_argument("--seconds", type=float, help="stop after this long (default: until Ctrl+C)")
    args = ap.parse_args()

    p = P.load_params(args.params)
    buf = StreamBuffer()
    if args.port:
        threading.Thread(target=serial_reader, args=(buf, args.port), daemon=True).start()
        print(f"Listening on {args.port} ... (Ctrl+C to stop)")
    else:
        threading.Thread(target=fake_reader, args=(buf, ROOT / args.fake, args.speed), daemon=True).start()
        print(f"Streaming {args.fake} as if it were live ... (Ctrl+C to stop)")

    t_wait = time.monotonic()
    while buf.n == 0:
        if buf.error:
            raise SystemExit(f"Could not read the node: {buf.error}")
        if time.monotonic() - t_wait > 5:
            raise SystemExit("No packets after 5 s: is the node plugged in, the firmware flashed, and no other "
                             "program (Arduino Serial Monitor, record.py) holding the port?")
        time.sleep(0.05)
    fs, n_ch = buf.fs, buf.data.shape[1]
    array = args.array or AUTO_ARRAY.get(n_ch)
    if array is None:
        raise SystemExit(f"stream has {n_ch} channels: no default layout, pass --array")
    cfg = load_array(array)
    if len(cfg["_xy"]) != n_ch:
        raise SystemExit(f"stream has {n_ch} channel(s) but {array} lists {len(cfg['_xy'])} sensors")
    delays, calibrated = P.load_delays(args.delays, n_ch)
    sigma_t = float(cfg["timing"]["sigma_s"])
    start = dt.datetime.fromtimestamp(buf.t_start, dt.timezone.utc)
    print(f"{n_ch} channel(s) at {fs} Hz -> {array} ({'single mic: no direction' if n_ch == 1 else cfg['_mode']} mode),"
          f" delays {'calibrated' if calibrated else 'NOT calibrated'}")

    out = ROOT / "data/runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    rows, det_rows, t_event, skipped, known_t = [], [], {}, [], []
    next_id = 0
    hold = int(0.6 * fs)                         # let a sound finish before labelling it
    guard = p.noise_win_s + 0.5                  # noise floor needs this much audio before a sound
    win = int(args.window * fs)

    sender = None
    if not args.no_atak:
        import atak as A
        sender = A.CotSender(args.atak or ["multicast"], args.iface)
        print("ATAK ->", ", ".join(f"{k}://{a[0]}:{a[1]}" for k, a in sender.targets))
    atak_sig, atak_t, last_beat = None, 0.0, 0.0

    def write_run():
        fixes = P.finish_fixes(rows, t_event)
        det = pd.DataFrame(det_rows, columns=P.DET_COLS)
        meta = dict(run=args.run, source="live", recording=f"data/recordings/{args.run}.npz", array=array,
                    mode=cfg["_mode"], medium=cfg["medium"], c_mps=round(cfg["_c_mps"], 2), fs_hz=fs,
                    sigma_t_s=sigma_t, start_time=start.isoformat(), duration_s=round(buf.n / fs, 3),
                    live_input=args.port or f"fake:{args.fake}", live_updated=dt.datetime.now(dt.timezone.utc).isoformat(),
                    dropped_packets=int(buf.dropped), adc_overflows=int(buf.overflows),
                    detect_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in D.asdict(p).items()},
                    channel_delays_s=[float(v) for v in delays], calibrated=calibrated,
                    labeler=("trained classifier (models/classifier.joblib)"
                             if (ROOT / "models/classifier.joblib").exists() else "rules-v0 placeholder (duration + clipping)"),
                    skipped_sounds=[dict(sound_id=s, reason=(f"heard on {r} sensor(s)" if isinstance(r, int) else r))
                                    for s, r in skipped])
        if n_ch == 1:
            meta["single_sensor"] = f"one mic: no direction; fixes sit on the node with a guessed {args.radius:g} m hearing radius"
        write_atomic(out / "fixes.csv", fixes.to_csv(index=False))
        write_atomic(out / "detections.csv", det.to_csv(index=False))
        write_atomic(out / "run.yaml", yaml.safe_dump(meta, sort_keys=False))
        return fixes

    fixes = write_run()
    t_stop = time.monotonic() + args.seconds if args.seconds else None
    try:
        while True:
            tick = time.monotonic()
            if buf.error:
                print(f"!! node stopped: {buf.error}")
                break
            end = buf.n - hold
            w0 = max(0, end - win)
            new = []
            if end - w0 > int(0.5 * fs):
                block = buf.window(w0, end)
                det, blk_rows, blk_t, blk_skip = P.locate_sounds(block, fs, cfg, p, delays, sigma_t,
                                                                 start + dt.timedelta(seconds=w0 / fs),
                                                                 single_radius_m=args.radius)
                for sid, g in (det.groupby("sound_id") if len(det) else []):
                    t_loc = float(g.t_s.min())
                    if w0 > 0 and t_loc < guard:
                        continue                      # too close to the window start: floor not settled
                    t_abs = w0 / fs + t_loc
                    if any(abs(t_abs - k) < p.min_gap_s for k in known_t):
                        continue                      # already reported in an earlier pass
                    known_t.append(t_abs)
                    gid = next_id
                    next_id += 1
                    for r in D.to_contract(g).itertuples(index=False):
                        det_rows.append(dict(sound_id=gid, ch=int(r.ch), t_s=round(r.t_s + w0 / fs, 6),
                                             sample_index=int(r.sample_index) + w0, peak=r.peak, snr=r.snr,
                                             clipped=bool(r.clipped)))
                    row = next((x for x in blk_rows if x["sound_id"] == int(sid)), None)
                    if row is None:
                        why = next((r for s, r in blk_skip if s == int(sid)), "not located")
                        skipped.append((gid, why))
                        print(f"  t={t_abs:6.1f}s  sound #{gid} heard but not located ({why if not isinstance(why, int) else f'on {why} sensor(s)'})")
                        continue
                    row = dict(row, sound_id=gid)
                    rows.append(row)
                    t_event[gid] = t_abs
                    new.append(row)
            if new:
                fixes = write_run()
                for r in new:
                    where = (f"bearing {r['bearing_deg']:.0f}° ± {2 * r['bearing_sigma_deg']:.0f}°" if r["kind"] == "bearing"
                             else f"x {r['x_m']:+.2f} m, y {r['y_m']:+.2f} m")
                    print(f"  t={t_event[r['sound_id']]:6.1f}s  sound #{r['sound_id']}: {r['label']} "
                          f"({r['confidence']:.0%}){'  -> PERSON-LIKE' if r['human_likely'] else ''}  {where}")
            if sender is not None:
                import dashboard as Db
                src = Db.PlaybackSource(out)
                contacts = Db.build_contacts(src.fixes)
                sig = [(c["cid"], len(c["sounds"])) for c in contacts]
                if sig != atak_sig or time.monotonic() - atak_t > 10:
                    err = sender.send(A.sensor_events(src, args.run) + A.source_events(src, contacts, {}, args.run))
                    if err or sig != atak_sig:
                        print(f"  ATAK: {len(contacts)} sound source(s) + {n_ch} sensor(s)" + (f"  ERRORS: {err}" if err else " sent"))
                    atak_sig, atak_t = sig, time.monotonic()
            if time.monotonic() - last_beat > 5:
                tail = buf.window(max(0, buf.n - fs), buf.n).astype(float)
                lvl = "  ".join(f"ch{k}: quiet {np.median(tail[:, k]):.0f}, swing {np.ptp(tail[:, k]):.0f}"
                                for k in range(n_ch))
                print(f"[{buf.n / fs:6.1f}s] {lvl}  | {len(rows)} sounds located"
                      + (f"  | dropped packets {buf.dropped}" if buf.dropped else "")
                      + (f"  | ADC clipping {buf.overflows}" if buf.overflows else ""))
                last_beat = time.monotonic()
            if (t_stop and time.monotonic() > t_stop) or (buf.done and end >= buf.n - hold and w0 + win >= end):
                if buf.done and args.fake:
                    print("fake stream finished")
                break
            time.sleep(max(0.0, args.every - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        buf.done = True
        write_run()
        rec_p = ROOT / "data/recordings" / f"{args.run}.npz"
        rec_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(rec_p, samples=buf.window(0, buf.n), fs_hz=fs, t_start_unix=buf.t_start,
                 dropped_packets=buf.dropped, adc_overflows=buf.overflows, notes=f"live.py run {args.run}")
        print(f"\nStopped after {buf.n / fs:.1f} s: {len(rows)} sounds located, {len(skipped)} heard but not located.")
        print(f"Recording saved: {rec_p.relative_to(ROOT)}   run: data/runs/{args.run}/")
        print(f"Reprocess offline: python pipeline.py {rec_p.relative_to(ROOT).as_posix()} --run {args.run}-full --array {array}")


if __name__ == "__main__":
    main()
