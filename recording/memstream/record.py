"""Record raw frames from the ESP32-S2 to an .npz file.

Usage:
    python tools/record.py --port /dev/ttyACM0 --seconds 20 --out data/recordings/stage1_knocks.npz

Prints a summary: sample rate, channels, dropped packets (seq gaps), ADC overflows.
Saves t_start_unix (computer clock time of the first sample) so recordings from two nodes,
made in two terminals at the same time, can be lined up by plot_recording.py.
Recordings stay out of git (data/ is in .gitignore).
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import serial

HDR = struct.Struct("<HHIIHBB")  # magic, seq, fs_hz, first_index, n_frames, n_ch, flags
MAGIC_BYTES = b"\xa5\x5a"        # 0x5AA5, little-endian


def read_packets(port: serial.Serial):
    """Yield (seq, fs_hz, first_index, overflows, samples[n_frames, n_ch]) forever."""
    buf = bytearray()
    while True:
        buf += port.read(4096)
        while True:
            i = buf.find(MAGIC_BYTES)
            if i < 0:
                del buf[:-1]
                break
            if len(buf) - i < HDR.size:
                break
            _, seq, fs_hz, first, n_frames, n_ch, flags = HDR.unpack_from(buf, i)
            if not (0 < n_ch <= 8 and 0 < n_frames <= 4096):
                del buf[: i + 1]          # false magic match; resync
                continue
            total = HDR.size + n_frames * n_ch * 2 + 2
            if len(buf) - i < total:
                break
            body = bytes(buf[i : i + total - 2])
            (checksum,) = struct.unpack_from("<H", buf, i + total - 2)
            if sum(body) & 0xFFFF != checksum:
                del buf[: i + 1]
                continue
            samples = np.frombuffer(body, dtype="<u2", offset=HDR.size).reshape(n_frames, n_ch)
            del buf[: i + total]
            yield seq, fs_hz, first, flags, samples.copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="data/recordings/stage1.npz")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    chunks, first_index, fs_hz, t_start = [], None, None, None
    last_seq, dropped, overflows = None, 0, 0
    with serial.Serial(args.port, 921600, timeout=0.2) as port:
        port.reset_input_buffer()
        t_end = time.time() + args.seconds
        print(f"Recording {args.seconds:.0f} s. Knock a few times near the sensor...")
        for seq, fs, first, ov, samples in read_packets(port):
            if first_index is None:
                first_index, fs_hz = first, fs
                t_start = time.time() - samples.shape[0] / fs  # first sample, approx (USB delay ~ms)
            elif seq != (last_seq + 1) % 65536:
                dropped += (seq - last_seq - 1) % 65536
            last_seq = seq
            overflows += ov
            chunks.append(samples)
            if time.time() > t_end:
                break

    data = np.concatenate(chunks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, samples=data, fs_hz=fs_hz, first_index=first_index, t_start_unix=t_start,
             dropped_packets=dropped, adc_overflows=overflows, notes=args.notes)

    print(f"Saved {out}: {data.shape[0]} samples x {data.shape[1]} channels at {fs_hz} Hz "
          f"({data.shape[0] / fs_hz:.1f} s)")
    print(f"Dropped packets: {dropped}   ADC overflows: {overflows}"
          + ("   OK" if dropped == 0 and overflows == 0 else "   <- should be 0"))


if __name__ == "__main__":
    main()
