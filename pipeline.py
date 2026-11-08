"""Recording -> detections.csv -> fixes.csv, ready for `dashboard.py --source replay`.

    python pipeline.py data/recordings/stage2.npz --run stage2
    python pipeline.py data/recordings/stage2.npz --run stage2-p45 --true-spot P45
    python pipeline.py data/recordings/tri1.npz --run tri1 --array config/array_triangle.yaml

Steps:  detect onsets (detect.py, tuned params)  ->  subtract per-sensor delays
(config/channel_delays.yaml, if calibrated)  ->  solve bearing (2 sensors) or
position (3+)  ->  label (RULES PLACEHOLDER until the Prompt 3 classifier)  ->
flag repeated knock patterns  ->  write the run folder.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import detect as D
from style import CHI2_95_2D
from arraycfg import ROOT, front_vector, load_array, local_angle_to_compass
from localize import solve_bearing, solve_position

FIX_COLS = ["sound_id", "kind", "wall_time", "bearing_deg", "bearing_sigma_deg", "x_m", "y_m", "cov_xx",
            "cov_xy", "cov_yy", "residual_m", "label", "confidence", "human_likely", "repeated",
            "true_x_m", "true_y_m", "true_label"]
HUMAN = {"knock_tap", "voice"}
DET_COLS = ["sound_id", "ch", "t_s", "sample_index", "peak", "snr", "clipped"]



# --------------------------------------------------------------------------- labels (placeholder)
def label_rules_v0(sound_rows: pd.DataFrame, fs: int, p: D.DetectParams):
    """PLACEHOLDER until Prompt 3. Uses only how long the sound stays above threshold
    and whether it clipped. Returns (label, confidence)."""
    dur = float(((sound_rows.end - sound_rows.trigger) / fs).median()) - p.release_s
    clipped = bool(sound_rows.clipped.any())
    if dur < 0.12 and not clipped:
        return "knock_tap", 0.6
    if clipped or dur < 0.3:
        return "impact", 0.5
    if dur < 1.5:
        return "voice", 0.5
    return "other", 0.4


def classify_sound(samples, fs, g, p):
    """Trained classifier (models/classifier.joblib) on 1 s from the loudest channel; else rules v0."""
    import classify as K
    if not K.MODEL_PATH.exists():
        return label_rules_v0(g, fs, p)
    best = g.loc[g.snr.idxmax()]
    i0 = int(round(max(best.t_s - 0.25, 0) * fs))
    audio = samples[i0:i0 + fs, int(best.ch)].astype(float)
    audio -= np.median(samples[:, int(best.ch)])
    return K.classify_audio(audio, fs)


def flag_repeated(fixes: pd.DataFrame, t_event: pd.Series, min_n=3, gap=(0.3, 2.5), max_cv=0.25):
    """Mark runs of >= min_n knocks with similar spacing (a person knocking on purpose)."""
    rep = pd.Series(False, index=fixes.index)
    k = fixes.index[fixes.label == "knock_tap"].tolist()
    t = t_event
    i = 0
    while i < len(k):
        j = i
        while j + 1 < len(k) and gap[0] <= t[k[j + 1]] - t[k[j]] <= gap[1]:
            j += 1
        run = k[i:j + 1]
        if len(run) >= min_n:
            d = np.diff([t[x] for x in run])
            if d.std() / d.mean() <= max_cv:
                rep[run] = True
        i = j + 1
    return rep


def timing_impossible(xy, t, c, sigma_t, k=4.0):
    """A time gap larger than the sensor spacing allows (plus k sigma) cannot come from
    a real sound location. Returns a short reason, or '' if timing is physically possible."""
    chs = sorted(t)
    for i, a in enumerate(chs):
        for b in chs[i + 1:]:
            limit = np.linalg.norm(xy[a] - xy[b]) / c + k * np.sqrt(2) * sigma_t
            if abs(t[a] - t[b]) > limit:
                return f"ch{a}-ch{b} gap {abs(t[a] - t[b]) * 1e3:.2f} ms > {limit * 1e3:.2f} ms possible"
    return ""


# --------------------------------------------------------------------------- main
def find_spot(name):
    spots = yaml.safe_load((ROOT / "config/cal_spots.yaml").read_text())
    for group in spots.values():
        if name in group:
            return float(group[name]["x"]), float(group[name]["y"])
    raise SystemExit(f"spot {name!r} not in config/cal_spots.yaml")


def load_delays(path, n_ch):
    p = ROOT / path
    if not p.exists():
        return np.zeros(n_ch), False
    d = np.asarray(yaml.safe_load(p.read_text())["delays_s"], float)[:n_ch]
    return np.pad(d, (0, n_ch - len(d))), True


def load_params(path):
    if not path:
        return D.DetectParams()
    raw = yaml.safe_load((ROOT / path).read_text())
    if "band_hz" in raw:
        raw["band_hz"] = tuple(raw["band_hz"])
    return D.DetectParams(**raw)


def locate_sounds(samples, fs, cfg, p, delays, sigma_t, start, truth_xy=None, true_label=None, single_radius_m=3.0):
    """Detect -> locate -> label on one block of samples. Times are seconds from the block start;
    wall_time = start + those seconds. Returns (det, rows, t_event, skipped).
    One sensor: no direction is possible, so the fix sits ON the node with a circle of
    single_radius_m (a guessed hearing range, said so in run.yaml and in ATAK)."""
    c, xy, mode = cfg["_c_mps"], cfg["_xy"], cfg["_mode"]
    n_ch = samples.shape[1]
    det, _, _ = D.detect_recording(samples, fs, p, xy, c)
    rows, t_event, skipped = [], {}, []
    if not len(det):
        return det, rows, t_event, skipped
    for sid, g in det.groupby("sound_id"):
        g = g.drop_duplicates("ch").sort_values("ch")
        need = 1 if n_ch == 1 else 2 if mode == "direction" else 3
        if len(g) < need:
            skipped.append((int(sid), len(g)))
            continue
        t = {int(r.ch): r.t_s - delays[int(r.ch)] for r in g.itertuples()}
        label, conf = classify_sound(samples, fs, g, p)
        row = dict(sound_id=int(sid), wall_time=(start + dt.timedelta(seconds=max(t.values()) + 0.3)).isoformat(),
                   label=label, confidence=conf, human_likely=label in HUMAN,
                   true_x_m=truth_xy[0] if truth_xy else np.nan, true_y_m=truth_xy[1] if truth_xy else np.nan,
                   true_label=true_label or (("knock_tap" if truth_xy else "")) or np.nan)
        if n_ch == 1:
            v = single_radius_m ** 2 / CHI2_95_2D
            row.update(kind="position", x_m=float(xy[0, 0]), y_m=float(xy[0, 1]), cov_xx=v, cov_xy=0.0, cov_yy=v,
                       residual_m=0.0)
        elif mode == "direction":
            b = solve_bearing(xy[0], xy[1], t[0], t[1], c, sigma_t, front_vector(xy[0], xy[1], cfg["bearing"]["front"]))
            row.update(kind="bearing", bearing_deg=round(local_angle_to_compass(b["theta_deg"],
                                                                                 cfg["geo"]["x_axis_bearing_deg"]), 2),
                       bearing_sigma_deg=round(b["sigma_deg"], 2))
        else:
            chs = sorted(t)
            bad = timing_impossible(xy, t, c, sigma_t)
            if bad:
                skipped.append((int(sid), f"inconsistent timing ({bad}); typical of soft voice onsets"))
                continue
            s = solve_position(xy[chs], np.array([t[k] for k in chs]), c, sigma_t)
            span = np.ptp(xy, axis=0).max()
            if np.hypot(s["x"] - xy[:, 0].mean(), s["y"] - xy[:, 1].mean()) > 20 * span:
                skipped.append((int(sid), "solver diverged (far outside the array)"))
                continue
            row.update(kind="position", x_m=round(s["x"], 4), y_m=round(s["y"], 4), cov_xx=s["cov"][0, 0],
                       cov_xy=s["cov"][0, 1], cov_yy=s["cov"][1, 1], residual_m=round(s["residual_m"], 4))
        rows.append(row)
        t_event[int(sid)] = float(g.t_s.min())
    return det, rows, t_event, skipped


def finish_fixes(rows, t_event):
    """Rows -> fixes DataFrame in contract order, with the repeated-knock flag."""
    fixes = pd.DataFrame(rows).reindex(columns=FIX_COLS)
    if len(fixes):
        fixes["repeated"] = flag_repeated(fixes.set_index("sound_id"),
                                          pd.Series(t_event)).reindex(fixes.sound_id).to_numpy()
    return fixes


def run_pipeline(recording, run, array="config/array.yaml", params=None, delays_file="config/channel_delays.yaml",
                 true_spot=None, true_label=None, start_time=None, sigma_t_s=None, extra_meta=None):
    cfg = load_array(array)
    rec = np.load(ROOT / recording if not Path(recording).is_absolute() else recording)
    samples, fs = rec["samples"], int(rec["fs_hz"])
    samples = samples[:, None] if samples.ndim == 1 else samples
    n_ch = samples.shape[1]
    if n_ch != len(cfg["_xy"]):
        raise SystemExit(f"recording has {n_ch} channels but {array} lists {len(cfg['_xy'])} sensors")
    p = params or D.DetectParams()
    sigma_t = float(sigma_t_s or cfg["timing"]["sigma_s"])
    c, mode = cfg["_c_mps"], cfg["_mode"]
    delays, calibrated = load_delays(delays_file, n_ch)
    start = (dt.datetime.fromisoformat(start_time) if start_time
             else dt.datetime.now(dt.timezone.utc).replace(microsecond=0))
    truth_xy = find_spot(true_spot) if true_spot else None

    det, rows, t_event, skipped = locate_sounds(samples, fs, cfg, p, delays, sigma_t, start, truth_xy, true_label)
    fixes = finish_fixes(rows, t_event)

    out = ROOT / "data/runs" / run
    (out / "snippets").mkdir(parents=True, exist_ok=True)
    (D.to_contract(det) if len(det) else pd.DataFrame(columns=DET_COLS)).to_csv(out / "detections.csv", index=False)
    fixes.to_csv(out / "fixes.csv", index=False)
    for sid, g in (det.groupby("sound_id") if len(det) else []):
        i0 = int(round(max(g.t_s.min() - 0.25, 0) * fs))
        np.savez_compressed(out / "snippets" / f"{sid}.npz", samples=samples[i0:i0 + fs], fs_hz=fs,
                            sound_id=int(sid), t0_s=i0 / fs)
    meta = dict(run=run, source="pipeline", recording=str(recording), array=str(array), mode=mode,
                medium=cfg["medium"], c_mps=round(c, 2), fs_hz=fs, sigma_t_s=sigma_t,
                start_time=start.isoformat(), duration_s=round(samples.shape[0] / fs, 3),
                detect_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in D.asdict(p).items()},
                channel_delays_s=[float(v) for v in delays], calibrated=calibrated,
                labeler=("trained classifier (models/classifier.joblib)" if (ROOT / "models/classifier.joblib").exists()
                         else "rules-v0 placeholder (duration + clipping)"),
                true_spot=true_spot, skipped_sounds=[dict(sound_id=s, reason=(f"heard on {n} sensor(s)" if isinstance(n, int) else n))
                                                     for s, n in skipped])
    if n_ch == 1:
        meta["single_sensor"] = "one mic: no direction; fixes sit on the node with a guessed 3 m hearing radius"
    meta.update(extra_meta or {})
    (out / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return det, fixes, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="npz with samples [n, n_ch] and fs_hz")
    ap.add_argument("--run", required=True, help="output folder name under data/runs/")
    ap.add_argument("--array", default="config/array.yaml")
    ap.add_argument("--params", help="detect params yaml (the tuning notebook writes detect_params.yaml)")
    ap.add_argument("--delays", default="config/channel_delays.yaml", help="per-sensor delays from calibration")
    ap.add_argument("--true-spot", help="name in config/cal_spots.yaml, if every knock came from one taped spot")
    ap.add_argument("--true-label", help="label of every sound in a test recording, e.g. knock_tap")
    ap.add_argument("--start-time", help="ISO time the recording started (default: now)")
    args = ap.parse_args()
    det, fixes, meta = run_pipeline(args.recording, args.run, args.array, load_params(args.params), args.delays,
                                    args.true_spot, args.true_label, args.start_time)
    print(f"{meta['mode']} mode, {meta['duration_s']:.1f} s at {meta['fs_hz']} Hz, "
          f"delays {'calibrated' if meta['calibrated'] else 'NOT calibrated (zeros)'}")
    print(f"{det.sound_id.nunique() if len(det) else 0} sounds detected, {len(fixes)} located, "
          f"{len(meta['skipped_sounds'])} skipped")
    for sk in meta["skipped_sounds"]:
        print(f"  skipped sound {sk['sound_id']}: {sk['reason']}")
    if len(fixes):
        print(fixes.groupby("label").size().to_string())
        print(f"repeated-pattern knocks: {int(fixes.repeated.sum())}")
    print(f"-> data/runs/{args.run}/   view: python dashboard.py --source replay --run {args.run}")
