"""Scripted session: accuracy test + labelled data + demo material in ONE recording.

1) Work out coordinates from tape measurements (distances only, no grid needed):
     python session.py frame --d12 2.5 --d13 2.3 --d23 2.1 --spot A=1.2:1.5 B=1.9:1.1 C=1.4:2.0
   Node 1 is the origin, node 2 sits on the +x axis, node 3 and the spots are on its side.
   Spot A=1.2:1.5 means 1.2 m from node 1 and 1.5 m from node 2. It prints --pos and --spots to paste.

2) Rehearse the prompts without hardware (5x speed):
     python session.py run --rehearse --spots A=1.0,0.8 B=1.8,1.2 C=0.6,1.5

3) The real session (~2 min). Prompts tell you what to do and when:
     python session.py run --ports COM11 COM12 COM13 --layout data/layout.json ^
         --spots A=1.0,0.8 B=1.8,1.2 C=0.6,1.5 --out data/session1.npz

4) Re-run the analysis any time (labels and positions are stored next to the recording).
   Wrong coordinates? Re-analyse with the right ones, no need to re-record:
     python session.py report data/session1.npz --layout data/layout.json --spots A=x,y B=x,y C=x,y

Outputs next to the recording: <name>.labels.json (what was done when), <name>.results.csv (every
detected sound: block label, per-node onset/peak/SNR, position, error in cm), <name>.accuracy.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

import record_plot as rp

PREP_S = 4.0      # get-ready time before each block
ACTIVE_S = 12.0   # how long each block lasts
TOL_S = 0.8       # a detection this close outside a block still counts as inside it
QUIET_S = 5.0


# ------------------------------------------------------------------ schedule
def build_schedule(spots: dict):
    names = list(spots)
    blocks = [dict(kind="quiet", label="quiet", spot=None, truth=None, prep_s=0.0, start_s=0.0,
                   end_s=QUIET_S, prompt="STAY QUIET")]
    t = QUIET_S

    def add(kind, spot, text):
        nonlocal t
        start = t + PREP_S
        end = start + ACTIVE_S
        blocks.append(dict(kind=kind, label=f"{kind}_{spot}", spot=spot, truth=list(spots[spot]),
                           prep_s=t, start_s=start, end_s=end, prompt=text))
        t = end

    for s in names:
        add("knock", s, f"spot {s}: knock 5 times with something hard (a pan or a box), about 2 s apart")
    first, second = names[0], names[min(1, len(names) - 1)]
    add("voice", first, f"spot {first}: say \"help, I'm here\" 3 times, pausing between")
    add("cough", first, f"spot {first}: cough 3 times, pausing between")
    add("impact", second, f"spot {second}: drop a book flat on the floor 3 times (a slam, not a knock)")
    return blocks, t + 3.0


def run_prompts(blocks, t0, speed=1.0):
    def now():
        return (time.time() - t0) * speed

    print("\n>>> STAY QUIET for 5 s")
    events = []
    for b in blocks[1:]:
        events += [(b["prep_s"], "prep", b), (b["start_s"], "go", b), (b["end_s"], "stop", b)]
    for at, what, b in events:
        while now() < at:
            time.sleep(0.02)
        if what == "prep":
            print(f"\n>>> GET READY (4 s): {b['prompt']}")
        elif what == "go":
            print("\a>>> GO!")
        else:
            print(">>> stop")


# ------------------------------------------------------------------ helpers
def parse_spots(items):
    spots = {}
    for it in items or []:
        name, xy = it.split("=")
        x, y = xy.split(",")
        spots[name] = (float(x), float(y))
    return spots


def valid_fix(sol, pos, margin=2.0):
    """Reject divergent or poorly conditioned fixes (positions far outside the nodes, big residual)."""
    if not sol:
        return False
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if not (min(xs) - margin <= sol["x"] <= max(xs) + margin and min(ys) - margin <= sol["y"] <= max(ys) + margin):
        return False
    return sol["resid_m"] <= 0.05 and math.isfinite(sol["radius95"]) and sol["radius95"] <= 1.5


def block_for(rel, blocks):
    """Which block a detection at rel seconds belongs to. People start knocking during GET READY,
    so a block runs from 1.5 s after its prompt until 0.5 s after the next prompt."""
    if rel <= blocks[0]["end_s"]:
        return blocks[0]
    for i, b in enumerate(blocks[1:], 1):
        hi = (blocks[i + 1]["prep_s"] if i + 1 < len(blocks) else b["end_s"]) + 0.5
        if b["prep_s"] + 1.5 <= rel <= hi:
            return b
    return None


def cmd_frame(a):
    d12 = a.d12
    x3 = (d12 ** 2 + a.d13 ** 2 - a.d23 ** 2) / (2 * d12)
    y3 = math.sqrt(max(a.d13 ** 2 - x3 ** 2, 0.0))
    print(f"--pos 1=0,0 2={d12:g},0 3={x3:.2f},{y3:.2f}")
    out = []
    for it in a.spot or []:
        name, d = it.split("=")
        d1, d2 = (float(v) for v in d.split(":"))
        x = (d12 ** 2 + d1 ** 2 - d2 ** 2) / (2 * d12)
        y = math.sqrt(max(d1 ** 2 - x ** 2, 0.0))
        out.append(f"{name}={x:.2f},{y:.2f}")
    if out:
        print("--spots " + " ".join(out))


# ------------------------------------------------------------------ run
def cmd_run(a):
    spots = parse_spots(a.spots)
    if len(spots) < 1:
        raise SystemExit("Give at least one spot, e.g. --spots A=1.0,0.8 B=1.8,1.2 C=0.6,1.5")
    blocks, total = build_schedule(spots)
    if a.rehearse:
        print(f"Rehearsal at 5x speed ({total / 5:.0f} s). No recording.")
        run_prompts(blocks, time.time(), speed=5.0)
        return
    if not a.ports:
        raise SystemExit("Need --ports.")
    pos = rp.load_positions(a.pos, a.layout)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    print(f"Session takes about {total:.0f} s. Nodes must stay where they are for the whole session.")
    input("Stand at the first spot with your knocking object, then press Enter to start... ")
    th = threading.Thread(target=rp.record, args=(a.ports, total, str(out), a.settle), daemon=True)
    th.start()
    time.sleep(a.settle)
    t0 = time.time()
    run_prompts(blocks, t0)
    th.join(30)
    if not out.exists():
        raise SystemExit("No recording was saved. Read the messages above (ports, sketch, sync).")
    lab = out.with_suffix(".labels.json")
    lab.write_text(json.dumps(dict(pos={str(k): list(v) for k, v in pos.items()}, temp=a.temp,
                                   blocks=blocks, total_s=total), indent=1))
    print(f"\nlabels saved: {lab}")
    report(out, a.k, show=not a.no_show)


# ------------------------------------------------------------------ report
def report(npz: Path, k: float = 5.0, show: bool = True, pos_override=None, spots_override=None):
    lab = json.loads(npz.with_suffix(".labels.json").read_text())
    pos = pos_override or {int(n): tuple(v) for n, v in lab["pos"].items()}
    blocks = lab["blocks"]
    if spots_override:
        for b in blocks:
            if b["kind"] == "knock" and b["spot"] in spots_override:
                b["truth"] = list(spots_override[b["spot"]])
    data, results, c, t_start = rp.analyse(str(npz), pos, lab.get("temp", 20.0), k, "level", 6.0, True)
    nodes = sorted(data)

    rows = []
    for gi, r in enumerate(results):
        g, sol = r["group"], r["sol"]
        rel = (min(e["onset_us"] for e in g.values()) - t_start) / 1e6
        blk = block_for(rel, blocks)
        ok = valid_fix(sol, pos)
        row = dict(group=gi, t_s=round(rel, 3), block=blk["label"] if blk else "unlabelled",
                   kind=blk["kind"] if blk else "none", spot=blk["spot"] if blk else "",
                   n_nodes=len(g), fix="ok" if ok else ("rejected" if sol else ""),
                   x_m="", y_m="", radius_m="", err_cm="")
        for n in nodes:
            e = g.get(n)
            row[f"n{n}_onset_ms"] = round((e["onset_us"] - t_start) / 1000, 3) if e else ""
            row[f"n{n}_peak"] = round(e["peak"], 1) if e else ""
            row[f"n{n}_snr"] = round(e["snr"], 2) if e else ""
            row[f"n{n}_guided"] = int(bool(e.get("guided"))) if e else ""
        row["n_guided"] = sum(1 for e in g.values() if e.get("guided"))
        if ok:
            row.update(x_m=round(sol["x"], 3), y_m=round(sol["y"], 3), radius_m=round(sol["radius95"], 3))
            if blk and blk.get("truth"):
                row["err_cm"] = round(math.hypot(sol["x"] - blk["truth"][0], sol["y"] - blk["truth"][1]) * 100, 1)
        rows.append(row)

    csv_path = npz.with_suffix(".results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["group"])
        w.writeheader()
        w.writerows(rows)

    def stats(rs, truth):
        xs = np.array([x["x_m"] for x in rs], float)
        ys = np.array([x["y_m"] for x in rs], float)
        errs = np.hypot(xs - truth[0], ys - truth[1]) * 100
        mx, my = np.median(xs), np.median(ys)
        return errs, math.hypot(mx - truth[0], my - truth[1]) * 100, float(np.median(np.hypot(xs - mx, ys - my)) * 100)

    n3 = sum(1 for x in rows if x["n_nodes"] >= 3)
    bad = sum(1 for x in rows if x["fix"] == "rejected")
    print(f"\n3-node fixes: {n3}, rejected as nonsense (far outside the nodes / inconsistent timing): {bad}")
    print("\n================ ACCURACY (valid 3-node fixes vs tape marks) ================")
    print(f"{'spot':<5}{'truth (m)':<13}{'sounds':>7}{'fixes':>7}{'median err':>12}{'<50cm':>7}{'<100cm':>8}{'bias':>7}{'spread':>8}")
    all_err = []
    for b in blocks:
        if b["kind"] != "knock":
            continue
        rs = [x for x in rows if x["block"] == b["label"]]
        loc = [x for x in rs if x["fix"] == "ok"]
        tx, ty = b["truth"]
        if loc:
            errs, bias, spread = stats(loc, (tx, ty))
            all_err += errs.tolist()
            print(f"{b['spot']:<5}({tx:.2f},{ty:.2f}){'':<2}{len(rs):>7}{len(loc):>7}{np.median(errs):>9.0f} cm"
                  f"{100 * np.mean(errs < 50):>6.0f}%{100 * np.mean(errs < 100):>7.0f}%{bias:>5.0f}cm{spread:>6.0f}cm")
        else:
            print(f"{b['spot']:<5}({tx:.2f},{ty:.2f}){'':<2}{len(rs):>7}{0:>7}{'-':>12}")
    if all_err:
        e = np.array(all_err)
        print(f"\nall valid fixes ({len(e)}): median error {np.median(e):.0f} cm, "
              f"{100 * np.mean(e < 50):.0f}% within 50 cm, {100 * np.mean(e < 100):.0f}% within 100 cm")
        own = [x for x in rows if x["fix"] == "ok" and x["kind"] == "knock" and x["n_guided"] == 0]
        if own:
            eo = np.array([x["err_cm"] for x in own], float)
            print(f"fixes where all 3 nodes detected the knock on their own ({len(eo)}): median error {np.median(eo):.0f} cm")
    print("bias = distance from the MEDIAN of the fixes to the tape mark (systematic error);")
    print("spread = typical distance of the fixes from their own median (repeatability).")

    quiet = [x for x in rows if x["kind"] == "quiet"]
    print(f"\nfalse detections during the quiet 5 s: {len(quiet)}")
    for kind in ("voice", "cough", "impact"):
        rs = [x for x in rows if x["kind"] == kind]
        print(f"{kind:<7} block: {len(rs)} detections, {sum(1 for x in rs if x['n_nodes'] >= 2)} heard by 2+ nodes")
    print(f"\nsaved: {csv_path}")

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    for n, (x, y) in pos.items():
        ax.plot(x, y, "s", ms=13, color="0.25")
        ax.annotate(f"node {n}", (x, y), textcoords="offset points", xytext=(0, -18), ha="center")
    cmap = plt.get_cmap("tab10")
    for i, b in enumerate([b for b in blocks if b["kind"] == "knock"]):
        col = cmap(i)
        tx, ty = b["truth"]
        ax.plot(tx, ty, "o", ms=16, mfc="none", mec=col, mew=2)
        ax.annotate(f"spot {b['spot']}", (tx, ty), textcoords="offset points", xytext=(10, 10), color=col)
        for x in rows:
            if x["block"] == b["label"] and x["fix"] == "ok":
                ax.plot([tx, x["x_m"]], [ty, x["y_m"]], "-", color=col, alpha=0.3, lw=1)
                ax.plot(x["x_m"], x["y_m"], "*", ms=12, color=col)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Triangulated knocks (stars) vs tape marks (circles); nonsense fixes removed")
    fig.tight_layout()
    png = npz.with_suffix(".accuracy.png")
    fig.savefig(png, dpi=120)
    print(f"saved: {png}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frame")
    f.add_argument("--d12", type=float, required=True)
    f.add_argument("--d13", type=float, required=True)
    f.add_argument("--d23", type=float, required=True)
    f.add_argument("--spot", nargs="*", help="NAME=d_from_node1:d_from_node2")
    r = sub.add_parser("run")
    r.add_argument("--ports", nargs="+")
    r.add_argument("--pos", nargs="+", help="node positions (no default)")
    r.add_argument("--layout", help="layout.json from live.py calibration, instead of --pos")
    r.add_argument("--spots", nargs="+", required=True)
    r.add_argument("--out", default="data/session.npz")
    r.add_argument("--settle", type=float, default=3.0)
    r.add_argument("--temp", type=float, default=20.0)
    r.add_argument("--k", type=float, default=5.0)
    r.add_argument("--rehearse", action="store_true")
    r.add_argument("--no-show", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("path")
    p.add_argument("--pos", nargs="+", help="override node positions")
    p.add_argument("--layout", help="override node positions from a layout.json")
    p.add_argument("--spots", nargs="+", help="override tape-mark positions, e.g. A=1.0,0.8 B=1.8,1.2")
    p.add_argument("--k", type=float, default=5.0)
    p.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    if a.cmd == "frame":
        cmd_frame(a)
    elif a.cmd == "run":
        cmd_run(a)
    else:
        report(Path(a.path), a.k, show=not a.no_show,
               pos_override=rp.load_positions(a.pos, a.layout) if (a.pos or a.layout) else None,
               spots_override=parse_spots(a.spots) if a.spots else None)


if __name__ == "__main__":
    main()
