"""Evaluate rules and a learned model on emulated public data; train the model; try it on real recordings.

  python -m dnl_classify.evaluate --features data/features_public.csv --real data/quiet_spot6.npz data/quiet_spot7.npz

Writes data/classify_eval/{confusion_*.png,dur_vs_mod.png,report.md} and models/dnl_rf.joblib.
Split = by fold (ESC-50 folds; hashed speaker/source for the others), never by clip, so renders of
the same clip never sit in train and test.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .features import FEATURES, extract
from .rules import PERSON, classify_rules

HERE = Path(__file__).parent
LABELS = ["knock_tap", "voice", "cough", "breath", "impact", "steady", "alarm", "other"]
RULE_LABELS = ["knock_tap", "voice", "cough", "impact", "steady", "other"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([r["label"] for r in rows])
    return rows, X, y


def confusion(y, p, labels_true, labels_pred):
    M = np.zeros((len(labels_true), len(labels_pred)), int)
    for a, b in zip(y, p):
        if a in labels_true and b in labels_pred:
            M[labels_true.index(a), labels_pred.index(b)] += 1
    return M


def plot_cm(M, lt, lp, title, out):
    R = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(lp), 1 + 0.7 * len(lt)))
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(lt)):
        for j in range(len(lp)):
            ax.text(j, i, f"{R[i, j]:.2f}\n({M[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if R[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(lp)), lp, rotation=45, ha="right")
    ax.set_yticks(range(len(lt)), lt)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def person_metrics(y, p):
    t = np.array([a in PERSON for a in y])
    q = np.array([b in PERSON for b in p])
    recall = (t & q).sum() / max(t.sum(), 1)            # people we flag
    false_alarm = (~t & q).sum() / max((~t).sum(), 1)   # scene sounds we wrongly flag
    return recall, false_alarm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_public.csv")
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--out", default="data/classify_eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = yaml.safe_load(open(HERE / "classify.yaml"))
    rows, X, y = load(a.features)
    folds = np.array([int(r["fold"]) for r in rows])
    rep = [f"# Classification on emulated public data\n\n{len(rows)} sounds from "
           f"{len({r['file'] for r in rows})} clips. Per class: "
           + ", ".join(f"{c} {int((y == c).sum())}" for c in LABELS) + "\n"]

    # ---- rules (no training: evaluate on everything)
    feats = [dict(zip(FEATURES, x), snr=float(r["snr"])) for x, r in zip(X, rows)]
    for key in ("rise_ms", "rise1_ms"):
        p = np.array([classify_rules(f, cfg["rules"], key)[0] for f in feats])
        M = confusion(y, p, LABELS, RULE_LABELS)
        plot_cm(M, LABELS, RULE_LABELS, f"rules ({key})", f"{a.out}/confusion_rules_{key}.png")
        rec, fa = person_metrics(y, p)
        kn = (p[y == "knock_tap"] == "knock_tap").mean()
        rep.append(f"## Rules using {key}\nperson recall {rec:.2f}, false alarm on scene sounds {fa:.2f}, "
                   f"knock recall {kn:.2f}\n")

    # ---- random forest, leave-one-fold-out
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    def rf():
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                      n_jobs=-1, random_state=0)
    pred = np.empty(len(y), dtype=object)
    proba_max = np.zeros(len(y))
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        m = rf().fit(X[tr], y[tr])
        P = m.predict_proba(X[te])
        pred[te] = m.classes_[P.argmax(1)]
        proba_max[te] = P.max(1)
    acc = (pred == y).mean()
    M = confusion(y, pred, LABELS, LABELS)
    plot_cm(M, LABELS, LABELS, f"random forest, by-fold CV, acc {acc:.2f}", f"{a.out}/confusion_rf.png")
    rec, fa = person_metrics(y, pred)
    rep.append(f"## Random forest (leave-one-fold-out)\naccuracy {acc:.2f}; person recall {rec:.2f}, "
               f"false alarm {fa:.2f}\n\n| class | recall | precision | n |\n|---|---|---|---|")
    for i, c in enumerate(LABELS):
        rc = M[i, i] / max(M[i].sum(), 1)
        pr = M[i, i] / max(M[:, i].sum(), 1)
        rep.append(f"| {c} | {rc:.2f} | {pr:.2f} | {M[i].sum()} |")
    for prof in ("hi", "lo"):
        s = np.array([r["profile"] == prof for r in rows])
        rep.append(f"\nsensor profile {prof}: accuracy {(pred[s] == y[s]).mean():.2f}")
    for lo, hi in ((0, 10), (10, 20), (20, 40)):
        s = np.array([lo <= float(r["sim_snr_db"]) < hi for r in rows])
        rep.append(f"simulated SNR {lo}-{hi} dB: accuracy {(pred[s] == y[s]).mean():.2f}, "
                   f"person recall/false alarm {'/'.join(f'{v:.2f}' for v in person_metrics(y[s], pred[s]))}")

    final = rf().fit(X, y)
    imp = sorted(zip(final.feature_importances_, FEATURES), reverse=True)
    rep.append("\nFeature importance: " + ", ".join(f"{f} {v:.2f}" for v, f in imp))
    os.makedirs(os.path.dirname(cfg["model"]["path"]), exist_ok=True)
    joblib.dump(dict(model=final, features=FEATURES), cfg["model"]["path"])
    rep.append(f"\nModel saved to {cfg['model']['path']}\n")

    # ---- duration vs syllable rate (handbook 11.3 step 5)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in LABELS:
        s = y == c
        ax.scatter(X[s, FEATURES.index("dur_ms")], X[s, FEATURES.index("mod_hz")] + np.random.uniform(-.15, .15, s.sum()),
                   s=6, alpha=0.4, label=c)
    ax.set_xscale("log")
    ax.set_xlabel("duration (ms)")
    ax.set_ylabel("loudness rhythm mod_hz (jittered)")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("Emulated Grove sensor: duration vs syllable rate by true class")
    fig.tight_layout()
    fig.savefig(f"{a.out}/dur_vs_mod.png", dpi=110)

    # ---- real recordings (unlabelled knock sessions: we expect knock_tap on clean nodes)
    det = cfg["detection"]
    if a.real:
        rep.append("## Real recordings (knock sessions; labels are what we expect, not ground truth)\n"
                   "| file | node | sounds | rules(rise1) | model |\n|---|---|---|---|---|")
    for path in a.real:
        rec = np.load(path)
        for n in [int(v) for v in rec["nodes"]]:
            evs = extract(rec[f"n{n}_adc"].astype(float), float(rec[f"n{n}_fs"]), **det)
            if not evs:
                continue
            F = np.array([[e[f] for f in FEATURES] for e in evs])
            pm = final.predict(F)
            pr = [classify_rules(e, cfg["rules"], "rise1_ms")[0] for e in evs]

            def hist(v):
                u, c = np.unique(v, return_counts=True)
                return ", ".join(f"{a} {b}" for a, b in sorted(zip(u, c), key=lambda t: -t[1]))
            rep.append(f"| {Path(path).stem} | {n} | {len(evs)} | {hist(pr)} | {hist(pm)} |")
    open(f"{a.out}/report.md", "w").write("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
