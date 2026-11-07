"""Sound classifier: rules baseline vs trained model, evaluated by session.

Labels: knock_tap, voice, impact, other.  Clips: data/clips/<label>/<session>_<n>.npz.
Evaluation is leave-one-session-out: every clip is predicted by a model that never
saw its session (not even augmented copies of it).

    python classify.py            # evaluate, train on all sessions, save models/classifier.joblib
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import style

ROOT = Path(__file__).resolve().parent
LABELS = ["knock_tap", "voice", "impact", "other"]   # human-likely first
HUMAN = {"knock_tap", "voice"}
MODEL_PATH = ROOT / "models/classifier.joblib"


# --------------------------------------------------------------------------- data
def load_clips(root="data/clips"):
    rows = []
    for p in sorted(glob.glob(str(ROOT / root / "*" / "*.npz"))):
        z = np.load(p)
        label = str(z["label"]) if "label" in z else Path(p).parent.name
        if label not in LABELS:
            continue
        session = str(z["session"]) if "session" in z else Path(p).stem.rsplit("_", 1)[0]
        rows.append(dict(path=p, label=label, session=session,
                         environment=str(z["environment"]) if "environment" in z else "",
                         audio=z["audio"].astype(float), fs=int(z["fs_hz"])))
    return pd.DataFrame(rows)


def background_bank(fs, rng, n=8):
    """Background noise for augmentation: data/background/*.npz (recorded) if present, else synthetic."""
    bank = []
    for p in sorted(glob.glob(str(ROOT / "data/background/*.npz"))):
        z = np.load(p)
        a = z["audio"] if "audio" in z else z["samples"][:, 0]
        a = a.astype(float) - np.mean(a)
        bank += [a[i:i + fs] for i in range(0, len(a) - fs, fs)]
    if not bank:   # real background: the 'other' clips cut from recordings (make_clips.py)
        for p in sorted(glob.glob(str(ROOT / "data/clips/other/*.npz")))[:40]:
            a = np.load(p)["audio"].astype(float)
            bank.append(a - a.mean())
    if not bank:   # last resort only: synthetic noise
        import sim
        bank = [sim.other(rng, fs, n=fs, kind=k) for k in ("pink", "fan", "rumble", "pink", "fan", "rustle")][:n]
    return [b / (np.std(b) + 1e-12) for b in bank]


def augment(audio, fs, rng, bank, snr_db):
    """Mix background at snr_db (peak-over-noise-RMS) and scale amplitude down (distance)."""
    x = audio / (np.abs(audio).max() + 1e-12)
    noise = bank[rng.integers(len(bank))]
    y = x * 10 ** (snr_db / 20) + noise[: len(x)]
    return 0.9 * y / np.abs(y).max() * rng.uniform(0.2, 1.0)


def featurize(clips, augment_snrs=(), seed=0):
    """Feature table; with augment_snrs, also adds noisy copies (same session = same split)."""
    rng = np.random.default_rng(seed)
    rows = []
    bank = background_bank(int(clips.fs.iloc[0]), rng) if augment_snrs else None
    for r in clips.itertuples():
        rows.append(dict(F.extract(r.audio, r.fs), label=r.label, session=r.session, augmented=False))
        for snr in augment_snrs:
            rows.append(dict(F.extract(augment(r.audio, r.fs, rng, bank, snr), r.fs), label=r.label,
                             session=r.session, augmented=True))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- models
def rules_predict(feat: pd.DataFrame):
    """Interpretable baseline. Thresholds read off the training feature ranges."""
    out = []
    for r in feat.itertuples():
        if r.pitch_strength > 0.6 and r.voiced_frac > 0.1 and r.band_low < 0.6:
            out.append("voice")
        elif r.crest_factor > 6.5 and r.spectral_flatness < 0.12:
            out.append("knock_tap")
        elif r.spectral_flatness > 0.12 and r.duration_s < 0.2:
            out.append("impact")
        elif r.crest_factor > 5.5 and r.duration_s < 0.2:
            out.append("knock_tap")
        else:
            out.append("other")
    return np.array(out)


def make_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0)


def loso_eval(feat_all: pd.DataFrame):
    """Leave-one-session-out. Train on other sessions (+ their augmented copies), test on raw clips
    of the held-out session. Returns predictions for every raw clip."""
    cols = F.feature_names()
    raw = feat_all[~feat_all.augmented].copy()
    raw["pred_rules"] = rules_predict(raw)
    raw["pred_model"] = ""
    raw["conf_model"] = np.nan
    for s in sorted(raw.session.unique()):
        tr = feat_all[feat_all.session != s]
        te = raw.session == s
        m = make_model().fit(tr[cols], tr.label)
        proba = m.predict_proba(raw.loc[te, cols])
        raw.loc[te, "pred_model"] = m.classes_[proba.argmax(1)]
        raw.loc[te, "conf_model"] = proba.max(1)
    return raw


def report(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    p, r, f1, n = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    table = pd.DataFrame(dict(precision=p, recall=r, f1=f1, clips=n), index=LABELS).round(3)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    person = np.isin(yt, list(HUMAN))
    responder = dict(
        knock_recall=float(np.mean(yp[yt == "knock_tap"] == "knock_tap")),
        voice_recall=float(np.mean(yp[yt == "voice"] == "voice")),
        person_missed=float(np.mean(~np.isin(yp[person], list(HUMAN)))),        # person called noise
        false_alarm=float(np.mean(np.isin(yp[~person], list(HUMAN)))),          # noise called a person
    )
    return cm, table, responder


def train_final(feat_all):
    import joblib
    cols = F.feature_names()
    m = make_model().fit(feat_all[cols], feat_all.label)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(dict(model=m, features=cols, labels=LABELS, sessions=sorted(feat_all.session.unique())), MODEL_PATH)
    return m


# --------------------------------------------------------------------------- use in the pipeline
_CACHE = {}


def classify_audio(audio, fs):
    """(label, confidence) for one ~1 s mono clip. Uses the trained model if saved, else rules."""
    if "m" not in _CACHE:
        if MODEL_PATH.exists():
            import joblib
            _CACHE["m"] = joblib.load(MODEL_PATH)
        else:
            _CACHE["m"] = None
    f = pd.DataFrame([F.extract(audio, fs)])
    bundle = _CACHE["m"]
    if bundle is None:
        return str(rules_predict(f)[0]), 0.5
    proba = bundle["model"].predict_proba(f[bundle["features"]])[0]
    k = int(proba.argmax())
    return str(bundle["model"].classes_[k]), float(proba[k])


# --------------------------------------------------------------------------- plots
def plot_confusion(cm, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if pct[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_gallery(clips, per_label=4, seed=0):
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(LABELS), per_label, figsize=(3.1 * per_label, 2.3 * len(LABELS)), sharex=True,
                             sharey=True)
    for i, lab in enumerate(LABELS):
        sub = clips[clips.label == lab]
        if not len(sub):
            for j in range(per_label):
                axes[i, j].set_title(f"{lab}: no clips yet", fontsize=8, color="#999")
            continue
        pick = sub.iloc[rng.choice(len(sub), min(per_label, len(sub)), replace=False)]
        for j, r in enumerate(pick.itertuples()):
            f, t, S = spectrogram(r.audio, r.fs, nperseg=256, noverlap=192)
            ax = axes[i, j]
            ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma",
                          vmin=np.percentile(10 * np.log10(S + 1e-12), 30))
            ax.set_title(f"{lab} · {r.session} · {r.environment}", fontsize=8,
                         color=style.LABEL_COLOR[lab] if lab != "other" else "#666")
            if j == 0:
                ax.set_ylabel("kHz")
            if i == len(LABELS) - 1:
                ax.set_xlabel("s")
    fig.suptitle("Knocks are short vertical stripes, voices are stacked horizontal bands, impacts are one broad "
                 "burst, other is diffuse", fontsize=11)
    fig.tight_layout()
    return fig


def plot_scatter(feat, x="duration_s", y="pitch_strength", title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for lab in LABELS[::-1]:
        s = feat[feat.label == lab]
        ax.scatter(s[x], s[y], s=24, alpha=0.75 if lab in HUMAN else 0.5, color=style.LABEL_COLOR[lab],
                   label=f"{lab} ({len(s)})", edgecolors="none")
    ax.set_xlabel(x.replace("_s", " (s)").replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.legend(fontsize=8)
    ax.grid(color="#eee")
    ax.set_title(title or "", fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    clips = load_clips()
    print(f"{len(clips)} clips, sessions {sorted(clips.session.unique())}")
    counts = clips.label.value_counts().reindex(LABELS, fill_value=0)
    print("clips per label:", counts.to_dict())
    thin = [k for k, v in counts.items() if v < 10]
    if thin:
        print(f"WARNING: fewer than 10 clips for {thin}. The model cannot learn these labels, and the scores "
              "below are not meaningful for them. Record more (see make_clips.py).")
    feat = featurize(clips, augment_snrs=(20, 10, 5))
    res = loso_eval(feat)
    for name in ("rules", "model"):
        cm, table, resp = report(res.label, res[f"pred_{name}"])
        print(f"\n== {name} (leave-one-session-out) ==\n{table}\n{resp}")
    train_final(feat)
    print(f"\nsaved {MODEL_PATH.relative_to(ROOT)}")
