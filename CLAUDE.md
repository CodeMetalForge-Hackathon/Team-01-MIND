# Drop-and-Listen — shared context (read this before every task)

Code Metal Forge Hackathon 2026. Cheap, throwable listening nodes are scattered
at a disaster site and keep listening after responders leave. A trapped
person's knock or voice should become a location on the rescue team's map
(ATAK marker + our dashboard) within seconds.

**Team.** Two people. Siddhi: sensors, ESP32-S2 firmware (`recording/s2_stream`), recording tools
(`recording/record.py`, `plot_recording.py`) and the M0 math package (`dropnlisten`, not yet pushed;
`tools/m0_sim_plot.py` imports it). Sakshi (+ Claude): the whole software side: detection, localization
pipeline, classifier, calibration, accuracy analysis, dashboard and the ATAK link. Everything must run
**without hardware** (sim data, or recorded runs replayed from disk).

**Boundary.** Nodes *detect* (onset time, peak, SNR, clipping). Our code
*classifies* (label, confidence, human_likely, repeated) and writes the fix.
Only likely-human sounds (and hazards) go to ATAK.

**One source of truth for the math.** `arraycfg.py` and `localize.py` here are
stand-ins written to the same equations as the M0 doc. When the M0 package is
in this repo, import `config.py`, `speed_of_sound.py`, `bearing.py`, `tdoa.py`
from it and delete the stand-ins. Never keep two solvers.

---

## Setup config — `config/array.yaml` (agreed M0 schema, do not change shape)

```yaml
medium: air                       # air | water
conditions: {temperature_c: 20, humidity_pct: 50}
sampling: {fs_hz: 10000}
sensors: [{ch: 0, name: DNL-S1, x_m: -0.25, y_m: 0.0}, {ch: 1, name: DNL-S2, x_m: 0.25, y_m: 0.0}]
bearing: {front: left}            # side of the A->B line sounds are assumed to come from
timing: {sigma_s: 0.0001}         # per-sensor timing error; Stage 2 replaces with measured value
geo: {lat0: 42.36030, lon0: -71.05880, x_axis_bearing_deg: 90}
```

- 2 sensors → direction mode; ≥3 → position mode (decided from the list).
- No code hard-codes a sensor position, sample rate, or speed of sound.
- Speed of sound: air `331.4 + 0.6 T + 0.0124 RH` (344.0 m/s at 20 °C/50 %);
  water: Medwin, salinity 0 for fresh water (≈1482 m/s at 20 °C).
- `config/array_triangle.yaml`: Stage 3 position layout, 2 m triangle.
  Positions are **assumed** until taped out and measured.

## Data contracts — FIXED. Do not drift.

| Path | Contents |
|---|---|
| `data/recordings/*.npz` | `samples` [n, n_ch] int16 ADC counts, `fs_hz` |
| `data/clips/<label>/<session>_<n>.npz` | 1.0 s audio: `audio`, `fs_hz`, `label`, `session`, `environment` |
| `data/runs/<run>/detections.csv` | `sound_id, ch, t_s, sample_index, peak, snr, clipped` — one row per channel per sound |
| `data/runs/<run>/fixes.csv` | `sound_id, kind, wall_time, bearing_deg, bearing_sigma_deg, x_m, y_m, cov_xx, cov_xy, cov_yy, residual_m, label, confidence, human_likely, repeated, true_x_m, true_y_m, true_label` |
| `data/runs/<run>/snippets/*.npz` | 1.0 s of raw samples around each sound |
| `data/runs/<run>/run.yaml` | run metadata: array file, mode, medium, c_mps, fs_hz, sigma_t_s, start time |
| `config/cal_spots.yaml` | known (x, y) of taped spots, grouped `direction:` / `triangle:` |

### Interpretations (keep consistent; confirm against M0 `datatypes.py`)

- **ADC counts:** ESP32-S2 ADC1, 12-bit, 0–4095, quiet level ≈ 2048 (Grove
  sensor at mid-rail on 3.3 V). Stored int16. A sample at 0 or 4095 is clipped.
- **Clock:** `t_s` = seconds since recording start on a clock shared by all
  sensors; **sub-sample precision expected** (onset interpolated between
  samples, decision 2 in the M0 doc). `sample_index = round(t_s * fs_hz)`.
- **`ch`:** matches `ch` in the array file.
- **`peak`:** max |sample − 2048| in counts in the event window.
  **`snr`:** dB, `20·log10(peak / noise_rms)`, noise from just before onset.
- **`clipped` (detections):** ADC saturation. **Not** the same as the bearing
  solver's endfire clip (time gap larger than the spacing allows). The endfire
  flag currently has no column in fixes.csv — open question, see below.
- **`kind`:** `position` (x, y, cov, residual filled) or `bearing`
  (bearing_deg, bearing_sigma_deg filled).
- **Bearing:** `bearing_deg` is a **compass bearing, clockwise from north**
  (glossary definition; what ATAK expects), from the pair midpoint, resolved to
  `bearing.front`. Local map angle = `x_axis_bearing_deg − bearing_deg`.
  σ: `c·√2·σ_t / (d·sin φ)`. Dashboard label: "assumes sounds from the front side."
- **Covariance:** m², `σ_t²·c²·(JᵀJ)⁻¹`. 95 % ellipse = 1-σ axes × √5.991.
  Its long radius is the ATAK uncertainty circle.
- **`residual_m`:** RMS range residual (m); always 0 with exactly 3 sensors.
- **`wall_time`:** ISO-8601 UTC. **Booleans:** `True`/`False`.
  **`true_*`:** empty for field runs; filled in tests and sims.
- **Snippets:** `snippets/<sound_id>.npz` with `samples`, `fs_hz`, `t0_s`.
- **Clips:** mono float32 [−1, 1] at the array's fs (10 kHz now). Session ids
  must not contain `_`. Split train/test **by session**.

### Open questions for the teammate

1. Do `datatypes.py` Detection / Fix fields match the CSV columns above? If
   not, the CSVs follow datatypes.py and this file gets updated.
2. Add an `endfire_clipped` column to fixes.csv? (Needed for the dashboard
   warning; today it lives only in run.yaml for sims.)
3. Is the clock shared across sensors in practice (same ESP32-S2 ADC = yes;
   separate nodes over ESP-NOW = needs sync)?

---

## Labels

Start with **four only:** `knock_tap`, `voice`, `impact`, `other`.
Later, only if asked: `structure_creak`, `nature`, `machinery`.
`human_likely` = label in {knock_tap, voice}.

## Style rules — every plot

Colors come from `style.py` (Okabe-Ito); never hard-code a hex elsewhere.
Sensors dark gray squares labeled DNL-S1…; knock_tap blue `#0072B2`, voice
green `#009E73`, impact orange `#E69F00` faded, machinery purple `#CC79A7`
faded, structure_creak vermillion `#D55E00` + hazard banner, other/nature
light gray `#BBBBBB` or hidden. Uncertainty (ellipse, ±2σ wedge): light fill of
the label color. History fades with age. True spot: black X. Warnings (low
SNR, large residual, ADC clipping, endfire): orange text. Axes with units,
equal aspect on maps, titles state the finding. PNGs to `docs/figures/`.
Same visual language as the M0 `m0_simulation.png`.

## Rules that cannot be broken

1. `data/` is gitignored — audio of people. Share by USB/drive. Consenting
   teammates only.
2. Split by session; augmented copies stay in their original's split.
3. `k_ratio`, `low_frac`, filter band, sub-sample onset: editable at the top of
   the tuning notebook.
4. Dashboard runs with `--source sim` and `--source replay`.
5. Fixes always carry uncertainty and residual.

---

## Timeline and build order (hardware stages from the M0 doc)

| Hardware stage (teammate) | When | What Sakshi must have ready |
|---|---|---|
| 1 Bench bring-up: 1 sensor streaming, 20 s with knocks | Sat → 20:00 | **Prompt 1 DONE** (`detect.py`, `01_tune_detection.ipynb`); run it on her first recording |
| 2 Direction node: pair 1.0 m apart, tune detection, measure jitter (20 knocks) | Sat 20:00 → 24:00 | Jitter cell in 01 notebook; **Prompt 2 DONE** (`dashboard.py`, sim + replay, both modes) |
| 3 Position mode: 2 m triangle, calibrate delays, repeatability | Sun 00:00 → 04:00 | **Prompt 5** calibration + repeatability notebook; dashboard position mode |
| 4 Into ATAK, classifier rules gate markers | Sun 04:00 → 08:00 | **Prompt 3** rules baseline classifier (trained model optional) |
| 5 Field/demo ready, feature freeze 08:00, tag by 11:30 | Sun 08:00 → 11:30 | Figures + demo script; Prompt 4 heatmap only if time |

M0 panel D already covers Plot 10 (error vs jitter, air vs water) and panel C
covers ellipse honesty in sim — reuse them, don't rebuild.

## Repo layout

```
recording/         Siddhi: firmware (s2_stream), record.py, plot_recording.py, test recordings (mockup/*.npz)
tools/             Siddhi: m0_sim_plot.py (needs the dropnlisten package)
sim.py             synthetic runs + clips
detect.py          onset detection + tuning view (Plot 1)
pipeline.py        recording -> detections.csv -> fixes.csv (detect, calibrate, locate, classify)
localize.py*       TDOA position / bearing solver        arraycfg.py*  array.yaml loader, speed of sound
classify.py        classifier (rules vs model, by-session eval)  features.py  clip features
calibrate.py       delays, jitter, repeatability (Plots 9, 4)   accuracy.py  heatmap + error vs jitter (Plots 3, 10)
dashboard.py       mission-control web app (+ assets/dashboard.css)
atak.py            CoT sender: sound sources + sensors to ATAK (multicast 239.2.3.1:6969 or phone:4242)
config/            array.yaml (M0 schema), array_triangle.yaml, cal_spots.yaml, channel_delays.yaml (from calibration)
notebooks/         01 tune detection, 02 calibration, 03 accuracy map, 04 classification
docs/figures/      exported PNGs (shown in the dashboard Evidence tab)
data/              GITIGNORED: recordings from record.py, runs, clips
models/            GITIGNORED-worthy: classifier.joblib (rebuild with python classify.py)
* stand-ins: switch to dropnlisten.tdoa / bearing / config once it is pushed (one solver only)
```

## Real hardware so far (recording/mockup, 1 channel, 10 kHz)

- `stage1_knocks`: quiet level ~40 counts, knocks barely above noise (only k_ratio 3 detects).
- `*_pin9`: quiet ~220 counts, 35% of samples stuck at 0 (bottom half of the wave clipped), 93% of energy < 100 Hz.
- `knockmuff_pin9`: quiet ~1000-1280 counts, no clipping at 0: best so far; loud knocks reach the 4074 top.
- Aim: quiet level 1500-2500 counts, knocks well clear of both 0 and 4095. Then tune k_ratio / band in 01.
- Firmware samples channels one after another (1/(fs*n_ch) apart): a fixed inter-channel offset that
  02_calibration removes.

## Commands

```
pip install -r requirements.txt
python sim.py        # sim-direction (array.yaml), sim-position (array_triangle.yaml), clips
python sim.py --array config/array.yaml --run my-run --no-clips
python recording/record.py --port COM5 --seconds 20 --out data/recordings/stage2.npz   # Siddhi's recorder
python pipeline.py data/recordings/stage2.npz --run stage2                            # -> data/runs/stage2
python classify.py                                                                    # train classifier
python dashboard.py --source replay --run stage2                                      # or Setup tab: Process
python atak.py --test                                                                 # test marker in ATAK
```
