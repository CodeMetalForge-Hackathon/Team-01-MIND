# Drop-and-Listen

Cheap, throwable listening nodes keep listening at a disaster site after responders leave. A trapped
person's knock or voice becomes a location on the rescue team's map (our dashboard and ATAK) within seconds.

## Quick start

```
pip install -r requirements.txt
python sim.py                                   # synthetic data (no hardware needed)
python classify.py                              # train the sound classifier
python dashboard.py --source sim --array config/array_triangle.yaml
```
Open http://127.0.0.1:8050. **Setup tab** → process a recording, switch runs, turn on the ATAK link.

## With hardware

```
python recording/record.py --port COM5 --seconds 20 --out data/recordings/stage2.npz
python pipeline.py data/recordings/stage2.npz --run stage2
python dashboard.py --source replay --run stage2
```

## Status

| Piece | Status |
|---|---|
| Sensor streaming (ESP32-S2, 10 kHz) | 1 channel recording; signal level being fixed |
| Detection + tuning view | done (`detect.py`, notebook 01) |
| Direction / position solver | done in sim (`localize.py`); M0 package to be merged |
| Calibration + honest uncertainty | done in sim (notebook 02) |
| Classifier | done on synthetic clips; needs real clips |
| Dashboard (map, sources, log, evidence, setup) | done |
| ATAK link (CoT multicast) | built, needs on-phone test |

See `CLAUDE.md` for data formats, conventions and the full layout.
