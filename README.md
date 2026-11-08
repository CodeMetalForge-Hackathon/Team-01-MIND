# MIND: Mesh Integrated Node Deployment

**Drop it. Walk away. Keep listening.**

MIND is a mesh of small, cheap sensor nodes that keeps listening after the search team moves on. Nodes are
air-dropped by helicopter, placed by hand, or carried on aircraft. Each node combines a **microphone**, a
**humidity/temperature sensor** and a **thermal imaging camera**. When a hidden person makes a sound
(footsteps, movement, a knock, a voice), the nodes work out where it came from, check that it is human and
not falling debris, confirm a warm body with the thermal camera, and put it on the team's map (our dashboard
and ATAK) within seconds.

Code Metal Forge Hackathon 2026 · Team 1 (Sakshi: software, Siddhi: hardware)

## Why

A victim is trapped inside a building. A rescue helicopter flies right past, and never finds them.
Searchers can't see through walls, rubble or forest, and they can't stay at every site. The same problem
shows up in disaster relief, urban combat and hostage recovery: people are hidden, and no team can search
everywhere at once. Sound gets through where sight can't.

## A moving mesh

The mesh doesn't have to stand still. It mixes two kinds of node:

| Node | How it gets there | Role |
|---|---|---|
| **Fixed** | air-dropped or hand-placed | listens for days at one spot; locates sounds with its neighbours |
| **Airborne** | carried on an aircraft or drone | joins the mesh from above, relays alerts from fixed nodes that can't reach the gateway, extends coverage as it moves |

Nodes share one clock over ESP-NOW (node 1 is the time master), so sounds heard by different nodes line up
on one timeline. Alerts hop node to node to a gateway, then onto every team's ATAK map.

## Sensors

| Sensor | Why |
|---|---|
| Microphone (Grove sound sensor; ICS-43434 I2S option) | hears footsteps, movement, knocks, voice; timing gives location |
| Humidity + temperature | speed of sound changes with air conditions; using the measured value keeps locations accurate |
| Thermal imaging camera | confirms a warm, human-sized body near the located sound; cuts false alarms |

## How it works

```
 nodes (mic + ESP32)  ──ESP-NOW clock──▶  nodes_to_run.py  ──▶  detect.py  ──▶  localize.py  ──▶  classify  ──▶  dashboard + ATAK
 fixed or airborne      shared timeline     one time grid       onsets          TDOA position     human or      map, contacts,
                                                                (sub-sample)    + 95% ellipse     not?          CoT markers
```

1. **Detect**: band-pass → envelope → adaptive noise floor → precise onset time for every sound. Within each
   travel-time window, each node keeps its clearest onset. If two nodes clearly heard a sound and the third
   missed it (for example, it was briefly deaf after a noise trigger), a **guided search** looks for it on the
   third node in the short window where it must have arrived.
2. **Locate (TDOA)**: sound travels ~343 m/s (corrected for the measured temperature and humidity):

   c = 331.4 + 0.6·T + 0.0124·RH  (m/s)

   The nearer node hears it first; the time gaps give a direction (2 mics) or a position (3+ nodes). Every
   fix comes with an honest 95% uncertainty ellipse from the real timing jitter.
3. **Classify + confirm**: a small model separates human sounds from debris and background noise; the
   thermal camera confirms a warm body. Only human-likely contacts reach responders.
4. **Group**: repeated sounds from the same spot become one *contact* ("#2 · likely person"), not
   fifty separate alerts.
5. **Report**: the dashboard shows contacts on a close-up, street or satellite map; `atak.py` sends them to
   ATAK as Cursor-on-Target markers with their accuracy circle.

## Real results (Nov 8, 3 nodes)

Session 1: three nodes 1.4–2.5 m apart, knocks, a cough and a dropped book at three measured spots, 104 s
recording. Error = distance from our estimate to the true spot.

| Spot | Sounds located | Median error | Best |
|---|---|---|---|
| knock_A | 6 | 72 cm | 47 cm |
| knock_B | 4 | 11 cm | 7 cm |
| knock_C | 5 | 104 cm | 98 cm |
| cough_A | 1 | 61 cm | |
| impact_B | 1 | 68 cm | |
| **all** | **17** | **70 cm** | **7 cm** |

Spot C is off by almost the same ~1 m every time, which points to a fixed timing offset on one node (or a
mis-measured spot), not random noise. Per-node calibration (`calibrate.py`) is built to remove exactly
that; next we calibrate on session 1 and test on session 2. Accuracy shown in the video is illustrative;
the numbers above are measured.

Reproduce:
```
python run_labeled.py datasets/2026-11-08/recordings/session1.npz
python dashboard.py --source replay --run session1
```

## Quick start (no hardware)

```
pip install -r requirements.txt
python sim.py                                   # synthetic recordings + ground truth
python classify.py                              # train the sound classifier
python dashboard.py --source sim --array config/array_triangle.yaml
```
Open http://127.0.0.1:8050. In the **Setup** tab you can process a recording, switch runs and turn on
the ATAK link.

## With hardware (3 synced nodes)

```
python recording/record_plot.py record --ports COM9 COM10 COM11 --seconds 20 --out data/run1.npz
python nodes_to_run.py data/run1.npz --pos 1=0,0 2=2.5,0 3=1.43,1.8
python pipeline.py data/run1_merged.npz --run real1 --array config/array_nodes.yaml --params config/params_nodes.yaml
python dashboard.py --source replay --run real1
```
`--pos` = where the nodes actually are, in metres (node id = x,y). Firmware: `dnl_stream_node.ino` on all
three boards (S2 = node 1 / time master, C5 = node 2, S3 = node 3). Set `FORCE_NODE_ID` to 0 or the right
number before flashing.

## On a Raspberry Pi

```
sudo apt install -y python3-venv python3-full
python3 -m venv ~/dnl-venv && source ~/dnl-venv/bin/activate
pip install numpy scipy pandas pyyaml matplotlib plotly dash scikit-learn pyserial
sudo usermod -aG dialout $USER && sudo reboot          # lets you read the USB nodes
```
After that, `source ~/dnl-venv/bin/activate` in every new terminal. The nodes appear as `/dev/ttyACM0..2`
(use them instead of COM ports). Run the dashboard with `--host 0.0.0.0` and open `http://<pi-ip>:8050`
from any device on the same Wi-Fi.

## ATAK

No TAK server needed. Markers go out on the standard ATAK multicast (239.2.3.1:6969), so any ATAK or
WinTAK device on the same Wi-Fi or hotspot sees them. Tested Nov 7 with WinTAK-CIV 5.8.

```
python atak.py --test                               # one test marker at the map origin
python atak.py --run session1                       # replay a run into ATAK
python atak.py --test --to 192.168.43.1:4242        # straight to one phone if multicast is blocked
python atak.py --node-coords                        # NODE_LAT / NODE_LON for the firmware
```
In ATAK you get one marker per contact, its 95% accuracy shape, and the nodes as friendly equipment.
Put Wi-Fi names and passwords in the firmware on your machine only, never in git.

## Repo layout

| Path | What |
|---|---|
| `sim.py` | synthetic recordings with ground truth |
| `nodes_to_run.py` | 3 synced node recordings → one multi-channel recording + array config |
| `detect.py`, `localize.py`, `pipeline.py` | onsets → positions → `fixes.csv` |
| `run_labeled.py` | runs a labelled session end to end and prints real accuracy |
| `features.py`, `classify.py`, `make_clips.py` | sound classifier and real-clip tooling |
| `calibrate.py`, `accuracy.py` | timing calibration and accuracy maps |
| `dashboard.py`, `assets/` | mission-control dashboard |
| `atak.py` | ATAK / Cursor-on-Target link |
| `config/` | array geometry, map origin, calibration spots |
| `recording/` | firmware, `record_plot.py`, debug sketches (Siddhi) |
| `datasets/` | shared recordings of the team (consented): knocks, voice, cough, impacts |
| `notebooks/` | tuning, calibration, accuracy, classification |

`data/` (working recordings, runs) and `models/` are git-ignored. Only consented team recordings go in
`datasets/`. See `CLAUDE.md` for data formats and conventions.

## Status

| Piece | Status |
|---|---|
| 3 nodes on a shared ESP-NOW clock | working (Siddhi): 0% dropped packets in session 1 |
| Converter into our pipeline | done (`nodes_to_run.py`) |
| Detection with guided search | done; on session 1: 22 sounds located (6 before) |
| Position solver + honest uncertainty | done; real median error 70 cm, best 7 cm |
| Calibration on real data | next (calibrate on session 1, test on session 2) |
| Classifier (knock, voice, impact, background) | trained on synthetic clips; real clips now available |
| Footsteps / movement class | planned: needs recordings |
| Dashboard (close-up / street / satellite) | done |
| ATAK link | done, tested in WinTAK |
| Runs on Raspberry Pi | set up Nov 8 |
| Humidity / temperature sensor | being integrated (values set in config today) |
| Thermal camera confirmation | being integrated |
| Alert relay across the mesh + gateway | planned |
| Airborne nodes | planned: needs GPS position at each sound |
| Live (real-time) mode | planned: today = record, then process |
