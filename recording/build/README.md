# Drop-and-Listen: three-node recording and triangulation

Three Grove sound nodes side by side, all on one shared clock, recorded together on the laptop.

| Node | Board | Grove sound (yellow) | Power |
| --- | --- | --- | --- |
| 1 (time master) | Adafruit Feather ESP32-S2 | pin **9** | red 3V, black GND |
| 2 | Seeed XIAO ESP32-C5 | **D0** | red 3V3, black GND |
| 3 | Seeed XIAO ESP32-S3 Plus | **D0** | red 3V3, black GND |

White Grove wire unused. Each node on its own breadboard, each plugged into the laptop by USB.

## Firmware (Arduino IDE, esp32 board package 3.3.5+, USB CDC On Boot: Enabled)

| Sketch | Use |
| --- | --- |
| `firmware/node1_s2_debug` | Text debug for node 1 alone (Serial Monitor 115200) |
| `firmware/node2_c5_debug` | Text debug for node 2 alone |
| `firmware/node3_s3_debug` | Text debug for node 3 alone |
| `firmware/dnl_stream_node` | **Deploy: the same sketch on all three.** Streams raw samples with shared-clock timestamps |

Debug sketches print stats every 2 s ("ok: signal is alive") and one line per sound with its features
and a guess (knock, voice, cough...). Run all three at once in three Serial Monitors.

Deploy sketch node IDs are automatic by chip: S2 = node 1, C5 = node 2, S3 = node 3. No edits
needed. Close all Serial Monitors before recording: the deploy sketch sends binary.

## Python

```bash
cd python
pip install -r requirements.txt
python record_plot.py both --ports COM9 COM10 COM11 --seconds 20 --pos 1=0,0 2=2.5,0 3=1.25,2
```

`--pos` = each node's position in meters, measured from one corner. Other commands:
`record` (only record), `plot data/run.npz --pos ...` (re-analyse a saved run), `--sound 2`
(plot a different sound), `--temp 22` (air temperature), `--k 6` (stricter detector), `--save-only`.

Output: a table of every sound with each node's arrival time and the triangulated position, and per
node a figure saved in `data/run/plots/nodeN.png`:

1. ADC vs time (ms), whole recording, detected onsets in red
2. ADC vs time (ms), zoomed on the chosen sound
3. ADC vs distance (m) the sound travelled since it was made, with the computed distance from the
   source to that node in green. The knock starting at the green line means the timing and the
   triangulation agree.

## Layout

Three nodes in a triangle, 2-3 m per side, on the floor or tables, mics facing up. Knock inside the
triangle on something you hold (not the table the nodes sit on). Measure the node positions with a
tape measure and pass them with `--pos`.

## Notes

- With exactly 3 nodes the position always fits perfectly ("fit error 0"); the ±radius is the
  uncertainty from timing error. A 4th node would give a real quality check.
- Check the recording summary: each node should be "synced 100%", dropped 0, overflows 0.
- Data stays out of git (`data/`).
