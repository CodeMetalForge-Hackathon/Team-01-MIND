"""ATAK link: send sound sources (and the sensors) to ATAK / WinTAK as Cursor-on-Target (CoT) markers.

No TAK server needed: by default markers are broadcast on the ATAK "mesh" (UDP multicast
239.2.3.1:6969), which every ATAK device on the same Wi-Fi/hotspot listens to out of the box.
You can also send straight to one phone (UDP to <phone-ip>:4242) or to a TAK server's plain
UDP/TCP input.

    python atak.py --test                                   # one test marker at the map origin
    python atak.py --run sim-position                       # replay a run, markers appear as it plays
    python atak.py --run stage2 --to 192.168.43.17:4242     # send to one phone instead of multicast
    python atak.py --run sim-position --print               # just print the CoT XML

What appears in ATAK:
  * one spot marker per SOUND SOURCE (not per knock), callsign like "DNL #3 Knocking (likely person)",
    blue for knocks, green for voice; accuracy = the long radius of the 95% ellipse (CoT "ce")
  * the 95% ellipse (position mode) or the +/-2 sigma direction wedge (direction mode) as a drawn shape
  * the sensors as friendly equipment markers
  * a source marked rescued in the dashboard turns grey with "RESCUED" in its name
Only human-likely sources are sent (the classifier gates what reaches responders).
"""
from __future__ import annotations

import argparse
import datetime as dt
import socket
import time
from xml.sax.saxutils import escape

import numpy as np

STALE_S = 300                      # ATAK hides markers that are not refreshed within this time
MCAST = ("239.2.3.1", 6969)        # ATAK default SA mesh
SPOT = "b-m-p-s-m"                 # spot map marker
SENSOR = "a-f-G-E-S"               # friendly ground equipment: sensor
SHAPE = "u-d-f"                    # drawn freehand polygon/polyline
COLORS = {"knock_tap": (0, 114, 178), "voice": (0, 158, 115), "resolved": (130, 130, 130), "sensor": (64, 64, 64)}


def argb(rgb, a=255):
    r, g, b = rgb
    v = (a << 24) | (r << 16) | (g << 8) | b
    return v - (1 << 32) if v >= (1 << 31) else v      # ATAK wants a signed 32-bit int


def _t(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def event(uid, cot_type, lat, lon, ce, detail, how="m-g"):
    now = dt.datetime.now(dt.timezone.utc)
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<event version="2.0" uid="{escape(uid)}" type="{cot_type}" how="{how}" time="{_t(now)}" '
            f'start="{_t(now)}" stale="{_t(now + dt.timedelta(seconds=STALE_S))}">'
            f'<point lat="{lat:.8f}" lon="{lon:.8f}" hae="9999999.0" ce="{ce:.2f}" le="9999999.0"/>'
            f'<detail>{detail}</detail></event>')


def _shape(uid, name, lats, lons, rgb, parent_uid):
    links = "".join(f'<link point="{la:.8f},{lo:.8f}"/>' for la, lo in zip(lats, lons))
    detail = (f'{links}<strokeColor value="{argb(rgb)}"/><strokeWeight value="3.0"/>'
              f'<fillColor value="{argb(rgb, 70)}"/><contact callsign="{escape(name)}"/>'
              f'<labels_on value="false"/><link uid="{escape(parent_uid)}" relation="p-p" type="{SPOT}"/>')
    return event(uid, SHAPE, float(np.mean(lats)), float(np.mean(lons)), 9999999.0, detail, how="h-e")


def source_events(src, contacts, acks, run):
    """CoT for every sound source: a spot marker + its uncertainty shape."""
    import dashboard as Db
    from arraycfg import compass_to_local_angle
    from localize import ellipse_points
    out = []
    xb = src.cfg["geo"]["x_axis_bearing_deg"]
    _, _, reach = Db.map_bounds(src)
    for c in contacts:
        state = acks.get(c["cid"], "")
        lab = c["labels"].most_common(1)[0][0]
        rgb = COLORS["resolved"] if state == "resolved" else COLORS.get(lab, COLORS["knock_tap"])
        word = {"KNOCKING": "Knocking", "CALLING": "Calling"}.get(Db.headline(c).split(" ·")[0], "Knocks+voice")
        name = f"DNL #{c['num']} {word}" + (" RESCUED" if state == "resolved" else " (likely person)")
        where, acc = Db.where_text(src, c)
        if c["kind"] == "position":
            x, y, ce = c["x"], c["y"], c["r95"]
            e = ellipse_points(c["x"], c["y"], c["cov"], n=36)
        else:   # direction only: marker along the bearing, wedge shows the real uncertainty
            mid, _ = Db.pair_geometry(src)
            th = np.radians(compass_to_local_angle(c["bearing"], xb))
            x, y = mid + reach * 0.6 * np.array([np.cos(th), np.sin(th)])
            ce = reach
            s2 = np.radians(min(2 * c["sigma"], 89.0))
            angs = np.linspace(th - s2, th + s2, 24)
            e = np.c_[np.r_[mid[0], mid[0] + reach * np.cos(angs), mid[0]],
                      np.r_[mid[1], mid[1] + reach * np.sin(angs), mid[1]]]
        lat, lon = Db.to_latlon(src, [x], [y])
        uid = f"DNL-{run}-{c['cid']}"
        remarks = (f"{where} ({acc}). {len(c['sounds'])} sounds"
                   + (", regular knock pattern" if c["repeated"] else "") + f". Confidence {c['conf']:.0%}."
                   + (" Direction only: the person is somewhere in the wedge." if c["kind"] == "bearing" else "")
                   + (" One mic only: no direction; the circle is a guessed hearing range around the node."
                      if len(src.cfg["_xy"]) == 1 else ""))
        detail = (f'<contact callsign="{escape(name)}"/><remarks>{escape(remarks)}</remarks>'
                  f'<color argb="{argb(rgb)}"/><usericon iconsetpath="COT_MAPPING_SPOTMAP/{SPOT}/{argb(rgb)}"/>'
                  f'<precisionlocation altsrc="???" geopointsrc="DNL-TDOA"/>')
        out.append(event(uid, SPOT, float(lat[0]), float(lon[0]), ce, detail))
        la, lo = Db.to_latlon(src, e[:, 0], e[:, 1])
        out.append(_shape(uid + "-area", name + " area", la, lo, rgb, uid))
    return out


def sensor_events(src, run):
    import dashboard as Db
    xy = src.cfg["_xy"]
    la, lo = Db.to_latlon(src, xy[:, 0], xy[:, 1])
    return [event(f"DNL-{run}-sensor-{n}", SENSOR, float(a), float(b), 0.5,
                  f'<contact callsign="{escape(n)}"/><remarks>Drop-and-Listen listening node</remarks>'
                  f'<color argb="{argb(COLORS["sensor"])}"/>')
            for n, a, b in zip(src.cfg["_names"], la, lo)]


def test_event(lat, lon):
    return event("DNL-test-marker", SPOT, lat, lon, 1.0,
                 f'<contact callsign="DNL TEST MARKER"/><remarks>If you can see this, the ATAK link works.</remarks>'
                 f'<color argb="{argb((213, 94, 0))}"/>')


class CotSender:
    """targets: list like ["multicast", "192.168.43.17:4242", "tcp:10.0.0.5:8087"]."""

    def __init__(self, targets=("multicast",), iface_ip=None):
        self.targets = []
        for t in targets:
            t = t.strip()
            if not t:
                continue
            if t == "multicast":
                self.targets.append(("udp", MCAST))
            elif t.startswith("tcp:"):
                h, p = t[4:].rsplit(":", 1)
                self.targets.append(("tcp", (h, int(p))))
            else:
                h, p = t.rsplit(":", 1) if ":" in t else (t, "4242")
                self.targets.append(("udp", (h, int(p))))
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.udp.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        if iface_ip:   # pick the hotspot's network adapter when the laptop has several
            self.udp.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface_ip))

    def send(self, xmls):
        errors = []
        for kind, addr in self.targets:
            try:
                if kind == "udp":
                    for x in xmls:
                        self.udp.sendto(x.encode("utf-8"), addr)
                else:
                    with socket.create_connection(addr, timeout=2) as s:
                        for x in xmls:
                            s.sendall(x.encode("utf-8"))
            except OSError as e:
                errors.append(f"{kind}://{addr[0]}:{addr[1]}: {e}")
        return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder under data/runs/ to replay")
    ap.add_argument("--to", action="append", help="multicast (default), <ip>:<port> (UDP), or tcp:<ip>:<port>")
    ap.add_argument("--iface", help="IP of this laptop on the hotspot (only if multicast goes out the wrong adapter)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--test", action="store_true", help="send one test marker at the map origin and exit")
    ap.add_argument("--print", action="store_true", help="print the CoT instead of sending")
    ap.add_argument("--node-coords", metavar="ARRAY", nargs="?", const="config/array.yaml",
                    help="print NODE_LAT / NODE_LON for the ESP32 firmware (center of the sensors in ARRAY)")
    args = ap.parse_args()
    from arraycfg import ROOT, load_array
    if args.node_coords:
        import dashboard as Db
        cfg = load_array(args.node_coords)

        class _S:  # minimal stand-in so to_latlon can read the geo block
            pass
        s = _S()
        s.cfg = cfg
        la, lo = Db.to_latlon(s, cfg["_xy"][:, 0], cfg["_xy"][:, 1])
        print(f"static const double NODE_LAT = {la.mean():.7f};\nstatic const double NODE_LON = {lo.mean():.7f};")
        return
    sender = CotSender(args.to or ["multicast"], args.iface)
    if args.test or not args.run:
        g = load_array("config/array.yaml")["geo"]
        x = test_event(g["lat0"], g["lon0"])
        print(x) if args.print else print("sent test marker", sender.send([x]) or "OK", "->", sender.targets)
        return
    import dashboard as Db
    src = Db.PlaybackSource(ROOT / "data/runs" / args.run)
    t0, last_sig, last_send = time.monotonic(), None, 0.0
    while True:
        now = (time.monotonic() - t0) * args.speed
        fixes, _ = src.snapshot(now)
        contacts = Db.build_contacts(fixes)
        sig = [(c["cid"], len(c["sounds"])) for c in contacts]
        if sig != last_sig or time.monotonic() - last_send > 10:
            xmls = sensor_events(src, args.run) + source_events(src, contacts, {}, args.run)
            if args.print:
                print("\n".join(xmls))
            else:
                err = sender.send(xmls)
                print(f"t={now:5.1f}s sent {len(contacts)} sources + {len(src.cfg['_names'])} sensors"
                      + (f"  ERRORS: {err}" if err else ""))
            last_sig, last_send = sig, time.monotonic()
        if now > src.duration_s or args.print:
            break
        time.sleep(0.25)


if __name__ == "__main__":
    main()