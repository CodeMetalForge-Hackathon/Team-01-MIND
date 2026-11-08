// Drop-and-Listen node firmware: ICS-43434 I2S MEMS microphone(s) on an ESP32-S2 Feather.
//
// Streams audio to the laptop over USB in the SAME packet format as before, so record.py,
// pipeline.py and the dashboard need no changes. Samples are mapped to 12-bit-style counts
// (quiet = 2048, range 0..4095) so the existing software sees what it expects.
// Optional Plan B: the node joins a Wi-Fi hotspot and announces itself in ATAK every 10 s.
//
// Wiring (Adafruit ICS-43434 breakout -> Feather):
//   3V  -> 3V          GND -> GND
//   BCLK -> I2S_BCLK pin    LRCL (WS) -> I2S_WS pin    DOUT -> I2S_DIN pin
//   SEL -> GND  for mic 0 (left slot)
// Second mic (2-channel direction mode): wire it IN PARALLEL to the same BCLK, LRCL and DOUT
// pins, but with its SEL -> 3V (right slot). Both mics then share one clock: perfectly in sync.
// Arduino IDE: board "Adafruit Feather ESP32-S2", esp32 core 3.x (needs ESP_I2S.h).

#include <Arduino.h>
#include <ESP_I2S.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <time.h>

// ---------- TODO(hardware): CHECK these against your wires ----------
static const int I2S_BCLK = 5;      // mic BCLK
static const int I2S_WS = 6;        // mic LRCL / WS
static const int I2S_DIN = 9;       // mic DOUT (data INTO the ESP32)
static const int N_CH = 1;          // 1 = one mic (SEL->GND). 2 = two mics (second SEL->3V)
static const uint32_t FS_HZ = 24000;  // ICS-43434 works from ~23 to ~51 kHz; 24 kHz = 42 us per sample
static const int GAIN_SHIFT = 10;   // bigger = quieter. Aim: knocks peak ~800-1500 counts from 2048.
static const bool SWAP_LR = false;  // if tapping mic 0 shows up on channel 1, set true
// ---------------------------------------------------------------------

// ---------- Plan B: announce this node in ATAK over Wi-Fi (optional) ----------
// Set ENABLE_ATAK to false for USB-only. DO NOT commit a real hotspot password to git.
static const bool ENABLE_ATAK = true;
static const char *WIFI_SSID = "YOUR_HOTSPOT_NAME";
static const char *WIFI_PASS = "YOUR_HOTSPOT_PASSWORD";
static const char *NODE_NAME = "DNL-NODE-1";
static const double NODE_LAT = 42.36030;   // from: python atak.py --node-coords
static const double NODE_LON = -71.05880;
static const uint32_t ATAK_EVERY_MS = 10000;
static const int LOUD_COUNTS = 400;        // a swing this big on ch0 counts as "loud sound"
// ------------------------------------------------------------------------------

static const uint16_t N_FRAMES = 256;   // frames per USB packet (~10.7 ms at 24 kHz)
static const uint16_t MAGIC = 0x5AA5;

#pragma pack(push, 1)
struct Header {
  uint16_t magic;
  uint16_t seq;
  uint32_t fs_hz;
  uint32_t first_index;
  uint16_t n_frames;
  uint8_t n_ch;
  uint8_t flags;      // count of samples that hit 0 or 4095 in this packet (clipping), max 255
};
#pragma pack(pop)

static I2SClass i2s;
static int32_t rx[N_FRAMES * 2];               // stereo int32 slots: L, R, L, R, ...
static uint16_t frames[N_FRAMES][N_CH];
static uint32_t firstIndex = 0;
static uint16_t seq = 0;
static uint32_t clipsThisPacket = 0;
static uint32_t clipsTotal = 0;

// Plan B state
static WiFiUDP udp;
static uint32_t lastAtakMs = 0;
static uint32_t lastLoudMs = 0;
static bool timeStarted = false;

static inline uint16_t toCounts(int32_t slot) {
  // ICS-43434: 24-bit two's complement, left-justified in the 32-bit slot.
  int32_t v = (slot >> 8) >> GAIN_SHIFT;       // 24-bit sample, then scale down
  v += 2048;                                   // quiet = 2048, like the old 12-bit ADC data
  if (v <= 0) { clipsThisPacket++; return 0; }
  if (v >= 4095) { clipsThisPacket++; return 4095; }
  return (uint16_t)v;
}

static void sendPacket() {
  Header h;
  h.magic = MAGIC;
  h.seq = seq++;
  h.fs_hz = FS_HZ;
  h.first_index = firstIndex;
  h.n_frames = N_FRAMES;
  h.n_ch = N_CH;
  h.flags = clipsThisPacket > 255 ? 255 : (uint8_t)clipsThisPacket;
  clipsTotal += clipsThisPacket;
  clipsThisPacket = 0;

  uint16_t sum = 0;
  const uint8_t *hb = (const uint8_t *)&h;
  for (size_t i = 0; i < sizeof(h); i++) sum += hb[i];
  const uint8_t *sb = (const uint8_t *)frames;
  for (size_t i = 0; i < sizeof(frames); i++) sum += sb[i];

  Serial.write(hb, sizeof(h));
  Serial.write(sb, sizeof(frames));
  Serial.write((const uint8_t *)&sum, 2);
  firstIndex += N_FRAMES;
}

// ---------- Plan B: CoT status marker (UDP multicast to the ATAK mesh 239.2.3.1:6969) ----------
static void cotTime(char *out, size_t n, time_t t) {
  struct tm g;
  gmtime_r(&t, &g);
  strftime(out, n, "%Y-%m-%dT%H:%M:%S.000Z", &g);
}

static void sendAtakStatus() {
  time_t now = time(nullptr);
  if (now < 1700000000) return;                // no internet time yet: ATAK would drop the marker
  char tNow[32], tStale[32];
  cotTime(tNow, sizeof(tNow), now);
  cotTime(tStale, sizeof(tStale), now + 60);   // vanishes from ATAK if the node goes silent for 60 s

  char loud[48];
  if (lastLoudMs) snprintf(loud, sizeof(loud), "last loud sound %lus ago", (unsigned long)((millis() - lastLoudMs) / 1000));
  else snprintf(loud, sizeof(loud), "no loud sound yet");

  char xml[900];
  snprintf(xml, sizeof(xml),
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    "<event version=\"2.0\" uid=\"%s-%s\" type=\"a-f-G-E-S\" how=\"h-g-i-g-o\" time=\"%s\" start=\"%s\" stale=\"%s\">"
    "<point lat=\"%.7f\" lon=\"%.7f\" hae=\"9999999.0\" ce=\"5.0\" le=\"9999999.0\"/>"
    "<detail><contact callsign=\"%s\"/>"
    "<remarks>Drop-and-Listen node: %d mic(s), %lu Hz I2S, streaming, %s, Wi-Fi %d dBm</remarks>"
    "<color argb=\"-12566464\"/></detail></event>",
    NODE_NAME, WiFi.macAddress().c_str(), tNow, tNow, tStale, NODE_LAT, NODE_LON, NODE_NAME,
    N_CH, (unsigned long)FS_HZ, loud, (int)WiFi.RSSI());

  udp.beginPacket(IPAddress(239, 2, 3, 1), 6969);
  udp.write((const uint8_t *)xml, strlen(xml));
  udp.endPacket();
}

static void atakTick() {
  if (!ENABLE_ATAK || WiFi.status() != WL_CONNECTED) return;
  if (!timeStarted) {
    configTime(0, 0, "pool.ntp.org", "time.google.com");
    timeStarted = true;
  }
  if (millis() - lastAtakMs >= ATAK_EVERY_MS) {
    lastAtakMs = millis();
    sendAtakStatus();
  }
}
// ------------------------------------------------------------------------------------------------

void setup() {
  Serial.begin(921600);  // baud is ignored on native USB CDC

  if (ENABLE_ATAK) {                           // non-blocking: audio starts streaming right away
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }

  // Always read both I2S slots; mic 0 is the left slot (SEL->GND), mic 1 the right (SEL->3V).
  i2s.setPins(I2S_BCLK, I2S_WS, -1, I2S_DIN);
  if (!i2s.begin(I2S_MODE_STD, FS_HZ, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO, I2S_STD_SLOT_BOTH)) {
    while (true) delay(1000);                  // I2S failed to start (check pins): no packets will arrive
  }
  // The mic needs a moment after the clock starts: throw away ~100 ms.
  for (int i = 0; i < 10; i++) i2s.readBytes((char *)rx, sizeof(rx));
}

void loop() {
  atakTick();                                  // returns immediately except once every 10 s

  size_t got = i2s.readBytes((char *)rx, sizeof(rx));     // blocks until N_FRAMES stereo frames
  size_t nFrames = got / (2 * sizeof(int32_t));
  for (size_t f = 0; f < nFrames && f < N_FRAMES; f++) {
    for (int k = 0; k < N_CH; k++) {
      frames[f][k] = toCounts(rx[2 * f + (SWAP_LR ? 1 - k : k)]);
    }
    if (ENABLE_ATAK && abs((int)frames[f][0] - 2048) > LOUD_COUNTS) lastLoudMs = millis();
  }
  if (nFrames >= N_FRAMES) sendPacket();
}