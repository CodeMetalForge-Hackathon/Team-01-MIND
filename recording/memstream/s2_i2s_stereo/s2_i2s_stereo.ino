// Drop-and-Listen, Stage 2: two ICS-43434 I2S MEMS mics in stereo on one ESP32-S2 Feather.
//
// Both mics share BCLK, WS and DATA. Mic A has SEL tied to GND (left slot),
// mic B has SEL tied to 3V (right slot). One I2S clock samples both mics at
// exactly the same instant, so there is no channel skew to correct.
//
// Board package: Arduino-ESP32 core 3.x. Board: "Adafruit Feather ESP32-S2"
// (or ESP32S2 Dev Module), USB CDC On Boot: Enabled.
//
// Output: same packet format as s2_stream (works with record.py unchanged).
// Samples are signed 16-bit audio sent as uint16 with a +32768 offset
// ("offset binary"), so silence sits around 32768. n_ch = 2: ch0 = mic A, ch1 = mic B.
// flags = number of packets' worth of time that probably went missing before this packet
// (a long stall, e.g. USB blocked). record.py reports it as "ADC overflows"; it should be 0.
//
// LED: solid on = streaming. Fast blink = I2S failed to start (check pins).

#include <Arduino.h>
#include <ESP_I2S.h>

// ---------- TODO(hardware): match your wiring ----------
static const int PIN_BCLK = 5;     // Feather pin labeled 5  -> both mics' BCLK
static const int PIN_WS   = 6;     // Feather pin labeled 6  -> both mics' LRCL / WS
static const int PIN_DATA = 11;    // Feather pin labeled 11 -> both mics' DOUT / DATA
static const uint32_t FS_HZ = 48000;   // try 32000 if record.py reports dropped packets
static const int SHIFT = 16;           // 32-bit slot -> 16-bit sample. Use 14 for 4x more gain if knocks look tiny
// -------------------------------------------------------

static const uint8_t N_CH = 2;
static const uint16_t N_FRAMES = 256;  // 5.3 ms per packet at 48 kHz
static const uint16_t MAGIC = 0x5AA5;
static const uint32_t PACKET_US = (uint32_t)(1000000ULL * N_FRAMES / FS_HZ);

#pragma pack(push, 1)
struct Header {
  uint16_t magic;
  uint16_t seq;
  uint32_t fs_hz;
  uint32_t first_index;
  uint16_t n_frames;
  uint8_t n_ch;
  uint8_t flags;
};
#pragma pack(pop)

I2SClass i2s;
static int32_t rx[N_FRAMES * N_CH];
static uint16_t out[N_FRAMES][N_CH];
static uint32_t firstIndex = 0;
static uint16_t seq = 0;
static uint32_t lastPacketUs = 0;

static void sendPacket(uint8_t flags) {
  Header h = {MAGIC, seq++, FS_HZ, firstIndex, N_FRAMES, N_CH, flags};
  uint16_t sum = 0;
  const uint8_t *hb = (const uint8_t *)&h;
  for (size_t i = 0; i < sizeof(h); i++) sum += hb[i];
  const uint8_t *sb = (const uint8_t *)out;
  for (size_t i = 0; i < sizeof(out); i++) sum += sb[i];
  Serial.write(hb, sizeof(h));
  Serial.write(sb, sizeof(out));
  Serial.write((const uint8_t *)&sum, 2);
  firstIndex += N_FRAMES;
}

// Fill rx completely. A short read must not be dropped: dropping part of a
// buffer would lose samples silently and could shift left/right (swap mics).
static bool readFullBuffer() {
  size_t have = 0;
  uint32_t start = millis();
  while (have < sizeof(rx)) {
    size_t got = i2s.readBytes((char *)rx + have, sizeof(rx) - have);
    have += got;
    if (got == 0 && millis() - start > 500) return false;  // I2S stopped delivering
  }
  return true;
}

void setup() {
  Serial.begin(921600);  // ignored on native USB
  pinMode(LED_BUILTIN, OUTPUT);
  i2s.setPins(PIN_BCLK, PIN_WS, -1, PIN_DATA);
  if (!i2s.begin(I2S_MODE_STD, FS_HZ, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    while (true) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(100);
    }
  }
  digitalWrite(LED_BUILTIN, HIGH);
  // Let the mics settle, then discard what they produced while starting up.
  delay(150);
  readFullBuffer();
  lastPacketUs = micros();
}

void loop() {
  if (!readFullBuffer()) return;

  // A gap much longer than one packet means the DMA buffers probably overflowed.
  uint32_t now = micros();
  uint32_t dt = now - lastPacketUs;
  lastPacketUs = now;
  uint32_t lost = dt > 4 * PACKET_US ? dt / PACKET_US - 1 : 0;
  uint8_t flags = lost > 255 ? 255 : (uint8_t)lost;

  for (int i = 0; i < N_FRAMES; i++) {
    for (int k = 0; k < N_CH; k++) {
      int32_t s = rx[i * N_CH + k] >> SHIFT;   // 24-bit audio sits in the top of each 32-bit slot
      if (s > 32767) s = 32767;
      if (s < -32768) s = -32768;
      out[i][k] = (uint16_t)(s + 32768);
    }
  }
  sendPacket(flags);
}
