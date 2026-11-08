// dnl_stream_node: ONE sketch for all three nodes.
// Streams raw Grove sound samples over USB, stamped with a SHARED clock:
//   node 1 (the S2) is the time master and broadcasts its clock over ESP-NOW every 0.5 s;
//   nodes 2 and 3 sync to it. The laptop then lines up all three recordings to microseconds.
// Boards: Feather ESP32-S2 (pin 9) = node 1, XIAO ESP32-C5 (D0) = node 2,
// XIAO ESP32-S3 Plus (D0) = node 3. IDs are automatic by chip. USB CDC On Boot: Enabled.
// Node 1 (S2) also reads an analog vibration sensor on GPIO8, sampled by the same ADC right
// after each sound sample (same clock). Its packets carry channel 1 in the header's last byte;
// sound packets carry 0, so older laptop tools that ignore that byte keep working.
// The only text this sketch prints is one boot line; everything else is binary packets.
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "soc/soc_caps.h"
#include "esp_adc/adc_continuous.h"

// ------------------------------------------------------------------ settings
#define FORCE_NODE_ID 0          // 0 = automatic by chip (S2 = 1, C5 = 2, S3 = 3). Only force an ID
                                 // when flashing ONE board, and set it back to 0 before flashing the next:
                                 // two boards with the same ID get ignored, and with no node 1 nobody syncs.
#define FS_HZ 10000              // samples per second
#define N_PER_PACKET 250         // 25 ms per packet
#define ESPNOW_CHANNEL 1

struct MacId { uint8_t mac[6]; uint8_t id; };
// After the first flash, each C5 prints "MAC xx:xx:..". Put them here and reflash, or use FORCE_NODE_ID.
static const MacId MAC_TABLE[] = {
  {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 2},   // TODO: node 2 C5 MAC
  {{0x00, 0x00, 0x00, 0x00, 0x00, 0x01}, 3},   // TODO: node 3 C5 MAC
};

#if CONFIG_IDF_TARGET_ESP32S2
#define SOUND_PIN 9              // Feather ESP32-S2: pin 9 (ADC1)
#define VIB_PIN 8                // vibration sensor: GPIO8 (ADC1_CH7). -1 = no vibration sensor
#define DEFAULT_ID 1
#elif CONFIG_IDF_TARGET_ESP32C5
#define SOUND_PIN 1              // XIAO ESP32-C5: D0 = GPIO1
#define DEFAULT_ID 2
#elif CONFIG_IDF_TARGET_ESP32S3
#define SOUND_PIN 1              // Seeed XIAO ESP32-S3 Plus: D0 = GPIO1 (ADC1)
#define DEFAULT_ID 3             // node 3 automatically
#else
#define SOUND_PIN 1
#define DEFAULT_ID 2
#endif
#ifndef VIB_PIN
#define VIB_PIN -1               // only node 1 has a vibration sensor
#endif
#define N_CH (VIB_PIN >= 0 ? 2 : 1)
#define CH_SOUND 0               // StreamHeader.reserved: which sensor the packet holds
#define CH_VIB 1

#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32S2
#define ADC_FORMAT ADC_DIGI_OUTPUT_FORMAT_TYPE1
#define ADC_CH(p) ((p)->type1.channel)
#define ADC_DATA(p) ((p)->type1.data)
#else
#define ADC_FORMAT ADC_DIGI_OUTPUT_FORMAT_TYPE2
#define ADC_CH(p) ((p)->type2.channel)
#define ADC_DATA(p) ((p)->type2.data)
#endif

// ------------------------------------------------------------------ packets
#pragma pack(push, 1)
struct Beacon {                  // ESP-NOW, node 1 -> others
  uint8_t magic;                 // 0xB5
  uint32_t seq;
  int64_t master_us;
};
struct StreamHeader {            // USB, node -> laptop
  uint16_t magic;                // 0x5AA6
  uint8_t node;
  uint8_t synced;
  uint16_t seq;
  uint32_t fs_hz;
  uint32_t first_index;
  int64_t first_t_us;            // shared-clock time of the first sample in this packet
  uint16_t n;
  uint8_t overflows;
  uint8_t reserved;              // channel: 0 = sound, 1 = vibration
};
#pragma pack(pop)

static const uint8_t BROADCAST[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static uint8_t nodeId = DEFAULT_ID;
static bool isMaster = false;

// ------------------------------------------------------------------ clock sync (nodes 2, 3)
static portMUX_TYPE syncMux = portMUX_INITIALIZER_UNLOCKED;
static const int NB = 8;
static int64_t bLoc[NB], bMas[NB];
static int bHead = 0, bCount = 0;
static int64_t lastBeaconLocal = 0, refX = 0, refY = 0;
static double fitA = 0, fitB = 1;

static void fitClock() {
  int64_t x0 = bLoc[0], y0 = bMas[0];
  double mx = 0, my = 0;
  for (int i = 0; i < bCount; i++) { mx += (double)(bLoc[i] - x0); my += (double)(bMas[i] - y0); }
  mx /= bCount; my /= bCount;
  double sxy = 0, sxx = 0;
  for (int i = 0; i < bCount; i++) {
    double dx = (double)(bLoc[i] - x0) - mx, dy = (double)(bMas[i] - y0) - my;
    sxy += dx * dy; sxx += dx * dx;
  }
  fitB = sxx > 0 ? sxy / sxx : 1.0;
  fitA = my - fitB * mx;
  refX = x0; refY = y0;
}

static bool isSynced() {
  if (isMaster) return true;
  return bCount >= 2 && (esp_timer_get_time() - lastBeaconLocal) < 3000000;
}

static int64_t toShared(int64_t localUs) {
  if (isMaster) return localUs;
  portENTER_CRITICAL(&syncMux);
  int64_t out = bCount >= 2 ? refY + (int64_t)llround(fitA + fitB * (double)(localUs - refX)) : localUs;
  portEXIT_CRITICAL(&syncMux);
  return out;
}

static void onRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  int64_t now = esp_timer_get_time();           // capture first
  if (isMaster || len < (int)sizeof(Beacon) || data[0] != 0xB5) return;
  const Beacon *b = (const Beacon *)data;
  portENTER_CRITICAL(&syncMux);
  bLoc[bHead] = now; bMas[bHead] = b->master_us;
  bHead = (bHead + 1) % NB;
  if (bCount < NB) bCount++;
  lastBeaconLocal = now;
  if (bCount >= 2) fitClock();
  portEXIT_CRITICAL(&syncMux);
}

// ------------------------------------------------------------------ ADC
static adc_continuous_handle_t adc = nullptr;
static adc_channel_t soundChannel;
static adc_channel_t vibChannel = (adc_channel_t)-1;
static volatile uint32_t overflows = 0;
static uint32_t lastOverflows = 0;
static uint8_t adcBuf[256 * SOC_ADC_DIGI_RESULT_BYTES];

static bool IRAM_ATTR onOverflow(adc_continuous_handle_t, const adc_continuous_evt_data_t *, void *) {
  overflows++;
  return false;
}

static void setupAdc() {
  adc_continuous_handle_cfg_t hcfg = {};
  hcfg.max_store_buf_size = 4096 * N_CH;
  hcfg.conv_frame_size = sizeof(adcBuf);
  ESP_ERROR_CHECK(adc_continuous_new_handle(&hcfg, &adc));
  adc_unit_t unit;
  ESP_ERROR_CHECK(adc_continuous_io_to_channel(SOUND_PIN, &unit, &soundChannel));
  adc_digi_pattern_config_t pattern[2] = {};
  pattern[0].atten = ADC_ATTEN_DB_12;
  pattern[0].channel = soundChannel;
  pattern[0].unit = unit;
  pattern[0].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
#if VIB_PIN >= 0
  adc_unit_t vunit;
  ESP_ERROR_CHECK(adc_continuous_io_to_channel(VIB_PIN, &vunit, &vibChannel));
  pattern[1] = pattern[0];       // same unit (ADC1), attenuation and width; sampled right after sound
  pattern[1].channel = vibChannel;
#endif
  adc_continuous_config_t cfg = {};
  cfg.pattern_num = N_CH;
  cfg.adc_pattern = pattern;
  cfg.sample_freq_hz = FS_HZ * N_CH;   // total conversions/s: FS_HZ per sensor
  cfg.conv_mode = ADC_CONV_SINGLE_UNIT_1;
  cfg.format = ADC_FORMAT;
  ESP_ERROR_CHECK(adc_continuous_config(adc, &cfg));
  adc_continuous_evt_cbs_t cbs = {};
  cbs.on_pool_ovf = onOverflow;
  adc_continuous_register_event_callbacks(adc, &cbs, nullptr);
  ESP_ERROR_CHECK(adc_continuous_start(adc));
}

// ------------------------------------------------------------------ streaming
static uint16_t pkt[N_PER_PACKET];
static int pktFill = 0;
static uint16_t seq = 0;
static int64_t totalSamples = 0;
// vibration (node 1 only): own buffer, sample count and packet sequence
static uint16_t vpkt[N_PER_PACKET];
static int vpktFill = 0;
static uint16_t vseq = 0;
static int64_t vTotalSamples = 0;

// ---- sample timing: sample index -> local microseconds ----------------------------------
// A line fitted over the last ~1.5 s of (sample index, time the ADC buffer arrived), using only
// reads that actually waited for new data (backlog reads arrive too early). The fit restarts
// after an ADC overflow or a stall, so lost samples or a blocked USB never skew the timeline.
static const int TW = 64;
static int64_t twIdx[TW], twT[TW];
static int twHead = 0, twCount = 0;
static int64_t tRefIdx = 0, tRefT = 0, lastIdxEnd = -1, lastTEnd = 0;
static double tFitA = 0, dtUs = 1e6 / FS_HZ;
static uint32_t seenOverflows = 0;

static void timingReset() { twHead = 0; twCount = 0; }

static void timingAdd(int64_t idxEnd, int64_t tEnd) {
  twIdx[twHead] = idxEnd; twT[twHead] = tEnd;
  twHead = (twHead + 1) % TW;
  if (twCount < TW) twCount++;
  if (twCount < 8) return;
  int64_t x0 = twIdx[(twHead - twCount + TW) % TW], y0 = twT[(twHead - twCount + TW) % TW];
  double mx = 0, my = 0;
  for (int i = 0; i < twCount; i++) {
    int k = (twHead - twCount + i + TW) % TW;
    mx += (double)(twIdx[k] - x0); my += (double)(twT[k] - y0);
  }
  mx /= twCount; my /= twCount;
  double sxy = 0, sxx = 0;
  for (int i = 0; i < twCount; i++) {
    int k = (twHead - twCount + i + TW) % TW;
    double dx = (double)(twIdx[k] - x0) - mx, dy = (double)(twT[k] - y0) - my;
    sxy += dx * dy; sxx += dx * dx;
  }
  if (sxx <= 0) return;
  double b = sxy / sxx;
  if (b < 0.5e6 / FS_HZ || b > 2.0e6 / FS_HZ) return;   // ignore nonsense fits
  dtUs = b;
  tFitA = my - b * mx;
  tRefIdx = x0; tRefT = y0;
}

static int64_t sampleLocalTime(int64_t idx) {
  if (twCount >= 8) return tRefT + (int64_t)llround(tFitA + dtUs * (double)(idx - tRefIdx));
  return lastTEnd - (int64_t)llround((double)(lastIdxEnd - idx) * dtUs);   // until the fit is ready
}

// ch = CH_SOUND or CH_VIB. Both sensors share the sound channel's sample timing: vibration
// sample k is converted right after sound sample k, i.e. half a sample period later.
static void sendPacket(const uint16_t *buf, int64_t firstSampleIndex, uint8_t ch, uint16_t &seqRef) {
  const size_t bytes = N_PER_PACKET * sizeof(uint16_t);
#if ARDUINO_USB_MODE
  // C5 / S3 USB: only send a packet if it fits completely; otherwise skip it whole.
  // (A half-sent packet would corrupt the stream; a skipped one just shows as "dropped".)
  const int need = sizeof(StreamHeader) + bytes + 2;
  if (Serial.availableForWrite() < need) { seqRef++; return; }
#endif
  StreamHeader h;
  h.magic = 0x5AA6;
  h.node = nodeId;
  h.synced = isSynced();
  h.seq = seqRef++;
  h.fs_hz = (uint32_t)llround(1e6 / dtUs);   // measured rate (some chips don't sample exactly at FS_HZ)
  h.first_index = (uint32_t)firstSampleIndex;
  int64_t tLocal = sampleLocalTime(firstSampleIndex);
  if (ch == CH_VIB) tLocal += (int64_t)llround(dtUs / N_CH);
  h.first_t_us = toShared(tLocal);
  h.n = N_PER_PACKET;
  uint8_t ovOut = 0;
  if (ch == CH_SOUND) {                       // overflows are reported once, on the sound stream
    uint32_t ov = overflows;
    uint32_t newOv = ov - lastOverflows;
    lastOverflows = ov;
    ovOut = newOv > 255 ? 255 : newOv;
  }
  h.overflows = ovOut;
  h.reserved = ch;
  uint16_t sum = 0;
  const uint8_t *hb = (const uint8_t *)&h;
  for (size_t i = 0; i < sizeof(h); i++) sum += hb[i];
  const uint8_t *sb = (const uint8_t *)buf;
  for (size_t i = 0; i < bytes; i++) sum += sb[i];
  Serial.write(hb, sizeof(h));
  Serial.write(sb, bytes);
  Serial.write((const uint8_t *)&sum, 2);
}

static uint8_t lookupId() {
  if (FORCE_NODE_ID) return FORCE_NODE_ID;
#if CONFIG_IDF_TARGET_ESP32S2
  return 1;
#else
  uint8_t mac[6];
  WiFi.macAddress(mac);
  for (const MacId &m : MAC_TABLE)
    if (memcmp(m.mac, mac, 6) == 0) return m.id;
  return DEFAULT_ID;
#endif
}

void setup() {
#if ARDUINO_USB_MODE
  Serial.setTxBufferSize(4096); // C5 / S3 USB: room for several packets (must be before begin)
#endif
  Serial.begin(921600);
#if ARDUINO_USB_MODE
  Serial.setTxTimeoutMs(0);     // never block when no laptop is reading
#endif
  delay(800);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
#ifdef SOC_WIFI_SUPPORT_5G
  esp_wifi_set_band_mode(WIFI_BAND_MODE_2G_ONLY);
#endif
  esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  nodeId = lookupId();
  isMaster = (nodeId == 1);
  if (esp_now_init() == ESP_OK) {
    esp_now_register_recv_cb(onRecv);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BROADCAST, 6);
    peer.channel = ESPNOW_CHANNEL;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
  }
  Serial.printf("\nDNL node %u %s MAC %s pin %d vibration pin %d\n", nodeId, isMaster ? "(time master)" : "",
                WiFi.macAddress().c_str(), SOUND_PIN, VIB_PIN);
  setupAdc();
}

void loop() {
  if (isMaster) {
    static uint32_t lastBeacon = 0, beaconSeq = 0;
    if (millis() - lastBeacon >= 500) {
      lastBeacon = millis();
      Beacon b = {0xB5, beaconSeq++, esp_timer_get_time()};
      esp_now_send(BROADCAST, (const uint8_t *)&b, sizeof(b));
    }
  }

  uint32_t n = 0;
  if (adc_continuous_read(adc, adcBuf, sizeof(adcBuf), &n, 20) != ESP_OK) return;
  int64_t tEnd = esp_timer_get_time();

  // count this read's samples first, so the timing is known before packets are sent
  int ours = 0;
  int results = n / SOC_ADC_DIGI_RESULT_BYTES;
  for (int i = 0; i < results; i++) {
    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&adcBuf[i * SOC_ADC_DIGI_RESULT_BYTES];
    if (ADC_CH(p) == soundChannel) ours++;
  }
  if (ours == 0) return;
  int64_t idxEnd = totalSamples + ours - 1;

  uint32_t ov = overflows;
  bool stalled = lastIdxEnd >= 0 && (tEnd - lastTEnd) > 100000;   // >100 ms without data
  if (ov != seenOverflows || stalled) { timingReset(); seenOverflows = ov; }
  double expectedUs = ours * (1e6 / FS_HZ);
  bool waited = lastIdxEnd < 0 || (tEnd - lastTEnd) > 0.5 * expectedUs;  // not a backlog read
  lastIdxEnd = idxEnd;
  lastTEnd = tEnd;
  if (waited) timingAdd(idxEnd, tEnd);

  for (int i = 0; i < results; i++) {
    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&adcBuf[i * SOC_ADC_DIGI_RESULT_BYTES];
    uint32_t chn = ADC_CH(p);
    if (chn == (uint32_t)soundChannel) {
      pkt[pktFill++] = ADC_DATA(p);
      totalSamples++;
      if (pktFill == N_PER_PACKET) {
        sendPacket(pkt, totalSamples - N_PER_PACKET, CH_SOUND, seq);
        pktFill = 0;
      }
    } else if (VIB_PIN >= 0 && chn == (uint32_t)vibChannel) {
      // keep vibration index k paired with sound index k (the conversion right after it)
      if (vTotalSamples >= totalSamples) continue;          // lost its sound partner: skip
      if (vTotalSamples < totalSamples - 1) {               // sound got ahead (lost vib): realign
        vTotalSamples = totalSamples - 1;
        vpktFill = 0;
      }
      if (vpktFill == 0 && vTotalSamples % N_PER_PACKET != 0) continue;   // start packets aligned
      vpkt[vpktFill++] = ADC_DATA(p);
      vTotalSamples++;
      if (vpktFill == N_PER_PACKET) {
        sendPacket(vpkt, vTotalSamples - N_PER_PACKET, CH_VIB, vseq);
        vpktFill = 0;
      }
    }
  }
}
