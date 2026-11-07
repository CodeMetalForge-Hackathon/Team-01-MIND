#include <Arduino.h>
#include "esp_adc/adc_continuous.h"

// ---------- TODO(hardware): match your wiring ----------
static const int PINS[] = {9};          // ADC1 pins (GPIO1..GPIO10 on ESP32-S2). Stage 2: {1, 2}
static const uint32_t FS_HZ = 10000;    // samples per second PER CHANNEL. Try 20000 later.
// -------------------------------------------------------

static const int N_CH = sizeof(PINS) / sizeof(PINS[0]);
static const uint16_t N_FRAMES = 256;   // frames per USB packet (25.6 ms at 10 kHz)
static const uint16_t MAGIC = 0x5AA5;

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

static adc_continuous_handle_t adc = nullptr;
static volatile uint32_t overflows = 0;
static uint8_t chanMap[16];                    // ADC channel number -> our channel index (0xFF = unused)
static uint8_t raw[N_CH * 128 * SOC_ADC_DIGI_RESULT_BYTES];
static uint16_t frames[N_FRAMES][N_CH];
static uint16_t frameIdx = 0;
static int expectK = 0;                        // channel we expect next (pattern alternates strictly)
static uint32_t firstIndex = 0;
static uint16_t seq = 0;
static uint32_t lastOverflows = 0;

static bool IRAM_ATTR onOverflow(adc_continuous_handle_t, const adc_continuous_evt_data_t *, void *) {
  overflows++;
  return false;
}

static void sendPacket() {
  Header h;
  h.magic = MAGIC;
  h.seq = seq++;
  h.fs_hz = FS_HZ;
  h.first_index = firstIndex;
  h.n_frames = N_FRAMES;
  h.n_ch = N_CH;
  uint32_t ov = overflows;
  uint32_t newOv = ov - lastOverflows;
  lastOverflows = ov;
  h.flags = newOv > 255 ? 255 : (uint8_t)newOv;

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

void setup() {
  Serial.begin(921600);  // baud is ignored on native USB CDC
  memset(chanMap, 0xFF, sizeof(chanMap));

  adc_continuous_handle_cfg_t hcfg = {};
  hcfg.max_store_buf_size = 8192;
  hcfg.conv_frame_size = sizeof(raw);   // a multiple of N_CH results, so a dropped frame keeps channel order
  ESP_ERROR_CHECK(adc_continuous_new_handle(&hcfg, &adc));

  adc_digi_pattern_config_t pattern[N_CH];
  for (int i = 0; i < N_CH; i++) {
    adc_unit_t unit;
    adc_channel_t ch;
    ESP_ERROR_CHECK(adc_continuous_io_to_channel(PINS[i], &unit, &ch));
    pattern[i].atten = ADC_ATTEN_DB_11;          // full ~0-3.1 V range (named DB_12 in newer IDF)
    pattern[i].channel = ch;
    pattern[i].unit = unit;                      // must be ADC1: use GPIO1..GPIO10
    pattern[i].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
    chanMap[ch] = i;
  }

  adc_continuous_config_t cfg = {};
  cfg.pattern_num = N_CH;
  cfg.adc_pattern = pattern;
  cfg.sample_freq_hz = FS_HZ * N_CH;              // total conversion rate across channels
  cfg.conv_mode = ADC_CONV_SINGLE_UNIT_1;
  cfg.format = ADC_DIGI_OUTPUT_FORMAT_TYPE1;      // ESP32 and ESP32-S2 use TYPE1
  ESP_ERROR_CHECK(adc_continuous_config(adc, &cfg));

  adc_continuous_evt_cbs_t cbs = {};
  cbs.on_pool_ovf = onOverflow;
  ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(adc, &cbs, nullptr));
  ESP_ERROR_CHECK(adc_continuous_start(adc));
}

void loop() {
  uint32_t n = 0;
  if (adc_continuous_read(adc, raw, sizeof(raw), &n, 100) != ESP_OK) return;

  for (uint32_t i = 0; i + SOC_ADC_DIGI_RESULT_BYTES <= n; i += SOC_ADC_DIGI_RESULT_BYTES) {
    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&raw[i];
    uint32_t ch = p->type1.channel;
    if (ch >= 16 || chanMap[ch] == 0xFF) continue;
    int k = chanMap[ch];

    if (k != expectK) {          // lost a result: drop the partial frame and wait for channel 0
      expectK = 0;
      if (k != 0) continue;
    }
    frames[frameIdx][k] = p->type1.data;
    expectK = (k + 1) % N_CH;
    if (k == N_CH - 1 && ++frameIdx == N_FRAMES) {
      sendPacket();
      frameIdx = 0;
    }
  }
}
