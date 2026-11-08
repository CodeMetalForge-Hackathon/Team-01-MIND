// Raw I2S check for two ICS-43434 mics. Prints left/right words as hex.
#include <Arduino.h>
#include "driver/i2s_std.h"

#if CONFIG_IDF_TARGET_ESP32S2        // Adafruit Feather ESP32-S2
static const int PIN_BCLK = 5, PIN_WS = 6, PIN_DATA = 11;
#elif CONFIG_IDF_TARGET_ESP32C5      // Seeed XIAO ESP32-C5: D8, D9, D10
static const int PIN_BCLK = 8, PIN_WS = 9, PIN_DATA = 10;
#else
#error "Set the pins for this board"
#endif

static i2s_chan_handle_t rx_chan;
static int32_t buf[16];  // 8 frames: left, right, left, right...

void setup() {
  Serial.begin(115200);
  i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  i2s_new_channel(&cc, NULL, &rx_chan);
  i2s_std_config_t sc = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(48000),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)PIN_BCLK,
      .ws   = (gpio_num_t)PIN_WS,
      .dout = I2S_GPIO_UNUSED,
      .din  = (gpio_num_t)PIN_DATA,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
  i2s_channel_init_std_mode(rx_chan, &sc);
  i2s_channel_enable(rx_chan);
}

void loop() {
  size_t n;
  i2s_channel_read(rx_chan, buf, sizeof(buf), &n, portMAX_DELAY);
  for (int i = 0; i < 16; i += 2) {
    Serial.printf("L %08lX   R %08lX\n", (unsigned long)buf[i], (unsigned long)buf[i + 1]);
  }
  Serial.println("---");
  delay(300);
}