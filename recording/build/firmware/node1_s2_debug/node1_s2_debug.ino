// Node 1 debug: Adafruit Feather ESP32-S2, Grove sound on pin 9.
// Text-only debug: run all three side by side, one Serial Monitor each.
// Wiring (XIAO ESP32-C5): yellow -> D0, red -> 3V3, black -> GND, white unused.
// Open the Serial Monitor at 115200. Every 2 s it prints signal stats and a verdict;
// every detected sound prints its features and a guess (knock, voice, ...).
// Set PLOTTER = true and use Tools > Serial Plotter to watch the signal live.
#include <Arduino.h>

const int SENSOR_PIN = 9;    // Feather ESP32-S2: pin 9 (ADC1)
const char *NODE_LABEL = "NODE 1 (S2)";

const bool PLOTTER = false;          // true: stream raw/envelope for the Serial Plotter
const uint32_t FS_HZ = 4000;         // samples per second
const uint32_t PERIOD_US = 1000000UL / FS_HZ;

// detector settings (same idea as the node firmware)
const float K_TRIGGER = 4.0f;        // trigger at 4x the noise floor
const float ABS_MIN = 15.0f;         // ...and at least 15 ADC counts
const float LOW_FRAC = 0.3f;         // walk back to 30% of the trigger level for the onset
const uint32_t QUIET_END_US = 150000;  // sound ends after 150 ms of quiet
const uint32_t MAX_SOUND_US = 2000000; // or after 2 s

float baseline = -1, env = 0, envSlow = 0, noiseFloor = 5;
const float A_FAST = 1.0f - expf(-1.0f / (FS_HZ * 0.0007f));  // ~0.7 ms
const float A_SLOW = 1.0f - expf(-1.0f / (FS_HZ * 0.015f));   // ~15 ms
const float A_FLOOR = 1.0f - expf(-1.0f / (FS_HZ * 1.0f));    // ~1 s
const float A_BASE = 1.0f - expf(-1.0f / (FS_HZ * 1.0f));     // ~1 s

const int HIST = 24;                 // ~6 ms of history for the walk-back
float histEnv[HIST];
uint32_t histT[HIST];
int histPos = 0;

bool active = false;
uint32_t onsetT = 0, peakT = 0, lastLoudT = 0, refractoryUntil = 0;
float peakEnv = 0, thrAtOnset = 0, floorAtOnset = 0;
float slowTrack[220];
int slowN = 0, decim = 0;
int eventCount = 0;

// 2-second statistics
uint32_t statStart = 0;
int rawMin = 4095, rawMax = 0;
double sum = 0, sumSq = 0;
uint32_t nStat = 0;

const char *guess(float durMs, float riseMs, float modHz) {
  if (durMs >= 1500 && modHz < 1.5f) return "steady noise (not a person)";
  if (riseMs <= 6 && durMs <= 150) return "knock / tap";
  if (riseMs <= 6) return "impact (slam, drop)";
  if (durMs >= 150 && durMs <= 2500 && modHz >= 2.5f && modHz <= 9) return "voice";
  if (durMs >= 80 && durMs <= 700 && modHz < 2.5f) return "cough";
  return "other";
}

int countPeaks(const float *v, int n) {
  float mx = 0;
  for (int i = 0; i < n; i++) mx = max(mx, v[i]);
  if (mx <= 0) return 0;
  float hyst = 0.25f * mx, ext = v[0];
  bool rising = true;
  int peaks = 0;
  for (int i = 1; i < n; i++) {
    if (rising) {
      if (v[i] > ext) ext = v[i];
      else if (ext - v[i] > hyst) { if (ext > 0.35f * mx) peaks++; rising = false; ext = v[i]; }
    } else {
      if (v[i] < ext) ext = v[i];
      else if (v[i] - ext > hyst) { rising = true; ext = v[i]; }
    }
  }
  return peaks;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  analogReadResolution(12);
  analogSetPinAttenuation(SENSOR_PIN, ADC_11db);
  if (!PLOTTER) {
    Serial.printf("\n%s: Grove sound test on pin %d at %lu Hz\n", NODE_LABEL, SENSOR_PIN, (unsigned long)FS_HZ);
    Serial.println("Stay quiet for 2 s, then knock, talk, cough, clap...");
  }
  statStart = micros();
}

void loop() {
  static uint32_t next = micros();
  while ((int32_t)(micros() - next) < 0) {}
  next += PERIOD_US;
  uint32_t t = micros();
  int raw = analogRead(SENSOR_PIN);

  // statistics
  rawMin = min(rawMin, raw);
  rawMax = max(rawMax, raw);
  sum += raw;
  sumSq += (double)raw * raw;
  nStat++;

  // envelope
  if (baseline < 0) baseline = raw;
  float a = fabsf(raw - baseline);
  env += A_FAST * (a - env);
  envSlow += A_SLOW * (env - envSlow);
  histEnv[histPos] = env;
  histT[histPos] = t;
  histPos = (histPos + 1) % HIST;

  if (!active) {
    baseline += A_BASE * (raw - baseline);
    float thr = max(K_TRIGGER * noiseFloor, ABS_MIN);
    if ((int32_t)(t - refractoryUntil) >= 0 && env > thr) {
      onsetT = t;
      for (int k = 1; k < HIST; k++) {             // walk back to the start of the rise
        int idx = (histPos - 1 - k + HIST) % HIST;
        if (histEnv[idx] < LOW_FRAC * thr) break;
        onsetT = histT[idx];
      }
      active = true;
      peakEnv = env; peakT = t; lastLoudT = t;
      thrAtOnset = thr; floorAtOnset = noiseFloor;
      slowN = 0; decim = 0;
    } else if (env < thr) {
      noiseFloor += A_FLOOR * (env - noiseFloor);
      if (noiseFloor < 1) noiseFloor = 1;
    }
  } else {
    if (env > peakEnv) { peakEnv = env; peakT = t; }
    if (envSlow > 1.8f * floorAtOnset) lastLoudT = t;
    if (++decim >= (int)(FS_HZ / 100)) { decim = 0; if (slowN < 220) slowTrack[slowN++] = envSlow; }
    bool ended = (t - lastLoudT) > QUIET_END_US || (t - onsetT) > MAX_SOUND_US;
    if (ended) {
      active = false;
      refractoryUntil = t + 120000;
      float durMs = (lastLoudT - onsetT) / 1000.0f;
      float riseMs = (peakT - onsetT) / 1000.0f;
      int peaks = durMs > 250 ? countPeaks(slowTrack, slowN) : 0;
      float modHz = peaks >= 2 ? peaks / (durMs / 1000.0f) : 0;   // one burst = no rhythm
      eventCount++;
      if (!PLOTTER) {
        Serial.printf("%s SOUND #%d  peak %.0f  floor %.1f  snr %.0fx  dur %.0f ms  rise %.1f ms  mod %.1f Hz  -> %s\n",
                      NODE_LABEL, eventCount, peakEnv, floorAtOnset, peakEnv / floorAtOnset, durMs, riseMs, modHz,
                      guess(durMs, riseMs, modHz));
      }
    }
  }

  if (PLOTTER) {
    static uint32_t lastPlot = 0;
    if (t - lastPlot >= 5000) {
      lastPlot = t;
      Serial.printf("raw:%d,envelope:%.1f,threshold:%.1f\n", raw, env,
                    max(K_TRIGGER * noiseFloor, ABS_MIN));
    }
    return;
  }

  if (t - statStart >= 2000000) {
    double mean = sum / nStat;
    double sd = sqrt(max(0.0, sumSq / nStat - mean * mean));
    Serial.printf("%s [stats] raw min %d  max %d  mean %.0f  spread %.1f  | noise floor %.1f  | sounds so far %d\n",
                  NODE_LABEL, rawMin, rawMax, mean, sd, noiseFloor, eventCount);
    if (rawMax <= 5) Serial.println("  ! reads ~0 V: check yellow wire on the right pin, red on 3V3, black on GND");
    else if (rawMin >= 4090) Serial.println("  ! stuck at maximum: sensor on 5V or signal shorted to 3V3");
    else if (rawMax - rawMin < 3) Serial.println("  ! flat: wrong pin, or the sensor has no power");
    else Serial.println("  ok: signal is alive");
    statStart = t;
    rawMin = 4095; rawMax = 0; sum = 0; sumSq = 0; nStat = 0;
  }
}
