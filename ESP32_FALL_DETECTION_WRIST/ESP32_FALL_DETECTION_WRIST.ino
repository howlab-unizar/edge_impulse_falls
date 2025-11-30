// Edge Impulse Fall Detection (Wrist)
// Authors: Pablo Sanz, Alberto Gascon, Roberto Casas.
// Institution: University of Zaragoza - Aragon Institute of Engineering Research - Spain
// Dependencies:
// - GFX Library for Arduino (Moon On Our Nation) v1.4.9
// - SensorLib (Lewis He) v0.1.6
// - LVGL (kisvegabor) v8.4.0
// Import ei-falldetection_wrist-arduino-1.0.4
#include <Arduino.h>
#include <Wire.h>
#include <FallDetection_Wrist_inferencing.h>
#include "SensorQMI8658.hpp"
#include "pin_config.h"
#include <Arduino_GFX_Library.h>
#include <cstring>
#include <cstdio>
#include <memory>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "Arduino_DriveBus_Library.h"

struct ButtonArea;

// ==================== Hardware configuration (I2C) ====================
#define IIC_SDA 11
#define IIC_SCL 10

// ==================== Signal conditioning ====================
#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE_G 4.0f
static constexpr bool OUTPUT_UNITS_MS2 = true;

// ==================== Inference parameters ====================
// Wrist-tuned thresholds: higher EMA smoothing and stricter post-impact quietness to avoid arm swings.
static constexpr float FALL_EMA_ALPHA = 0.25f;  // exponential smoothing for p(FALL)
static constexpr float FALL_H_ARM     = 0.90f;  // arm episode when EMA crosses this level
static constexpr float FALL_H_HOLD    = 0.70f;  // keep episode armed while EMA stays above
static constexpr uint8_t FALL_HOLD_M  = 3;      // windows evaluated during hold phase
static constexpr uint8_t FALL_HOLD_N  = 2;      // minimum windows >= hold threshold
static constexpr float FALL_P_HIGH    = 0.95f;  // raw prob to consider an impact
static constexpr float FALL_P_LOW     = 0.30f;  // max raw prob after impact to confirm fall
static constexpr uint8_t FALL_POST_M  = 4;      // windows observed after impact
static constexpr uint8_t FALL_POST_N  = 3;      // minimum windows <= FALL_P_LOW in that group
static constexpr uint8_t HELP_HIGH_MIN = 2;     // ask user help if pFALL stays high this many times
static constexpr float FALL_THRESHOLD = FALL_P_HIGH; // kept for legacy help logic
static constexpr size_t AXES_PER_SAMPLE = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME; // 3 axes
static constexpr size_t WINDOW_SAMPLES = EI_CLASSIFIER_RAW_SAMPLE_COUNT;     // 497 samples
// Analysis frequency depending on state
static constexpr size_t SLICE_SAMPLES_IDLE   = WINDOW_SAMPLES / 8;   // ~8 inferences / 10 s
static constexpr size_t SLICE_SAMPLES_VERIFY = WINDOW_SAMPLES / 10;  // ~10 inferences / 10 s during episodes
static size_t current_slice_target = SLICE_SAMPLES_IDLE;
static size_t slice_sample_accum   = 0;
static constexpr size_t WINDOW_STRIDE = AXES_PER_SAMPLE;                     // 3 floats per sample
static constexpr uint8_t HISTORY_SIZE = 6;                                    // displayed rows
static constexpr uint32_t FALL_REFRACTORY_MS = 15000;                         // wait time after alert
static constexpr float QUIET_STD_THRESHOLD = 1.0f;                            // std(|a|) max for quietness
static_assert(FALL_HOLD_N <= FALL_HOLD_M, "FALL_HOLD_N must be <= FALL_HOLD_M");
static_assert(FALL_POST_N <= FALL_POST_M, "FALL_POST_N must be <= FALL_POST_M");

// ==================== Power / Alerts ====================
static constexpr int POWER_HOLD_PIN = 35;
static constexpr int POWER_BOOST_PIN = 38;
static constexpr uint32_t ALERT_DURATION_MS = 5000;
static constexpr int VIBRATION_PIN = -1; // change to the real GPIO if a motor is present
static constexpr uint16_t COLOR_NORMAL = 0x0000; // black
static constexpr uint16_t COLOR_ALERT  = 0xF800; // red
static constexpr uint16_t COLOR_HIGHLIGHT = 0x07E0; // green for threshold crossings

// ==================== IMU and buffers ====================
SensorQMI8658 qmi;
IMUdata acc = {};
static float inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {0};
static size_t samples_collected = 0;        // total valid samples in buffer
static float fall_history_raw[HISTORY_SIZE] = {0};
static float fall_history_ema_vals[HISTORY_SIZE] = {0};
static uint8_t history_index = 0;
static uint8_t history_count = 0;

// ==================== Control ====================
static const bool debug_nn = false;
static bool alert_active = false;
static uint32_t alert_start_ms = 0;
static float fall_ema = 0.0f;
static bool fall_ema_initialized = false;
static uint32_t refractory_until_ms = 0;
static float last_slice_std = 0.0f;
static bool last_slice_quiet = false;
static uint32_t help_prompt_start_ms = 0;
static uint32_t help_prompt_suppress_until_ms = 0;
static constexpr uint32_t HELP_TIMEOUT_MS = 5000;
static volatile bool help_answer_yes = false;
static volatile bool help_answer_no  = false;

enum class FallDecisionState : uint8_t { Idle, AwaitHold, AwaitLow, AwaitHelp };
static FallDecisionState fall_state = FallDecisionState::Idle;
static uint8_t hold_windows_seen = 0;
static uint8_t hold_windows_hits = 0;
static uint8_t post_windows_seen = 0;
static uint8_t post_low_hits = 0;
static uint8_t post_quiet_hits = 0;
static uint8_t post_high_hits = 0;
static bool fall_event_pending = false;

// ==================== BLE / Streaming ====================
#define FALL_DEVICE_BASE_NAME "ESP32S3_FALL"
#define FALL_SERVICE_UUID     "5b3bda10-6f96-4e92-a21b-7adf92a12c30"
#define FALL_DATA_CHAR_UUID   "5b3bda11-6f96-4e92-a21b-7adf92a12c30"

static BLEServer *fall_ble_server = nullptr;
static BLECharacteristic *fall_ble_data_char = nullptr;
static BLE2902 *fall_ble_ccc = nullptr;
static bool fall_ble_connected = false;
static uint32_t fall_ble_seq = 0;

class FallServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *s) override {
    (void)s;
    fall_ble_connected = true;
  }

  void onDisconnect(BLEServer *s) override {
    (void)s;
    fall_ble_connected = false;
    BLEDevice::startAdvertising();
  }
};

static inline bool fall_notifications_ready() {
  if (!fall_ble_connected || fall_ble_data_char == nullptr) {
    return false;
  }
  if (!fall_ble_ccc) {
    return true;
  }
  return fall_ble_ccc->getNotifications();
}

static void init_fall_ble() {
  String name = String(FALL_DEVICE_BASE_NAME) + "_" + String((uint32_t)ESP.getEfuseMac(), HEX);
  BLEDevice::init(name.c_str());
  BLEDevice::setMTU(185);

  fall_ble_server = BLEDevice::createServer();
  fall_ble_server->setCallbacks(new FallServerCallbacks());

  BLEService *svc = fall_ble_server->createService(FALL_SERVICE_UUID);
  fall_ble_data_char = svc->createCharacteristic(
    FALL_DATA_CHAR_UUID,
    BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ
  );
  fall_ble_ccc = new BLE2902();
  fall_ble_ccc->setNotifications(false);
  fall_ble_data_char->addDescriptor(fall_ble_ccc);
  fall_ble_data_char->setValue("FALL_READY");

  svc->start();

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  if (adv != nullptr) {
    adv->addServiceUUID(FALL_SERVICE_UUID);
    adv->setScanResponse(true);
    adv->setMinPreferred(0x06);
    adv->setMaxPreferred(0x12);
  }

  BLEDevice::startAdvertising();
  Serial.printf("[BLE] Advertised %s (%s)\\n", name.c_str(), FALL_SERVICE_UUID);
}

static void notify_fall_metrics(float fall_score, float ema_score) {
  if (!fall_notifications_ready()) {
    return;
  }

  char payload[96];
  const uint32_t seq = fall_ble_seq++;
  const uint32_t now_ms = millis();
  const int written = snprintf(
    payload,
    sizeof(payload),
    "SEQ:%lu,T:%lu,PFALL:%.4f,EMA:%.4f,STATE:%u,STD:%.3f,QUIET:%u,FALL_EVT:%u",
    (unsigned long)seq,
    (unsigned long)now_ms,
    (double)fall_score,
    (double)ema_score,
    static_cast<unsigned>(fall_state),
    (double)last_slice_std,
    last_slice_quiet ? 1U : 0U,
    fall_event_pending ? 1U : 0U
  );
  if (written <= 0) {
    return;
  }
  const size_t used = (written < static_cast<int>(sizeof(payload))) ? static_cast<size_t>(written) : sizeof(payload);
  fall_ble_data_char->setValue((uint8_t*)payload, used);
  fall_ble_data_char->notify();
  fall_event_pending = false;
}

// ==================== Display ====================
Arduino_DataBus *bus = new Arduino_ESP32SPI(LCD_DC, LCD_CS, LCD_SCK, LCD_MOSI);
Arduino_GFX *gfx     = new Arduino_ST7789(bus, LCD_RST, 0, true /*IPS*/, LCD_WIDTH, LCD_HEIGHT, 0, 20, 0, 0);

struct ButtonArea {
  int16_t x;
  int16_t y;
  int16_t w;
  int16_t h;
};

static ButtonArea help_btn_yes = {};
static ButtonArea help_btn_no  = {};

static std::shared_ptr<Arduino_IIC_DriveBus> help_touch_bus;
static std::unique_ptr<Arduino_IIC> help_touch_panel;
static bool help_touch_ready = false;
static bool help_touch_pressed = false;

static void help_touch_isr(void);
static void init_help_touch();

// ==================== Utilities ====================
static inline float signf(float x) { return (x >= 0.0f) ? 1.0f : -1.0f; }

static void render_history();

static void push_history(float raw_score, float ema_score) {
  fall_history_raw[history_index] = raw_score;
  fall_history_ema_vals[history_index] = ema_score;
  history_index = (history_index + 1) % HISTORY_SIZE;
  if (history_count < HISTORY_SIZE) {
    history_count++;
  }
  render_history();
}

static void enter_alert_mode() {
  alert_active = true;
  alert_start_ms = millis();

  digitalWrite(LCD_BL, HIGH);
  gfx->fillScreen(COLOR_ALERT);
  if (VIBRATION_PIN >= 0) {
    digitalWrite(VIBRATION_PIN, HIGH);
  }
}

static void maintain_alert_state() {
  if (!alert_active) {
    return;
  }
  if (millis() - alert_start_ms >= ALERT_DURATION_MS) {
    alert_active = false;
    gfx->fillScreen(COLOR_NORMAL);
    if (VIBRATION_PIN >= 0) {
      digitalWrite(VIBRATION_PIN, LOW);
    }
    render_history();
  }
}

static void render_history() {
  if (alert_active) {
    return; // mantener pantalla en red durante la alerta
  }

  gfx->fillScreen(COLOR_NORMAL);
  gfx->setTextSize(2);
  gfx->setTextColor(0xFFFF, COLOR_NORMAL);
  gfx->setCursor(4, 6);
  gfx->print("#");
  gfx->setCursor(60, 6);
  gfx->print("pFALL");
  gfx->setCursor(150, 6);
  gfx->print("EMA");

  const int y_offset = 32;
  if (history_count == 0) {
    gfx->setCursor(4, y_offset);
    gfx->print("waiting for data...");
    gfx->setTextColor(0xFFFF, COLOR_NORMAL);
    return;
  }

  for (uint8_t i = 0; i < history_count; ++i) {
    uint8_t idx = (history_index + HISTORY_SIZE - 1 - i) % HISTORY_SIZE;
    const float raw_score = fall_history_raw[idx];
    const float ema_score = fall_history_ema_vals[idx];
    const int row_y = y_offset + static_cast<int>(i) * 22;

    gfx->setTextColor(0xFFFF, COLOR_NORMAL);
    gfx->setCursor(4, row_y);
    gfx->print(String(i + 1) + ":");

    gfx->setCursor(60, row_y);
    gfx->setTextColor(raw_score >= FALL_P_HIGH ? COLOR_HIGHLIGHT : 0xFFFF, COLOR_NORMAL);
    gfx->print(String(raw_score * 100.0f, 1) + "%");

    gfx->setCursor(150, row_y);
    gfx->setTextColor(ema_score >= FALL_H_ARM ? COLOR_HIGHLIGHT : 0xFFFF, COLOR_NORMAL);
    gfx->print(String(ema_score * 100.0f, 1) + "%");
  }

  gfx->setTextColor(0xFFFF, COLOR_NORMAL);
}

static inline bool help_point_in_button(int16_t x, int16_t y, const ButtonArea &btn) {
  return (x >= btn.x) && (x < (btn.x + btn.w)) && (y >= btn.y) && (y < (btn.y + btn.h));
}

static void help_draw_centered_text(const char *text, int16_t cx, int16_t cy, uint8_t size) {
  const int16_t width  = (int16_t)strlen(text) * 6 * size;
  const int16_t height = 8 * size;
  gfx->setTextSize(size);
  gfx->setTextColor(0xFFFF, COLOR_NORMAL);
  gfx->setCursor(cx - width / 2, cy - height / 2);
  gfx->print(text);
}

static void draw_help_button(const ButtonArea &btn, uint16_t color, const char *label) {
  const uint8_t radius = 18;
  gfx->fillRoundRect(btn.x, btn.y, btn.w, btn.h, radius, color);
  gfx->drawRoundRect(btn.x, btn.y, btn.w, btn.h, radius, 0xFFFF);
  help_draw_centered_text(label, btn.x + btn.w / 2, btn.y + btn.h / 2, 6);
}

// Direct coordinate read via Wire as a last resort if the driver fails.
static bool help_touch_read_fallback(int16_t &tx, int16_t &ty) {
  Wire.begin(IIC_SDA, IIC_SCL);
  Wire.beginTransmission(CST816T_DEVICE_ADDRESS);
  Wire.write((uint8_t)CST816x_RD_DEVICE_FINGERNUM);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  if (Wire.requestFrom((uint8_t)CST816T_DEVICE_ADDRESS, (uint8_t)6) != 6) {
    return false;
  }
  uint8_t fingers = Wire.read();
  uint8_t xh = Wire.read();
  uint8_t xl = Wire.read();
  uint8_t yh = Wire.read();
  uint8_t yl = Wire.read();
  (void)Wire.read(); // dummy

  if ((fingers & 0x0F) == 0) {
    return false;
  }

  const int16_t rx = ((xh & 0x0F) << 8) | xl;
  const int16_t ry = ((yh & 0x0F) << 8) | yl;

  tx = (int16_t)constrain(rx, 0, (int32_t)LCD_WIDTH - 1);
  ty = (int16_t)constrain(ry, 0, (int32_t)LCD_HEIGHT - 1);
  Serial.printf("[Touch][raw] fingers=%u raw=(%d,%d) clip=(%d,%d)\n",
    (unsigned)(fingers & 0x0F), (int)rx, (int)ry, (int)tx, (int)ty);
  return true;
}

static void show_help_prompt() {
  alert_active = false;
  help_answer_yes = false;
  help_answer_no = false;
  help_prompt_start_ms = millis();
  help_touch_pressed = false;
  init_help_touch();

  gfx->fillScreen(COLOR_NORMAL);
  help_draw_centered_text("Request help", LCD_WIDTH / 2, 36, 2);

  const int16_t margin = 12;
  const int16_t top = 80;
  const int16_t button_height = LCD_HEIGHT - top - margin;
  const int16_t button_width = (LCD_WIDTH - (margin * 3)) / 2;

  help_btn_yes = { margin, top, button_width, button_height };
  help_btn_no  = { margin * 2 + button_width, top, button_width, button_height };

  draw_help_button(help_btn_yes, COLOR_HIGHLIGHT, "YES");
  draw_help_button(help_btn_no, COLOR_ALERT, "NO");

  if (help_touch_panel) {
    help_touch_panel->IIC_Read_Device_Value(
      help_touch_panel->Arduino_IIC_Touch::Value_Information::TOUCH_FINGER_NUMBER);
    help_touch_panel->IIC_Interrupt_Flag = false;
  }
}

static void reset_fall_counters() {
  hold_windows_seen = 0;
  hold_windows_hits = 0;
  post_windows_seen = 0;
  post_low_hits = 0;
  post_quiet_hits = 0;
  post_high_hits = 0;
}

static void help_touch_isr(void) {
  if (help_touch_panel) {
    help_touch_panel->IIC_Interrupt_Flag = true;
  }
}

static void init_help_touch() {
  help_touch_ready = false;
  help_touch_pressed = false;

  // Align pin setup with the test sketch: INT pull-up and RST pulse if present.
  if (TP_INT >= 0) {
    pinMode(TP_INT, INPUT_PULLUP);
  }
  if (TP_RST >= 0) {
    pinMode(TP_RST, OUTPUT);
    digitalWrite(TP_RST, LOW);
    delay(5);
    digitalWrite(TP_RST, HIGH);
    delay(50);
  }

  if (!help_touch_bus) {
    help_touch_bus = std::make_shared<Arduino_HWIIC>(IIC_SDA, IIC_SCL, &Wire);
  }

  help_touch_panel.reset(new Arduino_CST816x(help_touch_bus, CST816T_DEVICE_ADDRESS, TP_RST, TP_INT, help_touch_isr));
  if (help_touch_panel && help_touch_panel->begin()) {
    help_touch_ready = true;
    Serial.println("[Touch] Touch panel CST816T initialized");
  } else {
    Serial.println("[Touch][WARN] Could not start the touch panel (timeout only)");
  }
}

static void poll_help_prompt_inputs() {
  if (fall_state != FallDecisionState::AwaitHelp) {
    help_touch_pressed = false;
    return;
  }
  int16_t tx = 0, ty = 0;
  bool got_point = false;

  if (help_touch_ready && help_touch_panel) {
    int32_t fingers = help_touch_panel->IIC_Read_Device_Value(
      help_touch_panel->Arduino_IIC_Touch::Value_Information::TOUCH_FINGER_NUMBER);

    // Debug: log detections while in the help screen.
    static int32_t last_fingers = -1;
    if (fingers != last_fingers) {
      Serial.printf("[Touch] fingers=%ld (pressed=%d)\n", (long)fingers, help_touch_pressed ? 1 : 0);
      last_fingers = fingers;
    }

    if (fingers > 0) {
      const int32_t raw_x = help_touch_panel->IIC_Read_Device_Value(
        help_touch_panel->Arduino_IIC_Touch::Value_Information::TOUCH_COORDINATE_X);
      const int32_t raw_y = help_touch_panel->IIC_Read_Device_Value(
        help_touch_panel->Arduino_IIC_Touch::Value_Information::TOUCH_COORDINATE_Y);

      tx = (int16_t)constrain(raw_x, 0, (int32_t)LCD_WIDTH - 1);
      ty = (int16_t)constrain(raw_y, 0, (int32_t)LCD_HEIGHT - 1);
      Serial.printf("[Touch] raw=(%ld,%ld) clip=(%d,%d)\n", (long)raw_x, (long)raw_y, (int)tx, (int)ty);
      got_point = true;
    }
  }

  //   // If the driver did not return a point, try a direct Wire read.
  if (!got_point) {
    static uint32_t last_retry_ms = 0;
    const uint32_t now_ms = millis();
    if (now_ms - last_retry_ms > 500) {
      Serial.println("[Touch] Reinit touch panel (not ready)");
      init_help_touch();
      last_retry_ms = now_ms;
    }

    if (!help_touch_read_fallback(tx, ty)) {
      help_touch_pressed = false;
      return;
    }
  }

  if (help_touch_pressed) {
    return;
  }

  if (help_point_in_button(tx, ty, help_btn_yes)) {
    help_answer_yes = true;
    help_touch_pressed = true;
    Serial.println("[Touch] YES hit");
  } else if (help_point_in_button(tx, ty, help_btn_no)) {
    help_answer_no = true;
    help_touch_pressed = true;
    Serial.println("[Touch] NO hit");
  }
}

static void resolve_help_prompt(bool needs_help, const char *log_msg) {
  if (log_msg) {
    Serial.println(log_msg);
  }
  if (needs_help) {
    enter_alert_mode();
    fall_event_pending = true;
  } else {
    render_history();
    help_prompt_suppress_until_ms = millis() + HELP_TIMEOUT_MS;
  }
  fall_state = FallDecisionState::Idle;
  reset_fall_counters();
  current_slice_target = SLICE_SAMPLES_IDLE;
  slice_sample_accum = 0;
  help_answer_yes = false;
  help_answer_no = false;
  help_touch_pressed = false;
}

static void update_help_prompt_state() {
  if (fall_state != FallDecisionState::AwaitHelp) {
    return;
  }

  poll_help_prompt_inputs();

  // Resolver en cuanto haya respuesta, sin esperar al timeout.
  if (help_answer_yes) {
    resolve_help_prompt(true, "[EI] User confirms help is needed");
    return;
  }
  if (help_answer_no) {
    resolve_help_prompt(false, "[EI] User dismisses the fall");
    return;
  }

  const uint32_t now_ms = millis();
  const bool timeout_elapsed = (now_ms - help_prompt_start_ms) >= HELP_TIMEOUT_MS;

  if (timeout_elapsed) {
    resolve_help_prompt(true, "[EI] No response after 5 s, requesting help automatically");
  }
}

// // Blocking loop while we wait for the touch-screen answer.
static void await_help_blocking() {
  // Reinit to ensure touch is alive for this session.
  init_help_touch();
  help_touch_pressed = false;
  help_prompt_start_ms = millis();
  help_answer_yes = false;
  help_answer_no = false;

  while (fall_state == FallDecisionState::AwaitHelp) {
    update_help_prompt_state();
    maintain_alert_state();
    delay(5);
  }
}

static float compute_recent_slice_stddev() {
  if (samples_collected < WINDOW_SAMPLES) {
    return 0.0f;
  }

  const size_t slice_span = (current_slice_target == 0) ? SLICE_SAMPLES_IDLE : current_slice_target;
  const size_t base_index = (WINDOW_SAMPLES - slice_span) * WINDOW_STRIDE;
  const float *ptr = inference_buffer + base_index;

  float sum = 0.0f;
  float sum_sq = 0.0f;

  for (size_t i = 0; i < slice_span; i++) {
    const float ax = ptr[i * WINDOW_STRIDE + 0];
    const float ay = ptr[i * WINDOW_STRIDE + 1];
    const float az = ptr[i * WINDOW_STRIDE + 2];
    const float mag = sqrtf(ax * ax + ay * ay + az * az);
    sum += mag;
    sum_sq += mag * mag;
  }

  const float mean = sum / static_cast<float>(slice_span);
  float variance = (sum_sq / static_cast<float>(slice_span)) - (mean * mean);
  if (variance < 0.0f) {
    variance = 0.0f;
  }
  return sqrtf(variance);
}

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  delay(100); // allows boot without USB connected
  Serial.println("\n[EI] Continuous fall detection - ESP32-S3");

  current_slice_target = SLICE_SAMPLES_IDLE;
  slice_sample_accum = 0;

  Wire.begin(IIC_SDA, IIC_SCL);
  if (!qmi.begin(Wire, QMI8658_L_SLAVE_ADDRESS, IIC_SDA, IIC_SCL)) {
    Serial.println("[EI][ERR] IMU not found (QMI8658)");
    while (1) { delay(1000); }
  }

  qmi.configAccelerometer(
    SensorQMI8658::ACC_RANGE_4G,
    SensorQMI8658::ACC_ODR_1000Hz,
    SensorQMI8658::LPF_MODE_0,
    true
  );
  qmi.enableAccelerometer();

  pinMode(POWER_HOLD_PIN, OUTPUT);
  digitalWrite(POWER_HOLD_PIN, HIGH);
  pinMode(POWER_BOOST_PIN, OUTPUT);
  digitalWrite(POWER_BOOST_PIN, HIGH);

  pinMode(LCD_BL, OUTPUT);
  digitalWrite(LCD_BL, HIGH);
  gfx->begin();
  gfx->fillScreen(COLOR_NORMAL);
  gfx->setTextWrap(false);
  gfx->setTextSize(2);
  gfx->setTextColor(0xFFFF, COLOR_NORMAL);

  if (VIBRATION_PIN >= 0) {
    pinMode(VIBRATION_PIN, OUTPUT);
    digitalWrite(VIBRATION_PIN, LOW);
  }

  Serial.printf("[EI] Model: %s | Fusion: %s\\n",
    EI_CLASSIFIER_PROJECT_NAME, EI_CLASSIFIER_FUSION_AXES_STRING);
  Serial.printf("[EI] Window: %u samples @ %u Hz | Slice Idle: %u | Slice Verify: %u\\n",
    (unsigned)WINDOW_SAMPLES,
    (unsigned)EI_CLASSIFIER_FREQUENCY,
    (unsigned)SLICE_SAMPLES_IDLE,
    (unsigned)SLICE_SAMPLES_VERIFY);
  Serial.printf("[EI] EMA alpha=%.2f | Arm >= %.2f | Hold >= %.2f (%u/%u)\n",
    (double)FALL_EMA_ALPHA, (double)FALL_H_ARM, (double)FALL_H_HOLD,
    (unsigned)FALL_HOLD_N, (unsigned)FALL_HOLD_M);
  Serial.printf("[EI] Post impact: raw>=%.2f -> raw<=%.2f in %u/%u | Quiet std<=%.2f m/s^2\n",
    (double)FALL_P_HIGH, (double)FALL_P_LOW,
    (unsigned)FALL_POST_N, (unsigned)FALL_POST_M,
    (double)QUIET_STD_THRESHOLD);
  Serial.printf("[EI] Refractory: %lu ms\n", (unsigned long)FALL_REFRACTORY_MS);

  init_fall_ble();
  init_help_touch();
  render_history();
}

// ==================== Bucle principal ====================
void loop() {
  //   // While waiting for the help screen response, stop IMU/inference.
  if (fall_state == FallDecisionState::AwaitHelp) {
    await_help_blocking();
    return;
  }

  const int64_t next_tick = (int64_t)micros() + ((int64_t)EI_CLASSIFIER_INTERVAL_MS * 1000);

  if (qmi.getDataReady()) {
    qmi.getAccelerometer(acc.x, acc.y, acc.z);
  } else {
    qmi.getAccelerometer(acc.x, acc.y, acc.z);
  }

  float ax = acc.x;
  float ay = acc.y;
  float az = acc.z;

  if (fabsf(ax) > MAX_ACCEPTED_RANGE_G) ax = signf(ax) * MAX_ACCEPTED_RANGE_G;
  if (fabsf(ay) > MAX_ACCEPTED_RANGE_G) ay = signf(ay) * MAX_ACCEPTED_RANGE_G;
  if (fabsf(az) > MAX_ACCEPTED_RANGE_G) az = signf(az) * MAX_ACCEPTED_RANGE_G;

  if (OUTPUT_UNITS_MS2) {
    ax *= CONVERT_G_TO_MS2;
    ay *= CONVERT_G_TO_MS2;
    az *= CONVERT_G_TO_MS2;
  }

  if (samples_collected < WINDOW_SAMPLES) {
    size_t offset = samples_collected * WINDOW_STRIDE;
    inference_buffer[offset + 0] = ax;
    inference_buffer[offset + 1] = ay;
    inference_buffer[offset + 2] = az;
    samples_collected++;
  } else {
    std::memmove(
      inference_buffer,
      inference_buffer + WINDOW_STRIDE,
      (EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - WINDOW_STRIDE) * sizeof(float)
    );
    inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3] = ax;
    inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 2] = ay;
    inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 1] = az;
  }

  if (samples_collected >= WINDOW_SAMPLES) {
    slice_sample_accum++;
    const size_t slice_target = (current_slice_target == 0) ? SLICE_SAMPLES_IDLE : current_slice_target;
    if (slice_sample_accum >= slice_target) {
      slice_sample_accum = 0;

      signal_t signal;
      int err = numpy::signal_from_buffer(inference_buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
      if (err != 0) {
        Serial.printf("[EI][ERR] signal_from_buffer (%d)\n", err);
      } else {
        ei_impulse_result_t result = { 0 };
        err = run_classifier(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK) {
          Serial.printf("[EI][ERR] run_classifier (%d)\n", err);
        } else {
          Serial.printf("[EI] Inference (DSP: %d ms | NN: %d ms)\n",
            result.timing.dsp, result.timing.classification);

          float fall_score = 0.0f;
          for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            const char *label = result.classification[i].label;
            float score = result.classification[i].value;
            Serial.printf("  %s: %.4f\n", label, (double)score);
            if (strcmp(label, "FALL") == 0) {
              fall_score = score;
            }
          }

          if (!fall_ema_initialized) {
            fall_ema = fall_score;
            fall_ema_initialized = true;
          } else {
            fall_ema = FALL_EMA_ALPHA * fall_score + (1.0f - FALL_EMA_ALPHA) * fall_ema;
          }

          last_slice_std = compute_recent_slice_stddev();
          last_slice_quiet = (last_slice_std <= QUIET_STD_THRESHOLD);

          push_history(fall_score, fall_ema);

          const uint32_t now_ms = millis();
          const bool in_refractory = (now_ms < refractory_until_ms);

          Serial.printf("[EI] FALL raw=%.3f | EMA=%.3f | std=%.3f | quiet=%d | state=%u\n",
            (double)fall_score, (double)fall_ema, (double)last_slice_std,
            last_slice_quiet ? 1 : 0, static_cast<unsigned>(fall_state));

          notify_fall_metrics(fall_score, fall_ema);

          switch (fall_state) {
            case FallDecisionState::Idle:
              reset_fall_counters();
              if (!in_refractory && fall_score >= FALL_P_HIGH && fall_ema >= FALL_H_ARM) {
                fall_state = FallDecisionState::AwaitHold;
                hold_windows_seen = 1;
                hold_windows_hits = (fall_ema >= FALL_H_HOLD) ? 1 : 0;
                current_slice_target = SLICE_SAMPLES_VERIFY;
                slice_sample_accum = 0;
                Serial.printf("[EI] Episode armed: raw>=%.2f & EMA>=%.2f (hold %u/%u)\n",
                  (double)FALL_P_HIGH, (double)FALL_H_HOLD,
                  (unsigned)hold_windows_hits, (unsigned)FALL_HOLD_N);
              } else if (in_refractory && fall_score >= FALL_P_HIGH) {
                Serial.println("[EI] Refractory active, peak ignored");
              }
              break;

            case FallDecisionState::AwaitHold:
              hold_windows_seen++;
              if (fall_ema >= FALL_H_HOLD) {
                hold_windows_hits++;
              }
              Serial.printf("[EI] Hold window %u/%u (hits %u/%u)\n",
                (unsigned)hold_windows_seen, (unsigned)FALL_HOLD_M,
                (unsigned)hold_windows_hits, (unsigned)FALL_HOLD_N);

              if (hold_windows_hits >= FALL_HOLD_N) {
                fall_state = FallDecisionState::AwaitLow;
                post_windows_seen = 0;
                post_low_hits = 0;
                post_quiet_hits = 0;
                post_high_hits = 0;
                Serial.println("[EI] Hold passed, evaluating post-impact");
              } else if (hold_windows_seen >= FALL_HOLD_M) {
                Serial.println("[EI] Hold failed, episode discarded");
                fall_state = FallDecisionState::Idle;
                reset_fall_counters();
                current_slice_target = SLICE_SAMPLES_IDLE;
                slice_sample_accum = 0;
              }
              break;

            case FallDecisionState::AwaitLow:
              post_windows_seen++;
              if (fall_score <= FALL_P_LOW) {
                post_low_hits++;
              }
              if (last_slice_quiet) {
                post_quiet_hits++;
              }
              if (fall_score >= FALL_P_HIGH && fall_ema >= FALL_H_ARM) {
                post_high_hits++;
              } else {
                post_high_hits = 0;
              }

              Serial.printf("[EI] Post window %u/%u: raw=%.3f (low %u/%u) quiet %u/%u\n",
                (unsigned)post_windows_seen, (unsigned)FALL_POST_M,
                (double)fall_score,
                (unsigned)post_low_hits, (unsigned)FALL_POST_N,
                (unsigned)post_quiet_hits, (unsigned)FALL_POST_N);

              if (post_low_hits >= FALL_POST_N && post_quiet_hits >= FALL_POST_N) {
                Serial.println("[EI] Fall confirmed (quiet)");
                enter_alert_mode();
                refractory_until_ms = now_ms + FALL_REFRACTORY_MS;
                fall_event_pending = true;
                fall_state = FallDecisionState::Idle;
                reset_fall_counters();
                current_slice_target = SLICE_SAMPLES_IDLE;
                slice_sample_accum = 0;
              }
              else if (post_high_hits >= HELP_HIGH_MIN && now_ms >= help_prompt_suppress_until_ms) {
                Serial.println("[EI] High probability persists -> asking the user");
                fall_state = FallDecisionState::AwaitHelp;
                show_help_prompt();
                current_slice_target = SLICE_SAMPLES_IDLE;
                slice_sample_accum = 0;
              }
              else if (post_windows_seen >= FALL_POST_M) {
                Serial.println("[EI] Post evaluation without fall (episode discarded)");
                fall_state = FallDecisionState::Idle;
                reset_fall_counters();
                current_slice_target = SLICE_SAMPLES_IDLE;
                slice_sample_accum = 0;
              }
              break;
            case FallDecisionState::AwaitHelp:
              //               // Should not arrive here because AwaitHelp is handled in await_help_blocking()
              break;
          }
        }
      }
    }
  }

  maintain_alert_state();

  int64_t wait_time = next_tick - (int64_t)micros();
  if (wait_time > 0) {
    delayMicroseconds((uint32_t)wait_time);
  }
}

#if !defined(EI_CLASSIFIER_SENSOR) || (EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_FUSION && EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER)
#error "Model not compatible with current sensor (expected FUSION/ACCELEROMETER)"
#endif




