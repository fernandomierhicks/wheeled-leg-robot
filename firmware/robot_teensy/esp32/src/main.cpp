#include <Arduino.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include <FastLED.h>
#include <WiFi.h>
#include <Wire.h>
#include <VL53L1X.h>
#include <driver/dac.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "CommLink.h"
#include "comm_protocol.h"
#include "udp_stream.h"
#include "wifi_config.h"

// ── NeoPixel ──────────────────────────────────────────────────────────────────

#define PIN_NEO    13
#define NUM_LEDS   30
#define BRIGHTNESS 200

// Robot state values — must match RobotStateEnum in teensy/src/robot_state.h
enum : uint8_t {
    RS_STARTUP     = 0,
    RS_CALIBRATION = 1,
    RS_STANDBY     = 2,
    RS_RUNNING     = 3,
    RS_ESTOP       = 4,
    RS_MANUAL      = 5,
    RS_CMD_REJECT  = 6,
};

static CRGB leds[NUM_LEDS];
static volatile uint8_t  g_robot_state      = RS_STARTUP;
static volatile bool     g_teensy_ever_heard = false;
static volatile uint32_t g_last_teensy_ms   = 0;
static volatile bool     g_display_dirty    = false;

static void neo_task(void*) {
    FastLED.addLeds<WS2812B, PIN_NEO, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);

    uint32_t tick = 0;

    for (;;) {
        uint8_t state  = g_robot_state;
        bool    linked = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < 1000);

        CRGB base;
        bool blink_fast = false;
        bool blink_slow = false;

        if (!linked) {
            base = CRGB(80, 80, 80);
        } else {
            switch (state) {
                case RS_STARTUP:     base = CRGB(255, 255, 255); break;
                case RS_CALIBRATION: base = CRGB(0,   0,   255); break;
                case RS_STANDBY:     base = CRGB(255, 200,   0); break;
                case RS_RUNNING:     base = CRGB(0,   255,   0); break;
                case RS_ESTOP:       base = CRGB(255,   0,   0); blink_slow = true; break;
                case RS_MANUAL:      base = CRGB(0,   200, 255); break;
                case RS_CMD_REJECT:  base = CRGB(255,  80,   0); blink_fast = true; break;
                default:             base = CRGB(255, 255, 255); break;
            }
        }

        if (blink_slow) {
            bool on = (tick % 25) < 12;
            fill_solid(leds, NUM_LEDS, on ? base : CRGB::Black);
        } else if (blink_fast) {
            bool on = (tick % 6) < 3;
            fill_solid(leds, NUM_LEDS, on ? base : CRGB::Black);
        } else {
            uint8_t phase = (uint8_t)((tick % 100) * 256 / 100);
            uint8_t bri   = sin8(phase);
            bri = map(bri, 0, 255, 30, 255);
            CRGB c = base;
            c.nscale8(bri);
            fill_solid(leds, NUM_LEDS, c);
        }

        FastLED.show();
        tick++;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// ── ToF sensors (VL53L1X ×4) ─────────────────────────────────────────────────
// Wiring: SDA=GPIO21  SCL=GPIO22  (400 kHz)
//   XSHUT[0]=GPIO14 → 0x30  (forward-left)
//   XSHUT[1]=GPIO27 → 0x31  (forward-right)
//   XSHUT[2]=GPIO26 → 0x32  (rear-left)
//   XSHUT[3]=GPIO25 → 0x33  (rear-right)

static const uint8_t TOF_XSHUT[4] = {14, 27, 26, 25};
static const uint8_t TOF_ADDR[4]  = {0x30, 0x31, 0x32, 0x33};

static VL53L1X g_tof_sensors[4];
static bool    g_tof_sensor_ok[4] = {};

// Shared ToF readings (written by tof_task on core 0, read by loop() on core 1).
// Individual uint16_t writes/reads are atomic on ESP32 Xtensa LX6.
static portMUX_TYPE       g_tof_mux = portMUX_INITIALIZER_UNLOCKED;
static volatile uint16_t  g_tof_dist[4]    = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
static volatile uint16_t  g_tof_front_min  = 0xFFFF;
static volatile uint16_t  g_tof_rear_min   = 0xFFFF;
static volatile bool      g_tof_updated    = false;

static bool tof_init_sensor(uint8_t i) {
    pinMode(TOF_XSHUT[i], OUTPUT);
    digitalWrite(TOF_XSHUT[i], HIGH);
    delay(10);
    g_tof_sensors[i].setAddress(0x29);  // reset internal address to boot default
    if (!g_tof_sensors[i].init()) return false;
    g_tof_sensors[i].setAddress(TOF_ADDR[i]);
    g_tof_sensors[i].setTimeout(500);
    g_tof_sensors[i].setDistanceMode(VL53L1X::Short);
    g_tof_sensors[i].setMeasurementTimingBudget(33000);  // 33 ms → ~30 Hz
    g_tof_sensors[i].startContinuous(33);
    return true;
}

static void tof_task(void*) {
    // Disable DAC on GPIO25/26 so XSHUT lines work as digital outputs
    dac_output_disable(DAC_CHANNEL_1);  // GPIO25
    dac_output_disable(DAC_CHANNEL_2);  // GPIO26

    // Hold all sensors in reset then bring up one at a time
    for (int i = 0; i < 4; i++) { pinMode(TOF_XSHUT[i], OUTPUT); digitalWrite(TOF_XSHUT[i], LOW); }
    delay(10);
    Wire.begin(21, 22, 400000UL);
    for (int i = 0; i < 4; i++) g_tof_sensor_ok[i] = tof_init_sensor(i);

    uint16_t dist[4]   = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};

    for (;;) {
        // Re-assert XSHUT HIGH each iteration (guards against DAC reconfiguring GPIO25/26)
        for (int i = 0; i < 4; i++) if (g_tof_sensor_ok[i]) digitalWrite(TOF_XSHUT[i], HIGH);

        for (int i = 0; i < 4; i++) {
            if (!g_tof_sensor_ok[i]) continue;
            if (g_tof_sensors[i].dataReady()) {
                g_tof_sensors[i].read(false);  // non-blocking; clears interrupt internally
                uint8_t st = g_tof_sensors[i].ranging_data.range_status;
                dist[i] = (st == 0) ? g_tof_sensors[i].ranging_data.range_mm : 0xFFFF;
            }
        }

        // Build processed values
        uint16_t fmin = min(dist[0], dist[1]);
        uint16_t rmin = min(dist[2], dist[3]);

        taskENTER_CRITICAL(&g_tof_mux);
        for (int i = 0; i < 4; i++) g_tof_dist[i] = dist[i];
        g_tof_front_min = fmin;
        g_tof_rear_min  = rmin;
        g_tof_updated   = true;
        taskEXIT_CRITICAL(&g_tof_mux);

        vTaskDelay(pdMS_TO_TICKS(5));  // 200 Hz poll; sensors fire at ~30 Hz each
    }
}

// ── CommLink instances ────────────────────────────────────────────────────────

static CommLink g_teensy(Serial2, COMM_SRC_ESP32);
static CommLink g_usb(Serial,    COMM_SRC_ESP32);

// ── Telemetry state ───────────────────────────────────────────────────────────

static volatile float    g_telem_pitch_rad  = 0.0f;
static volatile float    g_telem_roll_rad   = 0.0f;
static volatile float    g_telem_pitch_rate = 0.0f;
static volatile float    g_telem_wheel_vel  = 0.0f;
static volatile float    g_telem_hip_l_rad  = 0.0f;
static volatile float    g_telem_hip_r_rad  = 0.0f;
static volatile float    g_telem_hip_l_curr = 0.0f;
static volatile float    g_telem_hip_r_curr = 0.0f;
static volatile uint8_t  g_telem_ibus_alive = 0;
static volatile uint8_t  g_fault_code       = FAULT_NONE;
static volatile float    g_telem_wm_l_vel   = 0.0f;
static volatile float    g_telem_wm_r_vel   = 0.0f;
static volatile float    g_telem_wm_l_vbus  = 0.0f;
static volatile float    g_telem_wm_r_vbus  = 0.0f;
static volatile uint32_t g_telem_wm_l_error = 0;
static volatile uint32_t g_telem_wm_r_error = 0;
static volatile uint8_t  g_telem_wm_l_state = 0;
static volatile uint8_t  g_telem_wm_r_state = 0;
static volatile uint8_t  g_telem_wm_mode    = 0;

// ── WiFi / TCP / UDP ──────────────────────────────────────────────────────────

static WiFiServer  g_tcp_server(CMD_TCP_PORT);
static WiFiClient  g_tcp_client;
static CommLink*   g_comm_tcp   = nullptr;
static UDPStream   g_udp_stream;
static CommLink    g_telem_udp(g_udp_stream, COMM_SRC_ESP32);
static bool        g_wifi_inited = false;

// ── Command routing ───────────────────────────────────────────────────────────

static const char* mode_name(uint8_t state);  // forward decl

#define CMD_LOG_SIZE 3
static char     g_cmd_log[CMD_LOG_SIZE][24];
static uint32_t g_cmd_log_ms[CMD_LOG_SIZE];
static uint8_t  g_cmd_log_count = 0;
static volatile bool g_cmd_log_dirty = false;

static void decode_command(const uint8_t* payload, uint16_t len, char* out, size_t outlen) {
    if (len < 1) { snprintf(out, outlen, "CMD ?"); return; }
    uint8_t cmd_id = payload[0];
    switch (cmd_id) {
        case CMD_ID_SET_MODE:
            if (len >= 2) snprintf(out, outlen, "MODE -> %s", mode_name(payload[1]));
            else          snprintf(out, outlen, "SET MODE");
            break;
        case CMD_ID_HIP:
            if (len >= 3) {
                const char* motor = (payload[1] == HIP_MOTOR_L) ? "L"
                                   : (payload[1] == HIP_MOTOR_R) ? "R" : "BOTH";
                const char* sub;
                switch (payload[2]) {
                    case HIP_SUB_DISABLE: sub = "DISABLE"; break;
                    case HIP_SUB_ENABLE:  sub = "ENABLE";  break;
                    case HIP_SUB_ZERO:    sub = "ZERO";    break;
                    case HIP_SUB_MIT:     sub = "MIT";     break;
                    default:              sub = "?";       break;
                }
                snprintf(out, outlen, "HIP %s %s", motor, sub);
            } else { snprintf(out, outlen, "HIP"); }
            break;
        case CMD_ID_REBOOT:
            snprintf(out, outlen, "REBOOT");
            break;
        case CMD_ID_PARAM_SET:
            if (len >= 3) { uint16_t pid; memcpy(&pid, payload + 1, 2); snprintf(out, outlen, "PSET 0x%04X", pid); }
            else           snprintf(out, outlen, "PARAM SET");
            break;
        case CMD_ID_PARAM_GET:
            if (len >= 3) {
                uint16_t pid; memcpy(&pid, payload + 1, 2);
                if (pid == 0xFFFF) snprintf(out, outlen, "PARAM DUMP");
                else               snprintf(out, outlen, "PGET 0x%04X", pid);
            } else { snprintf(out, outlen, "PARAM GET"); }
            break;
        default:
            snprintf(out, outlen, "CMD 0x%02X", cmd_id);
            break;
    }
}

static void log_command(const uint8_t* payload, uint16_t len) {
    for (int i = CMD_LOG_SIZE - 1; i > 0; i--) {
        memcpy(g_cmd_log[i], g_cmd_log[i - 1], sizeof(g_cmd_log[i]));
        g_cmd_log_ms[i] = g_cmd_log_ms[i - 1];
    }
    decode_command(payload, len, g_cmd_log[0], sizeof(g_cmd_log[0]));
    g_cmd_log_ms[0] = millis();
    if (g_cmd_log_count < CMD_LOG_SIZE) g_cmd_log_count++;
    g_cmd_log_dirty = true;
}

static void forward_to_teensy(uint8_t type, uint8_t version, uint8_t /*source*/,
                               const uint8_t* payload, uint16_t len) {
    if (type == COMM_TYPE_COMMAND) {
        g_teensy.send(type, version, payload, len);
        log_command(payload, len);
    }
}

// ── Teensy → all outputs ──────────────────────────────────────────────────────

static void on_teensy_packet(uint8_t type, uint8_t version, uint8_t /*source*/,
                              const uint8_t* payload, uint16_t len) {
    if (Serial.availableForWrite() >= (int)(len + 9))
        g_usb.send(type, version, payload, len);
    if (g_comm_tcp && g_tcp_client.connected())
        g_comm_tcp->send(type, version, payload, len);
    if (type == COMM_TYPE_TELEMETRY && g_wifi_inited)
        g_telem_udp.send(type, version, payload, len);

    if (type == COMM_TYPE_TELEMETRY && version == TELEM_PAYLOAD_V4
            && len >= sizeof(TelemetryPayload)) {
        TelemetryPayload pkt;
        memcpy(&pkt, payload, sizeof(TelemetryPayload));

        uint8_t new_state = pkt.robot_state;
        if (new_state != g_robot_state || pkt.fault_code != g_fault_code)
            g_display_dirty = true;

        g_robot_state       = new_state;
        g_fault_code        = pkt.fault_code;
        g_telem_pitch_rad   = pkt.pitch_rad;
        g_telem_roll_rad    = pkt.roll_rad;
        g_telem_pitch_rate  = pkt.pitch_rate_rads;
        g_telem_wheel_vel   = pkt.wheel_vel_avg_ms;
        g_telem_hip_l_rad   = pkt.hip_l_pos_rad;
        g_telem_hip_r_rad   = pkt.hip_r_pos_rad;
        g_telem_hip_l_curr  = pkt.hip_l_current_a;
        g_telem_hip_r_curr  = pkt.hip_r_current_a;
        g_telem_ibus_alive  = pkt.ibus_alive;
        g_telem_wm_l_vel    = pkt.wm_l_vel_turns_s;
        g_telem_wm_r_vel    = pkt.wm_r_vel_turns_s;
        g_telem_wm_l_vbus   = pkt.wm_l_vbus;
        g_telem_wm_r_vbus   = pkt.wm_r_vbus;
        g_telem_wm_l_error  = pkt.wm_l_error;
        g_telem_wm_r_error  = pkt.wm_r_error;
        g_telem_wm_l_state  = pkt.wm_l_state;
        g_telem_wm_r_state  = pkt.wm_r_state;
        g_telem_wm_mode     = pkt.wm_mode;
    }
    if (!g_teensy_ever_heard) g_display_dirty = true;
    g_last_teensy_ms    = millis();
    g_teensy_ever_heard = true;
}

// ── Display ───────────────────────────────────────────────────────────────────

static TFT_eSPI    tft;
static TFT_eSprite ah_sprite(&tft);   // 170×156 off-screen AH buffer
static TFT_eSprite wm_sprite(&tft);   // 70×156 off-screen wheel motor widget buffer

// Layout (landscape 320×240, rotation=1)
#define BANNER_H   34
#define AH_X        0
#define AH_Y       BANNER_H
#define AH_W      170
#define AH_H      156
#define HIP_X     170
#define HIP_Y      BANNER_H
#define HIP_W      80
#define HIP_H     AH_H
#define GRAPH_X   250
#define GRAPH_Y    BANNER_H
#define GRAPH_W    70
#define GRAPH_H   AH_H
#define FOOTER_Y  190   // = BANNER_H + AH_H
#define FOOTER_H   50

// Hip display range [rad] — tune to match calibrated travel
#define HIP_DISPLAY_MIN  0.0f
#define HIP_DISPLAY_MAX  2.0f

// Colour palette (set in initDisplay() via tft.color565())
static uint16_t COL_SKY, COL_GROUND, COL_AIRCRAFT;
static uint16_t COL_GRAPH_BG, COL_PITCH_LINE, COL_ZERO_LINE;
static uint16_t COL_HIP_FILL, COL_HIP_EMPTY;
static uint16_t COL_FOOTER_BG, COL_DIM, COL_DIVIDER;
static uint16_t COL_TICK_SKY, COL_TICK_GND;

// ── Text / state helpers ──────────────────────────────────────────────────────

static uint16_t mode_color(uint8_t state) {
    switch (state) {
        case RS_STARTUP:     return TFT_WHITE;
        case RS_CALIBRATION: return TFT_BLUE;
        case RS_STANDBY:     return TFT_YELLOW;
        case RS_RUNNING:     return TFT_GREEN;
        case RS_ESTOP:       return TFT_RED;
        case RS_MANUAL:      return TFT_CYAN;
        case RS_CMD_REJECT:  return tft.color565(255, 100, 0);
        default:             return TFT_WHITE;
    }
}

static const char* mode_name(uint8_t state) {
    switch (state) {
        case RS_STARTUP:     return "STARTUP";
        case RS_CALIBRATION: return "CALIB";
        case RS_STANDBY:     return "STANDBY";
        case RS_RUNNING:     return "RUNNING";
        case RS_ESTOP:       return "ESTOP";
        case RS_MANUAL:      return "MANUAL";
        case RS_CMD_REJECT:  return "REJECTED";
        default:             return "UNKNOWN";
    }
}

static const char* fault_description(uint8_t code) {
    switch (code) {
        case FAULT_NONE:               return "";
        case FAULT_IMU_ERROR:          return "IMU error during startup";
        case FAULT_HIP_INIT_TIMEOUT:   return "Hip CAN timeout at boot";
        case FAULT_HIP_FEEDBACK_LOST:  return "Hip feedback lost";
        case FAULT_HIP_LARGE_POS_CMD:  return "Hip position jump too large";
        case FAULT_CALIBRATION_TIMEOUT:return "Hardstop not found";
        case FAULT_HUMAN_ESTOP:        return "User ESTOP";
        default:                       return "Unknown fault";
    }
}

// ── Mode Banner ───────────────────────────────────────────────────────────────

static void drawModeBanner(uint8_t state, bool active, uint8_t fault) {
    static uint8_t prev_state  = 0xFF;
    static bool    prev_active = false;
    static uint8_t prev_fault  = 0xFF;
    if (state == prev_state && active == prev_active && fault == prev_fault) return;
    prev_state = state; prev_active = active; prev_fault = fault;

    tft.fillRect(0, 0, 320, BANNER_H, tft.color565(8, 10, 20));

    bool show_fault = active && state == RS_ESTOP && fault != FAULT_NONE;
    uint16_t col = active ? mode_color(state) : COL_DIM;

    // Left accent stripe
    tft.fillRect(0, 0, 5, BANNER_H, col);

    const char* label    = active ? mode_name(state) : "NO TEENSY";
    int         tsize    = show_fault ? 2 : 3;
    int         char_w   = 6 * tsize;
    int         char_h   = 8 * tsize;
    int         label_px = strlen(label) * char_w;
    int         cx       = max(8, (320 - label_px) / 2);

    tft.setTextSize(tsize);
    tft.setTextColor(col, tft.color565(8, 10, 20));  // bg colour erases previous text
    tft.setCursor(cx, show_fault ? 3 : (BANNER_H - char_h) / 2);
    tft.print(label);

    if (show_fault) {
        tft.setTextSize(1);
        tft.setTextColor(TFT_YELLOW, tft.color565(8, 10, 20));
        tft.setCursor(8, BANNER_H - 10);
        tft.print(fault_description(fault));
    }
}

// ── Artificial Horizon (sprite-based, no flicker) ─────────────────────────────

static void drawArtificialHorizon(float pitch_rad, float roll_rad) {
    const int w  = AH_W, h = AH_H;
    const int cx = w / 2, cy = h / 2;

    // Positive pitch = nose up → horizon shifts down in display (3× sensitivity)
    int pitch_px = (int)(pitch_rad * 180.0f);
    pitch_px = constrain(pitch_px, -(h / 2 - 6), h / 2 - 6);
    int hy = cy + pitch_px;

    float cos_r = cosf(roll_rad);
    float sin_r = sinf(roll_rad);

    // Scanline fill entirely into sprite RAM (no SPI until pushSprite)
    for (int y = 0; y < h; y++) {
        if (fabsf(sin_r) < 0.02f) {
            ah_sprite.drawFastHLine(0, y, w, (y <= hy) ? COL_SKY : COL_GROUND);
        } else {
            // Horizon line through (cx, hy) with direction (cos_r, sin_r)
            float xh_f = cx + (float)(y - hy) / sin_r * cos_r;

            // Signed dist of left edge (x=0) from the horizon: sky side is negative
            float d = sin_r * (float)cx + cos_r * (float)(y - hy);
            bool left_is_sky = (d < 0);

            if (xh_f <= 0.0f || xh_f >= (float)w) {
                // Horizon off-screen: entire row is one colour
                ah_sprite.drawFastHLine(0, y, w, left_is_sky ? COL_SKY : COL_GROUND);
            } else {
                int xh = (int)xh_f;
                if (left_is_sky) {
                    ah_sprite.drawFastHLine(0,  y, xh,     COL_SKY);
                    ah_sprite.drawFastHLine(xh, y, w - xh, COL_GROUND);
                } else {
                    ah_sprite.drawFastHLine(0,  y, xh,     COL_GROUND);
                    ah_sprite.drawFastHLine(xh, y, w - xh, COL_SKY);
                }
            }
        }
    }

    // Pitch reference ticks at ±10° and ±20°
    const float px_per_rad = 180.0f;
    const float deg10_px   = 10.0f * (float)M_PI / 180.0f * px_per_rad;
    for (int tick = -2; tick <= 2; tick++) {
        if (tick == 0) continue;
        int ty = hy - (int)(tick * deg10_px);
        if (ty <= 1 || ty >= h - 1) continue;
        int tw = (abs(tick) == 1) ? 24 : 36;
        uint16_t tc = (ty < hy) ? COL_TICK_SKY : COL_TICK_GND;
        ah_sprite.drawFastHLine(cx - tw / 2, ty, tw, tc);
        ah_sprite.drawFastHLine(cx - 4, ty, 8, (ty <= hy) ? COL_SKY : COL_GROUND);
    }

    // Fixed aircraft crosshair
    ah_sprite.drawFastHLine(cx - 36, cy, 26, COL_AIRCRAFT);
    ah_sprite.drawFastHLine(cx + 10, cy, 26, COL_AIRCRAFT);
    ah_sprite.fillRect(cx - 3, cy - 3, 7, 7, COL_AIRCRAFT);

    // White border
    ah_sprite.drawRect(0, 0, w, h, TFT_WHITE);

    // Blit to display in one SPI transaction
    ah_sprite.pushSprite(AH_X, AH_Y);
}

// ── Hip Bars ──────────────────────────────────────────────────────────────────

static void drawHipBars(float hip_l, float hip_r, float curr_l, float curr_r) {
    const int bar_h   = 108;
    const int bar_w   = 26;
    const int bar_l_x = HIP_X + 6;
    const int bar_r_x = HIP_X + 46;
    const int bar_top = HIP_Y + 16;

    auto pct = [](float rad) -> float {
        return constrain((rad - HIP_DISPLAY_MIN) / (HIP_DISPLAY_MAX - HIP_DISPLAY_MIN),
                         0.0f, 1.0f);
    };

    float pl = pct(hip_l), pr = pct(hip_r);
    int   fl = (int)(pl * bar_h), fr = (int)(pr * bar_h);

    // Left bar
    tft.fillRect(bar_l_x, bar_top,              bar_w, bar_h - fl, COL_HIP_EMPTY);
    tft.fillRect(bar_l_x, bar_top + bar_h - fl, bar_w, fl,         COL_HIP_FILL);
    tft.drawRect(bar_l_x - 1, bar_top - 1, bar_w + 2, bar_h + 2, TFT_WHITE);

    // Right bar
    tft.fillRect(bar_r_x, bar_top,              bar_w, bar_h - fr, COL_HIP_EMPTY);
    tft.fillRect(bar_r_x, bar_top + bar_h - fr, bar_w, fr,         COL_HIP_FILL);
    tft.drawRect(bar_r_x - 1, bar_top - 1, bar_w + 2, bar_h + 2, TFT_WHITE);

    // Percentage labels
    int pct_y = bar_top + bar_h + 4;
    tft.fillRect(HIP_X, pct_y, HIP_W, 12, TFT_BLACK);
    tft.setTextSize(1);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setCursor(bar_l_x, pct_y);
    tft.printf("%3d%%", (int)(pl * 100));
    tft.setCursor(bar_r_x, pct_y);
    tft.printf("%3d%%", (int)(pr * 100));

    // Current labels
    int curr_y = pct_y + 12;
    tft.fillRect(HIP_X, curr_y, HIP_W, 11, TFT_BLACK);
    tft.setTextColor(tft.color565(255, 160, 0), TFT_BLACK);
    tft.setCursor(bar_l_x, curr_y);
    tft.printf("%.1fA", curr_l);
    tft.setCursor(bar_r_x, curr_y);
    tft.printf("%.1fA", curr_r);
}

// ── Wheel Motor Widget (sprite-based, 70×156) ─────────────────────────────────

static void drawWheelMotors(float vel_l, float vel_r,
                             float vbus_l, float vbus_r,
                             uint8_t state_l, uint8_t state_r,
                             uint32_t err_l, uint32_t err_r,
                             uint8_t mode) {
    const int w = GRAPH_W, h = GRAPH_H;

    wm_sprite.fillSprite(COL_GRAPH_BG);
    wm_sprite.setTextSize(1);

    // Title
    wm_sprite.setTextColor(COL_ZERO_LINE);
    wm_sprite.setCursor(4, 2);
    wm_sprite.print("WHEELS");

    // Velocity bars (bidirectional, zero centred)
    const float VEL_MAX  = 20.0f;   // turns/s full scale
    const int   bar_w    = 26;
    const int   bar_h    = 80;
    const int   bar_top  = 13;
    const int   bar_cy   = bar_top + bar_h / 2;
    const int   bar_lx   = 4;
    const int   bar_rx   = 38;
    const uint16_t COL_BAR_EMPTY = wm_sprite.color565(20, 20, 28);

    auto drawVelBar = [&](int bx, float vel) {
        float cv = constrain(vel / VEL_MAX, -1.0f, 1.0f);
        int fill  = (int)(fabsf(cv) * (bar_h / 2));
        uint16_t fc = (vel >= 0.0f) ? wm_sprite.color565(0, 200, 100)
                                     : wm_sprite.color565(255, 100, 0);
        wm_sprite.fillRect(bx, bar_top, bar_w, bar_h, COL_BAR_EMPTY);
        if (fill > 0) {
            int fy = (vel >= 0.0f) ? bar_cy - fill : bar_cy;
            wm_sprite.fillRect(bx, fy, bar_w, fill, fc);
        }
        wm_sprite.drawFastHLine(bx, bar_cy, bar_w, COL_ZERO_LINE);
        wm_sprite.drawRect(bx - 1, bar_top - 1, bar_w + 2, bar_h + 2, TFT_WHITE);
    };

    drawVelBar(bar_lx, vel_l);
    drawVelBar(bar_rx, vel_r);

    // Velocity numbers
    int num_y = bar_top + bar_h + 3;
    wm_sprite.setTextColor(TFT_WHITE);
    wm_sprite.setCursor(bar_lx, num_y);
    wm_sprite.printf("%.1f", vel_l);
    wm_sprite.setCursor(bar_rx, num_y);
    wm_sprite.printf("%.1f", vel_r);

    // State dots (green = closed loop, yellow = idle/other)
    int dot_y = num_y + 11;
    wm_sprite.fillCircle(bar_lx + bar_w / 2, dot_y, 4,
                         (state_l == 8) ? TFT_GREEN : TFT_YELLOW);
    wm_sprite.fillCircle(bar_rx + bar_w / 2, dot_y, 4,
                         (state_r == 8) ? TFT_GREEN : TFT_YELLOW);

    // Mode label
    const char* mode_str;
    switch (mode) {
        case 1: mode_str = "VEL";  break;
        case 2: mode_str = "POS";  break;
        case 3: mode_str = "TRQ";  break;
        default: mode_str = "IDLE"; break;
    }
    wm_sprite.setTextColor(wm_sprite.color565(160, 160, 160));
    wm_sprite.setCursor(2, dot_y + 7);
    wm_sprite.printf("M:%s", mode_str);

    // Bus voltage (average L+R — same battery)
    float vbus = (vbus_l + vbus_r) * 0.5f;
    uint16_t vc = (vbus > 20.0f) ? TFT_GREEN
                : (vbus > 15.0f) ? TFT_YELLOW
                                 : TFT_RED;
    wm_sprite.setTextColor(vc);
    wm_sprite.setCursor(2, dot_y + 17);
    wm_sprite.printf("%.1fV", vbus);

    // Error indicator
    if (err_l || err_r) {
        wm_sprite.setTextColor(TFT_RED);
        wm_sprite.setCursor(2, dot_y + 27);
        wm_sprite.print("ERR!");
    }

    wm_sprite.drawRect(0, 0, w, h, TFT_WHITE);
    wm_sprite.pushSprite(GRAPH_X, GRAPH_Y);
}

// ── Footer Status Bar ─────────────────────────────────────────────────────────

static void drawFooter(uint8_t state, bool active, bool rc_alive,
                       float wheel_vel, uint32_t pkt_age_ms) {
    const int fy = FOOTER_Y;

    // Heartbeat — pulses at display update rate when connected
    static bool hb_phase = false;
    if (active) hb_phase = !hb_phase;

    uint16_t hb_col = active ? mode_color(state) : COL_DIM;
    int      hb_r   = active ? (hb_phase ? 8 : 5) : 3;
    tft.fillRect(2, fy + 2, 22, 22, COL_FOOTER_BG);
    tft.fillCircle(12, fy + 13, hb_r, hb_col);

    // Packet age
    tft.fillRect(26, fy + 6, 48, 11, COL_FOOTER_BG);
    tft.setTextSize(1);
    tft.setTextColor(active ? TFT_WHITE : COL_DIM, COL_FOOTER_BG);
    tft.setCursor(26, fy + 7);
    if (active)
        tft.printf("%4lums", (unsigned long)min((uint32_t)9999, pkt_age_ms));
    else
        tft.print("NO PKT");

    // UART dot + label
    tft.fillRect(78, fy + 5, 46, 14, COL_FOOTER_BG);
    tft.fillCircle(83, fy + 12, 4, active ? TFT_GREEN : TFT_YELLOW);
    tft.setTextColor(active ? TFT_GREEN : TFT_YELLOW, COL_FOOTER_BG);
    tft.setCursor(90, fy + 7);
    tft.print(active ? "UART" : "WAIT");

    // RC dot + label
    tft.fillRect(130, fy + 5, 34, 14, COL_FOOTER_BG);
    tft.fillCircle(135, fy + 12, 4, rc_alive ? TFT_GREEN : TFT_RED);
    tft.setTextColor(rc_alive ? TFT_GREEN : TFT_RED, COL_FOOTER_BG);
    tft.setCursor(142, fy + 7);
    tft.print(rc_alive ? "RC" : "RC!");

    // WiFi dot + label
    tft.fillRect(174, fy + 5, 38, 14, COL_FOOTER_BG);
    uint16_t wifi_col = g_wifi_inited ? tft.color565(0, 150, 255) : COL_DIM;
    tft.fillCircle(179, fy + 12, 4, wifi_col);
    tft.setTextColor(wifi_col, COL_FOOTER_BG);
    tft.setCursor(186, fy + 7);
    tft.print("WiFi");

    // Wheel velocity bar
    const float VEL_MAX = 2.0f;
    const int   bar_x   = 4;
    const int   bar_y   = fy + 28;
    const int   bar_w   = 312;
    const int   bar_h   = 13;
    const int   bar_cx  = bar_x + bar_w / 2;

    tft.fillRect(bar_x, bar_y, bar_w, bar_h, COL_FOOTER_BG);
    tft.drawRect(bar_x, bar_y, bar_w, bar_h, tft.color565(60, 60, 60));

    float    cv  = constrain(wheel_vel, -VEL_MAX, VEL_MAX);
    int      fpx = (int)(cv / VEL_MAX * (bar_w / 2));
    uint16_t vc  = (wheel_vel >= 0) ? TFT_CYAN : TFT_MAGENTA;
    if (fpx > 0)
        tft.fillRect(bar_cx,       bar_y + 1, fpx,  bar_h - 2, vc);
    else if (fpx < 0)
        tft.fillRect(bar_cx + fpx, bar_y + 1, -fpx, bar_h - 2, vc);
    tft.drawFastVLine(bar_cx, bar_y, bar_h, TFT_WHITE);

    tft.fillRect(bar_x + 90, bar_y + bar_h + 1, 130, 9, COL_FOOTER_BG);
    tft.setTextColor(TFT_WHITE, COL_FOOTER_BG);
    tft.setCursor(bar_x + 112, bar_y + bar_h + 2);
    tft.printf("%.2f m/s", wheel_vel);
}

// ── One-time static display init ──────────────────────────────────────────────

static void initDisplay() {
    COL_SKY        = tft.color565( 20,  80, 200);
    COL_GROUND     = tft.color565(120,  65,  12);
    COL_AIRCRAFT   = tft.color565(255, 230,   0);
    COL_GRAPH_BG   = tft.color565(  4,   8,  18);
    COL_PITCH_LINE = tft.color565(  0, 200, 255);
    COL_ZERO_LINE  = tft.color565( 55,  55,  55);
    COL_HIP_FILL   = tft.color565(  0, 210, 210);
    COL_HIP_EMPTY  = tft.color565( 18,  18,  22);
    COL_FOOTER_BG  = tft.color565(  4,   6,  12);
    COL_DIM        = tft.color565( 65,  65,  65);
    COL_DIVIDER    = tft.color565( 35,  35,  40);
    COL_TICK_SKY   = tft.color565(120, 180, 255);
    COL_TICK_GND   = tft.color565(200, 140,  80);

    // Allocate sprite buffers
    ah_sprite.setColorDepth(16);
    ah_sprite.createSprite(AH_W, AH_H);

    wm_sprite.setColorDepth(16);
    wm_sprite.createSprite(GRAPH_W, GRAPH_H);

    tft.fillScreen(TFT_BLACK);

    // Column dividers
    tft.drawFastVLine(HIP_X,   AH_Y, AH_H, COL_DIVIDER);
    tft.drawFastVLine(GRAPH_X, AH_Y, AH_H, COL_DIVIDER);
    tft.drawFastHLine(0, FOOTER_Y, 320, COL_DIVIDER);

    // Hip axis labels (static — sprites don't cover this area)
    tft.setTextSize(1);
    tft.setTextColor(tft.color565(160, 160, 160), TFT_BLACK);
    tft.setCursor(HIP_X + 10, HIP_Y + 5);
    tft.print("L");
    tft.setCursor(HIP_X + 50, HIP_Y + 5);
    tft.print("R");

    tft.fillRect(0, FOOTER_Y + 1, 320, FOOTER_H - 1, COL_FOOTER_BG);
}

// ── Main display update (~10 Hz) ──────────────────────────────────────────────

static void update_display() {
    uint8_t  state     = g_robot_state;
    uint8_t  fault     = g_fault_code;
    bool     active    = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < 1000);
    float    pitch     = g_telem_pitch_rad;
    float    roll      = g_telem_roll_rad;
    float    hip_l     = g_telem_hip_l_rad;
    float    hip_r     = g_telem_hip_r_rad;
    float    curr_l    = g_telem_hip_l_curr;
    float    curr_r    = g_telem_hip_r_curr;
    float    wheel_vel = g_telem_wheel_vel;
    uint8_t  rc_alive  = g_telem_ibus_alive;
    uint32_t pkt_age   = millis() - g_last_teensy_ms;

    float    wm_l_vel   = g_telem_wm_l_vel;
    float    wm_r_vel   = g_telem_wm_r_vel;
    float    wm_l_vbus  = g_telem_wm_l_vbus;
    float    wm_r_vbus  = g_telem_wm_r_vbus;
    uint32_t wm_l_error = g_telem_wm_l_error;
    uint32_t wm_r_error = g_telem_wm_r_error;
    uint8_t  wm_l_state = g_telem_wm_l_state;
    uint8_t  wm_r_state = g_telem_wm_r_state;
    uint8_t  wm_mode    = g_telem_wm_mode;

    drawModeBanner(state, active, fault);
    drawArtificialHorizon(active ? pitch : 0.0f, active ? roll : 0.0f);
    drawHipBars(hip_l, hip_r, curr_l, curr_r);
    drawWheelMotors(active ? wm_l_vel : 0.0f, active ? wm_r_vel : 0.0f,
                    wm_l_vbus, wm_r_vbus,
                    active ? wm_l_state : 0, active ? wm_r_state : 0,
                    active ? wm_l_error : 0, active ? wm_r_error : 0,
                    active ? wm_mode : 0);
    drawFooter(state, active, rc_alive, active ? wheel_vel : 0.0f, pkt_age);
}

// ── Setup / loop ──────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    Serial2.setRxBufferSize(4096);
    Serial2.begin(TEENSY_UART_BAUD, SERIAL_8N1, TEENSY_UART_RX, TEENSY_UART_TX);

    g_teensy.onPacket(on_teensy_packet);
    g_usb.onPacket(forward_to_teensy);

    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    pinMode(TFT_BLK, OUTPUT);
    digitalWrite(TFT_BLK, HIGH);
    tft.init();
    tft.setRotation(3);
    initDisplay();

    xTaskCreatePinnedToCore(neo_task, "neo", 2048, nullptr, 1, nullptr, 0);
    xTaskCreatePinnedToCore(tof_task, "tof", 4096, nullptr, 1, nullptr, 0);

    Serial.println("[ESP32] ready");
}

void loop() {
    g_teensy.update();
    g_usb.update();

    if (g_comm_tcp && g_tcp_client.connected())
        g_comm_tcp->update();

    if (WiFi.status() == WL_CONNECTED && !g_wifi_inited) {
        g_tcp_server.begin();
        g_udp_stream.beginSend("255.255.255.255", TELEM_UDP_PORT);
        g_wifi_inited = true;
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
    }

    static uint32_t last_reconnect_ms = 0;
    if (WiFi.status() != WL_CONNECTED) {
        g_wifi_inited = false;
        if (millis() - last_reconnect_ms > 5000) {
            WiFi.begin(WIFI_SSID, WIFI_PASS);
            last_reconnect_ms = millis();
        }
    }

    // Send ToF packet to Teensy (and USB) at 20 Hz when new sensor data is ready
    static uint32_t last_tof_ms = 0;
    if (g_tof_updated && (millis() - last_tof_ms >= 50)) {
        last_tof_ms = millis();
        TofPayload tpkt;
        taskENTER_CRITICAL(&g_tof_mux);
        for (int i = 0; i < 4; i++) tpkt.dist_mm[i] = g_tof_dist[i];
        tpkt.front_min_mm = g_tof_front_min;
        tpkt.rear_min_mm  = g_tof_rear_min;
        g_tof_updated = false;
        taskEXIT_CRITICAL(&g_tof_mux);
        g_teensy.send(COMM_TYPE_TOF, TOF_PAYLOAD_V1, &tpkt, sizeof(tpkt));
    }

    static uint32_t last_disp_ms = 0;
    if (g_display_dirty || (millis() - last_disp_ms >= 100)) {
        g_display_dirty = false;
        update_display();
        last_disp_ms = millis();
    }

    if (g_wifi_inited && g_tcp_server.hasClient()) {
        WiFiClient c = g_tcp_server.available();
        if (c) {
            g_tcp_client = c;
            delete g_comm_tcp;
            g_comm_tcp = new CommLink(g_tcp_client, COMM_SRC_ESP32);
            g_comm_tcp->onPacket(forward_to_teensy);
            Serial.print("[TCP] client: ");
            Serial.println(g_tcp_client.remoteIP());
        }
    }
}
