#include <Arduino.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include <FastLED.h>
#include <WiFi.h>
#include <Wire.h>
#include <VL53L1X.h>
#include <driver/dac.h>
#include <driver/uart.h>
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
    RS_JUMPING     = 7,
};

// ── Strip geometry ────────────────────────────────────────────────────────────
// Strip wraps clockwise around a square chassis top.  FRONT face is split:
// LEDs 0-3 at the strip head and 27-29 at the tail.
// Order: FRONT(0-3) → RIGHT(4-11) → REAR(12-19) → LEFT(20-26) → FRONT(27-29)

#define SIDE_FRONT 0
#define SIDE_RIGHT 1
#define SIDE_REAR  2
#define SIDE_LEFT  3
static const uint8_t SIDE_LENS[4] = {7, 8, 8, 7};

static uint8_t side_led(uint8_t side, uint8_t i) {
    static const uint8_t F[7] = {27, 28, 29, 0, 1, 2, 3};
    if (side == SIDE_FRONT) return F[i];
    if (side == SIDE_RIGHT) return  4 + i;
    if (side == SIDE_REAR)  return 12 + i;
    /* SIDE_LEFT */         return 20 + i;
}

static void fill_side(CRGB* buf, uint8_t side, CRGB color) {
    for (uint8_t i = 0; i < SIDE_LENS[side]; i++) buf[side_led(side, i)] = color;
}

// ── LED globals ───────────────────────────────────────────────────────────────

static CRGB leds[NUM_LEDS];
static volatile uint8_t  g_robot_state       = RS_STARTUP;
static volatile bool     g_teensy_ever_heard  = false;
static volatile uint32_t g_last_teensy_ms    = 0;
static volatile bool     g_display_dirty     = false;
static volatile uint32_t g_reject_end_ms     = 0;
static volatile uint16_t g_tof_front_min     = 0xFFFF;
static volatile uint16_t g_tof_rear_min      = 0xFFFF;

// ── Display personality ───────────────────────────────────────────────────────
enum DisplayPersonality : uint8_t { PERS_ENGINEERING = 0, PERS_FACE = 1 };
static volatile DisplayPersonality g_display_personality =
    (DisplayPersonality)DEFAULT_DISPLAY_PERSONALITY;

// ── Telemetry state ───────────────────────────────────────────────────────────
// Declared here (before animation functions) because neo_task uses g_fault_code.

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
static volatile bool     g_version_mismatch = false;

// ── UART health (written core 1 on_teensy_packet / display_task, read display_task core 0)
// 32-bit reads/writes are atomic on Xtensa — no mutex needed for these counters.
static volatile uint32_t g_uart_crc_drops  = 0;  // lifetime CommLink CRC/frame errors
static volatile uint32_t g_uart_seq_gaps   = 0;  // lifetime packets lost (loop_count jumps)
static volatile uint8_t  g_uart_crc_rate   = 0;  // CRC drops in last 2 s window
static volatile uint8_t  g_uart_gap_rate   = 0;  // seq gaps  in last 2 s window

// ── Distance → color (green → yellow → red) ──────────────────────────────────

static CRGB dist_to_color(uint16_t d) {
    if (d == 0xFFFF || d > 600) return CRGB(0, 200, 0);
    if (d >= 200) {
        uint8_t t = (uint8_t)((600u - d) * 255u / 400u);
        return CRGB(t, 200, 0);
    }
    uint8_t t = (uint8_t)((uint32_t)(200u - d) * 255u / 200u);
    return CRGB(200, (uint8_t)(200u - (uint32_t)t * 200u / 255u), 0);
}

// ── Animations ────────────────────────────────────────────────────────────────

// DISCONNECTED — dim gray ghost comet orbiting the perimeter
static void anim_ghost_comet(CRGB* buf, uint32_t tick) {
    fill_solid(buf, NUM_LEDS, CRGB::Black);
    int head = (int)((tick / 4u) % NUM_LEDS);
    const uint8_t kBri[7] = {90, 65, 48, 34, 22, 13, 6};
    for (int t = 0; t < 7; t++)
        buf[(head - t + NUM_LEDS) % NUM_LEDS] = CRGB(kBri[t], kBri[t], kBri[t]);
}

// STARTUP — white cascade: each side breathes with 90° phase offset (CW wave)
static void anim_cascade_startup(CRGB* buf, uint32_t tick) {
    for (uint8_t s = 0; s < 4; s++) {
        uint8_t phase = (uint8_t)((tick % 200u) * 256u / 200u + s * 64u);
        uint8_t bri   = map(sin8(phase), 0, 255, 20, 255);
        fill_side(buf, s, CRGB(bri, bri, bri));
    }
}

// CALIBRATION — blue scanner bar advancing around the ring
static void anim_scanner_calibration(CRGB* buf, uint32_t tick) {
    fill_solid(buf, NUM_LEDS, CRGB(0, 0, 25));
    int head = (int)((tick / 2u) % NUM_LEDS);
    const CRGB kBar[4] = {CRGB(90, 90, 255), CRGB(0, 0, 200), CRGB(0, 0, 130), CRGB(0, 0, 60)};
    for (int t = 0; t < 4; t++)
        buf[(head + t) % NUM_LEDS] = kBar[t];
}

// STANDBY — slow amber ripple: a bright crest sweeps the full ring every ~3 s,
// rising from a dim baseline so the strip always glows warmly.
static void anim_marquee_standby(CRGB* buf, uint32_t tick) {
    // Crest position: one full lap every 150 ticks = 3 s
    float crest = (float)(tick % 60u) / 60.0f * (float)NUM_LEDS;
    for (int i = 0; i < NUM_LEDS; i++) {
        // Angular distance from crest (0..NUM_LEDS/2)
        float d = fabsf(fmodf((float)i - crest + (float)NUM_LEDS, (float)NUM_LEDS));
        if (d > (float)NUM_LEDS / 2.0f) d = (float)NUM_LEDS - d;
        // Cosine envelope: peak = 1.0 at crest, 0.0 at opposite side
        float env = 0.5f + 0.5f * cosf(d / ((float)NUM_LEDS / 2.0f) * (float)M_PI);
        uint8_t bri = (uint8_t)(20.0f + 235.0f * env * env);  // dim baseline → full amber
        buf[i] = CRGB(bri, (uint8_t)(bri * 110u / 255u), 0);  // amber: full red, ~43% green
    }
}

// RUNNING — two green comets orbiting in opposite directions, meeting exactly at
// front (LED 0) and rear (LED 15) each half-revolution, creating a bounce effect.
static void anim_running_tof(CRGB* buf, uint32_t tick) {
    fill_solid(buf, NUM_LEDS, CRGB::Black);

    // Advance 1 LED every 2 ticks → full lap in 60 ticks = 1.2 s.
    // CW head at pos; CCW head at (NUM_LEDS - pos) % NUM_LEDS.
    // They share the same LED at pos == 0 (front) and pos == 15 (rear).
    int pos = (int)((tick / 2u) % (uint32_t)NUM_LEDS);
    int cw  = pos;
    int ccw = (NUM_LEDS - pos) % NUM_LEDS;

    const uint8_t kTail[8] = {255, 170, 105, 60, 32, 16, 7, 2};

    for (int t = 0; t < 8; t++) {
        uint8_t b   = kTail[t];
        int cw_idx  = (cw  - t + NUM_LEDS) % NUM_LEDS;
        int ccw_idx = (ccw + t)             % NUM_LEDS;
        // Additive green (+ faint teal tint) so overlapping heads bloom bright
        buf[cw_idx].g  = qadd8(buf[cw_idx].g,  b);
        buf[cw_idx].b  = qadd8(buf[cw_idx].b,  b / 6);
        buf[ccw_idx].g = qadd8(buf[ccw_idx].g, b);
        buf[ccw_idx].b = qadd8(buf[ccw_idx].b, b / 6);
    }

    // Orange strobe on front/rear when obstacle is dangerously close (< 100 mm)
    uint16_t fd = g_tof_front_min;
    if (fd != 0xFFFF && fd < 100 && (tick % 5u) >= 2u)
        fill_side(buf, SIDE_FRONT, CRGB(255, 60, 0));
    uint16_t rd = g_tof_rear_min;
    if (rd != 0xFFFF && rd < 100 && (tick % 5u) >= 2u)
        fill_side(buf, SIDE_REAR, CRGB(255, 60, 0));
}

// ESTOP — red strobe + orange racing comet overlay
static void anim_estop_alarm(CRGB* buf, uint32_t tick) {
    fill_solid(buf, NUM_LEDS, (tick % 25u < 10u) ? CRGB(200, 0, 0) : CRGB::Black);
    int head = (int)(tick % NUM_LEDS);
    // Orange comet: R stays high, G tapers from ~140 to 10, B=0
    const uint8_t kR[5] = {255, 230, 180, 110,  50};
    const uint8_t kG[5] = {140, 110,  75,  40,  10};
    for (int t = 0; t < 5; t++)
        buf[(head - t + NUM_LEDS) % NUM_LEDS] = CRGB(kR[t], kG[t], 0);
}

// FAULT — deep red slow pulse, all LEDs
static void anim_fault(CRGB* buf, uint32_t tick) {
    // sin8 gives 0-255; map to 60-255 so it never fully blacks out
    uint8_t bri = map(sin8((uint8_t)(tick * 3u)), 0, 255, 60, 255);
    fill_solid(buf, NUM_LEDS, CRGB(bri, 0, 0));
}

// MANUAL — cyan knight rider ping-pong on all four sides simultaneously
static void anim_knight_rider_manual(CRGB* buf, uint32_t tick) {
    fill_solid(buf, NUM_LEDS, CRGB(0, 6, 18));
    const int DOT_W = 3;
    const uint8_t kBri[3] = {255, 150, 60};
    uint32_t t2 = tick / 2u;
    for (uint8_t s = 0; s < 4; s++) {
        int L = SIDE_LENS[s];
        int travel = L - DOT_W;
        int period = 2 * travel;
        int pos    = (int)(t2 % (uint32_t)period);
        if (pos > travel) pos = period - pos;
        for (int d = 0; d < DOT_W; d++) {
            int local = pos + d;
            if (local >= L) continue;
            uint8_t b = kBri[d];
            buf[side_led(s, (uint8_t)local)] = CRGB(0, (uint8_t)((uint16_t)b * 180u / 255u), b);
        }
    }
}

// CMD_REJECT — one-shot fast orange alternating strobe (overlay, 1 s)
static void anim_reject_strobe(CRGB* buf, uint32_t tick) {
    int offset = (int)(tick % 2u);
    for (int i = 0; i < NUM_LEDS; i++)
        buf[i] = ((i + offset) % 2 == 0) ? CRGB(255, 80, 0) : CRGB::Black;
}

// JUMPING — full rainbow wheel spinning fast + dual counter-rotating white comets + sparkles
static void anim_rainbow_jump(CRGB* buf, uint32_t tick) {
    // Full-spectrum rainbow wheel: 2+ rotations/sec (10 hue units × 50 fps = 500 hue/s)
    uint8_t base_hue = (uint8_t)(tick * 10u);
    for (int i = 0; i < NUM_LEDS; i++) {
        uint8_t hue = base_hue + (uint8_t)((uint32_t)i * 256u / NUM_LEDS);
        buf[i] = CHSV(hue, 255, 255);
    }
    // Two white comets orbiting in opposite directions at different speeds
    const uint8_t kCW[6] = {255, 220, 170, 110, 60, 20};
    int cw  = (int)((tick * 3u) % NUM_LEDS);
    int ccw = (int)(NUM_LEDS - (tick * 2u) % NUM_LEDS) % NUM_LEDS;
    for (int t = 0; t < 6; t++) {
        buf[(cw  + t)              % NUM_LEDS] = CRGB(kCW[t], kCW[t], kCW[t]);
        buf[(ccw - t + NUM_LEDS)   % NUM_LEDS] = CRGB(kCW[t], kCW[t], kCW[t]);
    }
    // Sparkles: pseudo-random positions flash full white every 2 ticks
    if (tick % 2u == 0) {
        buf[(tick * 7u  +  3u) % NUM_LEDS] = CRGB::White;
        buf[(tick * 13u + 11u) % NUM_LEDS] = CRGB::White;
        buf[(tick * 5u  + 17u) % NUM_LEDS] = CRGB::White;
        buf[(tick * 11u +  7u) % NUM_LEDS] = CRGB::White;
    }
}

// ── NeoPixel task ─────────────────────────────────────────────────────────────

static void neo_task(void*) {
    FastLED.addLeds<WS2812B, PIN_NEO, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);

    static CRGB neo_buf[NUM_LEDS];
    static CRGB neo_snap[NUM_LEDS];

    uint32_t tick          = 0;
    uint8_t  last_state    = 0xFF;
    bool     last_linked   = false;
    uint32_t trans_start   = 0;
    bool     in_transition = false;
    const uint32_t TRANS_TICKS = 15;  // 300 ms

    for (;;) {
        uint8_t state  = g_robot_state;
        bool    linked = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < 3000);

        if (state != last_state || linked != last_linked) {
            memcpy(neo_snap, leds, sizeof(leds));
            trans_start   = tick;
            in_transition = true;
            last_state    = state;
            last_linked   = linked;
        }

        fill_solid(neo_buf, NUM_LEDS, CRGB::Black);
        if (!linked) {
            anim_ghost_comet(neo_buf, tick);
        } else {
            switch (state) {
                case RS_STARTUP:     anim_cascade_startup(neo_buf, tick);         break;
                case RS_CALIBRATION: anim_scanner_calibration(neo_buf, tick);     break;
                case RS_STANDBY:     anim_marquee_standby(neo_buf, tick);         break;
                case RS_RUNNING:     anim_running_tof(neo_buf, tick);             break;
                case RS_ESTOP:
                    if (g_fault_code != FAULT_NONE && g_fault_code != FAULT_HUMAN_ESTOP)
                        anim_fault(neo_buf, tick);
                    else
                        anim_estop_alarm(neo_buf, tick);
                    break;
                case RS_MANUAL:      anim_knight_rider_manual(neo_buf, tick);     break;
                case RS_JUMPING:     anim_rainbow_jump(neo_buf, tick);            break;
                default:             fill_solid(neo_buf, NUM_LEDS, CRGB::White);  break;
            }
        }

        if (millis() < g_reject_end_ms)
            anim_reject_strobe(neo_buf, tick);

        if (in_transition) {
            uint32_t age = tick - trans_start;
            if (age >= TRANS_TICKS) {
                in_transition = false;
            } else {
                uint8_t amt = (uint8_t)(age * 255u / TRANS_TICKS);
                for (int i = 0; i < NUM_LEDS; i++)
                    neo_buf[i] = blend(neo_snap[i], neo_buf[i], amt);
            }
        }

        memcpy(leds, neo_buf, sizeof(leds));
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

// ── Telemetry snapshot (written by on_teensy_packet on core 1, read by loop on core 1)
// Both accesses are on the same core so no mutex is needed.
static TelemetryPayload s_telem_snap;
static bool             s_telem_a_rx   = false;  // TELEM_A received, waiting for B
static bool             s_telem_fresh  = false;  // complete A+B pair ready in s_telem_snap

// ── Teensy → all outputs ──────────────────────────────────────────────────────

static void on_teensy_packet(uint8_t type, uint8_t version, uint8_t /*source*/,
                              const uint8_t* payload, uint16_t len) {
    // Forward every packet upstream regardless of type
    g_usb.send(type, version, payload, len);
#if WIFI_ENABLED
    if (g_comm_tcp && g_tcp_client.connected())
        g_comm_tcp->send(type, version, payload, len);
    if ((type == COMM_TYPE_TELEM_A || type == COMM_TYPE_TELEM_B) && g_wifi_inited)
        g_telem_udp.send(type, version, payload, len);
#endif

    // Version mismatch: flag on either half so display updates immediately
    if ((type == COMM_TYPE_TELEM_A || type == COMM_TYPE_TELEM_B) && version != TELEM_VERSION) {
        if (!g_version_mismatch) {
            g_version_mismatch = true;
            g_display_dirty    = true;
            Serial.printf("[VERSION MISMATCH] Teensy telem v%u, expected v%u — reflash both boards\n",
                          version, TELEM_VERSION);
        }
        if (!g_teensy_ever_heard) g_display_dirty = true;
        g_last_teensy_ms    = millis();
        g_teensy_ever_heard = true;
        return;
    }

    // Reassemble split telemetry into s_telem_snap; signal loop() to process once.
    // All global updates happen in loop() so only the freshest snapshot is acted on.
    if (type == COMM_TYPE_TELEM_A && len == TELEM_A_LEN) {
        memcpy(&s_telem_snap, payload, TELEM_A_LEN);
        s_telem_a_rx = true;
    } else if (type == COMM_TYPE_TELEM_B && len == TELEM_B_LEN && s_telem_a_rx) {
        memcpy(((uint8_t*)&s_telem_snap) + TELEM_A_LEN, payload, TELEM_B_LEN);
        s_telem_a_rx  = false;
        s_telem_fresh = true;   // loop() will pick this up after draining the UART buffer
    }

    if (!g_teensy_ever_heard) g_display_dirty = true;
    g_last_teensy_ms    = millis();
    g_teensy_ever_heard = true;
}

// ── Display ───────────────────────────────────────────────────────────────────

static TFT_eSPI    tft;
static TFT_eSprite ah_sprite(&tft);   // 170×156 off-screen AH buffer
static TFT_eSprite wm_sprite(&tft);   // 70×156 off-screen wheel motor widget buffer
static TFT_eSprite eye_sprite(&tft);  // 100×100 face personality (reused for L and R)
static TFT_eSprite mouth_sprite(&tft);// 130×60  face personality mouth

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

// ── Face personality geometry (320×240 landscape) ─────────────────────────────
#define FACE_EYE_L_CX      90
#define FACE_EYE_R_CX     230
#define FACE_EYE_CY       105
#define FACE_EYE_RX        37   // ellipse half-width
#define FACE_EYE_RY        35   // ellipse half-height
#define FACE_PUPIL_R       22   // pupil circle radius
#define FACE_PUPIL_TRAVEL  18   // max pupil offset from eye center (px)
#define FACE_EYE_SPR_W    100   // sprite bounding box (gives 12 px clearance around ellipse)
#define FACE_EYE_SPR_H    100
#define FACE_EYE_SPR_CX    50   // eye center within sprite
#define FACE_EYE_SPR_CY    50
#define FACE_MOUTH_CX     160
#define FACE_MOUTH_CY     195
#define FACE_MOUTH_SPR_W  130
#define FACE_MOUTH_SPR_H   60
#define FACE_EYE_L_SPR_X  (FACE_EYE_L_CX - FACE_EYE_SPR_W/2)
#define FACE_EYE_R_SPR_X  (FACE_EYE_R_CX - FACE_EYE_SPR_W/2)
#define FACE_EYE_SPR_Y    (FACE_EYE_CY   - FACE_EYE_SPR_H/2)
#define FACE_MOUTH_SPR_X  (FACE_MOUTH_CX - FACE_MOUTH_SPR_W/2)
#define FACE_MOUTH_SPR_Y  (FACE_MOUTH_CY - FACE_MOUTH_SPR_H/2)

// Colour palette (set in initDisplay() via tft.color565())
static uint16_t COL_SKY, COL_GROUND, COL_AIRCRAFT;
static uint16_t COL_GRAPH_BG, COL_PITCH_LINE, COL_ZERO_LINE;
static uint16_t COL_HIP_FILL, COL_HIP_EMPTY;
static uint16_t COL_FOOTER_BG, COL_DIM, COL_DIVIDER;
static uint16_t COL_TICK_SKY, COL_TICK_GND;

// ── Face personality data structures ─────────────────────────────────────────

enum FaceMood : uint8_t {
    MOOD_SLEEPY     = RS_STARTUP,
    MOOD_FOCUSED    = RS_CALIBRATION,
    MOOD_CALM       = RS_STANDBY,
    MOOD_ALERT      = RS_RUNNING,
    MOOD_SCARED     = RS_ESTOP,
    MOOD_DETERMINED = RS_MANUAL,
    MOOD_ANNOYED    = RS_CMD_REJECT,
    MOOD_EXCITED    = RS_JUMPING,
};

struct MoodParams {
    float    lid_base;       // resting lid coverage (0=fully open, 1=fully closed)
    float    lid_squint;     // additional bottom-up squint
    uint32_t blink_min_ms;
    uint32_t blink_max_ms;
    uint32_t blink_close_ms;
    uint32_t blink_open_ms;
    float    pupil_scale;    // pitch/roll telemetry sensitivity (0=use idle drift)
    uint16_t sclera_color;
    uint16_t pupil_color;
    int8_t   mouth_shape;    // 0=neutral, 1=smile, -1=frown, 2=big smile, -2=wavy
};

// Filled at runtime in initFaceDisplay() because tft.color565() is not constexpr.
static MoodParams kMoodTable[8];

struct EyeAnim {
    float lid_pos;   // current lid coverage (0=open, 1=closed)
    float pupil_x;   // current pupil offset from eye center (px, float)
    float pupil_y;
    float pupil_tx;  // target pupil position
    float pupil_ty;
};

enum BlinkPhase : uint8_t { BLINK_OPEN, BLINK_CLOSING, BLINK_CLOSED, BLINK_OPENING };

struct FaceAnim {
    EyeAnim    eye[2];
    FaceMood   mood;
    uint8_t    prev_robot_state;
    BlinkPhase blink_phase;
    uint32_t   blink_next_ms;
    uint32_t   blink_phase_ms;
    uint32_t   pupil_retarget_ms;
    uint8_t    annoy_blink_count;
    uint32_t   annoy_next_ms;
};

static FaceAnim g_face;

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
        case RS_JUMPING:     return tft.color565(200, 0, 255);  // magenta
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
        case RS_JUMPING:     return "JUMP!";
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
        case FAULT_PARAM_OUT_OF_BOUNDS:return "Param out of bounds";
        case FAULT_PITCH_WATCHDOG:     return "Pitch watchdog ESTOP";
        case FAULT_WHEEL_RUNAWAY:      return "Wheel runaway ESTOP";
        default:                       return "Unknown fault";
    }
}

// ── Mode Banner ───────────────────────────────────────────────────────────────

static void drawModeBanner(uint8_t state, bool active, uint8_t fault, bool mismatch) {
    static uint8_t prev_state    = 0xFF;
    static bool    prev_active   = false;
    static uint8_t prev_fault    = 0xFF;
    static bool    prev_mismatch = false;
    if (state == prev_state && active == prev_active && fault == prev_fault
            && mismatch == prev_mismatch) return;
    prev_state = state; prev_active = active; prev_fault = fault; prev_mismatch = mismatch;

    tft.fillRect(0, 0, 320, BANNER_H, tft.color565(8, 10, 20));

    if (mismatch) {
        tft.fillRect(0, 0, 5, BANNER_H, TFT_RED);
        tft.setTextSize(2);
        tft.setTextColor(TFT_RED, tft.color565(8, 10, 20));
        tft.setCursor(8, (BANNER_H - 16) / 2);
        tft.print("VER MISMATCH");
        return;
    }

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

    // UART health — right side of footer top strip (x 214-316), only shown when non-zero.
    // Orange = historical errors, Red = errors happening right now (last 2 s).
    tft.fillRect(214, fy + 5, 104, 14, COL_FOOTER_BG);
    uint32_t crc_tot = g_uart_crc_drops;
    uint32_t gap_tot = g_uart_seq_gaps;
    if (crc_tot || gap_tot) {
        bool     active = (g_uart_crc_rate > 0 || g_uart_gap_rate > 0);
        uint16_t ucol   = active ? TFT_RED : tft.color565(255, 120, 0);
        tft.setTextSize(1);
        tft.setTextColor(ucol, COL_FOOTER_BG);
        tft.setCursor(214, fy + 7);
        // Format: "CRC:5 GAP:2" or "CRC:5!" (! = active)
        if (crc_tot && gap_tot)
            tft.printf("CRC:%lu GAP:%lu%s", crc_tot, gap_tot, active ? "!" : "");
        else if (crc_tot)
            tft.printf("CRC:%lu%s", crc_tot, active ? "!" : "");
        else
            tft.printf("GAP:%lu%s", gap_tot, active ? "!" : "");
    }

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

// ── Display init helpers ──────────────────────────────────────────────────────

static void init_color_palette() {
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
}

static void initEngineeringDisplay() {
    ah_sprite.setColorDepth(16);
    ah_sprite.createSprite(AH_W, AH_H);

    wm_sprite.setColorDepth(16);
    wm_sprite.createSprite(GRAPH_W, GRAPH_H);

    tft.fillScreen(TFT_BLACK);

    tft.drawFastVLine(HIP_X,   AH_Y, AH_H, COL_DIVIDER);
    tft.drawFastVLine(GRAPH_X, AH_Y, AH_H, COL_DIVIDER);
    tft.drawFastHLine(0, FOOTER_Y, 320, COL_DIVIDER);

    tft.setTextSize(1);
    tft.setTextColor(tft.color565(160, 160, 160), TFT_BLACK);
    tft.setCursor(HIP_X + 10, HIP_Y + 5);
    tft.print("L");
    tft.setCursor(HIP_X + 50, HIP_Y + 5);
    tft.print("R");

    tft.fillRect(0, FOOTER_Y + 1, 320, FOOTER_H - 1, COL_FOOTER_BG);
}

static void initFaceDisplay() {
    kMoodTable[MOOD_SLEEPY]     = { 0.55f, 0.0f,  8000, 12000, 200, 600, 0.0f, TFT_WHITE, TFT_BLACK,  0 };
    kMoodTable[MOOD_FOCUSED]    = { 0.15f, 0.25f, 5000,  8000, 100, 150, 0.0f, TFT_WHITE, TFT_BLUE,   0 };
    kMoodTable[MOOD_CALM]       = { 0.0f,  0.0f,  3000,  5000, 100, 180, 0.3f, TFT_WHITE, TFT_BLACK,  0 };
    kMoodTable[MOOD_ALERT]      = { 0.0f,  0.0f,  4000,  7000,  80, 120, 1.0f, TFT_WHITE, TFT_BLACK,  1 };
    kMoodTable[MOOD_SCARED]     = { 0.0f,  0.0f,  1500,  2500,  60,  90, 0.0f, TFT_WHITE, TFT_RED,   -1 };
    kMoodTable[MOOD_DETERMINED] = { 0.1f,  0.2f,  6000,  9000, 100, 160, 0.5f, TFT_WHITE, TFT_CYAN,  -1 };
    kMoodTable[MOOD_ANNOYED]    = { 0.05f, 0.0f,   500,   800,  60,  80, 0.0f, TFT_WHITE, TFT_BLACK, -2 };
    kMoodTable[MOOD_EXCITED]    = { 0.0f,  0.0f,   800,  1500,  50,  70, 0.0f,
                                    tft.color565(255, 255, 200), TFT_BLACK, 2 };

    eye_sprite.setColorDepth(16);
    eye_sprite.createSprite(FACE_EYE_SPR_W, FACE_EYE_SPR_H);

    mouth_sprite.setColorDepth(16);
    mouth_sprite.createSprite(FACE_MOUTH_SPR_W, FACE_MOUTH_SPR_H);

    tft.fillScreen(TFT_BLACK);

    memset(&g_face, 0, sizeof(g_face));
    g_face.mood             = MOOD_CALM;
    g_face.prev_robot_state = 0xFF;  // sentinel: forces mood sync on first update
    g_face.blink_phase      = BLINK_OPEN;
    g_face.blink_next_ms    = millis() + 3000;
    g_face.pupil_retarget_ms= millis() + 2000;
}

static void switch_personality(DisplayPersonality p) {
    if (p == g_display_personality) return;

    switch (g_display_personality) {
        case PERS_ENGINEERING:
            ah_sprite.deleteSprite();
            wm_sprite.deleteSprite();
            break;
        case PERS_FACE:
            eye_sprite.deleteSprite();
            mouth_sprite.deleteSprite();
            break;
    }

    g_display_personality = p;
    tft.fillScreen(TFT_BLACK);

    switch (p) {
        case PERS_ENGINEERING: initEngineeringDisplay(); break;
        case PERS_FACE:        initFaceDisplay();        break;
    }
    g_display_dirty = true;
}

static void initDisplay() {
    init_color_palette();
    switch (g_display_personality) {
        case PERS_ENGINEERING: initEngineeringDisplay(); break;
        case PERS_FACE:        initFaceDisplay();        break;
    }
}

// ── Face personality animation ────────────────────────────────────────────────

static void face_update_mood() {
    uint8_t state = g_robot_state;
    if (state == g_face.prev_robot_state) return;

    g_face.mood             = (FaceMood)constrain((int)state, 0, 7);
    g_face.prev_robot_state = state;

    if (g_face.mood == MOOD_ANNOYED) {
        g_face.annoy_blink_count = 4;
        g_face.annoy_next_ms     = millis();
    }
    g_face.pupil_retarget_ms = millis();
}

static void face_update_blink(uint32_t now) {
    const MoodParams& mp = kMoodTable[g_face.mood];

    // Annoyed rapid blink overrides normal schedule
    if (g_face.mood == MOOD_ANNOYED && g_face.annoy_blink_count > 0
            && now >= g_face.annoy_next_ms) {
        if (g_face.blink_phase == BLINK_OPEN) {
            g_face.blink_phase    = BLINK_CLOSING;
            g_face.blink_phase_ms = now;
        }
    }

    switch (g_face.blink_phase) {
        case BLINK_OPEN:
            if (now >= g_face.blink_next_ms) {
                g_face.blink_phase    = BLINK_CLOSING;
                g_face.blink_phase_ms = now;
            }
            break;

        case BLINK_CLOSING: {
            float t = (float)(now - g_face.blink_phase_ms) / (float)mp.blink_close_ms;
            float lid = constrain(t, 0.0f, 1.0f);
            g_face.eye[0].lid_pos = lid;
            g_face.eye[1].lid_pos = lid;
            if (now - g_face.blink_phase_ms >= mp.blink_close_ms) {
                g_face.blink_phase    = BLINK_CLOSED;
                g_face.blink_phase_ms = now;
            }
            break;
        }

        case BLINK_CLOSED:
            g_face.eye[0].lid_pos = 1.0f;
            g_face.eye[1].lid_pos = 1.0f;
            if (now - g_face.blink_phase_ms >= 20) {
                g_face.blink_phase    = BLINK_OPENING;
                g_face.blink_phase_ms = now;
            }
            break;

        case BLINK_OPENING: {
            float t = (float)(now - g_face.blink_phase_ms) / (float)mp.blink_open_ms;
            float lid = constrain(1.0f - t, 0.0f, 1.0f);
            g_face.eye[0].lid_pos = lid;
            g_face.eye[1].lid_pos = lid;
            if (now - g_face.blink_phase_ms >= mp.blink_open_ms) {
                g_face.eye[0].lid_pos = 0.0f;
                g_face.eye[1].lid_pos = 0.0f;
                g_face.blink_phase    = BLINK_OPEN;
                if (g_face.annoy_blink_count > 0) {
                    g_face.annoy_blink_count--;
                    g_face.annoy_next_ms = now + 80;
                } else {
                    uint32_t interval = mp.blink_min_ms +
                        (uint32_t)random((int32_t)(mp.blink_max_ms - mp.blink_min_ms));
                    g_face.blink_next_ms = now + interval;
                }
            }
            break;
        }
    }
}

static void face_update_pupils(uint32_t now) {
    const MoodParams& mp = kMoodTable[g_face.mood];
    bool connected = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < 1000);

    if (g_face.mood == MOOD_ALERT && connected) {
        float tx = constrain(g_telem_roll_rad  * mp.pupil_scale * FACE_PUPIL_TRAVEL / 0.3f,
                             -(float)FACE_PUPIL_TRAVEL, (float)FACE_PUPIL_TRAVEL);
        float ty = constrain(-g_telem_pitch_rad * mp.pupil_scale * FACE_PUPIL_TRAVEL / 0.3f,
                             -(float)FACE_PUPIL_TRAVEL, (float)FACE_PUPIL_TRAVEL);
        g_face.eye[0].pupil_tx = tx;
        g_face.eye[0].pupil_ty = ty;
        g_face.eye[1].pupil_tx = tx;
        g_face.eye[1].pupil_ty = ty;

    } else if (g_face.mood == MOOD_DETERMINED && connected && mp.pupil_scale > 0.0f) {
        float tx = constrain(g_telem_roll_rad  * mp.pupil_scale * FACE_PUPIL_TRAVEL / 0.3f,
                             -(float)FACE_PUPIL_TRAVEL, (float)FACE_PUPIL_TRAVEL);
        float ty = constrain(-g_telem_pitch_rad * mp.pupil_scale * FACE_PUPIL_TRAVEL / 0.3f,
                             -(float)FACE_PUPIL_TRAVEL, (float)FACE_PUPIL_TRAVEL);
        g_face.eye[0].pupil_tx = tx;
        g_face.eye[0].pupil_ty = ty;
        g_face.eye[1].pupil_tx = tx;
        g_face.eye[1].pupil_ty = ty;

    } else if (g_face.mood == MOOD_SCARED) {
        for (int i = 0; i < 2; i++) {
            g_face.eye[i].pupil_tx = (float)(random(9) - 4);
            g_face.eye[i].pupil_ty = (float)(random(9) - 4);
        }

    } else if (g_face.mood == MOOD_FOCUSED) {
        float phase = (float)(now % 2000u) / 2000.0f;
        float sweep = (phase < 0.5f) ? phase * 2.0f : 2.0f - phase * 2.0f;
        float tx = (sweep - 0.5f) * 2.0f * FACE_PUPIL_TRAVEL * 0.8f;
        g_face.eye[0].pupil_tx =  tx;
        g_face.eye[1].pupil_tx = -tx;
        g_face.eye[0].pupil_ty = 0.0f;
        g_face.eye[1].pupil_ty = 0.0f;

    } else {
        if (now >= g_face.pupil_retarget_ms) {
            float tx = (float)(random((int)(FACE_PUPIL_TRAVEL * 2 + 1))) - FACE_PUPIL_TRAVEL;
            float ty = (float)(random((int)(FACE_PUPIL_TRAVEL + 1))) - FACE_PUPIL_TRAVEL * 0.5f;
            for (int i = 0; i < 2; i++) {
                g_face.eye[i].pupil_tx = tx;
                g_face.eye[i].pupil_ty = ty;
            }
            g_face.pupil_retarget_ms = now + (uint32_t)random(1500, 4000);
        }
    }

    const float LERP = 0.15f;
    for (int i = 0; i < 2; i++) {
        g_face.eye[i].pupil_x += (g_face.eye[i].pupil_tx - g_face.eye[i].pupil_x) * LERP;
        g_face.eye[i].pupil_y += (g_face.eye[i].pupil_ty - g_face.eye[i].pupil_y) * LERP;
    }
}

static void draw_eye_sprite(int eye_idx, const EyeAnim& ea, FaceMood mood) {
    const MoodParams& mp = kMoodTable[mood];
    const int CX = FACE_EYE_SPR_CX;
    const int CY = FACE_EYE_SPR_CY;

    eye_sprite.fillSprite(TFT_BLACK);
    eye_sprite.fillEllipse(CX, CY, FACE_EYE_RX, FACE_EYE_RY, mp.sclera_color);

    // Pupil — clamped so it never exits the sclera
    int px = CX + (int)ea.pupil_x;
    int py = CY + (int)ea.pupil_y;
    px = constrain(px, CX - (FACE_EYE_RX - FACE_PUPIL_R - 2), CX + (FACE_EYE_RX - FACE_PUPIL_R - 2));
    py = constrain(py, CY - (FACE_EYE_RY - FACE_PUPIL_R - 2), CY + (FACE_EYE_RY - FACE_PUPIL_R - 2));
    eye_sprite.fillCircle(px, py, FACE_PUPIL_R, mp.pupil_color);
    // Glint
    eye_sprite.fillCircle(px + 7, py - 7, 5, TFT_WHITE);

    // Upper eyelid mask (slides down from top of ellipse)
    float effective_lid = ea.lid_pos + mp.lid_base;
    effective_lid = constrain(effective_lid, 0.0f, 1.0f);
    int lid_h = (int)(effective_lid * (float)(FACE_EYE_RY * 2));
    if (lid_h > 0) {
        int lid_top = CY - FACE_EYE_RY;
        eye_sprite.fillRect(0, lid_top, FACE_EYE_SPR_W, lid_h, TFT_BLACK);
    }

    // Bottom squint mask (rides up from bottom of ellipse)
    if (mp.lid_squint > 0.0f) {
        int sq_h = (int)(mp.lid_squint * (float)(FACE_EYE_RY * 2));
        int sq_y = CY + FACE_EYE_RY - sq_h;
        eye_sprite.fillRect(0, sq_y, FACE_EYE_SPR_W, sq_h + 6, TFT_BLACK);
    }

    // EXCITED: triangular arch mask over the top half of the ellipse
    if (mood == MOOD_EXCITED) {
        int outer_x = (eye_idx == 0) ? CX - FACE_EYE_RX : CX + FACE_EYE_RX;
        int inner_x = (eye_idx == 0) ? CX + FACE_EYE_RX : CX - FACE_EYE_RX;
        eye_sprite.fillTriangle(outer_x, CY, CX, CY - FACE_EYE_RY - 4, inner_x, CY, TFT_BLACK);
    }

    // ANNOYED: red angled brows above each eye, slanting inward
    if (mood == MOOD_ANNOYED) {
        uint16_t brow = tft.color565(200, 0, 0);
        int brow_y = CY - FACE_EYE_RY - 8;
        if (eye_idx == 0) {
            eye_sprite.drawLine(CX - 26, brow_y,     CX + 6, brow_y - 10, brow);
            eye_sprite.drawLine(CX - 26, brow_y + 1, CX + 6, brow_y - 9,  brow);
        } else {
            eye_sprite.drawLine(CX + 26, brow_y,     CX - 6, brow_y - 10, brow);
            eye_sprite.drawLine(CX + 26, brow_y + 1, CX - 6, brow_y - 9,  brow);
        }
    }
}

static void draw_mouth() {
    const int CX = FACE_MOUTH_SPR_W / 2;
    const int CY = FACE_MOUTH_SPR_H / 2;
    uint16_t MC = tft.color565(220, 120, 80);
    int8_t shape = kMoodTable[g_face.mood].mouth_shape;

    mouth_sprite.fillSprite(TFT_BLACK);

    if (shape == 0) {
        mouth_sprite.drawFastHLine(CX - 25, CY,     50, MC);
        mouth_sprite.drawFastHLine(CX - 25, CY + 1, 50, MC);

    } else if (shape == 1 || shape == 2) {
        float r = (shape == 2) ? 36.0f : 28.0f;
        int   n = (shape == 2) ? 10     : 8;
        for (int i = -n/2; i < n/2; i++) {
            float a0 = (float)i     / (float)(n/2) * 1.4f;
            float a1 = (float)(i+1) / (float)(n/2) * 1.4f;
            int x0 = CX + (int)(cosf(a0) * r);
            int y0 = CY + (int)(sinf(a0) * r * 0.55f);
            int x1 = CX + (int)(cosf(a1) * r);
            int y1 = CY + (int)(sinf(a1) * r * 0.55f);
            mouth_sprite.drawLine(x0, y0,     x1, y1,     MC);
            mouth_sprite.drawLine(x0, y0 + 1, x1, y1 + 1, MC);
        }

    } else if (shape == -1) {
        for (int i = -4; i < 4; i++) {
            float a0 = (float)i     / 4.0f * 1.4f;
            float a1 = (float)(i+1) / 4.0f * 1.4f;
            int x0 = CX + (int)(cosf(a0) * 28.0f);
            int y0 = CY - (int)(sinf(a0) * 16.0f) + 10;
            int x1 = CX + (int)(cosf(a1) * 28.0f);
            int y1 = CY - (int)(sinf(a1) * 16.0f) + 10;
            mouth_sprite.drawLine(x0, y0,     x1, y1,     MC);
            mouth_sprite.drawLine(x0, y0 + 1, x1, y1 + 1, MC);
        }

    } else {  // shape == -2: wavy/annoyed
        const int px[] = {-30, -15, 0, 15, 30};
        const int py[] = {  0,   8, 0,  8,  0};
        for (int i = 0; i < 4; i++) {
            mouth_sprite.drawLine(CX + px[i], CY + py[i],     CX + px[i+1], CY + py[i+1],     MC);
            mouth_sprite.drawLine(CX + px[i], CY + py[i] + 1, CX + px[i+1], CY + py[i+1] + 1, MC);
        }
    }

    mouth_sprite.pushSprite(FACE_MOUTH_SPR_X, FACE_MOUTH_SPR_Y);
}

static void update_face_display() {
    uint32_t now = millis();
    face_update_mood();
    face_update_blink(now);
    face_update_pupils(now);

    draw_eye_sprite(0, g_face.eye[0], g_face.mood);
    eye_sprite.pushSprite(FACE_EYE_L_SPR_X, FACE_EYE_SPR_Y);

    draw_eye_sprite(1, g_face.eye[1], g_face.mood);
    eye_sprite.pushSprite(FACE_EYE_R_SPR_X, FACE_EYE_SPR_Y);

    draw_mouth();
}

// ── Main display update (~10 Hz) ──────────────────────────────────────────────

static void update_display() {
    if (g_display_personality == PERS_FACE) {
        update_face_display();
        return;
    }

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

    drawModeBanner(state, active, fault, g_version_mismatch);
    drawArtificialHorizon(active ? pitch : 0.0f, active ? roll : 0.0f);
    drawHipBars(hip_l, hip_r, curr_l, curr_r);
    drawWheelMotors(active ? wm_l_vel : 0.0f, active ? wm_r_vel : 0.0f,
                    wm_l_vbus, wm_r_vbus,
                    active ? wm_l_state : 0, active ? wm_r_state : 0,
                    active ? wm_l_error : 0, active ? wm_r_error : 0,
                    active ? wm_mode : 0);
    drawFooter(state, active, rc_alive, active ? wheel_vel : 0.0f, pkt_age);
}

// ── Display task (core 0) ─────────────────────────────────────────────────────
// Owns all TFT SPI operations. loop() on core 1 never touches the display,
// so UART parsing runs uninterrupted regardless of render time.

static void display_task(void*) {
    pinMode(TFT_BLK, OUTPUT);
    digitalWrite(TFT_BLK, HIGH);
    tft.init();
    tft.setRotation(3);
    initDisplay();

    uint32_t last_disp_ms   = 0;
    uint32_t last_health_ms = 0;
    uint32_t prev_crc_drops = 0;
    uint32_t prev_seq_gaps  = 0;
    for (;;) {
        uint32_t now = millis();

        // Refresh UART health rates every 2 s
        if (now - last_health_ms >= 2000) {
            uint32_t cur_crc = g_teensy.rx_drops();
            uint32_t cur_gap = g_uart_seq_gaps;
            g_uart_crc_drops = cur_crc;
            g_uart_crc_rate  = (uint8_t)min((uint32_t)255, cur_crc - prev_crc_drops);
            g_uart_gap_rate  = (uint8_t)min((uint32_t)255, cur_gap - prev_seq_gaps);
            prev_crc_drops   = cur_crc;
            prev_seq_gaps    = cur_gap;
            last_health_ms   = now;
            if (g_uart_crc_rate || g_uart_gap_rate) g_display_dirty = true;
        }

        // Face mode: update every 10ms tick for smooth animation.
        // Engineering mode: update at 10 Hz or on dirty flag.
        bool time_to_update = g_display_dirty ||
            (g_display_personality == PERS_FACE ? true : (now - last_disp_ms >= 100));
        if (time_to_update) {
            g_display_dirty = false;
            update_display();
            last_disp_ms = millis();
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ── Setup / loop ──────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(921600);
    Serial2.setRxBufferSize(4096);
    Serial2.begin(TEENSY_UART_BAUD, SERIAL_8N1, TEENSY_UART_RX, TEENSY_UART_TX);
    uart_set_rx_full_threshold(UART_NUM_2, 32);  // fire ISR every 32 bytes (80 µs at 4 Mbaud) leaving 96 bytes / 192 µs of headroom vs the default 120-byte threshold (8 bytes / 16 µs)
    // Fix 5: flush boot-noise before the parser starts
    delay(10);
    while (Serial2.available()) Serial2.read();

    g_teensy.onPacket(on_teensy_packet);
    g_usb.onPacket(forward_to_teensy);

#if WIFI_ENABLED
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
#endif

    xTaskCreatePinnedToCore(neo_task,     "neo",  4096, nullptr, 1, nullptr, 0);
    xTaskCreatePinnedToCore(tof_task,     "tof",  4096, nullptr, 1, nullptr, 0);
    xTaskCreatePinnedToCore(display_task, "disp", 6144, nullptr, 1, nullptr, 0);

    Serial.println("[ESP32] ready");
}

void loop() {
    g_teensy.update();
    g_usb.update();

    // Process freshest complete telemetry snapshot. If multiple A+B pairs arrived
    // during the previous loop() iteration, all but the last are already overwritten
    // in s_telem_snap — we only act on what's there now.
    if (s_telem_fresh) {
        s_telem_fresh = false;
        const TelemetryPayload& pkt = s_telem_snap;

        if (g_version_mismatch) { g_version_mismatch = false; g_display_dirty = true; }

        uint8_t new_state = pkt.robot_state;
        if (new_state != g_robot_state || pkt.fault_code != g_fault_code)
            g_display_dirty = true;
        if (new_state == RS_CMD_REJECT && g_robot_state != RS_CMD_REJECT)
            g_reject_end_ms = millis() + 1000;

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

        // Gap detection via loop_count (in TELEM_B): each pair should advance by ~10
        static uint32_t s_last_lc  = 0;
        static bool     s_lc_valid = false;
        if (s_lc_valid) {
            uint32_t delta = pkt.loop_count - s_last_lc;
            if (delta > 15) g_uart_seq_gaps += (delta / 10) - 1;
        }
        s_last_lc  = pkt.loop_count;
        s_lc_valid = true;
    }

#if WIFI_ENABLED
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
#endif

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

#if WIFI_ENABLED
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
#endif
}
