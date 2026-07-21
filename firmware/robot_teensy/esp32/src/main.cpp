#include <Arduino.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include <FastLED.h>
#include <WiFi.h>
#include <Wire.h>
#include <VL53L1X.h>
#include <driver/dac.h>
#include <driver/uart.h>
#include <esp_system.h>
#include <esp_task_wdt.h>  // uplink_task watchdog — see esp_task_wdt_init() in setup()
#include <lwip/sockets.h>  // select()/fd_set for the TCP write-readiness pre-check below
#include <freertos/queue.h>
#include <freertos/semphr.h>
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
    RS_STANDING_UP = 8,
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
static volatile uint8_t  g_telem_active_profile = 0;
static volatile bool     g_version_mismatch = false;

// Extended telemetry — new fields used by redesigned display
static volatile uint16_t g_telem_health_flags  = 0;
static volatile float    g_telem_hip_l_cmd_rad = 0.0f;
static volatile float    g_telem_hip_r_cmd_rad = 0.0f;
static volatile float    g_telem_v_ref          = 0.0f;
static volatile float    g_telem_omega_cmd_rds = 0.0f;
static volatile float    g_telem_ff1_out        = 0.0f;
static volatile float    g_telem_ff2_out        = 0.0f;
static volatile uint8_t  g_telem_jump_state     = 0;

// ── UART health (written core 1 on_teensy_packet / display_task, read display_task core 0)
// 32-bit reads/writes are atomic on Xtensa — no mutex needed for these counters.
static volatile uint32_t g_uart_crc_drops  = 0;  // lifetime CommLink CRC/frame errors
static volatile uint32_t g_uart_seq_gaps   = 0;  // lifetime frames lost (CommLink seq gaps; mirror of g_teensy.rx_seq_gaps())
static volatile uint8_t  g_uart_crc_rate   = 0;  // CRC drops in last 2 s window
static volatile uint8_t  g_uart_gap_rate   = 0;  // seq gaps  in last 2 s window

// ── SD-log GET transfer state (written on_teensy_packet, read neo_task/display_task) ──
// The Teensy pauses its normal 50 Hz telemetry while a GET is streaming (see
// teensy/src/main.cpp loop()), so g_last_teensy_ms/"linked" goes stale during
// every download unless something else tells the neo/display tasks "this is
// an expected quiet spell, not a dead link" — that's what these track, derived
// straight from the LOG_INFO/LOG_DATA frames passing through regardless.
static volatile bool     g_log_xfer_active      = false;
static volatile uint32_t g_last_log_activity_ms = 0;
static volatile uint32_t g_log_total_chunks     = 0;
static volatile uint32_t g_log_chunks_done      = 0;
static volatile bool     g_log_relay_busy       = false;
// Set once when leaving the download banner so drawModeBanner's own
// value-based redraw cache (which has no idea the screen was just showing
// something else) is forced to repaint instead of concluding "unchanged".
static volatile bool     g_banner_force_redraw  = false;

// Set from LOG_SUB_START/STOP commands seen passing through forward_to_teensy()
// — direct, immediate signal (no round-trip ack needed), independent of
// telemetry. A duration-bound recording (the common case — see tuning.md's
// Test Protocol) auto-stops firmware-side with no explicit STOP command ever
// sent (mirrors sd_logger.cpp's own g_stop_deadline_ms), so g_recording_active
// alone would get stuck true forever after the first timed trial — use
// recording_is_active() below, which also honors g_recording_deadline_ms.
static volatile bool     g_recording_active      = false;
static volatile uint32_t g_recording_deadline_ms = 0;  // 0 = no deadline (indefinite, STOP-gated)

// True while a recording is genuinely still running — honors the deadline
// so a duration-bound recording's indicator clears itself on schedule even
// though no explicit STOP command ever arrives for those.
static bool recording_is_active() {
    if (!g_recording_active) return false;
    if (g_recording_deadline_ms != 0 && (int32_t)(millis() - g_recording_deadline_ms) >= 0) return false;
    return true;
}

// True while a GET is actively streaming; bounded by a staleness fallback
// (3 s with no log traffic at all) in case an XFER_END is ever lost.
static bool log_xfer_is_downloading() {
    return g_log_xfer_active &&
           (g_log_relay_busy || millis() - g_last_log_activity_ms < 3000);
}
static uint8_t log_xfer_percent() {
    uint32_t total = g_log_total_chunks;
    if (total == 0) return 0;
    uint32_t done = g_log_chunks_done;
    if (done > total) done = total;
    return (uint8_t)((uint64_t)done * 100u / total);
}

// ── WiFi diagnostics (loop.cpp, core 1) ───────────────────────────────────────
static volatile uint16_t g_loop_max_us            = 0;  // max loop() duration since last diag tick
static volatile uint16_t g_udp_send_max_us        = 0;  // max time inside UDPStream::write() since last diag tick
static volatile uint16_t g_wifi_reconnect_count   = 0;
// §2b: which link the GUI last announced it's reading — WIFI_DIAG_TRANSPORT_*.
// BOTH_LEGACY until a CMD_ID_SET_TELEM_TRANSPORT arrives (gating disabled or no announcement yet).
static volatile uint8_t  g_active_telem_transport = WIFI_DIAG_TRANSPORT_BOTH_LEGACY;

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

// DISCONNECTED (no Teensy heard) — dim red slow pulse, all LEDs. Deliberately a
// different color family from STARTUP (white) and every state color, and dimmer/
// slower than anim_fault's alarm pulse — this means "no Teensy link", not an
// active fault.
static void anim_ghost_comet(CRGB* buf, uint32_t tick) {
    uint8_t bri = map(sin8((uint8_t)tick), 0, 255, 8, 70);
    fill_solid(buf, NUM_LEDS, CRGB(bri, 0, 0));
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

// DOWNLOADING SD LOG — blue progress ring (fills as chunks arrive) + a
// pulsing leading edge, on a dim blue base so the strip clearly reads "busy,
// not dead" even at 0%. Takes over from every other animation, including the
// "not linked" ghost-comet — see log_xfer_is_downloading()'s use in neo_task().
static void anim_download(CRGB* buf, uint32_t tick, uint8_t pct) {
    fill_solid(buf, NUM_LEDS, CRGB(0, 4, 16));
    int lit = (int)((uint32_t)pct * NUM_LEDS / 100u);
    for (int i = 0; i < lit; i++)
        buf[i] = CRGB(0, 120, 255);
    if (lit < NUM_LEDS) {
        uint8_t pulse = map(sin8((uint8_t)(tick * 12u)), 0, 255, 60, 255);
        buf[lit] = CRGB(0, pulse, pulse);
    }
}

// RECORDING overlay — periodic additive blue flash blended onto whatever the
// base animation currently is (state animation, ghost-comet, etc.) so "SD
// card is actively writing" stays visible without hiding it. Unlike
// anim_reject_strobe (a brief, fully-overriding one-shot event), a recording
// runs for the whole trial, so a full takeover would hide the state
// animation for a large fraction of every session — additive blue instead
// leaves R/G untouched, reading as "the usual pattern, tinted blue".
static void anim_recording_strobe(CRGB* buf, uint32_t tick) {
    if ((tick % 25u) >= 5u) return;  // ~200 ms flash, repeating every 500 ms
    for (int i = 0; i < NUM_LEDS; i++)
        buf[i].b = qadd8(buf[i].b, 220);
}

// ── NeoPixel task ─────────────────────────────────────────────────────────────

static void neo_task(void*) {
    FastLED.addLeds<WS2812B, PIN_NEO, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);

    // One-shot boot rainbow (~1.5 s) so a physical reboot/reflash is
    // unmistakable on the bench — distinct from every state animation and
    // from anim_ghost_comet's dim red "not linked yet" indicator, which is
    // easy to miss if you're not staring at the strip at that exact moment.
    for (uint32_t t = 0; t < 75; t++) {  // 75 * 20 ms = 1.5 s
        uint8_t base_hue = (uint8_t)(t * 10u);
        for (int i = 0; i < NUM_LEDS; i++)
            leds[i] = CHSV(base_hue + (uint8_t)((uint32_t)i * 256u / NUM_LEDS), 255, 255);
        FastLED.show();
        vTaskDelay(pdMS_TO_TICKS(20));
    }

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
        bool    linked = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < TEENSY_LINK_TIMEOUT_MS);

        if (state != last_state || linked != last_linked) {
            memcpy(neo_snap, leds, sizeof(leds));
            trans_start   = tick;
            in_transition = true;
            last_state    = state;
            last_linked   = linked;
        }

        fill_solid(neo_buf, NUM_LEDS, CRGB::Black);
        if (log_xfer_is_downloading()) {
            anim_download(neo_buf, tick, log_xfer_percent());
        } else if (!linked) {
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

        if (recording_is_active() && !log_xfer_is_downloading())
            anim_recording_strobe(neo_buf, tick);

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

// Dedicated diagnostics links — TX-only (never .update()'d), so they never steal
// bytes from g_usb/g_teensy's RX parsers. Kept separate from g_telem_udp so their
// own _seq_tx counters don't interleave into the telemetry seq stream that the
// A/B pairing metric depends on.
static CommLink g_usb_diag(Serial, COMM_SRC_ESP32);

// ── WiFi / TCP / UDP ──────────────────────────────────────────────────────────

static WiFiServer  g_tcp_server(CMD_TCP_PORT);
static WiFiClient  g_tcp_client;
static CommLink*   g_comm_tcp   = nullptr;
static UDPStream   g_udp_stream;
static UDPStream   g_udp_diag_stream;
static CommLink    g_wifi_diag(g_udp_diag_stream, COMM_SRC_ESP32);
static CommLink    g_telem_udp(g_udp_stream, COMM_SRC_ESP32);
static bool        g_wifi_inited = false;

// ── Uplink queue (decouples the UART parse path from network/USB sends) ──────
// Normal telemetry and ACK-controlled bulk logs are sent by separate core-0
// tasks, keeping all potentially blocking USB/TCP I/O out of the UART parser.
typedef struct {
    uint8_t  type, version;
    uint8_t  destinations;
    uint16_t generation;
    uint16_t len;
    uint8_t  payload[COMM_MAX_PAYLOAD];
} UplinkFrame;

enum : uint8_t {
    UPLINK_DEST_USB = 0x01,
    UPLINK_DEST_TCP = 0x02,
    UPLINK_DEST_ALL = UPLINK_DEST_USB | UPLINK_DEST_TCP,
};

static QueueHandle_t      g_uplink_q        = nullptr;
static uint32_t           g_uplink_drops    = 0;   // frames dropped because queue was full
static QueueHandle_t      g_log_uplink_q    = nullptr;
static uint32_t           g_log_uplink_drops = 0;
static uint16_t           g_log_generation  = 0;

typedef struct {
    uint16_t file_index;
    uint32_t chunk_index;
    uint16_t generation;
} LogRelayAck;
static QueueHandle_t      g_log_ack_q = nullptr;

// LOG_INFO/LOG_DATA belong only on the transport which issued the most recent
// LOG command. Sending every 498-byte chunk to both USB and TCP made WiFi GETs
// spend most of their 8 ms pacing window blocking on the unused 921600-baud
// USB UART, and let a stale TCP client interfere with ESP32-USB downloads.
// Both command parsers run from loop() on core 1, as does on_teensy_packet(),
// so reading/writing this byte needs no cross-core lock.
static uint8_t            g_log_uplink_dest   = UPLINK_DEST_ALL;
static uint32_t           g_tcp_send_skipped = 0;  // TCP sends skipped because the socket
                                                    // wasn't write-ready — see the select()
                                                    // pre-check in uplink_task: WiFiClient::write()
                                                    // has a hardcoded 10x1s blocking retry loop
                                                    // (framework source, not ours to change) that
                                                    // was observed hanging uplink_task for so long
                                                    // the task watchdog fired — this counter tracks
                                                    // how often that's being avoided.
static volatile uint16_t  g_tcp_send_max_us = 0;   // like g_udp_send_max_us; core 0 writes,
                                                    // core 1 reads/resets in send_wifi_diag —
                                                    // volatile 16-bit access is atomic on Xtensa,
                                                    // same reasoning as g_udp_send_max_us below.

// Guards g_comm_tcp / g_tcp_client: loop() (core 1) replaces them on new client
// accept and reads via g_comm_tcp->update(), while uplink_task (core 0) sends
// through them — without this they race. Concurrent WiFiClient read (loop) +
// write (uplink_task) on the same socket from different cores corrupts lwIP's
// pbuf chain (assert "pbuf_free: p->ref > 0" -> reboot). Phase 7, UARTplat.md.
static SemaphoreHandle_t g_tcp_mutex = nullptr;
// Last connection state observed while holding g_tcp_mutex. Core 1 uses this
// cached byte when the sender currently owns the mutex; link-status telemetry
// must never wait behind a potentially slow TCP write and stop draining UART2.
static volatile bool g_tcp_connected_cached = false;

// Guards every write to Serial (UART0/USB-CDC to the GUI): g_usb.send(),
// g_usb_diag.send(), and the one-shot debug Serial.printf lines below.
// Arduino's Serial.write() isn't thread-safe against concurrent callers —
// log_uplink_task writes frames alongside uplink_task, so an
// unguarded interleaved write corrupts the CommLink framing
// on the wire (bytes from two frames mixed together), which the receiver
// can only see as a checksum failure / dropped frame. Same reasoning as
// g_tcp_mutex, just for the USB link instead of the TCP one.
static SemaphoreHandle_t g_usb_mutex = nullptr;

// Pending one-shot debug lines set by loop()/on_teensy_packet (core 1) and
// printed by uplink_task (core 0). g_usb_mutex keeps these plain-text writes
// from interleaving with binary frames from log_uplink_task.
static volatile bool g_dbg_version_mismatch_pending = false;
static volatile uint8_t g_dbg_mismatch_version       = 0;
static char           g_dbg_wifi_ip[24]              = {0};
static volatile bool  g_dbg_wifi_ip_pending           = false;
static char           g_dbg_tcp_client_ip[24]         = {0};
static volatile bool  g_dbg_tcp_client_pending        = false;

// Enqueue one frame for uplink_task. Never blocks: on a full queue, drops the
// oldest queued frame to make room (never blocks the caller — see 2.1).
static void enqueue_uplink(uint8_t type, uint8_t version, const void* payload, uint16_t len) {
    UplinkFrame f;
    f.type    = type;
    f.version = version;
    f.destinations = UPLINK_DEST_ALL;
    f.generation = 0;
    f.len     = len;
    memcpy(f.payload, payload, len);
    if (xQueueSend(g_uplink_q, &f, 0) != pdTRUE) {
        static UplinkFrame scratch;
        xQueueReceive(g_uplink_q, &scratch, 0);
        xQueueSend(g_uplink_q, &f, 0);
        ++g_uplink_drops;
    }
}

// Bulk log frames use a lossless, ACK-controlled lane. If this small queue is
// ever full, dropping the new frame is safe: the Teensy retains the chunk and
// resends it after its ACK timeout. Never evict XFER_BEGIN or an older chunk.
static void enqueue_uplink_log(uint8_t type, uint8_t version,
                               const void* payload, uint16_t len) {
    UplinkFrame f;
    f.type = type;
    f.version = version;
    f.destinations = g_log_uplink_dest;
    f.generation = g_log_generation;
    f.len = len;
    memcpy(f.payload, payload, len);
    if (xQueueSend(g_log_uplink_q, &f, 0) != pdTRUE) ++g_log_uplink_drops;
}

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

// TEST ONLY (Phase 9, UARTplat.md stress testing): armed by CMD_ID_TEST_INJECT_CORRUPT
// target=1 (WiFi), intercepted below — never forwarded to the Teensy. Consumed
// one-at-a-time in uplink_task's TELEM_A UDP send to confirm the GUI's own
// parser defenses actually detect and drop a deliberately malformed datagram.
static volatile uint8_t g_test_corrupt_wifi_remaining = 0;
static volatile uint8_t g_test_corrupt_wifi_mode = 1;  // CommLink::send() corrupt_mode_for_test value

static void forward_to_teensy_from(uint8_t incoming_dest, uint8_t type, uint8_t version,
                                   const uint8_t* payload, uint16_t len) {
    if (type == COMM_TYPE_COMMAND) {
#if WIFI_TRANSPORT_GATING
        // §2b: GUI telling the ESP32 which link it's reading. Intercepted here,
        // before the blind forward — the Teensy must never see this cmd_id.
        if (len >= 1 && payload[0] == CMD_ID_SET_TELEM_TRANSPORT) {
            bool suppress = (len >= 2) && (payload[1] != 0);
            g_active_telem_transport = suppress ? WIFI_DIAG_TRANSPORT_USB_ONLY
                                                 : WIFI_DIAG_TRANSPORT_WIFI_ONLY;
            return;
        }
#endif
        // TEST ONLY: target=1 (WiFi) is handled locally, never reaches the
        // Teensy — see g_test_corrupt_wifi_remaining declaration above.
        if (len >= 3 && payload[0] == CMD_ID_TEST_INJECT_CORRUPT && payload[2] == 1) {
            g_test_corrupt_wifi_remaining = payload[1];
            g_test_corrupt_wifi_mode = (len >= 4) ? payload[3] : 1;
            return;
        }
        // Peek (don't intercept — still forwarded below) at LOG_SUB_START/STOP
        // for g_recording_active: direct signal off the command itself, no
        // round-trip ack needed. See its declaration for what reads it.
        if (len >= 2 && payload[0] == CMD_ID_LOG) {
            // Route every response for this operation back to its requester.
            // This includes LIST/STATUS as well as bulk GET frames.
            g_log_uplink_dest = incoming_dest;
            if (payload[1] == LOG_SUB_GET) {
                // A repair GET supersedes every queued frame/ACK from the old
                // stream. Generation tagging also rejects one frame which may
                // already be blocked in the relay task when this reset occurs.
                ++g_log_generation;
                if (g_log_uplink_q) xQueueReset(g_log_uplink_q);
                if (g_log_ack_q) xQueueReset(g_log_ack_q);
            }
            if (payload[1] == LOG_SUB_START) {
                g_recording_active = true;
                uint32_t dur = 0;
                if (len >= 6) memcpy(&dur, payload + 2, sizeof(dur));
                // +500 ms matches the GUI's own auto-stop-fire margin
                // (log_transfer.py: duration_ms + 500) — covers command
                // transit time so this deadline doesn't fire early.
                g_recording_deadline_ms = (dur != 0) ? (millis() + dur + 500) : 0;
            } else if (payload[1] == LOG_SUB_STOP) {
                g_recording_active = false;
            }
        }
        g_teensy.send(type, version, payload, len);
        log_command(payload, len);
    }
}

static void forward_usb_to_teensy(uint8_t type, uint8_t version, uint8_t /*source*/,
                                  const uint8_t* payload, uint16_t len) {
    forward_to_teensy_from(UPLINK_DEST_USB, type, version, payload, len);
}

static void forward_tcp_to_teensy(uint8_t type, uint8_t version, uint8_t /*source*/,
                                  const uint8_t* payload, uint16_t len) {
    forward_to_teensy_from(UPLINK_DEST_TCP, type, version, payload, len);
}

// ── Telemetry snapshot (written by on_teensy_packet on core 1, read by loop on core 1)
// Both accesses are on the same core so no mutex is needed.
static TelemetryPayload s_telem_snap;
static bool             s_telem_a_rx   = false;  // TELEM_A received, waiting for B
static uint8_t          s_telem_a_seq  = 0;      // link seq of that TELEM_A (pairing check)
static bool             s_telem_fresh  = false;  // complete A+B pair ready in s_telem_snap

static bool send_uplink_frame(uint8_t type, uint8_t version, const uint8_t* payload,
                              uint16_t len, uint8_t destinations);

// ── Teensy → all outputs ──────────────────────────────────────────────────────

static void on_teensy_packet(uint8_t type, uint8_t version, uint8_t /*source*/,
                              const uint8_t* payload, uint16_t len) {
    // The relay task forwards bulk frames outside this UART callback. Unlike
    // the old firehose queue, the Teensy now waits for an ACK generated only
    // after each LOG_DATA frame is delivered, so this queue cannot be overrun
    // by a USB/TCP pause and the parser never blocks on an uplink write.
    if (type == COMM_TYPE_LOG_DATA || type == COMM_TYPE_LOG_INFO) {
        enqueue_uplink_log(type, version, payload, len);
    } else {
        // Forward every other packet upstream regardless of type — but only
        // enqueue here; uplink_task (core 0) performs the actual sends so
        // this parse path never blocks. See Phase 1, UARTplat.md.
        enqueue_uplink(type, version, payload, len);
    }

    // Track SD-log GET progress independent of telemetry (which the Teensy
    // pauses for the duration of a transfer — see teensy/src/main.cpp loop()).
    if (type == COMM_TYPE_LOG_INFO && len >= 1) {
        g_last_log_activity_ms = millis();
        uint8_t info_type = payload[0];
        if (info_type == LOG_INFO_XFER_BEGIN && len >= sizeof(LogInfoPayload)) {
            uint32_t total;
            memcpy(&total, payload + 7, sizeof(total));  // LogInfoPayload::total_chunks offset
            g_log_total_chunks = total;
            g_log_chunks_done  = 0;
            g_log_xfer_active  = true;
        } else if (info_type == LOG_INFO_XFER_END) {
            g_log_xfer_active = false;
        }
    } else if (type == COMM_TYPE_LOG_DATA && len >= sizeof(LogDataHeader)) {
        g_last_log_activity_ms = millis();
        uint32_t chunk_index;
        memcpy(&chunk_index, payload + 2, sizeof(chunk_index));  // LogDataHeader::chunk_index offset
        if (chunk_index + 1 > g_log_chunks_done) g_log_chunks_done = chunk_index + 1;
    }

    // Version mismatch: flag on either half so display updates immediately
    if ((type == COMM_TYPE_TELEM_A || type == COMM_TYPE_TELEM_B) && version != TELEM_VERSION) {
        if (!g_version_mismatch) {
            g_version_mismatch             = true;
            g_display_dirty                = true;
            g_dbg_mismatch_version         = version;
            g_dbg_version_mismatch_pending = true;
        }
        if (!g_teensy_ever_heard) g_display_dirty = true;
        g_last_teensy_ms    = millis();
        g_teensy_ever_heard = true;
        return;
    }

    // Reassemble split telemetry into s_telem_snap; signal loop() to process once.
    // All global updates happen in loop() so only the freshest snapshot is acted on.
    // Pairing requires B.seq == A.seq + 1 (Teensy sends the halves back-to-back) —
    // otherwise a lost B followed by a lost A would pair halves from different ticks.
    if (type == COMM_TYPE_TELEM_A && len == TELEM_A_LEN) {
        memcpy(&s_telem_snap, payload, TELEM_A_LEN);
        s_telem_a_seq = g_teensy.last_rx_seq();
        s_telem_a_rx  = true;
    } else if (type == COMM_TYPE_TELEM_B && len == TELEM_B_LEN && s_telem_a_rx) {
        s_telem_a_rx = false;
        if ((uint8_t)(g_teensy.last_rx_seq() - s_telem_a_seq) == 1) {
            memcpy(((uint8_t*)&s_telem_snap) + TELEM_A_LEN, payload, TELEM_B_LEN);
            s_telem_fresh = true;   // loop() will pick this up after draining the UART buffer
        }
        // Non-adjacent halves: discard the pair; the loss itself is already
        // counted by g_teensy.rx_seq_gaps().
    }

    if (!g_teensy_ever_heard) g_display_dirty = true;
    g_last_teensy_ms    = millis();
    g_teensy_ever_heard = true;
}

// Sends one frame to the selected output(s). Called by the normal and bulk-log
// uplink tasks; the bool reports whether at least one selected output accepted it.
static bool send_uplink_frame(uint8_t type, uint8_t version, const uint8_t* payload,
                              uint16_t len, uint8_t destinations) {
    bool delivered = false;
    if (type == COMM_TYPE_WIFI_DIAG) {
        if (xSemaphoreTake(g_usb_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            g_usb_diag.send(type, version, payload, len);
            delivered = true;
            xSemaphoreGive(g_usb_mutex);
        }
#if WIFI_ENABLED
        if (g_wifi_inited) {
            g_wifi_diag.send(type, version, payload, len);
            delivered = true;
        }
#endif
    } else if (type == COMM_TYPE_TELEM_FULL_WIFI) {
#if WIFI_ENABLED
        if (g_wifi_inited) {
            uint32_t t0 = micros();
            g_telem_udp.send(type, version, payload, len);
            delivered = true;
            uint16_t dt = (uint16_t)min((uint32_t)0xFFFF, (uint32_t)(micros() - t0));
            if (dt > g_udp_send_max_us) g_udp_send_max_us = dt;
        }
#endif
    } else {
        if ((destinations & UPLINK_DEST_USB) &&
            xSemaphoreTake(g_usb_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            delivered = g_usb.send(type, version, payload, len);
            xSemaphoreGive(g_usb_mutex);
        }
#if WIFI_ENABLED
        // Bounded, NOT portMAX_DELAY: the normal and bulk relay tasks
        // can contend for this mutex. If it is held unexpectedly long, skip
        // instead of blocking the UART parser or tripping a task watchdog.
        if ((destinations & UPLINK_DEST_TCP) &&
            xSemaphoreTake(g_tcp_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            bool tcp_connected = g_comm_tcp && g_tcp_client.connected();
            g_tcp_connected_cached = tcp_connected;
            if (tcp_connected) {
                // WiFiClient::write() (framework source) retries up to 10x
                // with a 1 s select() each — up to 10 s blocking — if the
                // peer stops draining the socket. That happens while
                // holding g_tcp_mutex, which then also stalls loop()'s
                // g_comm_tcp->update()/accept path on core 1, and since
                // uplink_task is the sole drain point for the WiFi UDP/USB
                // queue too, EVERYTHING (not just TCP) went dark until the
                // 10 s retry gave up — root-caused via esp_task_wdt, which
                // caught uplink_task stuck mid-send() every time (Phase 7,
                // UARTplat.md). Fix: our own quick (1 ms) write-readiness
                // check first, skip (drop and count) instead of risking it.
                int fd = g_tcp_client.fd();
                bool writable = false;
                if (fd >= 0) {
                    fd_set wset;
                    struct timeval tv = {0, 1000};  // 1 ms
                    FD_ZERO(&wset);
                    FD_SET(fd, &wset);
                    writable = select(fd + 1, nullptr, &wset, nullptr, &tv) > 0
                               && FD_ISSET(fd, &wset);
                }
                if (writable) {
                    uint32_t t0 = micros();
                    delivered = g_comm_tcp->send(type, version, payload, len);
                    uint16_t dt = (uint16_t)min((uint32_t)0xFFFF, (uint32_t)(micros() - t0));
                    if (dt > g_tcp_send_max_us) g_tcp_send_max_us = dt;
                } else {
                    ++g_tcp_send_skipped;
                }
            }
            xSemaphoreGive(g_tcp_mutex);
        }
#if !WIFI_TELEM_COMBINED
        if ((type == COMM_TYPE_TELEM_A || type == COMM_TYPE_TELEM_B) && g_wifi_inited
#if WIFI_TRANSPORT_GATING
            && g_active_telem_transport != WIFI_DIAG_TRANSPORT_USB_ONLY
#endif
        ) {
            // TEST ONLY (Phase 9, UARTplat.md): corrupt just the A half so the
            // GUI's parser has exactly one bad datagram per armed count.
            bool corrupt_wifi_now = (type == COMM_TYPE_TELEM_A) && g_test_corrupt_wifi_remaining > 0;
            uint8_t corrupt_wifi_mode = corrupt_wifi_now ? g_test_corrupt_wifi_mode : 0;
            if (corrupt_wifi_now) g_test_corrupt_wifi_remaining--;
            uint32_t t0 = micros();
            g_telem_udp.send(type, version, payload, len, corrupt_wifi_mode);
            uint16_t dt = (uint16_t)min((uint32_t)0xFFFF, (uint32_t)(micros() - t0));
            if (dt > g_udp_send_max_us) g_udp_send_max_us = dt;
        }
#endif
#endif
    }
    return delivered;
}

// ── Uplink task (core 0) ──────────────────────────────────────────────────────
// Writes the TCP client and telemetry/diag UDP sockets (see Phase 1,
// UARTplat.md); shares Serial (UART0/USB-CDC) with log_uplink_task,
// guarded by g_usb_mutex (see its declaration). Drains frames enqueued by
// on_teensy_packet() and send_wifi_diag() (both core 1) and performs the
// sends that used to run inline in the UART parse path. Uses a bounded wait
// (not portMAX_DELAY) so the pending debug-print flags below still get
// serviced promptly even if the Teensy link itself is down (e.g. WiFi/TCP
// connect events with no Teensy present).
static void uplink_task(void*) {
    UplinkFrame f;
    for (;;) {
        // Subscribed to the task watchdog in setup(): if anything below ever
        // blocks for esp_task_wdt_init()'s timeout, the panic handler reboots
        // us instead of the whole uplink pipeline (USB + WiFi) silently going
        // dark forever. This isn't just theoretical — it's exactly how the
        // g_comm_tcp->send() hang below was root-caused (Phase 7, UARTplat.md).
        esp_task_wdt_reset();

        if (xQueueReceive(g_uplink_q, &f, pdMS_TO_TICKS(100)) == pdTRUE) {
            send_uplink_frame(f.type, f.version, f.payload, f.len, f.destinations);
        }

        if ((g_dbg_version_mismatch_pending || g_dbg_wifi_ip_pending || g_dbg_tcp_client_pending)
            && xSemaphoreTake(g_usb_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (g_dbg_version_mismatch_pending) {
                g_dbg_version_mismatch_pending = false;
                Serial.printf("[VERSION MISMATCH] Teensy telem v%u, expected v%u — reflash both boards\n",
                              g_dbg_mismatch_version, TELEM_VERSION);
            }
            if (g_dbg_wifi_ip_pending) {
                g_dbg_wifi_ip_pending = false;
                Serial.print("[WiFi] IP: ");
                Serial.println(g_dbg_wifi_ip);
            }
            if (g_dbg_tcp_client_pending) {
                g_dbg_tcp_client_pending = false;
                Serial.print("[TCP] client: ");
                Serial.println(g_dbg_tcp_client_ip);
            }
            xSemaphoreGive(g_usb_mutex);
        }
    }
}

// Lossless bulk relay. The matching Teensy transfer keeps each chunk in RAM
// until the ACK below returns, so this task may safely wait on a slow USB/TCP
// output without blocking UART parsing or allowing later chunks to overrun it.
static void log_uplink_task(void*) {
    UplinkFrame f;
    for (;;) {
        if (xQueueReceive(g_log_uplink_q, &f, portMAX_DELAY) != pdTRUE) continue;
        if (f.generation != g_log_generation) continue;

        g_log_relay_busy = true;
        bool delivered = send_uplink_frame(f.type, f.version, f.payload,
                                           f.len, f.destinations);
        g_log_relay_busy = false;

        if (delivered && f.type == COMM_TYPE_LOG_DATA &&
            f.len >= sizeof(LogDataHeader)) {
            LogDataHeader hdr;
            memcpy(&hdr, f.payload, sizeof(hdr));
            LogRelayAck ack{hdr.file_index, hdr.chunk_index, f.generation};
            // If core 1 has not drained the previous ACK yet, the Teensy's
            // timeout will resend this same chunk and give us another chance.
            xQueueSend(g_log_ack_q, &ack, 0);
        }
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

static TFT_eSPI    tft;
static TFT_eSprite banner_sprite(&tft); // 320×44  mode banner  ~28 KB
static TFT_eSprite ah_sprite(&tft);     // 160×148 artificial horizon ~47 KB
static TFT_eSprite panel_sprite(&tft);  // 80×148  shared: hip panel then wm widget ~23 KB
static TFT_eSprite footer_sprite(&tft); // 320×48  footer status bar ~30 KB
// Total engineering sprites ~128 KB — panel_sprite is reused for hip then wm sequentially.
static TFT_eSprite eye_sprite(&tft);    // 100×100 face personality (reused L and R)
static TFT_eSprite mouth_sprite(&tft);  // 130×60  face personality mouth

// Layout (landscape 320×240)  44 + 148 + 48 = 240 ✓   160 + 80 + 80 = 320 ✓
#define BANNER_H   44
#define AH_X        0
#define AH_Y       BANNER_H
#define AH_W      160
#define AH_H      148
#define HIP_X     160
#define HIP_Y      BANNER_H
#define HIP_W      80
#define HIP_H     AH_H
#define GRAPH_X   240
#define GRAPH_Y    BANNER_H
#define GRAPH_W    80
#define GRAPH_H   AH_H
#define FOOTER_Y  192   // = BANNER_H + AH_H
#define FOOTER_H   48

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
        case RS_STANDING_UP: return tft.color565(255, 60, 0);   // red-orange
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
        case RS_STANDING_UP: return "STANDUP";
        default:             return "UNKNOWN";
    }
}

// MIRROR: _FAULT_DESCRIPTIONS in software/gui/tabs/telem_format.py — keep wording in
// sync whenever a fault is added/changed (ASCII only, no UTF-8, TFT font can't render it).
static const char* fault_description(uint8_t code) {
    switch (code) {
        case FAULT_NONE:               return "";
        case FAULT_IMU_ERROR:          return "IMU reported ERROR during startup";
        case FAULT_HIP_INIT_TIMEOUT:   return "No CAN reply from hip motors within 2 s of boot";
        case FAULT_HIP_FEEDBACK_LOST:  return "Hip CAN feedback timed out during operation (> 20 ms)";
        case FAULT_HIP_LARGE_POS_CMD:  return "Commanded hip position jump exceeded MAX_HIP_DELTA_RAD";
        case FAULT_CALIBRATION_TIMEOUT:return "Hardstop not found within CALIB_SAFETY_BOUND_RAD";
        case FAULT_HUMAN_ESTOP:        return "Human triggered ESTOP";
        case FAULT_PITCH_WATCHDOG:     return "|pitch| > 50 deg for > 200 ms - robot tipped";
        case FAULT_WHEEL_RUNAWAY:      return "Wheel velocity exceeded 2x soft governor limit";
        case FAULT_IMU_LOST:           return "IMU left NOMINAL while RUNNING/JUMPING (silence or heavy loss)";
        case FAULT_WHEEL_FEEDBACK_LOST:return "Wheel encoder timeout or ODrive error during operation";
        case FAULT_WHEEL_INIT_TIMEOUT: return "No CAN reply from wheel motors within 2 s of boot";
        case FAULT_STANDUP_FAILED:     return "Standup denied or failed - pitch out of recoverable range";
        default:                       return "Unknown fault";
    }
}

// ── Mode Banner helpers ───────────────────────────────────────────────────────

// Linear interpolate between two color565 values; t: 0=c0, 255=c1.
static uint16_t lerpColor(uint16_t c0, uint16_t c1, uint8_t t) {
    int r = ((c0 >> 11) & 0x1F) + ((int)((c1 >> 11 & 0x1F) - (c0 >> 11 & 0x1F)) * t >> 8);
    int g = ((c0 >>  5) & 0x3F) + ((int)((c1 >>  5 & 0x3F) - (c0 >>  5 & 0x3F)) * t >> 8);
    int b = ( c0        & 0x1F) + ((int)((c1        & 0x1F) - (c0        & 0x1F)) * t >> 8);
    return ((uint16_t)r << 11) | ((uint16_t)g << 5) | (uint16_t)b;
}

// Shift a color565 value brighter (+) or darker (-); G has 6 bits so it shifts twice.
static uint16_t brighten565(uint16_t c, int shift) {
    int r = ((c >> 11) & 0x1F) + shift;
    int g = ((c >>  5) & 0x3F) + shift * 2;
    int b = ( c        & 0x1F) + shift;
    return ((uint16_t)constrain(r, 0, 31) << 11)
         | ((uint16_t)constrain(g, 0, 63) <<  5)
         |  (uint16_t)constrain(b, 0, 31);
}

// ── Icon helpers ──────────────────────────────────────────────────────────────

// All icon helpers accept TFT_eSPI* so they work with both the display directly
// and any TFT_eSprite (which is a subclass of TFT_eSPI).

static void drawWifiIcon(TFT_eSPI* spr, int cx, int cy, bool active) {
    uint16_t col = active ? spr->color565(0, 150, 255) : spr->color565(55, 55, 65);
    spr->fillCircle(cx, cy + 11, 2, col);
    for (int ri = 0; ri < 3; ri++) {
        int r = 4 + ri * 4;
        for (int ad = -45; ad <= 45; ad += 5) {
            float ar = ad * (float)M_PI / 180.0f;
            int px = cx + (int)(r * sinf(ar));
            int py = (cy + 11) - (int)(r * cosf(ar));
            spr->drawFastHLine(px, py, 1, col);
        }
    }
}

static void drawCellBarsIcon(TFT_eSPI* spr, int x, int y, bool active) {
    uint16_t col = active ? TFT_GREEN : TFT_RED;
    const int heights[4] = {4, 7, 10, 13};
    for (int i = 0; i < 4; i++)
        spr->fillRect(x + i * 5, y + 13 - heights[i], 4, heights[i], col);
}

static void drawHipJointIcon(TFT_eSPI* spr, int x, int y) {
    uint16_t col = spr->color565(200, 200, 200);
    spr->drawLine(x - 7, y + 9, x, y, col);
    spr->drawLine(x, y, x + 7, y + 9, col);
    spr->fillCircle(x, y, 2, col);
}

static void drawWheelIcon(TFT_eSprite* spr, int cx, int cy, int r) {
    spr->drawCircle(cx, cy, r, TFT_WHITE);
    for (int i = 0; i < 4; i++) {
        float a = i * (float)M_PI / 2.0f;
        spr->drawLine(cx, cy, cx + (int)(r * cosf(a)), cy + (int)(r * sinf(a)), TFT_WHITE);
    }
    spr->fillCircle(cx, cy, 2, TFT_WHITE);
}

static void drawMiniArcGauge(TFT_eSPI* spr, int cx, int cy, int r,
                               float actual_pct, float cmd_pct) {
    uint16_t track = spr->color565(45, 45, 55);
    for (int ad = 0; ad <= 180; ad += 3) {
        float ar = (270 + ad) * (float)M_PI / 180.0f;
        int px = cx + (int)(r * sinf(ar));
        int py = cy - (int)(r * cosf(ar));
        spr->drawFastHLine(px, py, 1, track);
    }
    // Actual needle — thick cyan
    float a_rad = (270.0f + actual_pct * 180.0f) * (float)M_PI / 180.0f;
    int anx = cx + (int)(r * sinf(a_rad));
    int any = cy - (int)(r * cosf(a_rad));
    uint16_t act_col = spr->color565(0, 200, 255);
    spr->drawLine(cx, cy, anx, any, act_col);
    spr->drawLine(cx + 1, cy, anx + 1, any, act_col);
    // Commanded needle — thin yellow
    float c_rad = (270.0f + cmd_pct * 180.0f) * (float)M_PI / 180.0f;
    int cnx = cx + (int)((r - 2) * sinf(c_rad));
    int cny = cy - (int)((r - 2) * cosf(c_rad));
    spr->drawLine(cx, cy, cnx, cny, TFT_YELLOW);
    spr->fillCircle(cx, cy, 2, TFT_WHITE);
}

static void drawControllerButton(TFT_eSPI* spr, int x, int y,
                                  const char* label, bool active) {
    uint16_t bg  = active ? spr->color565(0, 110, 0)   : spr->color565(14, 14, 20);
    uint16_t fg  = active ? TFT_WHITE                   : spr->color565(75, 75, 80);
    uint16_t brd = active ? spr->color565(0, 190, 0)   : spr->color565(45, 45, 55);
    spr->fillRoundRect(x, y, 14, 18, 2, bg);
    spr->drawRoundRect(x, y, 14, 18, 2, brd);
    spr->setTextColor(fg);
    spr->setTextSize(1);
    spr->setCursor(x + 1, y + 5);
    spr->print(label);
}

// ── Mode Banner ───────────────────────────────────────────────────────────────

// Banner override while an SD-log GET is streaming — telemetry is paused for
// the whole transfer (see teensy/src/main.cpp loop()), so the normal banner
// would just sit on frozen values; this makes the intentional quiet spell
// obvious instead, with a live percent readout. See log_xfer_is_downloading().
static void drawDownloadBanner(uint8_t pct) {
    static uint8_t prev_pct = 0xFF;
    if (pct == prev_pct) return;
    prev_pct = pct;

    banner_sprite.fillSprite(tft.color565(4, 10, 24));

    const int BAR_X = 20, BAR_Y = BANNER_H - 14, BAR_W = 280, BAR_H = 8;
    banner_sprite.drawRect(BAR_X, BAR_Y, BAR_W, BAR_H, TFT_WHITE);
    int fill_w = (int)((uint32_t)(BAR_W - 2) * pct / 100u);
    banner_sprite.fillRect(BAR_X + 1, BAR_Y + 1, fill_w, BAR_H - 2, tft.color565(0, 150, 255));

    char label[24];
    snprintf(label, sizeof(label), "DOWNLOADING %u%%", (unsigned)pct);
    int tsize    = 2;
    int label_px = (int)strlen(label) * 6 * tsize;
    int cx       = max(4, (320 - label_px) / 2);
    banner_sprite.setTextColor(TFT_WHITE);
    banner_sprite.setTextSize(tsize);
    banner_sprite.setCursor(cx, 4);
    banner_sprite.print(label);

    banner_sprite.pushSprite(0, 0);
}

static void drawModeBanner(uint8_t state, bool active, uint8_t fault, bool mismatch, float vbus, uint8_t profile) {
    static uint8_t  prev_state      = 0xFF;
    static bool     prev_active     = false;
    static uint8_t  prev_fault      = 0xFF;
    static bool     prev_mismatch   = false;
    static int      prev_batt_bars  = -99;
    static bool     prev_blink_on   = true;
    static uint8_t  prev_profile    = 0xFF;
    static uint32_t prev_mq_tick    = 0xFFFFFFFF;
    static bool     prev_rec_on     = false;

    bool rec_on = recording_is_active() && ((millis() / 400) % 2 == 0);

    // 6S LiPo: 4.2V/cell × 6 = 25.2V full, 3.5V/cell × 6 = 21.0V empty
    const float BATT_VMAX = 25.2f, BATT_VMIN = 21.0f;
    int  batt_bars = (vbus > 5.0f)
        ? (int)(constrain((vbus - BATT_VMIN) / (BATT_VMAX - BATT_VMIN), 0.0f, 1.0f) * 5.0f + 0.5f)
        : -1;
    bool blink_on  = (batt_bars >= 0 && batt_bars <= 1) ? ((millis() / 500) % 2 == 0) : true;

    // Per-mode animation interval — drives dirty-check frame rate
    uint32_t anim_interval;
    if (!active || mismatch) {
        anim_interval = 100;
    } else {
        switch (state) {
            case RS_STANDBY:    anim_interval = 80;  break;
            case RS_ESTOP:      anim_interval = 16;  break;
            case RS_CMD_REJECT: anim_interval = 62;  break;
            case RS_JUMPING:    anim_interval = 16;  break;
            default: { static const uint32_t kMs[3] = {30, 20, 10};
                       anim_interval = kMs[profile < 3 ? profile : 0]; break; }
        }
    }
    uint32_t mq_tick = millis() / anim_interval;

    if (!g_banner_force_redraw &&
        state == prev_state && active == prev_active && fault == prev_fault
            && mismatch == prev_mismatch && batt_bars == prev_batt_bars
            && blink_on == prev_blink_on && profile == prev_profile
            && mq_tick == prev_mq_tick && rec_on == prev_rec_on) return;
    g_banner_force_redraw = false;
    prev_state = state; prev_active = active; prev_fault = fault;
    prev_mismatch = mismatch; prev_batt_bars = batt_bars; prev_blink_on = blink_on;
    prev_profile = profile; prev_mq_tick = mq_tick; prev_rec_on = rec_on;

    uint32_t now_ms = millis();

    // ── Per-mode gradient background ──────────────────────────────────────────

    if (!active) {
        // Disconnected: slow breathing pulse between dark slate and dim blue
        uint8_t bri = sin8((uint8_t)(now_ms / 8));
        uint16_t lo = tft.color565( 8, 10, 20);
        uint16_t hi = tft.color565(18, 26, 55);
        banner_sprite.fillSprite(lerpColor(lo, hi, bri));

    } else if (mismatch) {
        banner_sprite.fillSprite(TFT_RED);

    } else switch (state) {

        case RS_STARTUP: {
            // Deep blue → cyan gradient comet sweeping left→right (~1 s period)
            uint16_t lo = tft.color565(0, 15, 80);
            uint16_t hi = tft.color565(0, 190, 255);
            int offset  = (int)(now_ms / 3) % 320;
            for (int x = 0; x < 320; x++) {
                uint8_t t = sin8((uint8_t)(((x + 320 - offset) % 320) * 256 / 320));
                banner_sprite.drawFastVLine(x, 0, BANNER_H, lerpColor(lo, hi, t));
            }
            break;
        }

        case RS_CALIBRATION: {
            // Gaussian spotlight beam sweeping left↔right over blue base
            uint16_t base = tft.color565(0, 10, 160);
            banner_sprite.fillSprite(base);
            int t       = (int)(now_ms % 2400u);
            int sweep_x = (t < 1200) ? (t * 320 / 1200) : ((2400 - t) * 320 / 1200);
            const int BW = 22;
            for (int bx = max(0, sweep_x - BW); bx <= min(319, sweep_x + BW); bx++) {
                int d    = abs(bx - sweep_x);
                int frac = (BW - d) * 255 / (BW + 1);
                // Gaussian falloff: frac * frac / 255
                uint8_t v = (uint8_t)((frac * frac) >> 8);
                banner_sprite.drawFastVLine(bx, 0, BANNER_H, lerpColor(base, TFT_WHITE, v));
            }
            break;
        }

        case RS_STANDBY: {
            // Brown→yellow dense diagonal shimmer; old P3 speed is now P1
            static const uint32_t kStandbyDiv[3] = {13, 6, 3};
            uint16_t lo = tft.color565(130, 55, 5);
            uint16_t hi = tft.color565(250, 210, 0);
            uint32_t t  = now_ms / kStandbyDiv[profile < 3 ? profile : 0];
            for (int y = 0; y < BANNER_H; y++) {
                for (int x = 0; x < 320; x++) {
                    uint8_t phase = (uint8_t)(((x + y * 2) * 3 + t) & 0xFF);
                    banner_sprite.drawFastHLine(x, y, 1, lerpColor(lo, hi, sin8(phase)));
                }
            }
            break;
        }

        case RS_RUNNING: {
            // Light-green comets with comet tails bounce side→centre→side
            static const uint32_t kRunPeriod[3] = {2000, 1200, 700};
            uint32_t period = kRunPeriod[profile < 3 ? profile : 0];
            uint32_t ph     = now_ms % period;
            uint32_t half   = period / 2;
            int32_t bounce  = (ph < half)
                ? (int32_t)(ph * 160 / half)
                : 160 - (int32_t)((ph - half) * 160 / half);
            int pos_a = (int)bounce;
            int pos_b = 319 - (int)bounce;
            // dir: +1 = comet moving right, -1 = moving left
            int dir_a = (ph < half) ? +1 : -1;
            int dir_b = (ph < half) ? -1 : +1;

            uint16_t col_bg    = tft.color565(0, 140, 25);
            uint16_t col_comet = tft.color565(210, 255, 210);
            banner_sprite.fillSprite(col_bg);

            for (int x = 0; x < 320; x++) {
                // Tail fades gently behind, sharp cutoff in front
                int dx_a = x - pos_a;
                int ba = (dx_a * dir_a > 0) ? max(0, 255 - abs(dx_a) * 8)
                                             : max(0, 255 - abs(dx_a) * 55);
                int dx_b = x - pos_b;
                int bb = (dx_b * dir_b > 0) ? max(0, 255 - abs(dx_b) * 8)
                                             : max(0, 255 - abs(dx_b) * 55);
                int bri = min(255, ba + bb);
                if (bri > 0)
                    banner_sprite.drawFastVLine(x, 0, BANNER_H,
                        lerpColor(col_bg, col_comet, (uint8_t)bri));
            }
            break;
        }

        case RS_ESTOP: {
            // Radial ripple expanding from center, red
            const int cx = 160, cy = BANNER_H / 2;
            uint8_t phase = (uint8_t)(now_ms / 3);
            uint16_t lo = tft.color565(45, 0, 0);
            uint16_t hi = tft.color565(230, 0, 0);
            for (int y = 0; y < BANNER_H; y++) {
                int dy2 = (y - cy) * (y - cy);
                for (int x = 0; x < 320; x++) {
                    int dx   = x - cx;
                    int dist = (int)sqrtf((float)(dx * dx + dy2));
                    banner_sprite.drawFastHLine(x, y, 1, lerpColor(lo, hi, sin8((uint8_t)(dist * 8 - phase))));
                }
            }
            break;
        }

        case RS_MANUAL: {
            // Deep violet → magenta, very slow diagonal scroll
            uint16_t lo = tft.color565(38, 0, 75);
            uint16_t hi = tft.color565(175, 0, 200);
            int offset  = (int)(now_ms * 0.04f) & 0xFF;
            for (int y = 0; y < BANNER_H; y++) {
                for (int x = 0; x < 320; x++) {
                    uint8_t phase = (uint8_t)((x - y + offset) & 0xFF);
                    banner_sprite.drawFastHLine(x, y, 1, lerpColor(lo, hi, sin8(phase)));
                }
            }
            break;
        }

        case RS_CMD_REJECT: {
            // Rapid orange/white strobe at ~8 Hz, auto-clears
            bool flash = (now_ms / 62) & 1;
            uint16_t col = flash ? tft.color565(255, 165, 0) : tft.color565(55, 28, 0);
            banner_sprite.fillSprite(col);
            break;
        }

        case RS_JUMPING: {
            // Fast vertical sweep lime → white, top→bottom
            uint16_t lo = tft.color565(20, 200, 0);
            uint16_t hi = tft.color565(210, 255, 180);
            int offset  = (int)(now_ms / 3) % BANNER_H;
            for (int y = 0; y < BANNER_H; y++) {
                uint8_t t = sin8((uint8_t)(((y + offset) % BANNER_H) * 256 / BANNER_H));
                banner_sprite.drawFastHLine(0, y, 320, lerpColor(lo, hi, t));
            }
            break;
        }

        default:
            banner_sprite.fillSprite(tft.color565(15, 17, 25));
            break;
    }

    // ── Text overlay ──────────────────────────────────────────────────────────
    if (mismatch) {
        banner_sprite.setTextColor(TFT_WHITE);
        banner_sprite.setTextSize(2);
        banner_sprite.setCursor(10, (BANNER_H - 16) / 2);
        banner_sprite.print("VER MISMATCH");
    } else {
        bool show_fault = active && state == RS_ESTOP && fault != FAULT_NONE;

        const char* label = active ? mode_name(state) : "NO TEENSY";
        int tsize    = show_fault ? 2 : 3;
        int label_px = (int)strlen(label) * 6 * tsize;
        int cx       = max(8, (320 - label_px) / 2);
        int text_y   = show_fault ? 3 : (BANNER_H - 8 * tsize) / 2;

        const int tpad = 3;
        banner_sprite.fillRect(cx - tpad, text_y - tpad, label_px + tpad * 2, 8 * tsize + tpad * 2,
                               tft.color565(10, 10, 15));
        banner_sprite.setTextColor(TFT_WHITE);
        banner_sprite.setTextSize(tsize);
        banner_sprite.setCursor(cx, text_y);
        banner_sprite.print(label);

        if (active && !show_fault) {
            uint16_t dot_col;
            switch (profile) {
                case 0:  dot_col = tft.color565(  0, 230,  80); break;
                case 1:  dot_col = TFT_YELLOW;                   break;
                default: dot_col = tft.color565(255,  80,   0); break;
            }
            banner_sprite.fillRect(2, 3, 40, BANNER_H - 6, TFT_BLACK);
            banner_sprite.fillCircle(9, BANNER_H / 2, 4, dot_col);
            banner_sprite.setTextColor(TFT_WHITE);
            banner_sprite.setTextSize(2);
            banner_sprite.setCursor(16, (BANNER_H - 16) / 2);
            banner_sprite.printf("P%u", (unsigned)(profile + 1u));
        }

        if (show_fault) {
            // Drawn last (and skips the profile badge above) so nothing overlaps it.
            const char* fdesc     = fault_description(fault);
            int         fdesc_px  = (int)strlen(fdesc) * 6;
            int         fdesc_cx  = max(2, (320 - fdesc_px) / 2);
            banner_sprite.setTextColor(TFT_WHITE);
            banner_sprite.setTextSize(1);
            banner_sprite.setCursor(fdesc_cx, BANNER_H - 10);
            banner_sprite.print(fdesc);
        }
    }

    // ── Recording icon — small blinking dot, left of the battery icon. Sits in
    // the gap between the mode label and battery (uncrowded at every label
    // width tuning.md's states use). Blink phase (rec_on) is already folded
    // into the dirty-check above, so this repaints on every on/off transition.
    if (rec_on) {
        banner_sprite.fillCircle(254, BANNER_H / 2, 6, tft.color565(0, 120, 255));
        banner_sprite.drawCircle(254, BANNER_H / 2, 6, TFT_WHITE);
    }

    // ── Battery icon ──────────────────────────────────────────────────────────
    const int BX = 276, BY = 8, BW = 34, BH = 28;
    const uint16_t BATT_BG = tft.color565(8, 10, 20);
    banner_sprite.fillRect(BX - 2, BY - 2, BW + 8, BH + 4, tft.color565(80, 85, 100));
    banner_sprite.fillRect(BX - 1, BY - 1, BW + 6, BH + 2, BATT_BG);
    if (batt_bars >= 0 && blink_on) {
        float    pct = batt_bars / 5.0f;
        uint16_t bc  = (pct > 0.5f) ? TFT_GREEN : (pct > 0.2f) ? TFT_YELLOW : TFT_RED;
        banner_sprite.drawRect(BX, BY, BW, BH, TFT_WHITE);
        banner_sprite.fillRect(BX + BW, BY + 9, 4, 10, TFT_WHITE);
        banner_sprite.fillRect(BX + 2, BY + 2, BW - 4, BH - 4, tft.color565(15, 15, 20));
        for (int i = 0; i < 5; i++) {
            int bx = BX + 2 + i * 6;
            if (i < batt_bars) banner_sprite.fillRect(bx, BY + 3, 5, BH - 6, bc);
        }
    } else if (batt_bars >= 0 && !blink_on) {
        banner_sprite.drawRect(BX, BY, BW, BH, tft.color565(120, 0, 0));
        banner_sprite.fillRect(BX + BW, BY + 9, 4, 10, tft.color565(120, 0, 0));
    }

    banner_sprite.pushSprite(0, 0);
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

// ── Hip Panel (panel_sprite, 80×148, pushed to HIP_X,HIP_Y) ─────────────────
// panel_sprite is shared with drawWheelMotors — draw hip first, then wm.

static void drawHipPanel(float hip_l, float hip_r, float curr_l, float curr_r,
                          float cmd_l, float cmd_r, uint16_t health_flags) {
    panel_sprite.fillSprite(COL_GRAPH_BG);

    drawHipJointIcon(&panel_sprite, 40, 10);

    const int cy = 58, r = 22;
    auto hip_pct = [](float rad) -> float {
        return constrain((rad - HIP_DISPLAY_MIN) / (HIP_DISPLAY_MAX - HIP_DISPLAY_MIN), 0.0f, 1.0f);
    };
    drawMiniArcGauge(&panel_sprite, 20, cy, r, hip_pct(hip_l), hip_pct(cmd_l));
    drawMiniArcGauge(&panel_sprite, 60, cy, r, hip_pct(hip_r), hip_pct(cmd_r));

    panel_sprite.setTextSize(1);
    panel_sprite.setTextColor(COL_DIM);
    panel_sprite.setCursor(17, cy + 4);
    panel_sprite.print("L");
    panel_sprite.setCursor(57, cy + 4);
    panel_sprite.print("R");

    const int bar_y = 88, bar_h = 8, bar_w = 34;
    const float CURR_MAX = 8.0f;
    auto curr_col = [&](float c) -> uint16_t {
        if (c < 3.0f) return TFT_GREEN;
        if (c < 6.0f) return TFT_YELLOW;
        return TFT_RED;
    };
    auto drawCurrBar = [&](int bx, float curr) {
        int fill = (int)(constrain(curr / CURR_MAX, 0.0f, 1.0f) * bar_w);
        panel_sprite.fillRect(bx, bar_y, bar_w, bar_h, panel_sprite.color565(15, 15, 20));
        if (fill > 0) panel_sprite.fillRect(bx, bar_y, fill, bar_h, curr_col(curr));
        panel_sprite.drawRect(bx - 1, bar_y - 1, bar_w + 2, bar_h + 2, panel_sprite.color565(45, 45, 55));
        panel_sprite.setTextColor(panel_sprite.color565(190, 190, 190));
        panel_sprite.setCursor(bx, bar_y + bar_h + 3);
        panel_sprite.printf("%.1fA", curr);
    };
    drawCurrBar(3,  curr_l);
    drawCurrBar(43, curr_r);

    const int dot_y = 116;
    panel_sprite.fillCircle(20, dot_y, 4, (health_flags & HEALTH_HIP_L_OK) ? TFT_GREEN : TFT_RED);
    panel_sprite.fillCircle(60, dot_y, 4, (health_flags & HEALTH_HIP_R_OK) ? TFT_GREEN : TFT_RED);

    panel_sprite.setTextSize(1);
    panel_sprite.setTextColor(panel_sprite.color565(140, 140, 150));
    panel_sprite.setCursor(2, 128);
    panel_sprite.printf("%.2f", hip_l);
    panel_sprite.setCursor(42, 128);
    panel_sprite.printf("%.2f", hip_r);

    panel_sprite.drawRect(0, 0, HIP_W, HIP_H, TFT_WHITE);
    panel_sprite.pushSprite(HIP_X, HIP_Y);
}

// ── Wheel Motor Widget (sprite-based, 80×148) ─────────────────────────────────

static void drawWheelMotors(float vel_l, float vel_r,
                             uint8_t state_l, uint8_t state_r,
                             uint32_t err_l, uint32_t err_r,
                             uint8_t mode, float v_ref, float wheel_vel_avg) {
    panel_sprite.fillSprite(COL_GRAPH_BG);
    panel_sprite.setTextSize(1);

    // Wheel icon
    drawWheelIcon(&panel_sprite, 40, 11, 8);

    // Velocity bars (bidirectional, zero centred, 26×60 each)
    const float VEL_MAX  = 20.0f;
    const int   bar_w    = 26;
    const int   bar_h    = 60;
    const int   bar_top  = 24;
    const int   bar_cy   = bar_top + bar_h / 2;
    const int   bar_lx   = 4;
    const int   bar_rx   = 50;
    const uint16_t COL_EMPTY = panel_sprite.color565(20, 20, 28);

    auto drawVelBar = [&](int bx, float vel) {
        float cv = constrain(vel / VEL_MAX, -1.0f, 1.0f);
        int fill = (int)(fabsf(cv) * (bar_h / 2));
        uint16_t fc = (vel >= 0.0f) ? panel_sprite.color565(0, 200, 100)
                                     : panel_sprite.color565(255, 100, 0);
        panel_sprite.fillRect(bx, bar_top, bar_w, bar_h, COL_EMPTY);
        if (fill > 0) {
            int fy = (vel >= 0.0f) ? bar_cy - fill : bar_cy;
            panel_sprite.fillRect(bx, fy, bar_w, fill, fc);
        }
        panel_sprite.drawFastHLine(bx, bar_cy, bar_w, COL_ZERO_LINE);
        panel_sprite.drawRect(bx - 1, bar_top - 1, bar_w + 2, bar_h + 2, TFT_WHITE);
    };
    drawVelBar(bar_lx, vel_l);
    drawVelBar(bar_rx, vel_r);

    // Velocity numbers
    int num_y = bar_top + bar_h + 3;
    panel_sprite.setTextColor(TFT_WHITE);
    panel_sprite.setCursor(bar_lx, num_y);
    panel_sprite.printf("%.1f", vel_l);
    panel_sprite.setCursor(bar_rx, num_y);
    panel_sprite.printf("%.1f", vel_r);

    // State dots
    int dot_y = num_y + 11;
    auto state_col = [](uint8_t s) -> uint16_t {
        return (s == 8) ? TFT_GREEN : (s == 0) ? TFT_YELLOW : TFT_RED;
    };
    panel_sprite.fillCircle(bar_lx + bar_w / 2, dot_y, 4, state_col(state_l));
    panel_sprite.fillCircle(bar_rx + bar_w / 2, dot_y, 4, state_col(state_r));

    // Mode label centred between bars
    const char* mode_str;
    switch (mode) {
        case 1: mode_str = "VEL"; break;
        case 2: mode_str = "POS"; break;
        case 3: mode_str = "TRQ"; break;
        default: mode_str = "IDL"; break;
    }
    panel_sprite.setTextColor(panel_sprite.color565(150, 150, 160));
    panel_sprite.setCursor(34, dot_y - 4);
    panel_sprite.print(mode_str);

    // Full-width speed bar (±2.5 m/s, wheel_vel_avg actual + v_ref command tick)
    const float SPD_MAX = 2.5f;
    const int   sb_x    = 2;
    const int   sb_y    = dot_y + 13;
    const int   sb_w    = 76;
    const int   sb_h    = 10;
    const int   sb_cx   = sb_x + sb_w / 2;
    panel_sprite.fillRect(sb_x, sb_y, sb_w, sb_h, panel_sprite.color565(15, 15, 20));
    panel_sprite.drawRect(sb_x - 1, sb_y - 1, sb_w + 2, sb_h + 2, panel_sprite.color565(45, 45, 55));
    float sv  = constrain(wheel_vel_avg, -SPD_MAX, SPD_MAX);
    int   spx = (int)(sv / SPD_MAX * (sb_w / 2));
    uint16_t svc = (wheel_vel_avg >= 0.0f) ? TFT_CYAN : TFT_MAGENTA;
    if (spx > 0)       panel_sprite.fillRect(sb_cx, sb_y + 1, spx, sb_h - 2, svc);
    else if (spx < 0)  panel_sprite.fillRect(sb_cx + spx, sb_y + 1, -spx, sb_h - 2, svc);
    panel_sprite.drawFastVLine(sb_cx, sb_y, sb_h, TFT_WHITE);
    // v_ref command tick (yellow)
    float vrf = constrain(v_ref, -SPD_MAX, SPD_MAX);
    int   vrx = sb_cx + (int)(vrf / SPD_MAX * (sb_w / 2));
    panel_sprite.drawFastVLine(vrx, sb_y - 2, sb_h + 4, TFT_YELLOW);

    panel_sprite.setTextColor(TFT_WHITE);
    panel_sprite.setCursor(2, sb_y + sb_h + 3);
    panel_sprite.printf("%.2fm/s", wheel_vel_avg);

    if (err_l || err_r) {
        panel_sprite.setTextColor(TFT_RED);
        panel_sprite.setCursor(2, sb_y + sb_h + 13);
        panel_sprite.print("ERR!");
    }

    panel_sprite.drawRect(0, 0, GRAPH_W, GRAPH_H, TFT_WHITE);
    panel_sprite.pushSprite(GRAPH_X, GRAPH_Y);
}

// ── Footer Status Bar (footer_sprite, 320×48, pushed to 0,FOOTER_Y) ──────────

static void drawFooter(uint8_t state, bool active, bool rc_alive, uint32_t pkt_age_ms,
                       uint16_t health_flags, float v_ref, float omega_cmd_rds,
                       float ff1_out, float ff2_out, uint8_t jump_state) {
    footer_sprite.fillSprite(COL_FOOTER_BG);
    footer_sprite.setTextSize(1);

    // ── Row 1 (y 0-22): connection status ─────────────────────────────────────

    // Heartbeat dot
    static bool hb_phase = false;
    if (active) hb_phase = !hb_phase;
    uint16_t hb_col = active ? mode_color(state) : COL_DIM;
    footer_sprite.fillCircle(10, 11, active ? (hb_phase ? 7 : 5) : 3, hb_col);

    // Packet age
    footer_sprite.setTextColor(active ? TFT_WHITE : COL_DIM);
    footer_sprite.setCursor(22, 7);
    if (active)
        footer_sprite.printf("%4lums", (unsigned long)min((uint32_t)9999, pkt_age_ms));
    else
        footer_sprite.print("NO PKT");

    // UART dot
    footer_sprite.fillCircle(72, 11, 4, active ? TFT_GREEN : TFT_YELLOW);
    footer_sprite.setTextColor(active ? TFT_GREEN : TFT_YELLOW);
    footer_sprite.setCursor(79, 7);
    footer_sprite.print(active ? "UART" : "WAIT");

    // WiFi arc icon
    drawWifiIcon(&footer_sprite, 114, 0, g_wifi_inited);

    // RC bars icon
    drawCellBarsIcon(&footer_sprite, 136, 5, rc_alive);

    // UART health (CRC/GAP) — right side
    uint32_t crc_tot = g_uart_crc_drops;
    uint32_t gap_tot = g_uart_seq_gaps;
    if (crc_tot || gap_tot) {
        bool     ua   = (g_uart_crc_rate > 0 || g_uart_gap_rate > 0);
        uint16_t ucol = ua ? TFT_RED : footer_sprite.color565(255, 120, 0);
        footer_sprite.setTextColor(ucol);
        footer_sprite.setCursor(162, 7);
        if (crc_tot && gap_tot)
            footer_sprite.printf("CRC:%lu G:%lu%s", crc_tot, gap_tot, ua ? "!" : "");
        else if (crc_tot)
            footer_sprite.printf("CRC:%lu%s", crc_tot, ua ? "!" : "");
        else
            footer_sprite.printf("GAP:%lu%s", gap_tot, ua ? "!" : "");
    }

    // Divider between rows
    footer_sprite.drawFastHLine(0, 23, 320, footer_sprite.color565(35, 35, 45));

    // ── Row 2 (y 25-46): controller radio buttons ─────────────────────────────
    bool lqr_on = (health_flags & HEALTH_LQR_ACTIVE) != 0;
    bool vp_on  = (v_ref != 0.0f);
    bool yp_on  = (omega_cmd_rds != 0.0f);
    bool f1_on  = (ff1_out != 0.0f);
    bool f2_on  = (ff2_out != 0.0f);
    bool jm_on  = (jump_state != 0);

    const char* labels[6]  = {"LQ", "VP", "YP", "F1", "F2", "JM"};
    bool        states_[6] = {lqr_on, vp_on, yp_on, f1_on, f2_on, jm_on};
    for (int i = 0; i < 6; i++)
        drawControllerButton(&footer_sprite, 2 + i * 16, 25, labels[i], states_[i]);

    footer_sprite.pushSprite(0, FOOTER_Y);
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
    banner_sprite.setColorDepth(16);
    banner_sprite.createSprite(320, BANNER_H);

    ah_sprite.setColorDepth(16);
    ah_sprite.createSprite(AH_W, AH_H);

    panel_sprite.setColorDepth(16);
    panel_sprite.createSprite(GRAPH_W, GRAPH_H);  // 80×148, reused for hip + wm

    footer_sprite.setColorDepth(16);
    footer_sprite.createSprite(320, FOOTER_H);

    tft.fillScreen(TFT_BLACK);

    tft.drawFastVLine(HIP_X,   AH_Y, AH_H, COL_DIVIDER);
    tft.drawFastVLine(GRAPH_X, AH_Y, AH_H, COL_DIVIDER);
    tft.drawFastHLine(0, FOOTER_Y, 320, COL_DIVIDER);
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
            banner_sprite.deleteSprite();
            ah_sprite.deleteSprite();
            panel_sprite.deleteSprite();
            footer_sprite.deleteSprite();
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
    bool connected = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < TEENSY_LINK_TIMEOUT_MS);

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

    static bool was_downloading = false;
    bool downloading = log_xfer_is_downloading();
    if (was_downloading && !downloading) g_banner_force_redraw = true;  // leaving: force the real banner back
    was_downloading = downloading;
    if (downloading) {
        drawDownloadBanner(log_xfer_percent());
        return;  // other panels reflect frozen (paused) telemetry — nothing new to redraw
    }

    uint8_t  state     = g_robot_state;
    uint8_t  fault     = g_fault_code;
    bool     active    = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < TEENSY_LINK_TIMEOUT_MS);
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
    uint8_t  profile    = g_telem_active_profile;
    uint16_t health     = g_telem_health_flags;
    float    hip_l_cmd  = g_telem_hip_l_cmd_rad;
    float    hip_r_cmd  = g_telem_hip_r_cmd_rad;
    float    v_ref      = g_telem_v_ref;
    float    omega_cmd  = g_telem_omega_cmd_rds;
    float    ff1_out    = g_telem_ff1_out;
    float    ff2_out    = g_telem_ff2_out;
    uint8_t  jmp_state  = g_telem_jump_state;

    // A disabled/unrequested wheel motor's vbus reads a permanent 0 (Teensy never
    // polls it) — exclude it from the average instead of dragging a real ~24V
    // reading down to ~12V. Mirrors the same fix in software/gui/main.py.
    constexpr float VBUS_MIN_VALID = 5.0f;
    float vbus_sum = 0.0f;
    uint8_t vbus_n = 0;
    if (wm_l_vbus > VBUS_MIN_VALID) { vbus_sum += wm_l_vbus; vbus_n++; }
    if (wm_r_vbus > VBUS_MIN_VALID) { vbus_sum += wm_r_vbus; vbus_n++; }
    float vbus_avg = (vbus_n > 0) ? (vbus_sum / vbus_n) : 0.0f;
    drawModeBanner(state, active, fault, g_version_mismatch, active ? vbus_avg : 0.0f, active ? profile : 0);
    drawArtificialHorizon(active ? pitch : 0.0f, active ? roll : 0.0f);
    drawHipPanel(hip_l, hip_r, curr_l, curr_r,
                 active ? hip_l_cmd : 0.0f, active ? hip_r_cmd : 0.0f,
                 active ? health : 0);
    drawWheelMotors(active ? wm_l_vel : 0.0f, active ? wm_r_vel : 0.0f,
                    active ? wm_l_state : 0, active ? wm_r_state : 0,
                    active ? wm_l_error : 0, active ? wm_r_error : 0,
                    active ? wm_mode : 0,
                    active ? v_ref : 0.0f, active ? wheel_vel : 0.0f);
    drawFooter(state, active, rc_alive, pkt_age,
               active ? health : 0,
               active ? v_ref : 0.0f, active ? omega_cmd : 0.0f,
               active ? ff1_out : 0.0f, active ? ff2_out : 0.0f,
               active ? jmp_state : 0);
}

// ── Display task (core 0) ─────────────────────────────────────────────────────
// Owns all TFT SPI operations. loop() on core 1 never touches the display,
// so UART parsing runs uninterrupted regardless of render time.

static void display_task(void*) {
    pinMode(TFT_BLK, OUTPUT);
    digitalWrite(TFT_BLK, HIGH);
    tft.init();
    tft.setRotation(3);

    // One-shot boot splash (~1 s) so a physical reboot/reflash is unmistakable
    // on the bench — the normal engineering/face display only appears once
    // initDisplay() below completes, easy to miss if you're not staring at
    // the screen at that exact instant. See neo_task's boot rainbow, same idea.
    {
        const char* label = "BOOTED";
        int tsize    = 4;
        int label_px = (int)strlen(label) * 6 * tsize;
        tft.fillScreen(TFT_BLACK);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.setTextSize(tsize);
        tft.setCursor(max(0, (tft.width() - label_px) / 2), (tft.height() - 8 * tsize) / 2);
        tft.print(label);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

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
            uint32_t cur_gap = g_teensy.rx_seq_gaps();  // per-link frame loss (audit W3)
            g_uart_seq_gaps  = cur_gap;
            g_uart_crc_drops = cur_crc;
            g_uart_crc_rate  = (uint8_t)min((uint32_t)255, cur_crc - prev_crc_drops);
            g_uart_gap_rate  = (uint8_t)min((uint32_t)255, cur_gap - prev_seq_gaps);
            prev_crc_drops   = cur_crc;
            prev_seq_gaps    = cur_gap;
            last_health_ms   = now;
            if (g_uart_crc_rate || g_uart_gap_rate) g_display_dirty = true;
        }

        // Face mode: update every 10ms tick for smooth animation.
        // Engineering mode: update at ~30 Hz or on dirty flag.
        bool time_to_update = g_display_dirty ||
            (g_display_personality == PERS_FACE ? true : (now - last_disp_ms >= 33));
        if (time_to_update) {
            g_display_dirty = false;
            update_display();
            last_disp_ms = millis();
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ── WiFi diagnostics ─────────────────────────────────────────────────────────

static void send_wifi_diag() {
    WifiDiagPayload d;
    d.esp_uptime_ms   = millis();
    d.rssi_dbm        = (WiFi.status() == WL_CONNECTED) ? (int8_t)WiFi.RSSI() : 0;
    d.wifi_channel    = (uint8_t)WiFi.channel();
    d.wifi_status     = (uint8_t)WiFi.status();
    d.tx_power_raw    = (uint8_t)WiFi.getTxPower();
    d.free_heap       = ESP.getFreeHeap();
    d.min_free_heap   = ESP.getMinFreeHeap();
    d.loop_max_us     = g_loop_max_us;
    d.udp_send_max_us = g_udp_send_max_us;
    g_loop_max_us     = 0;
    g_udp_send_max_us = 0;
    d.wifi_reconnect_count = g_wifi_reconnect_count;
    d.udp_send_fail_count  = (uint16_t)min((uint32_t)0xFFFF, g_udp_stream.failCount());
    d.build_variant_flags =
#if WIFI_TELEM_MODE
        WIFI_DIAG_FLAG_UNICAST |
#endif
#if WIFI_TELEM_COMBINED
        WIFI_DIAG_FLAG_COMBINED |
#endif
#if WIFI_TX_POWER_MAX
        WIFI_DIAG_FLAG_TX_POWER_MAX |
#endif
#if !NEO_ENABLED
        WIFI_DIAG_FLAG_NEO_DISABLED |
#endif
#if !DISPLAY_ENABLED
        WIFI_DIAG_FLAG_DISPLAY_DISABLED |
#endif
#if WIFI_TRANSPORT_GATING
        WIFI_DIAG_FLAG_TRANSPORT_GATING |
#endif
        0;
    d.active_telem_transport = g_active_telem_transport;

    // V2 — Teensy-link status, so the GUI can tell "ESP32 alive, Teensy silent"
    // apart from "whole link down" even when WIFI_DIAG is the only thing arriving.
    uint32_t since_teensy = millis() - g_last_teensy_ms;
    // Telemetry (and so since_teensy) intentionally goes stale for the whole
    // duration of an SD-log GET (see teensy/src/main.cpp loop()) — don't let
    // the GUI's link-health panel read that expected quiet spell as down.
    d.teensy_link_up     = (log_xfer_is_downloading() ||
                             (g_teensy_ever_heard && since_teensy < TEENSY_LINK_TIMEOUT_MS)) ? 1 : 0;
    d.reserved2          = 0;
    d.ms_since_teensy    = (uint16_t)min((uint32_t)0xFFFF, since_teensy);
    d.uart_crc_drops     = (uint16_t)min((uint32_t)0xFFFF, g_teensy.rx_drops());
    d.uart_seq_gaps      = (uint16_t)min((uint32_t)0xFFFF, g_teensy.rx_seq_gaps());
    d.uplink_queue_drops = (uint16_t)min((uint32_t)0xFFFF,
                                         g_uplink_drops + g_log_uplink_drops);
    d.tcp_send_max_us    = g_tcp_send_max_us;
    g_tcp_send_max_us    = 0;

    // Enqueued like every other uplink frame — uplink_task (core 0) is the only
    // writer to Serial/UDP now; a direct send here would race with it.
    enqueue_uplink(COMM_TYPE_WIFI_DIAG, WIFI_DIAG_PAYLOAD_V2, &d, sizeof(d));
}

// ESP32->Teensy link heartbeat + ESP32 health, independent of the Teensy link's
// own state — this is what lets the Teensy (and via telemetry, the GUI) tell
// "ESP32 alive, Teensy silent" apart from "whole link down". Sent directly on
// g_teensy (Serial2, core 1 — same core as the parser, tiny frame, non-blocking
// at 4 Mbaud), not through the uplink queue: that queue is for Teensy->PC
// traffic, this is ESP32->Teensy. Telemetry-only; never causes a fault.
static void send_esp32_status() {
    Esp32StatusPayload s;
    s.uptime_ms            = millis();
    s.wifi_ok              = (WiFi.status() == WL_CONNECTED) ? 1 : 0;
    // Never wait here: this runs on core 1, which must return to g_teensy.update()
    // fast enough to drain the 4 Mbaud UART continuously. A WiFiClient::write()
    // on core 0 can retain g_tcp_mutex for seconds despite the writable precheck.
    // If the mutex is busy, the last lock-protected observation is sufficient
    // for this diagnostic heartbeat.
    bool tcp_connected = g_tcp_connected_cached;
    if (xSemaphoreTake(g_tcp_mutex, 0) == pdTRUE) {
        tcp_connected = g_comm_tcp && g_tcp_client.connected();
        g_tcp_connected_cached = tcp_connected;
        xSemaphoreGive(g_tcp_mutex);
    }
    s.tcp_client_connected = tcp_connected ? 1 : 0;
    s.rssi_dbm             = (WiFi.status() == WL_CONNECTED) ? (int8_t)WiFi.RSSI() : 0;
    s.reserved             = 0;
    s.uplink_drops         = (uint16_t)min((uint32_t)0xFFFF, g_uplink_drops);
    s.uart_rx_drops        = (uint16_t)min((uint32_t)0xFFFF, g_teensy.rx_drops());
    s.uart_seq_gaps        = (uint16_t)min((uint32_t)0xFFFF, g_teensy.rx_seq_gaps());
    g_teensy.send(COMM_TYPE_ESP32_STATUS, ESP32_STATUS_PAYLOAD_V1, &s, sizeof(s));
}

// ── Setup / loop ──────────────────────────────────────────────────────────────

// Human-readable reset cause, printed once at boot — cheap insurance so a
// future crash/reboot investigation starts from this line instead of
// re-deriving cause from raw serial-log archaeology (see Phase 7, UARTplat.md,
// where the pbuf_free/g_tcp_mutex bug was found this way).
static const char* reset_reason_name(esp_reset_reason_t r) {
    switch (r) {
        case ESP_RST_POWERON:   return "POWERON";
        case ESP_RST_EXT:       return "EXT_PIN";
        case ESP_RST_SW:        return "SW (esp_restart)";
        case ESP_RST_PANIC:     return "PANIC (exception/abort)";
        case ESP_RST_INT_WDT:   return "INT_WDT";
        case ESP_RST_TASK_WDT:  return "TASK_WDT";
        case ESP_RST_WDT:       return "OTHER_WDT";
        case ESP_RST_DEEPSLEEP: return "DEEPSLEEP_WAKE";
        case ESP_RST_BROWNOUT:  return "BROWNOUT";
        case ESP_RST_SDIO:      return "SDIO";
        default:                return "UNKNOWN";
    }
}

void setup() {
    Serial.begin(921600);
    esp_reset_reason_t rr = esp_reset_reason();
    Serial.printf("[ESP32] reset reason: %d (%s)\n", (int)rr, reset_reason_name(rr));
    Serial2.setRxBufferSize(4096);
    Serial2.begin(TEENSY_UART_BAUD, SERIAL_8N1, TEENSY_UART_RX, TEENSY_UART_TX);
    uart_set_rx_full_threshold(UART_NUM_2, 32);  // fire ISR every 32 bytes (80 µs at 4 Mbaud) leaving 96 bytes / 192 µs of headroom vs the default 120-byte threshold (8 bytes / 16 µs)
    // Fix 5: flush boot-noise before the parser starts
    delay(10);
    while (Serial2.available()) Serial2.read();

    g_teensy.onPacket(on_teensy_packet);
    g_usb.onPacket(forward_usb_to_teensy);

    g_uplink_q = xQueueCreate(UPLINK_QUEUE_LEN, sizeof(UplinkFrame));
    g_log_uplink_q = xQueueCreate(LOG_UPLINK_QUEUE_LEN, sizeof(UplinkFrame));
    g_log_ack_q = xQueueCreate(LOG_UPLINK_QUEUE_LEN, sizeof(LogRelayAck));
    g_tcp_mutex = xSemaphoreCreateMutex();
    g_usb_mutex = xSemaphoreCreateMutex();
    TaskHandle_t uplink_task_handle = nullptr;
    xTaskCreatePinnedToCore(uplink_task, "uplink", 4096, nullptr, 2, &uplink_task_handle, 0);
    xTaskCreatePinnedToCore(log_uplink_task, "log_uplink", 4096, nullptr, 2, nullptr, 0);
    // Root-caused a WiFiClient::write() hang here (Phase 7, UARTplat.md) via
    // this exact watchdog — keeping it as permanent defense-in-depth: if
    // uplink_task ever blocks for any other reason, a 5 s panic-reboot beats
    // the whole USB+WiFi uplink pipeline going silent forever.
    esp_task_wdt_init(5, true);
    esp_task_wdt_add(uplink_task_handle);

#if WIFI_ENABLED
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
#if WIFI_TX_POWER_MAX
    WiFi.setTxPower(WIFI_POWER_19_5dBm);
#endif
#endif

#if NEO_ENABLED
    xTaskCreatePinnedToCore(neo_task,     "neo",  4096, nullptr, 1, nullptr, 0);
#endif
#if LASERS_ENABLED
    xTaskCreatePinnedToCore(tof_task,     "tof",  4096, nullptr, 1, nullptr, 0);
#endif
#if DISPLAY_ENABLED
    xTaskCreatePinnedToCore(display_task, "disp", 6144, nullptr, 1, nullptr, 0);
#endif

    Serial.println("[ESP32] ready");

    // One-shot "ESP32 booted" line in the GUI's robot log — Teensy already
    // announces itself there via comm_log(); this is the ESP32-originated
    // equivalent so a reboot/reflash shows up there too, not just as a
    // version-mismatch or link-state blip elsewhere in the GUI.
    {
        char msg[48];
        int n = snprintf(msg, sizeof(msg), "[ESP32] Booted (reset: %s)", reset_reason_name(rr));
        if (n > (int)sizeof(msg) - 1) n = (int)sizeof(msg) - 1;
        uint8_t buf[1 + sizeof(msg)];
        buf[0] = LOG_LEVEL_INFO;
        memcpy(buf + 1, msg, n);
        enqueue_uplink(COMM_TYPE_LOG, LOG_PAYLOAD_V1, buf, 1 + n);
    }
}

void loop() {
    uint32_t t_loop_start = micros();

    g_teensy.update();

    // Only core 1 writes g_teensy/Serial2. The relay task reports completed
    // chunks through this queue instead of touching CommLink from core 0.
    LogRelayAck relay_ack;
    while (xQueueReceive(g_log_ack_q, &relay_ack, 0) == pdTRUE) {
        if (relay_ack.generation != g_log_generation) continue;
        uint8_t cmd[8];
        cmd[0] = CMD_ID_LOG;
        cmd[1] = LOG_SUB_CHUNK_ACK;
        memcpy(cmd + 2, &relay_ack.file_index, 2);
        memcpy(cmd + 4, &relay_ack.chunk_index, 4);
        g_teensy.send(COMM_TYPE_COMMAND, CMD_PAYLOAD_V1, cmd, sizeof(cmd));
    }
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
        g_telem_wm_l_state      = pkt.wm_l_state;
        g_telem_wm_r_state      = pkt.wm_r_state;
        g_telem_wm_mode         = pkt.wm_mode;
        g_telem_active_profile  = pkt.active_profile;
        g_telem_health_flags    = pkt.health_flags;
        g_telem_hip_l_cmd_rad   = pkt.hip_l_cmd_pos_rad;
        g_telem_hip_r_cmd_rad   = pkt.hip_r_cmd_pos_rad;
        g_telem_v_ref           = pkt.v_ref;
        g_telem_omega_cmd_rds   = pkt.omega_cmd_rds;
        g_telem_ff1_out         = pkt.ff1_out;
        g_telem_ff2_out         = pkt.ff2_out;
        g_telem_jump_state      = pkt.jump_state;

#if WIFI_ENABLED && WIFI_TELEM_COMBINED
        // Reuses the already-reassembled TelemetryPayload — no new reassembly logic.
        // Enqueued (not sent directly) — uplink_task (core 0) owns g_telem_udp now.
        if (g_wifi_inited
#if WIFI_TRANSPORT_GATING
            && g_active_telem_transport != WIFI_DIAG_TRANSPORT_USB_ONLY
#endif
        ) {
            enqueue_uplink(COMM_TYPE_TELEM_FULL_WIFI, TELEM_VERSION, &pkt, sizeof(pkt));
        }
#endif

        // (Gap detection moved to CommLink's per-link seq counter — see
        // g_teensy.rx_seq_gaps(), mirrored into g_uart_seq_gaps by display_task.)
    }

#if WIFI_ENABLED
    // Core 1 owns the Teensy UART parser. Never wait for a TCP sender here: skip
    // one TCP RX poll and return on the next loop iteration instead.
    if (xSemaphoreTake(g_tcp_mutex, 0) == pdTRUE) {
        bool tcp_connected = g_comm_tcp && g_tcp_client.connected();
        g_tcp_connected_cached = tcp_connected;
        if (tcp_connected) g_comm_tcp->update();
        xSemaphoreGive(g_tcp_mutex);
    }

    if (WiFi.status() == WL_CONNECTED && !g_wifi_inited) {
        g_tcp_server.begin();
        // Unicast needs a real target IP; fall back to broadcast if it wasn't
        // baked in at flash time (e.g. a raw `pio run -t upload` outside the
        // GUI's Flash & Monitor tab, which auto-injects it) so telemetry never
        // silently goes dark.
        const char* telem_target = "255.255.255.255";
#if WIFI_TELEM_MODE
        if (WIFI_UNICAST_IP[0] != '\0') telem_target = WIFI_UNICAST_IP;
#endif
        g_udp_stream.beginSend(telem_target, TELEM_UDP_PORT);
        g_udp_diag_stream.beginSend(telem_target, TELEM_UDP_PORT);
        g_wifi_inited = true;
        WiFi.localIP().toString().toCharArray(g_dbg_wifi_ip, sizeof(g_dbg_wifi_ip));
        g_dbg_wifi_ip_pending = true;
    }

    static uint32_t last_reconnect_ms = 0;
    if (WiFi.status() != WL_CONNECTED) {
        g_wifi_inited = false;
        g_tcp_connected_cached = false;
        if (millis() - last_reconnect_ms > 5000) {
            WiFi.begin(WIFI_SSID, WIFI_PASS);
            last_reconnect_ms = millis();
            g_wifi_reconnect_count++;
        }
    }
#endif

    // WiFi diagnostics: over USB always, over WiFi UDP when connected (see send_wifi_diag()).
    static uint32_t last_diag_ms = 0;
    if (millis() - last_diag_ms >= (1000 / WIFI_DIAG_HZ)) {
        last_diag_ms = millis();
        send_wifi_diag();
    }

    // ESP32->Teensy link heartbeat (see send_esp32_status()).
    static uint32_t last_esp32_status_ms = 0;
    if (millis() - last_esp32_status_ms >= (1000 / ESP32_STATUS_HZ)) {
        last_esp32_status_ms = millis();
        send_esp32_status();
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

#if WIFI_ENABLED
    if (g_wifi_inited && g_tcp_server.hasClient() &&
        xSemaphoreTake(g_tcp_mutex, 0) == pdTRUE) {
        // Only remove the pending client from the server after acquiring the
        // mutex. If the sender is busy, leave it pending for the next loop.
        WiFiClient c = g_tcp_server.available();
        if (c) {
            g_tcp_client = c;
            delete g_comm_tcp;
            g_comm_tcp = new CommLink(g_tcp_client, COMM_SRC_ESP32);
            g_comm_tcp->onPacket(forward_tcp_to_teensy);
            g_tcp_client.remoteIP().toString().toCharArray(g_dbg_tcp_client_ip, sizeof(g_dbg_tcp_client_ip));
            g_tcp_connected_cached = true;
            g_dbg_tcp_client_pending = true;
        }
        xSemaphoreGive(g_tcp_mutex);
    }
#endif

    uint16_t loop_dt = (uint16_t)min((uint32_t)0xFFFF, (uint32_t)(micros() - t_loop_start));
    if (loop_dt > g_loop_max_us) g_loop_max_us = loop_dt;
}
