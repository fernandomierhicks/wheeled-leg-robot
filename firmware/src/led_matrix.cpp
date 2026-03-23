// led_matrix.cpp — 12×8 LED matrix status display for UNO R4 WiFi.
//
// The built-in matrix is 12 columns × 8 rows.
// We divide it into 4 horizontal bars (2 rows each):
//   Rows 0-1: WiFi      solid=connected, blink=connecting, off=disconnected
//   Rows 2-3: Host      solid=host connected
//   Rows 4-5: Heartbeat toggles at ~3 Hz (main loop alive indicator)
//   Rows 6-7: Fault     solid=fault active
//
// Update is called from the soft-RT section of the main loop at a
// decimated rate (e.g. every 250 ms) to avoid unnecessary SPI traffic.

#include "led_matrix.h"
#include "config.h"
#include "wifi_mgr.h"
#include <Arduino_LED_Matrix.h>

static ArduinoLEDMatrix matrix;

// Frame buffer: 8 rows × 12 columns, packed as uint8_t[8][12]
// but ArduinoLEDMatrix expects uint8_t frame[8][12] (row-major, 1=on).
static uint8_t frame[8][12];

// ── Helpers ─────────────────────────────────────────────────────────────────

// Fill two rows (bar) with on/off
static void set_bar(uint8_t start_row, bool on) {
    uint8_t val = on ? 1 : 0;
    for (uint8_t c = 0; c < 12; c++) {
        frame[start_row][c]     = val;
        frame[start_row + 1][c] = val;
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

void led_matrix_init() {
    matrix.begin();
    memset(frame, 0, sizeof(frame));
    matrix.renderBitmap(frame, 8, 12);
}

void led_matrix_update(const RobotState *state) {
    // WiFi bar (rows 0-1): solid if connected, blink 2Hz if connecting
    bool wifi_connected = wifi_is_connected();
    if (wifi_connected) {
        set_bar(0, true);
    } else {
        // Blink at ~2 Hz (toggle every 250 ms worth of ticks)
        bool blink = (state->tick / (LOOP_RATE_HZ / 4)) & 1;
        set_bar(0, blink);
    }

    // Host bar (rows 2-3): solid if host connected (placeholder — future telemetry)
    set_bar(2, state->host_connected);

    // Heartbeat bar (rows 4-5): toggles every call → 3 Hz blink at 6 Hz update rate
    static bool hb = false;
    hb = !hb;
    set_bar(4, hb);

    // Fault bar (rows 6-7): solid if in FAULT mode
    set_bar(6, state->mode == Mode::FAULT);

    matrix.renderBitmap(frame, 8, 12);
}
