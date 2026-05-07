// led_matrix.cpp — 12×8 LED matrix icon display for UNO R4 WiFi.
//
// Shows one priority-selected icon on the full 12×8 grid, with small
// WiFi and host overlays in the corners.  Updated at ~6 Hz from main loop.

#include "led_matrix.h"
#include "config.h"
#include "wifi_mgr.h"
#include <Arduino_LED_Matrix.h>

static ArduinoLEDMatrix matrix;
static uint8_t frame[8][12];

// ── Icon bitmaps (8 rows × 12 cols, row-major, 1 = on) ─────────────────────
// Designed for the UNO R4 WiFi's 12-wide × 8-tall LED matrix.

// Smiley face :)  — IDLE, all good
static const uint8_t ICON_SMILEY[8][12] = {
    {0,0,0,1,1,1,1,1,1,0,0,0},
    {0,0,1,0,0,0,0,0,0,1,0,0},
    {0,1,0,0,1,0,0,1,0,0,1,0},
    {0,1,0,0,0,0,0,0,0,0,1,0},
    {0,1,0,1,0,0,0,0,1,0,1,0},
    {0,1,0,0,1,1,1,1,0,0,1,0},
    {0,0,1,0,0,0,0,0,0,1,0,0},
    {0,0,0,1,1,1,1,1,1,0,0,0},
};

// X cross — FAULT
static const uint8_t ICON_CROSS[8][12] = {
    {0,0,1,0,0,0,0,0,0,1,0,0},
    {0,0,0,1,0,0,0,0,1,0,0,0},
    {0,0,0,0,1,0,0,1,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,1,0,0,1,0,0,0,0},
    {0,0,0,1,0,0,0,0,1,0,0,0},
    {0,0,1,0,0,0,0,0,0,1,0,0},
};

// ! exclamation — loop overrun flash
static const uint8_t ICON_BANG[8][12] = {
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
};

// ? question mark — IMU lost
static const uint8_t ICON_QUESTION[8][12] = {
    {0,0,0,0,1,1,1,1,0,0,0,0},
    {0,0,0,1,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,0,0,0},
};

// Right arrow → — DRIVE mode
static const uint8_t ICON_ARROW[8][12] = {
    {0,0,0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,0},
    {0,1,1,1,1,1,1,1,1,1,1,0},
    {0,1,1,1,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,1,0,0,0,0},
};

// Up arrow ↑ — JUMP mode
static const uint8_t ICON_UP_ARROW[8][12] = {
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,1,1,1,1,0,0,0,0},
    {0,0,0,1,0,1,1,0,1,0,0,0},
    {0,0,1,0,0,1,1,0,0,1,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
    {0,0,0,0,0,1,1,0,0,0,0,0},
};

// Heartbeat animation — BALANCE mode (2 frames, alternating)
// Frame 0: small pulse in center
static const uint8_t ICON_HEART_0[8][12] = {
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,1,1,1,0,0,0,0},
    {0,0,0,0,1,1,1,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
};

// Frame 1: expanded pulse
static const uint8_t ICON_HEART_1[8][12] = {
    {0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,1,1,1,1,1,0,0,0},
    {0,0,1,1,1,1,1,1,1,1,0,0},
    {0,1,1,1,1,1,1,1,1,1,1,0},
    {0,1,1,1,1,1,1,1,1,1,1,0},
    {0,0,1,1,1,1,1,1,1,1,0,0},
    {0,0,0,1,1,1,1,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0},
};

// WiFi wave arcs — 3×3 overlay for bottom-right corner
static const uint8_t WIFI_OVERLAY[3][3] = {
    {1,0,1},
    {0,1,0},
    {0,1,0},
};

// ── Helpers ─────────────────────────────────────────────────────────────────

static void load_icon(const uint8_t icon[8][12]) {
    memcpy(frame, icon, sizeof(frame));
}

// Stamp WiFi overlay into bottom-right 3×3
static void overlay_wifi(bool on) {
    if (!on) return;
    for (uint8_t r = 0; r < 3; r++)
        for (uint8_t c = 0; c < 3; c++)
            if (WIFI_OVERLAY[r][c])
                frame[5 + r][9 + c] = 1;
}

// Host dot: bottom-left single pixel
static void overlay_host(bool on) {
    if (on) frame[7][0] = 1;
}

// ── Public API ──────────────────────────────────────────────────────────────

void led_matrix_init() {
    matrix.begin();
    memset(frame, 0, sizeof(frame));
    matrix.renderBitmap(frame, 8, 12);
}

void led_matrix_update(const RobotState *state) {
    static uint8_t anim_phase = 0;
    anim_phase++;

    // ── Priority-based icon selection (highest priority first) ──

    if (state->mode == Mode::FAULT) {
        // Blinking X: on 75% of the time for visibility
        if ((anim_phase % 4) < 3)
            load_icon(ICON_CROSS);
        else
            memset(frame, 0, sizeof(frame));
    }
    else if (state->overrun_flash > 0) {
        load_icon(ICON_BANG);
        const_cast<RobotState*>(state)->overrun_flash--;
    }
    else if (!state->imu_ok) {
        // Blinking ? so it catches your eye
        if (anim_phase & 1)
            load_icon(ICON_QUESTION);
        else
            memset(frame, 0, sizeof(frame));
    }
    else if (state->mode == Mode::BALANCE) {
        // Pulsing heartbeat — alternates small/large every ~0.3 s
        if ((anim_phase / 2) & 1)
            load_icon(ICON_HEART_1);
        else
            load_icon(ICON_HEART_0);
    }
    else if (state->mode == Mode::DRIVE) {
        load_icon(ICON_ARROW);
    }
    else if (state->mode == Mode::JUMP) {
        load_icon(ICON_UP_ARROW);
    }
    else {
        // IDLE or STAND_UP — smiley
        load_icon(ICON_SMILEY);
    }

    // ── Corner overlays ──

    // WiFi: solid arcs if connected, blinking if connecting, off if disconnected
    bool wifi_conn = wifi_is_connected();
    if (wifi_conn) {
        overlay_wifi(true);
    } else {
        // Blink at ~2 Hz (toggle every other update at 6 Hz)
        overlay_wifi((anim_phase / 2) & 1);
    }

    overlay_host(state->host_connected);

    matrix.renderBitmap(frame, 8, 12);
}
