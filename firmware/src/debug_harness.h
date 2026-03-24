#pragma once
// debug_harness.h — Lightweight profiling, watchdog, and serial menu for debug builds.
//
// Usage:
//   PROFILE_BEGIN(imu);
//   imu_poll(&state);
//   PROFILE_END(imu);
//   ...
//   profiler_print_and_reset();  // call every N ticks

#include <Arduino.h>
#include "r_wdt.h"          // Renesas FSP WDT driver

// ── Compile guard: only available in debug builds ────────────────────────────
#ifndef DEBUG_BUILD
#error "debug_harness.h included without -DDEBUG_BUILD"
#endif

// ── Profiler ─────────────────────────────────────────────────────────────────
// Up to 16 named slots.  Each tracks min/max/sum/count in microseconds.

#define PROFILER_MAX_SLOTS 16

struct ProfileSlot {
    const char *name;
    uint32_t min_us;
    uint32_t max_us;
    uint32_t sum_us;
    uint32_t count;
    uint32_t last_us;       // most recent measurement
};

static ProfileSlot _prof_slots[PROFILER_MAX_SLOTS];
static uint8_t     _prof_slot_count = 0;
static uint32_t    _prof_tmp_start  = 0;     // scratch for PROFILE_BEGIN

// Register a named slot (call once at startup).  Returns slot index.
inline uint8_t profiler_register(const char *name) {
    uint8_t idx = _prof_slot_count++;
    if (idx >= PROFILER_MAX_SLOTS) {
        Serial.println("[Prof] TOO MANY SLOTS");
        while (1) delay(1000);
    }
    _prof_slots[idx].name   = name;
    _prof_slots[idx].min_us = UINT32_MAX;
    _prof_slots[idx].max_us = 0;
    _prof_slots[idx].sum_us = 0;
    _prof_slots[idx].count  = 0;
    _prof_slots[idx].last_us = 0;
    return idx;
}

inline void profiler_record(uint8_t idx, uint32_t elapsed_us) {
    ProfileSlot &s = _prof_slots[idx];
    s.last_us = elapsed_us;
    s.sum_us += elapsed_us;
    s.count++;
    if (elapsed_us < s.min_us) s.min_us = elapsed_us;
    if (elapsed_us > s.max_us) s.max_us = elapsed_us;
}

// Print all slots, then reset accumulators.
inline void profiler_print_and_reset() {
    Serial.println("──── Profile Report ────");
    for (uint8_t i = 0; i < _prof_slot_count; i++) {
        ProfileSlot &s = _prof_slots[i];
        if (s.count == 0) continue;
        Serial.print("  ");
        Serial.print(s.name);
        Serial.print(": avg=");
        Serial.print(s.sum_us / s.count);
        Serial.print(" min=");
        Serial.print(s.min_us);
        Serial.print(" max=");
        Serial.print(s.max_us);
        Serial.print(" cnt=");
        Serial.print(s.count);
        Serial.println(" us");
        // reset
        s.min_us = UINT32_MAX;
        s.max_us = 0;
        s.sum_us = 0;
        s.count  = 0;
    }
    Serial.println("────────────────────────");
}

// Convenience macros — use matched pairs.
// Declares a local start variable unique to each call site.
#define PROFILE_BEGIN(slot_idx) \
    uint32_t _prof_t0_##slot_idx = micros()

#define PROFILE_END(slot_idx, idx) \
    profiler_record((idx), micros() - _prof_t0_##slot_idx)

// ── Watchdog Timer ───────────────────────────────────────────────────────────
// Renesas RA4M1 WDT: once started it cannot be stopped until reset.
// We use the FSP register-level API (no HAL instance needed on UNO R4).
//
// The WDT clock is PCLKB/128 = 48 MHz / 128 = 375 kHz.
// With a 16384-cycle window → timeout ≈ 43.7 ms (well above 2 ms loop).
//
// NOTE: The UNO R4 WiFi bootloader does NOT enable the WDT by default,
// so we can start it and feed it ourselves.

static volatile bool _wdt_enabled = false;

inline void watchdog_init() {
    // The Renesas RA4M1 WDT is register-start mode by default.
    // Write the start register to begin counting.
    // Timeout: PCLKB/128, 16384 cycles ≈ 43 ms.
    // If the main loop hangs, the WDT will hard-reset the MCU.
    R_WDT->WDTCR = (0x3 << 0)    // Clock div = PCLKB/128
                  | (0x3 << 4)    // Timeout = 16384 cycles
                  | (0x3 << 8)    // Window end = 0%
                  | (0x3 << 12);  // Window start = 100%
    R_WDT->WDTSR = 0;            // Clear status
    R_WDT->WDTRCR = 0x80;        // Reset on underflow (not NMI)
    // Start the WDT by refreshing it
    R_WDT->WDTRR = 0x00;
    R_WDT->WDTRR = 0xFF;
    _wdt_enabled = true;
    Serial.println("[WDT] Watchdog started (~43 ms timeout)");
}

inline void watchdog_feed() {
    if (!_wdt_enabled) return;
    R_WDT->WDTRR = 0x00;
    R_WDT->WDTRR = 0xFF;
}

// ── Serial menu helper ──────────────────────────────────────────────────────
// Waits for a single character with timeout.  Returns '\0' on timeout.
inline char serial_wait_char(uint32_t timeout_ms) {
    uint32_t start = millis();
    while (millis() - start < timeout_ms) {
        if (Serial.available()) {
            return (char)Serial.read();
        }
    }
    return '\0';
}

// Drain any buffered serial input
inline void serial_flush_input() {
    while (Serial.available()) Serial.read();
}
