// main.cpp — Phase 0+1: 500 Hz superloop + WiFi/OTA + BNO086 IMU.
//
// Verification:
//   - LED toggles at 250 Hz (500 Hz / 2) — measure on scope
//   - OTA reflash works over WiFi
//   - Serial prints tick count every second

#include <Arduino.h>
#include "FspTimer.h"
#include "config.h"
#include "robot_state.h"
#include "wifi_mgr.h"
#include "led_matrix.h"
#include "imu.h"

// ── 500 Hz hardware timer ───────────────────────────────────────────────────
static volatile bool tick_flag = false;
static FspTimer hw_timer;

void timer_isr(timer_callback_args_t __attribute__((unused)) *p_args) {
    tick_flag = true;
}

static bool timer_init_500hz() {
    // Get a free GPT timer channel
    uint8_t type;
    int8_t ch = FspTimer::get_available_timer(type);
    if (ch < 0) {
        Serial.println("[Timer] No free timer channel!");
        return false;
    }

    // Configure for 500 Hz (period = 2000 µs)
    hw_timer.begin(TIMER_MODE_PERIODIC, type, ch,
                   LOOP_RATE_HZ, 0.0f, timer_isr);
    hw_timer.setup_overflow_irq();
    hw_timer.open();
    hw_timer.start();

    Serial.print("[Timer] Channel ");
    Serial.print(ch);
    Serial.println(" running at 500 Hz");
    return true;
}

// ── Global state ────────────────────────────────────────────────────────────
static RobotState state = {};

// ── Setup ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);  // let serial settle
    Serial.println("=== Wheeled-Leg Robot — Phase 0 Skeleton ===");

    // LED for timing verification
    pinMode(PIN_STATUS_LED, OUTPUT);

    // WiFi + OTA (blocking, ~3s)
    wifi_init();

    // LED matrix status display
    led_matrix_init();

    // IMU (BNO086 over SPI — blocking, ~1-10s)
    if (!imu_init()) {
        Serial.println("FATAL: IMU init failed");
        while (1) { delay(1000); }
    }

    // 500 Hz hardware timer
    if (!timer_init_500hz()) {
        Serial.println("FATAL: timer init failed");
        while (1) { delay(1000); }
    }

    state.mode = Mode::IDLE;
    state.tick = 0;

    Serial.println("[Main] Entering superloop");
}

// ── Main loop ───────────────────────────────────────────────────────────────
void loop() {
    // Non-blocking OTA poll (~50 µs)
    wifi_ota_poll();

    // Spin until 500 Hz tick
    if (!tick_flag) return;
    tick_flag = false;

    uint32_t t0 = micros();

    // ── Toggle LED every tick (250 Hz square wave on scope) ──
    digitalWrite(PIN_STATUS_LED, state.tick & 1);

    // ── IMU ──
    imu_poll(&state);

    // ── Placeholder: future modules plug in here ──
    // can_rx_poll(&state);
    // if (state.mode >= Mode::BALANCE) {
    //     velocity_pi_update(&state);
    //     lqr_update(&state);
    //     yaw_pi_update(&state);
    //     hip_ctrl_update(&state);
    // }
    // can_tx_commands(&state);
    // safety_check(&state);

    // ── Measure loop time ──
    state.dt_us = micros() - t0;
    state.tick++;

    // ── LED matrix status display (4 Hz) ──
    if (state.tick % (LOOP_RATE_HZ / 4) == 0) {
        led_matrix_update(&state);
    }

    // ── 1 Hz serial heartbeat ──
    if (state.tick % LOOP_RATE_HZ == 0) {
        Serial.print("[Tick] ");
        Serial.print(state.tick);
        Serial.print("  dt=");
        Serial.print(state.dt_us);
        Serial.print(" us  pitch=");
        Serial.print(state.pitch * 57.2958f, 1);  // rad → deg
        Serial.print("° rate=");
        Serial.print(state.pitch_rate * 57.2958f, 1);
        Serial.print("°/s roll=");
        Serial.print(state.roll * 57.2958f, 1);
        Serial.println("°");
    }
}
