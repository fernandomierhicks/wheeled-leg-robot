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
#include "telemetry.h"
#include "commands.h"

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
    Serial.begin(1000000);
    delay(500);  // let serial settle
    Serial.println("=== Wheeled-Leg Robot — Phase 0 Skeleton ===");

    // LED for timing verification
    pinMode(PIN_STATUS_LED, OUTPUT);

    // WiFi + OTA (blocking, ~3s)
    wifi_init();

    // LED matrix status display
    led_matrix_init();

    // Telemetry + commands (UDP over WiFi)
    telemetry_init();
    commands_init();

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
static uint32_t dt_ota_max, dt_imu_max, dt_telem_max, dt_cmd_max, dt_led_max, dt_total_max;

void loop() {
    // Spin until 500 Hz tick
    if (!tick_flag) return;
    tick_flag = false;

    uint32_t t0 = micros();

    // ── OTA poll (1 Hz — ArduinoOTA.poll() takes ~150 ms on R4 WiFi) ──
    uint32_t dt_ota = 0;
    if (state.tick % LOOP_RATE_HZ == 0) {
        uint32_t t_ota0 = micros();
        wifi_ota_poll();
        dt_ota = micros() - t_ota0;
    }

    // ── Toggle LED every tick (250 Hz square wave on scope) ──
    digitalWrite(PIN_STATUS_LED, state.tick & 1);

    // ── IMU ──
    uint32_t t_imu0 = micros();
    imu_poll(&state);
    uint32_t dt_imu = micros() - t_imu0;

    // ── Debug: noisy sine for telemetry rate check ──
    float t_s = state.tick * (1.0f / LOOP_RATE_HZ);
    state.debug_sine = sinf(2.0f * 3.14159265f * 1.0f * t_s)   // 1 Hz sine
                     + 0.3f * sinf(2.0f * 3.14159265f * 7.0f * t_s)  // 7 Hz harmonic
                     + 0.1f * (((float)random(-1000, 1000)) / 1000.0f);  // noise

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

    state.tick++;

    // ── Telemetry (50 Hz) ──
    uint32_t dt_telem = 0;
    if (state.tick % TELEMETRY_DIV == 0) {
        uint32_t t_tel0 = micros();
        telemetry_send(state);
        dt_telem = micros() - t_tel0;
    }

    // ── Commands (10 Hz) ──
    uint32_t dt_cmd = 0;
    if (state.tick % COMMAND_DIV == 0) {
        uint32_t t_cmd0 = micros();
        commands_receive(state);
        dt_cmd = micros() - t_cmd0;
    }

    // ── LED matrix status display (~6 Hz, millis-based to avoid tick-rate dependency) ──
    uint32_t dt_led = 0;
    static uint32_t led_next_ms = 0;
    uint32_t now_ms = millis();
    if (now_ms >= led_next_ms) {
        led_next_ms = now_ms + 167;   // 167 ms → ~6 Hz update, heartbeat toggles → 3 Hz blink
        uint32_t t_led0 = micros();
        led_matrix_update(&state);
        dt_led = micros() - t_led0;
    }

    // ── Measure total loop time (including WiFi ops) ──
    state.dt_us = micros() - t0;

    // Track worst-case per section
    if (dt_ota   > dt_ota_max)   dt_ota_max   = dt_ota;
    if (dt_imu   > dt_imu_max)   dt_imu_max   = dt_imu;
    if (dt_telem > dt_telem_max) dt_telem_max = dt_telem;
    if (dt_cmd   > dt_cmd_max)   dt_cmd_max   = dt_cmd;
    if (dt_led   > dt_led_max)   dt_led_max   = dt_led;
    if (state.dt_us > dt_total_max) dt_total_max = state.dt_us;

    // ── Loop-rate watchdog: fault if tick rate deviates from 500 Hz ──
    // Compare wall-clock elapsed vs tick-implied elapsed every 500 ticks.
    // Skip the first 1000 ticks (2 s) to let IMU/WiFi settle after boot.
    static uint32_t wd_tick0   = 0;
    static uint32_t wd_ms0     = 0;
    static bool     wd_primed  = false;
    if (!wd_primed && state.tick >= 1000) {
        wd_tick0  = state.tick;
        wd_ms0    = millis();
        wd_primed = true;
    }
    if (wd_primed && (state.tick - wd_tick0) >= LOOP_RATE_HZ) {
        uint32_t elapsed_ms   = millis() - wd_ms0;
        uint32_t expected_ms  = ((state.tick - wd_tick0) * 1000UL) / LOOP_RATE_HZ;
        // Fault if wall-clock drifts more than 20% from expected
        if (elapsed_ms > expected_ms * 6 / 5 || elapsed_ms < expected_ms * 4 / 5) {
            state.mode = Mode::FAULT;
            Serial.print("[FAULT] Loop rate deviation: expected ");
            Serial.print(expected_ms);
            Serial.print(" ms, actual ");
            Serial.print(elapsed_ms);
            Serial.println(" ms");
        }
        // Reset window
        wd_tick0 = state.tick;
        wd_ms0   = millis();
    }

    // ── 1 Hz serial heartbeat with timing breakdown (all µs) ──
    if (state.tick % LOOP_RATE_HZ == 0) {
        Serial.print("[Tick] ");
        Serial.print(state.tick);
        Serial.print("  max_us: total=");
        Serial.print(dt_total_max);
        Serial.print(" ota=");
        Serial.print(dt_ota_max);
        Serial.print(" imu=");
        Serial.print(dt_imu_max);
        Serial.print(" telem=");
        Serial.print(dt_telem_max);
        Serial.print(" cmd=");
        Serial.print(dt_cmd_max);
        Serial.print(" led=");
        Serial.print(dt_led_max);
        Serial.print("  pitch=");
        Serial.print(state.pitch * 57.2958f, 1);
        Serial.println("°");
        // Reset max counters
        dt_ota_max = dt_imu_max = dt_telem_max = dt_cmd_max = dt_led_max = dt_total_max = 0;
    }
}
