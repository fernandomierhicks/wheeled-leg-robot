// main.cpp — 500 Hz hard-timed superloop with balance controllers.
//
// Signal flow per tick:
//   1. IMU poll  →  pitch, pitch_rate, roll, roll_rate
//   2. VelocityPI  →  theta_ref
//   3. LQR  →  tau_sym
//   4. YawPI  →  tau_yaw
//   5. Wheel split:  tau_L = tau_sym - tau_yaw,  tau_R = tau_sym + tau_yaw
//   6. Suspension  →  tau_hip_L, tau_hip_R
//   7. (future) CAN output to motors
//
// Timing: hard 500 Hz from hardware timer ISR.
// If any tick overruns 2000 µs, the FAULT LED bar flashes but we continue.

#ifndef DEBUG_BUILD   // debug_main.cpp provides setup()/loop() in debug builds

#include <Arduino.h>
#include "FspTimer.h"
#include "config.h"
#include "robot_state.h"
#include "controllers.h"
#include "led_matrix.h"
#include "imu.h"
#include "wifi_fast.h"
#include "telemetry.h"
#include "commands.h"
#include "odesc_can.h"

// ── 500 Hz hardware timer ───────────────────────────────────────────────────
static volatile bool tick_flag = false;
static FspTimer hw_timer;

void timer_isr(timer_callback_args_t __attribute__((unused)) *p_args) {
    tick_flag = true;
}

static bool timer_init_500hz() {
    uint8_t type;
    int8_t ch = FspTimer::get_available_timer(type);
    if (ch < 0) {
        Serial.println("[Timer] No free timer channel!");
        return false;
    }
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

// ── Global state + controllers ──────────────────────────────────────────────
static RobotState state = {};
static LQRController         lqr;
static VelocityPI            vel_pi;
static YawPI                 yaw_pi;
static SuspensionController  suspension;

static uint32_t loop_overruns = 0;
static uint32_t s_tick_start_us = 0;  // micros() at start of current tick

// ── Setup ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(1000000);
    delay(500);
    Serial.println("=== Wheeled-Leg Robot ===");
#if USE_WIFI
    Serial.println("[Main] Comms: WiFi UDP (slack-time)");
#else
    Serial.println("[Main] Comms: USB-UART Serial");
#endif

    pinMode(PIN_STATUS_LED, OUTPUT);

    // WiFi + UDP sockets (blocking connect, ~3-10s)
#if USE_WIFI
    wifi_fast_init();
#endif

    led_matrix_init();

    // IMU (BNO086 over SPI — blocking, ~1-10s)
    if (!imu_init()) {
        Serial.println("FATAL: IMU init failed");
        while (1) { delay(1000); }
    }

    // CAN bus + ODESC wheels
    if (!odesc_can_init()) {
        Serial.println("FATAL: CAN init failed");
        while (1) { delay(1000); }
    }

    // Controllers
    lqr.init();
    vel_pi.init();
    yaw_pi.init();
    suspension.init();
    Serial.println("[Ctrl] Controllers initialised");

    // 500 Hz hardware timer (start last)
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
    // ── Idle spin: slack-time WiFi I/O (only when USE_WIFI=1) ──
    if (!tick_flag) {
#if USE_WIFI
        wifi_try_send(s_tick_start_us);
        wifi_try_receive(state, state.tick, s_tick_start_us);
        wifi_fast_ota_poll();
#endif
        return;
    }
    tick_flag = false;

    s_tick_start_us = micros();
    const float dt = 1.0f / LOOP_RATE_HZ;

    // ── Toggle LED every tick (250 Hz square wave on scope) ──
    digitalWrite(PIN_STATUS_LED, state.tick & 1);

    // ── IMU ──
    imu_poll(&state);

    // ── CAN RX: parse encoder feedback + heartbeats ──
    odesc_can_poll(state);

    // ── Controllers (only in BALANCE or DRIVE mode) ──
    if (state.mode == Mode::BALANCE || state.mode == Mode::DRIVE) {
        // Safety: check fall angle
        if (fabsf(state.pitch) > FALL_ANGLE_RAD) {
            state.mode = Mode::FAULT;
            vel_pi.reset();
            yaw_pi.reset();
            state.tau_sym = 0.0f;
            state.tau_yaw = 0.0f;
            state.tau_wheel_L = 0.0f;
            state.tau_wheel_R = 0.0f;
            state.tau_hip_L = 0.0f;
            state.tau_hip_R = 0.0f;
            odesc_can_disable();
        } else {
            // Convert wheel_vel from rad/s to m/s for velocity PI
            float v_measured = state.wheel_vel_avg * WHEEL_RADIUS;
            float v_ref_rad  = state.v_cmd / WHEEL_RADIUS;  // back to rad/s for LQR

            // 1. Velocity PI → lean angle reference
            state.theta_ref = vel_pi.update(state.v_cmd, v_measured, dt);

            // 2. LQR balance → symmetric wheel torque
            state.tau_sym = lqr.update(state.pitch, state.pitch_rate,
                                       state.wheel_vel_avg,
                                       state.theta_ref, v_ref_rad);

            // 3. Yaw PI → differential torque
            // NOTE: yaw_rate not yet available from IMU (BNO086 Game Rotation
            // Vector doesn't provide yaw).  Using 0 until CAN gyro or
            // wheel-differential yaw is implemented.
            float yaw_rate = 0.0f;
            state.tau_yaw = yaw_pi.update(state.omega_cmd, yaw_rate, dt);

            // 4. Wheel torque split
            state.tau_wheel_L = constrain(state.tau_sym - state.tau_yaw,
                                          -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX);
            state.tau_wheel_R = constrain(state.tau_sym + state.tau_yaw,
                                          -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX);

            // 5. Suspension (impedance + roll leveling)
            suspension.update(state.roll, state.roll_rate,
                              state.hip_q_L, state.hip_dq_L,
                              state.hip_q_R, state.hip_dq_R,
                              state.tau_hip_L, state.tau_hip_R);
        }
    } else {
        // IDLE / FAULT / etc — zero all outputs
        state.tau_sym = 0.0f;
        state.tau_yaw = 0.0f;
        state.tau_wheel_L = 0.0f;
        state.tau_wheel_R = 0.0f;
        state.tau_hip_L = 0.0f;
        state.tau_hip_R = 0.0f;
    }

    // ── CAN TX: motor commands ──
    odesc_can_send_torque(state.tau_wheel_L, state.tau_wheel_R);
    // TODO: can_send_hip(state.tau_hip_L, state.tau_hip_R);

    state.tick++;

    // ── Telemetry (50 Hz = every TELEMETRY_SEND_DIV ticks) ──
    if (state.tick % TELEMETRY_SEND_DIV == 0) {
#if USE_WIFI
        // Fill back-buffer for slack-time UDP send (no WiFi I/O here)
        wifi_fast_fill_telemetry(state);
        // Update profiling fields for heartbeat
        state.wifi_send_us = wifi_last_send_us();
        state.wifi_recv_us = wifi_last_recv_us();
        state.wifi_skips   = wifi_send_skips();
#else
        // USB-UART: send immediately via Serial (framed binary)
        telemetry_send(state);
#endif
    }

    // ── LED matrix status display (~6 Hz) ──
    static uint32_t led_next_ms = 0;
    uint32_t now_ms = millis();
    if (now_ms >= led_next_ms) {
        led_next_ms = now_ms + 167;
        led_matrix_update(&state);
    }

    // ── Hard timing check ──
    state.dt_us = micros() - s_tick_start_us;
    if (state.dt_us > LOOP_PERIOD_US) {
        loop_overruns++;
        // Flash fault bar for ~6 LED updates (~1 second)
        if (state.overrun_flash < 12) {
            state.overrun_flash = 12;
        }
    }

    // ── 1 Hz serial heartbeat ──
    static uint32_t sum_dt = 0;
    static uint32_t max_dt = 0;
    sum_dt += state.dt_us;
    if (state.dt_us > max_dt) max_dt = state.dt_us;

    if (state.tick % LOOP_RATE_HZ == 0) {
        Serial.print("[HB] tick=");
        Serial.print(state.tick);
        Serial.print("  dt_avg=");
        Serial.print(sum_dt / LOOP_RATE_HZ);
        Serial.print("  dt_max=");
        Serial.print(max_dt);
        Serial.print("  overruns=");
        Serial.print(loop_overruns);
        Serial.print("  pitch=");
        Serial.print(state.pitch * 57.2958f, 1);
        Serial.print("  mode=");
        Serial.print((uint8_t)state.mode);
        Serial.print("  tau_sym=");
        Serial.print(state.tau_sym, 3);
#if USE_WIFI
        Serial.print("  wifi_tx=");
        Serial.print(state.wifi_send_us);
        Serial.print("us  wifi_skip=");
        Serial.print(state.wifi_skips);
#endif
        Serial.println();
        sum_dt = 0;
        max_dt = 0;
    }
}

#endif // !DEBUG_BUILD
