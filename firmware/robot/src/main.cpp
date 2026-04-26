// main.cpp — 500 Hz hard-timed superloop with balance controllers.
//
// Signal flow per tick:
//   1. IMU poll  →  pitch, pitch_rate, roll, roll_rate
//   2. wheel_motors_poll  →  wm_L/wm_R feedback; auto-IDLE on CAN fault
//   3. VelocityPI  →  theta_ref
//   4. LQR  →  tau_sym
//   5. YawPI  →  tau_yaw
//   6. Wheel split:  tau_L = tau_sym - tau_yaw,  tau_R = tau_sym + tau_yaw
//   7. Suspension  →  tau_hip_L, tau_hip_R
//   8. wheel_motors_send + pet_watchdog  →  CAN TX to ODrive
//
// Timing: hard 500 Hz from hardware timer ISR.
// If any tick overruns 2000 µs, the FAULT LED bar flashes but we continue.
//
// IMU_ONLY build: define IMU_ONLY to strip CAN + controllers for bench testing.

#ifndef DEBUG_BUILD   // debug_main.cpp provides setup()/loop() in debug builds

#include <Arduino.h>
#include "FspTimer.h"
#include "config.h"
#include "robot_state.h"
#include "led_matrix.h"
#include "imu.h"
#include "telemetry.h"
#ifndef IMU_ONLY
#include "controllers.h"
#include "wifi_fast.h"
#include "commands.h"
#include "wheel_motors.h"
#endif

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
#ifndef IMU_ONLY
static LQRController         lqr;
static VelocityPI            vel_pi;
static YawPI                 yaw_pi;
static SuspensionController  suspension;
#endif

static uint32_t loop_overruns = 0;
static uint32_t s_tick_start_us = 0;  // micros() at start of current tick

// ── Setup ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(1000000);
    delay(500);
    Serial.println("=== Wheeled-Leg Robot ===");
#ifdef IMU_ONLY
    Serial.println("[Main] IMU_ONLY mode — CAN + controllers disabled");
#elif USE_WIFI
    Serial.println("[Main] Comms: WiFi UDP (slack-time)");
#else
    Serial.println("[Main] Comms: USB-UART Serial");
#endif

    pinMode(PIN_STATUS_LED, OUTPUT);

#if USE_WIFI && !defined(IMU_ONLY)
    wifi_fast_init();
#endif

    led_matrix_init();

    // IMU (BNO086 over SPI — blocking, ~1-10s)
    if (!imu_init()) {
        Serial.println("[IMU] WARNING: IMU not found — imu_ok=false, continuing without IMU");
        Serial.println("[IMU] Balance control disabled; direct torque via CMD_DRIVE ok");
        state.imu_ok = false;
    }

#ifndef IMU_ONLY
    if (!wheel_motors_init()) {
        Serial.println("FATAL: wheel CAN init failed");
        while (1) { delay(1000); }
    }
    commands_init();

    // Controllers
    lqr.init();
    vel_pi.init();
    yaw_pi.init();
    suspension.init();
    Serial.println("[Ctrl] Controllers initialised");
#endif

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
#if USE_WIFI && !defined(IMU_ONLY)
    if (!tick_flag) {
        wifi_try_send(s_tick_start_us);
        wifi_try_receive(state, state.tick, s_tick_start_us);
        wifi_fast_ota_poll();
        return;
    }
#else
    if (!tick_flag) {
#ifndef IMU_ONLY
        commands_receive(state);  // drain serial command buffer between ticks
#endif
        return;
    }
#endif
    tick_flag = false;

    s_tick_start_us = micros();
    const float dt = 1.0f / LOOP_RATE_HZ;

    // ── Toggle LED every tick (250 Hz square wave on scope) ──
    digitalWrite(PIN_STATUS_LED, state.tick & 1);

    // ── IMU ──
    imu_poll(&state);

#ifndef IMU_ONLY
    // ── CAN RX: pump encoder feedback + heartbeats; auto-IDLE on fault ──
    wheel_motors_poll();
    state.wheel_pos_L        = wm_L.pos_rad;
    state.wheel_pos_R        = wm_R.pos_rad;
    state.wheel_vel_L        = wm_L.vel_rad_s;
    state.wheel_vel_R        = wm_R.vel_rad_s;
    state.wheel_vel_avg      = (wm_L.vel_rad_s + wm_R.vel_rad_s) * 0.5f;
    state.wheel_ok           = wm_L.ok && wm_R.ok;
    state.odrive_axis_state  = wm_L.axis_state;
    state.odrive_axis_error  = wm_L.error;
    state.odrive_axis_state_R = wm_R.axis_state;
    state.odrive_axis_error_R = wm_R.error;

    // ── Controllers (only in BALANCE or DRIVE mode) ──
    if (state.mode == Mode::BALANCE || state.mode == Mode::DRIVE) {
        // Without IMU: open-loop — v_cmd is used as symmetric torque [N·m] directly.
        if (!state.imu_ok) {
            static constexpr float SAFE_TAU_MAX = 2.0f;
            float tau_sym = constrain(state.v_cmd,    -SAFE_TAU_MAX, SAFE_TAU_MAX);
            float tau_yaw = constrain(state.omega_cmd, -SAFE_TAU_MAX, SAFE_TAU_MAX);
            state.tau_wheel_L = constrain(tau_sym - tau_yaw, -SAFE_TAU_MAX, SAFE_TAU_MAX);
            state.tau_wheel_R = constrain(tau_sym + tau_yaw, -SAFE_TAU_MAX, SAFE_TAU_MAX);
        }
        // Safety: fall detection (only meaningful with IMU)
        else if (fabsf(state.pitch) > FALL_ANGLE_RAD) {
            state.mode = Mode::FAULT;
            vel_pi.reset();
            yaw_pi.reset();
            state.tau_sym     = 0.0f;
            state.tau_yaw     = 0.0f;
            state.tau_wheel_L = 0.0f;
            state.tau_wheel_R = 0.0f;
            state.tau_hip_L   = 0.0f;
            state.tau_hip_R   = 0.0f;
        } else {
            float v_measured = state.wheel_vel_avg * WHEEL_RADIUS;
            float v_ref_rad  = state.v_cmd / WHEEL_RADIUS;

            state.theta_ref = vel_pi.update(state.v_cmd, v_measured, dt);
            state.tau_sym   = lqr.update(state.pitch, state.pitch_rate,
                                         state.wheel_vel_avg,
                                         state.theta_ref, v_ref_rad);
            float yaw_rate  = 0.0f;
            state.tau_yaw   = yaw_pi.update(state.omega_cmd, yaw_rate, dt);
            state.tau_wheel_L = constrain(state.tau_sym - state.tau_yaw,
                                          -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX);
            state.tau_wheel_R = constrain(state.tau_sym + state.tau_yaw,
                                          -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX);
            suspension.update(state.roll, state.roll_rate,
                              state.hip_q_L, state.hip_dq_L,
                              state.hip_q_R, state.hip_dq_R,
                              state.tau_hip_L, state.tau_hip_R);
        }
    } else {
        state.tau_sym     = 0.0f;
        state.tau_yaw     = 0.0f;
        state.tau_wheel_L = 0.0f;
        state.tau_wheel_R = 0.0f;
        state.tau_hip_L   = 0.0f;
        state.tau_hip_R   = 0.0f;
    }

    // ── Wheel motor mode sync (on robot mode transitions) ──
    {
        static Mode s_prev_mode = Mode::IDLE;
        if (state.mode != s_prev_mode) {
            if (state.mode == Mode::BALANCE || state.mode == Mode::DRIVE) {
                wheel_motors_set_mode(WheelMode::TORQUE);
            } else {
                wheel_motors_set_mode(WheelMode::IDLE);
            }
            s_prev_mode = state.mode;
        }
    }

    // ── CAN TX: send wheel commands + keepalive ──
    if (wm_mode == WheelMode::VELOCITY) {
        // [turns/s] → rad/s
        wheel_motors_send(state.odrive_vel_L * TWO_PI, state.odrive_vel_R * TWO_PI);
    } else if (wm_mode == WheelMode::POSITION) {
        // [turns] → rad
        wheel_motors_send(state.odrive_pos_L * TWO_PI, state.odrive_pos_R * TWO_PI);
    } else if (wm_mode == WheelMode::TORQUE) {
        // direct torque [N·m] from dashboard
        wheel_motors_send(state.odrive_tau_L, state.odrive_tau_R);
    } else {
        // IDLE: balance controller outputs (or zeros)
        wheel_motors_send(state.tau_wheel_L, state.tau_wheel_R);
    }
    wheel_motors_pet_watchdog();
#endif // !IMU_ONLY

    state.tick++;

    // ── Debug sine (1 Hz, for telemetry pipeline verification) ──
    state.debug_sine = sinf(state.tick * (2.0f * 3.14159265f / LOOP_RATE_HZ));

    // ── Telemetry (50 Hz = every TELEMETRY_SEND_DIV ticks) ──
    if (state.tick % TELEMETRY_SEND_DIV == 0) {
#if USE_WIFI && !defined(IMU_ONLY)
        wifi_fast_fill_telemetry(state);
        state.wifi_send_us = wifi_last_send_us();
        state.wifi_recv_us = wifi_last_recv_us();
        state.wifi_skips   = wifi_send_skips();
#else
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
        Serial.print("  imu=");
        Serial.print(state.imu_ok ? "OK" : "NO");
        Serial.print("  wheel=");
        Serial.print(state.wheel_ok ? "OK" : "NO");
        Serial.print("  vel_L=");
        Serial.print(state.wheel_vel_L / TWO_PI, 2);
        Serial.print("t/s  mode=");
        Serial.print((uint8_t)state.mode);
        Serial.println();
        sum_dt = 0;
        max_dt = 0;
    }
}

#endif // !DEBUG_BUILD
