// debug_main.cpp — Debug harness for wheeled-leg robot firmware.
//
// Same module flow as main.cpp, but adds:
//   - Interactive boot menu (prove serial works before anything else)
//   - Per-module enable/disable at boot
//   - Per-function profiling (min/max/avg µs)
//   - Optional hardware watchdog (~43 ms)
//   - Continuous serial heartbeat (proves board is alive)
//   - Runtime serial commands to toggle modules / dump controller state
//
// Build with:  pio run -e debug
// Flash with:  pio run -e debug -t upload

#ifdef DEBUG_BUILD   // only compiled when -DDEBUG_BUILD is set

#include <Arduino.h>
#include "FspTimer.h"
#include "config.h"
#include "robot_state.h"
#include "controllers.h"
#include "imu.h"
#include "led_matrix.h"
// Future modules — uncomment as they are implemented:
// #include "wifi_mgr.h"
// #include "telemetry.h"
// #include "commands.h"
// #include "can_bus.h"

#include "debug_harness.h"

// ── Module enable flags ──────────────────────────────────────────────────────
static bool en_imu        = true;
static bool en_led_matrix = true;
static bool en_controllers = true;
static bool en_watchdog   = false;
static bool en_wifi       = false;
static bool en_telemetry  = false;
static bool en_commands   = false;
// Future:
// static bool en_can     = false;

// ── Profiler slot indices ────────────────────────────────────────────────────
static uint8_t p_total, p_imu, p_led, p_gpio, p_serial_cmd;
static uint8_t p_vel_pi, p_lqr, p_yaw_pi, p_suspension;

// ── 500 Hz hardware timer ────────────────────────────────────────────────────
static volatile bool tick_flag = false;
static FspTimer hw_timer;

void debug_timer_isr(timer_callback_args_t __attribute__((unused)) *p_args) {
    tick_flag = true;
}

static bool timer_init_500hz() {
    uint8_t type;
    int8_t ch = FspTimer::get_available_timer(type);
    if (ch < 0) {
        Serial.println("[Timer] No free channel!");
        return false;
    }
    hw_timer.begin(TIMER_MODE_PERIODIC, type, ch,
                   LOOP_RATE_HZ, 0.0f, debug_timer_isr);
    hw_timer.setup_overflow_irq();
    hw_timer.open();
    hw_timer.start();
    Serial.print("[Timer] Ch ");
    Serial.print(ch);
    Serial.println(" @ 500 Hz");
    return true;
}

// ── Global state + controllers ──────────────────────────────────────────────
static RobotState state = {};
static LQRController         lqr;
static VelocityPI            vel_pi;
static YawPI                 yaw_pi;
static SuspensionController  suspension;

// ── Heartbeat tracking ───────────────────────────────────────────────────────
static uint32_t heartbeat_count = 0;
static uint32_t loop_overruns   = 0;

// ── Boot menu ────────────────────────────────────────────────────────────────
static void print_boot_menu() {
    Serial.println();
    Serial.println("╔══════════════════════════════════════════╗");
    Serial.println("║   WHEELED-LEG ROBOT — DEBUG HARNESS     ║");
    Serial.println("╠══════════════════════════════════════════╣");
    Serial.println("║  Module toggles (press key to flip):     ║");
    Serial.print  ("║   [I] IMU (BNO086 SPI)       : "); Serial.println(en_imu         ? "ON  ║" : "OFF ║");
    Serial.print  ("║   [L] LED matrix             : "); Serial.println(en_led_matrix  ? "ON  ║" : "OFF ║");
    Serial.print  ("║   [K] Controllers (LQR+PI)   : "); Serial.println(en_controllers ? "ON  ║" : "OFF ║");
    Serial.print  ("║   [W] Watchdog timer (~43ms) : "); Serial.println(en_watchdog    ? "ON  ║" : "OFF ║");
    Serial.print  ("║   [F] WiFi + OTA             : "); Serial.println(en_wifi        ? "ON  ║" : "OFF ║");
    Serial.print  ("║   [T] Telemetry (UDP out)    : "); Serial.println(en_telemetry   ? "ON  ║" : "OFF ║");
    Serial.print  ("║   [C] Commands (UDP in)      : "); Serial.println(en_commands    ? "ON  ║" : "OFF ║");
    Serial.println("╠══════════════════════════════════════════╣");
    Serial.println("║   [G] GO — start superloop               ║");
    Serial.println("║   [D] Defaults — enable all safe modules ║");
    Serial.println("╚══════════════════════════════════════════╝");
    Serial.println("Waiting for input (auto-GO in 10s)...");
}

static void run_boot_menu() {
    print_boot_menu();

    uint32_t deadline = millis() + 10000;
    while (millis() < deadline) {
        char c = serial_wait_char(200);
        if (c == '\0') continue;

        switch (c | 0x20) {  // tolower
            case 'i': en_imu         = !en_imu;         break;
            case 'l': en_led_matrix  = !en_led_matrix;   break;
            case 'k': en_controllers = !en_controllers;  break;
            case 'w': en_watchdog    = !en_watchdog;     break;
            case 'f': en_wifi        = !en_wifi;         break;
            case 't': en_telemetry   = !en_telemetry;    break;
            case 'c': en_commands    = !en_commands;      break;
            case 'd':  // defaults: everything safe
                en_imu = en_led_matrix = en_controllers = true;
                en_watchdog = en_wifi = en_telemetry = en_commands = false;
                break;
            case 'g': goto done;
            default: continue;
        }
        print_boot_menu();
        deadline = millis() + 10000;
    }
done:
    Serial.println("\n>>> Starting...");
}

// ── Runtime serial commands ──────────────────────────────────────────────────
//   'p' — print profiler report
//   's' — state snapshot (IMU + controller outputs)
//   'c' — controller detail (gains, integrators)
//   'b' — toggle BALANCE mode (for bench testing without CAN)
//   'h' — help
static void process_serial_commands() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        switch (c | 0x20) {
            case 'p':
                profiler_print_and_reset();
                break;
            case 's':
                Serial.println("── State Snapshot ──");
                Serial.print("  tick=");       Serial.println(state.tick);
                Serial.print("  dt_us=");      Serial.println(state.dt_us);
                Serial.print("  mode=");       Serial.println((uint8_t)state.mode);
                Serial.print("  pitch=");      Serial.print(state.pitch * 57.2958f, 2); Serial.println(" deg");
                Serial.print("  pitch_rate="); Serial.print(state.pitch_rate * 57.2958f, 2); Serial.println(" deg/s");
                Serial.print("  roll=");       Serial.print(state.roll * 57.2958f, 2); Serial.println(" deg");
                Serial.print("  imu_ok=");     Serial.println(state.imu_ok ? "YES" : "NO");
                Serial.print("  wheel_vel=");  Serial.print(state.wheel_vel_avg, 3); Serial.println(" rad/s");
                Serial.print("  theta_ref=");  Serial.print(state.theta_ref * 57.2958f, 2); Serial.println(" deg");
                Serial.print("  tau_sym=");    Serial.print(state.tau_sym, 3); Serial.println(" Nm");
                Serial.print("  tau_yaw=");    Serial.print(state.tau_yaw, 3); Serial.println(" Nm");
                Serial.print("  tau_wh_L=");   Serial.print(state.tau_wheel_L, 3); Serial.println(" Nm");
                Serial.print("  tau_wh_R=");   Serial.print(state.tau_wheel_R, 3); Serial.println(" Nm");
                Serial.print("  tau_hip_L=");  Serial.print(state.tau_hip_L, 3); Serial.println(" Nm");
                Serial.print("  tau_hip_R=");  Serial.print(state.tau_hip_R, 3); Serial.println(" Nm");
                Serial.print("  overruns=");   Serial.println(loop_overruns);
                break;
            case 'c':
                Serial.println("── Controller Detail ──");
                Serial.print("  LQR K=[");
                Serial.print(lqr.K[0], 4); Serial.print(", ");
                Serial.print(lqr.K[1], 4); Serial.print(", ");
                Serial.print(lqr.K[2], 4); Serial.println("]");
                Serial.print("  VelPI: Kp="); Serial.print(vel_pi.Kp, 4);
                Serial.print(" Ki="); Serial.print(vel_pi.Ki, 4);
                Serial.print(" Kff="); Serial.print(vel_pi.Kff, 4);
                Serial.print(" int="); Serial.print(vel_pi.integral, 4);
                Serial.print(" theta_ref="); Serial.print(vel_pi.prev_theta_ref * 57.2958f, 2);
                Serial.println(" deg");
                Serial.print("  YawPI: Kp="); Serial.print(yaw_pi.Kp, 4);
                Serial.print(" Ki="); Serial.print(yaw_pi.Ki, 4);
                Serial.print(" int="); Serial.println(yaw_pi.integral, 4);
                Serial.print("  Susp: Ks="); Serial.print(suspension.K_s, 3);
                Serial.print(" Bs="); Serial.print(suspension.B_s, 3);
                Serial.print(" Kroll="); Serial.print(suspension.K_roll, 3);
                Serial.print(" Droll="); Serial.print(suspension.D_roll, 3);
                Serial.print(" q_nom="); Serial.print(suspension.q_nom * 57.2958f, 1);
                Serial.println(" deg");
                break;
            case 'b':
                if (state.mode == Mode::IDLE) {
                    state.mode = Mode::BALANCE;
                    vel_pi.reset();
                    yaw_pi.reset();
                    Serial.println("[Cmd] Mode → BALANCE");
                } else if (state.mode == Mode::BALANCE) {
                    state.mode = Mode::IDLE;
                    Serial.println("[Cmd] Mode → IDLE");
                } else if (state.mode == Mode::FAULT) {
                    state.mode = Mode::IDLE;
                    Serial.println("[Cmd] Mode → IDLE (cleared FAULT)");
                } else {
                    Serial.print("[Cmd] Mode is ");
                    Serial.print((uint8_t)state.mode);
                    Serial.println(" — switch to IDLE first");
                }
                break;
            case 'h':
            case '?':
                Serial.println("── Debug Commands ──");
                Serial.println("  p = profiler report");
                Serial.println("  s = state snapshot");
                Serial.println("  c = controller detail (gains + integrators)");
                Serial.println("  b = toggle BALANCE/IDLE mode");
                Serial.println("  h = this help");
                break;
            default:
                break;
        }
    }
}

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(1000000);

    uint32_t serial_wait_start = millis();
    while (!Serial && (millis() - serial_wait_start < 3000)) {
        delay(10);
    }

    Serial.println();
    Serial.println("========================================");
    Serial.println("  DEBUG HARNESS — Serial OK");
    Serial.print("  Board alive at t=");
    Serial.print(millis());
    Serial.println(" ms");
    Serial.println("========================================");

    // ── Boot menu ──
    serial_flush_input();
    run_boot_menu();

    // ── Register profiler slots ──
    p_total      = profiler_register("total");
    p_imu        = profiler_register("imu");
    p_vel_pi     = profiler_register("vel_pi");
    p_lqr        = profiler_register("lqr");
    p_yaw_pi     = profiler_register("yaw_pi");
    p_suspension = profiler_register("suspension");
    p_led        = profiler_register("led_matrix");
    p_gpio       = profiler_register("gpio");
    p_serial_cmd = profiler_register("serial_cmd");

    // ── Initialise enabled modules ──
    pinMode(PIN_STATUS_LED, OUTPUT);

    if (en_led_matrix) {
        Serial.print("[Init] LED matrix... ");
        uint32_t t0 = micros();
        led_matrix_init();
        Serial.print("OK (");
        Serial.print(micros() - t0);
        Serial.println(" us)");
    }

    if (en_imu) {
        Serial.print("[Init] IMU (BNO086 SPI)... ");
        uint32_t t0 = millis();
        bool ok = imu_init();
        Serial.print(ok ? "OK" : "FAILED");
        Serial.print(" (");
        Serial.print(millis() - t0);
        Serial.println(" ms)");
        if (!ok) {
            Serial.println("[Init] IMU failed — disabling IMU module");
            en_imu = false;
        }
    }

    if (en_controllers) {
        lqr.init();
        vel_pi.init();
        yaw_pi.init();
        suspension.init();
        Serial.println("[Init] Controllers OK");
        Serial.print("  LQR K=[");
        Serial.print(lqr.K[0], 4); Serial.print(", ");
        Serial.print(lqr.K[1], 4); Serial.print(", ");
        Serial.print(lqr.K[2], 4); Serial.println("]");
    }

    // WiFi (currently disabled in firmware)
    if (en_wifi) {
        Serial.println("[Init] WiFi — NOT YET IMPLEMENTED, skipping");
        en_wifi = false;
    }
    if (en_telemetry) {
        Serial.println("[Init] Telemetry — requires WiFi, skipping");
        en_telemetry = false;
    }
    if (en_commands) {
        Serial.println("[Init] Commands — requires WiFi, skipping");
        en_commands = false;
    }

    // ── 500 Hz timer ──
    if (!timer_init_500hz()) {
        Serial.println("FATAL: timer init failed");
        while (1) delay(1000);
    }

    // ── Watchdog (last, after all init) ──
    if (en_watchdog) {
        watchdog_init();
    }

    state.mode = Mode::IDLE;
    state.tick = 0;

    Serial.println();
    Serial.println("[Debug] Entering superloop. Press 'h' for runtime commands.");
    Serial.println();
}

// ── Main loop ────────────────────────────────────────────────────────────────
void loop() {
    if (!tick_flag) return;
    tick_flag = false;

    watchdog_feed();

    PROFILE_BEGIN(total);
    const float dt = 1.0f / LOOP_RATE_HZ;

    // ── GPIO toggle (scope probe point) ──
    PROFILE_BEGIN(gpio);
    digitalWrite(PIN_STATUS_LED, state.tick & 1);
    PROFILE_END(gpio, p_gpio);

    // ── IMU ──
    if (en_imu) {
        PROFILE_BEGIN(imu);
        imu_poll(&state);
        PROFILE_END(imu, p_imu);
    }

    // ── Controllers ──
    if (en_controllers && (state.mode == Mode::BALANCE || state.mode == Mode::DRIVE)) {
        // Safety: fall detection
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
        } else {
            float v_measured = state.wheel_vel_avg * WHEEL_RADIUS;
            float v_ref_rad  = state.v_cmd / WHEEL_RADIUS;

            // VelocityPI → theta_ref
            PROFILE_BEGIN(vpi);
            state.theta_ref = vel_pi.update(state.v_cmd, v_measured, dt);
            PROFILE_END(vpi, p_vel_pi);

            // LQR → tau_sym
            PROFILE_BEGIN(lqr_p);
            state.tau_sym = lqr.update(state.pitch, state.pitch_rate,
                                       state.wheel_vel_avg,
                                       state.theta_ref, v_ref_rad);
            PROFILE_END(lqr_p, p_lqr);

            // YawPI → tau_yaw
            PROFILE_BEGIN(ypi);
            float yaw_rate = 0.0f;  // TODO: get from IMU or wheel differential
            state.tau_yaw = yaw_pi.update(state.omega_cmd, yaw_rate, dt);
            PROFILE_END(ypi, p_yaw_pi);

            // Wheel torque split
            state.tau_wheel_L = constrain(state.tau_sym - state.tau_yaw,
                                          -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX);
            state.tau_wheel_R = constrain(state.tau_sym + state.tau_yaw,
                                          -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX);

            // Suspension → hip torques
            PROFILE_BEGIN(susp);
            suspension.update(state.roll, state.roll_rate,
                              state.hip_q_L, state.hip_dq_L,
                              state.hip_q_R, state.hip_dq_R,
                              state.tau_hip_L, state.tau_hip_R);
            PROFILE_END(susp, p_suspension);
        }
    } else if (state.mode == Mode::IDLE || state.mode == Mode::FAULT) {
        state.tau_sym = 0.0f;
        state.tau_yaw = 0.0f;
        state.tau_wheel_L = 0.0f;
        state.tau_wheel_R = 0.0f;
        state.tau_hip_L = 0.0f;
        state.tau_hip_R = 0.0f;
    }

    // ── LED matrix (~6 Hz) ──
    if (en_led_matrix) {
        static uint32_t led_next_ms = 0;
        uint32_t now_ms = millis();
        if (now_ms >= led_next_ms) {
            led_next_ms = now_ms + 167;
            PROFILE_BEGIN(led);
            led_matrix_update(&state);
            PROFILE_END(led, p_led);
        }
    }

    state.tick++;

    // ── Process serial commands (non-blocking) ──
    PROFILE_BEGIN(scmd);
    process_serial_commands();
    PROFILE_END(scmd, p_serial_cmd);

    // ── Measure total ──
    PROFILE_END(total, p_total);
    state.dt_us = _prof_slots[p_total].last_us;

    // ── Hard timing check ──
    if (state.dt_us > LOOP_PERIOD_US) {
        loop_overruns++;
        if (state.overrun_flash < 12) {
            state.overrun_flash = 12;
        }
    }

    // ── 1 Hz heartbeat ──
    if (state.tick % LOOP_RATE_HZ == 0) {
        heartbeat_count++;
        Serial.print("[HB] #");
        Serial.print(heartbeat_count);
        Serial.print("  tick=");
        Serial.print(state.tick);
        Serial.print("  dt_avg=");
        Serial.print(_prof_slots[p_total].count > 0
                     ? _prof_slots[p_total].sum_us / _prof_slots[p_total].count : 0);
        Serial.print("  dt_max=");
        Serial.print(_prof_slots[p_total].max_us);
        Serial.print("  overruns=");
        Serial.print(loop_overruns);
        if (en_imu) {
            Serial.print("  pitch=");
            Serial.print(state.pitch * 57.2958f, 1);
            Serial.print("°  imu=");
            Serial.print(state.imu_ok ? "OK" : "STALE");
        }
        Serial.print("  mode=");
        Serial.print((uint8_t)state.mode);
        if (en_controllers && (state.mode == Mode::BALANCE || state.mode == Mode::DRIVE)) {
            Serial.print("  tau_sym=");
            Serial.print(state.tau_sym, 3);
        }
        Serial.print("  wdt=");
        Serial.print(_wdt_enabled ? "ON" : "OFF");
        Serial.println();

        // Auto-print profiler every 5 seconds
        if (heartbeat_count % 5 == 0) {
            profiler_print_and_reset();
        }
    }
}

#endif // DEBUG_BUILD
