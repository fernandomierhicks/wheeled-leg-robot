#include "hip_motors.h"
#include "config.h"
#include "robot_state.h"
#include "comm_protocol.h"
#include "param_registry.h"
#include "state_machine.h"
#include <Arduino.h>
#include <FlexCAN_T4.h>

// Maximum allowed position step per command (rad). Larger jumps trigger ESTOP.
#define MAX_HIP_DELTA_RAD  1.5708f  // 90 deg

// Encoder-zero settling window. Zeroing redefines the motor's position frame:
// our next command is already in the new frame while replies still in flight
// carry the old one, so |cmd - feedback| is meaningless until the motor answers
// in the new frame. A reply is taken as confirmation once it is newer than the
// zero command AND reads near zero — matching the tolerance calibration uses
// for the same handshake. TIMEOUT_MS bounds how long the guard can stay down if
// confirmation never comes; after it, the guard resumes and a motor that really
// failed to zero faults on the next command, which is what we want.
#define HM_ZERO_SYNC_TOL_RAD    0.05f
#define HM_ZERO_SYNC_TIMEOUT_MS 200u

// AK45-10 MIT Cheetah protocol parameter limits.
//
// These are PER MOTOR MODEL. Position, Kp and Kd are common to the whole AK
// family, but speed and torque come from the model table in section 5.3 of the
// AK series driver manual (v1.0.18 — whose changelog specifically notes
// "Corrected the AK45-10 motor parameters", so older copies of that table are
// wrong for this motor). For the AK45-10: speed ±20 rad/s, torque ±8 N·m.
//
// Getting these wrong is silent and expensive in both directions: the motor
// decodes a command with its own constants (so a mismatched T_MAX scales every
// feedforward torque we send), and we decode its reply with ours (so the same
// mismatch scales everything we report). A previous set of values here (V ±65,
// T ±18, plus a separate I ±20 for the reply) came from another AK model and
// made reported hip torque read 2.5× high while delivering 44 % of commanded
// feedforward. Confirmed against bench log 20260728T053232: reconstructing the
// MIT impedance law from telemetry fit the reported value at 0.400 = 8/20, and
// position-derivative vs reported velocity fit 0.308 = 20/65.
#define P_MIN   -12.5f
#define P_MAX    12.5f
#define V_MIN   -20.0f
#define V_MAX    20.0f
#define KP_MIN    0.0f
#define KP_MAX  500.0f
#define KD_MIN    0.0f
#define KD_MAX    5.0f
#define T_MIN    -8.0f
#define T_MAX     8.0f

// ── No periodic MIT re-entry. Measured, not assumed (2026-08-07) ─────────────
// There used to be an unconditional `enter_mit()` every MIT_REENTER_MS (3000u)
// here, justified by "motor silently drops out of MIT mode after ~4 s without
// re-entry". That observation dates from f4e143c (2026-05-08), when
// hip_motors_poll() sent NO periodic frames at all — it only checked feedback
// timeouts and re-entered MIT, so the re-entry was the only CAN traffic the
// hips ever saw. The always-send setpoint/ping path landed a month later
// (6f1a515, 2026-06-09), making poll() emit a frame every tick at 500 Hz in
// every state where mit_active is set. Nobody re-tested the re-entry against
// that, so it survived as vestigial.
//
// It was not free: enter-motor-mode is a state transition, not an idempotent
// refresh, and it landed ~100 us before the next stiff setpoint frame — a
// periodic disturbance in every state, RUNNING included. Paired with a
// duplicate 4 s re-entry the GUI used to run in MANUAL (removed from
// hip_motors.py), the two beat into a double glitch every 4 s, which is how
// this was first noticed as an audible repetitive jerk.
//
// Bench measurement, right hip on a stand, USB transport, re-entry disabled,
// log data/logs/runs/20260808T034429_031257Z_HOST: 120 s of continuous 0.125 Hz
// sine commanding, MANUAL, no faults. Longest span where commanded position
// moved while actual position did not: 40 ms (sine turning points + the 16-bit
// 0.00038 rad/LSB quantisation). Zero of 6015 samples had |torque| < 0.05 N.m,
// mean 1.348 N.m — a motor out of MIT mode makes no torque at all. So MIT
// survived 30x the claimed 4 s dropout window, under load, on the 500 Hz
// stream alone.
//
// If MIT ever does drop out, the existing feedback watchdog is the correct
// response: the motor stops replying, hm_*.ok goes false after
// HIP_CAN_TIMEOUT_MS, and the state machine ESTOPs on FAULT_HIP_FEEDBACK_LOST.
// That surfaces the fault instead of papering over it on a timer. Do not
// reintroduce a blind periodic re-entry; enter_mit() still has its explicit
// call sites (boot, ESTOP recovery, GUI ENABLE).

// A cached setpoint not refreshed within this window is considered stale and
// poll() reverts to the safe zero-torque ping.
#define HIP_SETPOINT_TIMEOUT_MS  5000u

HipAxisState hm_L = {};
HipAxisState hm_R = {};
HipSetpoint  hm_sp_L = {};
HipSetpoint  hm_sp_R = {};
HipLimits    hm_limits_L = {};
HipLimits    hm_limits_R = {};

// CAN2 on Teensy 4.1 uses pins 1 (TX) and 0 (RX) — matches config.h PIN_CAN2_*.
static FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
static bool s_can_initialized = false;

// ── helpers ──────────────────────────────────────────────────────────────────

static uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    uint32_t max_val = (1u << bits) - 1;
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    return (uint16_t)((x - x_min) / (x_max - x_min) * (float)max_val);
}

static float uint_to_float(uint16_t x, float x_min, float x_max, int bits) {
    uint32_t max_val = (1u << bits) - 1;
    return (float)x / (float)max_val * (x_max - x_min) + x_min;
}

// Clamp a commanded position to the calibrated software limits, if valid.
static float clamp_to_limits(float pos, const HipLimits& lim) {
    if (!lim.valid) return pos;
    if (pos < lim.min_rad) return lim.min_rad;
    if (pos > lim.max_rad) return lim.max_rad;
    return pos;
}

// Public alias — callers that ramp from the measured pose need the same clamp
// the send path applies (see hip_motors.h).
float hip_motors_clamp_to_limits(float pos, const HipLimits& lim) {
    return clamp_to_limits(pos, lim);
}

// ── CAN TX health ────────────────────────────────────────────────────────────
// Same contract and reasoning as CAN3 in wheel_motors.cpp: FlexCAN_T4::write()
// returns 1 for "into a hardware mailbox" and -1 for "mailboxes all busy,
// queued to the bounded software TX queue", which drops silently once full.
// One deferral is normal, a long run means MIT setpoints have stopped reaching
// the hips. Threshold matches CAN3's (~200 ms of a bus that isn't draining).
static constexpr uint32_t HM_TX_DEFER_FAULT_RUN = 200;
static uint32_t s_tx_defer_total = 0;
static uint32_t s_tx_defer_run   = 0;
static bool     s_tx_stalled     = false;

static void note_tx(int rc) {
    if (rc == 1) { s_tx_defer_run = 0; return; }
    s_tx_defer_total++;
    if (s_tx_defer_run < UINT32_MAX) s_tx_defer_run++;
}

uint32_t hip_motors_tx_defer_count() { return s_tx_defer_total; }

static void send_raw(uint32_t id, const uint8_t data[8]) {
    CAN_message_t msg = {};
    msg.id  = id;
    msg.len = 8;
    memcpy(msg.buf, data, 8);
    note_tx(can2.write(msg));
}

static void pack_and_send(uint32_t id, float pos, float vel, float kp, float kd, float torque) {
    // Guard against large position jumps — fault and suppress the frame.
    const HipAxisState* ax = (id == AK45_ID_L) ? &hm_L : &hm_R;
    if (ax->ever_heard && !ax->zero_pending) {
        float delta = pos - ax->pos_rad;
        if (delta < 0) delta = -delta;
        if (delta > MAX_HIP_DELTA_RAD) {
            const char* side = (id == AK45_ID_L) ? "L" : "R";
            comm_log(LOG_LEVEL_ERROR, "FAULT: hip %s pos jump %.3f rad > %.3f", side, delta, MAX_HIP_DELTA_RAD);
            // Request the transition through the FSM rather than writing
            // g_state.state directly — a direct write desyncs it from the
            // StateMachine library's own private currentState index, which
            // then keeps re-running the state we were actually in (masking
            // this fault entirely; see calibration double-run bug).
            g_state.fault_code = FAULT_HIP_LARGE_POS_CMD;
            stateMachine_request_estop();
            return;
        }
    }

    // Left hip is physically mirrored relative to right — negate pos/vel/torque
    // here, once, so every caller (calibration, jump FSM, GUI MIT frames, radio
    // hip command) can work in one consistent frame where positive means the
    // same physical direction on both sides, same as WheelMotors does for R.
    float pos_hw = pos, vel_hw = vel, torque_hw = torque;
    if (id == AK45_ID_L) { pos_hw = -pos_hw; vel_hw = -vel_hw; torque_hw = -torque_hw; }

    uint16_t p   = float_to_uint(pos_hw,    P_MIN,  P_MAX,  16);
    uint16_t v   = float_to_uint(vel_hw,    V_MIN,  V_MAX,  12);
    uint16_t kp_ = float_to_uint(kp,        KP_MIN, KP_MAX, 12);
    uint16_t kd_ = float_to_uint(kd,        KD_MIN, KD_MAX, 12);
    uint16_t t   = float_to_uint(torque_hw, T_MIN,  T_MAX,  12);

    uint8_t buf[8];
    buf[0] = p >> 8;
    buf[1] = p & 0xFF;
    buf[2] = v >> 4;
    buf[3] = ((v & 0xF) << 4) | (kp_ >> 8);
    buf[4] = kp_ & 0xFF;
    buf[5] = kd_ >> 4;
    buf[6] = ((kd_ & 0xF) << 4) | (t >> 8);
    buf[7] = t & 0xFF;
    send_raw(id, buf);
}

// ── CAN RX callback (called from FlexCAN_T4 ISR) ─────────────────────────────

static void rx_callback(const CAN_message_t& msg) {
    if (msg.len < 6) return;

    HipAxisState* ax;
    if      (msg.id == AK45_ID_L) ax = &hm_L;
    else if (msg.id == AK45_ID_R) ax = &hm_R;
    else return;

    uint16_t raw_pos = ((uint16_t)msg.buf[1] << 8) | msg.buf[2];
    uint16_t raw_vel = ((uint16_t)msg.buf[3] << 4) | (msg.buf[4] >> 4);
    uint16_t raw_trq = ((uint16_t)(msg.buf[4] & 0xF) << 8) | msg.buf[5];

    float pos = uint_to_float(raw_pos, P_MIN, P_MAX, 16);
    float vel = uint_to_float(raw_vel, V_MIN, V_MAX, 12);
    // The manual's byte table labels this third field "current", but its own
    // reference decoder (section 5.3, unpack_reply) scales it by the model's
    // TORQUE range and names the result `torque`. It is shaft torque in N·m,
    // not amps — see the T_MIN/T_MAX note above.
    float trq = uint_to_float(raw_trq, T_MIN, T_MAX, 12);
    // Left hip is physically mirrored relative to right — negate feedback here
    // so downstream code (gain scheduling, FF1 torque sum, telemetry, GUI)
    // sees a consistent frame; must match the TX-side flip in pack_and_send().
    if (msg.id == AK45_ID_L) { pos = -pos; vel = -vel; trq = -trq; }
    ax->pos_rad    = pos;
    ax->vel_rad_s  = vel;
    ax->torque_nm  = trq;
    ax->last_fb_ms = millis();
    ax->feedback_seq++;
    ax->ever_heard = true;
}

// ── public API ────────────────────────────────────────────────────────────────

bool hip_motors_init() {
    can2.begin();
    can2.setBaudRate(CAN_BAUD);
    can2.setMaxMB(16);
    can2.enableFIFO();
    can2.enableFIFOInterrupt();
    can2.onReceive(rx_callback);
    s_can_initialized = true;
    Serial.print("[HipMotors] CAN2 init OK  ");
    Serial.print(CAN_BAUD / 1000);
    Serial.print(" kbps  id_L=0x");
    Serial.print(AK45_ID_L, HEX);
    Serial.print("  id_R=0x");
    Serial.println(AK45_ID_R, HEX);
    return true;
}

// End the encoder-zero settling window once the motor has answered in the new
// frame. Both conditions matter: feedback_seq must have moved past the reply
// that was already in flight when the zero went out (that one still carries the
// OLD frame), and the position it carries must read near zero, which only a
// genuinely re-zeroed motor reports.
static void clear_zero_pending_if_synced(HipAxisState* ax, uint32_t now) {
    if (!ax->zero_pending) return;
    if ((ax->feedback_seq != ax->zero_fb_seq &&
         fabsf(ax->pos_rad) <= HM_ZERO_SYNC_TOL_RAD) ||
        (uint32_t)(now - ax->zero_cmd_ms) > HM_ZERO_SYNC_TIMEOUT_MS) {
        ax->zero_pending = false;
    }
}

void hip_motors_poll() {
    uint32_t now = millis();

    // Latched CAN2 TX stall — reported once, not every tick. Clears ok on both
    // axes: feedback can keep arriving while nothing we send gets out, and a
    // hip we cannot command is not a hip we can hold a pose with.
    if (!s_tx_stalled && s_tx_defer_run >= HM_TX_DEFER_FAULT_RUN) {
        s_tx_stalled = true;
        comm_log(LOG_LEVEL_ERROR,
                 "HipMotors: CAN2 TX stalled — %lu consecutive deferrals (%lu total); "
                 "MIT setpoints are no longer reaching the bus",
                 (unsigned long)s_tx_defer_run, (unsigned long)s_tx_defer_total);
    }

    hm_L.ok = hm_L.ever_heard && !s_tx_stalled && (now - hm_L.last_fb_ms) < HIP_CAN_TIMEOUT_MS;
    hm_R.ok = hm_R.ever_heard && !s_tx_stalled && (now - hm_R.last_fb_ms) < HIP_CAN_TIMEOUT_MS;

    // Retire the encoder-zero settling window before this tick's frames go out,
    // so the jump guard is back in force the moment feedback is comparable again.
    clear_zero_pending_if_synced(&hm_L, now);
    clear_zero_pending_if_synced(&hm_R, now);

    // (No periodic enter_mit() here — see the note above MIT_REENTER_MS's
    // former home at the top of this file.)

    // ESTOP or a stale (unrefreshed) setpoint both fall back to the safe ping —
    // except during the brief, time-bounded gentle-cutoff ramp (see
    // state_machine.cpp on_estop()), which needs its tapering setpoint to
    // survive for its ~1 s duration instead of being zeroed the instant
    // STATE_ESTOP is entered.
    if (g_state.state == STATE_ESTOP && !stateMachine_is_estop_hip_ramping()) {
        hm_sp_L.active = false;
        hm_sp_R.active = false;
    }
    if (hm_sp_L.active && (now - hm_sp_L.last_cmd_ms) > HIP_SETPOINT_TIMEOUT_MS) hm_sp_L.active = false;
    if (hm_sp_R.active && (now - hm_sp_R.last_cmd_ms) > HIP_SETPOINT_TIMEOUT_MS) hm_sp_R.active = false;

    // While a setpoint is active, re-send it every tick so it isn't overridden
    // by the zero-torque ping below. Otherwise ping with current-position +
    // zero-torque so the AK45 returns feedback every frame. Each side is gated
    // independently so a disabled/absent motor never gets CAN traffic.
    if (hm_L.mit_active && param_get(PARAM_HIP_L_ENABLE) >= 0.5f) {
        if (hm_sp_L.active)
            pack_and_send(AK45_ID_L, hm_sp_L.p, hm_sp_L.v, hm_sp_L.kp, hm_sp_L.kd, hm_sp_L.tff);
        else
            pack_and_send(AK45_ID_L, hm_L.pos_rad, 0.0f, 0.0f, 0.0f, 0.0f);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (hm_R.mit_active && param_get(PARAM_HIP_R_ENABLE) >= 0.5f) {
        if (hm_sp_R.active)
            pack_and_send(AK45_ID_R, hm_sp_R.p, hm_sp_R.v, hm_sp_R.kp, hm_sp_R.kd, hm_sp_R.tff);
        else
            pack_and_send(AK45_ID_R, hm_R.pos_rad, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}

void hip_motors_enter_mit() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    bool l_en = param_get(PARAM_HIP_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_HIP_R_ENABLE) >= 0.5f;
    if (l_en) {
        send_raw(AK45_ID_L, cmd);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (r_en) send_raw(AK45_ID_R, cmd);
    hm_L.mit_active = l_en;
    hm_R.mit_active = r_en;
}

void hip_motors_exit_mit() {
    bool delay_between = hm_L.mit_active && hm_R.mit_active && s_can_initialized;
    hip_motor_disable_L();
    if (delay_between) delayMicroseconds(CAN_INTER_FRAME_US);
    hip_motor_disable_R();
}

void hip_motor_disable_L() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    hm_sp_L.active = false;
    if (hm_L.mit_active && s_can_initialized) send_raw(AK45_ID_L, cmd);
    hm_L.mit_active = false;
}

void hip_motor_disable_R() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    hm_sp_R.active = false;
    if (hm_R.mit_active && s_can_initialized) send_raw(AK45_ID_R, cmd);
    hm_R.mit_active = false;
}

// Arm the settling window described at HM_ZERO_SYNC_TOL_RAD. Must be called for
// every axis a zero command goes out to, or the position-jump guard will fault
// on the frame discontinuity the zero just created.
static void mark_zero_pending(HipAxisState* ax) {
    ax->zero_pending = true;
    ax->zero_cmd_ms  = millis();
    ax->zero_fb_seq  = ax->feedback_seq;
}

void hip_motors_zero() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    bool l_en = param_get(PARAM_HIP_L_ENABLE) >= 0.5f;
    bool r_en = param_get(PARAM_HIP_R_ENABLE) >= 0.5f;
    if (l_en) {
        mark_zero_pending(&hm_L);
        send_raw(AK45_ID_L, cmd);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (r_en) {
        mark_zero_pending(&hm_R);
        send_raw(AK45_ID_R, cmd);
    }
    comm_log(LOG_LEVEL_INFO, "Hip encoders zeroed (L=%d R=%d)", (int)l_en, (int)r_en);
}

void hip_motor_zero_L() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    mark_zero_pending(&hm_L);
    send_raw(AK45_ID_L, cmd);
    comm_log(LOG_LEVEL_INFO, "Hip encoder zeroed (L)");
}

void hip_motor_zero_R() {
    static const uint8_t cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    mark_zero_pending(&hm_R);
    send_raw(AK45_ID_R, cmd);
    comm_log(LOG_LEVEL_INFO, "Hip encoder zeroed (R)");
}

void hip_motors_send(float pos_L, float vel_L, float kp_L, float kd_L, float trq_L,
                     float pos_R, float vel_R, float kp_R, float kd_R, float trq_R) {
    // Gate each motor independently — with only one leg enabled (bench config)
    // the active side must still receive its frame.
    if (hm_L.mit_active && param_get(PARAM_HIP_L_ENABLE) >= 0.5f) {
        pack_and_send(AK45_ID_L, clamp_to_limits(pos_L, hm_limits_L), vel_L, kp_L, kd_L, trq_L);
        delayMicroseconds(CAN_INTER_FRAME_US);
    }
    if (hm_R.mit_active && param_get(PARAM_HIP_R_ENABLE) >= 0.5f)
        pack_and_send(AK45_ID_R, clamp_to_limits(pos_R, hm_limits_R), vel_R, kp_R, kd_R, trq_R);
}

void hip_motor_send_L(float pos, float vel, float kp, float kd, float torque) {
    if (!hm_L.mit_active || param_get(PARAM_HIP_L_ENABLE) < 0.5f) return;
    pack_and_send(AK45_ID_L, clamp_to_limits(pos, hm_limits_L), vel, kp, kd, torque);
}

void hip_motor_send_R(float pos, float vel, float kp, float kd, float torque) {
    if (!hm_R.mit_active || param_get(PARAM_HIP_R_ENABLE) < 0.5f) return;
    pack_and_send(AK45_ID_R, clamp_to_limits(pos, hm_limits_R), vel, kp, kd, torque);
}

void hip_motors_set_setpoint_L(float pos, float vel, float kp, float kd, float torque) {
    if (param_get(PARAM_HIP_L_ENABLE) < 0.5f) { hm_sp_L.active = false; return; }
    hm_sp_L = {clamp_to_limits(pos, hm_limits_L), vel, kp, kd, torque, true, millis()};
}

void hip_motors_set_setpoint_R(float pos, float vel, float kp, float kd, float torque) {
    if (param_get(PARAM_HIP_R_ENABLE) < 0.5f) { hm_sp_R.active = false; return; }
    hm_sp_R = {clamp_to_limits(pos, hm_limits_R), vel, kp, kd, torque, true, millis()};
}

void hip_motors_clear_setpoints() {
    hm_sp_L.active = false;
    hm_sp_R.active = false;
}

void hip_cmd_to_setpoints(float t, float* pos_L, float* pos_R) {
    float span_L = hm_limits_L.max_rad - hm_limits_L.min_rad;
    float span_R = hm_limits_R.max_rad - hm_limits_R.min_rad;
    float dir_L  = CALIB_L_SEEK_DIR;
    float dir_R  = CALIB_R_SEEK_DIR;
    *pos_L = (dir_L > 0.0f) ? (hm_limits_L.max_rad - t * span_L)
                             : (hm_limits_L.min_rad + t * span_L);
    *pos_R = (dir_R > 0.0f) ? (hm_limits_R.max_rad - t * span_R)
                             : (hm_limits_R.min_rad + t * span_R);
}

bool hip_motors_ok() {
    bool l_ok = (param_get(PARAM_HIP_L_ENABLE) < 0.5f) || hm_L.ok;
    bool r_ok = (param_get(PARAM_HIP_R_ENABLE) < 0.5f) || hm_R.ok;
    return l_ok && r_ok;
}

void hip_motors_forgive_feedback_stall() {
    // AK45 MIT feedback is request-response: the motor only replies to a
    // command frame. A deliberate, known-blocking main-loop operation (opening
    // or finalizing an SD log) can freeze the 500 Hz tick for tens of ms, so no
    // command goes out and last_fb_ms goes stale — which the next poll() would
    // otherwise read as a motor dropout and ESTOP. Reset the freshness clock so
    // that self-inflicted stall isn't blamed on the motor. This grants exactly
    // one HIP_CAN_TIMEOUT_MS of grace; a genuinely dead motor still faults on
    // the following interval because no real reply refreshes last_fb_ms.
    uint32_t now = millis();
    hm_L.last_fb_ms = now;
    hm_R.last_fb_ms = now;
}
