#include <Arduino.h>
#include "state_machine.h"
#include "robot_state.h"
#include "IMU.h"
#include "hip_motors.h"
#include "calibration.h"
#include "comm_protocol.h"
#include "config.h"
#include "param_registry.h"
#include "control_loop.h"
#include "wheel_motors.h"
#include "Buzzer.h"
#include <StateMachine.h>

extern Buzzer g_buzzer;

static const BuzzerNote ARMED_MELODY[] = {
    {79, 80, 20},   // G5
    {86, 120,  0},  // D6 — rising fifth: "armed"
};
static const BuzzerNote DISARMED_MELODY[] = {
    {86, 80, 20},   // D6
    {79, 120,  0},  // G5 — falling fifth: "safe"
};
static const BuzzerNote REJECT_MELODY[] = {
    {60, 100, 40},  // C4
    {55, 180,  0},  // G3 — low descending: "denied"
};
static const BuzzerNote ESTOP_MELODY[] = {
    {72, 80,  20},  // C5
    {72, 80,  20},  // C5
    {72, 80,  20},  // C5
    {60, 400,  0},  // C4 — three short blasts then low long: "danger"
};
static const BuzzerNote JUMP_MELODY[] = {
    // Launch countdown: quick ascending run
    {72,  80, 30},  // C5
    {74,  80, 30},  // D5
    {76,  80, 30},  // E5
    {79,  80, 30},  // G5
    // Power-up hits
    {84, 150, 40},  // C6
    {83, 150, 40},  // B5
    {84, 150, 40},  // C6
    // Big launch arc
    {86, 250, 50},  // D6
    {84, 250, 50},  // C6
    {81, 250, 50},  // A5
    // Victory fanfare
    {79, 120, 20},  // G5
    {84, 120, 20},  // C6
    {88, 400,  0},  // E6 — hold: ~2.6 s total, state timer holds to 3 s
};
static const BuzzerNote STANDUP_MELODY[] = {
    // Fallen -> recovering: low, unresolved — outcome (capture vs. fail) unknown yet
    {48,  90, 30},  // C3
    {50,  90, 30},  // D3
    {53, 150,  0},  // F3
};

// ── State machine ─────────────────────────────────────────────────────────────

static StateMachine sm;

static State* S_STARTUP;
static State* S_STANDBY;
static State* S_MANUAL;
static State* S_CALIBRATION;
static State* S_RUNNING;
static State* S_ESTOP;
static State* S_CMD_REJECT;
static State* S_JUMPING;
static State* S_STANDING_UP;
static State* S_DISARMING;

// ── ESTOP hip-disable tracking ────────────────────────────────────────────────
static bool s_estop_hip_disabled = false;  // set when MIT was killed on ESTOP entry
static bool s_hip_disarm_ramping = false;  // set while ramping hip torque down before STANDBY
static bool s_disarming_calibration = false;

// ── ESTOP gentle hip-cutoff ramp ───────────────────────────────────────────────
// Compatibility hook for stateMachine_estop_hip_ramping(). ESTOP is now an
// immediate cutoff; normal hip ramp-down belongs only to DISARMING.
static bool s_estop_hip_ramping = false;
static bool s_disarm_done = false;

// ── Pending mode-change requests (set by command handler) ─────────────────────
static volatile bool s_req_manual        = false;
static volatile bool s_req_standby       = false;
static volatile bool s_req_reset         = false;
static volatile bool s_req_calibration   = false;
static volatile bool s_req_running       = false;
static volatile bool s_req_disarm_running = false;
static volatile bool s_req_disarm_calibration = false;
static volatile bool s_req_estop         = false;
static volatile bool s_req_cmd_reject    = false;
static volatile bool s_req_soft_clear    = false;
static volatile bool s_req_jump          = false;

// ── CMD_REJECT auto-exit timer ────────────────────────────────────────────────
static uint32_t s_cmd_reject_deadline_ms = 0;

// ── JUMPING auto-exit timer ───────────────────────────────────────────────────
static uint32_t s_jump_deadline_ms = 0;

// ── Jump FSM state (Phase 7) ──────────────────────────────────────────────────
typedef enum : uint8_t {
    JP_CROUCH  = 0,
    JP_EXTEND  = 1,
    JP_RETRACT = 2,
    JP_DONE    = 3,
} JumpPhase;

static JumpPhase s_jp_phase        = JP_DONE;
static uint32_t  s_jp_phase_ms     = 0;       // millis() when current phase started
static float     s_jp_nom_L        = 0.0f;    // hip pos at jump trigger (RETRACT target)
static float     s_jp_nom_R        = 0.0f;
static float     s_jp_ret_L        = 0.0f;    // calibrated retracted pos (CROUCH target)
static float     s_jp_ret_R        = 0.0f;
static float     s_jp_from_L       = 0.0f;    // hip pos when RETRACT began
static float     s_jp_from_R       = 0.0f;

// ── Standing-up FSM state ─────────────────────────────────────────────────────
typedef enum : uint8_t { SU_CROUCH = 0, SU_RECOVER = 1, SU_PAUSE = 2 } StandupPhase;
// Divergence debounce — fixed duration (not a tunable param), same rationale
// as control_loop.cpp's PITCH_WATCHDOG_MS: a safety-timeout, not a tuning knob.
static constexpr uint32_t STANDUP_DIVERGE_MS = 50;
static StandupPhase s_su_phase             = SU_CROUCH;
static uint32_t     s_su_phase_ms          = 0;
static float        s_su_nom_L, s_su_nom_R;       // hip pos snapshot at entry
static float        s_su_ret_L, s_su_ret_R;       // calibrated retracted target
static uint8_t      s_su_attempt           = 0;   // 1-based RECOVER attempt counter
static uint32_t     s_su_capture_since_ms  = 0;   // 0 = not currently in-band
static uint32_t     s_su_diverge_start_ms  = 0;   // 0 = not currently past the divergence limit
static bool         s_su_captured          = false;

// ── MANUAL GUI watchdog ───────────────────────────────────────────────────────
// Fed by any COMM_TYPE_COMMAND packet; the GUI sends CMD_ID_PING at 10 Hz so a
// GUI crash/disconnect exits MANUAL (and stops wheels) within ~500 ms.
static uint32_t s_last_gui_packet_ms  = 0;
static const uint32_t MANUAL_GUI_TIMEOUT_MS = 500;

// ── State actions ─────────────────────────────────────────────────────────────

static void on_startup()  {
    bool entering = (g_state.state != STATE_STARTUP);
    g_state.state = STATE_STARTUP;
    if (entering) {
        g_state.fault_code = FAULT_NONE;
        comm_log(LOG_LEVEL_INFO, "-> STARTUP");
        // Defensive cleanup for firmware upgraded from the former ESTOP-ramp
        // behavior: never carry a cached hip command across reset.
        if (s_estop_hip_ramping) {
            s_estop_hip_ramping = false;
            hip_motors_clear_setpoints();
        }
    }
    if (s_estop_hip_disabled) {
        s_estop_hip_disabled = false;
        if (param_get(PARAM_ESTOP_HIP_DISABLE) >= 0.5f) hip_motors_enter_mit();
    }
}
static void on_standby()  {
    bool entering      = (g_state.state != STATE_STANDBY);
    bool from_calib    = (g_state.state == STATE_CALIBRATION);
    bool calib_done    = from_calib && calibration_done();
    bool from_estop    = (g_state.state == STATE_ESTOP);  // soft-clear path
    bool from_manual   = (g_state.state == STATE_MANUAL);
    bool from_startup  = (g_state.state == STATE_STARTUP);
    g_state.state = STATE_STANDBY;
    hip_motors_clear_setpoints();  // DISARMING has already completed any normal ramp

    g_state.whl_tau_l = 0.0f;
    g_state.whl_tau_r = 0.0f;
    if (from_calib && !calib_done) calibration_abort();
    if (from_manual) {
        wheel_motors_send(0.0f, 0.0f);
        wheel_motors_set_mode(WheelMode::IDLE);
        wheel_motors_clear_errors();
    }
    // Startup succeeded: IMU (if enabled) reached NOMINAL and both hip motors
    // replied — startup_ok() is the only path into STANDBY from STARTUP.
    if (entering && from_startup) comm_log(LOG_LEVEL_INFO, "Startup complete");
    if (entering) comm_log(LOG_LEVEL_INFO, "-> STANDBY");
    if (from_estop) {
        // Soft-clear: clear fault and restore hip MIT mode skipped by bypassing STARTUP.
        g_state.fault_code = FAULT_NONE;
        // Defensive cleanup for the former ESTOP-ramp behavior.
        s_estop_hip_ramping = false;
        if (s_estop_hip_disabled) {
            s_estop_hip_disabled = false;
            if (param_get(PARAM_ESTOP_HIP_DISABLE) >= 0.5f) hip_motors_enter_mit();
        }
    }
}

static void on_disarming() {
    bool entering = (g_state.state != STATE_DISARMING);
    bool from_calib = (g_state.state == STATE_CALIBRATION);
    g_state.state = STATE_DISARMING;
    if (entering) {
        comm_log(LOG_LEVEL_INFO, "-> DISARMING");
        wheel_motors_send(0.0f, 0.0f);
        wheel_motors_set_mode(WheelMode::IDLE);
        g_state.whl_tau_l = g_state.whl_tau_r = g_state.tau_sym = g_state.tau_yaw = 0.0f;
        s_jp_phase = JP_DONE;
        s_su_captured = false;
        s_disarming_calibration = from_calib;
        if (s_disarming_calibration) {
            calibration_begin_disarm();
            s_hip_disarm_ramping = true;
            s_disarm_done = false;
        } else {
            controlLoop_reset_hip_disarm_ramp();
            s_hip_disarm_ramping =
                param_get(PARAM_HIP_RUNNING_RAMP_TIME_S) > 0.0f;
            s_disarm_done = !s_hip_disarm_ramping;
            if (s_disarm_done) hip_motors_clear_setpoints();
        }
    }
    if (s_hip_disarm_ramping) {
        const bool ramping = s_disarming_calibration
            ? calibration_run_disarm()
            : controlLoop_run_hip_disarm_ramp();
        if (!ramping) {
            s_hip_disarm_ramping = false;
            s_disarm_done = true;
            hip_motors_clear_setpoints();
        }
    }
}
static void on_manual() {
    bool entering = (g_state.state != STATE_MANUAL);
    g_state.state = STATE_MANUAL;
    if (entering) {
        s_last_gui_packet_ms = millis();  // start watchdog fresh; don't inherit stale timestamp
        g_hip_cmd.pending = false;        // discard any hip cmd latched before MANUAL entry
        comm_log(LOG_LEVEL_INFO, "-> MANUAL");
    }
    if (!g_hip_cmd.pending) return;
    g_hip_cmd.pending = false;
    switch (g_hip_cmd.sub_cmd) {
        case HIP_SUB_ENABLE:  hip_motors_enter_mit(); break;
        case HIP_SUB_DISABLE: hip_motors_exit_mit();  break;
        case HIP_SUB_ZERO:    hip_motors_zero();       break;
        case HIP_SUB_MIT:
            if      (g_hip_cmd.motor_id == HIP_MOTOR_L)
                hip_motors_set_setpoint_L(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            else if (g_hip_cmd.motor_id == HIP_MOTOR_R)
                hip_motors_set_setpoint_R(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            else {
                hip_motors_set_setpoint_L(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
                hip_motors_set_setpoint_R(g_hip_cmd.p, g_hip_cmd.v, g_hip_cmd.kp, g_hip_cmd.kd, g_hip_cmd.tff);
            }
            break;
        default: break;
    }
}
static void on_calibration() {
    bool entering = (g_state.state != STATE_CALIBRATION);
    g_state.state = STATE_CALIBRATION;
    if (entering) {
        comm_log(LOG_LEVEL_INFO, "-> CALIBRATION");
        calibration_start();
    }
    calibration_update();
}
static void on_running() {
    bool entering         = (g_state.state != STATE_RUNNING);
    bool from_jumping     = (g_state.state == STATE_JUMPING);
    bool from_standing_up = (g_state.state == STATE_STANDING_UP);
    g_state.state = STATE_RUNNING;
    if (entering) {
        comm_log(LOG_LEVEL_INFO, "-> RUNNING (armed)");
        g_buzzer.play(ARMED_MELODY, sizeof(ARMED_MELODY) / sizeof(ARMED_MELODY[0]));
        wheel_motors_set_mode(WheelMode::TORQUE);
        controlLoop_reset();  // C1: clear integrators/rate-limit state so each arm starts clean
        // Skip on a jump landing or a successful standup catch — the hips are
        // already stiffly holding the post-jump/post-catch position; re-ramping
        // kp from 0 here would loosen them right when they need to hold.
        if (!from_jumping && !from_standing_up) controlLoop_reset_hip_ramp();
    }
    controlLoop_run();
}
static void on_cmd_reject() {
    bool entering = (g_state.state != STATE_CMD_REJECT);
    g_state.state = STATE_CMD_REJECT;
    if (entering) {
        comm_log(LOG_LEVEL_WARN, "-> CMD_REJECT");
        g_buzzer.play(REJECT_MELODY, sizeof(REJECT_MELODY) / sizeof(REJECT_MELODY[0]));
        s_cmd_reject_deadline_ms = millis() + 1000;
    }
}
static void on_jumping() {
    bool entering = (g_state.state != STATE_JUMPING);
    g_state.state = STATE_JUMPING;

    if (entering) {
        comm_log(LOG_LEVEL_INFO, "-> JUMPING");
        g_buzzer.play(JUMP_MELODY, sizeof(JUMP_MELODY) / sizeof(JUMP_MELODY[0]));
        s_jump_deadline_ms = millis() + 3000;
        // Snapshot hip positions and calibrated retracted target
        s_jp_phase  = JP_CROUCH;
        s_jp_phase_ms = millis();
        s_jp_nom_L  = hm_L.pos_rad;
        s_jp_nom_R  = hm_R.pos_rad;
        hip_cmd_to_setpoints(0.0f, &s_jp_ret_L, &s_jp_ret_R);  // t=0 → retracted
    }

    // LQR wheel balance runs throughout (identical to RUNNING — keeps robot upright on real robot)
    controlLoop_run();

    g_state.jump_state = (uint8_t)s_jp_phase;

    // ── Master gate: leave at 0 until ready to test ───────────────────────────
    if (param_get(PARAM_JUMP_ENABLE) < 0.5f) return;

    // Require valid calibration limits before any hip torque is applied
    if (!hm_limits_L.valid || !hm_limits_R.valid) {
        comm_log(LOG_LEVEL_WARN, "JUMP: limits not valid — skipping hip cmds");
        return;
    }

    float hip_q_L  = hm_L.pos_rad;
    float hip_q_R  = hm_R.pos_rad;
    float hip_dq_L = hm_L.vel_rad_s;
    float hip_dq_R = hm_R.vel_rad_s;
    float elapsed  = (millis() - s_jp_phase_ms) / 1000.0f;
    float kp       = param_get(PARAM_JUMP_KP);
    float kd       = param_get(PARAM_JUMP_KD);

    if (s_jp_phase == JP_CROUCH) {
        float crouch_t = param_get(PARAM_JUMP_CROUCH_TIME_S);
        float alpha    = elapsed / crouch_t;
        if (alpha > 1.0f) alpha = 1.0f;
        float dq_L = (s_jp_ret_L - s_jp_nom_L) / crouch_t;
        float dq_R = (s_jp_ret_R - s_jp_nom_R) / crouch_t;
        hip_motors_set_setpoint_L(s_jp_nom_L + alpha * (s_jp_ret_L - s_jp_nom_L), dq_L, kp, kd, 0.0f);
        hip_motors_set_setpoint_R(s_jp_nom_R + alpha * (s_jp_ret_R - s_jp_nom_R), dq_R, kp, kd, 0.0f);
        if (elapsed >= crouch_t) {
            s_jp_phase    = JP_EXTEND;
            s_jp_phase_ms = millis();
        }
    }
    else if (s_jp_phase == JP_EXTEND) {
        // seek_dir points toward the retract switch; extension is opposite.
        float dir_L = CALIB_L_SEEK_DIR;
        float dir_R = CALIB_R_SEEK_DIR;

        // Extended limit is the configured range from switch zero.
        float lim_L = (dir_L > 0.0f) ? hm_limits_L.min_rad : hm_limits_L.max_rad;
        float lim_R = (dir_R > 0.0f) ? hm_limits_R.min_rad : hm_limits_R.max_rad;

        float dist_L = fabsf(hip_q_L - lim_L);
        float dist_R = fabsf(hip_q_R - lim_R);

        float ramp_dn = param_get(PARAM_JUMP_RAMP_DOWN_RAD);
        float margin  = param_get(PARAM_JUMP_HARDSTOP_MARGIN);
        float w_max   = param_get(PARAM_JUMP_OMEGA_MAX);
        float max_trq = param_get(PARAM_JUMP_TORQUE_MAX);
        float ext_kd  = param_get(PARAM_JUMP_EXTEND_KD);

        // Torque softstart
        float ramp_in = elapsed / param_get(PARAM_JUMP_RAMP_UP_S);
        if (ramp_in > 1.0f) ramp_in = 1.0f;

        // Torque ramp-to-zero near extended limit
        float ramp_out_L = dist_L / ramp_dn;
        if (ramp_out_L > 1.0f) ramp_out_L = 1.0f;
        float ramp_out_R = dist_R / ramp_dn;
        if (ramp_out_R > 1.0f) ramp_out_R = 1.0f;

        // Speed taper: torque → 0 as hip velocity approaches omega_max
        float tap_L = 1.0f - fabsf(hip_dq_L) / w_max;
        if (tap_L < 0.0f) tap_L = 0.0f;
        float tap_R = 1.0f - fabsf(hip_dq_R) / w_max;
        if (tap_R < 0.0f) tap_R = 0.0f;

        float tff_L = dir_L * max_trq * ramp_in * ramp_out_L * tap_L;
        float tff_R = dir_R * max_trq * ramp_in * ramp_out_R * tap_R;

        // Layer 4: hard cutoff — zero torque if within absolute margin of limit
        bool cutoff = false;
        if (dist_L < margin) { tff_L = 0.0f; cutoff = true; }
        if (dist_R < margin) { tff_R = 0.0f; cutoff = true; }

        // kp=0: pure torque + small damping; pos set to current so MITprotocol is well-formed
        hip_motors_set_setpoint_L(hip_q_L, 0.0f, 0.0f, ext_kd, tff_L);
        hip_motors_set_setpoint_R(hip_q_R, 0.0f, 0.0f, ext_kd, tff_R);

        bool near_ext = (dist_L < ramp_dn && dist_R < ramp_dn);
        bool timed_out = (elapsed >= param_get(PARAM_JUMP_EXTEND_TIMEOUT_S));
        if (near_ext || cutoff || timed_out) {
            s_jp_from_L   = hip_q_L;
            s_jp_from_R   = hip_q_R;
            s_jp_phase    = JP_RETRACT;
            s_jp_phase_ms = millis();
        }
    }
    else if (s_jp_phase == JP_RETRACT) {
        static constexpr float RETRACT_TIME_S = 0.20f;
        float alpha = elapsed / RETRACT_TIME_S;
        if (alpha > 1.0f) alpha = 1.0f;
        float dq_L = (s_jp_nom_L - s_jp_from_L) / RETRACT_TIME_S;
        float dq_R = (s_jp_nom_R - s_jp_from_R) / RETRACT_TIME_S;
        hip_motors_set_setpoint_L(s_jp_from_L + alpha * (s_jp_nom_L - s_jp_from_L), dq_L, kp, kd, 0.0f);
        hip_motors_set_setpoint_R(s_jp_from_R + alpha * (s_jp_nom_R - s_jp_from_R), dq_R, kp, kd, 0.0f);
        if (elapsed >= RETRACT_TIME_S) {
            s_jp_phase = JP_DONE;
        }
    }
    else {
        // JP_DONE: hold at Q_NOM with stiff position hold until jump_done() timer fires → RUNNING
        hip_motors_set_setpoint_L(s_jp_nom_L, 0.0f, kp, kd, 0.0f);
        hip_motors_set_setpoint_R(s_jp_nom_R, 0.0f, kp, kd, 0.0f);
    }
}
// Self-contained recovery sequence for arming while fallen — see standing_up.md.
// Never calls controlLoop_run(): pitch is large/outside the LQR's linearization
// region during the energetic phase, so this codes its own minimal wheel-torque
// law inline, the same way on_jumping() codes its own hip logic inline.
static void on_standing_up() {
    bool entering = (g_state.state != STATE_STANDING_UP);
    g_state.state = STATE_STANDING_UP;

    if (entering) {
        comm_log(LOG_LEVEL_INFO, "-> STANDING_UP");
        g_buzzer.play(STANDUP_MELODY, sizeof(STANDUP_MELODY) / sizeof(STANDUP_MELODY[0]));
        s_su_phase            = SU_CROUCH;
        s_su_phase_ms         = millis();
        s_su_attempt          = 0;
        s_su_captured         = false;
        s_su_capture_since_ms = 0;
        s_su_diverge_start_ms = 0;
        s_su_nom_L = hm_L.pos_rad;
        s_su_nom_R = hm_R.pos_rad;
        hip_cmd_to_setpoints(0.0f, &s_su_ret_L, &s_su_ret_R);  // t=0 → retracted
        wheel_motors_set_mode(WheelMode::TORQUE);
    }

    g_state.standup_state = (uint8_t)s_su_phase;

    // Refuse to command hips without valid calibrated limits — mirrors the
    // identical guard in on_jumping(). Without it, hip_cmd_to_setpoints(0.0f)
    // silently returns pos=0 (span==0), not "retracted", which would snap the
    // hip to an arbitrary pose at full stiffness. Unlike on_jumping(), there's
    // no LQR fallback to keep running here (pitch is outside its
    // linearization region), so an invalid-limits abort goes straight to
    // ESTOP rather than just skipping hip commands for the tick.
    if (!hm_limits_L.valid || !hm_limits_R.valid) {
        comm_log(LOG_LEVEL_WARN, "STANDING_UP: limits not valid — aborting");
        g_state.fault_code = FAULT_STANDUP_FAILED;
        stateMachine_request_estop();
        return;
    }

    float pitch      = g_state.pitch_rad;
    float pitch_rate = g_state.pitch_rate_rads;
    // Legs are pinned at the retracted target for the entire STANDING_UP
    // sequence, so the retracted-anchor balance trim applies throughout — no
    // need to gain-schedule by leg height the way control_loop.cpp does.
    float trim = param_get(PARAM_LQR_PITCH_TRIM_RET);

    // Divergence check (no retry): pitch grew past the recoverable range
    // mid-attempt — a wrong-direction/unstable response, not just a slow one.
    // Deliberately looser than the arm-time-only entry gate
    // (standup_pitch_fwd/bwd) so a saturated catch has overshoot budget
    // instead of self-tripping the instant it starts, and debounced
    // (STANDUP_DIVERGE_MS) like every other fault instead of instantaneous,
    // so one noisy IMU sample can't abort a good catch. Checked every phase,
    // including CROUCH, not just RECOVER/PAUSE.
    if (pitch > param_get(PARAM_STANDUP_DIVERGE_PITCH_FWD_RAD) ||
        pitch < -param_get(PARAM_STANDUP_DIVERGE_PITCH_BWD_RAD)) {
        if (s_su_diverge_start_ms == 0) s_su_diverge_start_ms = millis();
        if (millis() - s_su_diverge_start_ms > STANDUP_DIVERGE_MS) {
            g_state.fault_code = FAULT_STANDUP_FAILED;
            stateMachine_request_estop();
            return;
        }
    } else {
        s_su_diverge_start_ms = 0;
    }

    // Same hip gains RUNNING will use — not a separate standup_crouch_kp/kd —
    // so capture handoff to RUNNING (on_running(), from_standing_up) doesn't
    // also have to absorb a stiffness step on top of the hip position step.
    float kp      = param_get(PARAM_HIP_RUNNING_KP);
    float kd      = param_get(PARAM_HIP_RUNNING_KD);
    float elapsed = (millis() - s_su_phase_ms) / 1000.0f;

    if (s_su_phase == SU_CROUCH) {
        float crouch_t = param_get(PARAM_STANDUP_CROUCH_TIME_S);
        float alpha    = elapsed / crouch_t;
        if (alpha > 1.0f) alpha = 1.0f;
        float dq_L = (s_su_ret_L - s_su_nom_L) / crouch_t;
        float dq_R = (s_su_ret_R - s_su_nom_R) / crouch_t;
        hip_motors_set_setpoint_L(s_su_nom_L + alpha * (s_su_ret_L - s_su_nom_L), dq_L, kp, kd, 0.0f);
        hip_motors_set_setpoint_R(s_su_nom_R + alpha * (s_su_ret_R - s_su_nom_R), dq_R, kp, kd, 0.0f);
        wheel_motors_send(0.0f, 0.0f);  // no wheel motion until legs are known-good
        g_state.whl_tau_l = g_state.whl_tau_r = g_state.tau_sym = 0.0f;
        if (elapsed >= crouch_t) {
            s_su_phase    = SU_RECOVER;
            s_su_phase_ms = millis();
            s_su_attempt  = 1;
        }
        return;
    }

    // SU_RECOVER / SU_PAUSE: hold hips at the retracted target throughout
    hip_motors_set_setpoint_L(s_su_ret_L, 0.0f, kp, kd, 0.0f);
    hip_motors_set_setpoint_R(s_su_ret_R, 0.0f, kp, kd, 0.0f);

    if (s_su_phase == SU_RECOVER) {
        // Dedicated runaway hard-backup, independent of PARAM_WHEEL_VEL_LIMIT_TURNS_S
        // so tuning one doesn't loosen the other's safety margin.
        float hard_limit = param_get(PARAM_STANDUP_WHEEL_VEL_LIMIT_TURNS_S) * 2.0f;
        if (fabsf(wm_L.vel_turns_s) > hard_limit || fabsf(wm_R.vel_turns_s) > hard_limit) {
            g_state.fault_code = FAULT_WHEEL_RUNAWAY;
            stateMachine_request_estop();
            return;
        }

        // x0 mirrors the LQR's own pitch error term (control_loop.cpp:
        // pitch - theta_ref - pitch_trim, with theta_ref=0 here — standup
        // isn't tracking a velocity command) so the catch regulates to the
        // same equilibrium RUNNING will hold, instead of to pitch=0.
        float x0  = pitch - trim;
        float tau = param_get(PARAM_STANDUP_K_PITCH) * x0 + param_get(PARAM_STANDUP_K_RATE) * pitch_rate;
        float trq_limit = param_get(PARAM_STANDUP_TORQUE_LIMIT);
        if (tau >  trq_limit) tau =  trq_limit;
        if (tau < -trq_limit) tau = -trq_limit;
        if (tau >  MOTOR_TRQ_MAX) tau =  MOTOR_TRQ_MAX;
        if (tau < -MOTOR_TRQ_MAX) tau = -MOTOR_TRQ_MAX;
        wheel_motors_send(tau, tau);  // symmetric, no yaw/differential — straight push only
        g_state.whl_tau_l = g_state.whl_tau_r = g_state.tau_sym = tau;

        // Capture check: in-band relative to the same trim the catch is
        // regulating to (not pitch=0), continuously for the hold time =>
        // good enough for LQR. A trim-blind band centered on zero would
        // capture mid-roll, since x0=0 there requires sustained forward
        // acceleration rather than being an equilibrium.
        bool in_band = fabsf(x0)         < param_get(PARAM_STANDUP_CAPTURE_PITCH_RAD) &&
                        fabsf(pitch_rate) < param_get(PARAM_STANDUP_CAPTURE_RATE_RADS);
        if (in_band) {
            if (s_su_capture_since_ms == 0) s_su_capture_since_ms = millis();
            if (millis() - s_su_capture_since_ms >= (uint32_t)(param_get(PARAM_STANDUP_CAPTURE_HOLD_S) * 1000.0f)) {
                s_su_captured = true;
            }
        } else {
            s_su_capture_since_ms = 0;
        }

        // Attempt timeout: stayed within range but didn't converge in time
        if (elapsed >= param_get(PARAM_STANDUP_ATTEMPT_TIMEOUT_S) && !s_su_captured) {
            if (s_su_attempt >= (uint8_t)param_get(PARAM_STANDUP_MAX_RETRIES) + 1) {
                g_state.fault_code = FAULT_STANDUP_FAILED;
                stateMachine_request_estop();
                return;
            }
            s_su_phase    = SU_PAUSE;
            s_su_phase_ms = millis();
        }
    }
    else {  // SU_PAUSE
        wheel_motors_send(0.0f, 0.0f);
        g_state.whl_tau_l = g_state.whl_tau_r = g_state.tau_sym = 0.0f;
        if (elapsed >= param_get(PARAM_STANDUP_RETRY_PAUSE_S)) {
            s_su_attempt++;
            s_su_phase    = SU_RECOVER;
            s_su_phase_ms = millis();
        }
    }
}
static bool standup_captured() { return s_su_captured; }
// Immediate hip cutoff used by ESTOP. Normal exits use STATE_DISARMING.
static void estop_cut_hip() {
    if (param_get(PARAM_ESTOP_HIP_DISABLE) >= 0.5f) {
        hip_motors_exit_mit();
        s_estop_hip_disabled = true;
    }
}

static void on_estop() {
    bool entering     = (g_state.state != STATE_ESTOP);
    bool from_startup = (g_state.state == STATE_STARTUP);
    g_state.state = STATE_ESTOP;
    if (entering) {
        // Discard ALL queued requests so nothing stale fires after a future
        // reset/soft-clear — e.g. a CH6 jump latched the same tick a fault
        // fired would otherwise jump the instant the robot is re-armed.
        s_req_cmd_reject     = false;
        s_req_manual         = false;
        s_req_running        = false;
        s_req_calibration    = false;
        s_req_jump           = false;
        s_req_disarm_running = false;
        s_req_disarm_calibration = false;
        s_req_standby        = false;
        s_req_reset          = false;
        s_req_soft_clear     = false;
        // Startup failed (or was aborted by ESTOP) before it could resolve —
        // the specific reason (IMU/hip) was already logged by startup_fail()
        // just before this transition fired.
        if (from_startup) comm_log(LOG_LEVEL_ERROR, "Startup complete with errors");
        comm_log(LOG_LEVEL_ERROR, "-> ESTOP [fault 0x%02X]", g_state.fault_code);
        g_buzzer.play(ESTOP_MELODY, sizeof(ESTOP_MELODY) / sizeof(ESTOP_MELODY[0]));

        // Wheels: kill power immediately — no reason to keep driving a wheel
        // through an emergency stop.
        wheel_motors_set_mode(WheelMode::IDLE);

        // ESTOP is the immediate path, distinct from normal DISARMING. Clear
        // every active hip setpoint in this entry action and apply the
        // configured MIT-disable policy now; no torque ramp may delay the
        // emergency output invariant.
        s_hip_disarm_ramping = false;
        s_disarming_calibration = false;
        s_estop_hip_ramping = false;
        hip_motors_clear_setpoints();
        estop_cut_hip();
    }
}

// ── Transition conditions ─────────────────────────────────────────────────────

static bool startup_ok() {
    bool imu_ok = (param_get(PARAM_IMU_ENABLE) < 0.5f) || (imu_state() == ImuState::NOMINAL);
    return imu_ok && hip_motors_ok() && wheel_motors_ok();
}
static bool startup_fail() {
    if (param_get(PARAM_IMU_ENABLE) >= 0.5f && imu_state() == ImuState::ERROR) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: IMU init failed");
        g_state.fault_code = FAULT_IMU_ERROR;
        return true;
    }
    bool l_missing = (param_get(PARAM_HIP_L_ENABLE) >= 0.5f) && !hm_L.ever_heard;
    bool r_missing = (param_get(PARAM_HIP_R_ENABLE) >= 0.5f) && !hm_R.ever_heard;
    if (millis() > 2000 && (l_missing || r_missing)) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: hip init timeout (L=%d R=%d)", (int)hm_L.ever_heard, (int)hm_R.ever_heard);
        g_state.fault_code = FAULT_HIP_INIT_TIMEOUT;
        return true;
    }
    bool wl_missing = (param_get(PARAM_WHEEL_L_ENABLE) >= 0.5f) && !wm_L.ever_heard;
    bool wr_missing = (param_get(PARAM_WHEEL_R_ENABLE) >= 0.5f) && !wm_R.ever_heard;
    if (millis() > 2000 && (wl_missing || wr_missing)) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: wheel init timeout (L=%d R=%d)", (int)wm_L.ever_heard, (int)wm_R.ever_heard);
        g_state.fault_code = FAULT_WHEEL_INIT_TIMEOUT;
        return true;
    }
    return false;
}
// Checked from nearly every state (see stateMachine_init()) — a motor feedback
// dropout is a global ESTOP trigger, not just a RUNNING-time concern.
static bool motor_feedback_fault() {
    if (!hip_motors_ok()) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: hip feedback lost");
        g_state.fault_code = FAULT_HIP_FEEDBACK_LOST;
        return true;
    }
    if (!wheel_motors_ok()) {
        comm_log(LOG_LEVEL_ERROR, "FAULT: wheel feedback lost L(ok=%d e=%lu) R(ok=%d e=%lu)",
                 (int)wm_L.ok, (unsigned long)wm_L.error, (int)wm_R.ok, (unsigned long)wm_R.error);
        g_state.fault_code = FAULT_WHEEL_FEEDBACK_LOST;
        return true;
    }
    return false;
}
// All energetic states, including DISARMING: a frozen pitch estimate is unsafe,
// so ESTOP the moment the required IMU leaves NOMINAL.
static bool running_imu_fault() {
    if (param_get(PARAM_IMU_ENABLE) < 0.5f) return false;
    if (imu_state() == ImuState::NOMINAL) return false;
    comm_log(LOG_LEVEL_ERROR, "FAULT: IMU lost (imu_state=%d)", (int)imu_state());
    g_state.fault_code = FAULT_IMU_LOST;
    return true;
}
static bool req_manual()        { bool v = s_req_manual;  s_req_manual  = false; return v; }
static bool req_standby()       { bool v = s_req_standby; s_req_standby = false; return v; }
static bool req_reset()         { bool v = s_req_reset;   s_req_reset   = false; return v; }
static bool req_calibration() {
    bool v = s_req_calibration;
    s_req_calibration = false;
    return v;
}
static bool req_cmd_reject() { bool v = s_req_cmd_reject; s_req_cmd_reject = false; return v; }
static bool cmd_reject_done() { return (millis() >= s_cmd_reject_deadline_ms); }
static bool manual_gui_timeout() {
    if (millis() - s_last_gui_packet_ms < MANUAL_GUI_TIMEOUT_MS) return false;
    comm_log(LOG_LEVEL_WARN, "MANUAL: GUI timeout -> STANDBY");
    return true;
}
static bool req_running_checks_common() {
    if (param_get(PARAM_IMU_ENABLE) < 0.5f) {
        comm_log(LOG_LEVEL_WARN, "Running mode denied: IMU disabled (imu_enable=0)");
        stateMachine_request_cmd_reject();
        return false;
    }
    if (!hm_limits_L.valid || !hm_limits_R.valid) {
        if (param_get(PARAM_CALIB_BYPASS_EN) < 0.5f) {
            comm_log(LOG_LEVEL_WARN, "Running mode denied: calibrate first (limits not valid).");
            stateMachine_request_cmd_reject();
            return false;
        }
        comm_log(LOG_LEVEL_WARN, "Running mode ARMED without calibration (calib_bypass_en=1) — hip limits unenforced.");
    }
    bool bypass_hip_check   = param_get(PARAM_CALIB_BYPASS_EN)         >= 0.5f;
    bool bypass_wheel_check = param_get(PARAM_RUNNING_WHEEL_BYPASS_EN) >= 0.5f;
    if ((!bypass_hip_check &&
         (param_get(PARAM_HIP_L_ENABLE) < 0.5f || param_get(PARAM_HIP_R_ENABLE) < 0.5f)) ||
        (!bypass_wheel_check &&
         (param_get(PARAM_WHEEL_L_ENABLE) < 0.5f || param_get(PARAM_WHEEL_R_ENABLE) < 0.5f))) {
        comm_log(LOG_LEVEL_WARN, "Running mode denied: bench-test mode (a motor is disabled)");
        stateMachine_request_cmd_reject();
        return false;
    }
    return true;
}
// Arm while STANDUP_ENABLE=1: gate on recoverable pitch range, then route
// through STANDING_UP instead of straight to RUNNING.
static bool req_running_to_standing_up() {
    if (!s_req_running) return false;
    if (param_get(PARAM_STANDUP_ENABLE) < 0.5f) return false;  // let _direct handle it
    s_req_running = false;
    return true;
}
// Today's exact behavior, byte-identical, when STANDUP_ENABLE=0 (the default).
static bool req_running_direct() {
    if (!s_req_running) return false;
    if (param_get(PARAM_STANDUP_ENABLE) >= 0.5f) return false;  // handled above
    s_req_running = false;
    return true;
}
static bool req_disarm_running() { bool v = s_req_disarm_running; s_req_disarm_running = false; return v; }
static bool req_disarm_calibration() {
    bool v = s_req_disarm_calibration;
    s_req_disarm_calibration = false;
    return v;
}
static bool disarm_done()        { return s_disarm_done; }
static bool req_jump()           { bool v = s_req_jump;           s_req_jump           = false; return v; }
static bool jump_done()          { return (millis() >= s_jump_deadline_ms); }
static bool req_estop() {
    if (!s_req_estop) return false;
    s_req_estop = false;
    if (g_state.fault_code == FAULT_NONE) g_state.fault_code = FAULT_HUMAN_ESTOP;
    return true;
}
static bool req_soft_clear() {
    if (!s_req_soft_clear) return false;
    s_req_soft_clear = false;
    if (fault_severity(g_state.fault_code) != FAULT_SEVERITY_SOFT) {
        comm_log(LOG_LEVEL_WARN, "Soft clear denied: fault 0x%02X is not SOFT severity", g_state.fault_code);
        return false;
    }
    comm_log(LOG_LEVEL_INFO, "Soft clear ESTOP [0x%02X] -> STANDBY", g_state.fault_code);
    return true;
}

static bool calibration_done_fn() {
    return calibration_done();
}
static bool calibration_failed_fn() {
    if (!calibration_failed()) return false;
    g_state.fault_code = FAULT_CALIBRATION_TIMEOUT;
    return true;
}

// ── Init / update ─────────────────────────────────────────────────────────────

void stateMachine_init() {
    S_STARTUP     = sm.addState(on_startup);
    S_STANDBY     = sm.addState(on_standby);
    S_MANUAL      = sm.addState(on_manual);
    S_CALIBRATION = sm.addState(on_calibration);
    S_RUNNING     = sm.addState(on_running);
    S_ESTOP       = sm.addState(on_estop);
    S_CMD_REJECT  = sm.addState(on_cmd_reject);
    S_JUMPING     = sm.addState(on_jumping);
    S_STANDING_UP = sm.addState(on_standing_up);
    S_DISARMING   = sm.addState(on_disarming);

    S_STARTUP->addTransition(req_estop,    S_ESTOP);
    S_STARTUP->addTransition(startup_ok,   S_STANDBY);
    S_STARTUP->addTransition(startup_fail, S_ESTOP);

    S_STANDBY->addTransition(req_estop,            S_ESTOP);
    S_STANDBY->addTransition(motor_feedback_fault, S_ESTOP);
    S_STANDBY->addTransition(req_manual,           S_MANUAL);
    S_STANDBY->addTransition(req_calibration,      S_CALIBRATION);
    S_STANDBY->addTransition(req_running_to_standing_up, S_STANDING_UP);
    S_STANDBY->addTransition(req_running_direct,   S_RUNNING);
    S_STANDBY->addTransition(req_cmd_reject,       S_CMD_REJECT);

    S_MANUAL ->addTransition(req_estop,            S_ESTOP);
    S_MANUAL ->addTransition(motor_feedback_fault, S_ESTOP);
    S_MANUAL ->addTransition(req_standby,          S_STANDBY);
    S_MANUAL ->addTransition(manual_gui_timeout,   S_STANDBY);

    S_CALIBRATION->addTransition(req_estop,             S_ESTOP);
    S_CALIBRATION->addTransition(motor_feedback_fault,  S_ESTOP);
    S_CALIBRATION->addTransition(calibration_failed_fn, S_ESTOP);
    S_CALIBRATION->addTransition(calibration_done_fn,   S_STANDBY);
    S_CALIBRATION->addTransition(req_disarm_calibration, S_DISARMING);
    S_CALIBRATION->addTransition(req_standby,           S_STANDBY);

    S_RUNNING->addTransition(req_estop,            S_ESTOP);
    S_RUNNING->addTransition(motor_feedback_fault, S_ESTOP);
    S_RUNNING->addTransition(running_imu_fault,    S_ESTOP);
    S_RUNNING->addTransition(req_disarm_running,   S_DISARMING);
    S_RUNNING->addTransition(req_jump,             S_JUMPING);

    S_JUMPING->addTransition(req_estop,            S_ESTOP);
    S_JUMPING->addTransition(motor_feedback_fault, S_ESTOP);
    S_JUMPING->addTransition(running_imu_fault,    S_ESTOP);
    S_JUMPING->addTransition(req_disarm_running,   S_DISARMING);
    S_JUMPING->addTransition(jump_done,            S_RUNNING);

    S_STANDING_UP->addTransition(req_estop,            S_ESTOP);
    S_STANDING_UP->addTransition(motor_feedback_fault, S_ESTOP);
    S_STANDING_UP->addTransition(running_imu_fault,    S_ESTOP);
    S_STANDING_UP->addTransition(req_disarm_running,   S_DISARMING);
    S_STANDING_UP->addTransition(standup_captured,     S_RUNNING);

    S_DISARMING->addTransition(req_estop,            S_ESTOP);
    S_DISARMING->addTransition(motor_feedback_fault, S_ESTOP);
    S_DISARMING->addTransition(running_imu_fault,    S_ESTOP);
    S_DISARMING->addTransition(disarm_done,          S_STANDBY);

    S_ESTOP  ->addTransition(req_soft_clear,     S_STANDBY);
    S_ESTOP  ->addTransition(req_reset,          S_STARTUP);

    // req_estop/motor_feedback_fault checked first, same as every other state —
    // without these, an ESTOP request during the ~1 s reject transient would
    // sit latched and only take effect once cmd_reject_done() fires.
    S_CMD_REJECT->addTransition(req_estop,            S_ESTOP);
    S_CMD_REJECT->addTransition(motor_feedback_fault, S_ESTOP);
    S_CMD_REJECT->addTransition(cmd_reject_done,      S_STANDBY);

    g_state.state = STATE_STARTUP;
}

void stateMachine_update() {
    sm.run();
}

// ── Public request API ────────────────────────────────────────────────────────

// stateMachine_request_manual/_calibration/_running only latch while actually
// in STANDBY. Their matching req_X() transition conditions (which clear the
// flag) are only ever evaluated from S_STANDBY, so a stale/duplicate request
// arriving while already past STANDBY (e.g. the GUI's dual WiFi+USB send
// delivering one click twice) would otherwise sit latched indefinitely and
// immediately re-fire the instant we return to STANDBY for any reason.
bool stateMachine_request_manual() {
    if (g_state.state != STATE_STANDBY) return false;
    s_req_manual = true;
    return true;
}
// req_standby() is only evaluated from S_MANUAL/S_CALIBRATION — guard so a
// stray "Standby" request sent from some other state (e.g. RUNNING, where
// disarm_running is the correct path) doesn't latch and fire the moment we
// next happen to enter MANUAL or CALIBRATION.
bool stateMachine_exit_manual() {
    if (g_state.state != STATE_MANUAL && g_state.state != STATE_CALIBRATION) return false;
    s_req_standby = true;
    return true;
}
// req_reset() is only evaluated from S_ESTOP — guard so a stray reset request
// doesn't sit latched and silently clear a later, unrelated ESTOP fault.
bool stateMachine_request_reset() {
    if (g_state.state != STATE_ESTOP) return false;
    s_req_reset = true;
    return true;
}
bool stateMachine_request_calibration() {
    if (g_state.state != STATE_STANDBY) return false;
    if (param_get(PARAM_HIP_L_ENABLE) < 0.5f && param_get(PARAM_HIP_R_ENABLE) < 0.5f) {
        comm_log(LOG_LEVEL_WARN, "Calibration denied: no hip motors enabled (hip_l_enable=0, hip_r_enable=0)");
        stateMachine_request_cmd_reject();
        return false;
    }
    s_req_calibration = true;
    return true;
}
bool stateMachine_request_running() {
    if (g_state.state != STATE_STANDBY) return false;
    if (imu_state() != ImuState::NOMINAL || millis() - imu_last_update_ms() > 50) {
        comm_log(LOG_LEVEL_WARN, "Running mode denied: IMU not nominal/fresh (state=%d age=%lu ms)",
                 (int)imu_state(), (unsigned long)(millis() - imu_last_update_ms()));
        stateMachine_request_cmd_reject();
        return false;
    }
    if (!req_running_checks_common()) return false;
    if (param_get(PARAM_STANDUP_ENABLE) >= 0.5f) {
        float pitch = g_state.pitch_rad;
        if (pitch > param_get(PARAM_STANDUP_MAX_PITCH_FWD_RAD) ||
            pitch < -param_get(PARAM_STANDUP_MAX_PITCH_BWD_RAD)) {
            comm_log(LOG_LEVEL_WARN, "Standing-up denied: pitch %.3f rad outside recoverable range", pitch);
            stateMachine_request_cmd_reject();
            return false;
        }
    }
    s_req_running = true;
    return true;
}
// req_disarm_running() is only evaluated from S_RUNNING — guard so a stray
// disarm request doesn't sit latched and fire the instant we next arm,
// disarming immediately after an operator arms RUNNING.
bool stateMachine_disarm_running() {
    if (g_state.state != STATE_RUNNING && g_state.state != STATE_JUMPING &&
        g_state.state != STATE_STANDING_UP) return false;
    s_req_disarm_running = true;
    return true;
}
bool stateMachine_disarm_calibration() {
    if (g_state.state != STATE_CALIBRATION) return false;
    s_req_disarm_calibration = true;
    return true;
}
// Deliberately unguarded: ESTOP must be reachable from any state (it already
// is, from every state's transition list) and re-requesting it is harmless —
// idempotent, and there is no origin state where a "stray" estop is wrong.
bool stateMachine_request_estop() {
    // Idempotent in ESTOP, and crucially do not leave a stale event that can
    // fire immediately after RESET enters STARTUP.
    if (g_state.state == STATE_ESTOP) return true;
    s_req_estop = true;
    return true;
}
// Only ever called synchronously from within req_calibration()/req_running(),
// which themselves only run while currentState == S_STANDBY — no external
// caller, so no stale-latch risk here.
void stateMachine_request_cmd_reject()  { s_req_cmd_reject     = true; }
// req_soft_clear() is only evaluated from S_ESTOP — guard so a stray soft
// clear doesn't sit latched and silently clear a later, unrelated SOFT fault.
bool stateMachine_request_soft_clear() {
    if (g_state.state != STATE_ESTOP) return false;
    if (fault_severity(g_state.fault_code) != FAULT_SEVERITY_SOFT) {
        comm_log(LOG_LEVEL_WARN, "Soft clear denied: fault 0x%02X is not SOFT severity", g_state.fault_code);
        return false;
    }
    s_req_soft_clear = true;
    return true;
}
// req_jump() is only evaluated from S_RUNNING — guard so a stray jump request
// doesn't sit latched and fire an unintended jump the instant we next arm.
// jump_enable is the master gate: with it 0, don't even enter STATE_JUMPING —
// on_jumping() would skip hip commands, leaving the hips uncommanded for 3 s.
bool stateMachine_request_jump() {
    if (param_get(PARAM_JUMP_ENABLE) < 0.5f) {
        comm_log(LOG_LEVEL_WARN, "JUMP: blocked — jump_enable=0");
        return false;
    }
    if (g_state.state != STATE_RUNNING) return false;
    s_req_jump = true;
    return true;
}
void stateMachine_ping_gui_watchdog()   { s_last_gui_packet_ms = millis(); }
uint32_t stateMachine_ms_since_gui_packet() { return millis() - s_last_gui_packet_ms; }
bool stateMachine_is_estop_hip_ramping() { return s_estop_hip_ramping; }
