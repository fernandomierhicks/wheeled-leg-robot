#include "calibration.h"
#include <Arduino.h>
#include <math.h>
#include "config.h"
#include "hip_motors.h"
#include "robot_state.h"
#include "Buzzer.h"
#include "comm_protocol.h"
#include "param_registry.h"

extern Buzzer g_buzzer;

enum CalibAxisState : uint8_t {
    CAL_SEEK_BOTTOM,
    CAL_SEEK_TOP,
    CAL_RETURN_HOME,
    // Reached home; kp/kd ramping from hold values down to zero before
    // CAL_DONE. calibration_done() stays false throughout so STATE_CALIBRATION
    // doesn't exit to STANDBY (which drops the setpoint entirely) until torque
    // is already at zero — see PARAM_CALIB_RAMPDOWN_TIME_S.
    CAL_RAMPDOWN,
    CAL_DONE,
    CAL_FAULT,
    // Retract found and zeroed, but PARAM_CALIB_EXTEND_ENABLE=0 — hold here
    // indefinitely. Deliberately NOT CAL_DONE: no limits were computed, and
    // calibration_done() must stay false so STATE_CALIBRATION doesn't
    // auto-exit to STANDBY. Operator exits manually once satisfied.
    CAL_HOLD_RETRACT,
};

struct CalibAxis {
    CalibAxisState state;
    float          ramp_target;
    float          prev_pos;
    uint8_t        stall_count;
    float          seek_dir;     // sign toward the "bottom" hardstop
    float          home_target;  // midpoint of the computed limits
    float          start_pos;    // position when calibration_start() was called
    uint32_t       rampdown_start_ms;  // millis() when CAL_RAMPDOWN began
    float          kp_out;        // slew-limited kp actually being sent (SEEK_*/RETURN_HOME)
    float          kp0_rampdown;  // kp CAL_RAMPDOWN ramps down from — whatever kp_out was
    float          kd0_rampdown;  // at the instant home was reached, not a fixed hold_kp/kd
};

static CalibAxis ax_L, ax_R;
static bool s_done_announced  = false;
static bool s_fault_announced = false;

// Slew-rate-limits kp/kd toward their per-phase target instead of jumping
// instantly at SEEK_BOTTOM<->SEEK_TOP<->RETURN_HOME boundaries (each phase
// uses a different kp for its own stall-detection tuning) — otherwise
// commanding a large position error at a suddenly-different stiffness is a
// felt torque jerk. Reuses PARAM_CALIB_RAMPDOWN_TIME_S as the "how gentle"
// knob so there's one setting for calibration smoothness; 0 = instant
// (old behavior). Scaled off calib_kp_top since it's the largest constant
// in play, so a full 0->top swing takes ramp_s and smaller swings scale down.
static float slew_toward(float current, float target, float max_step) {
    float d = target - current;
    if (d >  max_step) d =  max_step;
    if (d < -max_step) d = -max_step;
    return current + d;
}
static float kp_slew_step() {
    float ramp_s = param_get(PARAM_CALIB_RAMPDOWN_TIME_S);
    if (ramp_s <= 0.0f) return 1.0e9f;  // instant
    return param_get(PARAM_CALIB_KP_TOP) / (ramp_s * CONTROL_HZ);
}

// ── Buzzer cues ──────────────────────────────────────────────────────────────

static const BuzzerNote START_CHIME[] = {
    {69, 80, 20},  // A4
    {76, 80,  0},  // E5
};
static const BuzzerNote DONE_MELODY[] = {
    {72,  80, 20}, // C5
    {76,  80, 20}, // E5
    {79, 120,  0}, // G5
};
static const BuzzerNote FAULT_MELODY[] = {
    {60, 150, 50}, // C4
    {55, 200,  0}, // G3
};

// ── Per-axis update ──────────────────────────────────────────────────────────

static void update_axis(CalibAxis& ax, HipAxisState& hm, HipLimits& lim,
                         void (*send)(float, float, float, float, float),
                         void (*zero)(), const char* tag, uint8_t axis_id) {
    const float dt = 1.0f / CONTROL_HZ;

    switch (ax.state) {
        case CAL_SEEK_BOTTOM:
        case CAL_SEEK_TOP: {
            float seek_speed   = param_get(PARAM_CALIB_SEEK_SPEED);
            float kd           = param_get(PARAM_CALIB_KD);
            float stall_db     = param_get(PARAM_CALIB_STALL_DEADBAND);
            // SEEK_BOTTOM (retract) is weight-assisted: less kp, more sensitive
            // (lower) stall threshold. SEEK_TOP (extend) fights robot weight:
            // more kp, less sensitive (higher) stall threshold.
            float kp           = (ax.state == CAL_SEEK_BOTTOM)
                                      ? param_get(PARAM_CALIB_KP_BOTTOM)
                                      : param_get(PARAM_CALIB_KP_TOP);
            float stall_cur    = (ax.state == CAL_SEEK_BOTTOM)
                                      ? param_get(PARAM_CALIB_STALL_CUR_BOTTOM)
                                      : param_get(PARAM_CALIB_STALL_CUR_TOP);
            float safety_bound = (ax.state == CAL_SEEK_BOTTOM)
                                      ? param_get(PARAM_CALIB_SAFETY_BOUND)
                                      : param_get(PARAM_CALIB_SAFETY_BOUND_TOP);
            int   stall_ticks  = (int)param_get(PARAM_CALIB_STALL_TICKS);
            float margin       = param_get(PARAM_CALIB_MARGIN);

            float dir = (ax.state == CAL_SEEK_BOTTOM) ? ax.seek_dir : -ax.seek_dir;
            ax.ramp_target += dir * seek_speed * dt;
            ax.kp_out = slew_toward(ax.kp_out, kp, kp_slew_step());
            send(ax.ramp_target, 0.0f, ax.kp_out, kd, 0.0f);

            bool stalled = fabsf(hm.pos_rad - ax.prev_pos) < stall_db &&
                           fabsf(hm.current_A) > stall_cur;
            ax.stall_count = stalled ? ax.stall_count + 1 : 0;
            ax.prev_pos    = hm.pos_rad;

            if (fabsf(ax.ramp_target - ax.start_pos) > safety_bound) {
                ax.state = CAL_FAULT;
                // Split into fully-spelled-out lines for readability
                // (comm_log messages are capped at 119 chars).
                comm_log(LOG_LEVEL_ERROR, "%s: FAULT - no hardstop found within %.1f rad", tag, safety_bound);
                comm_log(LOG_LEVEL_ERROR, "%s: current %.2f amps, threshold %.2f amps", tag, hm.current_A, stall_cur);
                comm_send_calib_event(axis_id, CALIB_EVENT_FAULT, ax.ramp_target, 0, 0);
                return;
            }

            if (ax.stall_count < stall_ticks) return;

            // Snapshot the trigger-instant diagnostics before zero()/state resets
            // below overwrite ramp_target/start_pos — these are what to look at
            // when tuning if a hardstop is declared too early:
            //   trig_err = commanded-vs-actual position error (ramp_target - pos_rad)
            //              at the instant of trigger — the open-loop ramp keeps
            //              advancing regardless of actual motion, so this is the
            //              gap that current is reacting to via kp.
            //   trig_d   = total distance traveled this phase (ramp_target - start_pos)
            //              — compare against the expected ~90 deg range; a hardstop
            //              declared after only a few degrees of travel is suspicious.
            float trig_err = ax.ramp_target - hm.pos_rad;
            float trig_d   = ax.ramp_target - ax.start_pos;
            float trig_cur = hm.current_A;

            if (ax.state == CAL_SEEK_BOTTOM) {
                zero();
                // Split across several fully-spelled-out lines for readability
                // (comm_log messages are capped at 119 chars).
                comm_log(LOG_LEVEL_INFO, "%s: bottom hardstop found and zeroed", tag);
                comm_log(LOG_LEVEL_INFO, "%s: current %.2f amps, threshold %.2f amps", tag, trig_cur, stall_cur);
                comm_log(LOG_LEVEL_INFO, "%s: position error %.2f rad, distance traveled %.2f rad", tag, trig_err, trig_d);
                comm_log(LOG_LEVEL_INFO, "%s: stall ticks required %d", tag, stall_ticks);
                comm_send_calib_event(axis_id, CALIB_EVENT_BOTTOM_FOUND, hm.pos_rad, 0, 0);
                g_buzzer.midi(84, 200, 120);  // C6
                ax.ramp_target = 0.0f;
                ax.prev_pos    = 0.0f;
                ax.stall_count = 0;
                // zero() just reset the AK45's own position reference to 0 here,
                // and ramp_target restarts at 0 to match — start_pos must follow
                // to the same new frame, or the SEEK_TOP safety-bound check below
                // would compare a fresh relative ramp_target against a stale
                // pre-zero absolute reading instead of "distance traveled this phase".
                ax.start_pos   = 0.0f;
                // The setpoint cache still holds the pre-zero-frame target from
                // the send() above; hip_motors_poll() would re-send it once in
                // the NEW frame (a large position error into the hardstop)
                // before the next update — refresh it to the new frame now.
                send(ax.ramp_target, 0.0f, kp, kd, 0.0f);
                if (param_get(PARAM_CALIB_EXTEND_ENABLE) >= 0.5f) {
                    ax.state = CAL_SEEK_TOP;
                } else {
                    ax.state = CAL_HOLD_RETRACT;
                    comm_log(LOG_LEVEL_WARN, "%s: extend disabled, holding at retract", tag);
                }
            } else {
                float range = hm.pos_rad;  // signed range from zero
                lim.min_rad = fminf(0.0f, range) + margin;
                lim.max_rad = fmaxf(0.0f, range) - margin;
                lim.valid   = true;
                comm_log(LOG_LEVEL_INFO, "%s: top hardstop found", tag);
                comm_log(LOG_LEVEL_INFO, "%s: current %.2f amps, threshold %.2f amps", tag, trig_cur, stall_cur);
                comm_log(LOG_LEVEL_INFO, "%s: position error %.2f rad, distance traveled %.2f rad", tag, trig_err, trig_d);
                comm_log(LOG_LEVEL_INFO, "%s: stall ticks required %d", tag, stall_ticks);
                comm_log(LOG_LEVEL_INFO, "%s: limits min %.3f rad, max %.3f rad", tag, lim.min_rad, lim.max_rad);
                comm_send_calib_event(axis_id, CALIB_EVENT_LIMITS, range, lim.min_rad, lim.max_rad);
                g_buzzer.midi(88, 200, 120);  // E6
                ax.home_target = 0.5f * (lim.min_rad + lim.max_rad);
                ax.state       = CAL_RETURN_HOME;
                ax.stall_count = 0;
            }
            break;
        }

        case CAL_RETURN_HOME: {
            float seek_speed = param_get(PARAM_CALIB_SEEK_SPEED);
            float kd         = param_get(PARAM_CALIB_KD);
            ax.kp_out = slew_toward(ax.kp_out, param_get(PARAM_CALIB_KP_BOTTOM), kp_slew_step());
            float step  = seek_speed * dt;
            float error = ax.home_target - ax.ramp_target;
            if (fabsf(error) <= step) {
                ax.ramp_target = ax.home_target;
                send(ax.ramp_target, 0.0f, ax.kp_out, kd, 0.0f);
                ax.state             = CAL_RAMPDOWN;
                ax.rampdown_start_ms = millis();
                // Continue smoothly from whatever kp/kd was actually in effect
                // this instant — not a fixed hold_kp/hold_kd, which would just
                // move the jump here instead of removing it.
                ax.kp0_rampdown      = ax.kp_out;
                ax.kd0_rampdown      = kd;
                comm_log(LOG_LEVEL_INFO, "%s: done, holding @ %.3f rad", tag, ax.ramp_target);
                comm_send_calib_event(axis_id, CALIB_EVENT_DONE, ax.ramp_target, lim.min_rad, lim.max_rad);
            } else {
                ax.ramp_target += (error > 0.0f) ? step : -step;
                send(ax.ramp_target, 0.0f, ax.kp_out, kd, 0.0f);
            }
            break;
        }

        // Hold position fixed at home; ramp kp/kd from whatever was in effect
        // when home was reached down to zero, so the eventual setpoint-clear
        // on entering STANDBY is a no-op torque-wise.
        case CAL_RAMPDOWN: {
            float ramp_s  = param_get(PARAM_CALIB_RAMPDOWN_TIME_S);
            float elapsed = (millis() - ax.rampdown_start_ms) / 1000.0f;
            float alpha   = (ramp_s > 0.0f) ? (1.0f - elapsed / ramp_s) : 0.0f;
            if (alpha < 0.0f) alpha = 0.0f;
            send(ax.ramp_target, 0.0f, alpha * ax.kp0_rampdown, alpha * ax.kd0_rampdown, 0.0f);
            if (alpha <= 0.0f) {
                ax.state = CAL_DONE;
                comm_log(LOG_LEVEL_INFO, "%s: torque ramped to zero", tag);
            }
            break;
        }

        case CAL_DONE:
            // Reached only once CAL_RAMPDOWN has already brought torque to
            // zero — send(0 kp/kd) here is a no-op vs. hip_motors_clear_setpoints()
            // on the STANDBY transition that follows immediately after.
            send(ax.ramp_target, 0.0f, 0.0f, 0.0f, 0.0f);
            break;

        // Retract-only bench test: hold at the zeroed retract hardstop indefinitely.
        // No stall detection, no safety-bound check, no further state transition —
        // deliberately inert until the operator exits calibration manually.
        case CAL_HOLD_RETRACT:
            send(ax.ramp_target, 0.0f, param_get(PARAM_CALIB_HOLD_KP), param_get(PARAM_CALIB_HOLD_KD), 0.0f);
            break;

        case CAL_FAULT:
            break;
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

void calibration_start() {
    bool l_en      = param_get(PARAM_HIP_L_ENABLE) >= 0.5f;
    bool r_en      = param_get(PARAM_HIP_R_ENABLE) >= 0.5f;
    bool retract_en = param_get(PARAM_CALIB_RETRACT_ENABLE) >= 0.5f;
    ax_L = {CAL_SEEK_BOTTOM, hm_L.pos_rad, hm_L.pos_rad, 0, param_get(PARAM_CALIB_L_SEEK_DIR), 0.0f, hm_L.pos_rad};
    ax_R = {CAL_SEEK_BOTTOM, hm_R.pos_rad, hm_R.pos_rad, 0, param_get(PARAM_CALIB_R_SEEK_DIR), 0.0f, hm_R.pos_rad};
    // A disabled/absent motor can't be seeked — mark its axis done immediately
    // rather than let it ramp until it trips the safety-bound fault.
    if (!l_en) {
        ax_L.state = CAL_DONE;
        comm_log(LOG_LEVEL_WARN, "Calib: L skipped (hip_l_enable=0)");
    } else if (!retract_en) {
        // SEEK_BOTTOM is the prerequisite phase (establishes the zero
        // reference) — without it there's nothing meaningful to seek.
        ax_L.state = CAL_DONE;
        comm_log(LOG_LEVEL_WARN, "Calib: L skipped (calib_retract_en=0)");
    }
    if (!r_en) {
        ax_R.state = CAL_DONE;
        comm_log(LOG_LEVEL_WARN, "Calib: R skipped (hip_r_enable=0)");
    } else if (!retract_en) {
        ax_R.state = CAL_DONE;
        comm_log(LOG_LEVEL_WARN, "Calib: R skipped (calib_retract_en=0)");
    }
    hm_limits_L.valid = false;
    hm_limits_R.valid = false;
    s_done_announced  = false;
    s_fault_announced = false;
    g_buzzer.play(START_CHIME, sizeof(START_CHIME) / sizeof(START_CHIME[0]), 200);
    comm_log(LOG_LEVEL_INFO, "Calib: starting hardstop search");
    comm_send_calib_event(HIP_MOTOR_BOTH, CALIB_EVENT_START, 0, 0, 0);
}

void calibration_update() {
    // Commands go through the setpoint cache (not a direct CAN send):
    // hip_motors_poll() then transmits exactly ONE MIT frame per motor per
    // tick — previously calibration sent directly AND poll pinged zero-torque,
    // chattering effective stiffness at 2 frames/tick during stall detection.
    if (param_get(PARAM_HIP_L_ENABLE) >= 0.5f)
        update_axis(ax_L, hm_L, hm_limits_L, hip_motors_set_setpoint_L, hip_motor_zero_L, "L", HIP_MOTOR_L);
    if (param_get(PARAM_HIP_R_ENABLE) >= 0.5f)
        update_axis(ax_R, hm_R, hm_limits_R, hip_motors_set_setpoint_R, hip_motor_zero_R, "R", HIP_MOTOR_R);

    if (calibration_done() && !s_done_announced) {
        s_done_announced = true;
        param_force_set(PARAM_CALIB_DONE, 1.0f);
        g_buzzer.play(DONE_MELODY, sizeof(DONE_MELODY) / sizeof(DONE_MELODY[0]), 200);
    }
    if (calibration_failed() && !s_fault_announced) {
        s_fault_announced = true;
        g_buzzer.play(FAULT_MELODY, sizeof(FAULT_MELODY) / sizeof(FAULT_MELODY[0]), 200);
    }
}

bool calibration_done() {
    bool l_done = (param_get(PARAM_HIP_L_ENABLE) < 0.5f) || (ax_L.state == CAL_DONE);
    bool r_done = (param_get(PARAM_HIP_R_ENABLE) < 0.5f) || (ax_R.state == CAL_DONE);
    return l_done && r_done;
}

bool calibration_failed() {
    bool l_bad = (param_get(PARAM_HIP_L_ENABLE) >= 0.5f) && (ax_L.state == CAL_FAULT);
    bool r_bad = (param_get(PARAM_HIP_R_ENABLE) >= 0.5f) && (ax_R.state == CAL_FAULT);
    return l_bad || r_bad;
}

void calibration_abort() {
    ax_L.state        = CAL_SEEK_BOTTOM;
    ax_R.state        = CAL_SEEK_BOTTOM;
    s_done_announced  = false;
    s_fault_announced = false;
    comm_log(LOG_LEVEL_INFO, "Calib: aborted");
}
