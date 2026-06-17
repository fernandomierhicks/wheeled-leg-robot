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
    CAL_DONE,
    CAL_FAULT,
};

struct CalibAxis {
    CalibAxisState state;
    float          ramp_target;
    float          prev_pos;
    uint8_t        stall_count;
    float          seek_dir;     // sign toward the "bottom" hardstop
    float          home_target;  // midpoint of the computed limits
    float          start_pos;    // position when calibration_start() was called
};

static CalibAxis ax_L, ax_R;
static bool s_done_announced  = false;
static bool s_fault_announced = false;

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
            float kp           = param_get(PARAM_CALIB_KP);
            float kd           = param_get(PARAM_CALIB_KD);
            float stall_db     = param_get(PARAM_CALIB_STALL_DEADBAND);
            float stall_cur    = param_get(PARAM_CALIB_STALL_CUR);
            float safety_bound = param_get(PARAM_CALIB_SAFETY_BOUND);
            int   stall_ticks  = (int)param_get(PARAM_CALIB_STALL_TICKS);
            float margin       = param_get(PARAM_CALIB_MARGIN);

            float dir = (ax.state == CAL_SEEK_BOTTOM) ? ax.seek_dir : -ax.seek_dir;
            ax.ramp_target += dir * seek_speed * dt;
            send(ax.ramp_target, 0.0f, kp, kd, 0.0f);

            bool stalled = fabsf(hm.pos_rad - ax.prev_pos) < stall_db &&
                           fabsf(hm.current_A) > stall_cur;
            ax.stall_count = stalled ? ax.stall_count + 1 : 0;
            ax.prev_pos    = hm.pos_rad;

            if (fabsf(ax.ramp_target - ax.start_pos) > safety_bound) {
                ax.state = CAL_FAULT;
                comm_log(LOG_LEVEL_ERROR, "%s: FAULT - hardstop not found within %.1f rad",
                         tag, safety_bound);
                comm_send_calib_event(axis_id, CALIB_EVENT_FAULT, ax.ramp_target, 0, 0);
                return;
            }

            if (ax.stall_count < stall_ticks) return;

            if (ax.state == CAL_SEEK_BOTTOM) {
                zero();
                comm_log(LOG_LEVEL_INFO, "%s: bottom hardstop found, zeroed", tag);
                comm_send_calib_event(axis_id, CALIB_EVENT_BOTTOM_FOUND, hm.pos_rad, 0, 0);
                g_buzzer.midi(84, 200, 120);  // C6
                ax.state       = CAL_SEEK_TOP;
                ax.ramp_target = 0.0f;
                ax.prev_pos    = 0.0f;
                ax.stall_count = 0;
            } else {
                float range = hm.pos_rad;  // signed range from zero
                lim.min_rad = fminf(0.0f, range) + margin;
                lim.max_rad = fmaxf(0.0f, range) - margin;
                lim.valid   = true;
                comm_log(LOG_LEVEL_INFO, "%s: limits [%.3f, %.3f] rad", tag, lim.min_rad, lim.max_rad);
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
            float kp         = param_get(PARAM_CALIB_KP);
            float kd         = param_get(PARAM_CALIB_KD);
            float hold_kp    = param_get(PARAM_CALIB_HOLD_KP);
            float hold_kd    = param_get(PARAM_CALIB_HOLD_KD);
            float step  = seek_speed * dt;
            float error = ax.home_target - ax.ramp_target;
            if (fabsf(error) <= step) {
                ax.ramp_target = ax.home_target;
                send(ax.ramp_target, 0.0f, hold_kp, hold_kd, 0.0f);
                ax.state = CAL_DONE;
                comm_log(LOG_LEVEL_INFO, "%s: done, holding @ %.3f rad", tag, ax.ramp_target);
                comm_send_calib_event(axis_id, CALIB_EVENT_DONE, ax.ramp_target, lim.min_rad, lim.max_rad);
            } else {
                ax.ramp_target += (error > 0.0f) ? step : -step;
                send(ax.ramp_target, 0.0f, kp, kd, 0.0f);
            }
            break;
        }

        case CAL_DONE:
            send(ax.ramp_target, 0.0f, param_get(PARAM_CALIB_HOLD_KP), param_get(PARAM_CALIB_HOLD_KD), 0.0f);
            break;

        case CAL_FAULT:
            break;
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

void calibration_start() {
    ax_L = {CAL_SEEK_BOTTOM, hm_L.pos_rad, hm_L.pos_rad, 0, param_get(PARAM_CALIB_L_SEEK_DIR), 0.0f, hm_L.pos_rad};
    ax_R = {CAL_SEEK_BOTTOM, hm_R.pos_rad, hm_R.pos_rad, 0, param_get(PARAM_CALIB_R_SEEK_DIR), 0.0f, hm_R.pos_rad};
    hm_limits_L.valid = false;
    hm_limits_R.valid = false;
    s_done_announced  = false;
    s_fault_announced = false;
    g_buzzer.play(START_CHIME, sizeof(START_CHIME) / sizeof(START_CHIME[0]), 200);
    comm_log(LOG_LEVEL_INFO, "Calib: starting hardstop search");
    comm_send_calib_event(HIP_MOTOR_BOTH, CALIB_EVENT_START, 0, 0, 0);
}

void calibration_update() {
    update_axis(ax_L, hm_L, hm_limits_L, hip_motor_send_L, hip_motor_zero_L, "L", HIP_MOTOR_L);
    update_axis(ax_R, hm_R, hm_limits_R, hip_motor_send_R, hip_motor_zero_R, "R", HIP_MOTOR_R);

    // Echo the commanded position [rad] back to the GUI via the otherwise-
    // unused telemetry cmd_l/cmd_r fields, so the Hip Motors tab can plot
    // the calibration ramp against actual position.
    g_state.cmd_l = ax_L.ramp_target;
    g_state.cmd_r = ax_R.ramp_target;

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
    return ax_L.state == CAL_DONE && ax_R.state == CAL_DONE;
}

bool calibration_failed() {
    return ax_L.state == CAL_FAULT || ax_R.state == CAL_FAULT;
}
