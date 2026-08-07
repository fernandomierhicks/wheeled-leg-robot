#include "calibration.h"

#include <Arduino.h>
#include <math.h>

#include "Buzzer.h"
#include "comm_protocol.h"
#include "config.h"
#include "hip_motors.h"
#include "limit_switches.h"
#include "param_registry.h"
#include "robot_state.h"

extern Buzzer g_buzzer;

enum CalibAxisState : uint8_t {
    CAL_RELEASE_SWITCH,
    CAL_SEEK_SWITCH,
    CAL_WAIT_ZERO_SYNC,
    CAL_BACKOFF,
    CAL_RAMPDOWN,
    CAL_DONE,
    CAL_FAULT,
};

struct CalibAxis {
    CalibAxisState state;
    float seek_dir;
    float ramp_target;
    float phase_start_pos;
    float home_target;
    float kp_out;
    float kp0_rampdown;
    float kd0_rampdown;
    uint32_t phase_start_ms;
    uint32_t rampdown_start_ms;
    uint32_t overtorque_since_ms;
    uint32_t zero_feedback_seq;
    bool release_seen;
    bool overtorque_active;
};

static CalibAxis ax_L;
static CalibAxis ax_R;
static bool s_done_announced  = false;
static bool s_fault_announced = false;
static HipSetpoint s_cancel_sp_L = {};
static HipSetpoint s_cancel_sp_R = {};
static uint32_t s_cancel_start_ms = 0;

static const BuzzerNote START_CHIME[] = {{69, 80, 20}, {76, 80, 0}};
static const BuzzerNote DONE_MELODY[] = {
    {72, 80, 20}, {76, 80, 20}, {79, 120, 0}
};
static const BuzzerNote FAULT_MELODY[] = {{60, 150, 50}, {55, 200, 0}};
static constexpr float ZERO_SYNC_TOL_RAD = 0.05f;

using SendFn = void (*)(float, float, float, float, float);
using ZeroFn = void (*)();

static float slew_toward(float current, float target, float max_step) {
    float delta = target - current;
    if (delta > max_step) delta = max_step;
    if (delta < -max_step) delta = -max_step;
    return current + delta;
}

// Match the original hardstop calibration's soft-authority profile: position
// advances at the configured low speed while Kp takes calib_rampdown_s to rise
// from zero to its weak calibration target. A zero ramp time means immediate.
static float kp_slew_step(float kp_target) {
    const float ramp_s = param_get(PARAM_CALIB_RAMPDOWN_TIME_S);
    if (ramp_s <= 0.0f) return 1.0e9f;
    return kp_target / (ramp_s * CONTROL_HZ);
}

static void set_phase(CalibAxis& ax, CalibAxisState state,
                      float position, uint32_t now_ms) {
    ax.state = state;
    ax.ramp_target = position;
    ax.phase_start_pos = position;
    ax.phase_start_ms = now_ms;
    ax.overtorque_active = false;
}

static void fault_axis(CalibAxis& ax, SendFn send, const char* tag,
                       uint8_t axis_id, float position, const char* reason) {
    ax.state = CAL_FAULT;
    send(position, 0.0f, 0.0f, 0.0f, 0.0f);
    comm_log(LOG_LEVEL_ERROR, "Calib %s: FAULT - %s", tag, reason);
    comm_send_calib_event(axis_id, CALIB_EVENT_FAULT, position, 0.0f, 0.0f);
}

static bool command_motion(CalibAxis& ax, HipAxisState& hm, SendFn send,
                           const char* tag, uint8_t axis_id, float target,
                           uint16_t torque_limit_param, uint16_t kp_param) {
    // Stall detection: the AK45 MIT reply field is shaft torque [N·m], not
    // amps (see hip_motors.cpp). The limit params are in the same units.
    const float torque_limit = param_get(torque_limit_param);
    const float torque = fabsf(hm.torque_nm);

    if (!hm.ok) {
        fault_axis(ax, send, tag, axis_id, hm.pos_rad, "hip feedback lost");
        return false;
    }

    if (torque > torque_limit) {
        const uint32_t now_ms = millis();
        const uint32_t trip_ms = (uint32_t)(
            param_get(PARAM_CALIB_TORQUE_TRIP_MS) + 0.5f);
        if (!ax.overtorque_active) {
            ax.overtorque_active = true;
            ax.overtorque_since_ms = now_ms;
        }
        const uint32_t overtorque_ms =
            (uint32_t)(now_ms - ax.overtorque_since_ms);
        if (trip_ms == 0 || overtorque_ms >= trip_ms) {
            comm_log(LOG_LEVEL_ERROR,
                     "Calib %s: torque %.2f Nm exceeded %.2f Nm for %lu ms",
                     tag, torque, torque_limit,
                     (unsigned long)overtorque_ms);
            fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                       "calibration torque safety limit exceeded");
            return false;
        }
    } else {
        ax.overtorque_active = false;
    }

    const float kp_target = param_get(kp_param);
    ax.kp_out = slew_toward(ax.kp_out, kp_target,
                            kp_slew_step(kp_target));
    send(target, 0.0f,
         ax.kp_out, param_get(PARAM_CALIB_KD), 0.0f);
    return true;
}

static void define_limits(CalibAxis& ax, HipLimits& limits, float range_rad) {
    const float away_dir = -ax.seek_dir;
    const float retracted = away_dir * param_get(PARAM_CALIB_BACKOFF_RAD);
    const float extended = away_dir * range_rad;
    limits.min_rad = fminf(retracted, extended);
    limits.max_rad = fmaxf(retracted, extended);
    limits.valid = true;
    ax.home_target = retracted;
}

static void update_axis(CalibAxis& ax, HipAxisState& hm, HipLimits& limits,
                        SendFn send, ZeroFn zero, bool switch_pressed,
                        float range_rad, const char* tag, uint8_t axis_id) {
    const uint32_t now_ms = millis();
    const float dt = 1.0f / CONTROL_HZ;
    const float seek_speed = param_get(PARAM_CALIB_SEEK_SPEED);
    const float move_speed = param_get(PARAM_CALIB_MOVE_SPEED);
    const float backoff = param_get(PARAM_CALIB_BACKOFF_RAD);
    const float max_seek = param_get(PARAM_CALIB_MAX_SEEK_TRAVEL_RAD);
    const float max_release = param_get(PARAM_CALIB_MAX_RELEASE_TRAVEL_RAD);
    const uint32_t seek_timeout_ms =
        (uint32_t)(param_get(PARAM_CALIB_SEEK_TIMEOUT_S) * 1000.0f);
    const uint32_t release_timeout_ms =
        (uint32_t)(param_get(PARAM_CALIB_RELEASE_TIMEOUT_S) * 1000.0f);
    const float away_dir = -ax.seek_dir;

    if (range_rad <= backoff) {
        fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                   "configured range must exceed backoff");
        return;
    }
    if (backoff > max_release) {
        fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                   "configured backoff exceeds max release travel");
        return;
    }

    switch (ax.state) {
        case CAL_RELEASE_SWITCH: {
            if (!switch_pressed && !ax.release_seen) {
                ax.release_seen = true;
                comm_log(LOG_LEVEL_INFO,
                         "Calib %s: switch RELEASED during initial backoff", tag);
            }

            if ((uint32_t)(now_ms - ax.phase_start_ms) > release_timeout_ms) {
                fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                           "initial switch backoff timed out");
                return;
            }

            ax.ramp_target += away_dir * move_speed * dt;
            const float commanded_travel =
                away_dir * (ax.ramp_target - ax.phase_start_pos);
            if (commanded_travel > max_release) {
                fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                           "initial switch backoff exceeded max commanded travel");
                return;
            }

            if (!command_motion(ax, hm, send, tag, axis_id, ax.ramp_target,
                                PARAM_CALIB_MOVE_TORQUE_LIMIT_NM,
                                PARAM_CALIB_MOVE_KP)) return;

            const float actual_travel =
                away_dir * (hm.pos_rad - ax.phase_start_pos);
            if (actual_travel >= backoff) {
                if (!ax.release_seen || switch_pressed) {
                    fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                               "switch did not release within initial backoff");
                    return;
                }
                comm_log(LOG_LEVEL_INFO,
                         "Calib %s: initial backoff %.1f deg complete; re-seeking",
                         tag, backoff * 57.2957795f);
                ax.kp_out = fminf(ax.kp_out,
                                  param_get(PARAM_CALIB_SEEK_KP));
                set_phase(ax, CAL_SEEK_SWITCH, hm.pos_rad, now_ms);
            }
            break;
        }

        case CAL_SEEK_SWITCH:
            if (switch_pressed) {
                ax.zero_feedback_seq = hm.feedback_seq;
                zero();
                // The motor's zero command changes its coordinate frame
                // immediately. Keep the local snapshot in that same frame,
                // but do not measure travel until a newer near-zero CAN frame
                // confirms that the motor applied the command.
                hm.pos_rad = 0.0f;
                hm.vel_rad_s = 0.0f;
                limits.valid = false;
                ax.release_seen = false;
                ax.home_target = away_dir * backoff;
                set_phase(ax, CAL_WAIT_ZERO_SYNC, 0.0f, now_ms);
                send(0.0f, 0.0f, ax.kp_out,
                     param_get(PARAM_CALIB_KD), 0.0f);
                comm_log(LOG_LEVEL_INFO,
                         "Calib %s: retract switch PRESSED; encoder zeroed", tag);
                comm_send_calib_event(axis_id, CALIB_EVENT_BOTTOM_FOUND,
                                      0.0f, 0.0f, 0.0f);
                return;
            }
            ax.ramp_target += ax.seek_dir * seek_speed * dt;
            if ((uint32_t)(now_ms - ax.phase_start_ms) > seek_timeout_ms ||
                fabsf(ax.ramp_target - ax.phase_start_pos) > max_seek) {
                fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                           "retract switch not found");
                return;
            }
            command_motion(ax, hm, send, tag, axis_id, ax.ramp_target,
                           PARAM_CALIB_SEEK_TORQUE_LIMIT_NM,
                           PARAM_CALIB_SEEK_KP);
            break;

        case CAL_WAIT_ZERO_SYNC:
            if ((uint32_t)(now_ms - ax.phase_start_ms) > release_timeout_ms) {
                fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                           "encoder zero feedback not confirmed");
                return;
            }
            if (!command_motion(ax, hm, send, tag, axis_id, 0.0f,
                                PARAM_CALIB_SEEK_TORQUE_LIMIT_NM,
                                PARAM_CALIB_SEEK_KP)) return;
            if (hm.feedback_seq != ax.zero_feedback_seq &&
                fabsf(hm.pos_rad) <= ZERO_SYNC_TOL_RAD) {
                comm_log(LOG_LEVEL_INFO,
                         "Calib %s: encoder zero feedback confirmed", tag);
                set_phase(ax, CAL_BACKOFF, hm.pos_rad, now_ms);
            }
            break;

        case CAL_BACKOFF: {
            if (!switch_pressed) ax.release_seen = true;
            if ((uint32_t)(now_ms - ax.phase_start_ms) > release_timeout_ms) {
                fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                           (!ax.release_seen || switch_pressed)
                               ? "switch did not release after zeroing"
                               : "post-zero backoff timed out");
                return;
            }

            ax.ramp_target += away_dir * move_speed * dt;
            const float commanded_travel =
                away_dir * (ax.ramp_target - ax.phase_start_pos);
            if (commanded_travel > max_release) {
                fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                           "post-zero backoff exceeded max commanded travel");
                return;
            }

            if (!command_motion(ax, hm, send, tag, axis_id, ax.ramp_target,
                                PARAM_CALIB_MOVE_TORQUE_LIMIT_NM,
                                PARAM_CALIB_MOVE_KP)) return;

            const float actual_travel =
                away_dir * (hm.pos_rad - ax.phase_start_pos);
            if (actual_travel >= backoff) {
                if (!ax.release_seen || switch_pressed) {
                    fault_axis(ax, send, tag, axis_id, hm.pos_rad,
                               "switch still pressed at backoff position");
                    return;
                }
                define_limits(ax, limits, range_rad);
                comm_log(LOG_LEVEL_INFO,
                         "Calib %s: backoff %.1f deg; limits [%.3f, %.3f] rad",
                         tag, backoff * 57.2957795f,
                         limits.min_rad, limits.max_rad);
                comm_send_calib_event(axis_id, CALIB_EVENT_LIMITS,
                                      away_dir * range_rad,
                                      limits.min_rad, limits.max_rad);
                ax.rampdown_start_ms = now_ms;
                ax.kp0_rampdown = ax.kp_out;
                ax.kd0_rampdown = param_get(PARAM_CALIB_KD);
                ax.state = CAL_RAMPDOWN;
                comm_send_calib_event(axis_id, CALIB_EVENT_DONE,
                                      ax.home_target,
                                      limits.min_rad, limits.max_rad);
            }
            break;
        }

        case CAL_RAMPDOWN: {
            const float ramp_s = param_get(PARAM_CALIB_RAMPDOWN_TIME_S);
            const float elapsed = (now_ms - ax.rampdown_start_ms) / 1000.0f;
            float alpha = (ramp_s > 0.0f) ? (1.0f - elapsed / ramp_s) : 0.0f;
            if (alpha < 0.0f) alpha = 0.0f;
            send(ax.home_target, 0.0f,
                 alpha * ax.kp0_rampdown,
                 alpha * ax.kd0_rampdown, 0.0f);
            if (alpha <= 0.0f) {
                ax.state = CAL_DONE;
                comm_log(LOG_LEVEL_INFO,
                         "Calib %s: complete at %.3f rad", tag, ax.home_target);
            }
            break;
        }

        case CAL_DONE:
            send(ax.home_target, 0.0f, 0.0f, 0.0f, 0.0f);
            break;

        case CAL_FAULT:
            send(hm.pos_rad, 0.0f, 0.0f, 0.0f, 0.0f);
            break;
    }
}

static CalibAxis make_axis(float position, float seek_dir, bool pressed) {
    CalibAxis axis{};
    axis.seek_dir = (seek_dir >= 0.0f) ? 1.0f : -1.0f;
    axis.state = pressed ? CAL_RELEASE_SWITCH : CAL_SEEK_SWITCH;
    axis.ramp_target = position;
    axis.phase_start_pos = position;
    axis.phase_start_ms = millis();
    return axis;
}

void calibration_start() {
    const bool l_en = param_get(PARAM_HIP_L_ENABLE) >= 0.5f;
    const bool r_en = param_get(PARAM_HIP_R_ENABLE) >= 0.5f;

    ax_L = make_axis(hm_L.pos_rad, CALIB_L_SEEK_DIR,
                     limit_switch_left_active());
    ax_R = make_axis(hm_R.pos_rad, CALIB_R_SEEK_DIR,
                     limit_switch_right_active());

    if (!l_en) {
        ax_L.state = CAL_DONE;
        comm_log(LOG_LEVEL_WARN, "Calib L: skipped (hip_l_enable=0)");
    }
    if (!r_en) {
        ax_R.state = CAL_DONE;
        comm_log(LOG_LEVEL_WARN, "Calib R: skipped (hip_r_enable=0)");
    }

    hm_limits_L.valid = false;
    hm_limits_R.valid = false;
    s_done_announced = false;
    s_fault_announced = false;

    g_buzzer.play(START_CHIME, sizeof(START_CHIME) / sizeof(START_CHIME[0]));
    comm_log(LOG_LEVEL_INFO, "Calib: starting retract-switch homing");
    if (l_en && limit_switch_left_active())
        comm_log(LOG_LEVEL_INFO, "Calib L: switch already PRESSED; releasing first");
    if (r_en && limit_switch_right_active())
        comm_log(LOG_LEVEL_INFO, "Calib R: switch already PRESSED; releasing first");
    comm_send_calib_event(HIP_MOTOR_BOTH, CALIB_EVENT_START, 0, 0, 0);
}

void calibration_update() {
    if (param_get(PARAM_HIP_L_ENABLE) >= 0.5f) {
        update_axis(ax_L, hm_L, hm_limits_L,
                    hip_motors_set_setpoint_L, hip_motor_zero_L,
                    limit_switch_left_active(),
                    param_get(PARAM_CALIB_RANGE_L_RAD), "L", HIP_MOTOR_L);
    }
    if (param_get(PARAM_HIP_R_ENABLE) >= 0.5f) {
        update_axis(ax_R, hm_R, hm_limits_R,
                    hip_motors_set_setpoint_R, hip_motor_zero_R,
                    limit_switch_right_active(),
                    param_get(PARAM_CALIB_RANGE_R_RAD), "R", HIP_MOTOR_R);
    }

    if (calibration_done() && !s_done_announced) {
        s_done_announced = true;
        comm_log(LOG_LEVEL_INFO, "*** CALIBRATION SUCCESSFUL ***");
        g_buzzer.play(DONE_MELODY, sizeof(DONE_MELODY) / sizeof(DONE_MELODY[0]));
    }
    if (calibration_failed() && !s_fault_announced) {
        s_fault_announced = true;
        g_buzzer.play(FAULT_MELODY, sizeof(FAULT_MELODY) / sizeof(FAULT_MELODY[0]));
    }
}

bool calibration_done() {
    const bool l_done =
        param_get(PARAM_HIP_L_ENABLE) < 0.5f || ax_L.state == CAL_DONE;
    const bool r_done =
        param_get(PARAM_HIP_R_ENABLE) < 0.5f || ax_R.state == CAL_DONE;
    return l_done && r_done;
}

bool calibration_failed() {
    const bool l_bad =
        param_get(PARAM_HIP_L_ENABLE) >= 0.5f && ax_L.state == CAL_FAULT;
    const bool r_bad =
        param_get(PARAM_HIP_R_ENABLE) >= 0.5f && ax_R.state == CAL_FAULT;
    return l_bad || r_bad;
}

static void mark_cancelled() {
    ax_L.state = CAL_FAULT;
    ax_R.state = CAL_FAULT;
    hm_limits_L.valid = false;
    hm_limits_R.valid = false;
    s_done_announced = false;
    s_fault_announced = false;
    comm_log(LOG_LEVEL_INFO, "CALIBRATION CANCELLED");
}

void calibration_begin_disarm() {
    s_cancel_sp_L = hm_sp_L;
    s_cancel_sp_R = hm_sp_R;
    s_cancel_start_ms = millis();
    mark_cancelled();
}

bool calibration_run_disarm() {
    if (!s_cancel_sp_L.active && !s_cancel_sp_R.active) {
        hip_motors_clear_setpoints();
        return false;
    }

    const float ramp_s = param_get(PARAM_CALIB_RAMPDOWN_TIME_S);
    const float elapsed = (millis() - s_cancel_start_ms) / 1000.0f;
    float t = (ramp_s > 0.0f) ? (elapsed / ramp_s) : 1.0f;
    if (t > 1.0f) t = 1.0f;
    const float alpha = 1.0f - t;

    if (s_cancel_sp_L.active) {
        hip_motors_set_setpoint_L(
            s_cancel_sp_L.p, 0.0f,
            alpha * s_cancel_sp_L.kp,
            alpha * s_cancel_sp_L.kd,
            alpha * s_cancel_sp_L.tff);
    }
    if (s_cancel_sp_R.active) {
        hip_motors_set_setpoint_R(
            s_cancel_sp_R.p, 0.0f,
            alpha * s_cancel_sp_R.kp,
            alpha * s_cancel_sp_R.kd,
            alpha * s_cancel_sp_R.tff);
    }

    if (t < 1.0f) return true;
    hip_motors_clear_setpoints();
    return false;
}

void calibration_abort() {
    mark_cancelled();
    hip_motors_clear_setpoints();
}
