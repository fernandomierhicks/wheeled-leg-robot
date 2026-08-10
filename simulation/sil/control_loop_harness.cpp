// Native harness that compiles and executes the production control_loop.cpp.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "control_loop.h"
#include "robot_state.h"
#include "hip_motors.h"
#include "wheel_motors.h"
#include "param_registry.h"
#include "live_tune.h"
#include "IMU.h"

static uint32_t s_now_ms = 0;
static float s_roll = 0.0f, s_roll_rate = 0.0f, s_yaw_rate = 0.0f;
uint32_t millis() { return s_now_ms; }

HipAxisState hm_L = {}, hm_R = {};
HipSetpoint hm_sp_L = {}, hm_sp_R = {};
HipLimits hm_limits_L = {0.0f, 1.0f, true};
HipLimits hm_limits_R = {0.0f, 1.0f, true};
WheelAxisState wm_L = {}, wm_R = {};
WheelMode wm_mode = WheelMode::TORQUE;
static float s_tau_l = 0.0f, s_tau_r = 0.0f;

void wheel_motors_send(float left, float right) { s_tau_l = left; s_tau_r = right; }
void hip_motors_set_setpoint_L(float p, float v, float kp, float kd, float tff) {
    hm_sp_L = {p, v, kp, kd, tff, true, millis()};
}
void hip_motors_set_setpoint_R(float p, float v, float kp, float kd, float tff) {
    hm_sp_R = {p, v, kp, kd, tff, true, millis()};
}
void hip_cmd_to_setpoints(float t, float* left, float* right) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    *left = hm_limits_L.max_rad - t * (hm_limits_L.max_rad - hm_limits_L.min_rad);
    *right = hm_limits_R.max_rad - t * (hm_limits_R.max_rad - hm_limits_R.min_rad);
}

bool stateMachine_request_estop() { return true; }
float live_tune_value(uint16_t id) { return param_get(id); }
float imu_roll() { return s_roll; }
float imu_roll_rate() { return s_roll_rate; }
float imu_yaw_rate() { return s_yaw_rate; }

static Param s_params[] = {
#include "generated_param_table.inc"
};
static constexpr size_t S_PARAM_COUNT = sizeof(s_params) / sizeof(s_params[0]);

static Param* find_param(uint16_t id) {
    for (size_t i = 0; i < S_PARAM_COUNT; ++i) if (s_params[i].id == id) return &s_params[i];
    return nullptr;
}
float param_get(uint16_t id) { Param* p = find_param(id); return p ? p->value : 0.0f; }
void param_force_set(uint16_t id, float value) {
    Param* p = find_param(id);
    if (!p) return;
    if (value < p->min_val) value = p->min_val;
    if (value > p->max_val) value = p->max_val;
    p->value = value;
}

static std::vector<float> parse(const std::string& line) {
    std::vector<float> values;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) values.push_back(std::strtof(field.c_str(), nullptr));
    return values;
}

int main() {
    // Watchdog behavior is unit-tested separately; vector equivalence focuses
    // on one-tick balance/drive/yaw/control ordering.
    param_force_set(PARAM_PITCH_WATCHDOG_ENABLE, 0.0f);
    param_force_set(PARAM_ROLL_WATCHDOG_EN, 0.0f);
    param_force_set(PARAM_HIP_RUNNING_RAMP_TIME_S, 0.0f);
    g_state.state = STATE_RUNNING;
    controlLoop_reset();
    controlLoop_reset_hip_ramp();

    std::string line;
    std::getline(std::cin, line);  // CSV header
    std::cout << "time_ms,tau_sym,tau_yaw,theta_ref,tau_l,tau_r,alpha,pitch_trim\n";
    std::cout << std::setprecision(9);
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const auto v = parse(line);
        if (v.size() != 14) return 2;
        s_now_ms = static_cast<uint32_t>(v[0]);
        g_state.pitch_rad = v[1];
        g_state.pitch_rate_rads = v[2];
        s_roll = v[3]; s_roll_rate = v[4]; s_yaw_rate = v[5];
        wm_L.vel_turns_s = v[6]; wm_R.vel_turns_s = v[7];
        const float alpha = v[8];
        hm_L.pos_rad = 1.0f - alpha; hm_R.pos_rad = 1.0f - alpha;
        hm_L.torque_nm = v[9]; hm_R.torque_nm = v[10];
        param_force_set(PARAM_V_CMD_MS, v[11]);
        param_force_set(PARAM_OMEGA_CMD_RDS, v[12]);
        param_force_set(PARAM_RADIO_HIP_CMD, v[13]);
        g_state.v_ref = v[11];
        controlLoop_run();
        std::cout << s_now_ms << ',' << g_state.tau_sym << ',' << g_state.tau_yaw
                  << ',' << g_state.theta_ref << ',' << s_tau_l << ',' << s_tau_r
                  << ',' << g_state.gain_sched_alpha << ','
                  << g_state.applied_pitch_trim << '\n';
    }
    return 0;
}
