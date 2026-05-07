#pragma once
// controllers.h — Balance, velocity, yaw, and suspension controllers.
//
// All controllers are pure structs with init()/update() methods.
// No dynamic allocation, no virtuals — suitable for hard-RT on MCU.
//
// Control signal flow (matches simulation/mujoco/master_sim/sim_loop.py):
//   1. VelocityPI  →  theta_ref  (lean angle from velocity error)
//   2. LQR         →  tau_sym    (symmetric wheel torque from balance error)
//   3. YawPI       →  tau_yaw    (differential wheel torque from yaw error)
//   4. Wheel split:   tau_L = tau_sym - tau_yaw,  tau_R = tau_sym + tau_yaw
//   5. Suspension  →  tau_hip_L, tau_hip_R  (impedance + roll leveling)

#include <Arduino.h>
#include "config.h"

// ── LQR Balance Controller (3-state) ────────────────────────────────────────
//
// State vector:  x = [pitch - theta_ref,  pitch_rate,  wheel_vel - v_ref]
// Control law:   tau_sym = -K · x   (clamped to torque limit)
//
// K gains must be pre-computed from simulation (export_gains.py) since
// solving the continuous algebraic Riccati equation on-MCU is impractical.

struct LQRController {
    float K[3];          // [k_pitch, k_pitch_rate, k_vel]
    float torque_max;

    void init() {
        K[0] = LQR_K_PITCH;
        K[1] = LQR_K_PITCH_RATE;
        K[2] = LQR_K_VEL;
        torque_max = WHEEL_TORQUE_MAX;
    }

    // Returns tau_sym [N·m]
    float update(float pitch, float pitch_rate, float wheel_vel_avg,
                 float theta_ref, float v_ref) {
        float x0 = pitch - theta_ref;
        float x1 = pitch_rate;
        float x2 = wheel_vel_avg - v_ref;
        float u = -(K[0] * x0 + K[1] * x1 + K[2] * x2);
        return constrain(u, -torque_max, torque_max);
    }

    void reset() {
        // LQR is memoryless — nothing to reset
    }
};

// ── Velocity PI (outer loop → lean angle command) ───────────────────────────
//
// Converts velocity error into a lean angle reference (theta_ref).
//   theta_ref = Kp·e + Ki·∫e + Kff·(dv_cmd/dt)
// with anti-windup and rate limiting.

struct VelocityPI {
    float Kp, Ki, Kff;
    float theta_max;     // [rad] max lean angle
    float int_max;       // [rad·s] integrator anti-windup
    float rate_lim;      // [rad/s] theta_ref rate limit
    float integral;
    float prev_v_desired;
    float prev_theta_ref;

    void init() {
        Kp       = VEL_PI_KP;
        Ki       = VEL_PI_KI;
        Kff      = VEL_PI_KFF;
        theta_max = VEL_PI_THETA_MAX;
        int_max  = VEL_PI_INT_MAX;
        rate_lim = VEL_PI_RATE_LIM;
        integral = 0.0f;
        prev_v_desired = 0.0f;
        prev_theta_ref = 0.0f;
    }

    // Returns theta_ref [rad]
    float update(float v_target, float v_measured, float dt) {
        float v_err = v_target - v_measured;

        integral += v_err * dt;
        integral = constrain(integral, -int_max, int_max);

        float dv_cmd_dt = (v_target - prev_v_desired) / dt;
        prev_v_desired = v_target;

        float theta_ref = Kp * v_err + Ki * integral + Kff * dv_cmd_dt;
        theta_ref = constrain(theta_ref, -theta_max, theta_max);

        // Rate limit
        float d_max = rate_lim * dt;
        theta_ref = constrain(theta_ref,
                              prev_theta_ref - d_max,
                              prev_theta_ref + d_max);
        prev_theta_ref = theta_ref;

        return theta_ref;
    }

    void reset() {
        integral = 0.0f;
        prev_v_desired = 0.0f;
        prev_theta_ref = 0.0f;
    }
};

// ── Yaw PI Controller (differential steering) ──────────────────────────────
//
// tau_yaw = Kp·(omega_cmd - yaw_rate) + Ki·∫error
// Wheel split: tau_L = tau_sym - tau_yaw,  tau_R = tau_sym + tau_yaw

struct YawPI {
    float Kp, Ki;
    float torque_max;
    float int_max;
    float integral;

    void init() {
        Kp         = YAW_PI_KP;
        Ki         = YAW_PI_KI;
        torque_max = YAW_PI_TORQUE_MAX;
        int_max    = YAW_PI_INT_MAX;
        integral   = 0.0f;
    }

    // Returns tau_yaw [N·m]
    float update(float omega_target, float yaw_rate, float dt) {
        float err = omega_target - yaw_rate;

        integral += err * dt;
        integral = constrain(integral, -int_max, int_max);

        float tau = Kp * err + Ki * integral;
        return constrain(tau, -torque_max, torque_max);
    }

    void reset() {
        integral = 0.0f;
    }
};

// ── Suspension Controller (impedance + roll leveling) ───────────────────────
//
// Each hip acts as a virtual spring-damper about the nominal angle:
//   tau_hip = -(K_s · (q - q_nom) + B_s · dq)
//
// Roll leveling offsets q_nom per side:
//   delta_q = K_roll · roll + D_roll · roll_rate
//   q_nom_L = q_nom + delta_q   (retract left when tilted left)
//   q_nom_R = q_nom - delta_q

struct SuspensionController {
    float K_s, B_s;          // impedance spring/damper
    float K_roll, D_roll;    // roll leveling gains
    float q_nom;             // nominal hip angle [rad]
    float torque_max;
    float hip_min, hip_max;  // hip angle limits [rad]

    void init() {
        K_s        = SUSP_K_S;
        B_s        = SUSP_B_S;
        K_roll     = SUSP_K_ROLL;
        D_roll     = SUSP_D_ROLL;
        q_nom      = Q_NOM;
        torque_max = HIP_IMP_TORQUE_MAX;
        hip_min    = Q_EXT;   // more negative = more extended
        hip_max    = Q_RET;   // less negative = more retracted
    }

    // Computes both hip torques in-place
    void update(float roll, float roll_rate,
                float hip_q_L, float hip_dq_L,
                float hip_q_R, float hip_dq_R,
                float &tau_L, float &tau_R) {
        // Roll leveling offset
        float delta_q = K_roll * roll + D_roll * roll_rate;
        float q_nom_L = constrain(q_nom + delta_q, hip_min, hip_max);
        float q_nom_R = constrain(q_nom - delta_q, hip_min, hip_max);

        // Impedance: virtual spring-damper
        tau_L = -(K_s * (hip_q_L - q_nom_L) + B_s * hip_dq_L);
        tau_R = -(K_s * (hip_q_R - q_nom_R) + B_s * hip_dq_R);
        tau_L = constrain(tau_L, -torque_max, torque_max);
        tau_R = constrain(tau_R, -torque_max, torque_max);
    }

    void reset() {
        // Stateless — nothing to reset
    }
};
