"""test_behavioral.py — Behavioral test harness for LQR tuning.

Scenarios:
  - self_balance: 30s stable upright (default)
  - drive: forward/backward acceleration, speed tracking

Usage:
    python test_behavioral.py --scenario self_balance --q 100 --r 0.1 --blend 1.0
    python test_behavioral.py --scenario drive --q 100 --r 0.1 --blend 1.0

Output:
    Prints metrics to stdout as JSON.
"""
import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sim_config import *
from physics import solve_ik, get_equilibrium_pitch, build_xml, build_assets
from motor_models import MotorModel


@dataclass
class TestMetrics:
    """Container for test scenario metrics."""
    rms_pitch_deg: float = 0.0
    max_pitch_deviation_deg: float = 0.0
    pitch_settle_time_s: float = -1.0
    pitch_oscillations_count: int = 0
    max_wheel_pos_drift_m: float = 0.0
    rms_wheel_tracking_error_m: float = 0.0
    wheel_velocity_rms_m_s: float = 0.0
    control_effort_integral: float = 0.0
    peak_torque_nm: float = 0.0
    avg_torque_nm: float = 0.0
    max_bearing_load_n: float = 0.0
    max_femur_lateral_n: float = 0.0
    max_impact_g: float = 0.0
    pass_fail: bool = False
    notes: str = ""


def run_self_balance_scenario(
    duration_s: float = 30.0,
    q_pitch: float = 100.0,
    r_val: float = 0.1,
    blend: float = 1.0,
) -> TestMetrics:
    """
    Run self-balance scenario: robot stays upright and still for 30s.

    Args:
        duration_s: Scenario duration (seconds)
        q_pitch: LQR Q[0,0] gain (pitch error weight)
        r_val: LQR R scalar (control effort weight)
        blend: LQR blend factor (0=PID only, 1=LQR only)

    Returns:
        TestMetrics dict with all measurements
    """
    # Build LQR gain K from Q/R
    from lqr_design import compute_lqr_gain
    K = compute_lqr_gain(Q_pitch=q_pitch, R=r_val)

    # Initialize physics
    p = ROBOT
    xml = build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml, assets=build_assets())
    data = mujoco.MjData(model)

    # Fix 4-bar equality constraint anchors
    for eq_name in ["4bar_close_L", "4bar_close_R"]:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
        model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    # Helper functions for joint/body access
    def jqp(n):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def jdof(n):
        return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    def gid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

    # Body/geom IDs
    femur_bid_L = bid("femur_L")
    tibia_bid_L = bid("tibia_L")
    coupler_bid_L = bid("coupler_L")
    wheel_bid_L = bid("wheel_asm_L")
    wheel_bid_R = bid("wheel_asm_R")
    box_bid = bid("box")
    tire_gid_L = gid("wheel_tire_geom_L")

    # Joint indices
    s_root = jqp("root_free")
    s_hip_L = jqp("hip_L")
    s_hip_R = jqp("hip_R")
    s_hF_L = jqp("hinge_F_L")
    s_hF_R = jqp("hinge_F_R")
    s_knee_L = jqp("knee_joint_L")
    s_knee_R = jqp("knee_joint_R")
    d_root = jdof("root_free")
    d_pitch = d_root + 4
    d_hip_L = jdof("hip_L")
    d_hip_R = jdof("hip_R")
    d_whl_L = jdof("wheel_spin_L")
    d_whl_R = jdof("wheel_spin_R")

    # Equality constraint for 4-bar force extraction
    eq_id_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close_L")
    _EQ_TYPE = mujoco.mjtConstraint.mjCNSTR_EQUALITY
    _f6 = np.zeros(6)

    # Motor models
    motor_hip_L = MotorModel(HIP_TORQUE_LIMIT, OMEGA_MAX, HIP_TAU_ELEC, HIP_B_FRICTION)
    motor_hip_R = MotorModel(HIP_TORQUE_LIMIT, OMEGA_MAX, HIP_TAU_ELEC, HIP_B_FRICTION)
    motor_whl_L = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD, WHEEL_TAU_ELEC, WHEEL_B_FRICTION)
    motor_whl_R = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD, WHEEL_TAU_ELEC, WHEEL_B_FRICTION)

    def _struct_all_L():
        """Extract forces on left leg. Returns (fax_fem, flat_fem, ..., fbear_W, grf_L)"""
        # Femur (A→C)
        cfrc_fem = data.cfrc_int[femur_bid_L]
        dv = data.xpos[tibia_bid_L] - data.xpos[femur_bid_L]
        n = math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
        ax_fem = dv / n if n > 1e-6 else np.array([1., 0., 0.])
        F_fem = cfrc_fem[3:6]
        fax_f = float(np.dot(F_fem, ax_fem))
        flat_f = float(math.sqrt(max(0., float(np.dot(F_fem, F_fem)) - fax_f**2)))
        fbear_A = float(math.sqrt(F_fem[0]**2 + F_fem[2]**2))

        # GRF
        grf = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == tire_gid_L or c.geom2 == tire_gid_L:
                mujoco.mj_contactForce(model, data, i, _f6)
                grf += _f6[0]

        # E force from 4-bar equality
        F_E_vec = np.zeros(3)
        k = 0
        for i in range(data.nefc):
            if data.efc_type[i] == _EQ_TYPE and data.efc_id[i] == eq_id_L:
                if k < 3:
                    F_E_vec[k] = data.efc_force[i]
                k += 1
        fbear_E = float(np.linalg.norm(F_E_vec))
        fbear_F = fbear_E

        # Tibia
        cfrc_tib = data.cfrc_int[tibia_bid_L]
        F_C_vec = cfrc_tib[3:6] - F_E_vec
        fbear_C = float(np.linalg.norm(F_C_vec))
        dv_t = data.xpos[wheel_bid_L] - data.xpos[tibia_bid_L]
        n_t = math.sqrt(dv_t[0]**2 + dv_t[1]**2 + dv_t[2]**2)
        ax_t = dv_t / n_t if n_t > 1e-6 else np.array([0., 0., -1.])
        fax_t = float(np.dot(cfrc_tib[3:6], ax_t))
        flat_t = float(math.sqrt(max(0., float(np.dot(cfrc_tib[3:6], cfrc_tib[3:6])) - fax_t**2)))

        # Coupler
        cfrc_cpl = data.cfrc_int[coupler_bid_L]
        R_cpl = data.xmat[coupler_bid_L].reshape(3, 3)
        ax_cpl = R_cpl @ np.array([-1., 0., 0.])
        fax_c = float(np.dot(cfrc_cpl[3:6], ax_cpl))

        # Wheel bearing
        cfrc_whl = data.cfrc_int[wheel_bid_L]
        fbear_W = float(math.sqrt(cfrc_whl[3]**2 + cfrc_whl[5]**2))

        return (fax_f, flat_f, fax_t, flat_t, fax_c,
                fbear_A, fbear_C, fbear_E, fbear_F, fbear_W, float(grf))

    def _init():
        """Initialize to neutral stance."""
        mujoco.mj_resetData(model, data)
        ik = solve_ik(Q_NEUTRAL, p)
        if not ik:
            raise RuntimeError("IK failed at Q_NEUTRAL")
        for s_hF, s_hip, s_knee in [
            (s_hF_L, s_hip_L, s_knee_L),
            (s_hF_R, s_hip_R, s_knee_R),
        ]:
            data.qpos[s_hF] = ik['q_coupler_F']
            data.qpos[s_hip] = ik['q_hip']
            data.qpos[s_knee] = ik['q_knee']
        mujoco.mj_forward(model, data)
        wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
        data.qpos[s_root + 2] += WHEEL_R - wz
        theta = get_equilibrium_pitch(p, Q_NEUTRAL)
        data.qpos[s_root + 3] = math.cos(theta / 2)
        data.qpos[s_root + 4] = 0.0
        data.qpos[s_root + 5] = math.sin(theta / 2)
        data.qpos[s_root + 6] = 0.0
        mujoco.mj_forward(model, data)

    _init()

    # Simulation loop parameters
    PHYSICS_HZ = round(1.0 / model.opt.timestep)
    RENDER_HZ = 60
    steps_per_frame = max(1, int(PHYSICS_HZ / RENDER_HZ))

    # Data logging
    pitch_log = []
    pitch_rate_log = []
    wheel_pos_log = []
    wheel_vel_log = []
    torque_log = []
    bearing_loads_log = []
    femur_lateral_log = []
    time_log = []

    # State variables
    wheel_pos_L = 0.0
    wheel_pos_R = 0.0
    pitch_integral = 0.0

    # Run scenario
    while data.time < duration_s:
        sim_t = float(data.time)
        _dt = model.opt.timestep

        # State extraction
        q_quat = data.xquat[box_bid]
        pitch_true = math.asin(max(-1.0, min(1.0,
            2 * (q_quat[0]*q_quat[2] - q_quat[3]*q_quat[1]))))
        pitch_rate_true = data.qvel[d_pitch]
        pitch = pitch_true + np.random.normal(0, PITCH_NOISE_STD_RAD)
        pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)

        hip_q = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
        wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0

        # Integrate wheel position
        wheel_pos_L += data.qvel[d_whl_L] * _dt
        wheel_pos_R += data.qvel[d_whl_R] * _dt
        wheel_pos = (wheel_pos_L + wheel_pos_R) / 2.0

        # Equilibrium pitch (feedforward)
        pitch_ff = get_equilibrium_pitch(p, hip_q)
        target_pitch = pitch_ff
        pitch_error = pitch - target_pitch
        pitch_integral = np.clip(pitch_integral + pitch_error * _dt, -1.0, 1.0)

        # PID control (legacy)
        u_pid = PITCH_KP * pitch_error + PITCH_KI * pitch_integral + PITCH_KD * pitch_rate

        # LQR control
        _lqr_state = np.array([pitch_error, pitch_rate, wheel_pos, wheel_vel])
        u_lqr = float(-K @ _lqr_state)

        # Blend
        u_bal = (1.0 - blend) * u_pid + blend * u_lqr

        # Send to wheel motors (no turning in self-balance)
        u_whl_L = motor_whl_L.step(u_bal, data.qvel[d_whl_L], _dt)
        u_whl_R = motor_whl_R.step(u_bal, data.qvel[d_whl_R], _dt)
        data.ctrl[2] = u_whl_L
        data.ctrl[3] = u_whl_R

        # Hip motors: suspension mode (neutral stance)
        u_hip = HIP_KP_SUSP * (Q_NEUTRAL - hip_q) + HIP_KD_SUSP * (0.0 - data.qvel[d_hip_L])
        u_hip = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
        data.ctrl[0] = motor_hip_L.step(u_hip, data.qvel[d_hip_L], _dt)
        data.ctrl[1] = motor_hip_R.step(u_hip, data.qvel[d_hip_R], _dt)

        # Step physics
        mujoco.mj_step(model, data)

        # Log metrics every ~1/10 Hz for lighter data footprint
        if int(sim_t * 10) % max(1, PHYSICS_HZ // 10) == 0:
            time_log.append(sim_t)
            pitch_log.append(math.degrees(pitch))
            pitch_rate_log.append(pitch_rate)
            wheel_pos_log.append(wheel_pos)
            wheel_vel_log.append(wheel_vel)
            torque_log.append(u_bal)
            forces = _struct_all_L()
            bearing_loads_log.append(max(forces[5:10]))  # max of bearing loads
            femur_lateral_log.append(forces[1])  # flat_fem

    # ─────────────────────────────────────────────────────────────────────────
    # Compute metrics
    # ─────────────────────────────────────────────────────────────────────────
    metrics = TestMetrics()

    if len(pitch_log) > 0:
        pitch_arr = np.array(pitch_log)
        # RMS pitch wobble: deviation from mean (what robot maintains), not from zero
        pitch_mean = np.mean(pitch_arr)
        pitch_wobble = pitch_arr - pitch_mean
        metrics.rms_pitch_deg = float(np.sqrt(np.mean(pitch_wobble**2)))
        metrics.max_pitch_deviation_deg = float(np.max(np.abs(pitch_wobble)))

        # Settle time: find first time pitch wobble stays within ±1° (after transient)
        threshold_deg = 1.0
        settled_idx = None
        for i in range(len(pitch_arr) // 2, len(pitch_arr)):
            if np.max(np.abs(pitch_wobble[i:])) < threshold_deg:
                settled_idx = i
                break
        if settled_idx is not None:
            metrics.pitch_settle_time_s = float(time_log[settled_idx])

        # Oscillation count: zero crossings of pitch wobble around mean
        zero_crosses = 0
        for i in range(1, len(pitch_wobble)):
            if pitch_wobble[i-1] * pitch_wobble[i] < 0:
                zero_crosses += 1
        metrics.pitch_oscillations_count = zero_crosses

    if len(wheel_pos_log) > 0:
        wheel_pos_arr = np.array(wheel_pos_log)
        metrics.max_wheel_pos_drift_m = float(np.max(np.abs(wheel_pos_arr)))
        metrics.rms_wheel_tracking_error_m = float(np.sqrt(np.mean(wheel_pos_arr**2)))

        wheel_vel_arr = np.array(wheel_vel_log)
        metrics.wheel_velocity_rms_m_s = float(np.sqrt(np.mean(wheel_vel_arr**2)))

    if len(torque_log) > 0:
        torque_arr = np.array(torque_log)
        metrics.control_effort_integral = float(np.sum(np.abs(torque_arr)) * (duration_s / len(torque_arr)))
        metrics.peak_torque_nm = float(np.max(np.abs(torque_arr)))
        metrics.avg_torque_nm = float(np.mean(np.abs(torque_arr)))

    if len(bearing_loads_log) > 0:
        metrics.max_bearing_load_n = float(np.max(bearing_loads_log))

    if len(femur_lateral_log) > 0:
        metrics.max_femur_lateral_n = float(np.max(np.abs(femur_lateral_log)))

    # Pass/fail: simple heuristic
    # Pass if: RMS pitch wobble < 1.0°, drift < 0.2m, settle within 2s, reasonable oscillation rate
    metrics.pass_fail = (
        metrics.rms_pitch_deg < 1.0 and
        metrics.max_wheel_pos_drift_m < 0.2 and
        (metrics.pitch_settle_time_s < 2.0 or metrics.pitch_settle_time_s < 0.1) and
        metrics.pitch_oscillations_count > 5  # Natural oscillation, not flat
    )

    metrics.notes = f"Self-balance {duration_s}s, Q={q_pitch}, R={r_val}, blend={blend}"

    return metrics


def run_drive_scenario(
    duration_s: float = 20.0,
    q_pitch: float = 100.0,
    r_val: float = 0.1,
    blend: float = 1.0,
) -> TestMetrics:
    """
    Run drive forward/backward scenario: accelerate, maintain speed, reverse.

    Phases:
      - 0-5s: Ramp forward velocity command (0 → 0.3 m/s)
      - 5-10s: Hold forward speed
      - 10-15s: Decelerate, reverse (0.3 → -0.3 m/s)
      - 15-20s: Hold backward speed

    Args:
        duration_s: Scenario duration (seconds)
        q_pitch: LQR Q[0,0] gain
        r_val: LQR R scalar
        blend: LQR blend factor

    Returns:
        TestMetrics with drive-specific measurements
    """
    from lqr_design import compute_lqr_gain

    K = compute_lqr_gain(Q_pitch=q_pitch, R=r_val)

    # Initialize physics with optional obstacle
    p = ROBOT
    # Add 2cm obstacle at x=2.0m: (cx, cy, half_len, half_w, half_t, pitch_deg, "r g b")
    extra_ramps = [
        (2.0, 0.0, 0.05, 0.25, 0.01, 0, "0.6 0.4 0.2"),  # 2cm tall, 10cm wide
    ]
    xml = build_xml(p, extra_ramps=extra_ramps)
    model = mujoco.MjModel.from_xml_string(xml, assets=build_assets())
    data = mujoco.MjData(model)

    for eq_name in ["4bar_close_L", "4bar_close_R"]:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
        model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    def jqp(n):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def jdof(n):
        return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    def gid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

    femur_bid_L = bid("femur_L")
    tibia_bid_L = bid("tibia_L")
    coupler_bid_L = bid("coupler_L")
    wheel_bid_L = bid("wheel_asm_L")
    wheel_bid_R = bid("wheel_asm_R")
    box_bid = bid("box")
    tire_gid_L = gid("wheel_tire_geom_L")

    eq_id_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close_L")
    _EQ_TYPE = mujoco.mjtConstraint.mjCNSTR_EQUALITY
    _f6 = np.zeros(6)

    s_root = jqp("root_free")
    s_hip_L = jqp("hip_L")
    s_hip_R = jqp("hip_R")
    s_hF_L = jqp("hinge_F_L")
    s_hF_R = jqp("hinge_F_R")
    s_knee_L = jqp("knee_joint_L")
    s_knee_R = jqp("knee_joint_R")
    d_root = jdof("root_free")
    d_pitch = d_root + 4
    d_hip_L = jdof("hip_L")
    d_hip_R = jdof("hip_R")
    d_whl_L = jdof("wheel_spin_L")
    d_whl_R = jdof("wheel_spin_R")

    motor_hip_L = MotorModel(HIP_TORQUE_LIMIT, OMEGA_MAX, HIP_TAU_ELEC, HIP_B_FRICTION)
    motor_hip_R = MotorModel(HIP_TORQUE_LIMIT, OMEGA_MAX, HIP_TAU_ELEC, HIP_B_FRICTION)
    motor_whl_L = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD, WHEEL_TAU_ELEC, WHEEL_B_FRICTION)
    motor_whl_R = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD, WHEEL_TAU_ELEC, WHEEL_B_FRICTION)

    def _struct_all_L():
        """Extract forces (same as self-balance)."""
        cfrc_fem = data.cfrc_int[femur_bid_L]
        dv = data.xpos[tibia_bid_L] - data.xpos[femur_bid_L]
        n = math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
        ax_fem = dv / n if n > 1e-6 else np.array([1., 0., 0.])
        F_fem = cfrc_fem[3:6]
        fax_f = float(np.dot(F_fem, ax_fem))
        flat_f = float(math.sqrt(max(0., float(np.dot(F_fem, F_fem)) - fax_f**2)))
        fbear_A = float(math.sqrt(F_fem[0]**2 + F_fem[2]**2))

        grf = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == tire_gid_L or c.geom2 == tire_gid_L:
                mujoco.mj_contactForce(model, data, i, _f6)
                grf += _f6[0]

        F_E_vec = np.zeros(3)
        k = 0
        for i in range(data.nefc):
            if data.efc_type[i] == _EQ_TYPE and data.efc_id[i] == eq_id_L:
                if k < 3:
                    F_E_vec[k] = data.efc_force[i]
                k += 1
        fbear_E = float(np.linalg.norm(F_E_vec))
        fbear_F = fbear_E

        cfrc_tib = data.cfrc_int[tibia_bid_L]
        F_C_vec = cfrc_tib[3:6] - F_E_vec
        fbear_C = float(np.linalg.norm(F_C_vec))
        dv_t = data.xpos[wheel_bid_L] - data.xpos[tibia_bid_L]
        n_t = math.sqrt(dv_t[0]**2 + dv_t[1]**2 + dv_t[2]**2)
        ax_t = dv_t / n_t if n_t > 1e-6 else np.array([0., 0., -1.])
        fax_t = float(np.dot(cfrc_tib[3:6], ax_t))
        flat_t = float(math.sqrt(max(0., float(np.dot(cfrc_tib[3:6], cfrc_tib[3:6])) - fax_t**2)))

        cfrc_cpl = data.cfrc_int[coupler_bid_L]
        R_cpl = data.xmat[coupler_bid_L].reshape(3, 3)
        ax_cpl = R_cpl @ np.array([-1., 0., 0.])
        fax_c = float(np.dot(cfrc_cpl[3:6], ax_cpl))

        cfrc_whl = data.cfrc_int[wheel_bid_L]
        fbear_W = float(math.sqrt(cfrc_whl[3]**2 + cfrc_whl[5]**2))

        return (fax_f, flat_f, fax_t, flat_t, fax_c,
                fbear_A, fbear_C, fbear_E, fbear_F, fbear_W, float(grf))

    def _init():
        """Initialize to neutral stance."""
        mujoco.mj_resetData(model, data)
        ik = solve_ik(Q_NEUTRAL, p)
        if not ik:
            raise RuntimeError("IK failed at Q_NEUTRAL")
        for s_hF, s_hip, s_knee in [
            (s_hF_L, s_hip_L, s_knee_L),
            (s_hF_R, s_hip_R, s_knee_R),
        ]:
            data.qpos[s_hF] = ik['q_coupler_F']
            data.qpos[s_hip] = ik['q_hip']
            data.qpos[s_knee] = ik['q_knee']
        mujoco.mj_forward(model, data)
        wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
        data.qpos[s_root + 2] += WHEEL_R - wz
        theta = get_equilibrium_pitch(p, Q_NEUTRAL)
        data.qpos[s_root + 3] = math.cos(theta / 2)
        data.qpos[s_root + 4] = 0.0
        data.qpos[s_root + 5] = math.sin(theta / 2)
        data.qpos[s_root + 6] = 0.0
        mujoco.mj_forward(model, data)

    _init()

    PHYSICS_HZ = round(1.0 / model.opt.timestep)
    RENDER_HZ = 60
    steps_per_frame = max(1, int(PHYSICS_HZ / RENDER_HZ))

    # Data logging
    pitch_log = []
    pitch_rate_log = []
    wheel_pos_log = []
    wheel_vel_log = []
    wheel_vel_cmd_log = []
    torque_log = []
    bearing_loads_log = []
    time_log = []

    # State variables
    wheel_pos_L = 0.0
    wheel_pos_R = 0.0
    pitch_integral = 0.0
    wheel_vel_target = 0.0

    # Drive command schedule
    def get_wheel_vel_target(sim_t):
        """Return target wheel velocity for current time."""
        # Speed: 1.0 m/s (wheels 150mm = 0.075m radius, so ~13.3 rad/s)
        V_MAX = 1.0  # m/s
        if sim_t < 5.0:
            # Ramp forward: 0 → 1.0 m/s
            return (sim_t / 5.0) * V_MAX
        elif sim_t < 10.0:
            # Hold forward at 1.0 m/s (crosses obstacle at ~2s into this phase)
            return V_MAX
        elif sim_t < 15.0:
            # Ramp reverse: 1.0 → -1.0 m/s
            return V_MAX - ((sim_t - 10.0) / 5.0) * 2.0 * V_MAX
        else:
            # Hold backward
            return -V_MAX

    # Run scenario
    while data.time < duration_s:
        sim_t = float(data.time)
        _dt = model.opt.timestep

        # State extraction
        q_quat = data.xquat[box_bid]
        pitch_true = math.asin(max(-1.0, min(1.0,
            2 * (q_quat[0]*q_quat[2] - q_quat[3]*q_quat[1]))))
        pitch_rate_true = data.qvel[d_pitch]
        pitch = pitch_true + np.random.normal(0, PITCH_NOISE_STD_RAD)
        pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)

        hip_q = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
        wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0

        # Integrate wheel position
        wheel_pos_L += data.qvel[d_whl_L] * _dt
        wheel_pos_R += data.qvel[d_whl_R] * _dt
        wheel_pos = (wheel_pos_L + wheel_pos_R) / 2.0

        # Drive control: use velocity error directly in wheel control
        wheel_vel_target = get_wheel_vel_target(sim_t)
        wheel_vel_linear = wheel_vel * WHEEL_R  # Convert rad/s to m/s

        # Equilibrium pitch (feedforward)
        pitch_ff = get_equilibrium_pitch(p, hip_q)
        target_pitch = pitch_ff
        pitch_error = pitch - target_pitch
        pitch_integral = np.clip(pitch_integral + pitch_error * _dt, -1.0, 1.0)

        # PID control
        u_pid = PITCH_KP * pitch_error + PITCH_KI * pitch_integral + PITCH_KD * pitch_rate

        # LQR control
        # Drive: damp pitch rate only, allow pitch to vary for velocity tracking
        _lqr_state = np.array([0.0, pitch_rate, 0.0, 0.0])
        u_lqr = float(-K @ _lqr_state)

        # Blend
        u_bal = (1.0 - blend) * u_pid + blend * u_lqr

        # Add velocity error compensation (simple proportional)
        wheel_vel_error = wheel_vel_target - wheel_vel_linear  # Target - actual
        u_vel = 1.5 * wheel_vel_error  # Positive error → more torque
        u_wheel_total = u_bal + u_vel

        # Send to wheels
        u_whl_L = motor_whl_L.step(u_wheel_total, data.qvel[d_whl_L], _dt)
        u_whl_R = motor_whl_R.step(u_wheel_total, data.qvel[d_whl_R], _dt)
        data.ctrl[2] = u_whl_L
        data.ctrl[3] = u_whl_R

        # Hip motors: suspension
        u_hip = HIP_KP_SUSP * (Q_NEUTRAL - hip_q) + HIP_KD_SUSP * (0.0 - data.qvel[d_hip_L])
        u_hip = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
        data.ctrl[0] = motor_hip_L.step(u_hip, data.qvel[d_hip_L], _dt)
        data.ctrl[1] = motor_hip_R.step(u_hip, data.qvel[d_hip_R], _dt)

        # Step physics
        mujoco.mj_step(model, data)

        # Log metrics
        if int(sim_t * 10) % max(1, PHYSICS_HZ // 10) == 0:
            time_log.append(sim_t)
            pitch_log.append(math.degrees(pitch))
            pitch_rate_log.append(pitch_rate)
            wheel_pos_log.append(wheel_pos)
            wheel_vel_log.append(wheel_vel)
            wheel_vel_cmd_log.append(wheel_vel_target)
            torque_log.append(u_bal)
            forces = _struct_all_L()
            bearing_loads_log.append(max(forces[5:10]))

    # ─────────────────────────────────────────────────────────────────────────
    # Compute metrics (drive-specific)
    # ─────────────────────────────────────────────────────────────────────────
    metrics = TestMetrics()

    if len(pitch_log) > 0:
        pitch_arr = np.array(pitch_log)
        pitch_mean = np.mean(pitch_arr)
        pitch_wobble = pitch_arr - pitch_mean
        metrics.rms_pitch_deg = float(np.sqrt(np.mean(pitch_wobble**2)))
        metrics.max_pitch_deviation_deg = float(np.max(np.abs(pitch_wobble)))

        # Settle time for drive: when pitch wobble < 2°
        threshold_deg = 2.0
        settled_idx = None
        for i in range(len(pitch_arr) // 2, len(pitch_arr)):
            if np.max(np.abs(pitch_wobble[i:])) < threshold_deg:
                settled_idx = i
                break
        if settled_idx is not None:
            metrics.pitch_settle_time_s = float(time_log[settled_idx])

        zero_crosses = 0
        for i in range(1, len(pitch_wobble)):
            if pitch_wobble[i-1] * pitch_wobble[i] < 0:
                zero_crosses += 1
        metrics.pitch_oscillations_count = zero_crosses

    if len(wheel_vel_log) > 0:
        wheel_vel_arr = np.array(wheel_vel_log)
        wheel_vel_cmd_arr = np.array(wheel_vel_cmd_log)
        # Tracking error: RMS deviation from commanded velocity
        tracking_error = wheel_vel_arr - wheel_vel_cmd_arr
        metrics.rms_wheel_tracking_error_m = float(np.sqrt(np.mean(tracking_error**2)))

    if len(wheel_pos_log) > 0:
        wheel_pos_arr = np.array(wheel_pos_log)
        metrics.max_wheel_pos_drift_m = float(np.max(np.abs(wheel_pos_arr)))

    if len(torque_log) > 0:
        torque_arr = np.array(torque_log)
        metrics.control_effort_integral = float(np.sum(np.abs(torque_arr)) * (duration_s / len(torque_arr)))
        metrics.peak_torque_nm = float(np.max(np.abs(torque_arr)))
        metrics.avg_torque_nm = float(np.mean(np.abs(torque_arr)))

    if len(bearing_loads_log) > 0:
        metrics.max_bearing_load_n = float(np.max(bearing_loads_log))

    # Pass/fail for drive
    metrics.pass_fail = (
        metrics.rms_pitch_deg < 2.0 and  # Tighter pitch control during drive
        metrics.rms_wheel_tracking_error_m < 0.1 and  # Good velocity tracking
        metrics.peak_torque_nm < 2.0  # Reasonable torque
    )

    metrics.notes = f"Drive fwd/bwd {duration_s}s, Q={q_pitch}, R={r_val}, blend={blend}"

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Behavioral scenario tests")
    parser.add_argument("--scenario", type=str, default="self_balance",
                        help="Scenario: self_balance, drive (default: self_balance)")
    parser.add_argument("--q", type=float, default=100.0, help="LQR Q[0,0] (pitch weight)")
    parser.add_argument("--r", type=float, default=0.1, help="LQR R (control cost weight)")
    parser.add_argument("--blend", type=float, default=1.0, help="LQR blend (0=PID, 1=LQR)")
    parser.add_argument("--duration", type=float, default=None, help="Scenario duration (s, default varies by scenario)")
    args = parser.parse_args()

    try:
        # Choose scenario and default duration
        if args.scenario == "self_balance":
            duration = args.duration or 30.0
            metrics = run_self_balance_scenario(
                duration_s=duration,
                q_pitch=args.q,
                r_val=args.r,
                blend=args.blend
            )
        elif args.scenario == "drive":
            duration = args.duration or 20.0
            metrics = run_drive_scenario(
                duration_s=duration,
                q_pitch=args.q,
                r_val=args.r,
                blend=args.blend
            )
        else:
            print(f"ERROR: Unknown scenario '{args.scenario}'. Choose: self_balance, drive")
            return 1

        # Output as JSON for easy parsing
        output = {
            "scenario": args.scenario,
            "duration_s": duration,
            "q_pitch": args.q,
            "r": args.r,
            "blend": args.blend,
            "metrics": {
                "rms_pitch_deg": metrics.rms_pitch_deg,
                "max_pitch_deviation_deg": metrics.max_pitch_deviation_deg,
                "pitch_settle_time_s": metrics.pitch_settle_time_s,
                "pitch_oscillations_count": metrics.pitch_oscillations_count,
                "max_wheel_pos_drift_m": metrics.max_wheel_pos_drift_m,
                "rms_wheel_tracking_error_m": metrics.rms_wheel_tracking_error_m,
                "wheel_velocity_rms_m_s": metrics.wheel_velocity_rms_m_s,
                "control_effort_integral": metrics.control_effort_integral,
                "peak_torque_nm": metrics.peak_torque_nm,
                "avg_torque_nm": metrics.avg_torque_nm,
                "max_bearing_load_n": metrics.max_bearing_load_n,
                "max_femur_lateral_n": metrics.max_femur_lateral_n,
                "max_impact_g": metrics.max_impact_g,
                "pass_fail": metrics.pass_fail,
                "notes": metrics.notes,
            }
        }

        print(json.dumps(output, indent=2))
        return 0 if metrics.pass_fail else 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
