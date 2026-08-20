"""Python transliteration of the Teensy balance/drive/yaw control path.

The equations and ordering mirror ``teensy/src/control_loop.cpp``: scheduled
limits and trim, velocity PI anti-windup, yaw PI, direct-gain LQR, backward
velocity guard/barrier, plant-ID chirp, clamps, then FF1/FF2 and wheel mixing.
The class owns only controller state; the MuJoCo/analytical plant owns physics,
delays, friction, actuator bandwidth, and noise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from v4_twin_279mm_baseline.params import RobotGeometry
from v4_twin_279mm_baseline.physics import alpha_to_hip_q

from .params_control import PARAMS_BY_NAME, default_values, validate_values


DT = 0.002
BACKWARD_TARGET_MARGIN_RAD = math.radians(3.0)
L_EFF_RET = 0.098915
L_EFF_EXT = 0.363396
M_BODY = 2.5620  # firmware value; 2026-08-09 scale inventory, battery in / wheels out
WHEEL_R = 0.056
MOTOR_TRQ_MAX = 7.0

GAIN_ALLOWLIST = frozenset({
    "lqr_k_pitch_ret", "lqr_k_pitch_ext", "lqr_k_rate_ret",
    "lqr_k_rate_ext", "lqr_k_vel", "lqr_pitch_trim_ret",
    "lqr_pitch_trim_ext", "lqr_barrier_k", "lqr_barrier_th_ret",
    "lqr_barrier_th_ext", "vel_pi_kp", "vel_pi_ki", "vel_pi_kff",
    "vel_pi_rate_lim", "vel_pi_int_max", "theta_max_fwd_ret",
    "theta_max_bwd_ret", "theta_max_fwd_ext", "theta_max_bwd_ext",
    "yaw_pi_kp", "yaw_pi_ki", "yaw_pi_torque_max", "yaw_pi_int_max",
    "roll_kp", "roll_kd", "roll_ki", "roll_int_max",
    "hip_running_kp", "hip_running_kd",
    "hip_running_tff_ret", "hip_running_tff_ext",
    "hip_roll_kp", "hip_roll_kd",
    "ff1_alpha", "ff2_alpha",
})


@dataclass(frozen=True)
class ControlInput:
    time_s: float
    pitch_rad: float = 0.0
    pitch_rate_rads: float = 0.0
    roll_rad: float = 0.0
    roll_rate_rads: float = 0.0
    yaw_rate_rads: float = 0.0
    wheel_l_turns_s: float = 0.0
    wheel_r_turns_s: float = 0.0
    hip_alpha: float = 0.0
    hip_l_torque_nm: float = 0.0
    hip_r_torque_nm: float = 0.0
    state: str = "RUNNING"
    velocity_offset_ms: float = 0.0
    jump_handoff_active: bool = False


@dataclass(frozen=True)
class ControlOutput:
    tau_sym: float = 0.0
    tau_yaw: float = 0.0
    tau_l: float = 0.0
    tau_r: float = 0.0
    theta_ref: float = 0.0
    theta_max_fwd: float = 0.0
    theta_max_bwd: float = 0.0
    pitch_trim: float = 0.0
    gain_sched_alpha: float = 0.0
    wheel_vel_avg_ms: float = 0.0
    ff1_out: float = 0.0
    ff2_out: float = 0.0
    hip_cmd_alpha: float = 0.0
    hip_l_setpoint_rad: float = 0.0
    hip_r_setpoint_rad: float = 0.0
    hip_kp: float = 0.0
    hip_kd: float = 0.0
    hip_tff: float = 0.0
    fault: str | None = None


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _schedule(ret: float, ext: float, alpha: float) -> float:
    return ret + alpha * (ext - ret)


def safe_backward_theta_limit(configured_limit: float, watchdog_bwd: float,
                              pitch_trim: float, margin: float) -> float:
    safe_limit = max(0.0, watchdog_bwd + pitch_trim - margin)
    return min(configured_limit, safe_limit)


def backward_velocity_term_guard(velocity_term: float, pitch: float,
                                 barrier_threshold: float,
                                 watchdog_bwd: float) -> float:
    if velocity_term >= 0.0 or pitch >= -barrier_threshold:
        return velocity_term
    span = watchdog_bwd - barrier_threshold
    if span <= 0.0:
        return 0.0
    fade = _clamp((watchdog_bwd + pitch) / span, 0.0, 1.0)
    return velocity_term * fade


def conditional_integral_step(integral: float, error: float, dt: float,
                              integral_max: float, ki: float,
                              non_integral: float, limit_fwd: float,
                              limit_bwd: float) -> float:
    candidate = _clamp(integral + error * dt, -integral_max, integral_max)
    integral_delta = ki * (candidate - integral)
    candidate_output = non_integral + ki * candidate
    unwinding = abs(candidate) < abs(integral)
    pushes_fwd = candidate_output > limit_fwd and integral_delta > 0.0
    pushes_bwd = candidate_output < -limit_bwd and integral_delta < 0.0
    return integral if (not unwinding and (pushes_fwd or pushes_bwd)) else candidate


class FirmwareController:
    """Stateful 500 Hz firmware-equivalent controller."""

    def __init__(self, params: Mapping[str, float] | None = None,
                 robot: RobotGeometry | None = None):
        self.params = default_values()
        if params:
            self.update_params(params)
        self.robot = robot or RobotGeometry()
        self._state = "STANDBY"
        self.reset(0.0, hip_alpha=0.0)

    def update_params(self, updates: Mapping[str, float]) -> None:
        values = {name: float(value) for name, value in updates.items()}
        validate_values(values)
        self.params.update(values)

    def set_param(self, name: str, value: float) -> None:
        self.update_params({name: value})

    def reset(self, time_s: float = 0.0, hip_alpha: float = 0.0) -> None:
        self.vel_integral = 0.0
        self.prev_v_desired = 0.0
        self.theta_ref_rlt = 0.0
        self.yaw_integral = 0.0
        self.roll_sp_rlt = 0.0
        self.roll_integral = 0.0
        self.hip_cmd_rlt = _clamp(hip_alpha, 0.0, 1.0)
        self.arm_time_s = float(time_s)
        self.pitch_fault_start_s: float | None = None
        self.roll_fault_start_s: float | None = None
        self.wheel_fault_start_s: float | None = None
        self.plant_id_start_s: float | None = None

    def enter_running(self, time_s: float, hip_alpha: float,
                      *, ramp_complete: bool = False) -> None:
        """Enter RUNNING explicitly, optionally after the stand-up ramp.

        A normal robot arm reaches RUNNING through STANDING_UP, where hip
        feedforward has already ramped in.  Starting a simulation directly in
        RUNNING should be able to represent that state without spending its
        first second with an artificial zero-feedforward hip transient.
        """
        self.reset(time_s, hip_alpha)
        if ramp_complete:
            self.arm_time_s -= max(0.0, self.params["hip_running_ramp_s"])
        self._state = "RUNNING"

    def _on_state(self, state: str, time_s: float, hip_alpha: float) -> None:
        state = state.upper()
        controlled = {"RUNNING", "JUMPING"}
        if state != self._state:
            if self._state in controlled:
                self.params["plant_id_en"] = 0.0
                self.plant_id_start_s = None
            # JUMPING -> RUNNING is a continuous controller handoff in the
            # firmware. Preserve the PI/rate-limit state instead of re-arming.
            if state == "RUNNING" and self._state not in controlled:
                self.reset(time_s, hip_alpha)
            self._state = state

    def _watchdogs(self, sample: ControlInput, pitch: float,
                   sched_fwd: float, sched_bwd: float,
                   soft_wheel_limit: float) -> str | None:
        now = sample.time_s
        if self.params["pitch_watchdog_en"] >= 0.5:
            if pitch > sched_fwd or pitch < -sched_bwd:
                self.pitch_fault_start_s = self.pitch_fault_start_s or now
                if now - self.pitch_fault_start_s > 0.200:
                    return "PITCH_WATCHDOG"
            else:
                self.pitch_fault_start_s = None
        if self.params["roll_watchdog_en"] >= 0.5:
            if abs(sample.roll_rad) > self.params["roll_watchdog_limit"]:
                self.roll_fault_start_s = self.roll_fault_start_s or now
                if now - self.roll_fault_start_s > 0.200:
                    return "ROLL_WATCHDOG"
            else:
                self.roll_fault_start_s = None
        runaway_enabled = self.params.get("wheel_runaway_en", 1.0) >= 0.5
        if runaway_enabled and (
                abs(sample.wheel_l_turns_s) > 2.0 * soft_wheel_limit or
                abs(sample.wheel_r_turns_s) > 2.0 * soft_wheel_limit):
            self.wheel_fault_start_s = self.wheel_fault_start_s or now
            if now - self.wheel_fault_start_s > 0.050:
                return "WHEEL_RUNAWAY"
        else:
            self.wheel_fault_start_s = None
        return None

    def step(self, sample: ControlInput) -> ControlOutput:
        p = self.params
        self._on_state(sample.state, sample.time_s, sample.hip_alpha)
        if sample.state.upper() not in {"RUNNING", "JUMPING"}:
            return ControlOutput(gain_sched_alpha=_clamp(sample.hip_alpha, 0.0, 1.0))

        alpha = 0.0 if p["alpha_force_ret_en"] >= 0.5 else _clamp(sample.hip_alpha, 0.0, 1.0)

        # Hip command slew and optional active-roll offset.
        hip_raw = _clamp(p["radio_hip_cmd"], 0.0, 1.0)
        rate = p["hip_cmd_rate_lim"]
        step = rate * DT
        self.hip_cmd_rlt += _clamp(hip_raw - self.hip_cmd_rlt, -step, step) if rate > 0.0 else hip_raw - self.hip_cmd_rlt
        pos_l = pos_r = alpha_to_hip_q(self.hip_cmd_rlt, self.robot)
        hip_kp = p["hip_running_kp"]
        hip_kd = p["hip_running_kd"]
        if ((sample.state.upper() == "RUNNING" or sample.jump_handoff_active)
                and p["roll_ctrl_en"] >= 0.5):
            dmax = p["roll_rate_lim"] * DT
            self.roll_sp_rlt += _clamp(p["roll_cmd_rad"] - self.roll_sp_rlt, -dmax, dmax)
            roll_err = self.roll_sp_rlt - sample.roll_rad
            offset_pd = p["roll_kp"] * roll_err - p["roll_kd"] * sample.roll_rate_rads
            stroke = abs(self.robot.Q_EXT - self.robot.Q_ALPHA_RET)
            off_lim = min(p["roll_offset_max"], min(self.hip_cmd_rlt, 1.0 - self.hip_cmd_rlt) * stroke)
            self.roll_integral = conditional_integral_step(
                self.roll_integral, roll_err, DT, p["roll_int_max"], p["roll_ki"],
                offset_pd, off_lim, off_lim)
            offset = _clamp(offset_pd + p["roll_ki"] * self.roll_integral, -off_lim, off_lim)
            pos_l -= offset
            pos_r += offset
            hip_kp = p["hip_roll_kp"]
            hip_kd = p["hip_roll_kd"]
        else:
            self.roll_sp_rlt = 0.0
            self.roll_integral = 0.0

        ramp_s = p["hip_running_ramp_s"]
        ramp_alpha = 1.0 if ramp_s <= 0.0 else _clamp((sample.time_s - self.arm_time_s) / ramp_s, 0.0, 1.0)
        hip_tff = ramp_alpha * _schedule(p["hip_running_tff_ret"], p["hip_running_tff_ext"], alpha)

        vel_avg_ms = 0.5 * (sample.wheel_l_turns_s + sample.wheel_r_turns_s) * 2.0 * math.pi * WHEEL_R
        pitch = p["sim_pitch_rad"] if p["enable_sim_pitch"] >= 0.5 else sample.pitch_rad
        pitch_rate = p["sim_pitch_rate"] if p["enable_sim_prate"] >= 0.5 else sample.pitch_rate_rads

        sched_fwd = _schedule(p["pitch_wd_fwd_ret"], p["pitch_wd_fwd_ext"], alpha)
        sched_bwd = _schedule(p["pitch_wd_bwd_ret"], p["pitch_wd_bwd_ext"], alpha)
        barrier_th = _schedule(p["lqr_barrier_th_ret"], p["lqr_barrier_th_ext"], alpha)
        pitch_trim = _schedule(p["lqr_pitch_trim_ret"], p["lqr_pitch_trim_ext"], alpha)
        soft_limit = p["wm_vel_limit"]
        if sample.jump_handoff_active and p["jmp_handoff_vel_lim"] > 0.0:
            soft_limit = p["jmp_handoff_vel_lim"]
        fault = self._watchdogs(sample, pitch, sched_fwd, sched_bwd, soft_limit)
        if fault:
            return ControlOutput(gain_sched_alpha=alpha, wheel_vel_avg_ms=vel_avg_ms,
                                 pitch_trim=pitch_trim, fault=fault)

        theta_ref = 0.0
        theta_max_fwd = 0.0
        theta_max_bwd = 0.0
        v_desired = p["v_cmd_ms"] + sample.velocity_offset_ms
        if p["vel_pi_en"] >= 0.5:
            v_err = v_desired - vel_avg_ms
            if v_desired * self.prev_v_desired < 0.0:
                self.vel_integral = 0.0
            theta_max_fwd = _schedule(p["theta_max_fwd_ret"], p["theta_max_fwd_ext"], alpha)
            theta_cfg_bwd = _schedule(p["theta_max_bwd_ret"], p["theta_max_bwd_ext"], alpha)
            theta_max_bwd = safe_backward_theta_limit(
                theta_cfg_bwd, sched_bwd, pitch_trim, BACKWARD_TARGET_MARGIN_RAD)
            non_integral = p["vel_pi_kp"] * v_err + p["vel_pi_kff"] * (v_desired - self.prev_v_desired) / DT
            self.vel_integral = conditional_integral_step(
                self.vel_integral, v_err, DT, p["vel_pi_int_max"], p["vel_pi_ki"],
                non_integral, theta_max_fwd, theta_max_bwd)
            raw = _clamp(non_integral + p["vel_pi_ki"] * self.vel_integral,
                         -theta_max_bwd, theta_max_fwd)
            dmax = p["vel_pi_rate_lim"] * DT
            self.theta_ref_rlt += _clamp(raw - self.theta_ref_rlt, -dmax, dmax)
            theta_ref = self.theta_ref_rlt
        else:
            self.vel_integral = 0.0
            self.theta_ref_rlt = 0.0
        self.prev_v_desired = v_desired

        tau_yaw = 0.0
        if p["yaw_pi_en"] >= 0.5:
            yaw_err = p["omega_cmd_rds"] - sample.yaw_rate_rads
            self.yaw_integral = _clamp(
                self.yaw_integral + yaw_err * DT,
                -p["yaw_pi_int_max"], p["yaw_pi_int_max"])
            tau_yaw = _clamp(
                p["yaw_pi_kp"] * yaw_err + p["yaw_pi_ki"] * self.yaw_integral,
                -p["yaw_pi_torque_max"], p["yaw_pi_torque_max"])
        else:
            self.yaw_integral = 0.0

        k_pitch = _schedule(p["lqr_k_pitch_ret"], p["lqr_k_pitch_ext"], alpha)
        k_rate = _schedule(p["lqr_k_rate_ret"], p["lqr_k_rate_ext"], alpha)
        if sample.jump_handoff_active:
            k_pitch *= p["jmp_handoff_kp_mul"]
            k_rate *= p["jmp_handoff_kr_mul"]
        v_ref = v_desired if p["vel_pi_en"] >= 0.5 else 0.0
        velocity_term = p["lqr_k_vel"] * (vel_avg_ms - v_ref)
        if sample.jump_handoff_active:
            velocity_term *= p["jmp_handoff_kv_mul"]
        velocity_term = backward_velocity_term_guard(
            velocity_term, pitch, barrier_th, sched_bwd)
        tau_sym = -(k_pitch * (pitch - theta_ref - pitch_trim)
                    + k_rate * pitch_rate + velocity_term)
        over = -pitch - barrier_th
        if over > 0.0:
            tau_sym -= p["lqr_barrier_k"] * over

        if p["plant_id_en"] >= 0.5:
            if self.plant_id_start_s is None:
                self.plant_id_start_s = sample.time_s
            elapsed = sample.time_s - self.plant_id_start_s
            duration = p["plant_id_dur"]
            if elapsed >= duration:
                p["plant_id_en"] = 0.0
                self.plant_id_start_s = None
            else:
                phase = 2.0 * math.pi * (
                    p["plant_id_f0"] * elapsed
                    + 0.5 * (p["plant_id_f1"] - p["plant_id_f0"])
                    * elapsed * elapsed / duration)
                tau_sym += p["plant_id_amp"] * math.sin(phase)
        else:
            self.plant_id_start_s = None

        torque_limit = p["lqr_torque_limit"]
        if sample.jump_handoff_active and p["jmp_handoff_torque"] > 0.0:
            torque_limit = p["jmp_handoff_torque"]
        tau_sym = _clamp(tau_sym, -torque_limit, torque_limit)
        l_eff = _schedule(L_EFF_RET, L_EFF_EXT, alpha)
        ff2 = p["ff2_alpha"] * M_BODY * 9.81 * l_eff * math.sin(pitch) if p["ff2_alpha"] > 0.0 else 0.0
        ff1 = (-p["ff1_alpha"] *
               (sample.hip_l_torque_nm + sample.hip_r_torque_nm) * p["ff1_kt_hip"] *
               (WHEEL_R / l_eff)) if p["ff1_alpha"] > 0.0 else 0.0

        tau_l = tau_r = 0.0
        if p["lqr_enable"] >= 0.5:
            tau_l = _clamp(tau_sym - tau_yaw + ff1 + ff2, -torque_limit, torque_limit)
            tau_r = _clamp(tau_sym + tau_yaw + ff1 + ff2, -torque_limit, torque_limit)
            tau_l = _clamp(tau_l, -MOTOR_TRQ_MAX, MOTOR_TRQ_MAX)
            tau_r = _clamp(tau_r, -MOTOR_TRQ_MAX, MOTOR_TRQ_MAX)
            if ((sample.wheel_l_turns_s > soft_limit and tau_l > 0.0) or
                    (sample.wheel_l_turns_s < -soft_limit and tau_l < 0.0)):
                tau_l = 0.0
            if ((sample.wheel_r_turns_s > soft_limit and tau_r > 0.0) or
                    (sample.wheel_r_turns_s < -soft_limit and tau_r < 0.0)):
                tau_r = 0.0

        return ControlOutput(
            tau_sym=tau_sym, tau_yaw=tau_yaw, tau_l=tau_l, tau_r=tau_r,
            theta_ref=theta_ref, theta_max_fwd=theta_max_fwd,
            theta_max_bwd=theta_max_bwd, pitch_trim=pitch_trim,
            gain_sched_alpha=alpha, wheel_vel_avg_ms=vel_avg_ms,
            ff1_out=ff1, ff2_out=ff2, hip_cmd_alpha=self.hip_cmd_rlt,
            hip_l_setpoint_rad=pos_l, hip_r_setpoint_rad=pos_r,
            hip_kp=ramp_alpha * hip_kp, hip_kd=hip_kd, hip_tff=hip_tff,
        )
