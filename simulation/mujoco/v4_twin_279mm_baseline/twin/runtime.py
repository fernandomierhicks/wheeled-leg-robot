"""Deterministic offline plant/scenario runtime for the v4 twin framework.

This low-order plant is intentionally parameter-identification friendly. It is
the headless path used for WLOG/replay/optimizer tests while the package's
MuJoCo model supplies full linkage/contact visualization. Provisional dynamics
are explicit in params_plant.py and are replaced test-by-test tomorrow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from ..robot_match import load_latest_firmware_params
from .firmware_control import ControlInput, FirmwareController, L_EFF_EXT, L_EFF_RET
from .params_control import default_values
from .params_plant import PlantParams
from .scenario import Scenario
from .wlog import WlogWriter


@dataclass
class PlantState:
    pitch_rad: float
    pitch_rate_rads: float = 0.0
    wheel_vel_ms: float = 0.0
    wheel_pos_m: float = 0.0
    yaw_rad: float = 0.0
    yaw_rate_rads: float = 0.0
    hip_alpha: float = 0.0
    hip_l_torque_nm: float = 0.0
    hip_r_torque_nm: float = 0.0
    tau_l_actual: float = 0.0
    tau_r_actual: float = 0.0


class AnalyticalPlant:
    def __init__(self, params: PlantParams, initial: dict[str, float] | None = None,
                 seed: int = 1):
        self.params = params
        initial = dict(initial or {})
        alpha = initial.pop("hip_alpha", 0.0)
        equilibrium = -math.atan2(params.cg_x_m, max(1e-6, params.cg_z_m))
        self.state = PlantState(
            pitch_rad=initial.pop("pitch_rad", equilibrium), hip_alpha=alpha,
            **initial)
        self.rng = random.Random(seed)
        self.track_m = 0.286
        self._sensor_delay = max(0, round(params.sensor_delay_s / 0.002))
        self._actuator_delay = max(0, round(params.actuator_delay_s / 0.002))
        initial_sensor = self._noisy_sensor()
        self._sensor_q = deque([initial_sensor] * (self._sensor_delay + 1),
                               maxlen=self._sensor_delay + 1)
        self._actuator_q = deque([(0.0, 0.0)] * (self._actuator_delay + 1),
                                 maxlen=self._actuator_delay + 1)

    def _noisy_sensor(self) -> tuple[float, float, float, float]:
        p, s = self.params, self.state
        return (
            s.pitch_rad + self.rng.gauss(0.0, p.pitch_noise_std_rad),
            s.pitch_rate_rads + self.rng.gauss(0.0, p.pitch_rate_noise_std_rad_s),
            s.yaw_rate_rads,
            s.hip_alpha,
        )

    def sensor(self) -> tuple[float, float, float, float]:
        self._sensor_q.append(self._noisy_sensor())
        return self._sensor_q[0]

    def apply(self, tau_l_cmd: float, tau_r_cmd: float, hip_cmd_alpha: float,
              hip_kp: float, dt: float) -> None:
        p, s = self.params, self.state
        self._actuator_q.append((tau_l_cmd, tau_r_cmd))
        delayed_l, delayed_r = self._actuator_q[0]
        lag = 1.0 if p.actuator_time_constant_s <= 0.0 else min(1.0, dt / p.actuator_time_constant_s)
        s.tau_l_actual += lag * (delayed_l * p.wheel_torque_scale - s.tau_l_actual)
        s.tau_r_actual += lag * (delayed_r * p.wheel_torque_scale - s.tau_r_actual)

        l_eff = L_EFF_RET + s.hip_alpha * (L_EFF_EXT - L_EFF_RET)
        equilibrium = -math.atan2(p.cg_x_m, max(1e-6, p.cg_z_m))
        tau_avg = 0.5 * (s.tau_l_actual + s.tau_r_actual)
        gravity_accel = 9.81 / max(0.05, l_eff) * math.sin(s.pitch_rad - equilibrium)
        pitch_ddot = gravity_accel - 2.0 * tau_avg / max(1e-5, p.body_inertia_axle_kgm2)
        s.pitch_rate_rads += pitch_ddot * dt
        s.pitch_rad += s.pitch_rate_rads * dt

        drive_force = (s.tau_l_actual + s.tau_r_actual) / p.wheel_radius_m
        rolling = p.ground_rolling_nm / p.wheel_radius_m
        if abs(s.wheel_vel_ms) > 1e-5:
            drive_force -= math.copysign(rolling, s.wheel_vel_ms)
        wheel_accel = drive_force / max(0.1, p.body_mass_kg + 2.0 * p.wheel_mass_kg)
        s.wheel_vel_ms += wheel_accel * dt
        s.wheel_pos_m += s.wheel_vel_ms * dt

        yaw_moment = (s.tau_r_actual - s.tau_l_actual) * self.track_m / (2.0 * p.wheel_radius_m)
        s.yaw_rate_rads += yaw_moment / max(1e-5, p.yaw_inertia_kgm2) * dt
        s.yaw_rad += s.yaw_rate_rads * dt

        max_alpha_rate = p.hip_speed_limit_rad_s / math.radians(80.0)
        alpha_step = max_alpha_rate * dt
        s.hip_alpha += max(-alpha_step, min(alpha_step, hip_cmd_alpha - s.hip_alpha))
        span = math.radians(80.0)
        hip_error_rad = (hip_cmd_alpha - s.hip_alpha) * span
        hip_torque = max(-p.hip_torque_limit_nm,
                         min(p.hip_torque_limit_nm, hip_kp * hip_error_rad))
        s.hip_l_torque_nm = hip_torque
        s.hip_r_torque_nm = hip_torque


def run_scenario(scenario: Scenario, output: Path, *, plant: PlantParams | None = None,
                 control_overrides: dict[str, float] | None = None,
                 seed: int = 1) -> dict:
    plant_params = plant or PlantParams()
    params = default_values()
    params.update(load_latest_firmware_params())
    params.update(scenario.initial_params)
    if control_overrides:
        params.update(control_overrides)
    controller = FirmwareController(params=params)
    model = AnalyticalPlant(plant_params, scenario.initial_state, seed=seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    events = iter(scenario.events)
    next_event = next(events, None)
    t = 0.0
    loop_count = 0
    robot_state = 3
    fault_code = 0
    dt_nom = 0.002
    with WlogWriter(output, controller.params) as writer:
        while t <= scenario.duration_s + 1e-12:
            while next_event is not None and next_event.time_s <= t + 1e-12:
                controller.update_params(next_event.params)
                for name, value in next_event.params.items():
                    writer.param_change(round(t * 1e6), name, value)
                next_event = next(events, None)

            pitch, pitch_rate, yaw_rate, alpha = model.sensor()
            s = model.state
            turns_s = s.wheel_vel_ms / (2.0 * math.pi * plant_params.wheel_radius_m)
            result = controller.step(ControlInput(
                time_s=t, pitch_rad=pitch, pitch_rate_rads=pitch_rate,
                yaw_rate_rads=yaw_rate, wheel_l_turns_s=turns_s,
                wheel_r_turns_s=turns_s, hip_alpha=alpha,
                hip_l_torque_nm=s.hip_l_torque_nm,
                hip_r_torque_nm=s.hip_r_torque_nm,
                state="RUNNING" if robot_state == 3 else "ESTOP"))
            if result.fault:
                robot_state = 4
                fault_code = {"PITCH_WATCHDOG": 8, "WHEEL_RUNAWAY": 9,
                              "ROLL_WATCHDOG": 14}[result.fault]

            dt = max(0.0005, dt_nom + model.rng.gauss(0.0, plant_params.tick_jitter_std_s))
            model.apply(result.tau_l, result.tau_r, result.hip_cmd_alpha, result.hip_kp, dt)
            writer.write(round(t * 1e6), {
                "timestamp_ms": round(t * 1000), "pitch_rad": s.pitch_rad,
                "pitch_rate_rads": s.pitch_rate_rads,
                "wheel_vel_avg": s.wheel_vel_ms,
                "hip_l_pos_rad": result.hip_l_setpoint_rad,
                "hip_r_pos_rad": result.hip_r_setpoint_rad,
                "whl_tau_l": result.tau_l, "whl_tau_r": result.tau_r,
                "roll_rad": 0.0, "yaw_rad": s.yaw_rad,
                "robot_state": robot_state, "fault_code": fault_code,
                "hip_l_torque_nm": s.hip_l_torque_nm,
                "hip_r_torque_nm": s.hip_r_torque_nm,
                "wm_l_vel_turns_s": turns_s, "wm_r_vel_turns_s": turns_s,
                "wm_l_pos_turns": s.wheel_pos_m / (2.0 * math.pi * plant_params.wheel_radius_m),
                "wm_r_pos_turns": s.wheel_pos_m / (2.0 * math.pi * plant_params.wheel_radius_m),
                "yaw_rate_rads": s.yaw_rate_rads,
                "hip_l_cmd_pos_rad": result.hip_l_setpoint_rad,
                "hip_r_cmd_pos_rad": result.hip_r_setpoint_rad,
                "hip_l_cmd_kp": result.hip_kp, "hip_r_cmd_kp": result.hip_kp,
                "hip_l_cmd_kd": result.hip_kd, "hip_r_cmd_kd": result.hip_kd,
                "hip_l_cmd_tff": result.hip_tff, "hip_r_cmd_tff": result.hip_tff,
                "theta_ref": result.theta_ref, "v_ref": controller.params["v_cmd_ms"],
                "omega_cmd_rds": controller.params["omega_cmd_rds"],
                "tau_sym": result.tau_sym, "tau_yaw": result.tau_yaw,
                "ff1_out": result.ff1_out, "ff2_out": result.ff2_out,
                "loop_count": loop_count, "pitch_trim_rad": result.pitch_trim,
                "gain_sched_alpha": result.gain_sched_alpha,
            })
            loop_count += 1
            t += dt

    return {"wlog": str(output), "params": str(output.with_suffix('.PARAMS')),
            "samples": loop_count, "fault_code": fault_code}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run a shared scenario in the offline twin")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--params", type=Path,
                        help="GUI-compatible parameter-export JSON to apply")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    scenario = Scenario.load(args.scenario)
    output = args.output or Path("data/twin") / f"{scenario.name}.WLOG"
    overrides = None
    if args.params:
        from .tools.param_snapshot import load_snapshot
        overrides = load_snapshot(args.params)
    print(json.dumps(run_scenario(scenario, output, control_overrides=overrides,
                                  seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
