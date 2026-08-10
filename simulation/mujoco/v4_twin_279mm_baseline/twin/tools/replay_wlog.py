"""Replay a hardware/twin WLOG open-loop or through firmware-equivalent control."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

from ..firmware_control import ControlInput, FirmwareController
from ..params_control import PARAMS_BY_NAME, default_values
from ..params_plant import PlantParams
from ..runtime import AnalyticalPlant


ROOT = Path(__file__).resolve().parents[5]
GUI_ROOT = ROOT / "software" / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))
from analysis.param_sidecar import load_matching_sidecar  # noqa: E402
from analysis.wlog_metrics import decode_wlog  # noqa: E402


def _nrmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = float(np.sqrt(np.mean(np.square(estimate - reference))))
    scale = float(np.percentile(reference, 95) - np.percentile(reference, 5))
    if scale < 1e-9:
        scale = max(float(np.sqrt(np.mean(np.square(reference)))), 1.0)
    return error / scale


def replay(path: Path, *, mode: str = "closed", plant_params: PlantParams | None = None,
           output_csv: Path | None = None) -> dict:
    run = decode_wlog(path)
    fields = run.fields
    sidecar = load_matching_sidecar(path)
    params = default_values()
    if sidecar:
        for name in sidecar.names:
            if name in PARAMS_BY_NAME:
                value = sidecar.initial_value(name)
                if value is not None:
                    params[name] = value
    controller = FirmwareController(params=params)
    alpha0 = (float(fields["gain_sched_alpha"][0])
              if run.has_gain_sched_alpha else 0.0)
    model = AnalyticalPlant(
        plant_params or PlantParams(),
        {"pitch_rad": float(fields["pitch_rad"][0]),
         "pitch_rate_rads": float(fields["pitch_rate_rads"][0]),
         "wheel_vel_ms": float(fields["wheel_vel_avg"][0]),
         "yaw_rad": float(fields["yaw_rad"][0]),
         "yaw_rate_rads": float(fields["yaw_rate_rads"][0]),
         "hip_alpha": alpha0},
        seed=0,
    )
    param_series = {}
    if sidecar:
        for name in sidecar.names:
            if name in PARAMS_BY_NAME and "command" not in PARAMS_BY_NAME[name].flags:
                series = sidecar.series(name, run.t_micros)
                if series is not None:
                    param_series[name] = series

    predicted = {name: [] for name in ("pitch_rad", "pitch_rate_rads",
                                        "wheel_vel_avg", "yaw_rate_rads", "tau_sym")}
    last_params: dict[str, float] = {}
    wheel_radius = model.params.wheel_radius_m
    for index, time_s in enumerate(run.t_s):
        state = model.state
        predicted["pitch_rad"].append(state.pitch_rad)
        predicted["pitch_rate_rads"].append(state.pitch_rate_rads)
        predicted["wheel_vel_avg"].append(state.wheel_vel_ms)
        predicted["yaw_rate_rads"].append(state.yaw_rate_rads)

        dt = (float(run.t_s[index + 1] - time_s)
              if index + 1 < run.count else 1.0 / run.sample_rate_hz)
        dt = min(0.02, max(0.0002, dt))
        for name, series in param_series.items():
            value = float(series[index])
            if last_params.get(name) != value:
                controller.set_param(name, value)
                last_params[name] = value
        controller.params["v_cmd_ms"] = float(fields["v_ref"][index])
        controller.params["omega_cmd_rds"] = float(fields["omega_cmd_rds"][index])

        if mode == "closed":
            pitch, pitch_rate, yaw_rate, alpha = model.sensor()
            turns_s = state.wheel_vel_ms / (2.0 * math.pi * wheel_radius)
            output = controller.step(ControlInput(
                time_s=float(time_s), pitch_rad=pitch, pitch_rate_rads=pitch_rate,
                yaw_rate_rads=yaw_rate, wheel_l_turns_s=turns_s,
                wheel_r_turns_s=turns_s, hip_alpha=alpha,
                hip_l_torque_nm=state.hip_l_torque_nm,
                hip_r_torque_nm=state.hip_r_torque_nm,
                state="RUNNING" if int(fields["robot_state"][index]) == 3 else "ESTOP",
            ))
            tau_l, tau_r = output.tau_l, output.tau_r
            hip_alpha, hip_kp = output.hip_cmd_alpha, output.hip_kp
            predicted["tau_sym"].append(output.tau_sym)
        else:
            tau_l = float(fields["whl_tau_l"][index])
            tau_r = float(fields["whl_tau_r"][index])
            hip_alpha = (float(fields["gain_sched_alpha"][index])
                         if run.has_gain_sched_alpha else alpha0)
            hip_kp = float(fields["hip_l_cmd_kp"][index]) if "hip_l_cmd_kp" in fields else 0.0
            predicted["tau_sym"].append(0.5 * (tau_l + tau_r))
        model.apply(tau_l, tau_r, hip_alpha, hip_kp, dt)

    arrays = {name: np.asarray(values) for name, values in predicted.items()}
    reference_tau = fields["tau_sym"]
    scores = {
        "pitch_rad": _nrmse(fields["pitch_rad"], arrays["pitch_rad"]),
        "pitch_rate_rads": _nrmse(fields["pitch_rate_rads"], arrays["pitch_rate_rads"]),
        "wheel_vel_avg": _nrmse(fields["wheel_vel_avg"], arrays["wheel_vel_avg"]),
        "yaw_rate_rads": _nrmse(fields["yaw_rate_rads"], arrays["yaw_rate_rads"]),
        "tau_sym": _nrmse(reference_tau, arrays["tau_sym"]),
    }
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            columns = tuple(arrays)
            writer.writerow(["time_s", *(f"recorded_{name}" for name in columns),
                             *(f"sim_{name}" for name in columns)])
            references = {**fields, "tau_sym": reference_tau}
            for index, time_s in enumerate(run.t_s):
                writer.writerow([time_s, *(references[name][index] for name in columns),
                                 *(arrays[name][index] for name in columns)])
    return {"ok": True, "mode": mode, "wlog": str(path), "samples": run.count,
            "nrmse": scores, "worst_nrmse": max(scores.values()),
            "plant_is_provisional": True,
            **({"overlay_csv": str(output_csv)} if output_csv else {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wlog", type=Path)
    parser.add_argument("--mode", choices=("open", "closed"), default="closed")
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.wlog, mode=args.mode, output_csv=args.output_csv), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
