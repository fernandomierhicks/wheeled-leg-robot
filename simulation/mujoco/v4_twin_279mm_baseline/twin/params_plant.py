"""Plant-only parameters identified by the DigitalTwin.md bench program.

Control gains never live here; they are generated from the firmware schema in
``params_control.py``. Values listed in ``PROVISIONAL_FIELDS`` are deliberate
placeholders and must be replaced by the named physical tests before claiming
sim-to-real fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random


@dataclass(frozen=True)
class PlantParams:
    # Catalog/CAD prior from RobotGeometry; replace with T0.1 scale data.
    body_mass_kg: float = 1.769488
    body_inertia_axle_kgm2: float = 0.0650
    cg_x_m: float = 0.0250
    cg_z_m: float = 0.1800
    wheel_mass_kg: float = 0.520
    # Geometric placeholder only: the motor's stationary/rotating mass split is
    # unknown, so T1.3 must replace this despite the improved translating mass.
    wheel_inertia_kgm2: float = 0.5 * 0.520 * 0.056**2
    wheel_radius_m: float = 0.056
    wheel_kt_nm_per_a: float = 9.55 / 70.0
    wheel_torque_scale: float = 1.0
    wheel_viscous_nm_per_rad_s: float = 0.001
    wheel_coulomb_nm: float = 0.002
    ground_rolling_nm: float = 0.01
    ground_friction: float = 0.8
    yaw_inertia_kgm2: float = 0.018
    sensor_delay_s: float = 0.002
    actuator_delay_s: float = 0.001
    actuator_time_constant_s: float = 0.001
    pitch_noise_std_rad: float = 0.000176
    pitch_rate_noise_std_rad_s: float = 0.002116
    roll_noise_std_rad: float = 0.000156
    accel_noise_std_ms2: float = 0.008
    hip_torque_limit_nm: float = 8.0
    hip_speed_limit_rad_s: float = 20.0
    hip_phase_resistance_ohm: float = 2.19
    hip_kt_output_nm_per_a: float = 1.27
    hip_gearbox_efficiency: float = 0.85
    hip_backlash_rad: float = 0.0
    hip_torque_quantization_nm: float = 16.0 / 4095.0
    tick_jitter_std_s: float = 0.0

    @property
    def total_loop_delay_s(self) -> float:
        return self.sensor_delay_s + self.actuator_delay_s


PROVISIONAL_FIELDS = {
    "body_mass_kg": "T0.1 mass inventory",
    "wheel_mass_kg": "T0.1 removable wheel-assembly mass",
    "body_inertia_axle_kgm2": "T1.1 locked-wheel pendulum swing",
    "cg_x_m": "T0.2 two-scale CG measurement",
    "cg_z_m": "T0.2 side-on CG measurement",
    "wheel_inertia_kgm2": "T1.3 free spin-up/coast-down",
    "wheel_torque_scale": "T2.1 blocked-wheel lever/scale",
    "wheel_viscous_nm_per_rad_s": "T1.3 coast-down fit",
    "wheel_coulomb_nm": "T1.3 coast-down fit",
    "ground_rolling_nm": "T4.5 rolling coast-down",
    "ground_friction": "T4.4 yaw-rate step",
    "sensor_delay_s": "T3.1 loop-delay chirp",
    "actuator_delay_s": "T2.2 wheel torque bandwidth",
    "actuator_time_constant_s": "T2.2 wheel torque bandwidth",
    "pitch_noise_std_rad": "T3.2 assembled/energized IMU noise",
    "pitch_rate_noise_std_rad_s": "T3.2 assembled/energized IMU noise",
    "roll_noise_std_rad": "T3.2 assembled/energized IMU noise",
    "accel_noise_std_ms2": "T3.2 assembled/energized IMU noise",
    "hip_gearbox_efficiency": "T2.4/T2.5 linkage load and bus-current fit",
    "tick_jitter_std_s": "T3.3 WLOG t_micros histogram",
}


def plant_ensemble(n: int = 5, seed: int = 1,
                   nominal: PlantParams | None = None) -> tuple[PlantParams, ...]:
    """Deterministic robust-design ensemble around the provisional plant."""
    if n < 1:
        raise ValueError("ensemble size must be positive")
    base = nominal or PlantParams()
    rng = random.Random(seed)
    plants = [base]
    for _ in range(n - 1):
        plants.append(replace(
            base,
            body_inertia_axle_kgm2=base.body_inertia_axle_kgm2 * rng.uniform(0.8, 1.2),
            sensor_delay_s=max(0.0, base.sensor_delay_s + rng.uniform(-0.001, 0.001)),
            actuator_delay_s=max(0.0, base.actuator_delay_s + rng.uniform(-0.001, 0.001)),
            wheel_torque_scale=base.wheel_torque_scale * rng.uniform(0.85, 1.15),
            cg_x_m=base.cg_x_m * rng.uniform(0.9, 1.1),
        ))
    return tuple(plants)
