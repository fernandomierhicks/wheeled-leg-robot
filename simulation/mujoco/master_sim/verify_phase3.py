"""verify_phase3.py — Verify sim_loop.run() reproduces S1 results.

Runs S1 (LQR pitch step) headlessly through the unified sim_loop and
prints metrics for comparison with latency_sensitivity baseline.
"""
import sys
import os
import math

# Ensure master_sim parent is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from master_sim.defaults import DEFAULT_PARAMS
from master_sim.scenarios.base import ScenarioConfig
from master_sim.scenarios.profiles import s1_dist_fn
from master_sim.physics import get_equilibrium_pitch
from master_sim import sim_loop

import mujoco


def s1_init_fn(model, data, params):
    """Apply +5° pitch perturbation on top of equilibrium."""
    robot = params.robot
    pitch_step_rad = params.scenarios.pitch_step_rad

    def _jqp(name):
        return model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    s_root = _jqp("root_free")

    theta = get_equilibrium_pitch(robot, robot.Q_NOM) + pitch_step_rad
    data.qpos[s_root + 3] = math.cos(theta / 2.0)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2.0)
    data.qpos[s_root + 6] = 0.0
    mujoco.mj_forward(model, data)


def s1_fitness(metrics: dict) -> float:
    """S1 fitness: ISE pitch + W_PITCH_RATE * ISE pitch_rate + W_SETTLE * settle_time."""
    W_FALL = 200.0
    W_PITCH_RATE = 0.05
    W_SETTLE = 0.01
    return (metrics['ise_pitch']
            + W_PITCH_RATE * metrics['ise_pitch_rate']
            + W_SETTLE * metrics['settle_time_s']
            + (W_FALL if metrics['fell'] else 0.0))


# ── S1 scenario config ──────────────────────────────────────────────────────
S1_CONFIG = ScenarioConfig(
    name="s01_lqr_pitch_step",
    display_name="S1: LQR Pitch Step",
    duration=DEFAULT_PARAMS.scenarios.s1_duration,
    active_controllers=frozenset({"lqr"}),  # VelocityPI OFF, YawPI OFF
    hip_mode="position",
    dist_fn=s1_dist_fn,
    init_fn=s1_init_fn,
    fitness_fn=s1_fitness,
    group="lqr",
    order=1.0,
)


def main():
    print("Phase 3 Verification — sim_loop.run() with S1")
    print("=" * 70)

    P = DEFAULT_PARAMS
    print(f"\nLQR gains: Q=[{P.gains.lqr.Q_pitch}, {P.gains.lqr.Q_pitch_rate}, "
          f"{P.gains.lqr.Q_vel}], R={P.gains.lqr.R}")
    print(f"Latency: sensor={P.latency.sensor_delay_s}s, "
          f"actuator={P.latency.actuator_delay_s}s")
    print(f"Duration: {S1_CONFIG.duration}s")

    print("\nRunning S1 headlessly...")
    metrics = sim_loop.run(P, S1_CONFIG, rng_seed=42)

    print(f"\n{'Metric':<28} {'Value':>12}")
    print("-" * 42)
    for k, v in metrics.items():
        print(f"  {k:<26} {str(v):>12}")

    fitness = s1_fitness(metrics)
    print(f"\n  {'fitness':<26} {fitness:>12.6f}")

    print(f"\nStatus: {metrics['status']}")
    if metrics['status'] == 'PASS':
        print("[OK] S1 completed successfully — robot survived full duration")
    else:
        print(f"[WARN] S1 failed: {metrics['fail_reason']}")

    print("\n" + "=" * 70)
    print("Phase 3 verification complete.")


if __name__ == "__main__":
    main()
