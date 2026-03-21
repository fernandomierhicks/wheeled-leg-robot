"""base.py — ScenarioConfig + WorldConfig dataclasses.

ScenarioConfig is pure data — no inheritance, no abstract methods.
The sim_loop reads its fields to decide which controllers are active,
which profiles to call, and what world geometry to build.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, FrozenSet


@dataclass(frozen=True)
class WorldConfig:
    """World geometry for MJCF builder (obstacles, bumps, floor size)."""
    obstacle_height: float = 0.0
    bumps: tuple = ()              # Tuple of S5Bump
    sandbox_obstacles: tuple = ()  # Tuple of S8Bump
    prop_bodies: tuple = ()        # Tuple of prop dicts (free-body objects)
    floor_size: tuple = (20.0, 20.0, 0.1)


@dataclass(frozen=True)
class ScenarioConfig:
    """Declarative scenario definition — composition over inheritance.

    The sim_loop reads these fields each tick to decide behaviour.
    Fitness is computed *after* the loop finishes via fitness_fn(metrics).

    Parameters
    ----------
    name           : machine key, e.g. "s01_lqr_pitch_step"
    display_name   : human-readable label for plots/logs
    duration       : simulation time [s]
    active_controllers : which controllers are ON
        - "lqr"         : LQR balance (always on for balance scenarios)
        - "velocity_pi" : outer velocity PI loop
        - "yaw_pi"      : differential yaw PI
    hip_mode       : "position" (stiff PD, S1–S7) or "impedance" (S8)
    v_profile      : callable(t) -> v_desired [m/s], or None (= 0.0)
    omega_profile  : callable(t) -> omega_desired [rad/s], or None (= 0.0)
    hip_profile    : callable(t) -> q_hip_target [rad], or None (= Q_NOM)
    dist_fn        : callable(t) -> force_x [N], or None (= 0.0)
    world          : WorldConfig for MJCF builder
    fitness_fn     : callable(metrics_dict) -> float
    group          : optimizer group: "lqr" | "velocity_pi" | "yaw_pi" | "suspension"
    order          : sort key for pipeline sequencing (1.0, 2.0, 2.1, ...)
    init_fn        : callable(model, data, params) -> None, or None
                     Called once after init_sim() to apply scenario-specific ICs
                     (e.g., pitch step perturbation for S1)
    """
    name: str
    display_name: str
    duration: float
    active_controllers: FrozenSet[str] = frozenset({"lqr"})
    hip_mode: str = "position"
    v_profile: Optional[Callable] = None
    omega_profile: Optional[Callable] = None
    hip_profile: Optional[Callable] = None
    hip_vel_profile: Optional[Callable] = None
    dist_fn: Optional[Callable] = None
    world: WorldConfig = field(default_factory=WorldConfig)
    fitness_fn: Optional[Callable] = None
    group: str = "lqr"
    order: float = 0.0
    init_fn: Optional[Callable] = None

    # Fitness weights / constants used by fitness_fn — stored here so
    # the scenario definition is fully self-contained.
    W_FALL: float = 200.0
    W_RMS: float = 1.0
    W_VEL_ERR: float = 3.0
    W_YAW_ERR: float = 3.0
    W_LIFTOFF: float = 50.0
    W_PITCH_RATE: float = 0.05
    W_SETTLE: float = 0.01
