"""Checks for the source-based v4 mass prior.

These are accounting tests, not claims about the as-built robot. T0.1 scale
measurements are still authoritative.
"""

import math

from v4_twin_279mm_baseline.params import RobotGeometry
from v4_twin_279mm_baseline.twin.params_plant import PlantParams, PROVISIONAL_FIELDS


def _tube_mass(od_m: float, wall_m: float, length_m: float) -> float:
    inner_m = od_m - 2.0 * wall_m
    return math.pi / 4.0 * (od_m**2 - inner_m**2) * length_m * 2700.0


def test_v4_tube_masses_follow_geometry_and_6061_density():
    robot = RobotGeometry()
    tibia_path_m = math.hypot(robot.L_tibia, robot.w_perp) + robot.L_stub

    assert math.isclose(robot.m_femur,
                        _tube_mass(0.014, 0.001, robot.L_femur), rel_tol=5e-5)
    assert math.isclose(robot.m_tibia,
                        _tube_mass(0.016, 0.001, tibia_path_m), rel_tol=5e-5)
    assert math.isclose(robot.m_coupler,
                        _tube_mass(0.010, 0.001, robot.Lc), rel_tol=5e-5)


def test_catalog_mass_accounting_is_internally_consistent():
    robot = RobotGeometry()
    plant = PlantParams()

    assert math.isclose(robot.m_wheel,
                        robot.wheel_motor_mass + robot.wheel_print_mass)
    assert math.isclose(robot.body_mass_excluding_wheels, 1.769488,
                        abs_tol=1e-6)
    assert math.isclose(robot.total_mass, 2.809488, abs_tol=1e-6)
    assert math.isclose(plant.body_mass_kg, robot.body_mass_excluding_wheels,
                        abs_tol=1e-6)
    assert math.isclose(plant.wheel_mass_kg, robot.m_wheel)
    assert "body_mass_kg" in PROVISIONAL_FIELDS
    assert "wheel_mass_kg" in PROVISIONAL_FIELDS
