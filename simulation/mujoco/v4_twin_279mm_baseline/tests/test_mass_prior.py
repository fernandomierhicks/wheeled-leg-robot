"""Checks for the 2026-08-09 as-built v4 mass inventory."""

import math

import pytest

from v4_twin_279mm_baseline.params import RobotGeometry
from v4_twin_279mm_baseline.twin.params_plant import PlantParams, PROVISIONAL_FIELDS


def test_measured_component_masses_are_preserved_separately_from_residual():
    robot = RobotGeometry()
    assert robot.m_box == 0.505
    assert robot.m_battery == 0.276
    assert robot.wheel_motor_mass == 0.418
    assert robot.wheel_tpu_mass == 0.029
    assert robot.wheel_rim_mass == 0.031
    assert robot.measured_femur_mass == 0.040
    assert robot.measured_tibia_mass == 0.085
    assert robot.measured_coupler_mass == 0.035
    assert robot.m_bearing == 0.018
    assert 2 * robot.bearings_per_leg == 16

    distributed = 2.0 * (
        (robot.m_femur - robot.measured_femur_mass)
        + (robot.m_tibia - robot.measured_tibia_mass)
        + (robot.m_coupler - robot.measured_coupler_mass))
    assert distributed == pytest.approx(robot.unassigned_mass)


def test_as_built_mass_accounting_matches_whole_robot_scale():
    robot = RobotGeometry()
    plant = PlantParams()

    assert math.isclose(robot.m_wheel,
                        robot.wheel_motor_mass + robot.wheel_print_mass)
    assert math.isclose(robot.wheel_print_mass,
                        robot.wheel_tpu_mass + robot.wheel_rim_mass)
    assert math.isclose(robot.total_mass_without_battery, 3.242,
                        abs_tol=1e-6)
    assert math.isclose(robot.total_mass, 3.518, abs_tol=1e-6)
    assert math.isclose(robot.body_mass_excluding_wheels, 2.562, abs_tol=1e-6)
    assert math.isclose(plant.body_mass_kg, robot.body_mass_excluding_wheels, abs_tol=1e-6)
    assert math.isclose(plant.wheel_mass_kg, robot.m_wheel)
    assert "body_mass_kg" not in PROVISIONAL_FIELDS
    assert "wheel_mass_kg" not in PROVISIONAL_FIELDS
