import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from tabs.generated_protocol import PARAM_BY_NAME
from tabs.params_tab import (
    _ANGLE_PARAM_NAMES,
    _ANGULAR_RATE_PARAM_NAMES,
    _BENCH_PRESETS,
    _DISPLAY_UNIT_BY_PARAM,
    _ParamRow,
    _effective_group,
    _get_subgroup,
    _migrate_import_value,
)


_APP = QApplication.instance() or QApplication([])


class ParamsTabAngleDisplayTests(unittest.TestCase):
    def test_every_declared_angle_name_resolves_to_a_generated_parameter(self):
        names = _ANGLE_PARAM_NAMES | _ANGULAR_RATE_PARAM_NAMES
        self.assertEqual(len(names), 33)
        self.assertEqual(
            {PARAM_BY_NAME[name] for name in names},
            set(_DISPLAY_UNIT_BY_PARAM),
        )

    def test_angle_rows_display_degrees_but_export_radians(self):
        row = _ParamRow(
            PARAM_BY_NAME["calib_range_l_rad"],
            "calib_range_l_rad",
            "Range [rad].",
            math.pi / 2,
            0.0,
            math.pi,
            0,
        )
        self.assertAlmostEqual(float(row._edit.text()), 90.0)
        self.assertIn("180", row._range_lbl._full_text)
        self.assertIn("deg", row._range_lbl._full_text)
        self.assertAlmostEqual(row.current_value(), math.pi / 2)
        self.assertAlmostEqual(row.export_entry()["value"], math.pi / 2)

        row.update_value(math.pi / 4, 0.0, math.pi, 0)
        self.assertAlmostEqual(float(row._edit.text()), 45.0)

    def test_degree_edit_is_converted_to_radians_when_sent(self):
        row = _ParamRow(
            PARAM_BY_NAME["omega_cmd_rds"],
            "omega_cmd_rds",
            "Yaw rate [rad/s].",
            0.0,
            -math.pi,
            math.pi,
            0,
        )
        row._edit.setText("180")

        with (
            patch("tabs.params_tab.send_param_set") as send_param_set,
            patch(
                "tabs.params_tab.send_reliable",
                side_effect=lambda send, **_: send(),
            ),
        ):
            row._send()

        send_param_set.assert_called_once()
        param_id, raw_value = send_param_set.call_args.args
        self.assertEqual(param_id, PARAM_BY_NAME["omega_cmd_rds"])
        self.assertAlmostEqual(raw_value, math.pi)

    def test_angle_gains_remain_in_firmware_units(self):
        row = _ParamRow(
            PARAM_BY_NAME["standup_k_pitch"],
            "standup_k_pitch",
            "Recovery gain [N.m/rad].",
            12.5,
            0.0,
            60.0,
            0,
        )
        self.assertEqual(row._edit.text(), "12.5")
        self.assertAlmostEqual(row.current_value(), 12.5)

    def test_signed_standup_bounds_display_and_edit_as_signed_degrees(self):
        row = _ParamRow(
            PARAM_BY_NAME["standup_pitch_min"],
            "standup_pitch_min",
            "Lower signed pitch bound [rad].",
            -math.pi / 4,
            -1.4,
            0.0,
            0,
        )
        self.assertAlmostEqual(float(row._edit.text()), -45.0)
        row._edit.setText("-37")

        with (
            patch("tabs.params_tab.send_param_set") as send_param_set,
            patch("tabs.params_tab.send_reliable", side_effect=lambda send, **_: send()),
        ):
            row._send()

        _, raw_value = send_param_set.call_args.args
        self.assertAlmostEqual(raw_value, math.radians(-37))

    def test_old_backward_magnitude_export_migrates_to_signed_minimum(self):
        param_id = PARAM_BY_NAME["standup_pitch_min"]
        old_entry = {"name": "standup_pitch_bwd", "value": 0.6}
        self.assertEqual(_migrate_import_value(param_id, old_entry, 0.6), -0.6)
        new_entry = {"name": "standup_pitch_min", "value": -0.7}
        self.assertEqual(_migrate_import_value(param_id, new_entry, -0.7), -0.7)

    def test_balance_trim_schedule_is_grouped_under_hip(self):
        for name in (
            "lqr_pitch_trim_ret",
            "lqr_pitch_trim_ext",
            "lqr_trim_curve",
        ):
            param_id = PARAM_BY_NAME[name]
            self.assertEqual(_effective_group(param_id), 0x02)
            self.assertEqual(_get_subgroup(param_id), "Pitch Trim vs Leg Height")

    def test_standup_divergence_limits_are_grouped_with_standing_up(self):
        for name in ("standup_div_fwd", "standup_div_bwd"):
            param_id = PARAM_BY_NAME[name]
            self.assertEqual(_effective_group(param_id), 0x08)
            self.assertEqual(_get_subgroup(param_id), "Divergence Limits")

    def test_lqr_barrier_parameters_have_a_control_subgroup(self):
        for name in ("lqr_barrier_k", "lqr_barrier_th_ret", "lqr_barrier_th_ext"):
            param_id = PARAM_BY_NAME[name]
            self.assertEqual(_effective_group(param_id), 0x04)
            self.assertEqual(_get_subgroup(param_id), "Backward Pitch Barrier")

    def test_plant_identification_parameters_are_diagnostics(self):
        for name in ("plant_id_en", "plant_id_amp", "plant_id_f0", "plant_id_f1", "plant_id_dur"):
            param_id = PARAM_BY_NAME[name]
            self.assertEqual(_effective_group(param_id), 0x09)
            self.assertEqual(_get_subgroup(param_id), "Plant Identification")

    def test_late_allocated_standup_parameter_is_not_left_in_command(self):
        self.assertEqual(_effective_group(PARAM_BY_NAME["standup_ret_gains"]), 0x08)


class ParamsTabBenchPresetTests(unittest.TestCase):
    @staticmethod
    def _values(key: str) -> dict[str, float]:
        return next(values for preset_key, _, _, _, values in _BENCH_PRESETS
                    if preset_key == key)

    def test_no_motors_is_safe_for_an_out_of_level_bench_arm(self):
        values = self._values("no_motors")

        self.assertEqual(values["hip_l_enable"], 0.0)
        self.assertEqual(values["hip_r_enable"], 0.0)
        self.assertEqual(values["wheel_l_enable"], 0.0)
        self.assertEqual(values["wheel_r_enable"], 0.0)
        self.assertEqual(values["calib_bypass_en"], 1.0)
        self.assertEqual(values["run_wheel_bypass_en"], 1.0)
        self.assertEqual(values["standup_enable"], 0.0)
        self.assertEqual(values["pitch_watchdog_en"], 0.0)
        self.assertEqual(values["roll_watchdog_en"], 0.0)
        self.assertEqual(values["wheel_runaway_en"], 0.0)

        # _apply_preset preserves this insertion order. Both 200 ms pose
        # watchdogs must be disabled before either arm bypass is opened.
        order = list(values)
        for watchdog in ("roll_watchdog_en", "pitch_watchdog_en"):
            self.assertLess(order.index(watchdog), order.index("calib_bypass_en"))
            self.assertLess(order.index(watchdog), order.index("run_wheel_bypass_en"))

    def test_full_robot_restores_safeties_disabled_by_no_motors(self):
        no_motors = self._values("no_motors")
        full = self._values("full")

        disabled_safeties = {
            name for name, value in no_motors.items()
            if name.endswith("watchdog_en") or name == "wheel_runaway_en"
            if value == 0.0
        }
        self.assertEqual(
            disabled_safeties,
            {"pitch_watchdog_en", "roll_watchdog_en", "wheel_runaway_en"},
        )
        self.assertTrue(all(full[name] == 1.0 for name in disabled_safeties))


if __name__ == "__main__":
    unittest.main()
