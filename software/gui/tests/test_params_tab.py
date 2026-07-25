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
    _DISPLAY_UNIT_BY_PARAM,
    _ParamRow,
)


_APP = QApplication.instance() or QApplication([])


class ParamsTabAngleDisplayTests(unittest.TestCase):
    def test_every_declared_angle_name_resolves_to_a_generated_parameter(self):
        names = _ANGLE_PARAM_NAMES | _ANGULAR_RATE_PARAM_NAMES
        self.assertEqual(len(names), 24)
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


if __name__ == "__main__":
    unittest.main()
