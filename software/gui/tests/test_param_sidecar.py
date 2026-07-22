import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.param_sidecar import (
    active_profile_series, find_matching_sidecar, load_param_sidecar,
)


class ParamSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name="LOG0001.PARAMS"):
        path = self.root / name
        path.write_text(
            "# t_micros,event,id,name,value\n"
            "990,DUMP,1298,profile1_torque_lim,0.1\n"
            "991,DUMP,1301,profile2_torque_lim,0.2\n"
            "992,DUMP,1304,profile3_torque_lim,0.3\n"
            "993,DUMP,1032,vel_pi_theta_max,0.5\n"
            "2500,CHANGE,1298,profile1_torque_lim,0.15\n",
            encoding="utf-8",
        )
        return path

    def test_loads_dump_and_change_as_sample_aligned_series(self):
        sidecar = load_param_sidecar(self._write())
        samples = np.asarray([1000, 2000, 3000, 4000], dtype=np.uint32)
        np.testing.assert_allclose(
            sidecar.series("profile1_torque_lim", samples),
            [0.1, 0.1, 0.15, 0.15],
        )
        self.assertEqual(sidecar.initial_value("vel_pi_theta_max"), 0.5)
        self.assertIsNone(sidecar.series("missing", samples))

    def test_active_profile_selects_the_matching_time_varying_limit(self):
        sidecar = load_param_sidecar(self._write())
        samples = np.asarray([1000, 2000, 3000, 4000], dtype=np.uint32)
        profiles = np.asarray([0, 1, 2, 0])
        values = active_profile_series(sidecar, "torque_lim", profiles, samples)
        np.testing.assert_allclose(values, [0.1, 0.2, 0.3, 0.15])

    def test_uint32_clock_rollover_is_aligned(self):
        path = self.root / "ROLL.PARAMS"
        path.write_text(
            "# t_micros,event,id,name,value\n"
            "4294967000,DUMP,1,gain,1\n"
            "150,CHANGE,1,gain,2\n",
            encoding="utf-8",
        )
        sidecar = load_param_sidecar(path)
        samples = np.asarray([4294967100, 50, 250], dtype=np.uint32)
        np.testing.assert_allclose(sidecar.series("gain", samples), [1, 1, 2])

    def test_finds_companion_case_insensitively(self):
        wlog = self.root / "Trial.WLOG"
        wlog.touch()
        sidecar = self._write("trial.params")
        self.assertEqual(find_matching_sidecar(wlog), sidecar)

    def test_rejects_malformed_rows_with_context(self):
        path = self.root / "BAD.PARAMS"
        path.write_text("1,DUMP,2,missing-value\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected 5 columns"):
            load_param_sidecar(path)


if __name__ == "__main__":
    unittest.main()
