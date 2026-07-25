import os
from pathlib import Path
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from analysis.param_sidecar import ParamEvent, ParamSidecar
from tabs.log_analyzer_tab import (
    _event_summary, _mask_intervals, _parameter_change_groups,
    _parameter_change_label,
)


class LogAnalyzerHelperTests(unittest.TestCase):
    def test_mask_intervals_compresses_contiguous_saturation_samples(self):
        t = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
        mask = np.asarray([False, True, True, False, True])
        self.assertEqual(_mask_intervals(t, mask), [(0.1, 0.3), (0.4, 0.5)])
        self.assertEqual(_event_summary(t, mask), "2 events, 0.300 s total")

    def test_empty_mask_reports_none(self):
        t = np.asarray([0.0, 0.1])
        self.assertEqual(_mask_intervals(t, np.asarray([False, False])), [])
        self.assertEqual(_event_summary(t, np.asarray([False, False])), "none")

    def test_parameter_change_groups_omit_dump_and_align_rollover(self):
        sidecar = ParamSidecar(Path("run.PARAMS"), [
            ParamEvent(0xFFFFFFF0, "DUMP", 1, "gain", 1.0),
            ParamEvent(4, "CHANGE", 1, "gain", 2.0),
            ParamEvent(4, "CHANGE", 2, "limit", 3.0),
        ])
        groups = _parameter_change_groups(
            sidecar,
            np.asarray([0xFFFFFFF0, 4, 24], dtype=np.uint32),
            np.asarray([0.0, 0.000020, 0.000040]),
        )
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(groups[0][0], 0.000020)
        self.assertEqual(_parameter_change_label(groups[0][1]), "gain=2, limit=3")


if __name__ == "__main__":
    unittest.main()
