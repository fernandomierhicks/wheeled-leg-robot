import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tabs.log_analyzer_tab import _event_summary, _mask_intervals


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


if __name__ == "__main__":
    unittest.main()
