import unittest

from tools.link_baseline import percentile


class LinkBaselineTests(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2)
        self.assertEqual(percentile([4, 1, 3, 2], 0.95), 4)


if __name__ == "__main__":
    unittest.main()
