import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from analysis.leg_height_sweep import (
    TRIM_TABLE_PARAMS, alpha_plateaus, band_rms, rate_limit_duty,
    trim_table_from_plateaus,
)


class AlphaPlateauTests(unittest.TestCase):
    FS = 500.0

    def _trace(self, *levels_and_seconds):
        return np.concatenate([np.full(int(secs * self.FS), level)
                               for level, secs in levels_and_seconds])

    def test_flat_stretches_separated_by_a_ramp_are_two_plateaus(self):
        alpha = np.concatenate([
            np.full(5000, 0.10),                  # 10 s
            np.linspace(0.10, 0.50, 1000),        #  2 s ramp
            np.full(5000, 0.50),                  # 10 s
        ])
        found = alpha_plateaus(alpha, self.FS)
        self.assertEqual(len(found), 2)
        self.assertAlmostEqual(alpha[slice(*found[0])].mean(), 0.10, places=3)
        self.assertAlmostEqual(alpha[slice(*found[1])].mean(), 0.50, places=3)

    def test_a_slow_steady_ramp_is_never_a_plateau(self):
        # 0.1 alpha/s across the stroke — slower than any ramp in a real run,
        # and per-sample steps of 2e-4 that no derivative threshold could
        # separate from noise. The drift test rejects it anyway.
        self.assertEqual(alpha_plateaus(np.linspace(0.0, 0.6, 3000), self.FS), [])

    def test_noise_does_not_fragment_a_genuine_plateau(self):
        rng = np.random.default_rng(0)
        alpha = 0.4 + rng.normal(0.0, 0.002, 10000)
        self.assertEqual(len(alpha_plateaus(alpha, self.FS)), 1)

    def test_stretches_shorter_than_the_minimum_are_dropped(self):
        alpha = np.concatenate([np.full(1000, 0.2),          # 2 s, too short
                                np.linspace(0.2, 0.6, 500),
                                np.full(4000, 0.6)])         # 8 s, kept
        found = alpha_plateaus(alpha, self.FS, min_s=4.0)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(alpha[slice(*found[0])].mean(), 0.6, places=3)


class BandRmsTests(unittest.TestCase):
    def test_a_pure_tone_lands_in_its_own_band_at_its_own_amplitude(self):
        fs, t = 500.0, np.arange(0, 40, 1 / 500.0)
        signal = 3.0 * np.sin(2 * np.pi * 0.6 * t)
        self.assertAlmostEqual(band_rms(signal, fs, 0.3, 1.5), 3.0 / np.sqrt(2), delta=0.15)
        self.assertLess(band_rms(signal, fs, 1.5, 4.0), 0.05)

    def test_bands_separate_two_superimposed_tones(self):
        fs, t = 500.0, np.arange(0, 40, 1 / 500.0)
        signal = 2.0 * np.sin(2 * np.pi * 0.6 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t)
        self.assertAlmostEqual(band_rms(signal, fs, 0.3, 1.5), 2.0 / np.sqrt(2), delta=0.12)
        self.assertAlmostEqual(band_rms(signal, fs, 1.5, 4.0), 0.5 / np.sqrt(2), delta=0.05)


class RateLimitTests(unittest.TestCase):
    FS = 500.0

    def test_a_triangle_wave_is_fully_slew_saturated_in_long_runs(self):
        # A rate limiter driven past its limit emits a triangle wave: every
        # sample is at the limit, in runs of half a period.
        limit, freq = 0.4, 0.5
        t = np.arange(0, 20, 1 / self.FS)
        theta = limit / (2 * freq) * (2 * np.abs(2 * (t * freq % 1) - 1) - 1)
        report = rate_limit_duty(theta, self.FS, rate_lim=limit)
        self.assertGreater(report["duty_frac"], 0.98)
        self.assertAlmostEqual(report["max_run_s"], 1.0, delta=0.05)  # half of 2 s
        self.assertFalse(report["inferred"])

    def test_a_smooth_signal_below_the_limit_never_saturates(self):
        t = np.arange(0, 20, 1 / self.FS)
        theta = 0.05 * np.sin(2 * np.pi * 0.2 * t)   # peak slew 0.063 rad/s
        self.assertEqual(rate_limit_duty(theta, self.FS, rate_lim=0.4)["duty_frac"], 0.0)

    def test_the_limit_is_inferred_from_the_data_when_not_supplied(self):
        limit = 0.4
        t = np.arange(0, 20, 1 / self.FS)
        theta = limit / 1.0 * (2 * np.abs(2 * (t * 0.5 % 1) - 1) - 1) / 2
        report = rate_limit_duty(theta, self.FS)
        self.assertTrue(report["inferred"])
        self.assertAlmostEqual(report["rate_lim"], limit, delta=0.02)


class TrimTableTests(unittest.TestCase):
    def test_points_on_the_table_alphas_come_back_exactly(self):
        alphas = [0.0, 0.25, 0.5, 1.0]
        balances = [-0.02, -0.085, -0.067, -0.03]
        table = trim_table_from_plateaus(alphas, balances)
        self.assertEqual(table["params"], TRIM_TABLE_PARAMS)
        self.assertAlmostEqual(table["points_rad"][0], -0.02, places=9)
        self.assertAlmostEqual(table["points_rad"][2], -0.085, places=9)
        self.assertAlmostEqual(table["points_rad"][4], -0.067, places=9)
        self.assertAlmostEqual(table["points_rad"][8], -0.03, places=9)
        self.assertAlmostEqual(table["points_rad"][1], -0.0525, places=9)  # midway 0..0.25
        self.assertFalse(any(table["extrapolated"]))

    def test_points_outside_the_measured_span_are_held_and_flagged(self):
        table = trim_table_from_plateaus([0.46, 0.22], [-0.095, -0.085])  # unsorted on purpose
        self.assertEqual(table["alpha_span"], (0.22, 0.46))
        self.assertAlmostEqual(table["points_rad"][0], -0.085, places=9)
        self.assertAlmostEqual(table["points_rad"][8], -0.095, places=9)
        self.assertEqual(table["extrapolated"],
                         [True, True, False, False, True, True, True, True, True])

    def test_no_measurement_is_an_error(self):
        with self.assertRaises(ValueError):
            trim_table_from_plateaus([], [])


if __name__ == "__main__":
    unittest.main()
