import unittest

import numpy as np

from analysis.jump_analysis import analyze_jumps, jump_focus_mask


def imu_fields(size: int) -> dict[str, np.ndarray]:
    return {
        "accel_x_ms2": np.zeros(size),
        "accel_y_ms2": np.zeros(size),
        "accel_z_ms2": np.zeros(size),
        "roll_rate_rads": np.zeros(size),
        "pitch_rate_rads": np.zeros(size),
        "yaw_rate_rads": np.zeros(size),
    }


class JumpAnalysisTests(unittest.TestCase):
    def test_modern_episode_uses_live_landing_and_handoff_phases(self):
        t = np.arange(8, dtype=float) * 0.1
        fields = imu_fields(t.size)
        fields["robot_state"] = np.asarray([3, 7, 7, 7, 7, 7, 3, 3])
        fields["jump_state"] = np.asarray([0, 0, 1, 2, 3, 4, 4, 4])

        episodes = analyze_jumps(t, fields)

        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertTrue(episode.modern_phases)
        self.assertEqual(episode.phase_entries, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
        self.assertEqual(episode.landing_index, 4)
        self.assertEqual(episode.landing_source, "live firmware phase")

    def test_legacy_gyro_landing_requires_two_changes_in_twelve_ms(self):
        t = np.asarray([0.0, 0.05, 0.10, 0.26, 0.266, 0.272, 0.35])
        fields = imu_fields(t.size)
        fields["robot_state"] = np.full(t.size, 7)
        fields["jump_state"] = np.asarray([0, 1, 2, 2, 2, 2, 3])
        fields["pitch_rate_rads"] = np.asarray([0.0, 0.0, 0.0, 1.6, 2.8, 2.8, 2.8])

        episode = analyze_jumps(t, fields, modern_phases=False)[0]

        self.assertEqual(episode.landing_index, 4)
        self.assertEqual(episode.landing_source, "gyro impulse")

    def test_legacy_accel_requires_airborne_latch_before_rebound(self):
        t = np.asarray([0.0, 0.05, 0.10, 0.26, 0.27, 0.29])
        fields = imu_fields(t.size)
        fields["robot_state"] = np.full(t.size, 7)
        fields["jump_state"] = np.asarray([0, 1, 2, 2, 2, 2])
        fields["accel_z_ms2"] = np.asarray([0.0, 0.0, 0.0, -9.8, 2.1, 2.1])

        episode = analyze_jumps(t, fields, modern_phases=False)[0]

        self.assertEqual(episode.landing_index, 4)
        self.assertEqual(episode.landing_source, "accel rebound")
        self.assertTrue(episode.airborne_seen)

    def test_focus_mask_includes_pre_and_post_jump_context(self):
        t = np.arange(11, dtype=float) * 0.1
        fields = imu_fields(t.size)
        fields["robot_state"] = np.asarray([3, 3, 3, 7, 7, 7, 3, 3, 3, 3, 3])
        fields["jump_state"] = np.asarray([0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 4])
        episode = analyze_jumps(t, fields)[0]

        mask = jump_focus_mask(t, [episode], before_s=0.1, after_s=0.2)

        np.testing.assert_array_equal(
            mask, [False, False, True, True, True, True, True, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
