"""Independent verifier catches corruption in provenance and protected timing."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("verify_retimed_dataset", Path(__file__).parents[1] / "scripts/verify_retimed_dataset.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class RetimingVerificationTests(unittest.TestCase):
    def test_shared_coordinate_reproduces_all_14_dimensions(self):
        path = np.arange(42).reshape(3, 14).astype(float)
        result = verifier.interpolate(path, [.25, 1.5, 2])
        np.testing.assert_allclose(result[0], .75 * path[0] + .25 * path[1])
        np.testing.assert_allclose(result[1], .5 * path[1] + .5 * path[2])
        np.testing.assert_array_equal(result[2], path[2])
        with self.assertRaises(AssertionError):
            verifier.interpolate(path, [2.01])

    def test_protected_duration_preserved_despite_output_grid_phase(self):
        scales = np.array([1.5, 1., 1., 1.5])
        knots = np.r_[0, np.cumsum(scales)] / 20
        frames = np.interp(np.arange(6) / 20, knots, np.arange(5))
        duration, protected = verifier.check_clock(frames[:-1], frames[1:], scales, [False, True, True, False], 20)
        self.assertAlmostEqual(duration, .25)
        self.assertAlmostEqual(protected, .1)
        self.assertNotEqual(frames[2], round(frames[2]))

    def test_protection_violation_rejected(self):
        with self.assertRaises(AssertionError):
            verifier.check_clock([0, 1], [1, 2], [1.1, 1], [True, False], 20)

    def test_camera_offsets_recomputed_at_fractional_source_clock(self):
        original = {"source_episode_time_s": .1, "overhead.offset_s": .01,
                    "overhead.source_path": "camera.jpg", "overhead.source_sha256": "original"}
        mapping = {"source_episode_time_s": .12, "overhead.offset_s": -.01,
                   "overhead.original_offset_s": .01, "overhead.source_path": "camera.jpg",
                   "overhead.source_sha256": "original"}
        verifier.check_camera_provenance(mapping, original, "overhead", 20)
        mapping["overhead.offset_s"] = .01
        with self.assertRaises(AssertionError):
            verifier.check_camera_provenance(mapping, original, "overhead", 20)

    def test_terminal_camera_clamp_allows_one_source_tick(self):
        original = {"source_episode_time_s": .1, "overhead.offset_s": -.01}
        mapping = {"source_episode_time_s": .145, "source_state_frame": 2.9,
                   "overhead.original_offset_s": -.01, "overhead.offset_s": -.055,
                   "image_source_terminal_clamped": True, "camera_alignment_extra_bound_s": .05}
        verifier.check_camera_provenance(mapping, original, "overhead", 20, source_count=3)
        mapping["image_source_terminal_clamped"] = False
        with self.assertRaises(AssertionError):
            verifier.check_camera_provenance(mapping, original, "overhead", 20, source_count=3)

    def test_precision_half_open_windows_and_visual_confidence(self):
        phases = [{"phase": "gear_insertion", "start_s": 1., "end_s": 2.,
                   "confidence": "visual", "visual_reviewer_confidence": "medium-low"},
                  {"phase": "pin_insertion", "start_s": 2., "end_s": 3.,
                   "confidence": "visual", "visual_reviewer_confidence": "medium"}]
        low = verifier.expected_precision_label(phases, 1.5, 3.)
        self.assertTrue(low["gear_insertion"])
        self.assertEqual(low["loss_weight"], 1.)
        boundary = verifier.expected_precision_label(phases, 2., 3.)
        self.assertFalse(boundary["gear_insertion"])
        self.assertTrue(boundary["pin_insertion"])
        self.assertEqual(boundary["loss_weight"], 3.)
        self.assertEqual(verifier.expected_precision_label(phases, 3., 3.)["loss_weight"], 1.)

    def test_cross_episode_or_different_action_clock_rejected(self):
        with self.assertRaises(AssertionError):
            verifier.check_clock([0, .5], [.6, 1.1], [2, 2], [False, False], 20)


if __name__ == "__main__":
    unittest.main()
