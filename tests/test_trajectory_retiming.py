"""Retiming invariants and finite-difference limits on known paths."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('trajectory_retiming', Path(__file__).resolve().parents[1] / 'scripts/trajectory_retiming.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class RetimingTests(unittest.TestCase):
    def assert_map(self, q, result):
        np.testing.assert_allclose(result['poses'][[0, -1]], q[[0, -1]])
        self.assertTrue(np.all(np.diff(result['source_times']) > 0))
        np.testing.assert_allclose(np.diff(result['output_times']), .05)
        self.assertGreaterEqual(result['metrics']['duration_scale'], 1 - 1e-12)
        self.assertLess(result['metrics']['protected_duration_error_seconds'], 1e-10)
        for j in range(14):
            np.testing.assert_allclose(result['poses'][:, j], np.interp(result['source_times'], np.arange(len(q)) / 20, q[:, j]), atol=1e-10)

    def test_stationary_has_identical_timing(self):
        q = np.full((20, 14), .3)
        r = timing.retime(q)
        self.assert_map(q, r)
        np.testing.assert_allclose(r['source_times'], np.arange(20) / 20)
        self.assertTrue(r['protected_mask'].all())
        self.assertTrue(r['metrics']['all_limits_satisfied'])

    def test_constant_speed_satisfies_caps(self):
        q = np.zeros((21, 14)); q[:, 0] = np.arange(21) * .1
        r = timing.retime(q, joint_velocity=.5)
        self.assert_map(q, r)
        self.assertTrue(r['metrics']['all_limits_satisfied'])
        self.assertAlmostEqual(r['metrics']['duration_scale'], 4.)

    def test_turn_requires_acceleration_dilation(self):
        q = np.zeros((21, 14)); q[:, 0] = np.r_[np.arange(11), np.arange(9, -1, -1)] * .05
        r = timing.retime(q, joint_velocity=1., joint_acceleration=2.)
        self.assert_map(q, r)
        self.assertTrue(r['metrics']['all_limits_satisfied'])
        self.assertLessEqual(r['metrics']['after']['joint_acceleration']['max'], 2.00002)

    def test_gripper_has_distinct_limit_shared_clock(self):
        q = np.zeros((15, 14)); q[:, 6] = np.arange(15) * .1; q[:, 0] = np.arange(15) * .01
        r = timing.retime(q, joint_velocity=10., gripper_velocity=.5)
        self.assert_map(q, r)
        self.assertAlmostEqual(r['metrics']['duration_scale'], 4.)
        self.assertLessEqual(r['metrics']['after']['gripper_velocity']['max'], .50001)
        np.testing.assert_allclose(r['poses'][:, 6], 10 * r['poses'][:, 0])

    def test_pause_and_explicit_contact_keep_exact_duration(self):
        q = np.zeros((20, 14)); q[:, 0] = np.r_[np.zeros(4), np.arange(12) * .06, np.full(4, .66)]
        protect = np.zeros(19, bool); protect[7:9] = True
        r = timing.retime(q, protected_mask=protect, max_iterations=20)
        self.assert_map(q, r)
        np.testing.assert_allclose(np.diff(r['knot_times'])[r['protected_mask']], .05)
        self.assertTrue(r['protected_mask'][7:9].all())
        self.assertGreater(r['metrics']['violations']['joint_velocity']['protected'], 0)
        self.assertFalse(r['metrics']['all_limits_satisfied'])

    def test_fully_protected_violations_remain_and_are_reported(self):
        q = np.zeros((4, 14)); q[:, 0] = [0, .1, -.1, 0]
        r = timing.retime(q, protected_mask=np.ones(3, bool))
        self.assert_map(q, r)
        np.testing.assert_allclose(r['poses'], q)
        self.assertTrue(r['metrics']['converged'])
        self.assertFalse(r['metrics']['all_limits_satisfied'])
        self.assertEqual(r['metrics']['violations']['joint_acceleration']['protected'], 2)

    def test_terminal_grid_rounding_may_exceed_nominal_interval_cap(self):
        q = np.zeros((4, 14)); q[:, 0] = [0., .2, .2, .2]
        r = timing.retime(q, joint_velocity=1., max_scale=2.3)
        self.assert_map(q, r)
        # Only the movement is editable. Its nominal 2.3-frame duration gets
        # rounded up to 3 frames; the two following pauses remain 1 frame each.
        self.assertGreater(r['interval_scales'].max(), 2.3)
        self.assertLess(r['interval_scales'].max(), 3.3)
        self.assertAlmostEqual(r['interval_scales'][0], 3.)
        np.testing.assert_allclose(r['interval_scales'][1:], 1.)
        self.assertEqual(r['metrics']['intervals_at_scale_cap'], 1)
        self.assertFalse(r['metrics']['all_limits_satisfied'])

    def test_terminal_pose_two_knots_and_invalid_inputs(self):
        q = np.zeros((2, 14)); q[1, 0] = .117
        r = timing.retime(q, joint_velocity=.6)
        self.assert_map(q, r)
        self.assertEqual(len(r['poses']), 5)
        for args in [dict(fps=0), dict(joint_velocity=-1), dict(protected_mask=[1]), dict(protect_joint_speed=-1)]:
            with self.assertRaises(ValueError): timing.retime(q, **args)
        with self.assertRaises(ValueError): timing.retime(np.full((3,14), np.nan))


if __name__ == '__main__': unittest.main()
