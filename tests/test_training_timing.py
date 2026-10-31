"""Check timing estimates against trajectories with known derivatives."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'training_timing', Path(__file__).resolve().parents[1] / 'scripts/analyze_training_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class TrainingTimingTests(unittest.TestCase):
    def test_constant_velocity_and_distinct_gripper_limits(self):
        q = np.zeros((5, 14))
        q[:, 0] = np.arange(5) * .1
        q[:, 6] = np.arange(5) * .2
        v, a, error = timing.derivatives(q[:-1], q[1:], 20)
        np.testing.assert_allclose(v[:, 0], 2.)
        np.testing.assert_allclose(a, 0., atol=1e-12)
        self.assertEqual(error, 0.)
        cost = timing.dilation_cost(v, a, (1., 2., 4., 8.))
        self.assertAlmostEqual(cost['local_velocity_interval_equivalents'], 8.)
        self.assertAlmostEqual(cost['uniform_episode_scale'], 2.)

    def test_acceleration_requires_square_root_dilation(self):
        q = np.zeros((4, 14))
        q[:, 0] = [0., .1, .3, .6]
        v, a, _ = timing.derivatives(q[:-1], q[1:], 10)
        np.testing.assert_allclose(a[:, 0], 10.)
        cost = timing.dilation_cost(v, a, (100., 2.5, 100., 100.))
        self.assertAlmostEqual(cost['uniform_episode_scale'], 2.)
        slower_v, slower_a, _ = timing.derivatives(q[:-1], q[1:], 5)
        self.assertLessEqual(np.abs(slower_v).max(), 100.)
        self.assertLessEqual(np.abs(slower_a).max(), 2.5 + 1e-12)

    def test_stationary_episode_is_not_stretched(self):
        q = np.ones((3, 14))
        v, a, _ = timing.derivatives(q[:-1], q[1:], 20)
        cost = timing.dilation_cost(v, a, (.6, 2., 2., 8.))
        self.assertEqual(cost['local_velocity_interval_equivalents'], 2.)
        self.assertEqual(cost['uniform_episode_scale'], 1.)

    def test_rejects_unrelated_actions_and_invalid_samples(self):
        q = np.zeros((3, 14))
        for states, actions, fps in [(q, q + 1, 20), (q, q, 0),
                                     (q[:1], q[:1], 20), (q * np.nan, q, 20)]:
            with self.assertRaises(ValueError):
                timing.derivatives(states, actions, fps)


if __name__ == '__main__':
    unittest.main()
