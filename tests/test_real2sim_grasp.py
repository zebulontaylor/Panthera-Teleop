"""Offline rigid grasp and planar calibration regression tests."""
import sys
import unittest
from pathlib import Path
try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
if np is not None:
    from real2sim_grasp import reconstruct_grasp, estimate_tabletop_grasp, fit_planar_base_transform


def pose(yaw=0, xyz=(0, 0, 0)):
    t = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t[:3, 3] = xyz
    return t


@unittest.skipIf(np is None, 'numpy unavailable')
class GraspTests(unittest.TestCase):
    def test_moving_tcp_stationary_object_before_and_after_grasp(self):
        tcp = np.array([pose(i*.3, (i*.1, i*.02, .4+i*.01)) for i in range(9)])
        offset = pose(.2, (.01, -.02, -.15))
        obj = tcp[3] @ offset
        a = reconstruct_grasp(np.arange(9), tcp, 3, 6, object_pose_at_pickup=obj)
        b = reconstruct_grasp(np.arange(9), tcp, 3, 6, tool_to_object=offset)
        np.testing.assert_allclose(a['object_poses'], b['object_poses'], atol=1e-12)
        np.testing.assert_allclose(a['object_poses'][:3], np.repeat(obj[None], 3, axis=0))
        np.testing.assert_allclose(a['object_poses'][3:7], tcp[3:7] @ offset)
        np.testing.assert_allclose(a['object_poses'][7:], np.repeat((tcp[6] @ offset)[None], 2, axis=0))
        self.assertTrue(a['metadata']['kinematic_estimate'])
        self.assertFalse(a['metadata']['physics_simulated'])

    def test_tabletop_origin_height_and_offset(self):
        tcp = pose(.7, (.4, .2, .5))
        estimate = estimate_tabletop_grasp(tcp, .02, .4, (.01, -.03), .12)
        obj = estimate['object_pose_at_pickup']
        np.testing.assert_allclose(obj[:3, 3], [.41, .17, .14])
        np.testing.assert_allclose(tcp @ estimate['tool_to_object'], obj, atol=1e-12)

    def test_fit_known_transform_and_heldout_not_in_fit(self):
        right = np.array([[0, 0, 0], [.2, .1, .04], [-.1, .3, .06], [.4, -.2, -.1]])
        expected = pose(.6, (.9, -.3, .12))
        left = right @ expected[:3, :3].T + expected[:3, 3]
        fitted = fit_planar_base_transform(right, left, heldout_right_points=right,
                                           heldout_left_points=left + 1)
        np.testing.assert_allclose(fitted['right_to_left'], expected, atol=1e-12)
        self.assertLess(fitted['fit_rms_m'], 1e-12)
        self.assertAlmostEqual(fitted['heldout_rms_m'], np.sqrt(3))
        self.assertFalse(fitted['heldout_used_in_fit'])

    def test_reject_degenerate_and_nonfinite_inputs(self):
        for value in [np.zeros((3, 3)), np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                fit_planar_base_transform(value, value)
        tcp = np.repeat(np.eye(4)[None], 3, axis=0)
        for times in [[0, 0, 1], [0, np.nan, 2]]:
            with self.assertRaises(ValueError):
                reconstruct_grasp(times, tcp, 0, tool_to_object=np.eye(4))
        with self.assertRaises(ValueError):
            reconstruct_grasp([0, 1, 2], tcp, 2, 1, tool_to_object=np.eye(4))
        with self.assertRaises(ValueError):
            reconstruct_grasp([0, 1, 2], tcp, 0, tool_to_object=np.full((4, 4), np.inf))
        with self.assertRaises(ValueError):
            estimate_tabletop_grasp(np.eye(4), object_yaw_rad=np.nan)


if __name__ == '__main__':
    unittest.main()
