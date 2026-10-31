"""Offline model, geometry, runtime and HTTP checks. No robot device access."""
import copy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from bounded_alignment_trial import ConstrainedVisualNudge, StereoPoseNudge
from constrained_visual_nudge import constrained_visual_proposal, checked_translation
from stereo_pose_nudge import triangulate, stereo_residual, stereo_pose_proposal, validate_calibration
from test_cartesian_nudge import fixture, LinearKinematics
from test_alignment_assist import advance


def geometry():
    config, snapshot, arms, limits, matrix = fixture()
    arms['right'] = copy.deepcopy(arms['right'])
    k = [[500., 0, 320], [0, 500., 240], [0, 0, 1.]]
    poses = []
    cameras = {}
    for name, side, x in [('left_wrist', 'left', -.05), ('right_wrist', 'right', .05)]:
        pose = np.eye(4); pose[0, 3] = x; poses.append(pose)
        cameras[name] = dict(side=side, intrinsics=k, distortion=[0.]*5, T_tool_camera=pose.tolist())
    cal = dict(verified=True, checkpoint_id=config['calibration']['checkpoint_id'],
        camera_mapping=config['calibration']['camera_mapping'], image_size=[640, 480],
        target_offset_left_base_m=[0.]*3, T_left_base_right_base=np.eye(4).tolist(), cameras=cameras)
    snapshot['geometry_context'] = dict(state_monotonic_ns=snapshot['monotonic_ns'],
        tool_poses={side: arms[side]['tool_pose_base'] for side in ('left', 'right')})
    world = [np.array([.001, .002, .5]), np.array([0., 0., .5])]
    for ci, pose in zip((1, 2), poses):
        for li, point in enumerate(world):
            local = np.linalg.inv(pose) @ np.r_[point, 1.]
            pixel = np.asarray(k) @ local[:3]
            snapshot['predictions']['gear']['keypoints_2d'][ci][li] = (pixel[:2]/pixel[2]).tolist()
    return config, snapshot, arms, limits, cal, poses


class AdditionalNudgeTests(unittest.TestCase):
    def test_constrained_solution_respects_forward_axis(self):
        _, _, _, limits, _ = fixture()
        j = np.zeros((4, 6)); j[1, 0] = 10000; j[3, 2] = 10000
        result = constrained_visual_proposal(np.zeros(6), [0, 10, 0, 10], j,
            LinearKinematics(), limits, {'forward_axis_base': [1, 0, 0]})
        self.assertGreaterEqual(result['delta_xyz_m'][0], -1e-9)
        self.assertLessEqual(np.linalg.norm(result['delta_xyz_m']), .000500001)
        self.assertLess(np.linalg.norm(result['predicted_residual']), np.sqrt(200))

    def test_retreat_and_invalid_axis_block(self):
        _, _, _, limits, _ = fixture()
        for settings in ({'forward_axis_base': [1, 0, 0]}, {'forward_axis_base': None}):
            with self.assertRaises(ValueError):
                checked_translation(np.zeros(6), [-.0002, 0, 0], LinearKinematics(), limits, settings)

    def test_stereo_recovers_metric_relative_translation(self):
        config, snapshot, _, _, cal, _ = geometry()
        validate_calibration(cal, config['calibration'])
        np.testing.assert_allclose(stereo_residual(snapshot, cal), [.001, .002, 0], atol=1e-10)

    def test_stereo_checks_checkpoint_calibration_and_timestamp(self):
        config, snapshot, _, _, cal, _ = geometry()
        bad = copy.deepcopy(cal); bad['verified'] = False
        with self.assertRaisesRegex(ValueError, 'calibration_required'):
            validate_calibration(bad, config['calibration'])
        bad = copy.deepcopy(cal); bad['checkpoint_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checkpoint_id_mismatch'):
            validate_calibration(bad, config['calibration'])
        snapshot['geometry_context']['state_monotonic_ns'] += 100_000_000
        with self.assertRaisesRegex(ValueError, 'timestamp_skew'):
            stereo_residual(snapshot, cal)

    def test_degenerate_and_inconsistent_stereo_block(self):
        _, snapshot, _, _, cal, poses = geometry()
        cameras = list(cal['cameras'].values())
        with self.assertRaisesRegex(ValueError, 'parallax'):
            triangulate([[320, 240], [320, 240]], [poses[0], poses[0]], cameras)
        snapshot['predictions']['gear']['keypoints_2d'][2][0][1] += 30
        with self.assertRaisesRegex(ValueError, 'reprojection'):
            stereo_residual(snapshot, cal)

    def test_pose_step_is_bounded_and_retreat_rejected(self):
        _, _, _, limits, _ = fixture()
        result = stereo_pose_proposal(np.zeros(6), [.001, 0, -.002], LinearKinematics(), limits,
            {'forward_axis_base': [0, 0, 1]})
        self.assertLessEqual(np.linalg.norm(result['delta_xyz_m']), .000500001)
        with self.assertRaisesRegex(ValueError, 'cartesian_step_constraint'):
            stereo_pose_proposal(np.zeros(6), [0, 0, .002], LinearKinematics(), limits,
                {'forward_axis_base': [0, 0, 1]})

    def test_missing_calibration_and_forward_axis_never_step(self):
        for cls in (ConstrainedVisualNudge, StereoPoseNudge):
            config, snapshot, arms, limits, _ = fixture()
            controller = cls(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
            np.testing.assert_array_equal(controller.tick(10.1, snapshot, arms, allowed=True), np.zeros(6))
            self.assertEqual(controller.reason, 'measured_forward_axis_required')
            self.assertEqual(controller.steps, 0)
        config['experiment'] = dict(axis_verified=True, forward_axis_base=[0, 0, 1])
        controller = StereoPoseNudge(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
        controller.tick(10.1, snapshot, arms, allowed=True)
        self.assertEqual(controller.reason, 'stereo_calibration_required')

    def test_constrained_runtime_never_repeats_step(self):
        config, snapshot, arms, limits, matrix = fixture()
        matrix[1] = [10000, 0, 0, 0, 0, 0]
        matrix[3] = [0, 0, 10000, 0, 0, 0]
        config['calibration']['image_jacobian_px_per_rad'] = matrix.tolist()
        config['experiment'] = dict(axis_verified=True, forward_axis_base=[1, 0, 0])
        controller = ConstrainedVisualNudge(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
        for i in range(100):
            now = 10+i*.05
            advance(snapshot, now)
            q = np.asarray(arms['left']['position_rad'])
            points = snapshot['predictions']['gear']['keypoints_2d']
            points[1][0][1] = 110+10000*q[0]
            points[2][0][1] = 110+10000*q[2]
            target = controller.tick(now, snapshot, arms, allowed=True)
            arms['left']['position_rad'] = target.tolist()
            arms['left']['tool_pose_base']['position_m'] = target[:3].tolist()
        self.assertEqual(controller.steps, 1)
        self.assertEqual(controller.reason, 'single_step_observed_holding')

    def test_each_dashboard_is_separate_and_rejects_legacy_torque(self):
        import dashboard_constrained
        import dashboard_stereo
        for module, label in ((dashboard_constrained, 'One constrained visual nudge and hold'),
                              (dashboard_stereo, 'One 3D position nudge and hold')):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            store = Mock(); store.snapshot.return_value = {}
            dashboard = module.Dashboard(store, None, port, demo=True)
            try:
                with urlopen(dashboard.url) as response:
                    self.assertIn(label, response.read().decode())
                request = Request(dashboard.url+'/api/alignment-nudge', data=json.dumps({}).encode(),
                    headers={'Content-Type': 'application/json', 'X-Session-Token': dashboard.token})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 409)
                error.exception.close()
                store.submit.assert_not_called()
            finally:
                dashboard.close()

    def test_stereo_runtime_one_measured_step_then_hold(self):
        config, snapshot, arms, limits, cal, _ = geometry()
        config['experiment'] = dict(axis_verified=True, forward_axis_base=[0, 0, 1], stereo_calibration=cal)
        controller = StereoPoseNudge(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
        for i in range(100):
            now = 10+i*.05
            advance(snapshot, now)
            q = np.asarray(arms['left']['position_rad'])
            snapshot['geometry_context'] = dict(state_monotonic_ns=snapshot['monotonic_ns'],
                tool_poses=copy.deepcopy({side: arms[side]['tool_pose_base'] for side in ('left', 'right')}))
            for ci, name in zip((1, 2), ('left_wrist', 'right_wrist')):
                camera = cal['cameras'][name]
                pose = np.asarray(camera['T_tool_camera']).copy()
                if camera['side'] == 'left': pose[:3, 3] += q[:3]
                for li, world in enumerate((np.array([.001, .002, .5])+q[:3], np.array([0., 0., .5]))):
                    local = np.linalg.inv(pose) @ np.r_[world, 1.]
                    pixel = np.asarray(camera['intrinsics']) @ local[:3]
                    snapshot['predictions']['gear']['keypoints_2d'][ci][li] = (pixel[:2]/pixel[2]).tolist()
            target = controller.tick(now, snapshot, arms, allowed=True)
            arms['left']['position_rad'] = target.tolist()
            arms['left']['tool_pose_base']['position_m'] = target[:3].tolist()
        self.assertEqual(controller.steps, 1)
        self.assertEqual(controller.reason, 'single_step_observed_holding')
        self.assertLessEqual(np.linalg.norm(target), .000500001)

    def test_measured_retreat_aborts_both_versions(self):
        for cls in (ConstrainedVisualNudge, StereoPoseNudge):
            config, snapshot, arms, limits, _ = fixture()
            config['experiment'] = dict(axis_verified=True, forward_axis_base=[1, 0, 0])
            controller = cls(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
            arms['left']['position_rad'][0] = -.0002
            arms['left']['tool_pose_base']['position_m'][0] = -.0002
            with self.assertRaisesRegex(ValueError, 'measured_retreat'):
                controller.tick(10.1, snapshot, arms, allowed=True)


if __name__ == '__main__':
    unittest.main()
