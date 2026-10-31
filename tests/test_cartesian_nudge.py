import sys
import unittest
from pathlib import Path
import json
import socket
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from cartesian_nudge import CartesianNudge, vertical_proposal
from cartesian_nudge_mode import PreviewOnlyAlignmentAssist, CartesianCommands
from test_supervised_alignment import supervised
from test_alignment_assist import advance


class LinearKinematics:
    def pose(self, q):
        return np.asarray(q[:3]), np.eye(3)

    def jacobian(self, q):
        return np.eye(6)


def fixture():
    c, s, a, limits = supervised()
    a.update(gravity_model_torque_nm=[0.]*6,
             tool_pose_base=dict(position_m=[0.]*3, rotation=np.eye(3).tolist()))
    other = dict(a, position_rad=[0.]*6)
    # Matching positive vertical sensitivity in the two views.
    matrix = np.zeros((4, 6)); matrix[1, 2] = matrix[3, 2] = 10000
    matrix[0, 0] = matrix[2, 1] = 1
    c.config['calibration']['side'] = 'left'
    c.config['calibration']['image_jacobian_px_per_rad'] = matrix.tolist()
    return c.config, s, dict(left=a, right=other), limits, matrix


class CartesianNudgeTests(unittest.TestCase):
    def test_tool_direction_is_vertical_and_bounded(self):
        _, _, _, limits, matrix = fixture()
        result = vertical_proposal(np.zeros(6), [0, 10, 0, 10], matrix, LinearKinematics(), limits)
        np.testing.assert_allclose(result['target'], [0, 0, -.0005, 0, 0, 0])

    def test_conflicting_views_block(self):
        _, _, _, limits, matrix = fixture()
        with self.assertRaisesRegex(ValueError, 'direction_disagreement'):
            vertical_proposal(np.zeros(6), [0, 10, 0, -10], matrix, LinearKinematics(), limits)

    def test_bad_kinematics_and_joint_limits_block(self):
        _, _, _, limits, matrix = fixture()
        class Singular(LinearKinematics):
            def jacobian(self, q): return np.zeros((6, 6))
        with self.assertRaisesRegex(ValueError, 'singularity'):
            vertical_proposal(np.zeros(6), [0, 10, 0, 10], matrix, Singular(), limits)
        limits = dict(lower=[-.0301]*6, upper=[1.]*6)
        with self.assertRaisesRegex(ValueError, 'joint_target_limit'):
            vertical_proposal(np.zeros(6), [0, 10, 0, 10], matrix, LinearKinematics(), limits)

    def test_only_one_step_even_with_persistent_error(self):
        config, s, arms, limits, _ = fixture()
        controller = CartesianNudge(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
        for i in range(80):
            now = 10+i*.05
            advance(s, now)
            q = np.asarray(arms['left']['position_rad'])
            points = s['predictions']['gear']['keypoints_2d']
            points[1][0][1] = points[2][0][1] = 110+10000*q[2]
            target = controller.tick(now, s, arms, allowed=True)
            arms['left']['position_rad'] = target.tolist()
            arms['left']['tool_pose_base']['position_m'] = target[:3].tolist()
        self.assertEqual(controller.steps, 1)
        self.assertFalse(controller.running)
        self.assertEqual(controller.reason, 'single_step_observed_holding')
        self.assertLessEqual(np.linalg.norm(target), .000500001)

    def test_measured_retreat_and_model_mismatch_block(self):
        config, s, arms, limits, _ = fixture()
        controller = CartesianNudge(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
        arms['left']['position_rad'][0] = .0003
        arms['left']['tool_pose_base']['position_m'][0] = .0003
        with self.assertRaisesRegex(ValueError, 'cartesian_bound'):
            controller.tick(10., s, arms, allowed=True)
        arms['left']['tool_pose_base']['position_m'][0] = 0
        with self.assertRaisesRegex(ValueError, 'model_mismatch'):
            controller.tick(10., s, arms, allowed=True)

    def test_missing_or_repeated_evidence_never_steps(self):
        config, s, arms, limits, _ = fixture()
        controller = CartesianNudge(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
        for i in range(110):
            controller.tick(10+i*.05, s, arms, allowed=True)
        self.assertEqual(controller.steps, 0)
        self.assertFalse(controller.running)

    def test_legacy_torque_is_suppressed_and_route_rejected(self):
        c, s, a, limits = supervised()
        result = PreviewOnlyAlignmentAssist(c.config).evaluate(10., s, a, limits,
            other_arm=dict(a, position_rad=[0.]*6), collecting=True,
            operator_confirmed=True, operator_press='test')
        self.assertFalse(result['active'])
        self.assertEqual(result['torque_nm'], [0.]*6)
        with self.assertRaisesRegex(ValueError, 'Legacy torque pulse disabled'):
            CartesianCommands(None).dispatch('alignment.nudge', {})

    def test_alternative_dashboard_disables_both_legacy_command_routes(self):
        from dashboard_cartesian import Dashboard
        store = Mock()
        store.snapshot.return_value = dict(active='trial', phase='recording', control_active=True,
            alignment_assist={'mode': 'operator_supervised'}, gear_grasp={'settled': True})
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        dashboard = Dashboard(store, None, port, demo=True)
        self.addCleanup(dashboard.close)
        url = 'http://127.0.0.1:'+str(dashboard.server.server_address[1])
        with urlopen(url) as response:
            html = response.read().decode()
            self.assertIn('One vertical nudge and hold', html)
            self.assertIn('1 mm limit', html)
        for endpoint, payload in (
            ('/api/alignment-nudge', dict(held=True, press_id='a', episode_id='trial')),
            ('/api/commands', dict(command='alignment.nudge', parameters=dict(held=True, press_id='a', episode_id='trial')))):
            request = Request(url+endpoint, data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json', 'X-Session-Token': dashboard.token})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 409)
            error.exception.close()
        store.submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
