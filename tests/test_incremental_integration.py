"""HTTP and launcher integration without opening cameras or robot devices."""
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from dashboard_cartesian import Dashboard


class IncrementalDashboardTests(unittest.TestCase):
    def dashboard(self, incremental, relaxed=False):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        store = Mock()
        store.snapshot.return_value = dict(active='episode', phase='recording', control_active=True,
            alignment_assist=dict(enabled=True, mode='operator_supervised'), gear_grasp=dict(settled=True))
        dashboard = Dashboard(store, None, port, demo=True, incremental=incremental, relaxed=relaxed)
        self.addCleanup(dashboard.close)
        return dashboard, store

    def post(self, dashboard, path, payload, token=True):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['X-Session-Token'] = dashboard.token
        return urlopen(Request(dashboard.url + path, data=json.dumps(payload).encode(), headers=headers))

    def test_opt_in_page_and_default_one_step_page(self):
        for incremental, label in [(True, 'Align vertically with repeated nudges'),
                                   (False, 'One vertical nudge and hold')]:
            dashboard, _ = self.dashboard(incremental)
            with urlopen(dashboard.url) as response:
                html = response.read().decode()
            self.assertIn(label, html)
            if incremental:
                self.assertIn('0.5 mm per step, 2 mm total path, 20 steps or 30 seconds', html)
                self.assertIn('both wrist views are within 3 pixels', html)
                self.assertIn('servo.tool_path_m', html)
                self.assertIn('servo.motor_command_active===false', html)
                self.assertIn('Observing · gravity assist', html)
                self.assertIn('servo.reason', html)
                self.assertIn('Stop nudges / release position hold', html)

    def test_start_and_release_keep_existing_authenticated_queue(self):
        dashboard, store = self.dashboard(True)
        with self.post(dashboard, '/api/alignment-servo', {'enabled': True}) as response:
            self.assertEqual(response.status, 202)
        kind, payload = store.submit.call_args.args
        self.assertEqual(kind, 'alignment_servo')
        self.assertTrue(payload['enabled'])
        self.assertEqual(payload['episode_id'], 'episode')
        store.snapshot.return_value = {}
        with self.post(dashboard, '/api/alignment-servo', {'enabled': False}) as response:
            self.assertEqual(response.status, 202)
        self.assertFalse(store.submit.call_args.args[1]['enabled'])
        with self.assertRaises(HTTPError) as error:
            self.post(dashboard, '/api/alignment-servo', {'enabled': False}, token=False)
        self.assertEqual(error.exception.code, 403)
        error.exception.close()

    def test_legacy_torque_and_widening_are_rejected_on_both_routes(self):
        dashboard, store = self.dashboard(True)
        requests = [('/api/alignment-nudge', {'held': True, 'press_id': 'x', 'episode_id': 'episode'}),
                    ('/api/alignment-trial-settings', {'settings': {'max_error_px': 100}}),
                    ('/api/commands', {'command': 'alignment.nudge', 'parameters': {'held': True}}),
                    ('/api/commands', {'command': 'alignment.settings', 'parameters': {'settings': {}}})]
        for path, payload in requests:
            with self.assertRaises(HTTPError) as error:
                self.post(dashboard, path, payload)
            self.assertEqual(error.exception.code, 409)
            error.exception.close()
        store.submit.assert_not_called()

    def test_incremental_kinematics_uses_backend_configuration(self):
        from hand_guide import make_alignment_controller
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'backend-left.yaml'
            config_path.write_text('urdf: {file_path: backend-selected.urdf}')
            with patch('cartesian_nudge.ToolKinematics') as kinematics, \
                 patch('incremental_nudge.IncrementalNudge') as controller:
                result = make_alignment_controller({}, 1., {}, {}, [1]*6,
                    incremental=True, robot_config_path=config_path)
                kinematics.assert_called_once_with({'urdf': {'file_path': 'backend-selected.urdf'}})
                controller.assert_called_once_with({'encoder_resolution_rad': 2*math.pi/10000}, 1., {}, {}, [1]*6,
                    kinematics=kinematics.return_value)
                self.assertIs(result, controller.return_value)
        with patch('hand_guide.AlignmentServo') as legacy:
            self.assertIs(make_alignment_controller({}, 1., {}, {}, [1]*6), legacy.return_value)
            legacy.assert_called_once_with({}, 1., {}, {}, [1]*6)

    def test_incremental_preflight_blocks_before_controller_or_hold(self):
        from hand_guide import start_alignment_controller
        arms = {'left': {'arm': 'left'}, 'right': {'arm': 'right'}}
        with patch('hand_guide.AlignmentAssist') as assist, \
             patch('hand_guide.make_alignment_controller') as factory:
            assist.return_value.evaluate.return_value = dict(reason='outside_calibration_envelope',
                active=False, error_px=[1, 2, 3, 4], target_distance_px=41.)
            controller, status = start_alignment_controller({}, 1., {}, arms, {}, [1]*6,
                incremental=True, robot_config_path='unused')
            self.assertIsNone(controller)
            factory.assert_not_called()
            self.assertEqual(status['stage'], 'preflight')
            self.assertEqual(status['steps'], 0)
            self.assertFalse(status['active'])
            self.assertFalse(status['correcting'])
            self.assertEqual(status['reason'], 'outside_calibration_envelope')
            self.assertEqual(status['error_px'], [1, 2, 3, 4])

    def test_preflight_accepts_ready_and_aligned_and_skips_legacy(self):
        from hand_guide import start_alignment_controller
        with patch('hand_guide.AlignmentAssist') as assist, \
             patch('hand_guide.make_alignment_controller') as factory:
            factory.return_value.status.return_value = {'active': True}
            for reason in ('nudging', 'aligned'):
                assist.return_value.evaluate.return_value = dict(reason=reason)
                controller, status = start_alignment_controller({}, 1., {}, {'left': {}, 'right': {}},
                    {}, [1]*6, incremental=True, robot_config_path='backend-left.yaml')
                self.assertIs(controller, factory.return_value)
                self.assertEqual(status, {'active': True})
            assist.reset_mock()
            start_alignment_controller({}, 1., {}, {}, {}, [1]*6)
            assist.assert_not_called()
            factory.assert_called_with({}, 1., {}, {}, [1]*6,
                incremental=False, robot_config_path=None)

    def test_exception_keeps_motion_and_visual_diagnostics(self):
        from hand_guide import alignment_exception_status
        controller = Mock()
        controller.status.return_value = dict(active=True, correcting=True, steps=4,
            error_px=[0, 5, 0, 7], tool_path_m=.0006, reason='nudging')
        status = alignment_exception_status(controller, ValueError('velocity_limit'))
        self.assertEqual(status['steps'], 4)
        self.assertEqual(status['tool_path_m'], .0006)
        self.assertEqual(status['error_px'], [0, 5, 0, 7])
        self.assertEqual(status['reason'], 'velocity_limit')
        self.assertFalse(status['active'])
        self.assertFalse(status['correcting'])
        self.assertTrue(controller.status.return_value['active'])

    def test_relaxed_profile_is_session_copy_used_by_preflight_and_controller(self):
        from hand_guide import effective_nudge_config, start_alignment_controller
        original = {'calibration': {'max_joint_distance_rad': .12}, 'max_error_px': 60}
        strict = effective_nudge_config(original)
        relaxed = effective_nudge_config(original, relaxed=True)
        self.assertEqual(strict, original)
        self.assertIsNot(strict['calibration'], original['calibration'])
        self.assertEqual(original['calibration']['max_joint_distance_rad'], .12)
        self.assertEqual(relaxed['calibration']['max_joint_distance_rad'], .4)
        self.assertEqual(relaxed['incremental_profile'], 'supervised_relaxed')
        self.assertTrue(relaxed['incremental_calibration_unverified'])
        with patch('incremental_nudge.evaluate_incremental_evidence', create=True) as evidence, \
             patch('hand_guide.make_alignment_controller') as factory:
            evidence.return_value = {'reason': 'nudging'}
            start_alignment_controller(relaxed, 1., {}, {'left': {}, 'right': {}}, {}, [1]*6,
                                       incremental=True)
            self.assertIs(evidence.call_args.args[0], relaxed)
            self.assertIs(factory.call_args.args[0], relaxed)

    def test_relaxed_engaged_grasp_does_not_require_transient_settling(self):
        from hand_guide import nudge_grasp_authorized
        grasp = Mock(episode='episode', settled=False)
        self.assertTrue(nudge_grasp_authorized(grasp, 'episode', relaxed=True))
        self.assertFalse(nudge_grasp_authorized(grasp, 'episode'))
        self.assertFalse(nudge_grasp_authorized(grasp, 'new-episode', relaxed=True))
        grasp.episode = None
        self.assertFalse(nudge_grasp_authorized(grasp, 'episode', relaxed=True))
        dashboard, store = self.dashboard(True, relaxed=True)
        store.snapshot.return_value['gear_grasp'] = {'active': True, 'settled': False}
        with self.post(dashboard, '/api/alignment-servo', {'enabled': True}) as response:
            self.assertEqual(response.status, 202)
        store.snapshot.return_value['gear_grasp']['active'] = False
        with self.assertRaises(HTTPError) as error:
            self.post(dashboard, '/api/alignment-servo', {'enabled': True})
        self.assertEqual(error.exception.code, 409)
        error.exception.close()

    def test_relaxed_dashboard_and_flag_guard(self):
        dashboard, _ = self.dashboard(True, relaxed=True)
        with urlopen(dashboard.url) as response:
            html = response.read().decode()
        self.assertIn('Supervised relaxed alignment: distance/speed limits, unverified calibration', html)
        self.assertIn('5 mm in 3D per step and 50 mm cumulative travel/displacement', html)
        self.assertIn('const relaxedNudge=true;', html)
        self.assertIn('servo?.minimum_encoder_step_ticks', html)
        self.assertIn('Holds within 3 px in both views', html)
        self.assertIn('No experimental response, calibration-envelope, orientation, step-count or duration stops', html)
        with urlopen(dashboard.url + '/api/status') as response:
            self.assertEqual(json.load(response)['incremental_profile'], 'supervised_relaxed')
        result = subprocess.run([sys.executable, str(ROOT / 'app/hand_guide.py'), '--relaxed-nudge'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--relaxed-nudge requires --incremental', result.stderr)

    def test_launcher_selects_incremental_runtime_without_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / 'start-incremental-nudge.sh'
            launcher.write_text((ROOT / launcher.name).read_text())
            interpreter = root / 'work/teleop-venv/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            interpreter.chmod(0o755)
            result = subprocess.run(['bash', str(launcher), '--demo', '--no-browser'],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), ['-u', str(root / 'app/hand_guide.py'),
                '--run', '--incremental', '--relaxed-nudge', '--alignment-assist', '--demo', '--no-browser'])
        result = subprocess.run([sys.executable, str(ROOT / 'app/hand_guide.py'), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--incremental', result.stdout)


if __name__ == '__main__':
    unittest.main()
