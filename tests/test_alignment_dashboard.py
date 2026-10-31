"""Exercise authenticated alignment toggle requests without cameras or motors."""
import importlib.util
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
spec = importlib.util.spec_from_file_location('alignment_dashboard', ROOT/'app/dashboard.py')
dashboard_module = importlib.util.module_from_spec(spec)
try:
    import cv2
except ImportError:
    # The HTTP handler has no OpenCV dependency; capture is deliberately unused.
    with patch.dict(sys.modules, {'cv2': Mock()}):
        spec.loader.exec_module(dashboard_module)
else:
    spec.loader.exec_module(dashboard_module)


class AlignmentDashboardTests(unittest.TestCase):
    def setUp(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.store = Mock()
        self.store.snapshot.return_value = {'active': None, 'alignment_assist': {'enabled': False}}
        self.dashboard = dashboard_module.Dashboard(self.store, None, port, True)
        self.addCleanup(self.dashboard.close)

    def request(self, payload, token=True, origin=None, endpoint='/api/alignment-assist'):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['X-Session-Token'] = self.dashboard.token
        if origin:
            headers['Origin'] = origin
        return Request(self.dashboard.url+endpoint,
                       data=json.dumps(payload).encode(), headers=headers)

    def test_toggle_queues_explicit_boolean_without_active_episode(self):
        for enabled in (True, False):
            with urlopen(self.request({'enabled': enabled})) as response:
                self.assertEqual(response.status, 202)
            self.store.submit.assert_called_with('alignment_assist', {'enabled': enabled})

    def test_rejects_invalid_boolean_and_non_object(self):
        for payload in ({'enabled': 'false'}, {'enabled': 1}, {}, [], None):
            with self.assertRaises(HTTPError) as error:
                urlopen(self.request(payload))
            self.assertEqual(error.exception.code, 409)
            error.exception.close()
        self.store.submit.assert_not_called()

    def test_requires_session_token_and_trusted_origin(self):
        for request in (self.request({'enabled': True}, token=False),
                        self.request({'enabled': True}, origin='https://example.com')):
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
        self.store.submit.assert_not_called()

    def test_supervised_press_is_bound_to_current_recording(self):
        self.store.snapshot.return_value = dict(active='trial', phase='recording',
            alignment_assist={'mode':'operator_supervised'})
        payload = dict(held=True, press_id='press', episode_id='trial')
        with urlopen(self.request(payload, endpoint='/api/alignment-nudge')) as response:
            self.assertEqual(response.status, 202)
        kind, data = self.store.submit.call_args.args
        self.assertEqual(kind, 'alignment_operator_hold')
        self.assertGreater(data['received_monotonic_ns'], 0)
        for change in (dict(episode_id='old'), dict(held=1), dict(press_id='')):
            with self.assertRaises(HTTPError) as error:
                urlopen(self.request(dict(payload, **change), endpoint='/api/alignment-nudge'))
            self.assertEqual(error.exception.code, 409)
            error.exception.close()

    def test_gear_grasp_requires_supervised_active_recording(self):
        payload = dict(action='close', episode_id='trial')
        with self.assertRaises(HTTPError) as error:
            urlopen(self.request(payload, endpoint='/api/gear-grasp'))
        self.assertEqual(error.exception.code,409)
        error.exception.close()
        self.store.snapshot.return_value = dict(active='trial',phase='recording',control_active=True,
            alignment_assist={'mode':'operator_supervised'})
        with urlopen(self.request(payload, endpoint='/api/gear-grasp')) as response:
            self.assertEqual(response.status,202)
        kind,data = self.store.submit.call_args.args
        self.assertEqual(kind,'gear_grasp')
        self.assertEqual(data['action'],'close')
        self.assertNotIn('right',data)

    def test_trial_settings_are_bounded_and_cannot_change_motion_caps(self):
        self.store.snapshot.return_value = dict(alignment_assist={'mode':'operator_supervised','active':False})
        with urlopen(self.request({'settings':{'max_joint_distance_rad':.35}}, endpoint='/api/alignment-trial-settings')) as response:
            self.assertEqual(response.status,202)
        self.store.submit.assert_called_with('alignment_trial_settings', {'max_joint_distance_rad':.35})
        for settings in ({'max_joint_distance_rad':.5},{'max_error_px':101.},{'max_torque_nm':1.},{'max_error_px':True}):
            with self.assertRaises(HTTPError) as error:
                urlopen(self.request({'settings':settings}, endpoint='/api/alignment-trial-settings'))
            self.assertEqual(error.exception.code,409)
            error.exception.close()

    def test_servo_requires_settled_grasp_and_release_is_always_available(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.request({'enabled':True},endpoint='/api/alignment-servo'))
        self.assertEqual(error.exception.code,409);error.exception.close()
        self.store.snapshot.return_value=dict(phase='recording',active='trial',control_active=True,
            alignment_assist={'mode':'operator_supervised'},gear_grasp={'settled':True})
        with urlopen(self.request({'enabled':True},endpoint='/api/alignment-servo')) as response:
            self.assertEqual(response.status,202)
        self.assertEqual(self.store.submit.call_args.args[0],'alignment_servo')
        self.store.snapshot.return_value={}
        with urlopen(self.request({'enabled':False},endpoint='/api/alignment-servo')) as response:
            self.assertEqual(response.status,202)


if __name__ == '__main__':
    unittest.main()
