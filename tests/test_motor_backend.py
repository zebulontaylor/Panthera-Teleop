"""Motor ownership and frontend isolation without devices, SDK, or cameras."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from motor_backend import MotorEngine, MotorServer, Handler, SETTINGS
from motor_client import MotorClient


def arm(position=0.):
    return dict(position_rad=[position]*6, velocity_rad_s=[0.]*6,
                gravity_model_torque_nm=[.2]*6, gravity_torque_command_nm=[.2]*6,
                command={}, gripper={'position': 0., 'velocity': 0., 'torque': 0.})


def robot():
    result = Mock()
    result.joint_limits = {'lower': np.full(6, -2.), 'upper': np.full(6, 2.)}
    result.gripper_limits = {'lower': 0., 'upper': 2.}
    return result


def frame():
    return {s: dict(joint=dict(mode='position', position_rad=[.1]*6,
                             speed_rad_s=.2, torque_nm=[1.]*6),
                    gripper={'mode': 'passive'}) for s in ('left', 'right')}


class MotorEngineTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.measured = {s: arm() for s in ('left', 'right')}
        self.robots = {s: robot() for s in self.measured}
        self.engine = MotorEngine(self.robots, json.loads(SETTINGS.read_text()), clock=lambda: self.now)
        self.feedback = patch('motor_backend.feedback', side_effect=lambda r, s, stale:
                              (np.array([.2]*6), deepcopy(self.measured[s])))
        self.feedback.start()
        self.addCleanup(self.feedback.stop)
        self.engine.tick()
        self.token = self.call('acquire', name='test')['token']

    def call(self, operation, **kwargs):
        return self.engine.dispatch(dict(version=1, operation=operation,
            token=getattr(self, 'token', None), **kwargs))

    def test_backend_exclusively_owns_control(self):
        with self.assertRaisesRegex(RuntimeError, 'Another frontend'):
            self.call('acquire', name='other')
        with self.assertRaises(RuntimeError):
            self.engine.dispatch(dict(version=1, operation='command', token='wrong', commands=frame()))
        self.call('command', commands=frame())
        self.engine.tick()
        self.assertEqual(self.engine.mode, 'client_control')

    def test_bad_frames_are_atomic_and_do_not_renew_lease(self):
        good = frame()
        self.call('command', commands=good)
        renewed = self.engine.renewed
        bad = []
        bad.append({'left': good['left']})
        for field, value in [('speed_rad_s', 100.), ('position_rad', [3.]*6),
                             ('position_rad', [float('nan')]*6), ('torque_nm', [99.]*6),
                             ('mode', 'arbitrary_sdk_call')]:
            command = deepcopy(good)
            command['right']['joint'][field] = value
            bad.append(command)
        command = deepcopy(good)
        command['right']['gripper'] = dict(mode='position', position_rad=3., speed_rad_s=1., torque_nm=.2)
        bad.append(command)
        self.now += .1
        for command in bad:
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.call('command', commands=command)
            self.assertEqual(self.engine.commands, good)
            self.assertEqual(self.engine.renewed, renewed)

    def test_expired_frontend_holds_measured_pose_and_cannot_resume(self):
        self.call('command', commands=frame())
        self.engine.tick()
        self.measured['left'] = arm(.04)
        self.now += .6
        self.engine.tick()
        np.testing.assert_array_equal(self.robots['left'].Joint_Pos_Vel.call_args.args[0], [.04]*6)
        self.assertIsNone(self.engine.owner)
        self.assertEqual(self.engine.mode, 'holding')
        with self.assertRaises(RuntimeError):
            self.call('command', commands=frame())
        self.measured['left'] = arm(.03)
        self.engine.tick()
        np.testing.assert_array_equal(self.robots['left'].Joint_Pos_Vel.call_args.args[0], [.04]*6)
        self.call('acquire', name='replacement')

    def test_late_frame_cannot_renew_an_expired_lease_before_next_tick(self):
        self.call('command', commands=frame())
        self.now += .6
        with self.assertRaises(RuntimeError):
            self.call('command', commands=frame())
        self.assertIsNone(self.engine.owner)

    def test_release_preserves_powered_backend_hold(self):
        self.call('command', commands=frame())
        self.call('release')
        self.engine.tick()
        self.robots['left'].set_stop.assert_not_called()
        self.assertEqual(self.engine.mode, 'holding')

    def test_gravity_is_recomputed_on_backend_ticks(self):
        command = frame()
        for side in command:
            command[side]['joint'] = dict(mode='gravity', assist_torque_nm=[.1]*6)
        self.call('command', commands=command)
        self.engine.tick()
        self.engine.tick()
        np.testing.assert_allclose(self.robots['left'].pos_vel_tqe_kp_kd.call_args.args[2], [.3]*6)
        self.assertEqual(self.robots['left'].pos_vel_tqe_kp_kd.call_count, 2)

    def test_feedback_fault_stops_both_arms_and_blocks_new_commands(self):
        self.feedback.stop()
        with patch('motor_backend.feedback', side_effect=RuntimeError('motor fault')):
            self.engine.tick()
        self.assertEqual(self.engine.mode, 'fault')
        for item in self.robots.values():
            item.set_stop.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'motor fault'):
            self.call('command', commands=frame())


class MotorSocketTests(unittest.TestCase):
    def test_frontend_process_crash_does_not_stop_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motor.sock'
            settings = json.loads(SETTINGS.read_text())
            settings['command_lease_seconds'] = .1
            engine = MotorEngine({s: robot() for s in ('left', 'right')}, settings)
            stopping = threading.Event()
            def feedback(r, s, stale):
                return np.array([.2]*6), arm()
            with patch('motor_backend.feedback', side_effect=feedback):
                engine.tick()
                with MotorServer(str(path), Handler) as server:
                    server.engine = engine
                    server.stopping = stopping
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    def loop():
                        while not stopping.is_set():
                            engine.tick()
                            stopping.wait(.005)
                    worker = threading.Thread(target=loop)
                    worker.start()
                    try:
                        code = '''import os, sys
sys.path.insert(0, sys.argv[1])
from motor_client import MotorClient
client = MotorClient('crashing frontend', sys.argv[2]).open(autostart=False)
client.rpc('command', commands=__import__('json').loads(sys.argv[3]))
os._exit(0)
'''
                        result = subprocess.run([sys.executable, '-c', code, str(ROOT/'app'), str(path),
                                                 json.dumps(frame())], capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        previous = engine.sequence
                        deadline = time.monotonic() + 2
                        while engine.owner and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertIsNone(engine.owner)
                        self.assertGreater(engine.sequence, previous)
                        self.assertEqual(engine.mode, 'holding')
                        client = MotorClient('replacement', path).open(autostart=False)
                        try:
                            samples = client.read()
                            for side, handle in client.robots.items():
                                handle.pos_vel_tqe_kp_kd(np.zeros(6), np.zeros(6), samples[side][0], np.zeros(6), np.zeros(6))
                                handle.gripper_control_MIT(0, 0, 0, 0, 0)
                            self.assertTrue(client.flush()['accepted'])
                            client.release()
                            self.assertEqual(client.rpc('status')['mode'], 'holding')
                        finally:
                            client.close()
                    finally:
                        stopping.set()
                        worker.join(2)
                        server.shutdown()
                        thread.join(2)


if __name__ == '__main__':
    unittest.main()
