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
from motor_client import MotorArm


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


class GuardedGravityTests(MotorEngineTests):
    """The independent motor loop, not the frontend, enforces load bounds."""
    def setUp(self):
        super().setUp()
        for sample in self.measured.values():
            sample['tool_pose_base'] = dict(position_m=[.2, .1, .1], rotation=np.eye(3).tolist())
        for _ in range(7):
            self.now += .01
            self.engine.tick()
        for item in self.robots.values():
            item.reset_mock()

    def guarded_frame(self, identity='load-1', **limits):
        command = frame()
        for side in command:
            command[side]['gripper'] = dict(mode='position', position_rad=.3 if side == 'left' else .7,
                speed_rad_s=.2, torque_nm=.2)
        guard = dict(id=identity, anchor_tool_position_m=[.2, .1, .1],
            anchor_tool_rotation=np.eye(3).tolist(), max_xy_m=.008, max_down_m=.005,
            max_up_m=.002, max_speed_m_s=.02, max_angular_rad=.1, max_duration_s=30.)
        guard.update(limits)
        command['left']['joint'] = dict(mode='guarded_gravity', assist_torque_nm=[.1]*6, guard=guard)
        return command

    def start_load(self, **limits):
        self.call('command', commands=self.guarded_frame(**limits))
        self.now += .01
        self.engine.tick()
        self.assertIsNone(self.engine.guard_stop)

    def assert_guard_hold(self, reason):
        self.assertEqual(self.engine.guard_stop['reason'], reason)
        self.assertIsNone(self.engine.error)
        self.assertEqual(self.engine.mode, 'holding')
        self.assertFalse(self.engine.active_guards)
        for side, grip in [('left', .3), ('right', .7)]:
            self.robots[side].set_stop.assert_not_called()
            np.testing.assert_allclose(self.robots[side].Joint_Pos_Vel.call_args.args[0], self.measured[side]['position_rad'])
            self.assertEqual(self.robots[side].gripper_control.call_args.args[0], grip)
        with self.assertRaisesRegex(RuntimeError, 'latched'):
            self.call('command', commands=self.guarded_frame())

    def test_explicit_guarded_mode_recomputes_gravity_and_advertises_capability(self):
        self.start_load()
        np.testing.assert_allclose(self.robots['left'].pos_vel_tqe_kp_kd.call_args.args[2], [.3]*6)
        status = self.call('status')
        self.assertTrue(status['capabilities']['guarded_cartesian_load'])
        self.assertEqual(status['active_guards'], {'left': 'load-1'})
        self.assertIsNone(status['guard_stop'])

    def test_guard_limits_and_complete_gripper_holds_are_validated_atomically(self):
        generous = self.guarded_frame(max_duration_s=300.)
        self.engine.validate_commands(generous)
        self.assertEqual(self.engine.description()['guarded_load_limits']['max_duration_s'],300.)
        bad = []
        for key, value in [('max_down_m', .006), ('max_xy_m', .031), ('max_up_m', .003),
                           ('max_speed_m_s', .051), ('max_angular_rad', .11),
                           ('max_duration_s', 301.), ('max_down_m', float('nan')),
                           ('anchor_tool_rotation', [[0]*3]*3), ('id', '')]:
            command = self.guarded_frame()
            command['left']['joint']['guard'][key] = value
            bad.append(command)
        command = self.guarded_frame()
        command['left']['joint']['assist_torque_nm'][5] = 1.21
        bad.append(command)
        command = self.guarded_frame()
        command['right']['gripper'] = {'mode': 'passive'}
        bad.append(command)
        renewed = self.engine.renewed
        for command in bad:
            with self.subTest(command=command), self.assertRaises((ValueError, TypeError)):
                self.call('command', commands=command)
            self.assertEqual(self.engine.renewed, renewed)
            self.assertFalse(self.engine.active_guards)

    def test_expanded_assist_limits_accept_intermediate_torques_and_reject_excess(self):
        expanded = np.array([3., 6., 6., 1.5, 1.2, 1.2])
        intermediate = np.array([2.25, 4.5, 4.5, 1.125, .9, .9])
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                command = self.guarded_frame()
                command['left']['joint']['assist_torque_nm'] = (sign*intermediate).tolist()
                self.call('command', commands=command)
                self.now += .01
                self.engine.tick()
                self.assertIsNone(self.engine.guard_stop)
                np.testing.assert_allclose(
                    self.robots['left'].pos_vel_tqe_kp_kd.call_args.args[2],
                    .2+sign*intermediate)
        for axis in range(6):
            for sign in (-1., 1.):
                with self.subTest(axis=axis, sign=sign):
                    command = self.guarded_frame()
                    command['left']['joint']['assist_torque_nm'][axis] = sign*(expanded[axis]+.001)
                    with self.assertRaisesRegex(ValueError, 'Guarded assist torque'):
                        self.engine.validate_commands(command)

    def test_contact_approach_limits_are_explicit_and_advertised(self):
        limits = dict(phase='contact_approach', max_down_m=.020,
                      max_xy_m=.005, max_duration_s=30.)
        guard = self.guarded_frame(**limits)['left']['joint']['guard']
        self.assertEqual(self.engine.validate_guard(guard), guard)
        description = self.engine.description()
        self.assertTrue(description['capabilities']['guarded_contact_approach'])
        self.assertEqual(description['guarded_load_limits']['contact_approach']['max_down_m'], .020)
        self.assertEqual(description['guarded_load_limits']['contact_approach']['max_xy_m'], .005)
        self.assertEqual(description['guarded_load_limits']['contact_approach']['max_duration_s'], 30.)
        loaded = description['guarded_load_limits']
        self.assertEqual((loaded['max_xy_m'], loaded['max_down_m'], loaded['max_duration_s']),
                         (.030, .005, 300.))
        for key, value in [('max_down_m', .0201), ('max_xy_m', .0051),
                           ('max_duration_s', 30.1), ('phase', 'unknown'), ('phase', None)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.engine.validate_guard({**guard, key: value})
        with self.assertRaises(ValueError):
            self.engine.validate_guard({k: v for k, v in guard.items() if k != 'phase'})

    def transition_frame(self, **changes):
        pose = self.engine.arms['left']['tool_pose_base']
        return self.guarded_frame(identity='loaded-1', phase='loaded_insertion',
            transition_from='load-1', max_up_m=.005,
            anchor_tool_position_m=pose['position_m'],
            anchor_tool_rotation=pose['rotation'], **changes)

    def start_approach(self):
        self.start_load(phase='contact_approach', max_down_m=.020, max_xy_m=.005)

    def test_continuous_transition_preserves_torque_without_position_hold(self):
        self.start_approach()
        self.measured['left']['tool_pose_base']['position_m'][2] -= .010
        self.now += .01
        self.engine.tick()
        self.robots['left'].reset_mock()
        old_started = self.engine.active_guards['left']['started']
        command = self.transition_frame()
        self.call('command', commands=command)
        self.assertIn('load-1', self.engine.latched_guard_ids)
        self.assertGreater(self.engine.active_guards['left']['started'], old_started)
        self.now += .01
        self.engine.tick()
        self.robots['left'].Joint_Pos_Vel.assert_not_called()
        self.robots['left'].pos_vel_tqe_kp_kd.assert_called_once()
        self.assertIsNone(self.engine.guard_stop)
        self.call('command', commands=command)  # Identical renewal stays valid.
        description = self.engine.description()
        self.assertTrue(description['capabilities']['guarded_continuous_insertion'])
        self.assertEqual(description['guarded_load_limits']['loaded_insertion']['max_up_m'], .005)
        self.assertEqual(description['guarded_load_limits']['max_up_m'], .002)
        self.measured['left']['tool_pose_base']['position_m'][2] += .0049
        self.now += .01
        self.engine.tick()
        self.assertIsNone(self.engine.guard_stop)
        self.measured['left']['tool_pose_base']['position_m'][2] += .0002
        self.now += .01
        self.engine.tick()
        self.assertEqual(self.engine.guard_stop['reason'], 'guard_up_travel_exceeded')

    def test_transition_rejections_leave_guard_commands_and_renewal_unchanged(self):
        self.start_approach()
        initial = deepcopy(self.engine.active_guards)
        commands = deepcopy(self.engine.commands)
        renewed = self.engine.renewed
        for changes in ({'transition_from': 'wrong'}, {'id': 'load-1'},
                        {'anchor_tool_position_m': [.202, .1, .1]},
                        {'anchor_tool_rotation': [[np.cos(.02), -np.sin(.02), 0],
                            [np.sin(.02), np.cos(.02), 0], [0, 0, 1]]}):
            command = self.transition_frame()
            command['left']['joint']['guard'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.call('command', commands=command)
            self.assertEqual(self.engine.active_guards, initial)
            self.assertEqual(self.engine.commands, commands)
            self.assertEqual(self.engine.renewed, renewed)
            self.assertNotIn('load-1', self.engine.latched_guard_ids)
        self.call('command', commands=self.transition_frame())
        command = self.transition_frame()
        command['left']['joint']['guard'].update(id='loaded-2', transition_from='loaded-1')
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            self.call('command', commands=command)
        self.assertEqual(self.engine.active_guards['left']['guard']['id'], 'loaded-1')

    def test_transition_cannot_bypass_old_duration_or_motion_limit(self):
        self.start_approach()
        initial = deepcopy(self.engine.active_guards)
        self.engine.arms['left']['tool_pose_base']['position_m'][2] -= .0201
        with self.assertRaisesRegex(RuntimeError, 'guard_down_travel_exceeded'):
            self.call('command', commands=self.transition_frame())
        self.assertEqual(self.engine.active_guards, initial)
        self.assertNotIn('load-1', self.engine.latched_guard_ids)
        self.engine.active_guards['left']['started'] = self.now - 31.
        with self.assertRaises(RuntimeError):
            self.call('command', commands=self.transition_frame())
        self.assertEqual(self.engine.guard_stop['reason'], 'guard_duration_exceeded')
        self.assertFalse(self.engine.active_guards)
        self.assertIn('load-1', self.engine.latched_guard_ids)
        self.assertNotIn('loaded-1', self.engine.latched_guard_ids)

    def test_two_arm_transition_rejection_is_atomic(self):
        command = self.guarded_frame(phase='contact_approach', max_down_m=.020, max_xy_m=.005)
        command['right']['joint'] = deepcopy(command['left']['joint'])
        command['right']['joint']['guard']['id'] = 'right-approach'
        self.call('command', commands=command)
        self.now += .01
        self.engine.tick()
        active = deepcopy(self.engine.active_guards)
        original_commands = deepcopy(self.engine.commands)
        renewed = self.engine.renewed
        transition = self.transition_frame()
        transition['right']['joint'] = deepcopy(transition['left']['joint'])
        transition['right']['joint']['guard'].update(id='right-loaded', transition_from='wrong')
        with self.assertRaises(RuntimeError):
            self.call('command', commands=transition)
        self.assertEqual(self.engine.active_guards, active)
        self.assertEqual(self.engine.commands, original_commands)
        self.assertEqual(self.engine.renewed, renewed)
        self.assertFalse(self.engine.latched_guard_ids)

    def test_transition_schema_requires_loaded_phase_and_valid_previous_id(self):
        command = self.transition_frame()
        guard = command['left']['joint']['guard']
        for changes in ({'phase': 'contact_approach', 'max_up_m': .002},
                        {'transition_from': ''}, {'transition_from': None},
                        {'transition_from': 1}, {'max_up_m': .0051}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.engine.validate_guard({**guard, **changes})
        with self.assertRaisesRegex(RuntimeError, 'requires an active'):
            self.call('command', commands=command)

    def test_contact_approach_downward_guard_uses_cumulative_measured_travel(self):
        limits = dict(phase='contact_approach', max_down_m=.020,
                      max_xy_m=.005, max_duration_s=30.)
        self.start_load(**limits)
        self.measured['left']['tool_pose_base']['position_m'][2] -= .019
        self.call('command', commands=self.guarded_frame(**limits))
        self.now += .01
        self.engine.tick()
        self.assertIsNone(self.engine.guard_stop)
        self.measured['left']['tool_pose_base']['position_m'][2] -= .002
        self.now += .01
        self.engine.tick()
        self.assert_guard_hold('guard_down_travel_exceeded')

    def test_contact_approach_xy_guard_uses_cumulative_fixed_anchor_travel(self):
        limits = dict(phase='contact_approach', max_down_m=.020,
                      max_xy_m=.005, max_duration_s=30.)
        self.start_load(**limits)
        pose = self.measured['left']['tool_pose_base']
        pose['position_m'][0] += .003
        self.call('command', commands=self.guarded_frame(**limits))
        self.now += .01
        self.engine.tick()
        self.assertIsNone(self.engine.guard_stop)
        # Renewal must retain the original anchor; the second small move
        # crosses the cumulative 5 mm envelope.
        pose['position_m'][0] += .0021
        self.call('command', commands=self.guarded_frame(**limits))
        self.now += .01
        self.engine.tick()
        self.assert_guard_hold('guard_xy_travel_exceeded')

    def test_downward_guard_uses_actual_fk_and_holds_without_power_release(self):
        self.start_load()
        self.measured['left']['tool_pose_base']['position_m'][2] -= .006
        self.measured['left']['position_rad'] = [.04]*6
        self.now += .01
        self.engine.tick()
        self.assert_guard_hold('guard_down_travel_exceeded')

    def test_xy_up_orientation_guards(self):
        for kind in ['xy', 'up', 'orientation']:
            # A new engine per subcase is essential: stopped ids never rearm.
            if kind != 'xy':
                self.tearDown()
                self.doCleanups()
                self.setUp()
            self.start_load()
            pose = self.measured['left']['tool_pose_base']
            if kind == 'xy':
                pose['position_m'][0] += .009
                expected = 'guard_xy_travel_exceeded'
            elif kind == 'up':
                pose['position_m'][2] += .003
                expected = 'guard_up_travel_exceeded'
            else:
                a = .11
                pose['rotation'] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
                expected = 'guard_orientation_exceeded'
            self.now += .01
            self.engine.tick()
            self.assert_guard_hold(expected)

    def test_command_timeout_cannot_be_hidden_by_late_command_renewal(self):
        self.start_load()
        self.now += .101
        with self.assertRaisesRegex(RuntimeError, 'latched'):
            self.call('command', commands=self.guarded_frame())
        self.engine.tick()
        self.assert_guard_hold('guard_command_stale')

    def test_guard_duration_does_not_restart_on_renewed_frames(self):
        self.start_load(max_duration_s=.025)
        self.now += .01
        self.call('command', commands=self.guarded_frame(max_duration_s=.025))
        self.engine.tick()
        self.now += .01
        self.engine.tick()
        self.assert_guard_hold('guard_duration_exceeded')

    def test_guard_parameters_are_immutable_and_new_id_requires_executed_hold(self):
        self.start_load()
        command = self.guarded_frame(max_down_m=.004)
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            self.call('command', commands=command)
        self.call('command', commands=frame())
        with self.assertRaisesRegex(RuntimeError, 'explicit position hold'):
            self.call('command', commands=self.guarded_frame(identity='new-load'))
        self.now += .01
        self.engine.tick()
        self.call('command', commands=self.guarded_frame(identity='new-load'))
        self.now += .01
        self.engine.tick()
        self.assertIsNone(self.engine.guard_stop)
        self.assertEqual(self.engine.active_guards['left']['guard']['id'], 'new-load')

    def test_guard_release_and_lease_loss_preserve_both_squeezes(self):
        self.start_load()
        self.call('release')
        self.now += .01
        self.engine.tick()
        self.assertEqual(self.engine.guard_stop['reason'], 'guard_lease_released')
        self.assertIsNone(self.engine.owner)
        for side, grip in [('left', .3), ('right', .7)]:
            self.assertEqual(self.robots[side].gripper_control.call_args.args[0], grip)
            self.robots[side].gripper_control_MIT.assert_not_called()
        self.now += .6
        self.engine.tick()
        self.assertEqual(self.robots['right'].gripper_control.call_args.args[0], .7)

    def test_total_torque_is_rejected_not_silently_clipped(self):
        self.feedback.stop()
        with patch('motor_backend.feedback', side_effect=lambda r, s, stale:
                   (np.array([4.95, .2, .2, .2, .2, .2]), deepcopy(self.measured[s]))):
            self.call('command', commands=self.guarded_frame())
            self.now += .01
            self.engine.tick()
        self.assert_guard_hold('guard_total_torque_exceeded')
        self.robots['left'].pos_vel_tqe_kp_kd.assert_not_called()

    def test_stale_motor_feedback_is_a_guard_hold_not_power_stop(self):
        self.start_load()
        self.engine.stale[('left', 2)] = (1., time.monotonic() - .11)
        self.now += .01
        self.engine.tick()
        self.assert_guard_hold('guard_motor_feedback_stale')

    def test_encoder_quantum_jitter_does_not_false_trip_speed_but_travel_is_immediate(self):
        self.start_load()
        self.measured['left']['tool_pose_base']['position_m'][0] += .00019
        self.now += 1 / 150
        self.engine.tick()
        self.assertIsNone(self.engine.guard_stop)
        self.measured['left']['tool_pose_base']['position_m'][0] += .008
        self.now += 1 / 150
        self.engine.tick()
        self.assert_guard_hold('guard_xy_travel_exceeded')

    def test_sustained_tool_speed_crossing_does_not_stop_guarded_load(self):
        self.start_load()
        for _ in range(8):
            self.now += .01
            self.measured['left']['tool_pose_base']['position_m'][0] += .0004
            if self.engine.active_guards:
                self.call('command', commands=self.guarded_frame())
            self.engine.tick()
            if self.engine.guard_stop:
                break
        self.assertIsNone(self.engine.guard_stop)
        self.assertTrue(self.engine.active_guards)

    def test_guarded_load_needs_no_velocity_history_but_retains_control_deadline(self):
        self.engine.previous_tools.clear()
        self.engine.tool_history.clear()
        self.start_load()
        self.engine.previous_tools['left'] = (self.now - .101, np.zeros(3))
        self.now += .01
        self.engine.tick()
        self.assert_guard_hold('guard_control_deadline')

    def test_client_refuses_guarded_stage_on_legacy_backend(self):
        client = MotorClient('fake')
        handle = MotorArm(client, 'left', dict(joint_limits={'lower': [-2]*6, 'upper': [2]*6},
            gripper_limits={'lower': 0., 'upper': 2.}))
        with self.assertRaisesRegex(RuntimeError, 'does not support'):
            handle.guarded_gravity([0]*6, self.guarded_frame()['left']['joint']['guard'])
        self.assertEqual(client.frames, {})
        client.capabilities = {'guarded_cartesian_load': True}
        handle.guarded_gravity([.1]*6, self.guarded_frame()['left']['joint']['guard'])
        self.assertEqual(client.frames['left']['joint']['mode'], 'guarded_gravity')

    def test_slow_path_checks_cannot_overlap_guarded_motor_ticks(self):
        self.engine.path_checks.append(('pending',))
        with self.assertRaisesRegex(RuntimeError, 'pending path checks'):
            self.call('command', commands=self.guarded_frame())
        self.assertFalse(self.engine.active_guards)
        self.engine.path_checks.clear()
        self.start_load()
        with self.assertRaisesRegex(RuntimeError, 'unavailable during guarded gravity'):
            self.call('validate_path', side='left', initial=[0]*6, target=[.1]*6)
        self.assertEqual(self.engine.path_checks, [])

    def test_guard_stop_cannot_be_bypassed_by_legacy_gravity_before_hold(self):
        self.start_load()
        self.now += .101
        self.engine.tick()
        legacy = frame()
        legacy['left']['joint'] = dict(mode='gravity', assist_torque_nm=[0]*6)
        with self.assertRaisesRegex(RuntimeError, 'explicit position hold'):
            self.call('command', commands=legacy)
        self.call('command', commands=frame())
        self.now += .01
        self.engine.tick()
        self.call('command', commands=legacy)

    def test_large_search_envelope_is_advertised_and_bounded_independently(self):
        self.start_load(max_xy_m=.030, max_speed_m_s=.050)
        limits = self.call('status')['guarded_load_limits']
        self.assertEqual(limits['max_xy_m'], .030)
        self.assertEqual(limits['max_speed_m_s'], .050)
        self.assertFalse(limits['speed_guards_enabled'])
        self.assertEqual(limits['max_down_m'], .005)
        self.assertEqual(limits['max_up_m'], .002)
        self.assertEqual(limits['command_fresh_seconds'], .1)
        self.assertEqual(limits['assist_torque_nm'], [3., 6., 6., 1.5, 1.2, 1.2])
        for _ in range(110):
            self.now += .01
            self.measured['left']['tool_pose_base']['position_m'][0] += .0001
            self.call('command', commands=self.guarded_frame(max_xy_m=.030, max_speed_m_s=.050))
            self.engine.tick()
            self.assertIsNone(self.engine.guard_stop)
        # More than 10mm of slow travel is supported; the expanded 30mm hard
        # guard is still checked immediately.
        self.measured['left']['tool_pose_base']['position_m'][0] = .2301
        self.now += .01
        self.engine.tick()
        self.assert_guard_hold('guard_xy_travel_exceeded')

    def test_guarded_load_ignores_legacy_speed_ceiling_within_travel_limits(self):
        self.start_load(max_xy_m=.030, max_speed_m_s=.050)
        for _ in range(20):
            self.now += .01
            self.measured['left']['tool_pose_base']['position_m'][0] += .0004  # 40mm/s.
            self.call('command', commands=self.guarded_frame(max_xy_m=.030, max_speed_m_s=.050))
            self.engine.tick()
            self.assertIsNone(self.engine.guard_stop)
        for _ in range(8):
            self.now += .01
            self.measured['left']['tool_pose_base']['position_m'][0] += .0007  # 70mm/s.
            if self.engine.active_guards:
                self.call('command', commands=self.guarded_frame(max_xy_m=.030, max_speed_m_s=.050))
            self.engine.tick()
            if self.engine.guard_stop:
                break
        self.assertIsNone(self.engine.guard_stop)
        self.assertTrue(self.engine.active_guards)


if __name__ == '__main__':
    unittest.main()
