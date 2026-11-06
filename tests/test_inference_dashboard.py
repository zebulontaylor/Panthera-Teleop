"""Policy-step timing, stop behavior and web authorization without robot devices."""
import json
import os
import signal
from pathlib import Path
import sys
import threading
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import inference_dashboard as dashboard


class FakeHardware:
    def __init__(self):
        self.lower = np.full(14, -1.)
        self.upper = np.full(14, 1.)
        self.config = dict(control_hz=150)
        self.cameras = None
        self.commands = []
        self.position = np.zeros(14)
        self.closed = threading.Event()

    def open(self):
        pass

    def read(self):
        return self.position.copy(), {}, {}

    def images(self):
        return {}

    def command(self, target):
        self.position = target.copy()
        self.commands.append(target.copy())

    def close(self):
        self.closed.set()


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for controller state')
        time.sleep(.005)


def fake_request(base, path, body=None, timeout=3):
    if path == '/health':
        return dict(status='ready', contract=dashboard.CONTRACT, checkpoint='saved')
    return dict(sequence=body['sequence'], fps=20, checkpoint='saved', mock=False,
                actions=[[.01] * 14] * 50)


class RecoveryHardware(FakeHardware):
    def __init__(self):
        super().__init__()
        self.position[:] = .2
        self.config['neutral_return'] = json.loads(
            (dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']

    def validate_reset_path(self, *_):
        raise RuntimeError('Recovery must not reset')


class HangingCloseHardware(RecoveryHardware):
    def close(self):
        time.sleep(30)


class SupervisorTests(unittest.TestCase):
    def create(self, hardware=RecoveryHardware):
        supervisor = dashboard.SessionSupervisor('unused', hardware, fake_request)
        self.addCleanup(supervisor.shutdown)
        return supervisor

    def test_recovery_reopens_without_confirmation_reset_or_replay(self):
        supervisor = self.create()
        old_pid = supervisor.process.pid
        supervisor.recover()
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding' and
                 supervisor.snapshot()['hold_status'] == 'commanded', 5)
        state = supervisor.snapshot()
        self.assertNotEqual(supervisor.process.pid, old_pid)
        self.assertIsNone(state['prediction'])
        self.assertEqual(state['steps_completed'], 0)
        np.testing.assert_allclose(state['target'], .2)
        self.assertEqual(state['hold_status'], 'commanded')

    def test_native_crash_keeps_supervisor_usable_and_can_recover(self):
        supervisor = self.create()
        supervisor.process.kill()
        supervisor.process.join(2)
        self.assertEqual(supervisor.snapshot()['mode'], 'fault')
        self.assertEqual(supervisor.snapshot()['hold_status'], 'unverified')
        supervisor.recover()
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 5)

    def test_foreground_interrupt_does_not_kill_robot_session(self):
        supervisor = self.create()
        supervisor.snapshot()  # Wait until the spawned child has installed handlers.
        os.kill(supervisor.process.pid, signal.SIGINT)
        os.kill(supervisor.process.pid, signal.SIGTERM)
        self.assertEqual(supervisor.snapshot()['mode'], 'stopped')
        self.assertTrue(supervisor.process.is_alive())

    def test_unresponsive_session_does_not_block_recovery_or_reuse_late_replies(self):
        supervisor = self.create()
        supervisor.snapshot()
        old_process = supervisor.process
        os.kill(old_process.pid, signal.SIGSTOP)
        self.assertEqual(supervisor.snapshot()['mode'], 'fault')
        self.assertEqual(supervisor.snapshot()['mode'], 'fault')
        supervisor.recover()
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 7)
        self.assertFalse(old_process.is_alive())

    def test_stuck_cleanup_is_replaced_only_after_old_process_exits(self):
        supervisor = self.create(HangingCloseHardware)
        supervisor.recover()
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 5)
        old_process = supervisor.process
        supervisor.recover()
        with self.assertRaises(RuntimeError):
            supervisor.recover()
        with self.assertRaises(RuntimeError):
            supervisor.connect()
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 7)
        self.assertFalse(old_process.is_alive())
        self.assertNotEqual(supervisor.process.pid, old_process.pid)

    def test_stop_cancels_recovery_without_reopening_hardware(self):
        supervisor = self.create(HangingCloseHardware)
        supervisor.recover()
        wait_for(lambda: supervisor.snapshot()['mode'] == 'holding', 5)
        supervisor.recover()
        started = time.monotonic()
        supervisor.stop()
        self.assertLess(time.monotonic() - started, .2)
        wait_for(lambda: not supervisor.snapshot()['recovering'], 7)
        self.assertEqual(supervisor.snapshot()['mode'], 'stopped')
        self.assertFalse(supervisor.snapshot()['connected'])


class DashboardShutdownTests(unittest.TestCase):
    def test_connected_dashboard_signal_exits_and_cleans_up(self):
        controller = Mock()
        controller.snapshot.return_value = {'connected': True}
        server = Mock()
        finished = threading.Event()
        server.shutdown.side_effect = finished.set
        handlers = {}

        def serve_forever(**_):
            handlers[signal.SIGTERM]()
            handlers[signal.SIGINT]()  # Repeated signals do not start more cleanup.
            self.assertTrue(finished.wait(2))

        server.serve_forever.side_effect = serve_forever
        with patch.object(sys, 'argv', ['inference_dashboard.py']), \
             patch.object(dashboard, 'SessionSupervisor', return_value=controller), \
             patch.object(dashboard, 'make_server', return_value=server), \
             patch.object(dashboard.signal, 'signal', side_effect=lambda sig, fn: handlers.update({sig: fn})):
            dashboard.main()
        server.shutdown.assert_called_once_with()
        server.server_close.assert_called_once_with()
        self.assertGreaterEqual(controller.shutdown.call_count, 1)
        controller.update.assert_not_called()


class StepTests(unittest.TestCase):
    def test_blend_crossfades_aligned_tail_and_preserves_bounds(self):
        old = np.full((40, 14), .2)
        new = np.full((50, 14), -.2)
        blended, overlap = dashboard.blend_predictions(new, old, 10)
        self.assertEqual(overlap, 10)
        self.assertTrue(np.all(np.diff(blended[:10, 0]) < 0))
        self.assertTrue(np.all(blended >= -.2))
        self.assertTrue(np.all(blended <= .2))
        np.testing.assert_array_equal(blended[9:], new[9:])
        np.testing.assert_array_equal(new, -.2)
        np.testing.assert_array_equal(old, .2)
        for tail, count in ((None, 10), (old, 0), (old[:0], 10)):
            result, overlap = dashboard.blend_predictions(new, tail, count)
            self.assertEqual(overlap, 0)
            np.testing.assert_array_equal(result, new)
        _, overlap = dashboard.blend_predictions(new, old[:3], 50)
        self.assertEqual(overlap, 3)
        for value in (-1, 51, True, 1.5, None):
            with self.assertRaises(ValueError):
                dashboard.blend_count(value)

    def test_count_rejects_fractional_boolean_and_out_of_range(self):
        for value in (0, 51, True, 1.5, '5', None):
            with self.assertRaises(ValueError):
                dashboard.action_count(value)

    def test_single_step_cannot_advance_into_the_next_action(self):
        actions = np.array([[.1] * 14, [-.1] * 14])
        step = dashboard.ActionStep(actions, 1, 10.)
        target = np.zeros(14)
        for tick in range(150):
            now = 10. + tick*.01
            previous = target.copy()
            target, done = step.tick(now, .01, target)
            self.assertTrue(np.all(target >= previous))
            self.assertTrue(np.all(target-previous <= step.speed*.01 + 1e-9))
            if done:
                break
        self.assertTrue(done)
        np.testing.assert_allclose(target, .1)
        final, done = step.tick(now + .01, .01, target)
        self.assertTrue(done)
        np.testing.assert_array_equal(final, target)

    def test_large_jump_is_smoothed_and_takes_longer_than_policy_time(self):
        step = dashboard.ActionStep(np.ones((50, 14)), 1, 0)
        target = np.zeros(14)
        for tick in range(6):
            previous = target.copy()
            target, done = step.tick(tick*.01, .01, target)
            self.assertFalse(done)
            self.assertTrue(np.all(np.abs(target-previous) <= step.speed*.01 + 1e-9))
        self.assertEqual(step.index, 0)


class CollectionResetTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']
        self.settings['arms'] = ['left', 'right']
        self.lower = np.full(14, -3.)
        self.upper = np.full(14, 3.)
        self.lower[[6, 13]] = 0
        self.upper[[6, 13]] = 2

    def test_reset_reaches_collection_pose_and_opening_with_speed_bounds(self):
        measured = np.zeros(14)
        measured[[6, 13]] = [1.94, .5]
        target = measured.copy()
        reset = dashboard.CollectionReset(self.settings, self.lower, self.upper, measured, 0)
        for tick in range(1, 600):
            previous = target.copy()
            target, done = reset.tick(tick*.01, .01, measured, target)
            speed = np.full(14, dashboard.CONTRACT['max_joint_speed_rad_s'])
            speed[[6, 13]] = dashboard.CONTRACT['gripper_speed_rad_s']
            self.assertTrue(np.all(np.abs(target-previous) <= speed*.01 + 1e-9))
            measured = target.copy()
            if done:
                break
        self.assertTrue(done)
        np.testing.assert_allclose(target, [0,.25,.25,0,0,0,1.2]*2)

    def test_gripper_obstruction_refuses_completion(self):
        measured = np.array([0,.25,.25,0,0,0,0.]*2)
        reset = dashboard.CollectionReset(self.settings, self.lower, self.upper, measured, 0)
        target = measured.copy()
        with self.assertRaisesRegex(RuntimeError, 'grippers'):
            for tick in range(1, 600):
                target, done = reset.tick(tick*.01, .01, measured, target)
                self.assertFalse(done)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.hardware = FakeHardware()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.controller = dashboard.Controller('http://unused', lambda: self.hardware,
                                               Path(directory.name) / 'steps.jsonl')
        self.addCleanup(self.stop)

    def stop(self):
        self.controller.shutdown()
        if self.controller.worker:
            self.controller.worker.join(2)

    def request(self, base, path, body=None, timeout=3):
        if path == '/health':
            return dict(status='ready', contract=dashboard.CONTRACT, checkpoint='saved')
        return dict(sequence=body['sequence'], fps=20, checkpoint='saved', mock=False,
                    actions=[[.01] * 14] * 50)

    def connect(self):
        self.controller.connect()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')

    def test_task_switch_requires_idle_hold_and_clears_prediction(self):
        controller = self.controller
        default = dashboard.CONTRACT['task']
        controller.update(mode='holding', supported_tasks=[default, 'four gears'],
                          prediction=[[0.] * 14], blended_actions=5)
        for values in (dict(mode='predicting'), dict(mode='executing'), dict(playing=True),
                       dict(rollout_active=True), dict(collection_pending=True)):
            controller.update(**values)
            with self.assertRaises(RuntimeError):
                controller.select_task('four gears')
            controller.update(mode='holding', playing=False, rollout_active=False, collection_pending=False)
        with self.assertRaises(ValueError):
            controller.select_task('unknown')
        controller.select_task('four gears')
        self.assertEqual(controller.snapshot()['task'], 'four gears')
        self.assertIsNone(controller.snapshot()['prediction'])
        self.assertEqual(controller.snapshot()['blended_actions'], 0)
        self.assertEqual(controller.task_generation, 1)

    def test_task_switch_without_reconnect_and_response_echo_validation(self):
        default = dashboard.CONTRACT['task']
        bodies = []
        def request(base, path, body=None, timeout=3):
            if path == '/health':
                return dict(status='ready', contract=dashboard.CONTRACT, checkpoint='saved',
                            supported_tasks=[default, 'four gears'])
            bodies.append(body.copy())
            result = self.request(base, path, body, timeout)
            result['task'] = default if len(bodies) > 1 else body['task']
            return result
        with patch.object(dashboard, 'request', request):
            self.connect()
            self.controller.submit('predict')
            wait_for(lambda: self.controller.snapshot()['prediction'] is not None)
            self.controller.select_task('four gears')
            self.controller.submit('step', 1)
            wait_for(lambda: 'task differs' in (self.controller.snapshot()['error'] or ''))
        self.assertEqual([body['task'] for body in bodies], [default, 'four gears'])
        self.assertIn('task differs', self.controller.snapshot()['error'])
        self.assertEqual(self.controller.snapshot()['steps_completed'], 0)

    def test_default_actions_per_step_and_expo_override(self):
        self.assertEqual(self.controller.snapshot()['actions_per_step'], 10)
        self.assertEqual(self.controller.snapshot()['step_count'], 10)
        expo = dashboard.Controller('unused', expo_record_dir='unused', expo_prefix_length=5)
        self.assertEqual(expo.snapshot()['actions_per_step'], 5)
        self.assertEqual(expo.snapshot()['blend_actions'], 0)

    def test_play_pause_resume_preserves_configured_settings(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            self.controller.configure(7, 4)
            for _ in range(2):
                completed = self.controller.snapshot()['steps_completed']
                self.controller.heartbeat()
                self.controller.submit('play')
                wait_for(lambda: self.controller.snapshot()['steps_completed'] > completed)
                self.assertEqual(self.controller.snapshot()['step_count'], 7)
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding' and
                         not self.controller.pause_event.is_set())
                self.assertEqual(self.controller.snapshot()['actions_per_step'], 7)
                self.assertEqual(self.controller.snapshot()['blend_actions'], 4)
            completed = self.controller.snapshot()['steps_completed']
            self.controller.submit('step', blend=0)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] > completed)
            self.assertEqual(self.controller.snapshot()['step_count'], 7)
            self.assertEqual(self.controller.snapshot()['blend_actions'], 0)

    def test_usb_pedal_double_tap_plays_and_disconnect_holds(self):
        pedal = Mock()
        pedal.devices = {42: set()}
        events = []
        initialized = threading.Event()
        def sample():
            batch = events[:]
            events.clear()
            initialized.set()
            return dict(connected=True, downward_force_requested=False,
                        sampled_monotonic_ns=time.monotonic_ns(), events=batch)
        pedal.snapshot.side_effect = sample
        with patch.object(dashboard, 'request', self.request), \
             patch.object(dashboard, 'FootPedal', return_value=pedal):
            self.connect()
            initialized.wait(1)
            self.controller.update(rollout_active=True)
            now = time.monotonic_ns()
            events.extend(dict(monotonic_ns=now+offset, pressed=pressed)
                          for offset, pressed in [(-200_000_000, True), (-150_000_000, False),
                                                  (-100_000_000, True), (-50_000_000, False)])
            wait_for(lambda: self.controller.snapshot()['playing'])
            self.assertTrue(self.controller.snapshot()['foot_pedal_connected'])
            self.assertFalse(self.controller.snapshot()['collection_pending'])
            pedal.snapshot.side_effect = RuntimeError('pedal unplugged')
            wait_for(lambda: not self.controller.snapshot()['foot_pedal_connected'])
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.assertFalse(self.controller.snapshot()['playing'])
            self.assertIn('unplugged', self.controller.snapshot()['foot_pedal_error'])
            pedal.close.assert_called_once()

    def test_play_repeats_fresh_steps_blends_and_applies_next_settings(self):
        observations = []
        def changing(base, path, body=None, timeout=3):
            result = self.request(base, path, body, timeout)
            if path == '/infer':
                observations.append(body['state'])
                result['actions'] = [[.001*(i+1)+body['state'][0]]*14 for i in range(50)]
            return result
        with patch.object(dashboard, 'request', changing):
            self.connect()
            self.controller.submit('play', 2, 10)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] >= 2)
            self.assertTrue(self.controller.snapshot()['playing'])
            self.controller.configure(3, 4)
            wait_for(lambda: self.controller.snapshot()['step_count'] == 3 and
                     self.controller.snapshot()['blended_actions'] == 4)
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.assertFalse(self.controller.snapshot()['playing'])
            held = self.hardware.position.copy()
            time.sleep(.08)
            np.testing.assert_array_equal(self.hardware.position, held)
            self.assertGreater(observations[1][0], observations[0][0])
            entries = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()]
            blends = [row for row in entries if row.get('event') == 'play_blend']
            self.assertEqual(blends[0]['blended_actions'], 0)
            expected, _ = dashboard.blend_predictions(
                np.array([row for row in entries if 'response' in row][1]['response']['actions']),
                np.array(blends[0]['actions'])[2:], 10)
            np.testing.assert_allclose(blends[1]['actions'], expected)

    def test_play_pause_discards_late_response_and_resume_starts_fresh(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def delayed(base, path, body=None, timeout=3):
            result = self.request(base, path, body, timeout)
            if path == '/infer':
                calls.append(body['sequence'])
                if len(calls) == 1:
                    entered.set()
                    release.wait(2)
                    result['actions'] = [[.9]*14]*50
            return result
        with patch.object(dashboard, 'request', delayed):
            try:
                self.connect()
                self.controller.submit('play', 2)
                self.assertTrue(entered.wait(1))
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
                release.set()
                time.sleep(.05)
                np.testing.assert_array_equal(self.hardware.position, 0)
                self.assertFalse(self.controller.snapshot()['playing'])
                self.controller.submit('play', 1)
                wait_for(lambda: self.controller.snapshot()['steps_completed'] >= 1)
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
                np.testing.assert_allclose(self.hardware.position, .01)
                blends = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()
                          if 'play_blend' in line]
                self.assertEqual(blends[0]['blended_actions'], 0)
            finally:
                release.set()

    def test_play_error_disables_repetition_until_explicit_resume(self):
        calls = []
        def invalid(base, path, body=None, timeout=3):
            result = self.request(base, path, body, timeout)
            if path == '/infer':
                calls.append(body)
                result['actions'][-1][0] = 1.1
            return result
        with patch.object(dashboard, 'request', invalid):
            self.connect()
            self.controller.submit('play', 1)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertFalse(self.controller.snapshot()['playing'])
            self.controller.heartbeat()
            time.sleep(.08)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(self.hardware.position, 0)

    def test_expired_overlap_is_discarded_after_retimed_motion(self):
        with patch.object(dashboard, 'request', self.request), patch.dict(
                dashboard.CONTRACT, max_observation_age_s=.06):
            self.connect()
            self.controller.submit('play', 2, 10)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] >= 2)
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            blends = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()
                      if 'play_blend' in line]
            self.assertEqual(blends[1]['blended_actions'], 0)

    def test_browser_loss_cancels_play_and_never_auto_resumes(self):
        with patch.object(dashboard, 'request', self.request), patch.object(dashboard, 'LEASE_SECONDS', .1):
            self.connect()
            self.controller.submit('play', 1)
            wait_for(lambda: not self.controller.snapshot()['playing'] and
                     self.controller.snapshot()['error'] is not None)
            held = self.hardware.position.copy()
            self.controller.heartbeat()
            time.sleep(.04)
            np.testing.assert_array_equal(self.hardware.position, held)
            self.assertFalse(self.hardware.closed.is_set())

    def test_camera_fault_clears_queued_play_before_camera_recovery(self):
        camera = {'ok': False, 'age_s': 0.}
        self.hardware.read = lambda: (self.hardware.position.copy(), {}, {'test': camera.copy()})
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            self.controller.submit('play', 1)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertFalse(self.controller.snapshot()['playing'])
            self.assertTrue(self.controller.commands.empty())
            camera['ok'] = True
            time.sleep(.08)
            self.assertEqual(self.controller.snapshot()['steps_completed'], 0)
            np.testing.assert_array_equal(self.hardware.position, 0)

    def test_preview_holds_and_one_action_step_pauses(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            self.controller.submit('predict')
            wait_for(lambda: self.controller.snapshot()['prediction'] is not None and
                     self.controller.snapshot()['mode'] == 'holding')
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            state = self.controller.snapshot()
            self.assertEqual(state['mode'], 'holding')
            self.assertEqual(state['action_index'], 1)
            self.assertGreater(float(np.max(self.hardware.position)), 0)
            held = self.hardware.position.copy()
            time.sleep(.08)
            np.testing.assert_array_equal(self.hardware.position, held)

    def test_gripper_overshoots_execute_at_configured_limits(self):
        def overshoot(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                action = [.01] * 14
                action[6], action[13] = -5., 8.
                result['actions'] = [action] * 50
            return result
        with patch.object(dashboard, 'request', overshoot):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            self.assertIsNone(self.controller.snapshot()['error'])
            np.testing.assert_allclose(self.hardware.position[[6, 13]], [-1., 1.])
            self.assertTrue(all(np.all(q >= self.hardware.lower) and
                                np.all(q <= self.hardware.upper) for q in self.hardware.commands))

    def test_bad_prediction_stops_before_any_nonzero_motion(self):
        def bad(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                result['actions'] = [[1.1] * 14] * 50
            return result
        with patch.object(dashboard, 'request', bad):
            self.connect()
            self.controller.submit('step', 5)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertEqual(self.controller.snapshot()['mode'], 'holding')
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))

    def test_one_action_executes_when_only_unselected_tail_is_outside_limits(self):
        def tail_invalid(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                result['actions'][-1] = [1.1] * 14
            return result
        with patch.object(dashboard, 'request', tail_invalid):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            # Completion is published within the tick before its final command is sent.
            wait_for(lambda: np.allclose(self.hardware.position, .01, atol=1e-9, rtol=0))
            self.assertIsNone(self.controller.snapshot()['error'])
            np.testing.assert_allclose(self.hardware.position, .01)
            self.controller.submit('step', 50)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertIn('action 50, left.joint1', self.controller.snapshot()['error'])
            np.testing.assert_allclose(self.hardware.position, .01)
            entries = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()]
            self.assertEqual(sum('response' in row for row in entries), 2)

    def test_stop_interrupts_a_stretched_large_motion_and_retains_hold(self):
        def large(*args, **kwargs):
            result = self.request(*args, **kwargs)
            if 'actions' in result:
                result['actions'] = [[.9]*14] * 50
            return result
        with patch.object(dashboard, 'request', large):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.hardware.position.max() > .01)
            self.assertEqual(self.controller.snapshot()['mode'], 'executing')
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
            held = self.hardware.position.copy()
            time.sleep(.08)
            np.testing.assert_array_equal(self.hardware.position, held)
            self.assertFalse(self.hardware.closed.is_set())
            self.assertLess(held.max(), .9)

    def test_browser_lease_expiry_keeps_hardware_connected_and_held(self):
        with patch.object(dashboard, 'request', self.request), patch.object(dashboard, 'LEASE_SECONDS', .05):
            self.connect()
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertIn('Browser connection lost', self.controller.snapshot()['error'])
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))

    def test_connect_resets_before_accepting_steps_and_stop_interrupts_reset(self):
        self.hardware.config['neutral_return'] = json.loads(
            (dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']
        self.hardware.validate_reset_path = lambda *_: None
        with patch.object(dashboard, 'request', self.request):
            self.controller.connect()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'resetting')
            with self.assertRaisesRegex(RuntimeError, 'wait'):
                self.controller.submit('step')
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
            self.assertFalse(self.hardware.closed.is_set())

    def test_stop_does_not_wait_for_network_inference(self):
        entered, release = threading.Event(), threading.Event()
        def delayed(*args, **kwargs):
            if args[1] == '/infer':
                entered.set()
                release.wait(2)
            return self.request(*args, **kwargs)
        with patch.object(dashboard, 'request', delayed):
            try:
                self.connect()
                self.controller.submit('step', 50)
                self.assertTrue(entered.wait(1))
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding', .3)
                self.assertTrue(self.controller.snapshot()['connected'])
                self.assertFalse(self.hardware.closed.is_set())
                self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))
            finally:
                release.set()

    def test_offline_policy_keeps_local_hardware_and_reset_available(self):
        with patch.object(dashboard, 'request', side_effect=OSError('AWS offline')), \
             patch.object(self.controller, 'collection_reset', return_value=None) as reset:
            self.connect()
            state = self.controller.snapshot()
            self.assertTrue(state['connected'])
            self.assertFalse(state['policy_ready'])
            self.assertIn('AWS offline', state['policy_error'])
            self.assertIsNone(state['checkpoint'])
            self.assertFalse(self.hardware.closed.is_set())
            for kind in ('predict', 'step', 'play'):
                with self.assertRaisesRegex(RuntimeError, 'Policy unavailable'):
                    self.controller.submit(kind)
            self.assertTrue(self.controller.commands.empty())
            # Reset is accepted independently of policy availability.
            self.controller.submit('reset')
            wait_for(lambda: reset.call_count == 2)
            self.assertTrue(self.controller.snapshot()['connected'])

    def test_offline_policy_does_not_create_expo_recording_or_auto_play(self):
        self.controller.expo_record_dir = Path('unused-offline-expo')
        with patch.object(dashboard, 'request', side_effect=OSError('AWS offline')), \
             patch.object(dashboard, 'ExplorationRolloutStore') as store:
            self.connect()
            store.assert_not_called()
            self.assertIsNone(self.controller.expo_recorder)
            for payload in (dict(operator='test', scene='bench'),
                            dict(operator='test', scene='bench', pedal_play=dict(count=15, blend=0))):
                with self.assertRaisesRegex(RuntimeError, 'Policy unavailable'):
                    self.controller.collection('begin', payload)
            self.assertTrue(self.controller.collection_commands.empty())

    def test_unready_or_invalid_server_does_not_enable_policy(self):
        for health in (dict(status='loading', error='checkpoint loading'),
                       dict(status='ready', contract={}, checkpoint='bad')):
            with self.subTest(health=health), patch.object(dashboard, 'request', return_value=health):
                self.connect()
                state = self.controller.snapshot()
                self.assertFalse(state['policy_ready'])
                self.assertTrue(state['policy_error'])
                self.assertEqual(state['mode'], 'holding')
                self.stop()

    def test_ready_server_enables_policy(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            state = self.controller.snapshot()
            self.assertTrue(state['policy_ready'])
            self.assertIsNone(state['policy_error'])
            self.assertEqual(state['checkpoint'], 'saved')

    def test_inference_network_error_retains_hold(self):
        def broken(*args, **kwargs):
            if args[1] == '/infer':
                raise OSError('network offline')
            return self.request(*args, **kwargs)
        with patch.object(dashboard, 'request', broken):
            self.connect()
            self.controller.submit('step', 1)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertEqual(self.controller.snapshot()['mode'], 'holding')
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            self.assertTrue(all(np.all(q == 0) for q in self.hardware.commands))

    def test_motor_feedback_failure_retains_sdk_and_blocks_motion(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            def broken():
                raise RuntimeError('stale motor feedback')
            self.hardware.read = broken
            wait_for(lambda: self.controller.snapshot()['mode'] == 'fault_hold')
            self.assertTrue(self.controller.snapshot()['connected'])
            self.assertFalse(self.hardware.closed.is_set())
            with self.assertRaises(RuntimeError):
                self.controller.submit('step')

    def test_failed_hold_is_reported_as_failed_instead_of_retained_hold(self):
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            def broken(*_):
                raise OSError('Bad file descriptor')
            self.hardware.command = broken
            wait_for(lambda: self.controller.snapshot()['hold_status'] == 'failed')
            self.assertIn('Powered hold is not reliable', self.controller.snapshot()['error'])
            self.assertEqual(self.controller.snapshot()['mode'], 'fault_hold')



class CollectionResetTimingTests(unittest.TestCase):
    def test_first_tick_before_reset_creation_retains_hold(self):
        settings = json.loads((dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']
        lower, upper = np.full(14, -4.), np.full(14, 4.)
        measured = np.zeros(14)
        reset = dashboard.CollectionReset(settings, lower, upper, measured, 10.)
        target, done = reset.tick(9.999, .006, measured, measured)
        np.testing.assert_array_equal(target, measured)
        self.assertFalse(done)
        target, done = reset.tick(10.001, .006, measured, measured)
        self.assertTrue(np.isfinite(target).all())


class WebTests(unittest.TestCase):
    def test_mutations_require_token_and_trusted_origin(self):
        controller = dashboard.Controller('unused')
        server = dashboard.make_server(controller, 0)
        port = server.server_port
        # The handler validates the configured port, so use the actual bound port.
        server.server_close()
        server = dashboard.make_server(controller, port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{port}'
        with urlopen(base + '/api/status') as response:
            token = json.load(response)['token']
        for headers in ({}, {'X-Session-Token': token, 'Origin': 'https://untrusted.example'}):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/api/stop', data=b'{}', headers=headers))
            self.assertEqual(error.exception.code, 403)
        self.assertFalse(controller.stop_event.is_set())
        with urlopen(Request(base + '/api/stop', data=b'{}', headers={'X-Session-Token': token})) as response:
            self.assertEqual(response.status, 202)
            self.assertTrue(controller.pause_event.is_set())
            self.assertFalse(controller.stop_event.is_set())
        headers = {'X-Session-Token': token, 'Content-Type': 'application/json'}
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base + '/api/alignment/force/spiral', data=b'{}', headers=headers))
        self.assertEqual(error.exception.code, 404)
        with patch.object(controller, 'recover', create=True) as recover:
            with urlopen(Request(base + '/api/recover', data=b'{}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            recover.assert_called_once_with()
        with patch.object(controller, 'submit') as submit:
            with urlopen(Request(base + '/api/play', data=b'{"count":5,"blend":8}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            submit.assert_called_once_with('play', 5, 8)
        with patch.object(controller, 'submit') as submit:
            with urlopen(Request(base + '/api/play', data=b'{}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            submit.assert_called_once_with('play', None, None)
        with patch.object(controller, 'configure') as configure:
            with urlopen(Request(base + '/api/configure', data=b'{"count":3,"blend":0}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            configure.assert_called_once_with(3, 0)
        controller.update(mode='holding', supported_tasks=[dashboard.CONTRACT['task'], 'four gears'])
        with urlopen(Request(base + '/api/task', data=b'{"task":"four gears"}', headers=headers)) as response:
            self.assertEqual(response.status, 202)
        self.assertEqual(controller.snapshot()['task'], 'four gears')
        controller.update(rollout_active=True)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base + '/api/task', data=json.dumps({'task': dashboard.CONTRACT['task']}).encode(), headers=headers))
        self.assertEqual(error.exception.code, 409)
        with patch.object(controller, 'stop') as stop:
            with urlopen(Request(base + '/api/pause', data=b'{}', headers=headers)) as response:
                self.assertEqual(response.status, 202)
            stop.assert_called_once_with()
        for payload in (b'{"count":true}', b'{"count":51}', b'{"blend":-1}'):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/api/play', data=payload, headers=headers))
            self.assertEqual(error.exception.code, 409)




class AlignmentAndRightFreezeTests(unittest.TestCase):
    def test_exact_alignment_frame_and_native_capture(self):
        controller = dashboard.Controller('unused', alignment_options={})
        monitor = Mock()
        monitor.analyzed_frame.return_value = ('7', b'jpeg')
        controller.alignment = monitor
        self.assertIsNone(controller.alignment_frame('6'))
        self.assertEqual(controller.alignment_frame('7'), ('7', b'jpeg'))
        hardware = Mock()
        hardware.config = {'alignment_assist': {'camera_mapping': {'left_wrist': 'wrist_port3', 'right_wrist': 'wrist_port2'}}}
        hardware.cameras.recording_frames.return_value = {
            'wrist_port2': {'jpeg': b'wrong-right', 'monotonic_ns': 99},
            'wrist_port3': {'jpeg': b'native', 'monotonic_ns': 100}}
        controller.capture_alignment(hardware, np.arange(14.), 110)
        kwargs = monitor.submit.call_args.kwargs
        self.assertEqual(kwargs['jpeg'], b'native')
        self.assertEqual(kwargs['state'], list(np.arange(14.)))
        self.assertEqual(kwargs['measured_timestamp_ns'], 110)
        self.assertEqual(kwargs['camera_id'], 'left_wrist')
        hardware.config['alignment_assist']['camera_mapping']['left_wrist'] = 'custom_left'
        hardware.cameras.recording_frames.return_value = {'custom_left': {'jpeg': b'mapped', 'monotonic_ns': 101}}
        controller.capture_alignment(hardware, np.arange(14.), 111)
        self.assertEqual(monitor.submit.call_args.kwargs['jpeg'], b'mapped')

    def test_freeze_rejects_unsafe_modes_and_projection_copies(self):
        controller = dashboard.Controller('unused')
        for mode in ('stopped', 'fault_hold', 'resetting'):
            controller.state['mode'] = mode
            with self.assertRaises(RuntimeError):
                controller.request_right_arm_freeze(True)
        controller.state['mode'] = 'holding'
        controller.request_right_arm_freeze(True)
        self.assertTrue(controller.snapshot()['right_arm_freeze_pending'])
        controller._right_freeze_target = np.arange(7.)
        actions = np.zeros((3, 14))
        projected = controller.project_right_hold(actions)
        np.testing.assert_array_equal(projected[:, :7], 0)
        np.testing.assert_array_equal(projected[:, 7:], np.tile(np.arange(7.), (3, 1)))
        np.testing.assert_array_equal(actions, 0)
        controller.clear_right_freeze('manual takeover')
        self.assertFalse(controller.snapshot()['right_arm_frozen'])
        self.assertIsNone(controller._freeze_requested)

    def test_control_thread_freeze_preserves_commanded_gripper_and_left_motion(self):
        class ContactHardware(FakeHardware):
            def read(self):
                measured = self.position.copy()
                measured[13] = .15  # Object prevents commanded closed aperture.
                return measured, {}, {}
        hardware = ContactHardware()
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, 'request', side_effect=fake_request):
            controller = dashboard.Controller('unused', lambda: hardware,
                                              log_path=Path(directory)/'steps.jsonl', data_dir=directory)
            controller.connect(reset_pose=False)
            try:
                wait_for(lambda: controller.snapshot()['mode'] == 'holding')
                controller.heartbeat()
                before = controller.snapshot()['target'][13]
                controller.request_right_arm_freeze(True)
                wait_for(lambda: controller.snapshot()['right_arm_frozen'])
                self.assertEqual(controller.snapshot()['right_arm_freeze_target'][6], before)
                controller.submit('step', 2, 0)
                wait_for(lambda: controller.snapshot()['steps_completed'] >= 1)
                self.assertGreater(controller.snapshot()['target'][0], 0)
                self.assertEqual(controller.snapshot()['target'][13], before)
                controller.stop()
                self.assertTrue(controller.snapshot()['right_arm_frozen'])
                controller.request_right_arm_freeze(False)
                wait_for(lambda: not controller.snapshot()['right_arm_frozen'])
            finally:
                controller.shutdown()
                controller.worker.join(2)

class ManualHoleCaptureTests(unittest.TestCase):
    def test_explicit_point_delegates_only_in_paused_frozen_hold(self):
        controller=dashboard.Controller('unused',alignment_options={})
        controller.state.update(mode='holding',right_arm_frozen=True)
        controller.heartbeat_at=time.monotonic()
        monitor=Mock();monitor.status.return_value=dict(status='waiting_for_hole',frame_id=17,image_size_wh=[640,480])
        controller.alignment=monitor
        self.assertTrue(controller.snapshot()['alignment_manual_capture_supported'])
        controller.capture_alignment_hole(17,[280.,310.])
        monitor.capture_hole.assert_called_once_with(17,hole_px=[280.,310.])
        for point in ([640,200],[-1,200],[200,480],[float('nan'),2],[1,2,3],[True,2],['3',2]):
            with self.assertRaises(ValueError):controller.capture_alignment_hole(17,point)
        monitor.capture_hole.reset_mock()
        controller.state['right_arm_frozen']=False
        with self.assertRaises(RuntimeError):controller.capture_alignment_hole(17,[280,310])
        monitor.capture_hole.assert_not_called()
        controller.state.update(right_arm_frozen=True,mode='force_holding')
        with self.assertRaises(RuntimeError):controller.capture_alignment_hole(17,[280,310])

    def test_http_string_query_serves_exact_retained_integer_frame_id(self):
        from pin_alignment_monitor import PinAlignmentMonitor
        monitor=PinAlignmentMonitor(config={'enabled':False})
        controller=dashboard.Controller('unused',alignment_options={})
        controller.alignment=monitor
        import io
        from PIL import Image
        buffers=[]
        for color in ('red','blue'):
            stream=io.BytesIO();Image.new('RGB',(8,8),color).save(stream,format='JPEG');buffers.append(stream.getvalue())
        old_jpeg,latest_jpeg=buffers
        stamp=time.monotonic_ns()
        with monitor._condition:
            monitor._snapshots={7:({'frame_id':7,'analyzed_timestamp_ns':stamp},old_jpeg),
                                8:({'frame_id':8,'analyzed_timestamp_ns':stamp},latest_jpeg)}
            monitor._jpeg_frame_id=8;monitor._jpeg=latest_jpeg
        server=dashboard.make_server(controller,0);port=server.server_port;server.server_close()
        server=dashboard.make_server(controller,port)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            base=f'http://127.0.0.1:{port}'
            with urlopen(base+'/api/alignment/image?frame_id=7') as response:
                self.assertEqual(response.status,200)
                self.assertEqual(response.headers['X-Alignment-Frame-Id'],'7')
                self.assertEqual(response.read(),old_jpeg)
            for query in ('','?frame_id=True','?frame_id=0','?frame_id=9'):
                with self.assertRaises(HTTPError) as error:urlopen(base+'/api/alignment/image'+query)
                self.assertEqual(error.exception.code,409)
            with monitor._condition:
                monitor._snapshots[7][0]['analyzed_timestamp_ns']=stamp-5_100_000_000
            with self.assertRaises(HTTPError) as error:urlopen(base+'/api/alignment/image?frame_id=7')
            self.assertEqual(error.exception.code,409)
        finally:
            server.shutdown();server.server_close();thread.join(2);monitor.close()

    def test_recent_exact_snapshot_is_used_without_substituting_newer_predictions(self):
        controller=dashboard.Controller('unused',alignment_options={})
        monitor=Mock()
        old=dict(frame_id=11,hole_px=[100,200],image_size_wh=[640,480])
        monitor.status_for_frame.return_value=old
        monitor.status.return_value=dict(frame_id=12,hole_px=[300,400])
        monitor.analyzed_frame.return_value=(11,b'older-exact-jpeg')
        controller.alignment=monitor
        self.assertEqual(controller.alignment_preview(11)['hole_px'],[100,200])
        self.assertEqual(controller.alignment_frame(11),(11,b'older-exact-jpeg'))
        monitor.analyzed_frame.assert_called_once_with(11)
        monitor.status_for_frame.return_value=None
        with self.assertRaises(RuntimeError):controller.alignment_preview(11)

    def test_pin_selection_and_independent_clear_routes_preserve_the_other_point(self):
        controller=dashboard.Controller('unused',alignment_options={})
        controller.state.update(mode='holding',right_arm_frozen=True)
        controller.heartbeat_at=time.monotonic()
        monitor=Mock();monitor.status_for_frame.return_value=dict(frame_id=17,image_size_wh=[640,480]);monitor.status.return_value={'status':'ready'}
        controller.alignment=monitor
        self.assertTrue(controller.snapshot()['alignment_manual_pin_supported'])
        controller.capture_alignment_pin('17',[270,308])
        monitor.capture_pin.assert_called_once_with(17,[270.,308.])
        controller.clear_alignment_hole()
        monitor.clear_hole.assert_called_once_with()
        monitor.clear_pin.assert_not_called();monitor.reset.assert_not_called()
        controller.clear_alignment_pin()
        monitor.clear_pin.assert_called_once_with()
        monitor.reset.assert_not_called()
        for point in ([True,2],['2',3],[640,0],[0,float('nan')]):
            with self.assertRaises(ValueError):controller.capture_alignment_pin(17,point)

    def test_operator_source_authorizes_geometry_without_promoting_cnn_fallback(self):
        q=np.zeros(14)
        proposal=OperatorAlignmentNudgeTests.proposal(q,hole_source='captured_operator_reference',
                   geometry={'valid':True,'unclamped_full_delta_left_m':[0,0,0]})
        dashboard.Controller.validate_nudge_preview(proposal,q,17)
        with self.assertRaises(RuntimeError):
            dashboard.Controller.validate_nudge_preview(dict(proposal,hole_source='cnn_unverified_fallback'),q,17)


class OperatorAlignmentNudgeTests(unittest.TestCase):
    @staticmethod
    def proposal(state, **values):
        result = dict(frame_id=17, analyzed_timestamp_ns=time.monotonic_ns(),
                      nudge_valid=True, measured_state14=list(state),
                      hole_source='captured_demo_reference', nudge_left_mm=[.3, -.4, .2])
        result.update(values)
        return result

    def test_left_carrier_cnn_sources_allow_bounded_nudge_without_demo_cache(self):
        q = np.zeros(14)
        for source in ('left_carrier_cnn_observed', 'left_carrier_opencv_observed', 'left_carrier_layout_inferred'):
            proposal = self.proposal(q, hole_source=source, cache=None)
            np.testing.assert_allclose(dashboard.Controller.validate_nudge_preview(proposal, q, 17), [.0003, -.0004, 0.])
            with self.assertRaises(RuntimeError):
                dashboard.Controller.validate_nudge_preview(dict(proposal, nudge_valid=False), q, 17)

    def test_retained_hole_requires_stationary_pose_for_nudge(self):
        q=np.zeros(14)
        proposal=self.proposal(q,hole_source='left_carrier_stationary_accumulated',
                               accumulation=dict(stationary=True))
        np.testing.assert_allclose(dashboard.Controller.validate_nudge_preview(proposal,q,17),[.0003,-.0004,0.])
        for stationary in (False,None):
            with self.assertRaisesRegex(RuntimeError,'stationary'):
                dashboard.Controller.validate_nudge_preview(dict(proposal,accumulation=dict(stationary=stationary)),q,17)

    def test_five_mm_setting_preserves_active_downward_load(self):
        controller=dashboard.Controller('unused',alignment_options={})
        controller.state.update(mode='force_holding',right_arm_frozen=True,
                                alignment_force=dict(active=True,force_n=3.))
        controller._force_xy_requested={'old':'queued'}
        controller.configure_alignment({'max_step_mm':5.})
        self.assertFalse(controller._force_stop_requested)
        self.assertIsNone(controller._force_xy_requested)
        self.assertEqual(controller.snapshot()['mode'],'force_holding')
        self.assertTrue(controller.snapshot()['alignment_force']['active'])
        q=np.zeros(14)
        proposal=self.proposal(q,nudge_left_mm=[3.,4.,7.])
        np.testing.assert_allclose(dashboard.Controller.validate_nudge_preview(proposal,q,17),[.003,.004,0.])
        with self.assertRaises(RuntimeError):
            dashboard.Controller.validate_nudge_preview(dict(proposal,nudge_left_mm=[3.01,4.,0.]),q,17)

    def test_proposal_guards_stale_state_source_frame_and_hard_cap(self):
        q = np.zeros(14)
        for values in (dict(frame_id='wrong'), dict(analyzed_timestamp_ns=1),
                       dict(hole_source='cnn_unverified_fallback'), dict(nudge_valid=False),
                       dict(nudge_left_mm=[10.1,0,0]), dict(nudge_left_mm=[0,0,0]),
                       dict(measured_state14=[.1]*14)):
            with self.assertRaises(RuntimeError):
                dashboard.Controller.validate_nudge_preview(self.proposal(q, **values), q, 17)
        np.testing.assert_allclose(dashboard.Controller.validate_nudge_preview(self.proposal(q), q, 17), [.0003,-.0004,0.])

    def test_older_nudge_frames_use_configured_age_limit_and_current_pose(self):
        now=10_000_000_000;q=np.zeros(14)
        old=self.proposal(q,analyzed_timestamp_ns=now-1_000_000_000,
                          config={'frequency_hz':2.,'max_frame_age_ms':1250.})
        dashboard.Controller.validate_nudge_preview(old,q,17,now_ns=now)
        for frequency in (.2,2.,10.):
            dashboard.Controller.validate_nudge_preview(dict(old,
                analyzed_timestamp_ns=now-4_900_000_000,
                config={'frequency_hz':frequency}),q,17,now_ns=now)
            dashboard.Controller.validate_nudge_preview(dict(old,
                analyzed_timestamp_ns=now-9_900_000_000,
                config={'frequency_hz':frequency,'max_frame_age_ms':10000.}),q,17,now_ns=now)
        for proposal in (dict(old,analyzed_timestamp_ns=now-1_300_000_000),
                         dict(old,analyzed_timestamp_ns=now-5_100_000_000,config={'frequency_hz':10.}),
                         dict(old,analyzed_timestamp_ns=now-10_100_000_000,config={'frequency_hz':.2,'max_frame_age_ms':20000.}),
                         dict(old,analyzed_timestamp_ns=now+1),
                         dict(old,measured_state14=[.021]*14)):
            with self.assertRaises(RuntimeError):dashboard.Controller.validate_nudge_preview(proposal,q,17,now_ns=now)
        controller=dashboard.Controller('unused')
        controller.state.update(mode='holding',right_arm_frozen=True)
        controller.heartbeat_at=time.monotonic()
        monitor=Mock();monitor.status.return_value=dict(status='pending',frame_id=None)
        monitor.status_for_frame.return_value=self.proposal(q,analyzed_timestamp_ns=time.monotonic_ns()-1_000_000_000,
                                                         config={'frequency_hz':2.,'max_frame_age_ms':1250.})
        controller.alignment=monitor
        controller.request_alignment_nudge(17)
        self.assertEqual(controller._nudge_requested['proposal']['frame_id'],17)
        self.assertEqual(controller.snapshot()['mode'],'nudge_queued')

    def test_actual_ik_and_bounded_trajectory_fake_hardware_only(self):
        class PoseHardware(FakeHardware):
            def __init__(self):
                super().__init__()
                self.lower = np.full(14,-5.)
                self.upper = np.full(14,5.)
                self.position = np.r_[[.2,1.,1.4,.3,.4,.2],.6,[.2,1.,1.4,.3,.4,.2],.7]
                self.motor_client = Mock()
        from pin_alignment_motion import NativeToolKinematics
        from pin_regrasp_motion import rotation_log
        hardware = PoseHardware()
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard,'request',side_effect=fake_request):
            controller = dashboard.Controller('unused',lambda:hardware,log_path=Path(directory)/'steps.jsonl',data_dir=directory)
            controller.connect(reset_pose=False)
            try:
                wait_for(lambda:controller.snapshot()['mode']=='holding')
                controller.heartbeat()
                initial = hardware.position.copy()
                controller.request_right_arm_freeze(True)
                wait_for(lambda:controller.snapshot()['right_arm_frozen'])
                monitor = Mock()
                monitor.status.return_value = self.proposal(initial)
                controller.alignment = monitor
                controller.request_alignment_nudge(17)
                with self.assertRaises(RuntimeError):
                    controller.submit('step',1,0)
                wait_for(lambda:controller.snapshot()['alignment_nudge']['status']=='done')
                final = hardware.position.copy()
                kinematics = NativeToolKinematics()
                p0,r0 = kinematics.pose(initial[:6]); p1,r1 = kinematics.pose(final[:6])
                np.testing.assert_allclose(p1-p0,[.0003,-.0004,0.],atol=2e-5)
                self.assertLess(np.linalg.norm(rotation_log(r1@r0.T)),2e-4)
                np.testing.assert_array_equal(final[6:],initial[6:])
                self.assertGreaterEqual(hardware.motor_client.validate_path.call_count,2)
                self.assertEqual(controller.snapshot()['steps_completed'],0)
                events = [json.loads(line) for line in (Path(directory)/'steps.jsonl').read_text().splitlines()]
                self.assertTrue(any(e.get('event')=='operator_nudge' for e in events))
            finally:
                controller.alignment = None
                controller.shutdown(); controller.worker.join(2)

    def test_five_mm_native_ik_path_and_submicron_residual_are_valid(self):
        hardware=FakeHardware();hardware.lower=np.full(14,-5.);hardware.upper=np.full(14,5.)
        q=np.r_[[.2,1.,1.4,.3,.4,.2],.6,[.2,1.,1.4,.3,.4,.2],.7]
        for delta in ([.005,0,0],[0,.005,0]):
            planned=dashboard.Controller.plan_operator_nudge(hardware,q,q,np.asarray(delta),q[7:])
            metadata=dashboard.Controller.validate_nudge_commit(planned,q)
            self.assertLessEqual(metadata['fresh_translation_norm_m'],.0051)
            self.assertLessEqual(metadata['fresh_max_path_translation_error_m'],.00001)
            self.assertLessEqual(metadata['fresh_max_path_height_error_m'],.00001)
            self.assertLessEqual(metadata['fresh_max_path_orientation_error_rad'],.01)
        controller=dashboard.Controller('unused')
        self.assertEqual(controller.snapshot()['alignment_limits']['max_nudge_mm'],5)
        proposal=self.proposal(q,nudge_left_mm=[5,0,0])
        dashboard.Controller.validate_nudge_preview(proposal,q,17)

    def test_commit_replays_encoder_noise_but_rejects_physical_drift(self):
        # Read-only 8093 hold samples, 2026-11-04: joint2 toggled one count.
        q=np.array([-.9236282110214233,1.6725839376449585,.9054070115089417,
                    -.11686725169420242,.1212654784321785,.23436281085014343])
        initial=np.r_[q,.6,q,.7]
        fresh=initial.copy();fresh[1]=1.6732122898101807
        hardware=FakeHardware();hardware.lower=np.full(14,-5.);hardware.upper=np.full(14,5.)
        planned=dashboard.Controller.plan_operator_nudge(hardware,initial,initial,np.array([.0004,0,0]),initial[7:])
        with self.assertRaises(dashboard.NudgeReanchorRequired):
            dashboard.Controller.validate_nudge_commit(planned,fresh)
        reanchored=dashboard.Controller.plan_operator_nudge(hardware,fresh,initial,np.array([.0004,0,0]),initial[7:])
        metadata=dashboard.Controller.validate_nudge_commit(reanchored,fresh)
        p_fresh,_=reanchored['kinematics'].pose(fresh[:6])
        p_end,_=reanchored['kinematics'].pose(reanchored['target14'][:6])
        self.assertLess(abs(p_end[2]-p_fresh[2]),.00001)
        self.assertLessEqual(metadata['fresh_translation_norm_m'],.002)
        self.assertLessEqual(metadata['fresh_max_path_translation_error_m'],.00005)
        self.assertEqual(metadata['fresh_path_samples'],101)
        moved=initial.copy();moved[1]+=2*(2*np.pi/10000)
        with self.assertRaises(dashboard.NudgeReanchorRequired):
            dashboard.Controller.validate_nudge_commit(planned,moved)
        moved[1]=initial[1]+.004
        with self.assertRaisesRegex(RuntimeError,'fresh-pose safety envelope.*tool drift.*orientation drift'):
            dashboard.Controller.validate_nudge_commit(planned,moved)
        # Quantization tolerance cannot increase the hard fresh-start displacement.
        kin=planned['kinematics'];p0,_=kin.pose(initial[:6]);p1,_=kin.pose(fresh[:6])
        class LinearKinematics:
            def pose(self,q):return np.asarray(q)[:3].copy(),np.eye(3)
        long_target=np.zeros(14);long_target[0]=.0049
        long_plan=dict(initial_state14=np.zeros(14),target14=long_target,
                       initial_position=np.zeros(3),initial_rotation=np.eye(3),kinematics=LinearKinematics())
        # Submicron solver residual is not physical motion or a useful stop.
        residual_plan=dict(long_plan,target14=np.r_[.0050007,np.zeros(13)])
        dashboard.Controller.validate_nudge_commit(residual_plan,np.zeros(14))
        two_mm_residual=dict(long_plan,target14=np.r_[.0020007,np.zeros(13)])
        dashboard.Controller.validate_nudge_commit(two_mm_residual,np.zeros(14))
        one_count=np.zeros(14);one_count[0]=-2*np.pi/10000
        with self.assertRaises(dashboard.NudgeReanchorRequired):
            dashboard.Controller.validate_nudge_commit(long_plan,one_count)
        # Even a small joint-space endpoint must retain the orientation check.
        bad_plan=dict(planned,target14=np.asarray(planned['target14']).copy())
        bad_plan['target14'][5]+=.02
        with self.assertRaisesRegex(RuntimeError,'Cartesian/orientation bounds'):
            dashboard.Controller.validate_nudge_commit(bad_plan,initial)

    def test_pause_cancels_async_plan_without_late_motion(self):
        hardware = FakeHardware()
        release = threading.Event()
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard,'request',side_effect=fake_request):
            controller = dashboard.Controller('unused',lambda:hardware,log_path=Path(directory)/'steps.jsonl',data_dir=directory)
            controller.connect(reset_pose=False)
            try:
                wait_for(lambda:controller.snapshot()['mode']=='holding')
                controller.heartbeat(); controller.request_right_arm_freeze(True)
                wait_for(lambda:controller.snapshot()['right_arm_frozen'])
                monitor = Mock(); monitor.status.return_value = self.proposal(hardware.position)
                controller.alignment = monitor
                def delayed(*args):
                    release.wait(1)
                    return dict(target14=np.r_[[.01]*6,[0]*8],metadata={})
                with patch.object(controller,'plan_operator_nudge',side_effect=delayed):
                    controller.request_alignment_nudge(17)
                    wait_for(lambda:controller.snapshot()['mode']=='nudge_planning')
                    controller.stop(); release.set()
                    wait_for(lambda:controller.snapshot()['mode']=='holding')
                    time.sleep(.05)
                    np.testing.assert_array_equal(hardware.position,np.zeros(14))
            finally:
                release.set();controller.alignment=None;controller.shutdown();controller.worker.join(2)

class GuardedForceHardware(FakeHardware):
    """No devices: records commands and supplies deterministic joint feedback."""
    supports_guarded_load = True
    guarded_load_limits = dict(max_xy_m=.03,max_speed_m_s=.05)
    guarded_load_status = None
    def __init__(self):
        super().__init__()
        self.position = np.r_[[.2,1.,1.4,.3,.4,.2],.6,[.2,1.,1.4,.3,.4,.2],.7]
        self.lower = np.full(14,-5.)
        self.upper = np.full(14,5.)
        self.load_commands = []
    def read(self):
        arm = dict(velocity_rad_s=[0.]*6,gravity_model_torque_nm=[0.]*6)
        return self.position.copy(), {'left':dict(arm),'right':dict(arm)}, {}
    def command_guarded_load(self,arms,target,assist,guard):
        self.position[6:] = np.asarray(target)[6:]
        self.load_commands.append(dict(assist=np.asarray(assist).copy(),guard=dict(guard),target=np.asarray(target).copy()))


class OperatorGuardedForceTests(unittest.TestCase):
    def run_controller(self, hardware, callback, request_fn=fake_request):
        with tempfile.TemporaryDirectory() as directory, patch.object(dashboard,'request',side_effect=request_fn):
            controller = dashboard.Controller('unused',lambda:hardware,log_path=Path(directory)/'steps.jsonl',data_dir=directory)
            controller.connect(reset_pose=False)
            try:
                wait_for(lambda:controller.snapshot()['mode']=='holding')
                controller.heartbeat();controller.request_right_arm_freeze(True)
                wait_for(lambda:controller.snapshot()['right_arm_frozen'])
                callback(controller)
            finally:
                controller.alignment = None
                controller.shutdown();controller.worker.join(2)

    def test_auto_insert_requires_monitor_and_backend_and_accepts_zero_only_auto(self):
        hardware = GuardedForceHardware()
        def check(controller):
            with self.assertRaises(RuntimeError):
                controller.request_alignment_insert(dict(pin_on_carrier=True, frame_id=17))
            monitor = Mock()
            proposal = OperatorAlignmentNudgeTests.proposal(hardware.position, nudge_left_mm=[0,0,0])
            monitor.status.return_value = proposal
            controller.alignment = monitor
            with self.assertRaises(RuntimeError):
                controller.request_alignment_nudge(17)
            monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(hardware.position, nudge_left_mm=[0,0,0])
            controller.state['alignment_force_supported'] = False
            with self.assertRaises(RuntimeError):
                controller.request_alignment_insert(dict(pin_on_carrier=True, frame_id=17))
            controller.state['alignment_force_supported'] = True
            controller.request_alignment_insert(dict(pin_on_carrier=True, frame_id=17))
            wait_for(lambda:len(hardware.load_commands)>=3)
            self.assertEqual(controller.snapshot()['mode'], 'force_auto')
            self.assertTrue(controller.snapshot()['alignment_force']['active'])
            wait_for(lambda:controller.snapshot()['alignment_force'].get('ramped_force_n', 0.) > 0.)
            controller.request_alignment_force_stop()
            wait_for(lambda:controller.snapshot()['mode']=='holding')
            count = len(hardware.load_commands)
            time.sleep(.03)
            self.assertEqual(len(hardware.load_commands), count)
        self.run_controller(hardware, check)

    def test_auto_insert_waits_for_detection_and_recovers_without_restarting_load(self):
        for changes in [dict(nudge_valid=False), dict(analyzed_timestamp_ns=0),
                        dict(hole_source='unsupported')]:
            with self.subTest(changes=changes):
                hardware = GuardedForceHardware()
                def check(controller):
                    monitor = Mock()
                    monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(
                        hardware.position, nudge_left_mm=[0,0,0], **changes)
                    controller.alignment = monitor
                    # No initial analyzed frame is needed while detection warms up.
                    controller.request_alignment_insert(dict(pin_on_carrier=True))
                    wait_for(lambda:len(hardware.load_commands)>=3)
                    wait_for(lambda:controller.snapshot()['alignment_force'].get('waiting_reason'))
                    self.assertEqual(controller.snapshot()['mode'], 'force_auto')
                    self.assertTrue(controller.snapshot()['alignment_force']['active'])
                    guard_id = hardware.load_commands[-1]['guard']['id']
                    monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(
                        hardware.position, frame_id=18, nudge_left_mm=[0,0,0])
                    wait_for(lambda:not controller.snapshot()['alignment_force'].get('waiting_reason'))
                    self.assertTrue(controller.snapshot()['alignment_force']['active'])
                    self.assertEqual(hardware.load_commands[-1]['guard']['id'], guard_id)
                    # Losing the carrier again pauses rather than stopping loading.
                    monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(
                        hardware.position, nudge_valid=False)
                    wait_for(lambda:controller.snapshot()['alignment_force'].get('waiting_reason'))
                    self.assertEqual(controller.snapshot()['mode'], 'force_auto')
                    controller.request_alignment_force_stop()
                    wait_for(lambda:controller.snapshot()['mode']=='holding')
                self.run_controller(hardware, check)

    def test_auto_insert_detection_loss_keeps_committed_xy_target(self):
        from pin_alignment_motion import NativeToolKinematics
        from pin_regrasp_motion import solve_pose
        hardware = GuardedForceHardware()
        def check(controller):
            monitor = Mock()
            monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(
                hardware.position, nudge_left_mm=[.4,0,0])
            controller.alignment = monitor
            kin = NativeToolKinematics()
            q = hardware.position[:6].copy()
            position, rotation = kin.pose(q)
            goal = position[:2] + [.0004,0]
            controller.request_alignment_insert(dict(pin_on_carrier=True))
            wait_for(lambda:controller.snapshot()['alignment_force'].get('nudge_count') == 1)
            guard_id = hardware.load_commands[-1]['guard']['id']
            monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(
                hardware.position, nudge_valid=False)
            # Simulate measured overshoot while the carrier is unavailable.
            # The accepted target must remain fixed so the controller can recover.
            hardware.position[:6] = solve_pose(kin, q, position+[.001,0,0], rotation,
                                               hardware.lower[:6], hardware.upper[:6])
            wait_for(lambda:controller.snapshot()['alignment_force'].get('preview_unavailable'))
            wait_for(lambda:np.allclose(controller.snapshot()['alignment_force']['xy_target'],
                                       goal, atol=1e-9, rtol=0))
            status = controller.snapshot()['alignment_force']
            np.testing.assert_allclose(status['xy_target'], goal, atol=1e-9, rtol=0)
            self.assertGreater(np.linalg.norm(np.asarray(status['measured_xyz'])[:2]-goal), .0004)
            self.assertEqual(status['nudge_count'], 1)
            self.assertTrue(status['active'])
            self.assertEqual(hardware.load_commands[-1]['guard']['id'], guard_id)
            controller.request_alignment_force_stop()
            wait_for(lambda:controller.snapshot()['mode']=='holding')
        self.run_controller(hardware, check)

    def test_auto_insert_three_loaded_button_passes_then_axial_without_detection_or_tracking_wait(self):
        from pin_alignment_insert import loaded_xy_nudge_goal
        from pin_alignment_motion import NativeToolKinematics
        hardware = GuardedForceHardware()
        original = hardware.position.copy()
        def check(controller):
            monitor = Mock()
            monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(
                hardware.position, frame_id=time.monotonic_ns(), nudge_left_mm=[.05,0,0])
            controller.alignment = monitor
            controller.request_alignment_insert(dict(pin_on_carrier=True))
            wait_for(lambda:controller.snapshot()['alignment_force'].get('phase') == 'axial_retry')
            status = controller.snapshot()['alignment_force']
            self.assertEqual(status['nudge_count'], 3)
            guard_id = hardware.load_commands[-1]['guard']['id']
            events = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
            nudges = [row for row in events if row.get('event') == 'operator_auto_insert_nudge']
            self.assertEqual(len(nudges), 3)
            anchor, _ = NativeToolKinematics().pose(original[:6])
            expected = loaded_xy_nudge_goal(anchor, [.00005,0,0], anchor)
            for row in nudges:
                np.testing.assert_allclose(row['goal_xy_m'], expected)
            monitor.status.side_effect = lambda: dict(nudge_valid=False)
            wait_for(lambda:controller.snapshot()['alignment_force'].get('axial_retry_mm', 0) > .7)
            status = controller.snapshot()['alignment_force']
            self.assertTrue(status['active'])
            self.assertEqual(status['nudge_count'], 3)
            self.assertFalse(status['preview_unavailable'])
            self.assertIsNone(status['waiting_reason'])
            self.assertGreater(status['ramped_force_n'], 0.)
            self.assertEqual(hardware.load_commands[-1]['guard']['id'], guard_id)
            # Stuck feedback never blocks the commanded axial movement.
            np.testing.assert_array_equal(hardware.position[:6], original[:6])
            controller.request_alignment_force_stop()
            wait_for(lambda:controller.snapshot()['mode']=='holding')
        with patch('pin_alignment_insert.AUTO_NUDGE_INTERVAL_S', .05):
            self.run_controller(hardware, check)

    def test_auto_insert_sustained_drop_stops_as_candidate(self):
        from pin_alignment_motion import NativeToolKinematics
        from pin_regrasp_motion import solve_pose
        hardware = GuardedForceHardware()
        def check(controller):
            monitor = Mock()
            monitor.status.side_effect = lambda: OperatorAlignmentNudgeTests.proposal(hardware.position, nudge_left_mm=[0,0,0])
            controller.alignment = monitor
            controller.request_alignment_insert(dict(pin_on_carrier=True, frame_id=17,
                drop_threshold_mm=.5, drop_debounce_s=.1))
            wait_for(lambda:controller.snapshot()['alignment_force'].get('baseline_z_m') is not None)
            kin = NativeToolKinematics()
            q = hardware.position[:6].copy()
            position, rotation = kin.pose(q)
            hardware.position[:6] = solve_pose(kin, q, position+[0,0,-.001], rotation,
                                                hardware.lower[:6], hardware.upper[:6])
            wait_for(lambda:not controller.snapshot()['alignment_force']['active'])
            status = controller.snapshot()['alignment_force']
            self.assertEqual(status['reason_code'], 'insertion_candidate')
            self.assertFalse(status['insertion_confirmed'])
            self.assertEqual(controller.snapshot()['mode'], 'holding')
        self.run_controller(hardware, check)

    def test_offline_policy_allows_local_guarded_force(self):
        hardware = GuardedForceHardware()
        def offline(*args, **kwargs):
            raise OSError('AWS offline')
        def check(controller):
            self.assertFalse(controller.snapshot()['policy_ready'])
            controller.request_alignment_force(dict(force_n=.5, pin_on_carrier=True))
            wait_for(lambda: len(hardware.load_commands) >= 3)
            self.assertEqual(controller.snapshot()['mode'], 'force_holding')
            self.assertTrue(controller.snapshot()['right_arm_frozen'])
            controller.request_alignment_force_stop()
            wait_for(lambda: controller.snapshot()['mode'] == 'holding')
            self.assertTrue(controller.snapshot()['connected'])
        self.run_controller(hardware, check, request_fn=offline)

    def test_load_configuration_generous_duration_and_auto_dither(self):
        force, config = dashboard.Controller.force_load_config(dict(pin_on_carrier=True,
            max_duration_s=300.,
            oscillation_amplitude_mm=2.,oscillation_frequency_hz=40.))
        self.assertEqual(config.max_duration_s,300.)
        self.assertEqual(config.oscillation_amplitude_m,.002)
        self.assertEqual(config.oscillation_frequency_hz,40.)
        _, defaults = dashboard.Controller.force_load_config(dict(pin_on_carrier=True))
        self.assertEqual(defaults.max_duration_s,120.)
        self.assertEqual(defaults.insertion_drop_m,.003)
        self.assertEqual(defaults.oscillation_amplitude_m,.0002)
        self.assertEqual(defaults.oscillation_frequency_hz,15.)

    def test_legacy_backend_and_missing_ack_refused(self):
        def check(controller):
            self.assertFalse(controller.snapshot()['alignment_force_supported'])
            with self.assertRaises(RuntimeError):
                controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            with self.assertRaises(ValueError):
                controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=False))
            with self.assertRaises(ValueError):
                controller.request_alignment_force(dict(force_n=3.1,pin_on_carrier=True))
        self.run_controller(FakeHardware(),check)

    def test_load_xy_nudge_and_stop(self):
        hardware = GuardedForceHardware()
        initial = hardware.position.copy()
        def check(controller):
            self.assertTrue(controller.snapshot()['alignment_force_supported'])
            controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            wait_for(lambda:len(hardware.load_commands)>=3)
            self.assertEqual(controller.snapshot()['mode'],'force_holding')
            with self.assertRaises(RuntimeError): controller.submit('play',1,0)
            with self.assertRaises(RuntimeError): controller.request_right_arm_freeze(False)
            monitor = Mock()
            proposal = OperatorAlignmentNudgeTests.proposal(initial,nudge_left_mm=[.5,0,1.])
            monitor.status.return_value = proposal
            controller.alignment = monitor
            controller.request_alignment_nudge(17)
            wait_for(lambda:controller.snapshot()['alignment_nudge']['status']=='executing')
            self.assertTrue(controller.snapshot()['alignment_nudge']['xy_only'])
            # Targets slew while the fake feedback stays still; Z remains compliant.
            xy0 = np.asarray(controller.snapshot()['alignment_force']['xy_target'])
            time.sleep(.025)
            xy1 = np.asarray(controller.snapshot()['alignment_force']['xy_target'])
            self.assertLessEqual(np.linalg.norm(xy1-xy0),.010*.06)
            controller.request_alignment_force_stop()
            wait_for(lambda:not controller.snapshot()['alignment_force']['active'])
            wait_for(lambda:controller.snapshot()['mode']=='holding')
            count=len(hardware.load_commands);time.sleep(.03)
            self.assertEqual(len(hardware.load_commands),count)
            np.testing.assert_array_equal(hardware.position[6:],initial[6:])
            self.assertEqual(controller.snapshot()['steps_completed'],0)
        self.run_controller(hardware,check)

    def test_removed_spiral_settings_are_rejected(self):
        for key, value in [('radius_mm', 1.), ('turns', 2.), ('speed_mm_s', 1.)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                dashboard.Controller.force_load_config(dict(pin_on_carrier=True, **{key:value}))

    def test_guard_latch_and_global_pause_remove_load(self):
        hardware = GuardedForceHardware()
        def check(controller):
            controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            wait_for(lambda:len(hardware.load_commands)>1)
            identity=hardware.load_commands[-1]['guard']['id']
            hardware.guarded_load_status=dict(id=identity,reason='test_guard_deadline')
            wait_for(lambda:not controller.snapshot()['alignment_force']['active'])
            self.assertIn('test_guard_deadline',controller.snapshot()['alignment_force']['reason'])
            controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            wait_for(lambda:controller.snapshot()['alignment_force']['active'])
            wait_for(lambda:hardware.load_commands[-1]['guard']['id']!=identity)
            self.assertNotEqual(hardware.load_commands[-1]['guard']['id'],identity)
            controller.stop()
            wait_for(lambda:not controller.snapshot()['alignment_force']['active'])
            count=len(hardware.load_commands);time.sleep(.025)
            self.assertEqual(len(hardware.load_commands),count)
            self.assertTrue(controller.snapshot()['right_arm_frozen'])
        self.run_controller(hardware,check)

    def test_guard_stop_between_read_and_flush_returns_to_hold(self):
        class RaceHardware(GuardedForceHardware):
            failed=False
            def command_guarded_load(self,arms,target,assist,guard):
                if not self.failed:
                    self.failed=True
                    fresh=self.position.copy();fresh[0]+=.0002
                    self.position[:6]=fresh[:6]
                    raise dashboard.GuardedLoadStopped(dict(id=guard['id'],reason='guard_race_test'),fresh)
                super().command_guarded_load(arms,target,assist,guard)
        hardware=RaceHardware();initial=hardware.position.copy()
        def check(controller):
            controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            wait_for(lambda:'guard_race_test' in str(controller.snapshot()['alignment_force'].get('reason')))
            self.assertEqual(controller.snapshot()['mode'],'holding')
            self.assertTrue(controller.worker.is_alive())
            wait_for(lambda:np.isclose(hardware.position[0],initial[0]+.0002,atol=1e-10,rtol=0))
            np.testing.assert_array_equal(hardware.position[6:],initial[6:])
            self.assertFalse(controller.snapshot()['alignment_force']['active'])
            controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            wait_for(lambda:len(hardware.load_commands)>0)
            controller.request_alignment_force_stop()
            wait_for(lambda:controller.snapshot()['mode']=='holding')
        self.run_controller(hardware,check)

    def test_unverified_guard_command_transport_error_stays_fault(self):
        class BrokenHardware(GuardedForceHardware):
            def command_guarded_load(self,*args):
                raise RuntimeError('simulated motor transport failure')
        def check(controller):
            controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            wait_for(lambda:controller.snapshot()['mode']=='fault_hold')
            self.assertIn('simulated motor transport failure',controller.snapshot()['error'])
            self.assertFalse(controller.snapshot()['alignment_force']['active'])
        self.run_controller(BrokenHardware(),check)

    def test_stop_retains_rejected_sample_speed_limit_and_logs(self):
        hardware=GuardedForceHardware()
        diagnostic=dict(tool_speed_m_s=.09,down_speed_m_s=.03,velocity_source='measured_pose_window',velocity_window_s=.05)
        def check(controller):
            with patch('pin_alignment_force.CartesianLoadController.step',return_value=dict(valid=False,reason='tool_speed_exceeded',diagnostics=diagnostic)):
                controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
                wait_for(lambda:controller.snapshot()['alignment_force'].get('reason_code')=='tool_speed_exceeded')
            force=controller.snapshot()['alignment_force']
            self.assertIn('90.00 mm/s > 50.00 mm/s',force['reason'])
            self.assertEqual(force['diagnostics']['sdk_tool_speed_m_s'],0)
            self.assertEqual(force['diagnostics']['tool_speed_m_s'],.09)
            entries=[json.loads(line) for line in controller.log_path.read_text().splitlines()]
            stopped=next(row for row in entries if row.get('event')=='operator_force_stop')
            self.assertEqual(len(stopped['measured_state14']),14)
            self.assertGreater(stopped['measured_timestamp_ns'],0)
            self.assertEqual(stopped['diagnostics']['tool_speed_m_s'],.09)
            controller.stop();wait_for(lambda:controller.snapshot()['mode']=='holding')
            time.sleep(.02)
            force=controller.snapshot()['alignment_force']
            self.assertEqual(force['last_stop']['reason_code'],'tool_speed_exceeded')
            self.assertEqual(force['diagnostics']['tool_speed_m_s'],.09)
        self.run_controller(hardware,check)

    def test_running_legacy_guard_ceiling_cannot_advertise_large_force_support(self):
        hardware=GuardedForceHardware();hardware.guarded_load_limits=dict(max_xy_m=.008,max_speed_m_s=.02)
        def check(controller):
            self.assertFalse(controller.snapshot()['alignment_force_supported'])
            with self.assertRaises(RuntimeError):controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
            self.assertEqual(controller.snapshot()['alignment_limits']['max_load_duration_s'],300)
        self.run_controller(hardware,check)

    def test_early_loading_descent_arms_lower_baseline_and_keeps_load_active(self):
        from pin_alignment_motion import NativeToolKinematics
        from pin_regrasp_motion import solve_pose
        hardware = GuardedForceHardware()
        def check(controller):
            controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True,drop_threshold_mm=.5,drop_debounce_s=.1))
            wait_for(lambda:len(hardware.load_commands)>1)
            # Simulate external downward compliance under load. The alignment
            # nudge planner deliberately rejects vertical movement requests.
            kin=NativeToolKinematics()
            q=hardware.position[:6].copy()
            position,rotation=kin.pose(q)
            hardware.position[:6]=solve_pose(kin,q,position+[0,0,-.001],rotation,
                                             hardware.lower[:6],hardware.upper[:6])
            lower_position, _ = kin.pose(hardware.position[:6])
            wait_for(lambda:controller.snapshot()['alignment_force'].get('baseline_z_m') is not None)
            status = controller.snapshot()['alignment_force']
            self.assertTrue(status['active'])
            self.assertAlmostEqual(status['baseline_z_m'], lower_position[2], delta=.0001)
            before = len(hardware.load_commands)
            # Stay at the lower loaded baseline longer than the candidate
            # debounce; the earlier descent must not become an insertion.
            time.sleep(.15)
            self.assertTrue(controller.snapshot()['alignment_force']['active'])
            self.assertIsNone(controller.snapshot()['alignment_force']['reason'])
            self.assertGreater(len(hardware.load_commands), before)
            controller.request_alignment_force_stop()
            wait_for(lambda:controller.snapshot()['mode']=='holding')
            self.assertFalse(controller.snapshot()['alignment_force']['active'])
        self.run_controller(hardware,check)

class RecordedOperatorAssistanceTests(unittest.TestCase):
    def test_real_rollout_store_records_nudge_and_load_without_ownership_fault(self):
        from hitl_demo import DemoHardware
        class RecordedHardware(DemoHardware):
            supports_guarded_load = True
            guarded_load_limits = dict(max_xy_m=.03,max_speed_m_s=.05)
            guarded_load_status = None
            def __init__(self):
                super().__init__()
                self.config['audio_cues']={'enabled':False}
                self.config['foot_pedal']={'enabled':False}
                self.config['episode_pedal']={'enabled':False}
                self.config['control_hz']=100
                self.config['start_delay_seconds']=0
                self.position=np.r_[[.2,1.,1.4,.3,.4,.2],.6,[.2,1.,1.4,.3,.4,.2],.7]
                self.loads=[]
            def command_guarded_load(self,arms,target,assist,guard):
                self.position[6:]=np.asarray(target)[6:]
                self.loads.append(dict(assist=list(assist),guard=guard))
        hardware=RecordedHardware()
        with tempfile.TemporaryDirectory() as directory,patch.object(dashboard,'request',side_effect=fake_request):
            controller=dashboard.Controller('unused',lambda:hardware,log_path=Path(directory)/'events.jsonl',data_dir=directory)
            controller.connect(reset_pose=False)
            try:
                wait_for(lambda:controller.snapshot()['mode']=='holding')
                controller.heartbeat()
                controller.collection('begin',dict(operator='test',scene='simulated recording integration'))
                wait_for(lambda:controller.snapshot()['rollout_active'])
                wait_for(lambda:controller.recorder.snapshot().get('active') is not None)
                controller.request_right_arm_freeze(True)
                wait_for(lambda:controller.snapshot()['right_arm_frozen'])
                monitor=Mock();monitor.status.return_value=OperatorAlignmentNudgeTests.proposal(hardware.position)
                controller.alignment=monitor
                controller.request_alignment_nudge(17)
                wait_for(lambda:controller.snapshot()['alignment_nudge']['status']=='done')
                time.sleep(.04)
                self.assertIsNone(controller.recorder.failure)
                controller.heartbeat()
                controller.request_alignment_force(dict(force_n=.5,pin_on_carrier=True))
                wait_for(lambda:len(hardware.loads)>=4)
                self.assertIsNone(controller.recorder.failure)
                controller.request_alignment_force_stop()
                wait_for(lambda:controller.snapshot()['mode']=='holding')
                controller.collection('finish',dict(label='review',success=False,quality=1,notes='simulated assistance test'))
                wait_for(lambda:not controller.snapshot()['rollout_active'])
                wait_for(lambda:bool(list(Path(directory).glob('review/*/states.jsonl'))))
                rows=[json.loads(line) for path in Path(directory).glob('review/*/states.jsonl') for line in path.read_text().splitlines()]
                sources={row['control_source'] for row in rows}
                self.assertIn('operator_nudge',sources)
                self.assertIn('operator_load',sources)
                assisted=[row for row in rows if row['control_source']=='operator_load']
                self.assertTrue(assisted)
                self.assertTrue(all(row['arms']['left']['command']['mode']=='guarded_gravity' for row in assisted))
                self.assertTrue(all(row['control_source_by_arm']['right']=='frozen_hold' for row in assisted))
                self.assertTrue(all(row['alignment_force']['force_sensor'] is False for row in assisted))
                self.assertIsNone(controller.recorder.failure)
            finally:
                controller.alignment=None;controller.shutdown();controller.worker.join(3)


class LoadedNudgeProgressTests(unittest.TestCase):
    def test_slew_completion_cannot_claim_stalled_arm_moved(self):
        job=dict(started=0.,frame_id=4,start_position=np.zeros(3),goal_xy=np.array([.003,0.]))
        target=job['goal_xy'].copy()
        result=dashboard.loaded_nudge_progress(job,np.zeros(3),target,.5)
        self.assertEqual(result['status'],'executing')
        self.assertEqual(result['tracking_error_mm'],3.)
        result=dashboard.loaded_nudge_progress(job,np.zeros(3),target,4.)
        self.assertEqual(result['status'],'stalled')
        self.assertEqual(result['measured_delta_xyz_mm'],[0.,0.,0.])

    def test_completion_requires_measured_convergence_and_settling(self):
        job=dict(started=0.,frame_id=4,start_position=np.zeros(3),goal_xy=np.array([.003,0.]))
        target=job['goal_xy'].copy()
        position=np.array([.0028,0.,-.0005])
        self.assertEqual(dashboard.loaded_nudge_progress(job,position,target,1.)['status'],'executing')
        result=dashboard.loaded_nudge_progress(job,position,target,1.21)
        self.assertEqual(result['status'],'done')
        np.testing.assert_allclose(result['measured_delta_xyz_mm'],[2.8,0.,-.5])
        self.assertEqual(dashboard.loaded_nudge_progress(job,np.zeros(3),target,1.3)['status'],'executing')


if __name__ == '__main__':
    unittest.main()
