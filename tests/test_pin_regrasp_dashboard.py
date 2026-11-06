"""Hardware-free controller integration, including human intervention boundaries."""
import json
import queue
from pathlib import Path
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import pin_regrasp_dashboard as dashboard


class FakePedal:
    def __init__(self, *_):
        pass

    def snapshot(self):
        return dict(connected=True, sampled_monotonic_ns=time.monotonic_ns(), events=[])

    def close(self):
        pass


class FakeRecorder:
    def __init__(self, root, **_):
        self.root = Path(root)
        self.episodes = []
        self.active = False

    def begin(self, metadata):
        self.episodes.append(dict(metadata=metadata, samples=[], complete=None, reason=None))
        self.active = True

    def append(self, phase, measured, target, jpeg, camera):
        assert self.active
        assert phase in ('return', 'close')
        assert camera['camera'] == 'left_wrist'
        self.episodes[-1]['samples'].append((phase, measured, target))

    def append_frames(self, phase, measured, target, frames):
        assert self.active
        assert set(frames) == set(dashboard.CAMERA_ROLES)
        assert all(frame['camera'] == role and frame['source_camera'] == 'physical_' + role
                   for role, frame in frames.items())
        self.episodes[-1]['samples'].append((phase, measured, target))

    def finish(self, complete, reason):
        self.episodes[-1].update(complete=complete, reason=reason)
        self.active = False

    def snapshot(self):
        return dict(active=self.active, samples=sum(len(e['samples']) for e in self.episodes))


class FakeGestures:
    def __init__(self):
        self.pending = queue.Queue()
        self.delivered = 0

    def update(self, snapshot):
        try:
            gesture = self.pending.get_nowait()
        except queue.Empty:
            return None
        self.delivered += 1
        return gesture() if callable(gesture) else gesture


class FakeKinematics:
    def __init__(self, *_):
        pass

    def pose(self, q):
        return np.asarray(q[:3]), np.eye(3)


class FakeHardware:
    def __init__(self, config_path):
        self.configs = {'left': config_path}
        self.config = {}
        self.lower = np.full(14, -1.)
        self.upper = np.ones(14)
        self.lower[6] = -.35
        self.measured = np.zeros(14)
        self.commands = []
        self.guides = []
        self.closed = False
        self.sequence = 0
        self.cameras = None

    def open(self):
        roles = dashboard.CAMERA_ROLES if self.task == 'alignment' else dashboard.CAMERA_ROLES[:1]
        self.camera_mapping = {role: 'physical_' + role for role in roles}

    def read(self):
        return self.measured.copy(), {}, {name: {'ok': True} for name in self.camera_mapping.values()}

    def frames(self):
        return {role: dict(self.frame(), camera=role, source_camera=name)
                for role, name in self.camera_mapping.items()}

    def frame(self):
        self.sequence += 1
        return dict(jpeg=b'test-jpeg', camera='physical_left_camera', sequence=self.sequence,
                    monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns())

    def command(self, target):
        self.commands.append(target.copy())
        self.measured = target.copy()

    def guide(self, arms, target, enabled):
        self.guides.append(enabled)

    def close(self):
        self.closed = True


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = Path(self.directory.name) / 'arm.yaml'
        config.write_text('{}')
        self.hw = FakeHardware(config)
        self.plans = []
        self.gestures = FakeGestures()

        def planner(anchor, kin, lower, upper, **kwargs):
            self.plans.append((anchor.copy(), kwargs))
            target = kwargs.get('current14', anchor).copy()
            result = []
            for phase, length in [('release', 2), ('retreat', 2), ('return', 45),
                                  ('close', 3), ('lift', 2)]:
                for _ in range(length):
                    result.append((phase, target.copy()))
            return result

        for name, replacement in [('ToolKinematics', FakeKinematics), ('FootPedal', FakePedal),
                                  ('PinRegraspEpisodeStore', FakeRecorder), ('plan_regrasp', planner),
                                  ('InferencePedal', lambda: self.gestures)]:
            patched = patch.object(dashboard, name, replacement)
            patched.start()
            self.addCleanup(patched.stop)
        self.controller = dashboard.Controller(Path(self.directory.name) / 'forward', lambda: self.hw)
        self.addCleanup(self.stop)
        self.controller.request('connect')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'holding')

    def stop(self):
        self.controller.request('disconnect')
        if self.controller.worker:
            self.controller.worker.join(2)
        self.assertFalse(self.controller.worker.is_alive())

    def wait(self, predicate, timeout=3, heartbeat=True):
        deadline = time.monotonic() + timeout
        while not predicate():
            if self.controller.snapshot()['mode'] == 'error':
                self.fail(self.controller.snapshot()['error'])
            if time.monotonic() > deadline:
                self.fail('Timed out: ' + json.dumps(self.controller.snapshot()))
            if heartbeat:
                self.controller.request('heartbeat')
            time.sleep(.005)

    def start_forward(self):
        self.controller.request('capture')
        self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
        self.controller.request('resume')
        self.wait(lambda: self.controller.recorder.active)

    def send_gesture(self, gesture):
        delivered = self.gestures.delivered
        self.gestures.pending.put(gesture)
        self.wait(lambda: self.gestures.delivered > delivered)

    def select_alignment(self):
        self.controller.request('disconnect')
        self.controller.worker.join(2)
        self.assertFalse(self.controller.worker.is_alive())
        self.controller.request('settings', {'task': 'alignment'})
        self.controller.request('connect')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'holding')

    def test_task_switch_requires_disconnect_and_clears_reference(self):
        self.controller.request('capture')
        self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
        original = self.controller.snapshot()['settings']
        with self.assertRaisesRegex(ValueError, 'Disconnect'):
            self.controller.request('settings', {'task': 'alignment', 'retreat_m': .03})
        self.assertEqual(self.controller.snapshot()['settings'], original)
        self.assertIsNotNone(self.controller.snapshot()['anchor'])
        self.controller.request('disconnect')
        self.controller.worker.join(2)
        self.controller.request('settings', {'task': 'alignment'})
        state = self.controller.snapshot()
        self.assertIsNone(state['anchor'])
        self.assertIsNone(state['recording'])
        self.assertEqual(state['cycle'], 0)
        self.assertEqual(Path(state['directory']), self.controller.root / 'alignment')
        conventional = dashboard.Controller(Path(self.directory.name) / 'pin-regrasp-forward')
        conventional.request('settings', {'task': 'alignment'})
        self.assertEqual(conventional.output_directory(), Path(self.directory.name) / 'pin-alignment-forward')

    def test_alignment_planner_records_task_in_separate_directory(self):
        calls = []

        def alignment_plan(anchor, kin, lower, upper, **kwargs):
            calls.append(kwargs)
            target = kwargs['current14'].copy()
            return [(phase, target.copy()) for phase in
                    ['release', 'retreat', 'return', 'return', 'lift']]

        with patch.object(dashboard, 'plan_alignment', alignment_plan):
            self.select_alignment()
            self.start_forward()
            recorder = self.controller.recorder
            self.wait(lambda: recorder.episodes[0]['complete'] is not None)
        self.assertTrue(calls)
        self.assertEqual(self.plans, [])
        self.assertEqual(calls[0]['config'].task, 'alignment')
        self.assertEqual(recorder.root, self.controller.root / 'alignment')
        episode = recorder.episodes[0]
        self.assertEqual(episode['metadata']['task'], 'alignment')
        self.assertEqual(episode['metadata']['camera_mapping'], self.hw.camera_mapping)
        self.assertEqual(set(self.controller.snapshot()['camera_roles']), set(dashboard.CAMERA_ROLES))
        self.assertEqual(episode['metadata']['settings']['task'], 'alignment')
        self.assertEqual([sample[0] for sample in episode['samples']], ['return', 'return'])
        self.assertTrue(episode['complete'])

    def test_missing_alignment_camera_prevents_running(self):
        self.select_alignment()
        original_read = self.hw.read
        def missing_stream():
            measured, arms, cameras = original_read()
            del cameras['physical_overhead']
            return measured, arms, cameras
        self.hw.read = missing_stream
        self.controller.request('capture')
        self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
        self.controller.request('resume')
        self.controller.worker.join(2)
        self.assertEqual(self.controller.snapshot()['mode'], 'error')
        self.assertIn('camera is stale', self.controller.snapshot()['error'])
        self.assertEqual(self.controller.recorder.episodes, [])
        state = self.controller.snapshot()
        self.assertIn('overhead (physical_overhead)', state['error'])
        self.assertIn('missing stream status/receipt', state['error'])
        self.assertTrue(state['camera_health']['overhead']['missing'])
        self.assertFalse(state['camera_health']['overhead']['ok'])
        self.assertIsNone(self.controller.hardware)

    def test_stale_alignment_camera_reports_role_age_and_capture_error(self):
        self.select_alignment()
        original_read = self.hw.read
        def stale_streams():
            measured, arms, cameras = original_read()
            cameras['physical_right_wrist'] = dict(ok=False, age_s=1.2345, sequence=18,
                                                  error='capture timed out')
            cameras['physical_overhead'] = dict(ok=False, age_s=None, sequence=0,
                                               error='device unavailable')
            return measured, arms, cameras
        self.hw.read = stale_streams
        self.controller.request('capture')
        self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
        self.controller.request('resume')
        self.controller.worker.join(2)
        state = self.controller.snapshot()
        self.assertEqual(state['mode'], 'error')
        self.assertIn('right_wrist (physical_right_wrist): frame age 1.234s', state['error'])
        self.assertIn('freshness limit 1.000s', state['error'])
        self.assertIn('capture error: capture timed out', state['error'])
        self.assertIn('overhead (physical_overhead): no captured frame receipt', state['error'])
        self.assertIn('device unavailable', state['error'])
        self.assertEqual(state['camera_health']['right_wrist']['sequence'], 18)
        self.assertEqual(state['camera_health']['right_wrist']['age_s'], 1.2345)
        self.assertFalse(state['camera_health']['overhead']['ok'])
        self.assertEqual(self.controller.recorder.episodes, [])
        self.assertIsNone(self.controller.hardware)

    def test_alignment_adjustment_guides_both_arms_and_holds_measured_pose(self):
        self.select_alignment()
        self.controller.request('capture')
        self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
        self.controller.request('adjust')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'adjusting')
        self.hw.measured[0] = .15
        self.hw.measured[7] = -.2
        self.hw.measured[13] = .3
        time.sleep(.08)
        self.assertEqual(self.hw.guides[-1], {'left': True, 'right': True})
        self.controller.request('pause')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        np.testing.assert_allclose(self.hw.commands[-1][[0, 7, 13]], [.15, -.2, .3])
        self.assertEqual(self.controller.recorder.episodes, [])

    def test_carrier_jitter_setting_limits_and_parallel_frame(self):
        self.controller.request('settings', {'carrier_jitter_m': .10})
        self.assertEqual(self.controller.snapshot()['settings']['carrier_jitter_m'], .10)
        self.assertEqual(self.controller.snapshot()['carrier_frame_mapping'], 'parallel_bases')
        with self.assertRaises(ValueError):
            self.controller.request('settings', {'carrier_jitter_m': .101})
        with self.assertRaisesRegex(ValueError, 'Unknown command'):
            self.controller.request('calibration_capture')

    def test_carrier_pedal_resume_and_live_jitter_settings_need_no_calibration(self):
        calls = []
        def alignment_plan(anchor, kin, lower, upper, **kwargs):
            calls.append(kwargs)
            return [('return', kwargs['current14'].copy()) for _ in range(100)]
        with patch.object(dashboard, 'plan_alignment', alignment_plan):
            self.select_alignment()
            self.hw.configs['right'] = self.hw.configs['left']
            self.controller.request('settings', {'carrier_jitter_m': .005})
            self.controller.request('capture')
            self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
            self.send_gesture('double')
            self.wait(lambda: self.controller.recorder.active)
            self.assertEqual(self.controller.snapshot()['mode'], 'running')
            self.controller.request('settings', {'carrier_jitter_m': .01})
            self.assertEqual(self.controller.snapshot()['settings']['carrier_jitter_m'], .01)
            self.controller.request('pause')
            self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        np.testing.assert_allclose(calls[0]['T_left_base_right_base'], np.eye(4))

    def test_alignment_next_cycle_keeps_current_carrier_and_records_transform(self):
        calls = []
        def carrier_plan(anchor, kin, lower, upper, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                self.controller.request('settings', {'carrier_jitter_m': 0.})
            target = kwargs['current14'].copy()
            target[7] += .01
            return [('return', target.copy()), ('lift', target.copy())]
        self.select_alignment()
        self.hw.configs['right'] = self.hw.configs['left']
        self.controller.request('settings', {'carrier_jitter_m': .005})
        with patch.object(dashboard, 'plan_alignment', carrier_plan):
            self.controller.request('capture')
            self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
            self.controller.request('resume')
            self.wait(lambda: len(calls) >= 2)
            self.controller.request('pause')
            self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        self.assertAlmostEqual(calls[1]['current14'][7], .01)
        self.assertEqual(calls[1]['config'].carrier_jitter_m, 0.)
        np.testing.assert_allclose(calls[1]['T_left_base_right_base'], np.eye(4))
        self.assertIn('right_kinematics', calls[0])
        np.testing.assert_allclose(calls[0]['T_left_base_right_base'], np.eye(4))
        metadata = self.controller.recorder.episodes[0]['metadata']
        self.assertEqual(metadata['T_left_base_right_base'], np.eye(4).tolist())
        self.assertEqual(metadata['carrier_frame_mapping'], 'parallel_bases')
        self.assertTrue(metadata['carrier_mapping_displacement_only'])

    def test_alignment_resume_scales_both_arms_together_within_speed_limits(self):
        def alignment_plan(anchor, kin, lower, upper, **kwargs):
            waypoint = kwargs['current14'].copy()
            waypoint[0], waypoint[7] = .02, .01
            return [('return', waypoint.copy()) for _ in range(100)]
        with patch.object(dashboard, 'plan_alignment', alignment_plan):
            self.select_alignment()
            self.controller.request('settings', {'joint_speed_rad_s': .06})
            self.start_forward()
            self.controller.request('pause')
            self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
            # Human displacement leaves unequal gaps to the same pending waypoint.
            self.hw.measured[0], self.hw.measured[7] = .002, .004
            self.controller.request('pause')
            self.wait(lambda: np.allclose(self.hw.commands[-1][[0, 7]], [.002, .004]))
            recorder = self.controller.recorder
            count = recorder.snapshot()['samples']
            self.controller.request('resume')
            self.wait(lambda: recorder.snapshot()['samples'] > count)
            self.controller.request('pause')
            self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        _, measured, emitted = recorder.episodes[-1]['samples'][0]
        delta = np.asarray(emitted) - measured
        # A shared scale preserves the 0.018 : 0.006 radian gap ratio.
        self.assertAlmostEqual(delta[0], 3 * delta[7])
        self.assertGreater(delta[7], 0.)
        self.assertAlmostEqual(delta[0], .06 / 30)
        self.assertLessEqual(np.max(np.abs(delta[np.r_[0:6, 7:13]])), .06 / 30 + 1e-12)

    def test_alignment_checks_right_arm_drift_while_planning(self):
        import threading
        release = threading.Event()
        entered = threading.Event()
        def delayed_plan(anchor, kin, lower, upper, **kwargs):
            entered.set()
            release.wait(1)
            return [('return', kwargs['current14'].copy())]
        with patch.object(dashboard, 'plan_alignment', delayed_plan):
            self.select_alignment()
            self.controller.request('capture')
            self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
            self.controller.request('resume')
            self.wait(entered.is_set)
            original_read = self.hw.read
            def moved_feedback():
                measured, arms, cameras = original_read()
                measured[7] += .05
                return measured, arms, cameras
            self.hw.read = moved_feedback
            time.sleep(.05)
            release.set()
            self.controller.worker.join(2)
        self.assertEqual(self.controller.snapshot()['mode'], 'error')
        self.assertIn('moved while planning', self.controller.snapshot()['error'])
        self.assertEqual(self.controller.recorder.episodes, [])

    def test_single_pedal_tap_pauses_running_and_stays_paused(self):
        self.start_forward()
        self.send_gesture('single')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        recorder = self.controller.recorder
        count = recorder.snapshot()['samples']
        self.assertFalse(recorder.active)
        self.assertEqual(recorder.episodes[0]['reason'], 'pause')
        self.send_gesture('single')
        time.sleep(.08)
        self.assertEqual(self.controller.snapshot()['mode'], 'paused')
        self.assertEqual(recorder.snapshot()['samples'], count)
        self.assertFalse(self.hw.guides)

    def test_single_pedal_tap_exits_adjustment_to_frozen_hold(self):
        self.start_forward()
        self.controller.request('adjust')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'adjusting')
        self.hw.measured[0] = .15
        self.send_gesture('single')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        guide_count = len(self.hw.guides)
        count = self.controller.recorder.snapshot()['samples']
        time.sleep(.08)
        self.assertEqual(len(self.hw.guides), guide_count)
        self.assertEqual(self.controller.recorder.snapshot()['samples'], count)
        self.assertAlmostEqual(self.hw.commands[-1][0], .15)
        self.assertEqual(len(self.plans), 1)

    def test_single_pedal_tap_overrides_same_tick_queued_resume(self):
        self.start_forward()
        self.controller.request('pause')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        count = self.controller.recorder.snapshot()['samples']

        def single_with_queued_resume():
            self.controller.request('resume')
            return 'single'

        self.send_gesture(single_with_queued_resume)
        time.sleep(.1)
        self.assertEqual(self.controller.snapshot()['mode'], 'paused')
        self.assertTrue(self.controller.commands.empty())
        self.assertEqual(self.controller.recorder.snapshot()['samples'], count)
        self.assertFalse(self.controller.recorder.active)

    def test_double_pedal_tap_still_toggles_pause_and_resume(self):
        self.start_forward()
        self.send_gesture('double')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        self.assertFalse(self.controller.recorder.active)
        self.send_gesture('double')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'running')
        self.wait(lambda: self.controller.recorder.active)

    def test_pause_resume_creates_partial_then_new_forward_episode(self):
        self.start_forward()
        recorder = self.controller.recorder
        self.controller.request('pause')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'paused')
        count = recorder.snapshot()['samples']
        time.sleep(.08)
        self.assertEqual(recorder.snapshot()['samples'], count)
        self.assertFalse(recorder.episodes[0]['complete'])
        self.assertEqual(recorder.episodes[0]['reason'], 'pause')
        self.controller.request('resume')
        self.wait(lambda: len(recorder.episodes) == 2 and recorder.active)
        self.assertTrue(recorder.episodes[1]['metadata']['resumed_mid_forward'])
        self.assertTrue(all(s[0] in ('return', 'close') for e in recorder.episodes for s in e['samples']))

    def test_adjust_drops_plan_recaptures_measured_pose_and_records_no_human_frames(self):
        self.start_forward()
        recorder = self.controller.recorder
        self.controller.request('adjust')
        self.wait(lambda: self.controller.snapshot()['mode'] == 'adjusting')
        before = recorder.snapshot()['samples']
        self.hw.measured[0] = .2
        self.hw.measured[6] = -.1
        time.sleep(.08)
        self.assertEqual(recorder.snapshot()['samples'], before)
        self.assertTrue(self.hw.guides)
        self.assertEqual(self.hw.guides[-1], {'left': True, 'right': False})
        self.controller.request('resume')
        self.wait(lambda: len(self.plans) >= 2)
        self.assertAlmostEqual(self.plans[1][0][0], .2)
        self.assertAlmostEqual(self.plans[1][0][6], -.1)
        self.assertAlmostEqual(self.controller.snapshot()['anchor'][0], .2)
        self.assertEqual(recorder.episodes[0]['reason'], 'adjust')
        self.wait(lambda: len(recorder.episodes) >= 2)
        self.assertFalse(recorder.episodes[1]['metadata']['resumed_mid_forward'])

    def test_only_forward_is_recorded_and_completion_is_explicit(self):
        self.start_forward()
        recorder = self.controller.recorder
        self.wait(lambda: recorder.episodes[0]['complete'] is not None)
        first = recorder.episodes[0]
        self.assertTrue(first['complete'])
        self.assertEqual(first['reason'], 'forward_complete')
        self.assertEqual([s[0] for s in first['samples']], ['return'] * 45 + ['close'] * 3)
        self.assertEqual(first['metadata']['closed_gripper_rad'], -.35)

    def test_capture_clamps_low_gripper_encoder_feedback(self):
        original_read = self.hw.read

        def low_feedback():
            measured, arms, cameras = original_read()
            measured[6] = self.hw.lower[6] - .006
            measured[13] = self.hw.lower[13] - .004
            return measured, arms, cameras

        self.hw.read = low_feedback
        self.controller.request('capture')
        self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
        anchor = self.controller.snapshot()['anchor']
        self.assertEqual(anchor[6], self.hw.lower[6])
        self.assertEqual(anchor[13], self.hw.lower[13])
        self.wait(lambda: self.hw.commands[-1][6] == self.hw.lower[6])
        self.assertEqual(self.hw.commands[-1][13], self.hw.lower[13])
        self.controller.request('resume')
        self.wait(lambda: len(self.plans) > 0)
        self.assertEqual(self.plans[0][0][6], self.hw.lower[6])
        self.assertEqual(self.plans[0][0][13], self.hw.lower[13])

    def test_failed_motor_command_does_not_append_training_sample(self):
        original_command = self.hw.command

        def reject_forward(target):
            if self.controller.recorder.active:
                raise RuntimeError('simulated command rejection')
            original_command(target)

        self.hw.command = reject_forward
        self.controller.request('capture')
        self.wait(lambda: self.controller.snapshot()['anchor'] is not None)
        self.controller.request('resume')
        # Error is the expected result, so do not use wait(), which rejects it.
        self.controller.worker.join(3)
        self.assertFalse(self.controller.worker.is_alive())
        state = self.controller.snapshot()
        self.assertEqual(state['mode'], 'error')
        self.assertIn('simulated command rejection', state['error'])
        recorder = self.controller.recorder
        self.assertEqual(len(recorder.episodes), 1)
        self.assertEqual(recorder.episodes[0]['samples'], [])
        self.assertFalse(recorder.episodes[0]['complete'])
        self.assertFalse(recorder.active)

    def test_real_planner_closes_to_calibrated_lower_limit(self):
        from pin_regrasp_motion import plan_regrasp, RegraspConfig, rotation_exp

        class CartesianKinematics:
            def pose(self, q):
                return np.asarray(q[:3]), rotation_exp(np.asarray(q[3:]))

            def jacobian(self, q):
                return np.eye(6)

        plan = plan_regrasp(np.zeros(14), CartesianKinematics(), self.hw.lower,
                            self.hw.upper, config=RegraspConfig(tool_axis='+z', cone_deg=0, angle_jitter_deg=0))
        closed = [target for phase, target in plan if phase == 'close'][-1]
        self.assertEqual(closed[6], self.hw.lower[6])
        self.assertNotEqual(closed[6], 0.)

    def test_exposed_speed_and_jitter_limits_validate_atomically(self):
        limits = dict(joint_speed_rad_s=.6, gripper_speed_rad_s=5.,
                      speed_m_s=.15, placement_jitter_m=.10)
        self.controller.request('settings', limits)
        accepted = self.controller.snapshot()['settings']
        for key, maximum in limits.items():
            self.assertEqual(accepted[key], maximum)
            with self.subTest(setting=key), self.assertRaises(ValueError):
                self.controller.request('settings', {key: maximum + .001, 'retreat_m': .03})
            self.assertEqual(self.controller.snapshot()['settings'], accepted)

    def test_measured_fk_outside_two_cm_prevents_closing_until_tool_arrives(self):
        lag = [.03]
        planning = []
        original_read = self.hw.read

        def lagging_read():
            measured, arms, cameras = original_read()
            if planning:
                measured[0] += lag[0]
            return measured, arms, cameras

        def closing_plan(anchor, kin, lower, upper, **kwargs):
            planning.append(True)
            # Make at least one fresh feedback tick occur before planning completes.
            time.sleep(.06)
            target = kwargs['current14'].copy()
            target[6] = -.2
            return [('close', target.copy()), ('lift', target.copy())]

        self.hw.read = lagging_read
        patched = patch.object(dashboard, 'plan_regrasp', closing_plan)
        patched.start()
        self.addCleanup(patched.stop)
        self.start_forward()
        recorder = self.controller.recorder
        self.wait(lambda: len(recorder.episodes[0]['samples']) >= 3)
        stalled = list(recorder.episodes[0]['samples'])
        self.assertTrue(all(phase == 'close' for phase, _, _ in stalled))
        self.assertTrue(all(measured[0] > .02 for _, measured, _ in stalled))
        self.assertTrue(all(target[6] == 0. for _, _, target in stalled))
        self.assertIsNone(recorder.episodes[0]['complete'])
        # The same pending close waypoint can advance once physical feedback arrives.
        lag[0] = .01
        self.wait(lambda: any(target[6] < 0. for _, _, target in recorder.episodes[0]['samples']))
        closing = [(measured, target) for _, measured, target in recorder.episodes[0]['samples']
                   if target[6] < 0.]
        self.assertTrue(closing)
        self.assertTrue(all(abs(measured[0]) <= .02 for measured, _ in closing))
        self.wait(lambda: recorder.episodes[0]['complete'] is True)

    def test_invalid_settings_rejected_atomically(self):
        original = self.controller.snapshot()['settings']
        for payload in ({'retreat_m': .5}, {'cone_deg': 90}, {'speed_m_s': 0},
                        {'placement_jitter_m': float('nan')}, {'tool_axis': 'bad'}, {'unknown': 1}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.controller.request('settings', payload)
            self.assertEqual(self.controller.snapshot()['settings'], original)
        self.controller.request('settings', {'retreat_m': .03})
        self.assertEqual(self.controller.snapshot()['settings']['retreat_m'], .03)

    def test_expired_heartbeat_pauses_and_closes_partial_episode(self):
        self.start_forward()
        self.controller.heartbeat_at = time.monotonic() - 3
        self.wait(lambda: self.controller.snapshot()['mode'] == 'paused', heartbeat=False)
        recorder = self.controller.recorder
        self.assertFalse(recorder.active)
        self.assertEqual(recorder.episodes[0]['reason'], 'pause')
        self.assertFalse(recorder.episodes[0]['complete'])


class CameraHardwareTests(unittest.TestCase):
    def test_task_camera_selection_and_copied_receipts(self):
        config = json.loads((dashboard.ROOT / 'config/recording.json').read_text())
        robot = SimpleNamespace(joint_limits={'lower': [-1]*6, 'upper': [1]*6},
                                gripper_limits={'lower': -1, 'upper': 1})
        for task, roles in [('regrasp', dashboard.CAMERA_ROLES[:1]), ('alignment', dashboard.CAMERA_ROLES)]:
            with self.subTest(task=task), patch.object(dashboard, 'Cameras') as cameras, patch.object(dashboard, 'MotorClient') as motor:
                motor.return_value.robots = {'left': robot, 'right': robot}
                hw = dashboard.RegraspHardware()
                hw.task = task
                hw.open()
                selected = cameras.call_args.args[1]['cameras']
                self.assertEqual(set(selected), {config['alignment_assist']['camera_mapping'][r] for r in roles})
                hw.cameras.lock = threading.Lock()
                hw.cameras.latest = {name: {'jpeg': b'jpeg', 'sequence': 1} for name in selected}
                frames = hw.frames()
                self.assertEqual(set(frames), set(roles))
                for role, receipt in frames.items():
                    self.assertEqual(receipt['camera'], role)
                    self.assertEqual(receipt['source_camera'], hw.camera_mapping[role])
                    receipt['sequence'] = 9
                    self.assertEqual(hw.cameras.latest[hw.camera_mapping[role]]['sequence'], 1)
                self.assertEqual(hw.frame()['camera'], 'left_wrist')
                controller = dashboard.Controller('/tmp/test')
                controller.hardware = hw
                controller.jpeg()
                hw.cameras.jpeg.assert_called_with(hw.camera_mapping['left_wrist'])
                with self.assertRaises(ValueError):
                    controller.jpeg('../camera')

    def test_invalid_alignment_mapping_fails_before_opening_devices(self):
        original = json.loads((dashboard.ROOT / 'config/recording.json').read_text())
        for change in ({'overhead': 'missing'}, {'overhead': original['alignment_assist']['camera_mapping']['left_wrist']}, {'overhead': None}):
            config = json.loads(json.dumps(original))
            config['alignment_assist']['camera_mapping'].update(change)
            with self.subTest(change=change), patch.object(dashboard.Path, 'read_text', return_value=json.dumps(config)), patch.object(dashboard, 'Cameras') as cameras, patch.object(dashboard, 'MotorClient') as motor:
                hw = dashboard.RegraspHardware()
                hw.task = 'alignment'
                with self.assertRaisesRegex(ValueError, 'camera mapping'):
                    hw.open()
                cameras.assert_not_called()
                motor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
