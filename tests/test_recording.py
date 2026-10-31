"""Exercise episode boundaries, classification, recovery and control guards without motors."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from recording import EpisodeStore
from dashboard import Dashboard
import hand_guide
import numpy as np
import urllib.request
import urllib.error


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = json.loads((ROOT/'config/recording.json').read_text())
        self.config['start_delay_seconds'] = 0
        self.config['motion_arms'] = ['left', 'right']
        self.config['neutral_return']['arms'] = ['left', 'right']
        self.config['audio_cues'] = {'enabled': False}
        self.config['minimum_free_bytes'] = 0
        self.config['neutral_return']['enabled'] = False
        self.store = EpisodeStore(self.temp.name, self.config, {'test': True})
        self.t = time.monotonic_ns()
        self.sequence = 0

    def tearDown(self):
        self.store.close('test_finished')
        self.temp.cleanup()

    def sample(self, q=0.0, velocity=0.0, dt=0.05, pedal=None, right_gripper=0.0):
        self.t += int(dt*1e9)
        self.sequence += 1
        sample = dict(monotonic_ns=self.t, unix_ns=time.time_ns(), sequence=self.sequence,
                      arms={side: dict(position_rad=[q]*6, velocity_rad_s=[velocity]*6,
                                       gripper=dict(position=0.0, velocity=0.0))
                            for side in ('left', 'right')})
        sample['arms']['right']['gripper']['position'] = right_gripper
        if pedal is not None:
            sample['foot_pedal'] = pedal
        self.store.submit('sample', sample)
        self.store.events.join()

    def trigger(self):
        self.sample()
        for _ in range(5):
            self.sample(.03, .1)
        self.assertEqual(self.store.snapshot()['phase'], 'recording')
        return self.store.snapshot()['active']

    def decision(self, label, identifier):
        self.store.submit('decision', dict(label=label, episode_id=identifier))
        self.store.events.join()

    def test_left_only_motion_trigger(self):
        self.config['motion_arms'] = ['left']
        for i in range(20):
            self.sample(right_gripper=i * .1)
        self.assertIsNone(self.store.active)
        self.trigger()

    def test_episode_modes_switch_trigger_delay_and_reset_arms(self):
        self.assertEqual(self.store.snapshot()['episode_mode'], 'regular')
        self.sample()
        self.assertTrue(self.store.pre_roll)
        self.store.set_episode_mode('single_bolt')
        self.assertFalse(self.store.pre_roll)
        self.assertIsNone(self.store.baseline)
        self.assertIsNone(self.store.trigger_since)
        self.assertEqual(self.config['neutral_return']['arms'], ['left'])
        self.assertEqual(self.config['start_delay_seconds'], 2)
        for i in range(20):
            self.sample(right_gripper=i * .1)
        self.assertIsNone(self.store.active)
        self.store.set_episode_mode('regular')
        self.assertEqual(self.config['neutral_return']['arms'], ['left', 'right'])
        self.assertEqual(self.config['start_delay_seconds'], 0)
        for i in range(20):
            self.sample(right_gripper=i * .1)
        self.assertIsNotNone(self.store.active)
        self.assertGreater(self.store.active['samples'], 0)

    def test_mode_change_rejects_busy_states_without_failing_writer(self):
        identifier = self.trigger()
        with self.assertRaisesRegex(RuntimeError, 'between takes'):
            self.store.set_episode_mode('single_bolt')
        self.decision('review', identifier)
        self.store.pending_pedal_decision = {'id': identifier}
        with self.assertRaisesRegex(RuntimeError, 'between takes'):
            self.store.set_episode_mode('single_bolt')
        self.store.pending_pedal_decision = None
        self.store.return_pending = True
        self.store.update(phase='returning_to_neutral')
        with self.assertRaisesRegex(RuntimeError, 'between takes'):
            self.store.set_episode_mode('single_bolt')
        self.store.update(phase='holding_neutral', handoff_ready={'left': False, 'right': False})
        self.store.set_episode_mode('single_bolt')
        self.assertTrue(self.store.return_pending)
        self.assertEqual(self.store.snapshot()['handoff_ready'], {'left': False, 'right': False})
        self.assertEqual(self.store.snapshot()['phase'], 'holding_neutral')
        self.assertIsNone(self.store.failure)

    def test_episode_metadata_preserves_mode_config_snapshot(self):
        identifier = self.trigger()
        metadata = self.store.active['metadata']
        self.decision('kept', identifier)
        self.store.set_episode_mode('single_bolt')
        self.assertEqual(metadata['episode_mode'], 'regular')
        self.assertEqual(metadata['config']['neutral_return']['arms'], ['left', 'right'])
        self.assertEqual(metadata['config']['start_delay_seconds'], 0)
        self.assertEqual(metadata['config']['motion_arms'], ['left', 'right'])

    def test_mode_change_queued_after_trigger_is_rejected(self):
        self.sample()
        # Queue enough motion to trigger before the mode request is handled.
        sample = dict(monotonic_ns=self.t + 1_000_000_000,
                      arms={side: dict(position_rad=[.03]*6, velocity_rad_s=[.1]*6,
                                       gripper=dict(position=0, velocity=0))
                            for side in ('left', 'right')})
        self.store.submit('sample', sample)
        self.store.submit('sample', dict(sample, monotonic_ns=self.t + 2_000_000_000))
        with self.assertRaisesRegex(RuntimeError, 'between takes'):
            self.store.set_episode_mode('single_bolt')
        self.assertEqual(self.store.snapshot()['episode_mode'], 'regular')

    def test_http_episode_mode_validation_auth_and_busy(self):
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        server = Dashboard(self.store, None, port, True)
        def post(payload, token=True, origin=None):
            headers = {'Content-Type': 'application/json'}
            if token:
                headers['X-Session-Token'] = server.token
            if origin:
                headers['Origin'] = origin
            request = urllib.request.Request(server.url + '/api/episode-mode',
                data=json.dumps(payload).encode(), headers=headers)
            return urllib.request.urlopen(request)
        try:
            for token, origin in [(False, None), (True, 'http://untrusted.example')]:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post({'mode': 'single_bolt'}, token, origin)
                self.assertEqual(error.exception.code, 403)
            for payload in ({}, {'mode': 'invalid'}, {'mode': []}, {'mode': 'regular', 'extra': True}):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post(payload)
                self.assertEqual(error.exception.code, 409)
            with post({'mode': 'single_bolt'}) as response:
                self.assertEqual(json.load(response)['episode_mode'], 'single_bolt')
            self.trigger()
            with self.assertRaises(urllib.error.HTTPError) as error:
                post({'mode': 'regular'})
            self.assertEqual(error.exception.code, 409)
            self.assertIsNone(self.store.failure)
        finally:
            server.close()

    def test_setup_window_excludes_states_and_images(self):
        self.config['start_delay_seconds'] = 2.0
        identifier = self.trigger()
        cutoff = self.store.active['trigger_ns'] + 2_000_000_000
        self.assertEqual(self.store.active['samples'], 0)
        for stamp in (cutoff - 1, cutoff, cutoff - 100, cutoff + 100):
            self.store.submit('frame', dict(camera='overhead', monotonic_ns=stamp,
                                            unix_ns=time.time_ns(), jpeg=b'fake-jpeg'))
        self.t = cutoff - 50_000_000
        self.sample(q=.04)
        self.store.events.join()
        self.decision('kept', identifier)
        directory = Path(self.temp.name) / 'kept' / identifier
        states = [json.loads(row) for row in (directory / 'states.jsonl').read_text().splitlines()]
        frames = [json.loads(row) for row in (directory / 'frames.jsonl').read_text().splitlines()]
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]['episode_time_s'], 0)
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(row['monotonic_ns'] >= cutoff for row in states + frames))
        self.assertEqual(len(list((directory / 'images/overhead').glob('*.jpg'))), 2)

    def test_setup_only_take_cannot_be_kept(self):
        self.config['start_delay_seconds'] = 2.0
        identifier = self.trigger()
        self.decision('kept', identifier)
        meta = json.loads((Path(self.temp.name) / 'review' / identifier / 'metadata.json').read_text())
        self.assertFalse(meta['complete'])
        self.assertEqual(meta['samples'], 0)

    def test_single_bolt_start_cue_follows_first_retained_state_once_per_episode(self):
        self.store.set_episode_mode('single_bolt')
        self.store.audio.play = Mock()
        for first_sample_offset in (0, 10_000_000):
            with self.subTest(first_sample_offset=first_sample_offset):
                identifier = self.trigger()
                cutoff = self.store.active['trigger_ns'] + 2_000_000_000
                self.assertEqual(self.store.active['samples'], 0)
                self.store.audio.play.assert_not_called()
                # A camera can arrive at the cutoff before the state stream.
                self.store.submit('frame', dict(camera='overhead', monotonic_ns=cutoff,
                                                unix_ns=time.time_ns(), jpeg=b'fake-jpeg'))
                self.store.events.join()
                self.store.audio.play.assert_not_called()
                self.t = cutoff - 1
                self.sample(q=.04, dt=0)
                self.assertEqual(self.store.active['samples'], 0)
                self.store.audio.play.assert_not_called()
                self.t = cutoff + first_sample_offset
                self.sample(q=.04, dt=0)
                self.assertEqual(self.store.active['samples'], 1)
                self.store.audio.play.assert_called_once_with('recording_start')
                self.sample(q=.05)
                self.store.submit('frame', dict(camera='overhead', monotonic_ns=cutoff - 1,
                                                unix_ns=time.time_ns(), jpeg=b'late-jpeg'))
                self.store.events.join()
                self.store.audio.play.assert_called_once_with('recording_start')
                self.decision('kept', identifier)
                for _ in range(20):
                    self.sample()
                self.assertEqual(self.store.snapshot()['phase'], 'armed')
                self.store.audio.play.reset_mock()

    def test_regular_episode_has_no_recording_start_cue(self):
        self.store.audio.play = Mock()
        self.trigger()
        self.assertGreater(self.store.active['samples'], 0)
        self.sample(q=.04, dt=2.1)
        self.store.audio.play.assert_not_called()

    def test_single_bolt_ended_during_setup_has_no_recording_start_cue(self):
        self.store.set_episode_mode('single_bolt')
        self.store.audio.play = Mock()
        identifier = self.trigger()
        self.decision('kept', identifier)
        self.store.audio.play.assert_called_once_with('kept')
        self.assertIsNone(self.store.active)

    def test_jitter_does_not_trigger(self):
        for i in range(40):
            self.sample(.001*(i%3))
        self.assertIsNone(self.store.snapshot()['active'])

    def test_stationary_pedal_does_not_start_recording(self):
        self.sample(pedal=dict(connected=True, physical_pressed=False, events=[]))
        for _ in range(20):
            self.sample(pedal=dict(connected=True, physical_pressed=True,
                                  events=[dict(monotonic_ns=self.t+1, pressed=True)]))
        self.assertIsNone(self.store.snapshot()['active'])
        self.assertEqual(self.store.snapshot()['phase'], 'armed')

    def test_pre_roll_classification_and_rearm(self):
        for label in ('kept', 'review', 'discarded'):
            identifier = self.trigger()
            self.store.submit('frame', dict(camera='overhead', monotonic_ns=self.t,
                                            unix_ns=time.time_ns(), jpeg=b'fake-jpeg'))
            self.store.events.join()
            self.decision(label, identifier)
            directory = Path(self.temp.name)/label/identifier
            meta = json.loads((directory/'metadata.json').read_text())
            states = [json.loads(s) for s in (directory/'states.jsonl').read_text().splitlines()]
            self.assertTrue(meta['complete'])
            self.assertEqual(meta['label'], label)
            self.assertLess(states[0]['monotonic_ns'], meta['trigger_monotonic_ns'])
            self.assertEqual(meta['camera_frames']['overhead'], 1)
            self.assertTrue((directory/'images/overhead/0000000.jpg').exists())
            # Continued motion after labeling must not create an accidental second take.
            for i in range(15):
                self.sample(.03+i*.025, .1)
            self.assertIsNone(self.store.snapshot()['active'])
            for _ in range(20):
                self.sample(0.0, 0.0)
            self.assertEqual(self.store.snapshot()['phase'], 'armed')

    def test_pedal_alone_ends_episode_and_requests_return(self):
        self.store.audio.play = Mock()
        identifier = self.trigger()
        def tap(position):
            press = self.t + 10_000_000
            release = press + 100_000_000
            self.sample(dt=.12, right_gripper=position,
                        pedal=dict(connected=True, physical_pressed=False,
                                   events=[dict(monotonic_ns=press, pressed=True),
                                           dict(monotonic_ns=release, pressed=False)]))
        self.sample(pedal=dict(connected=True, physical_pressed=False, events=[]))
        self.assertFalse(self.store.snapshot()['episode_end_detected'])
        self.assertEqual(self.store.snapshot()['active'], identifier)
        # First PRESS starts the return; later taps still classify during it.
        self.store.config['neutral_return']['enabled'] = True
        press = self.t + 10_000_000
        self.sample(dt=.02, right_gripper=1.7,
                    pedal=dict(connected=True, physical_pressed=True,
                               events=[dict(monotonic_ns=press, pressed=True)]))
        self.assertIsNone(self.store.snapshot()['active'])
        self.assertEqual(self.store.snapshot()['phase'], 'returning_to_neutral')
        self.assertEqual(self.store.control_requests.get_nowait(), dict(episode_id=identifier, label='pending'))
        self.assertTrue(self.store.snapshot()['pedal_decision_pending'])
        self.sample(dt=.1, right_gripper=1.7,
                    pedal=dict(connected=True, physical_pressed=False,
                               events=[dict(monotonic_ns=self.t+90_000_000, pressed=False)]))
        tap(1.7)
        self.sample(dt=.7, right_gripper=1.7,
                    pedal=dict(connected=True, physical_pressed=False, events=[]))
        self.assertIsNone(self.store.snapshot()['active'])
        self.assertEqual(self.store.snapshot()['phase'], 'returning_to_neutral')
        self.assertTrue(self.store.control_requests.empty())
        self.assertFalse(self.store.snapshot()['pedal_decision_pending'])
        directory = Path(self.temp.name)/'review'/identifier
        metadata = json.loads((directory/'metadata.json').read_text())
        self.assertEqual(metadata['end_reason'], 'foot_pedal')
        self.assertEqual(metadata['episode_end_detection']['method'], 'foot_pedal')
        states = [json.loads(row) for row in (directory/'states.jsonl').read_text().splitlines()]
        pedals = [row['foot_pedal'] for row in states if 'foot_pedal' in row]
        self.assertTrue(any(p['physical_pressed'] for p in pedals))
        self.assertTrue(all('downward_force_requested' not in p for p in pedals))
        self.assertTrue((directory/'pedal-decision-events.jsonl').exists())
        self.store.audio.play.assert_any_call('press')
        self.store.audio.play.assert_any_call('review')

    def arm_pedal_classification(self):
        identifier = self.trigger()
        self.sample(pedal=dict(connected=True, physical_pressed=False, events=[]))
        self.assertEqual(self.store.snapshot()['pedal_mode'], 'classification')
        self.assertFalse(self.store.snapshot()['episode_end_detected'])
        return identifier

    def test_gripper_opening_does_not_end_episode(self):
        identifier = self.trigger()
        for _ in range(120):
            self.sample()
        for _ in range(10):
            self.sample(right_gripper=1.7)
        self.assertEqual(self.store.snapshot()['active'], identifier)
        self.assertFalse(self.store.snapshot()['episode_end_detected'])
        self.assertFalse(self.store.snapshot().get('pedal_decision_pending', False))

    def classification_tap(self):
        press = self.t + 10_000_000
        self.sample(dt=.12, right_gripper=1.7,
                    pedal=dict(connected=True, physical_pressed=False,
                               events=[dict(monotonic_ns=press, pressed=True),
                                       dict(monotonic_ns=press+100_000_000, pressed=False)]))

    def test_pedal_keep_and_delete_relocate_the_finished_episode(self):
        for count, label in [(1, 'kept'), (3, 'discarded')]:
            identifier = self.arm_pedal_classification()
            for _ in range(count):
                self.classification_tap()
                self.assertIsNone(self.store.snapshot()['active'])
            self.sample(dt=.7, right_gripper=1.7)
            directory = Path(self.temp.name)/label/identifier
            metadata = json.loads((directory/'metadata.json').read_text())
            self.assertTrue(metadata['complete'])
            self.assertEqual(metadata['label'], label)
            self.assertEqual(self.store.snapshot()['last_decision'], dict(id=identifier, label=label))
            self.assertFalse((Path(self.temp.name)/'review'/identifier).exists())
            for _ in range(20):
                self.sample()

    def test_stop_during_pedal_group_preserves_incomplete_review(self):
        identifier = self.arm_pedal_classification()
        self.classification_tap()
        self.store.close('operator_stop')
        metadata = json.loads((Path(self.temp.name)/'review'/identifier/'metadata.json').read_text())
        self.assertFalse(metadata['complete'])
        self.assertEqual(metadata['end_reason'], 'foot_pedal_pending')

    def test_invalid_pedal_gesture_keeps_return_gate_until_handoff(self):
        self.arm_pedal_classification()
        self.store.config['neutral_return']['enabled'] = True
        press = self.t + 10_000_000
        self.sample(dt=.82, right_gripper=1.7,
                    pedal=dict(connected=True, physical_pressed=False,
                               events=[dict(monotonic_ns=press, pressed=True),
                                       dict(monotonic_ns=press+800_000_000, pressed=False)]))
        self.sample(dt=.7, right_gripper=1.7)
        self.assertFalse(self.store.snapshot()['pedal_decision_pending'])
        self.assertTrue(self.store.return_pending)
        self.assertEqual(self.store.snapshot()['phase'], 'returning_to_neutral')
        for _ in range(10):
            self.sample(.5, .2)
        self.assertIsNone(self.store.snapshot()['active'])
        self.assertIsNone(self.store.failure)

    def test_delayed_pedal_events_resolve_in_the_first_sample(self):
        for duration, label in [(0.1, 'kept'), (0.8, 'review')]:
            with self.subTest(duration=duration):
                identifier = self.arm_pedal_classification()
                press = self.t + 10_000_000
                self.sample(dt=duration+1, right_gripper=1.7,
                            pedal=dict(connected=True, physical_pressed=False,
                                       events=[dict(monotonic_ns=press, pressed=True),
                                               dict(monotonic_ns=press+round(duration*1e9), pressed=False)]))
                self.assertFalse(self.store.snapshot()['pedal_decision_pending'])
                self.assertTrue((Path(self.temp.name)/label/identifier).exists())
                for _ in range(20):
                    self.sample()

    def test_invalid_pedal_gesture_preserves_review_and_unblocks_recording(self):
        for durations in [(0.8,), (0.01,), (0.1,) * 4]:
            with self.subTest(durations=durations):
                identifier = self.arm_pedal_classification()
                self.store.audio.play = Mock()
                for duration in durations:
                    press = self.t + 10_000_000
                    self.sample(dt=duration + .02, right_gripper=1.7,
                                pedal=dict(connected=True, physical_pressed=False,
                                           events=[dict(monotonic_ns=press, pressed=True),
                                                   dict(monotonic_ns=press + round(duration*1e9), pressed=False)]))
                self.sample(dt=.7, right_gripper=1.7)
                self.assertFalse(self.store.snapshot()['pedal_decision_pending'])
                self.assertIsNone(self.store.pending_pedal_decision)
                self.assertEqual(self.store.snapshot()['pedal_mode'], 'idle')
                self.assertIn('review', self.store.snapshot()['pedal_notice'])
                metadata = json.loads((Path(self.temp.name)/'review'/identifier/'metadata.json').read_text())
                self.assertFalse(metadata['complete'])
                self.assertEqual(metadata['end_reason'], 'foot_pedal_invalid_gesture')
                self.store.audio.play.assert_any_call('cancel')
                for _ in range(20):
                    self.sample()
                self.assertEqual(self.store.snapshot()['phase'], 'armed')
                self.trigger()
                self.assertIsNone(self.store.snapshot()['pedal_notice'])
                self.decision('review', self.store.snapshot()['active'])
                for _ in range(20):
                    self.sample()

    def test_stale_decision_cannot_finish_next_episode(self):
        identifier = self.trigger()
        self.decision('kept', 'stale-id')
        self.assertEqual(self.store.snapshot()['active'], identifier)

    def test_unfinished_stop_is_review(self):
        identifier = self.trigger()
        self.store.close('operator_stop')
        meta = json.loads((Path(self.temp.name)/'review'/identifier/'metadata.json').read_text())
        self.assertFalse(meta['complete'])
        self.assertEqual(meta['end_reason'], 'operator_stop')

    def test_crash_recovery(self):
        pending = Path(self.temp.name)/'pending'/'interrupted'
        pending.mkdir()
        (pending/'metadata.json').write_text('{"id":"interrupted"}')
        self.store.close('test')
        self.store = EpisodeStore(self.temp.name, self.config, {})
        meta = json.loads((Path(self.temp.name)/'review/interrupted/metadata.json').read_text())
        self.assertEqual(meta['end_reason'], 'recovered_after_interruption')
        self.assertFalse(meta['complete'])

    def test_http_requires_token_and_matching_episode(self):
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        server = Dashboard(self.store, None, port, True)
        try:
            identifier = self.trigger()
            data = json.dumps(dict(label='kept', episode_id=identifier)).encode()
            request = urllib.request.Request(server.url+'/api/decision', data=data,
                                             headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
            request.add_header('X-Session-Token', server.token)
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 202)
            self.store.events.join()
            self.assertIsNone(self.store.snapshot()['active'])
        finally:
            server.close()


class ControlTests(unittest.TestCase):
    def fake(self):
        states = [SimpleNamespace(position=0.0, velocity=0.0, torque=0.0, time=1.0, fault=0, mode=0)
                  for _ in range(7)]
        motors = [SimpleNamespace(get_current_motor_state=lambda s=s:s, get_motor_id=lambda i=i:i+1)
                  for i, s in enumerate(states)]
        robot = SimpleNamespace(get_current_state=lambda:states[:6],
                                get_current_state_gripper=lambda:states[-1], Motors=motors,
                                joint_limits={'lower':[-1]*6, 'upper':[1]*6},
                                gripper_limits={'lower':0.0, 'upper':2.0},
                                get_Gravity=lambda q:np.full(6, 99.0),
                                forward_kinematics=lambda q:dict(position=[0.0]*3, rotation=np.eye(3)))
        return robot, states

    def test_torque_caps_and_passive_commands(self):
        robot, states = self.fake()
        torque, arm = hand_guide.feedback(robot, 'left', {})
        np.testing.assert_equal(torque, hand_guide.TORQUE_LIMIT)
        self.assertEqual(arm['command']['kp'], [0.0]*6)
        self.assertEqual(arm['command']['kd'], [0.0]*6)
        self.assertEqual(arm['gripper']['command'], [0.0]*5)

    def test_fault_nonfinite_and_limit_rejection(self):
        for field, value in [('fault',1), ('position',2), ('velocity',float('nan')), ('torque',float('nan'))]:
            robot, states = self.fake()
            setattr(states[0], field, value)
            with self.assertRaises(RuntimeError):
                hand_guide.feedback(robot, 'left', {})

    def test_stale_feedback_rejected(self):
        robot, _ = self.fake()
        stale = {('left',i):(1.0,time.monotonic()-1) for i in range(7)}
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            hand_guide.feedback(robot, 'left', stale)

    def test_gripper_placeholder_rejected(self):
        robot, states = self.fake()
        states[-1].position = 999.0
        with self.assertRaisesRegex(RuntimeError, 'gripper position'):
            hand_guide.feedback(robot, 'left', {})

    def test_passive_closed_gripper_encoder_offset_preserved(self):
        robot, states = self.fake()
        states[-1].position = -0.0094
        _, arm = hand_guide.feedback(robot, 'left', {})
        self.assertEqual(arm['gripper']['position'], -0.0094)
        self.assertEqual(arm['gripper']['command'], [0.0]*5)

    def test_stop_attempts_both_arms(self):
        called=[]
        def fail():
            called.append('left')
            raise RuntimeError('disconnected')
        hand_guide.stop_robots(dict(left=SimpleNamespace(set_stop=fail),
                                    right=SimpleNamespace(set_stop=lambda:called.append('right'))))
        self.assertEqual(called, ['left','right'])


if __name__ == '__main__':
    unittest.main()
