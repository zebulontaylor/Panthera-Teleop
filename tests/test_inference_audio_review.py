"""Pedal review and inference cues through the hardware-free recording path."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from hitl_demo import DemoHardware, demo_request
from inference_dashboard import Controller, LEASE_SECONDS


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for inference review state')
        time.sleep(.005)


class InferenceAudioReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        audio_patch = patch('recording.AudioCues')
        self.audio = audio_patch.start().return_value
        self.addCleanup(audio_patch.stop)
        request_patch = patch('inference_dashboard.request', demo_request)
        request_patch.start()
        self.addCleanup(request_patch.stop)
        self.controller = Controller('demo', DemoHardware,
            log_path=self.root/'steps.jsonl', data_dir=self.root/'raw')
        self.addCleanup(self.shutdown)
        self.controller.connect(False)
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        self.controller.hardware.auto_pull = False
        self.controller.hardware.config['takeover_hold'].update(
            baseline_seconds=.03, pull_confirm_seconds=.04)

    def shutdown(self):
        self.controller.shutdown()
        if self.controller.worker:
            self.controller.worker.join(3)

    def begin(self):
        self.controller.heartbeat()
        self.controller.collection('begin', dict(operator='tester', scene='audio review'))
        wait_for(lambda: self.controller.snapshot()['rollout_active'])
        wait_for(lambda: self.controller.recorder.snapshot()['active'] is not None)

    def finish(self, label='kept', success=True):
        self.controller.collection('finish', dict(label=label, success=success, quality=4))
        wait_for(lambda: not self.controller.snapshot()['rollout_active'])
        self.controller.recorder.events.join()
        self.assertIsNone(self.controller.recorder.failure)

    def metadata(self, label):
        return [json.loads((p/'metadata.json').read_text())
                for p in (self.root/'raw'/label).iterdir()]

    def test_play_pause_intervention_handoff_and_resume_cues(self):
        self.begin()
        self.audio.play.reset_mock()
        self.controller.submit('play', 1)
        wait_for(lambda: self.controller.snapshot()['playing'])
        self.controller.stop()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding'
                 and not self.controller.pause_event.is_set())
        self.controller.collection('intervene')
        wait_for(lambda: self.controller.snapshot()['handoff_baseline_ready'])
        # Merely requesting an intervention does not claim human assistance.
        self.assertFalse(self.controller.recorder.snapshot()['review_recommended'])
        self.controller.hardware.pull_torque['left'] = [.9] + [0.]*5
        wait_for(lambda: self.controller.snapshot()['handoff_ready'].get('left'))
        wait_for(lambda: self.controller.recorder.snapshot()['review_recommended'])
        self.controller.collection('resume')
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
        self.assertEqual(self.audio.play.call_args_list, [call('policy_play'),
            call('policy_pause'), call('intervention_start'), call('handoff'),
            call('intervention_end')])
        self.finish()
        outcome = self.metadata('kept')[0]['outcome']
        self.assertTrue(outcome['success'])
        self.assertTrue(outcome['human_assisted'])
        self.assertFalse(outcome['policy_success'])
        self.audio.play.assert_called_with('kept')
        self.begin()
        self.assertFalse(self.controller.recorder.snapshot()['review_recommended'])
        self.assertEqual(self.controller.recorder.snapshot()['human_interventions'], 0)
        self.finish()
        clean = [m for m in self.metadata('kept') if not m['outcome']['human_assisted']]
        self.assertEqual(len(clean), 1)
        self.assertTrue(clean[0]['outcome']['policy_success'])

    def test_review_pedal_preserves_context_and_queues_finish_before_reset(self):
        self.begin()
        milestones = dict(gear_pickup='success', gear_seating='failure')
        self.controller.heartbeat(dict(quality=2, notes='needs another attempt', milestones=milestones))
        reset = Mock()
        reset.tick.side_effect = lambda now, dt, measured, target: (target, False)
        order = []
        submit = self.controller.recorder.submit

        def record(kind, payload):
            if kind == 'rollout_end':
                order.append('finish')
            return submit(kind, payload)

        def start_reset(hardware, measured):
            self.assertFalse(self.controller.snapshot()['rollout_active'])
            self.assertEqual(len(self.metadata('review')), 1)
            order.append('reset')
            self.controller.update(mode='resetting')
            return reset

        saved_at_cue = []
        def cue(name):
            if name == 'review':
                saved_at_cue.extend(self.metadata('review'))

        self.audio.play.side_effect = cue
        with patch.object(self.controller.recorder, 'submit', side_effect=record), \
             patch.object(self.controller, 'collection_reset', side_effect=start_reset):
            self.controller.pedal_gesture('review')
            wait_for(lambda: self.controller.snapshot()['mode'] == 'resetting')
            self.controller.recorder.events.join()
            self.assertEqual(order, ['finish', 'reset'])
            self.assertEqual(len(saved_at_cue), 1)
            outcome = saved_at_cue[0]['outcome']
            self.assertEqual(outcome['label'], 'review')
            self.assertFalse(outcome['success'])
            self.assertEqual(outcome['quality'], 2)
            self.assertEqual(outcome['notes'], 'needs another attempt')
            self.assertEqual(outcome['milestones'], milestones)
            self.assertEqual(outcome['reason'], 'foot_pedal_review')
            self.assertTrue(outcome['reset_after_finish'])
            self.audio.play.assert_called_with('review')
            self.controller.stop()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')

    def test_review_pedal_guards_and_expo_approval_routing(self):
        # Test routing directly so EXPO approval does not create a second writer.
        with patch.object(self.controller, 'collection') as collection:
            self.controller.heartbeat()
            with self.assertRaisesRegex(RuntimeError, 'No active rollout'):
                self.controller.pedal_gesture('review')
            collection.assert_not_called()
            self.controller.update(rollout_active=True)
            self.controller.heartbeat_at = time.monotonic() - LEASE_SECONDS - 1
            self.controller.pedal_gesture('review')
            collection.assert_not_called()
            self.controller.heartbeat()
            self.controller.update(rollout_active=True)
            self.controller.update(collection_pending=True)
            self.controller.pedal_gesture('review')
            collection.assert_not_called()
            self.controller.update(collection_pending=False)
            self.controller.pause_event.set()
            self.controller.pedal_gesture('review')
            collection.assert_not_called()
            self.controller.pause_event.clear()
            self.controller.expo_record_dir = self.root/'expo'
            self.controller.pedal_gesture('review')
            args = collection.call_args.args
            self.assertEqual(args[0], 'finish')
            self.assertTrue(args[1]['review_approved'])
            self.assertFalse(args[1]['success'])
            self.assertTrue(args[1]['reset_after_finish'])
            self.controller.update(rollout_active=False)
            self.controller.expo_record_dir = None

    def test_review_waits_for_save_and_stop_cancels_pending_reset(self):
        self.begin()
        entered = threading.Event()
        release = threading.Event()
        handle = self.controller.recorder.handle

        def delayed_handle(kind, payload):
            if kind == 'rollout_end':
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('Test did not release recording writer')
            return handle(kind, payload)

        with patch.object(self.controller.recorder, 'handle', side_effect=delayed_handle), \
             patch.object(self.controller, 'collection_reset') as reset:
            try:
                self.controller.pedal_gesture('review')
                self.assertTrue(entered.wait(2))
                wait_for(lambda: self.controller.snapshot()['mode'] == 'saving')
                reset.assert_not_called()
                self.assertEqual(list((self.root/'raw/review').iterdir()), [])
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding'
                         and not self.controller.pause_event.is_set())
                release.set()
                self.controller.recorder.events.join()
                self.assertEqual(len(self.metadata('review')), 1)
                # Give the motor worker several ticks to observe the saved decision.
                time.sleep(.04)
                reset.assert_not_called()
                self.assertEqual(self.controller.snapshot()['mode'], 'holding')
                self.assertFalse(self.controller.snapshot()['rollout_active'])
            finally:
                release.set()

    def test_review_starts_reset_when_delayed_save_completes(self):
        self.begin()
        entered = threading.Event()
        release = threading.Event()
        handle = self.controller.recorder.handle
        reset = Mock()
        reset.tick.side_effect = lambda now, dt, measured, target: (target, False)

        def delayed_handle(kind, payload):
            if kind == 'rollout_end':
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('Test did not release recording writer')
            return handle(kind, payload)

        def start_reset(hardware, measured):
            self.assertEqual(len(self.metadata('review')), 1)
            self.controller.update(mode='resetting')
            return reset

        with patch.object(self.controller.recorder, 'handle', side_effect=delayed_handle), \
             patch.object(self.controller, 'collection_reset', side_effect=start_reset) as start:
            try:
                self.controller.pedal_gesture('review')
                self.assertTrue(entered.wait(2))
                wait_for(lambda: self.controller.snapshot()['mode'] == 'saving')
                start.assert_not_called()
                release.set()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'resetting')
                start.assert_called_once()
                self.assertFalse(self.controller.snapshot()['rollout_active'])
                self.controller.stop()
                wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            finally:
                release.set()


if __name__ == '__main__':
    unittest.main()
