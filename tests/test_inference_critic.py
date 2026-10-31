"""Dashboard critic integration remains observational and uses proposal inputs."""
import copy
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import inference_dashboard as dashboard
from test_inference_dashboard import FakeHardware, fake_request, wait_for


class CaptureHardware(FakeHardware):
    def inference_images(self):
        return ({'overhead': 'policy-crop'}, {'overhead': {'sequence': 19}},
                {'overhead': 'raw-full-frame'})


class FakeMonitor:
    instances = []

    def __init__(self, callback=None, **options):
        self.callback = callback
        self.submissions = []
        self.invalidations = 0
        self.closed = False
        self.instances.append(self)

    def submit(self, images, state, actions, fps=20, **identity):
        self.submissions.append(copy.deepcopy(dict(images=images, state=state,
            actions=actions, fps=fps, identity=identity)))
        return len(self.submissions)

    def invalidate(self):
        self.invalidations += 1

    def close(self):
        self.closed = True


class DashboardCriticTests(unittest.TestCase):
    def setUp(self):
        FakeMonitor.instances = []
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.hardware = CaptureHardware()
        self.monitor_patch = patch('action_critic_monitor.ActionCriticMonitor', FakeMonitor)
        self.monitor_patch.start()
        self.addCleanup(self.monitor_patch.stop)
        self.controller = dashboard.Controller('unused', lambda: self.hardware,
            Path(self.directory.name) / 'steps.jsonl', critic_options={})
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.controller.shutdown()
        if self.controller.worker:
            self.controller.worker.join(2)

    def connect(self):
        self.controller.connect()
        wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')

    def test_disabled_constructor_does_not_start_monitor(self):
        controller = dashboard.Controller('unused', critic_options=None)
        self.assertIsNone(controller.critic)
        self.assertEqual(FakeMonitor.instances, [])

    def test_proposal_uses_raw_images_and_request_state_and_does_not_wait_for_score(self):
        requested = []
        def infer(base, path, body=None, timeout=3):
            if path != '/health':
                requested.append(copy.deepcopy(body))
            return fake_request(base, path, body, timeout)
        with patch.object(dashboard, 'request', infer):
            self.connect()
            self.controller.submit('step', 1, 0)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            monitor = FakeMonitor.instances[0]
            self.assertEqual(len(monitor.submissions), 1)
            submitted = monitor.submissions[0]
            self.assertEqual(submitted['images'], {'overhead': 'raw-full-frame'})
            self.assertEqual(submitted['state'], requested[0]['state'])
            self.assertEqual(requested[0]['images'], {'overhead': 'policy-crop'})
            np.testing.assert_allclose(submitted['actions'], self.controller.snapshot()['prediction'])
            self.assertTrue(any(np.any(command) for command in self.hardware.commands))

    def test_invalid_chunk_is_not_scored(self):
        def infer(base, path, body=None, timeout=3):
            result = fake_request(base, path, body, timeout)
            if path != '/health':
                result['actions'][0][0] = float('nan')
            return result
        with patch.object(dashboard, 'request', infer):
            self.connect()
            self.controller.submit('step', 1, 0)
            wait_for(lambda: self.controller.snapshot()['error'] is not None)
            self.assertFalse(any(m.submissions for m in FakeMonitor.instances))
            self.assertFalse(any(np.any(command) for command in self.hardware.commands))

    def test_cleared_and_superseded_results_cannot_replace_current_status(self):
        score = lambda sequence: self.controller.score_prediction(
            {'overhead': 'raw'}, [0.] * 14, [[.01] * 14] * 50, 20, sequence, 1)
        score(1)
        monitor = FakeMonitor.instances[0]
        old_identity = monitor.submissions[-1]['identity']
        self.controller.clear_critic()
        monitor.callback(dict(old_identity, status='ok', score=.9))
        self.assertEqual(self.controller.snapshot()['critic']['status'], 'idle')
        self.assertEqual(monitor.invalidations, 1)
        score(2)
        monitor.callback(dict(old_identity, status='ok', score=.9))
        self.assertEqual(self.controller.snapshot()['critic']['status'], 'scoring')
        current_identity = monitor.submissions[-1]['identity']
        monitor.callback(dict(current_identity, status='ok', score=.2))
        self.assertEqual(self.controller.snapshot()['critic']['score'], .2)

    def test_failed_critic_submission_does_not_block_motion(self):
        with patch.object(dashboard, 'request', fake_request), patch.object(
                FakeMonitor, 'submit', side_effect=RuntimeError('sidecar unavailable')):
            self.connect()
            self.controller.submit('step', 1, 0)
            wait_for(lambda: self.controller.snapshot()['steps_completed'] == 1)
            state = self.controller.snapshot()
            self.assertIsNone(state['error'])
            self.assertEqual(state['critic']['status'], 'error')
            self.assertTrue(any(np.any(command) for command in self.hardware.commands))

    def test_score_history_preserves_rollout_through_stop_and_rejects_stale_results(self):
        def score(sequence, sampled_at, q):
            self.controller.score_prediction({'overhead': 'raw'}, [0.] * 14,
                [[.01] * 14] * 50, 20, sequence, 1)
            monitor = FakeMonitor.instances[0]
            identity = dict(monitor.submissions[-1]['identity'], sampled_at=sampled_at)
            monitor.callback(dict(identity, status='ok', q=q, current_v=.1, delta=q-.1))
            return monitor, identity
        monitor, first = score(1, 100., .2)
        score(2, 101.5, .4)
        history = self.controller.snapshot()['critic_history']
        self.assertEqual([point['q'] for point in history], [.2, .4])
        self.assertEqual([point['t'] for point in history], [0., 1.5])
        self.controller.stop()
        self.assertEqual(self.controller.snapshot()['critic_history'], history)
        monitor.callback(dict(first, status='ok', q=.9, current_v=.1, delta=.8))
        self.assertEqual(self.controller.snapshot()['critic_history'], history)
        self.controller.reset_critic_history()
        self.assertEqual(self.controller.snapshot()['critic_history'], [])
        score(3, 110., .5)
        self.assertEqual(self.controller.snapshot()['critic_history'][0]['t'], 0.)

    def test_history_is_bounded_and_keeps_latest_score_and_sample_count(self):
        self.controller.score_prediction({'overhead': 'raw'}, [0.] * 14,
            [[.01] * 14] * 50, 20, 1, 1)
        monitor = FakeMonitor.instances[0]
        identity = monitor.submissions[-1]['identity']
        for index in range(1205):
            monitor.callback(dict(identity, status='ok', sampled_at=100.+index,
                sequence=index, q=.2, current_v=.1, delta=.1))
        state = self.controller.snapshot()
        self.assertLessEqual(len(state['critic_history']), 1200)
        self.assertEqual(state['critic_samples'], 1205)
        self.assertEqual(state['critic_history'][-1]['sequence'], 1204)
        self.assertEqual(state['critic_history'][-1]['t'], 1204.)

    def test_stop_discards_late_inference_without_scoring(self):
        started, release = threading.Event(), threading.Event()
        def infer(base, path, body=None, timeout=3):
            if path != '/health':
                started.set()
                release.wait(2)
            return fake_request(base, path, body, timeout)
        self.addCleanup(release.set)
        with patch.object(dashboard, 'request', infer):
            self.connect()
            self.controller.submit('step', 1, 0)
            self.assertTrue(started.wait(1))
            self.controller.stop()
            release.set()
            wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.assertFalse(any(m.submissions for m in FakeMonitor.instances))
            self.assertEqual(self.controller.snapshot()['steps_completed'], 0)


if __name__ == '__main__':
    unittest.main()
