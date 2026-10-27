"""Model-side overlap transport, acknowledgement, expiry and controller integration."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_dashboard as dashboard
import rtc_chunking as rtc
import test_inference_dashboard as fixtures


class ProtocolTests(unittest.TestCase):
    def test_capability_requires_known_protocol_and_explicit_support(self):
        for health in ({}, {'rtc': None}, {'rtc': {'supported': True}},
                       {'rtc': {'supported': False, 'protocol': 'prefix-v1'}}):
            self.assertFalse(rtc.supports_rtc(health))
        self.assertTrue(rtc.supports_rtc({'rtc': {'supported': True, 'protocol': 'prefix-v1'}}))

    def test_tail_alignment_and_limit(self):
        actions = np.arange(50*14).reshape(50, 14)
        body, stamp = rtc.prepare_rtc((actions[3:], 10), 4, 11, 2)
        self.assertEqual(stamp, 10)
        np.testing.assert_array_equal(body['previous_actions'], actions[3:7])
        self.assertEqual(body['overlap'], 4)
        self.assertEqual(rtc.prepare_rtc((actions[49:], 10), 10, 11, 2)[0]['overlap'], 1)

    def test_disabled_expired_or_empty_overlap_has_no_guidance(self):
        for tail, overlap in ((None, 10), ((np.ones((3, 14)), 1), 0),
                              ((np.ones((3, 14)), -5), 10), ((np.empty((0, 14)), 1), 10)):
            self.assertEqual(rtc.prepare_rtc(tail, overlap, 2, 2), (None, None))

    def test_confirmation_fails_closed_on_ignored_mismatched_or_expired_prefix(self):
        body = {'overlap': 4}
        for result in ({}, {'rtc': {'overlap': 3}}, {'rtc': {'overlap': 4.0}}):
            with self.assertRaisesRegex(RuntimeError, 'confirm'):
                rtc.confirm_rtc(result, body, 1, 2, 2)
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            rtc.confirm_rtc({'rtc': body}, body, 1, 4, 2)
        self.assertEqual(rtc.confirm_rtc({'rtc': body}, body, 1, 2, 2), 4)
        self.assertEqual(rtc.confirm_rtc({}, None, None, 4, 2), 0)


class ControllerRTCTests(unittest.TestCase):
    def setUp(self):
        self.hardware = fixtures.FakeHardware()
        self.directory = tempfile.TemporaryDirectory()
        self.controller = dashboard.Controller('unused', lambda: self.hardware,
                                               Path(self.directory.name) / 'steps.jsonl')
        self.requests = []
        self.acknowledge = True

    def tearDown(self):
        self.controller.shutdown()
        if self.controller.worker:
            self.controller.worker.join(3)
        self.directory.cleanup()

    def request(self, base, path, body=None, timeout=3):
        if path == '/health':
            return dict(status='ready', checkpoint='saved', contract=dashboard.CONTRACT,
                        rtc=dict(supported=True, protocol='prefix-v1'))
        self.requests.append(body)
        actions = [[.001*(i+1)+.001*body['sequence']]*14 for i in range(50)]
        result = dict(sequence=body['sequence'], fps=20, checkpoint='saved', mock=False, actions=actions)
        if 'rtc' in body and self.acknowledge:
            result['rtc'] = dict(overlap=body['rtc']['overlap'])
        return result

    def connect(self):
        self.controller.connect()
        fixtures.wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')

    def test_model_guidance_replaces_crossfade_and_pause_discards_prefix(self):
        with patch.object(dashboard, 'request', self.request), \
             patch.object(dashboard, 'blend_predictions', side_effect=AssertionError('Client crossfade called')):
            self.connect()
            self.controller.submit('play', 2, 4)
            fixtures.wait_for(lambda: self.controller.snapshot()['steps_completed'] >= 2)
            self.controller.stop()
            fixtures.wait_for(lambda: self.controller.snapshot()['mode'] == 'holding')
            self.assertNotIn('rtc', self.requests[0])
            self.assertEqual(self.requests[1]['rtc']['overlap'], 4)
            np.testing.assert_allclose(self.requests[1]['rtc']['previous_actions'],
                                       [[.001*(i+1)]*14 for i in range(2, 6)])
            rows = [json.loads(line) for line in self.controller.log_path.read_text().splitlines()]
            guided = next(row for row in rows if row.get('event') == 'play_blend' and row['sequence'] == 1)
            np.testing.assert_allclose(guided['actions'][0], .002)
            self.assertEqual(guided['method'], 'rtc')
            self.assertEqual(guided['blended_actions'], 4)
            before = len(self.requests)
            self.controller.submit('play', 2, 4)
            fixtures.wait_for(lambda: len(self.requests) > before)
            self.assertNotIn('rtc', self.requests[before])
            self.controller.stop()

    def test_ignored_guidance_stops_before_executing_second_chunk(self):
        self.acknowledge = False
        with patch.object(dashboard, 'request', self.request):
            self.connect()
            self.controller.submit('play', 2, 4)
            fixtures.wait_for(lambda: bool(self.controller.snapshot()['error']))
            state = self.controller.snapshot()
            self.assertFalse(state['playing'])
            self.assertEqual(state['steps_completed'], 1)
            self.assertIn('confirm', state['error'])
            self.assertEqual(state['mode'], 'holding')


if __name__ == '__main__':
    unittest.main()
