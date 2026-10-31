"""Forward recording checks use generated JPEGs and no robot hardware."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from pin_regrasp_recording import PinRegraspEpisodeStore


class PinRegraspRecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = PinRegraspEpisodeStore(Path(self.directory.name) / 'pin-regrasp-forward')
        self.addCleanup(self.store.finish)
        ok, encoded = cv2.imencode('.jpg', np.full((48, 64, 3), 130, dtype=np.uint8))
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    def camera(self, sequence=1):
        return dict(camera='left_wrist', sequence=sequence, monotonic_ns=time.monotonic_ns(),
                    unix_ns=time.time_ns(), width=64, height=48)

    def append(self, phase='return', camera=None, jpeg=None):
        return self.store.append(phase, [0] * 14, [0.1] * 14,
                                 self.jpeg if jpeg is None else jpeg,
                                 self.camera() if camera is None else camera)

    def test_forward_video_and_evidence_remain_unverified(self):
        self.store.begin(dict(task='pin', label='success'))
        self.append()
        self.append('close', self.camera(2))
        result = self.store.finish(True, 'forward_finished')
        self.assertTrue(result['complete'])
        path = Path(result['path'])
        meta = json.loads((path / 'metadata.json').read_text())
        self.assertEqual(meta['label'], 'unverified')
        rows = [json.loads(line) for line in (path / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual([r['phase'] for r in rows], ['return', 'close'])
        self.assertEqual(rows[1]['camera']['sequence'], 2)
        self.assertEqual(rows[0]['action'], [0.1] * 14)
        self.assertEqual(rows[0]['state'], [0.0] * 14)
        capture = cv2.VideoCapture(str(path / 'left_wrist.avi'))
        try:
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            self.assertTrue(capture.read()[0])
        finally:
            capture.release()

    def test_pause_stops_episode_and_restart_creates_new_directory(self):
        first = self.store.begin({})['path']
        self.append()
        self.assertFalse(self.store.finish(False, 'manual_override')['complete'])
        with self.assertRaises(RuntimeError):
            self.append()
        second = self.store.begin({})['path']
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.snapshot()['samples'], 0)
        self.assertEqual(json.loads((Path(first) / 'metadata.json').read_text())['reason'], 'manual_override')

    def test_forbidden_phase_and_camera_fail_closed(self):
        for phase, camera in [('human', self.camera()), ('return', {}),
                              ('return', dict(self.camera(), camera='right_wrist')),
                              ('return', dict(self.camera(), monotonic_ns=1))]:
            with self.subTest(phase=phase, camera=camera):
                self.store.begin({})
                with self.assertRaises(ValueError):
                    self.append(phase, camera)
                self.assertFalse(self.store.snapshot()['active'])
                self.assertFalse(self.store.snapshot()['complete'])
                self.assertEqual(self.store.snapshot()['samples'], 0)

    def test_bad_jpeg_and_bad_state_fail_closed(self):
        self.store.begin({})
        with self.assertRaises(ValueError):
            self.append(jpeg=b'broken')
        self.store.begin({})
        with self.assertRaises(ValueError):
            self.store.append('return', [float('nan')] * 14, [0] * 14, self.jpeg, self.camera())
        self.assertFalse(self.store.snapshot()['active'])

    def test_reversed_frame_sequence_fails_closed(self):
        self.store.begin({})
        self.append(camera=self.camera(2))
        with self.assertRaises(ValueError):
            self.append(camera=self.camera(1))
        self.assertEqual(self.store.snapshot()['samples'], 1)
        self.assertFalse(self.store.snapshot()['complete'])


if __name__ == '__main__':
    unittest.main()
