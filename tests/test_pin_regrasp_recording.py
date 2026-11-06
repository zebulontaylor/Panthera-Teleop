"""Forward recording checks use generated JPEGs and no robot hardware."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

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

    def test_alignment_records_only_return_and_explicit_metadata(self):
        self.store.begin({'task': 'alignment'})
        self.append_frames()
        result = self.store.finish(True, 'forward_complete')
        path = Path(result['path'])
        metadata = json.loads((path / 'metadata.json').read_text())
        self.assertEqual(metadata['task'], 'alignment')
        self.assertEqual(metadata['phases'], ['return'])
        self.assertTrue(metadata['complete'])
        self.assertEqual(metadata['samples'], 1)
        rows = [json.loads(line) for line in (path / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual([row['phase'] for row in rows], ['return'])
        for phase in ('release', 'retreat', 'close', 'lift'):
            with self.subTest(phase=phase):
                self.store.begin({'task': 'alignment'})
                with self.assertRaises(ValueError):
                    self.append_frames(phase)
                self.assertFalse(self.store.snapshot()['active'])
                self.assertFalse(self.store.snapshot()['complete'])
                self.assertEqual(self.store.snapshot()['samples'], 0)

    def append_frames(self, phase='return', frames=None):
        if frames is None:
            frames = {name: dict(self.camera(), camera=name, jpeg=self.jpeg)
                      for name in ('left_wrist', 'right_wrist', 'overhead')}
        return self.store.append_frames(phase, [0] * 14, [0.1] * 14, frames)

    def test_alignment_three_streams(self):
        mapping = dict(left_wrist='usb-left', right_wrist='usb-right', overhead='usb-top')
        self.store.begin(dict(task='alignment', camera_mapping=mapping))
        expected = []
        for index in range(3):
            frames = {name: dict(self.camera(index + offset), camera=name, jpeg=self.jpeg,
                                 source_camera=mapping[name])
                      for offset, name in enumerate(mapping)}
            expected.append(frames)
            self.append_frames(frames=frames)
        path = Path(self.store.finish(True)['path'])
        meta = json.loads((path / 'metadata.json').read_text())
        self.assertEqual(meta['schema_version'], 2)
        self.assertEqual(meta['camera_mapping'], mapping)
        self.assertEqual(meta['cameras'], list(mapping))
        self.assertNotIn('video', meta)
        rows = [json.loads(line) for line in (path / 'samples.jsonl').read_text().splitlines()]
        for index, row in enumerate(rows):
            self.assertEqual(row['video_frame'], index)
            for name in mapping:
                self.assertEqual(row['cameras'][name], {k: v for k, v in expected[index][name].items()
                                                       if k != 'jpeg'})
        for name in mapping:
            capture = cv2.VideoCapture(str(path / meta['videos'][name]))
            try:
                self.assertTrue(capture.isOpened())
                count = 0
                while capture.read()[0]:
                    count += 1
                self.assertEqual(count, 3)
            finally:
                capture.release()

    def test_alignment_dimensions_are_independent_per_camera(self):
        self.store.begin({'task': 'alignment'})
        frames = {}
        for index, name in enumerate(('left_wrist', 'right_wrist', 'overhead')):
            height, width = 48 + index * 8, 64 + index * 8
            ok, jpeg = cv2.imencode('.jpg', np.full((height, width, 3), 100, dtype=np.uint8))
            self.assertTrue(ok)
            frames[name] = dict(self.camera(index), camera=name, width=width, height=height,
                                jpeg=jpeg.tobytes())
        self.append_frames(frames=frames)
        for receipt in frames.values():
            receipt['monotonic_ns'] = time.monotonic_ns()
        self.append_frames(frames=frames)
        self.assertTrue(self.store.finish(True)['complete'])

    def test_alignment_failure_is_atomic_and_releases_writers(self):
        names = ('left_wrist', 'right_wrist', 'overhead')
        for failure in ('missing', 'stale', 'invalid', 'reverse', 'dimensions', 'identity'):
            for bad_name in names:
                with self.subTest(failure=failure, camera=bad_name):
                    self.store.begin({'task': 'alignment'})
                    self.append_frames()
                    writers = list(self.store._writers.values())
                    frames = {name: dict(self.camera(2), camera=name, jpeg=self.jpeg) for name in names}
                    if failure == 'missing':
                        del frames[bad_name]
                    else:
                        key, value = {'stale': ('monotonic_ns', 1), 'invalid': ('jpeg', b'broken'),
                                      'reverse': ('sequence', 0), 'dimensions': ('width', 999),
                                      'identity': ('camera', 'wrong')}[failure]
                        frames[bad_name][key] = value
                    with self.assertRaises(ValueError):
                        self.append_frames(frames=frames)
                    self.assertFalse(self.store.active)
                    self.assertFalse(self.store.complete)
                    self.assertEqual(self.store.samples, 1)
                    self.assertTrue(all(not writer.isOpened() for writer in writers))
                    self.assertEqual(len((self.store.path / 'samples.jsonl').read_text().splitlines()), 1)
                    for name in names:
                        capture = cv2.VideoCapture(str(self.store.path / (name + '.avi')))
                        try:
                            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
                        finally:
                            capture.release()

    def test_alignment_rejects_single_append(self):
        self.store.begin({'task': 'alignment'})
        with self.assertRaises(ValueError):
            self.append()
        self.assertFalse(self.store.active)
        self.assertEqual(self.store.samples, 0)

    def test_alignment_writer_open_failure_releases_every_opened_writer(self):
        self.store.begin({'task': 'alignment'})
        first, failed = Mock(), Mock()
        first.isOpened.return_value = True
        failed.isOpened.return_value = False
        with patch('pin_regrasp_recording.cv2.VideoWriter', side_effect=[first, failed]):
            with self.assertRaises(RuntimeError):
                self.append_frames()
        first.release.assert_called_once()
        failed.release.assert_called_once()
        first.write.assert_not_called()
        self.assertFalse(self.store.active)
        self.assertEqual(self.store.samples, 0)

    def test_unknown_task_rejected_before_creating_episode(self):
        with self.assertRaises(ValueError):
            self.store.begin({'task': 'unknown'})
        self.assertFalse(self.store.root.exists())
        self.assertFalse(self.store.snapshot()['active'])

    def test_default_task_is_explicit_regrasp(self):
        result = self.store.begin({})
        metadata = json.loads((Path(result['path']) / 'metadata.json').read_text())
        self.assertEqual(metadata['task'], 'regrasp')
        self.assertEqual(metadata['phases'], ['return', 'close'])

    def test_forward_video_and_evidence_remain_unverified(self):
        self.store.begin(dict(task='pin', label='success'))
        self.append()
        self.append('close', self.camera(2))
        result = self.store.finish(True, 'forward_finished')
        self.assertTrue(result['complete'])
        path = Path(result['path'])
        meta = json.loads((path / 'metadata.json').read_text())
        self.assertEqual(meta['label'], 'unverified')
        self.assertEqual(meta['task'], 'regrasp')
        self.assertEqual(meta['schema_version'], 1)
        self.assertEqual(meta['video'], 'left_wrist.avi')
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
