import io
import base64
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from alignment_vision import AlignmentVision


class FakeProcess:
    def __init__(self, result):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(json.dumps(result) + '\n')
        self.dead = False

    def poll(self):
        return 0 if self.dead else None

    def terminate(self):
        self.dead = True

    def wait(self, timeout=None):
        return 0


class VisionTests(unittest.TestCase):
    def wait_ready(self, vision):
        deadline = time.monotonic() + 2
        while vision.snapshot()['status'] == 'loading' and time.monotonic() < deadline:
            time.sleep(.005)
        return vision.snapshot()

    def frames(self):
        return {name: dict(jpeg=name.encode(), monotonic_ns=timestamp)
                for name, timestamp in [('overhead', 100), ('wrist_port2', 110), ('wrist_port3', 120)]}

    def test_disabled_never_starts_worker_and_missing_frames_fail_closed(self):
        vision = AlignmentVision({})
        self.addCleanup(vision.close)
        with patch.object(vision, '_start') as start:
            self.assertFalse(vision.submit(self.frames(), {}))
            vision.set_enabled(True)
            self.assertFalse(vision.submit({}, {}))
            self.assertEqual(vision.snapshot()['status'], 'error')
            start.assert_not_called()

    def test_preserves_source_time_camera_order_and_real_fk_only(self):
        vision = AlignmentVision({})
        self.addCleanup(vision.close)
        process = FakeProcess(dict(status='ready', monotonic_ns=100, predictions={}))
        with patch.object(vision, '_start', return_value=process):
            vision.set_enabled(True)
            arms = {'left': dict(tool_pose_base=dict(position_m=[.1, .2, .3])),
                    'right': dict(tool_pose_base=dict(position_m=[.4, .5, .6]))}
            self.assertTrue(vision.submit(self.frames(), arms))
            self.assertEqual(self.wait_ready(vision)['monotonic_ns'], 100)
            request = json.loads(process.stdin.getvalue())
            self.assertEqual(request['camera_names'], ['overhead', 'wrist_port3', 'wrist_port2'])
            self.assertEqual([base64.b64decode(frame).decode() for frame in request['jpeg']],
                             ['overhead', 'wrist_port3', 'wrist_port2'])
            self.assertEqual(request['tool_xyz_m'], [.1, .2, .3, .4, .5, .6])
            self.assertFalse(vision.submit(self.frames(), arms))
            snapshot = vision.snapshot()
            snapshot['predictions']['invented'] = True
            self.assertNotIn('invented', vision.snapshot()['predictions'])
            preview = vision.preview()
            self.assertEqual(base64.b64decode(preview['frames']['wrist_port3']), b'wrist_port3')
            self.assertEqual(preview['prediction']['monotonic_ns'], 100)
            vision.set_enabled(False)
            self.assertEqual(vision.snapshot()['status'], 'disabled')
            self.assertEqual(vision.preview()['frames'], {})

    def test_explicit_camera_mapping_overrides_current_setup_default(self):
        mapping = dict(overhead='overhead', left_wrist='wrist_port2', right_wrist='wrist_port3')
        vision = AlignmentVision({'camera_mapping': mapping})
        self.addCleanup(vision.close)
        process = FakeProcess(dict(status='ready', predictions={}))
        with patch.object(vision, '_start', return_value=process):
            vision.set_enabled(True)
            vision.submit(self.frames(), {})
            self.assertEqual(self.wait_ready(vision)['status'], 'ready')
            request = json.loads(process.stdin.getvalue())
            self.assertEqual(request['camera_names'], ['overhead', 'wrist_port2', 'wrist_port3'])
            self.assertEqual([base64.b64decode(frame).decode() for frame in request['jpeg']],
                             ['overhead', 'wrist_port2', 'wrist_port3'])

    def test_recording_config_matches_current_setup_and_requires_operator_trial(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'config/recording.json').read_text())['alignment_assist']
        vision = AlignmentVision(config)
        self.addCleanup(vision.close)
        self.assertEqual(vision.camera_names, ('overhead', 'wrist_port3', 'wrist_port2'))
        self.assertTrue(config['enabled'])
        self.assertEqual(config['mode'], 'operator_supervised')
        self.assertFalse(config['calibration']['verified'])
        calibration = json.loads((Path(__file__).resolve().parents[1]/config['calibration_file']).read_text())
        self.assertFalse(calibration['verified'])
        self.assertEqual(calibration['scope'], 'operator_supervised')

    def test_missing_fk_is_explicit_and_worker_failure_is_visible(self):
        vision = AlignmentVision({})
        self.addCleanup(vision.close)
        process = FakeProcess(dict(status='error', error='dependency missing'))
        with patch.object(vision, '_start', return_value=process):
            vision.set_enabled(True)
            vision.submit(self.frames(), {})
            self.assertEqual(self.wait_ready(vision)['error'], 'dependency missing')
            self.assertIsNone(json.loads(process.stdin.getvalue())['tool_xyz_m'])

    def test_invalid_camera_order_rejected(self):
        with self.assertRaises(ValueError):
            AlignmentVision({'camera_names': ['same', 'same', 'overhead']})


if __name__ == '__main__':
    unittest.main()
