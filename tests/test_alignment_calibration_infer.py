"""Offline sample contract: supplied camera order and clock rejection."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'alignment_calibration_infer', Path(__file__).resolve().parents[1] / 'scripts/alignment_calibration_infer.py')
infer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer)


class OfflineCalibrationSamplingTests(unittest.TestCase):
    def test_default_order_restores_current_camera_roles(self):
        self.assertEqual(infer.DEFAULT_CAMERA_NAMES, ('overhead', 'wrist_port3', 'wrist_port2'))

    def test_supplied_camera_order_and_measured_state_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            origin = 1_000_000_000
            metadata = {'id': 'fixture', 'label': 'kept', 'complete': True,
                        'start_monotonic_ns': origin, 'end_monotonic_ns': origin + 500_000_000}
            (path / 'metadata.json').write_text(json.dumps(metadata))
            arms = {side: {'position_rad': [value] * 6, 'gripper': {'position': value},
                           'tool_pose_base': {'position_m': [value] * 3}}
                    for side, value in [('left', .1), ('right', .2)]}
            states = [{'monotonic_ns': origin + tick * 500_000_000, 'arms': arms}
                      for tick in range(2)]
            cameras = ['overhead', 'wrist_port3', 'wrist_port2']
            frames = [{'camera': camera, 'monotonic_ns': origin + 2_000_000,
                       'width': 640, 'height': 480, 'path': f'{camera}/0.jpg'} for camera in cameras]
            for name, rows in [('states', states), ('frames', frames)]:
                (path / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            rows = list(infer.episode_samples(path, cameras, 2))
            self.assertEqual(rows[0]['camera_mapping']['left_wrist'], 'wrist_port3')
            self.assertEqual(rows[0]['camera_mapping']['right_wrist'], 'wrist_port2')
            self.assertEqual(rows[0]['q_left'], [.1] * 6)
            self.assertEqual(rows[0]['arms'], arms)
            self.assertAlmostEqual(rows[0]['images']['wrist_port3']['skew_s'], .002)
            self.assertIn('rejection', rows[1])
            phase_rows = list(infer.episode_samples(path, cameras, 2,
                              phases={'gear': {'start_s': 0, 'end_s': .1}}))
            self.assertEqual(len(phase_rows), 1)
            self.assertEqual(phase_rows[0]['phases'], ['gear'])
            metadata['label'] = 'discarded'
            (path/'metadata.json').write_text(json.dumps(metadata))
            original = (path/'metadata.json').read_bytes()
            with self.assertRaises(ValueError):
                list(infer.episode_samples(path, cameras, 2))
            diagnostic = list(infer.episode_samples(path, cameras, 2, allow_non_kept=True))
            self.assertEqual(diagnostic[0]['q_left'], [.1]*6)
            self.assertEqual((path/'metadata.json').read_bytes(), original)
            metadata['complete'] = False
            (path/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                list(infer.episode_samples(path, cameras, 2, allow_non_kept=True))

    def test_stale_frames_rejected_and_ties_choose_earlier(self):
        early, late = {'monotonic_ns': 0}, {'monotonic_ns': 20}
        self.assertIs(infer.nearest([early, late], [0, 20], 10, 1.), early)
        with self.assertRaisesRegex(ValueError, 'skew'):
            infer.nearest([early], [0], 50_000_000, .04)
        with self.assertRaisesRegex(ValueError, 'empty'):
            infer.nearest([], [], 0, .04)


if __name__ == '__main__':
    unittest.main()
