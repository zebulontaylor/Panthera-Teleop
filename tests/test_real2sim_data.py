"""Saved demo sampling tests without robot or simulator dependencies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from real2sim_data import load_episode


class RealDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.metadata = dict(label='kept', complete=True, start_monotonic_ns=1000000000,
                             context=dict(mode='independent_gravity_assist'), calibration='uncalibrated')
        self.rows = [dict(monotonic_ns=1000000000 + t, arms={
            side: dict(position_rad=[i + offset] * 6, gripper=dict(position=i + offset + .5))
            for side, offset in [('left', 0), ('right', 10)]})
            for i, t in enumerate([0, 40000000, 60000000, 100000000, 130000000])]
        self.frames = [dict(camera='overhead', monotonic_ns=1000000000 + t,
                            path=f'images/overhead/{i:07d}.jpg', width=640, height=480,
                            frame_index=i) for i, t in enumerate([0, 100000000])]

    def write(self):
        (self.path / 'metadata.json').write_text(json.dumps(self.metadata))
        for name, rows in [('states', self.rows), ('frames', self.frames)]:
            (self.path / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return load_episode(self.path)

    def test_native_coordinates_next_state_ties_and_no_extrapolation(self):
        episode = self.write()
        np.testing.assert_array_equal(episode.grid_monotonic_ns, [1000000000, 1050000000])
        np.testing.assert_array_equal(episode.state[:, 0], [0, 1])
        np.testing.assert_array_equal(episode.action[:, 0], [1, 3])
        np.testing.assert_array_equal(episode.state[0], [0] * 6 + [.5] + [10] * 6 + [10.5])
        np.testing.assert_allclose(episode.state_skew_s, [0, -.01])
        np.testing.assert_allclose(episode.action_skew_s, [-.01, 0])
        np.testing.assert_allclose(episode.time_s, [0, .05])
        self.assertEqual(episode.metadata['calibration'], 'uncalibrated')

    def test_actual_frame_schema_and_earlier_tie(self):
        episode = self.write()
        frame = episode.nearest_image('overhead', 1050000000)
        self.assertEqual(frame['frame_index'], 0)
        self.assertEqual(frame['skew_s'], -.05)
        self.assertEqual(frame['path'], str(self.path / 'images/overhead/0000000.jpg'))
        self.assertIsNone(episode.nearest_image('overhead', 999999999))
        self.assertIsNone(episode.nearest_image('missing', 1000000000))
        self.assertIsNone(episode.nearest_image('overhead', 1050000000, .01))

    def test_reject_unreviewed_incomplete_and_simulation(self):
        for change in [dict(label='review'), dict(complete=False), dict(context=dict(mode='simulation'))]:
            with self.subTest(change=change):
                saved = self.metadata.copy()
                self.metadata.update(change)
                with self.assertRaises(ValueError):
                    self.write()
                self.metadata = saved
        self.rows[1]['arms']['right']['simulated'] = True
        with self.assertRaises(ValueError):
            self.write()

    def test_reject_bad_joints_gripper_and_timestamp(self):
        for change in ['joints', 'nan', 'gripper', 'timestamp']:
            with self.subTest(change=change):
                old = json.loads(json.dumps(self.rows))
                if change == 'joints':
                    self.rows[0]['arms']['left']['position_rad'] = [0] * 5
                elif change == 'nan':
                    self.rows[0]['arms']['left']['position_rad'][0] = float('nan')
                elif change == 'gripper':
                    self.rows[0]['arms']['right']['gripper']['position'] = float('inf')
                else:
                    self.rows[1]['monotonic_ns'] = self.rows[0]['monotonic_ns']
                with self.assertRaises(ValueError):
                    self.write()
                self.rows = old

    def test_camera_monotonicity_is_per_stream(self):
        self.frames.insert(1, dict(self.frames[0], camera='wrist', monotonic_ns=1000000000))
        self.write()
        self.frames.append(dict(self.frames[0]))
        with self.assertRaises(ValueError):
            self.write()

    def test_invalid_origin_and_context_rejected(self):
        for key, value in [('start_monotonic_ns', -1), ('start_monotonic_ns', 1000000001),
                           ('start_monotonic_ns', True), ('context', None)]:
            with self.subTest(key=key, value=value):
                saved = self.metadata.copy()
                self.metadata[key] = value
                with self.assertRaises(ValueError):
                    self.write()
                self.metadata = saved

    def test_short_episode_has_no_fabricated_action(self):
        self.rows = self.rows[:2]
        with self.assertRaisesRegex(ValueError, 'shorter'):
            self.write()


if __name__ == '__main__':
    unittest.main()
