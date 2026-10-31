import copy
import json
from pathlib import Path
import unittest

from app.alignment_q_features import (
    CHECKPOINT_IDS, FEATURE_NAMES, OBSERVATION_DIM, VISUAL_DIM,
    VISUAL_PREFIX_DIM, encode_observation, feature_spec,
)


class AlignmentQFeaturesTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'logs/alignment-calibration/predictions.phases10hz.corrected-order.jsonl'
        with path.open() as stream:
            self.row = json.loads(next(stream))

    def test_cached_snapshot_and_measured_order(self):
        row = self.row
        result = encode_observation(row, row['arms'])
        self.assertEqual(len(result), OBSERVATION_DIM)
        self.assertEqual(len(FEATURE_NAMES), OBSERVATION_DIM)
        self.assertEqual(result[0], row['predictions']['gear']['keypoints_2d'][0][0][0] / 640)
        self.assertEqual(result[1], row['predictions']['gear']['keypoints_2d'][0][0][1] / 480)
        self.assertEqual(result[VISUAL_DIM:VISUAL_PREFIX_DIM], [0.] * (2 * VISUAL_DIM + 1))
        self.assertEqual(result[VISUAL_PREFIX_DIM:VISUAL_PREFIX_DIM + 6], row['arms']['left']['position_rad'])
        self.assertEqual(result[-6:], row['arms']['right']['velocity_rad_s'])
        self.assertEqual(feature_spec()['checkpoint_ids'], CHECKPOINT_IDS)
        self.assertEqual(feature_spec()['names'], list(FEATURE_NAMES))

    def test_uncertain_coordinates_are_preserved(self):
        self.row['predictions']['gear']['visibility_probability'][0][0] = 0
        self.assertNotEqual(encode_observation(self.row, self.row['arms'])[0], 0)

    def test_history_uses_actual_dt(self):
        previous = copy.deepcopy(self.row)
        self.row['predictions']['gear']['keypoints_2d'][0][0][0] += 64
        result = encode_observation(self.row, self.row['arms'], previous, .2)
        self.assertAlmostEqual(result[2 * VISUAL_DIM], .5)
        self.assertEqual(result[VISUAL_PREFIX_DIM - 1], 1)
        for dt in (None, 0, -1, float('nan')):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                encode_observation(self.row, self.row['arms'], previous, dt)

    def test_native_size_identity_and_order(self):
        for key, value in [('image_size', [320, 240]), ('checkpoint_id', 'changed'),
                           ('camera_names', ['overhead', 'right_wrist', 'left_wrist']),
                           ('landmark_names', ['slot_center', 'gear_center'])]:
            row = copy.deepcopy(self.row)
            row['predictions']['gear'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                encode_observation(row, row['arms'])
        self.row['camera_names'].reverse()
        with self.assertRaises(ValueError):
            encode_observation(self.row, self.row['arms'])

    def test_incomplete_and_nonfinite_rejected(self):
        for key in ('keypoints_2d', 'visibility_probability', 'confidence', 'pixel_std'):
            row = copy.deepcopy(self.row)
            if key == 'keypoints_2d':
                row['predictions']['pin'][key][0][0][0] = float('nan')
            else:
                row['predictions']['pin'][key][0][0] = float('inf')
            with self.subTest(key=key), self.assertRaises(ValueError):
                encode_observation(row, row['arms'])
        del self.row['arms']['left']['velocity_rad_s']
        with self.assertRaises(ValueError):
            encode_observation(self.row, self.row['arms'])

    def test_not_ready_rejected(self):
        self.row['status'] = 'loading'
        with self.assertRaises(ValueError):
            encode_observation(self.row, self.row['arms'])


if __name__ == '__main__':
    unittest.main()
