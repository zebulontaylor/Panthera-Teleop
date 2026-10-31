"""Offline measured-response and reference-association regression checks."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_marked_alignment import analyze, errors


def fixture(reverse_holdout=False):
    rng = np.random.default_rng(873)
    j = rng.normal(size=(4, 6))*100
    rows = []
    for i in range(181):
        q = rng.normal(size=6)*.015
        e = j @ q
        if reverse_holdout and i > 120:
            e = -e
        points = np.full((3, 2, 2), 200.)
        points[1, 0] += e[:2]
        points[2, 0] += e[2:]
        rows.append(dict(episode_id='trial', time_s=i*.1, q_left=q.tolist(), q_right=[0.]*6,
            arms={'left':{'gripper':{'torque':0.}}},
            predictions={'gear':dict(camera_names=['overhead','left_wrist','right_wrist'],
                landmark_names=['gear_center','slot_center'], keypoints_2d=points.tolist(),
                visibility_probability=np.ones((3,2)).tolist(), confidence=np.ones((3,2)).tolist(),
                pixel_std=np.ones((3,2)).tolist())}))
    arms={s:dict(position_rad=[0.]*6) for s in ('left','right')}
    markers=[dict(label='reference', time_s=0., arms=arms),
             dict(label='motion-start', time_s=0.),dict(label='motion-end', time_s=18.),
             dict(label='reference', time_s=18., arms=arms)]
    return rows, markers


class AnalysisTests(unittest.TestCase):
    def test_measured_holdout_reversal_changes_validation_result(self):
        good = analyze(*fixture())
        bad = analyze(*fixture(True))
        self.assertGreater(good['heldout_improvement_fraction'], .99)
        self.assertLess(bad['heldout_improvement_fraction'], 0)
        self.assertFalse(good['verified'])
        self.assertFalse(good['candidate_usable'])

    def test_logical_camera_order_does_not_follow_array_order(self):
        rows, _ = fixture()
        row = rows[0]
        expected = errors(row)
        p = row['predictions']['gear']
        p['camera_names'] = ['right_wrist','overhead','left_wrist']
        p['keypoints_2d'] = [p['keypoints_2d'][i] for i in (2,0,1)]
        np.testing.assert_allclose(errors(row), expected)

    def test_missing_reference_predictions_are_rejected(self):
        rows, markers = fixture()
        with self.assertRaisesRegex(ValueError, 'marked reference'):
            analyze(rows[20:], markers)


if __name__ == '__main__':
    unittest.main()
