"""Replay harness tests: every injected fault starts from positive activation."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_alignment_gates import activate, advance, fault_checks, tick


class GateReplayTests(unittest.TestCase):
    def test_fault_matrix_has_positive_controls_and_specific_reasons(self):
        checks = fault_checks()
        self.assertEqual(len(checks), 30)
        self.assertTrue(all(x['positive_baseline_active'] and x['zero_torque'] for x in checks))
        self.assertEqual({x['task'] for x in checks}, {'gear', 'pin'})

    def test_budget_is_finite_with_fresh_evidence(self):
        for task in ('gear', 'pin'):
            c, s, a, limits, _ = activate(task)
            for stamp, active in ((10.509, True), (10.511, False), (10.7, False)):
                advance(s, stamp)
                result = tick(c, s, a, limits)
                self.assertEqual(result['active'], active)
                if not active:
                    self.assertEqual(result['reason'], 'nudge_budget_exhausted')
                    self.assertEqual(result['torque_nm'], [0.]*6)

    def test_config_cannot_raise_motion_caps(self):
        for task in ('gear', 'pin'):
            c, s, a, limits, _ = activate(task)
            c.config.update(max_delta_rad=10., max_torque_nm=10., spring_nm_per_rad=1000.)
            advance(s, 10.22)
            result = tick(c, s, a, limits)
            self.assertTrue(result['active'])
            self.assertLessEqual(np.linalg.norm(result['delta_rad']), .006000001)
            self.assertLessEqual(max(abs(x) for x in result['torque_nm']), .15)

    def test_named_prediction_views_can_reorder_without_wrong_coordinates(self):
        c, s, a, limits, prior = activate('pin')
        p = s['predictions']['pin']
        order = [0, 2, 1]
        for field in ('camera_names', 'keypoints_2d', 'visibility_probability', 'confidence', 'pixel_std'):
            p[field] = [p[field][i] for i in order]
        advance(s, 10.22)
        result = tick(c, s, a, limits)
        self.assertTrue(result['active'])
        np.testing.assert_allclose(result['delta_rad'], prior['delta_rad'])


if __name__ == '__main__':
    unittest.main()
