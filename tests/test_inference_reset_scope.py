"""Inference resets both arms independently of teleop's arm selection."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import inference_dashboard as dashboard
from robot_hardware import RobotHardware


class InferenceResetScopeTests(unittest.TestCase):
    def test_inference_reset_is_bilateral_without_mutating_teleop_settings(self):
        for side in ['left', 'right']:
            with self.subTest(side=side):
                settings = json.loads((dashboard.ROOT / 'config/recording.json').read_text())['neutral_return']
                settings['arms'] = [side]
                original = copy.deepcopy(settings)
                lower, upper = np.full(14, -3.), np.full(14, 3.)
                lower[[6, 13]], upper[[6, 13]] = 0., 2.
                measured = np.array([.1, -.1, .2, -.2, .3, -.3, .4] * 2)
                target = measured.copy()
                reset = dashboard.CollectionReset(settings, lower, upper, measured, 0.)
                self.assertEqual(set(reset.neutral.targets), {'left', 'right'})
                done = False
                for tick in range(1, 600):
                    target, done = reset.tick(tick*.01, .01, measured, target)
                    measured = target.copy()
                    if done:
                        break
                self.assertTrue(done)
                for reset_side, offset in [('left', 0), ('right', 7)]:
                    np.testing.assert_allclose(target[offset:offset+6], settings['position_rad'][reset_side])
                    self.assertAlmostEqual(target[offset+6], 1.2)
                self.assertEqual(settings, original)

    def test_path_validation_only_checks_selected_arm(self):
        for side, offset in [('left', 0), ('right', 7)]:
            with self.subTest(side=side):
                hardware = RobotHardware()
                hardware.motor_client = Mock()
                initial = np.arange(14, dtype=float)
                goal = np.full(6, .25)
                hardware.validate_reset_path(initial, {side: goal})
                hardware.motor_client.validate_path.assert_called_once()
                actual_side, actual_initial, actual_goal = hardware.motor_client.validate_path.call_args.args
                self.assertEqual(actual_side, side)
                np.testing.assert_array_equal(actual_initial, initial[offset:offset+6])
                np.testing.assert_array_equal(actual_goal, goal)


if __name__ == '__main__':
    unittest.main()
