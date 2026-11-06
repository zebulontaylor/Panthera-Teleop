"""Settling must follow carrier pickup and continuous fresh feedback."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from right_arm_settle import RightArmSettleDetector
from inference_dashboard import Controller


class SettleTests(unittest.TestCase):
    def setUp(self):
        self.detector = RightArmSettleDetector()
        self.detector.configure(True)
        self.neutral = np.zeros(14)
        self.neutral[8] = .25
        self.neutral[13] = 1.2
        self.held = self.neutral.copy()
        self.held[7:9] = [.65, 1.4]
        self.held[13] = 1.12
        self.detector.update(0., self.neutral, self.neutral, True)

    def feed(self, start, duration, pose=None, **kwargs):
        pose = self.held if pose is None else pose
        results = []
        for t in np.arange(start, start+duration, .02):
            results.append(self.detector.update(t, pose, pose, **dict(eligible=True, **kwargs)))
        return results

    def test_neutral_and_open_pickup_never_freeze(self):
        self.assertFalse(any(self.feed(.02, 2, self.neutral)))
        opened = self.held.copy(); opened[13] = 1.7
        self.assertFalse(any(self.feed(2.02, 2, opened)))

    def test_full_second_after_movement_not_reported_velocity(self):
        values = self.feed(.02, 1.1)
        self.assertFalse(any(values[:50]))
        self.assertTrue(any(values[50:]))

    def test_slow_drift_is_not_stable(self):
        for i, t in enumerate(np.arange(.02, 3, .02)):
            pose = self.held.copy(); pose[9] = i*.0005
            self.assertFalse(self.detector.update(t, pose, pose, True))

    def test_gap_pause_and_tracking_error_restart_window(self):
        self.feed(.02, .8)
        self.assertFalse(any(self.feed(2., .8)))
        self.detector.update(2.8, self.held, self.held, False)
        self.assertFalse(any(self.feed(2.82, .8)))
        target = self.held.copy(); target[9] += .2
        self.assertFalse(self.detector.update(3.62, self.held, target, True))
        self.assertFalse(any(self.feed(3.64, .8)))

    def test_manual_unfreeze_inhibits_until_rearmed(self):
        controller = Controller('unused')
        self.assertFalse(controller.state['right_arm_auto_freeze']['enabled'])
        controller.configure_right_auto_freeze(True)
        controller.state['mode'] = 'holding'
        controller.request_right_arm_freeze(False)
        self.assertTrue(controller.right_settle.inhibited)
        controller.configure_right_auto_freeze(True)
        self.assertFalse(controller.right_settle.inhibited)
        controller.clear_right_freeze('reset')
        self.assertIsNone(controller.right_settle.origin)
        with self.assertRaises(ValueError): controller.configure_right_auto_freeze(1)

if __name__ == '__main__': unittest.main()
