import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from takeover_hold import TakeoverHold


class GripperTakeoverTests(unittest.TestCase):
    def setUp(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'config/recording.json').read_text())
        self.hold = TakeoverHold({s: [0.]*6 for s in ('left', 'right')},
                                 config['neutral_return'], config, 0.)
        self.arms = {s: dict(position_rad=[0.]*6, measured_torque_nm=[0.]*6,
                            gripper=dict(position=1., velocity=0., torque=0.)) for s in ('left', 'right')}
        for tick in (0., .1, .2, .3, .4, .5, .6):
            self.hold.tick(tick, self.arms)

    def test_squeeze_releases_only_its_arm_and_latches(self):
        self.arms['left']['gripper']['torque'] = .2
        self.assertFalse(any(self.hold.tick(.7, self.arms).values()))
        self.assertEqual(self.hold.tick(.85, self.arms), dict(left=True, right=False))
        self.assertEqual(self.hold.diagnostics['left']['trigger'], 'gripper_force_change')
        self.arms['left']['gripper']['position'] = 1.
        self.arms['left']['position_rad'][0] = 1.
        self.assertEqual(self.hold.tick(1., self.arms), dict(left=True, right=False))

    def test_existing_grip_load_is_baselined_and_force_change_releases(self):
        hold = TakeoverHold(self.hold.targets, self.hold.settings, self.hold.motion_settings, 0.)
        for arm in self.arms.values():
            arm['gripper']['torque'] = .7
        for tick in (0., .1, .2, .3, .4, .5, .7, .9):
            self.assertFalse(any(hold.tick(tick, self.arms).values()))
        self.arms['left']['gripper']['torque'] = .9
        self.assertFalse(any(hold.tick(1., self.arms).values()))
        self.assertEqual(hold.tick(1.15, self.arms), dict(left=True, right=False))
        self.assertEqual(hold.diagnostics['left']['trigger'], 'gripper_force_change')
        self.assertAlmostEqual(hold.diagnostics['left']['gripper_torque_baseline_nm'], .7)

    def test_force_noise_and_brief_force_change_do_not_release(self):
        self.arms['left']['gripper']['torque'] = .1
        for tick in (.7, .9, 1.1):
            self.assertFalse(any(self.hold.tick(tick, self.arms).values()))
        self.arms['left']['gripper']['torque'] = -.2
        self.hold.tick(1.2, self.arms)
        self.arms['left']['gripper']['torque'] = 0.
        self.assertFalse(any(self.hold.tick(1.25, self.arms).values()))
        self.assertFalse(any(self.hold.tick(1.5, self.arms).values()))

    def test_force_decrease_can_release_and_threshold_is_configurable(self):
        settings = dict(self.hold.settings, gripper_pull_torque_threshold_nm=.3)
        hold = TakeoverHold(self.hold.targets, settings, self.hold.motion_settings, 0.)
        self.arms['right']['gripper']['torque'] = .7
        for tick in (0., .2, .5):
            hold.tick(tick, self.arms)
        self.arms['right']['gripper']['torque'] = .5
        for tick in (.7, .9):
            self.assertFalse(any(hold.tick(tick, self.arms).values()))
        self.arms['right']['gripper']['torque'] = .3
        hold.tick(1., self.arms)
        self.assertEqual(hold.tick(1.15, self.arms), dict(left=False, right=True))

    def test_invalid_force_feedback_faults(self):
        self.arms['left']['gripper']['torque'] = float('inf')
        with self.assertRaisesRegex(RuntimeError, 'invalid takeover gripper'):
            self.hold.tick(.7, self.arms)

    def test_position_and_velocity_changes_alone_do_not_release(self):
        self.arms['right']['gripper']['position'] = 1.1
        self.arms['left']['gripper']['velocity'] = .1
        for tick in (.7, .85, 1.05):
            self.assertFalse(any(self.hold.tick(tick, self.arms).values()))

    def test_small_noise_and_brief_motion_do_not_release(self):
        self.arms['left']['gripper']['position'] = .99
        for tick in (.7, .9, 1.1):
            self.assertFalse(any(self.hold.tick(tick, self.arms).values()))
        self.arms['left']['gripper']['position'] = .9
        self.hold.tick(1.2, self.arms)
        self.hold.tick(1.25, self.arms)
        self.arms['left']['gripper']['position'] = 1.
        self.assertFalse(any(self.hold.tick(1.26, self.arms).values()))
        self.assertFalse(any(self.hold.tick(1.5, self.arms).values()))

    def test_baseline_settling_does_not_trigger_and_invalid_feedback_faults(self):
        new = TakeoverHold(self.hold.targets, self.hold.settings, self.hold.motion_settings, 0.)
        self.arms['left']['gripper']['position'] = .8
        for tick in (0., .2, .4, .5, .7, .9):
            self.assertFalse(any(new.tick(tick, self.arms).values()))
        self.arms['left']['gripper']['position'] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'invalid takeover gripper'):
            new.tick(1., self.arms)
