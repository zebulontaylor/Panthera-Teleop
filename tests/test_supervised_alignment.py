"""Supervised trials require a fresh explicit press and retain physical bounds."""
import unittest

import numpy as np

from test_alignment_assist import fixture, advance
from alignment_assist import OperatorNudgeLease


def supervised():
    controller, snapshot, arm, limits = fixture()
    controller.config['mode'] = 'operator_supervised'
    controller.config['calibration'].update(verified=False, scope='operator_supervised', measured=True,
                                           held_gripper_position_rad=1.)
    for field in ('visibility_probability', 'confidence'):
        snapshot['predictions']['gear'][field] = np.zeros((3, 2)).tolist()
    arm['gripper']['torque'] = 0.
    return controller, snapshot, arm, limits


class SupervisedTests(unittest.TestCase):
    def evaluate(self, c, s, a, limits, now=10., **kwargs):
        return c.evaluate(now, s, a, limits, other_arm=a, collecting=True, **kwargs)

    def test_human_confirmation_is_required_and_does_not_verify_automatic_mode(self):
        c,s,a,l = supervised()
        self.assertEqual(self.evaluate(c,s,a,l)['reason'], 'hold_button_required')
        c.config['mode'] = 'automatic'
        self.assertEqual(self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='a')['reason'], 'calibration_required')

    def test_operator_trial_can_nudge_with_passive_torque_and_uncalibrated_heads(self):
        c,s,a,l = supervised()
        r = self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='a')
        self.assertTrue(r['active'])
        self.assertLessEqual(np.linalg.norm(r['delta_rad']), .006000001)
        self.assertLessEqual(max(abs(x) for x in r['torque_nm']), .15)
        self.assertAlmostEqual(max(abs(x) for x in r['torque_nm']), .15)

    def test_supervised_config_cannot_raise_torque_or_offset_caps(self):
        c,s,a,l = supervised()
        c.config.update(max_torque_nm=10., max_delta_rad=10.)
        r = self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='a')
        self.assertTrue(r['active'])
        self.assertLessEqual(np.linalg.norm(r['delta_rad']), .006000001)
        self.assertLessEqual(max(abs(x) for x in r['torque_nm']), .15)

    def test_stale_camera_evidence_and_changed_gripper_still_block(self):
        c,s,a,l = supervised()
        self.assertEqual(self.evaluate(c,s,a,l,now=11.,operator_confirmed=True,operator_press='a')['reason'], 'stale_prediction')
        self.assertTrue(self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='b')['active'])
        a['gripper']['position'] = 1.1
        self.assertEqual(self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='b')['reason'], 'gripper_changed')

    def test_operator_can_confirm_a_new_stable_grasp(self):
        c,s,a,l = supervised()
        a['gripper']['position'] = .7
        self.assertTrue(self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='new-grasp')['active'])

    def test_measured_travel_stops_burst_and_does_not_reset_on_sensor_fault(self):
        c,s,a,l = supervised()
        self.assertTrue(self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='a')['active'])
        a['position_rad'][0] = .007
        r = self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='a')
        self.assertEqual(r['reason'], 'measured_travel_limit')
        self.assertEqual(r['torque_nm'], [0.]*6)
        a['position_rad'][0] = 0.
        self.assertEqual(self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='a')['reason'], 'trial_press_exhausted')

    def test_expired_press_cannot_rearm_by_renewing_or_delayed_messages(self):
        lease = OperatorNudgeLease()
        lease.update(10., 'episode', 'press', True)
        self.assertTrue(lease.active(10.1, 'episode'))
        self.assertFalse(lease.active(10.21, 'episode'))
        lease.update(10.25, 'episode', 'press', True)
        self.assertFalse(lease.active(10.26, 'episode'))
        self.assertFalse(lease.active(10.31, 'episode'))
        lease.update(10.4, 'episode', 'press', True)
        self.assertFalse(lease.active(10.4, 'episode'))
        lease.update(10.5, 'episode', 'press', False)
        lease.update(10.6, 'episode', 'press', True)
        self.assertFalse(lease.active(10.6, 'episode'))
        lease.update(10.7, 'episode', 'next', True)
        self.assertTrue(lease.active(10.7, 'episode'))
        self.assertFalse(lease.active(10.7, 'different-episode'))

    def test_release_stops_and_carrier_envelope_blocks(self):
        c,s,a,l = supervised()
        self.assertTrue(self.evaluate(c,s,a,l,operator_confirmed=True,operator_press='a')['active'])
        self.assertEqual(self.evaluate(c,s,a,l)['torque_nm'], [0.]*6)
        other = dict(a, position_rad=[.2]*6)
        r = c.evaluate(10.,s,a,l,other_arm=other,collecting=True,operator_confirmed=True,operator_press='b')
        self.assertEqual(r['reason'], 'other_arm_outside_calibration_envelope')


if __name__ == '__main__':
    unittest.main()
