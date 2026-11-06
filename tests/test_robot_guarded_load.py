"""Hardware adapter checks using staged command mocks, never devices."""
from pathlib import Path
import sys
import unittest
import time
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from robot_hardware import RobotHardware, GuardedLoadStopped


class GuardedLoadAdapterTests(unittest.TestCase):
    def hardware(self, supported=True):
        hardware = RobotHardware()
        hardware.motor_client = Mock(capabilities={'guarded_cartesian_load': supported}, guard_stop=None)
        hardware.robots = {side: Mock() for side in ('left', 'right')}
        return hardware

    def test_atomic_left_torque_right_hold_and_both_grips(self):
        hardware = self.hardware()
        target = np.arange(14)/10
        assist = np.array([.1, -.2, .1, 0., 0., 0.])
        guard = {'id': 'test-session'}
        hardware.command_guarded_load({}, target, assist, guard)
        hardware.robots['left'].guarded_gravity.assert_called_once_with(assist.tolist(), guard)
        hardware.robots['left'].Joint_Pos_Vel.assert_not_called()
        self.assertEqual(hardware.robots['left'].gripper_control.call_args.args[0], target[6])
        np.testing.assert_array_equal(hardware.robots['right'].Joint_Pos_Vel.call_args.args[0], target[7:13])
        self.assertEqual(hardware.robots['right'].gripper_control.call_args.args[0], target[13])
        hardware.motor_client.flush.assert_called_once_with()

    def test_legacy_backend_never_receives_unprotected_force(self):
        hardware = self.hardware(False)
        with self.assertRaisesRegex(RuntimeError, 'guarded load'):
            hardware.command_guarded_load({}, np.zeros(14), np.zeros(6), {})
        hardware.motor_client.flush.assert_not_called()
        hardware.robots['left'].guarded_gravity.assert_not_called()

    def test_invalid_values_do_not_stage_or_flush(self):
        hardware = self.hardware()
        with self.assertRaises(ValueError):
            hardware.command_guarded_load({}, np.zeros(14), [float('nan')]*6, {})
        hardware.robots['left'].guarded_gravity.assert_not_called()
        hardware.motor_client.flush.assert_not_called()

    def stopped_hardware(self, **changes):
        hardware = self.hardware()
        hardware.motor_client.flush.side_effect = RuntimeError('Command rejected')
        status = dict(error=None, guard_stop={'id': 'active', 'reason': 'travel'},
                      sampled_monotonic_ns=time.monotonic_ns(), arms={
                          side: dict(position_rad=[.1]*6, gripper={'position': .2})
                          for side in ('left', 'right')})
        status.update(changes)
        hardware.motor_client.rpc.return_value = status
        return hardware

    def test_matching_guard_stop_carries_fresh_measured_pose(self):
        hardware = self.stopped_hardware()
        with self.assertRaises(GuardedLoadStopped) as caught:
            hardware.command_guarded_load({}, np.zeros(14), np.zeros(6), {'id': 'active'})
        self.assertEqual(caught.exception.guard_stop['reason'], 'travel')
        np.testing.assert_array_equal(caught.exception.measured_state14, ([.1]*6+[.2])*2)

    def test_stale_future_missing_or_unrelated_stop_remains_fault(self):
        for changes in [dict(sampled_monotonic_ns=time.monotonic_ns()-200_000_000),
                        dict(sampled_monotonic_ns=time.monotonic_ns()+10_000_000_000),
                        dict(sampled_monotonic_ns=None), dict(error='feedback stale'),
                        dict(guard_stop={'id': 'previous'}), dict(arms={})]:
            with self.subTest(changes=changes):
                hardware = self.stopped_hardware(**changes)
                with self.assertRaises(RuntimeError) as caught:
                    hardware.command_guarded_load({}, np.zeros(14), np.zeros(6), {'id': 'active'})
                self.assertNotIsInstance(caught.exception, GuardedLoadStopped)

    def test_transport_error_is_not_recovered_as_guard_stop(self):
        hardware = self.stopped_hardware()
        hardware.motor_client.rpc.side_effect = RuntimeError('Socket lost')
        with self.assertRaisesRegex(RuntimeError, 'Command rejected'):
            hardware.command_guarded_load({}, np.zeros(14), np.zeros(6), {'id': 'active'})


if __name__ == '__main__':
    unittest.main()
