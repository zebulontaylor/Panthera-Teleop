"""Closed-loop tests use measured simulated poses, never robot devices."""
import unittest
from unittest.mock import patch
import numpy as np
from test_cartesian_nudge import fixture, LinearKinematics
from test_alignment_assist import advance
from incremental_nudge import (IncrementalNudge, incremental_proposal, sdk_position_ticks,
                               sdk_grid_target, SDK_POSITION_QUANTUM, quantized_proposal, _quantized_candidate, nudge_bounds, evaluate_incremental_evidence)


class IncrementalNudgeTests(unittest.TestCase):
    def make(self):
        config, self.snapshot, self.arms, self.limits, self.matrix = fixture()
        self.controller = IncrementalNudge(config, 10., self.arms, self.limits,
                                           [5]*6, kinematics=LinearKinematics())
        return self.controller

    def run_loop(self, gain=1., initial=10., iterations=600, follow=True):
        targets = []
        for i in range(iterations):
            now = 10+i*.05
            advance(self.snapshot, now)
            q = np.asarray(self.arms['left']['position_rad'])
            points = self.snapshot['predictions']['gear']['keypoints_2d']
            points[1][0][1] = points[2][0][1] = 100+initial+gain*10000*q[2]
            target = self.controller.tick(now, self.snapshot, self.arms, allowed=True)
            targets.append(q.copy() if target is None else target.copy())
            if follow and target is not None:
                self.arms['left']['position_rad'] = target.tolist()
                self.arms['left']['tool_pose_base']['position_m'] = target[:3].tolist()
            if not self.controller.running:
                break
        return np.array(targets)

    def test_converges_with_imperfect_scale(self):
        for gain in (.5, 1., 2.):
            with self.subTest(gain=gain):
                c = self.make()
                targets = self.run_loop(gain)
                self.assertEqual(c.reason, 'aligned_holding')
                self.assertGreater(c.steps, 1)
                self.assertLessEqual(max(abs(np.array(c.error_px)[[1, 3]])), 3)
                self.assertLessEqual(np.max(np.abs(np.diff(targets[:, 2]))), .0002500001)
                self.assertLessEqual(c.path_m, .002)

    def test_one_aligned_view_and_unequal_magnitude(self):
        self.make()
        result = incremental_proposal(np.zeros(6), [0, 10, 0, 1], self.matrix,
                                      LinearKinematics(), self.limits)
        self.assertLess(result['dz_m'], 0)
        self.assertLessEqual(abs(result['predicted_height_error_px'][1]), 3)
        result = incremental_proposal(np.zeros(6), [0, 20, 0, 4], self.matrix,
                                      LinearKinematics(), self.limits)
        self.assertLess(result['dz_m'], 0)

    def test_conflicting_views_and_deadband_edge_block(self):
        self.make()
        for error, reason in (([0, 10, 0, -10], 'direction_disagreement'),
                              ([0, 10, 0, -3], 'step_too_small')):
            with self.assertRaisesRegex(ValueError, reason):
                incremental_proposal(np.zeros(6), error, self.matrix,
                                     LinearKinematics(), self.limits)

    def test_large_overshoot_and_wrong_direction_stop(self):
        for gain, reason in ((10., 'image_error_worsened'),
                             (-1., 'image_response_wrong_direction'),
                             (0., 'image_response_no_progress')):
            with self.subTest(gain=gain):
                c = self.make()
                self.run_loop(gain)
                self.assertEqual(c.reason, reason)
                self.assertLessEqual(c.steps, 2)

    def test_hold_does_not_restart(self):
        c = self.make()
        self.run_loop(initial=2.)
        self.assertEqual(c.reason, 'aligned_holding')
        self.run_loop(initial=12.)
        self.assertEqual(c.steps, 0)

    def test_cumulative_path_limit(self):
        c = self.make()
        self.run_loop(gain=.1)
        self.assertEqual(c.reason, 'cumulative_travel_limit')
        self.assertLessEqual(c.commanded_path_m, .002)

    def test_step_confirmation_uses_step_origin(self):
        c = self.make()
        # Complete a first step, then leave the second one unexecuted.
        for i in range(200):
            now = 10+i*.05
            advance(self.snapshot, now)
            q = np.asarray(self.arms['left']['position_rad'])
            points = self.snapshot['predictions']['gear']['keypoints_2d']
            points[1][0][1] = points[2][0][1] = 110+10000*q[2]
            target = c.tick(now, self.snapshot, self.arms, allowed=True)
            if c.steps == 1:
                self.arms['left']['position_rad'] = target.tolist()
                self.arms['left']['tool_pose_base']['position_m'] = target[:3].tolist()
            if not c.running:
                break
        self.assertEqual(c.steps, 2)
        self.assertEqual(c.reason, 'physical_step_not_confirmed')

    def test_moving_near_target_cannot_issue_next_step(self):
        c = self.make()
        self.run_loop(iterations=8)
        self.assertEqual(c.steps, 1)
        self.arms['left']['velocity_rad_s'] = [0., 0., .003, 0., 0., 0.]
        self.run_loop(iterations=30)
        self.assertEqual(c.steps, 1)
        self.assertEqual(len(c.samples), 0)

    def test_repeated_camera_frames_do_not_count_as_fresh(self):
        c = self.make()
        for i in range(650):
            now = 10+i*.05
            self.snapshot['monotonic_ns'] = int(now*1e9)
            c.tick(now, self.snapshot, self.arms, allowed=True)
        self.assertEqual(c.steps, 0)
        self.assertEqual(c.reason, 'trial_timeout')

    def test_original_calibration_envelope_cannot_be_bypassed(self):
        c = self.make()
        c.config['calibration']['reference_joint_position_rad'] = [1.]*6
        self.run_loop()
        self.assertEqual(c.reason, 'outside_calibration_envelope')
        self.assertEqual(c.steps, 0)

    def test_stationary_holding_jitter_does_not_exhaust_path(self):
        c = self.make()
        self.run_loop(initial=2.)
        for i in range(4500):
            q = np.zeros(6)
            q[2] = .00001 if i % 2 else -.00001
            self.arms['left']['position_rad'] = q.tolist()
            self.arms['left']['tool_pose_base']['position_m'] = q[:3].tolist()
            c.tick(11.+i/150., self.snapshot, self.arms, allowed=True)
        self.assertEqual(c.reason, 'aligned_holding')
        self.assertEqual(c.path_m, 0.)

    def test_measured_step_excursion_blocks_overshoot(self):
        c = self.make()
        self.run_loop(iterations=8)
        q = np.asarray(self.arms['left']['position_rad'])
        q[2] = -.0005
        self.arms['left']['position_rad'] = q.tolist()
        self.arms['left']['tool_pose_base']['position_m'] = q[:3].tolist()
        with self.assertRaisesRegex(ValueError, 'measured_step_excursion_limit'):
            c.tick(11., self.snapshot, self.arms, allowed=True)

    def test_buffered_evidence_expires_during_camera_stall(self):
        c = self.make()
        for now in (10., 10.2, 10.25):
            advance(self.snapshot, now)
            c.tick(now, self.snapshot, self.arms, allowed=True)
        self.assertEqual(len(c.samples), 2)
        advance(self.snapshot, 11.)
        points = self.snapshot['predictions']['gear']['keypoints_2d']
        points[1][0][1] = points[2][0][1] = 90
        c.tick(11., self.snapshot, self.arms, allowed=True)
        self.assertEqual(c.steps, 0)
        self.assertEqual(len(c.samples), 1)

    def test_startup_never_commands_position_before_evidence(self):
        c = self.make()
        for now in (10., 10.2, 10.25):
            advance(self.snapshot, now)
            self.assertIsNone(c.tick(now, self.snapshot, self.arms, allowed=True))
        self.assertFalse(c.status()['motor_command_active'])
        c.config['calibration']['reference_joint_position_rad'] = [1.]*6
        advance(self.snapshot, 10.3)
        self.assertIsNone(c.tick(10.3, self.snapshot, self.arms, allowed=True))
        self.assertEqual(c.steps, 0)

    def test_sdk_hold_roundtrip_preserves_every_tick(self):
        # Includes recorded feedback values that lose a tick if resent directly.
        ticks = np.arange(-32000, 32001)
        desired = ticks*SDK_POSITION_QUANTUM
        decoded, command = sdk_grid_target(desired)
        np.testing.assert_array_equal(sdk_position_ticks(command), ticks)
        np.testing.assert_array_equal(np.rint(decoded/SDK_POSITION_QUANTUM), ticks)
        self.assertTrue(np.any(sdk_position_ticks(decoded) != ticks))

    def test_quantized_closed_loop_uses_decoded_endpoint(self):
        class ScaledKinematics(LinearKinematics):
            def pose(self, q): return np.asarray(q[:3])*.1, np.eye(3)
            def jacobian(self, q): return np.diag([.1, .1, .1, 1., 1., 1.])
        config, self.snapshot, self.arms, self.limits, matrix = fixture()
        matrix[1, 2] = matrix[3, 2] = 1000.
        config['calibration']['image_jacobian_px_per_rad'] = matrix.tolist()
        config['encoder_resolution_rad'] = SDK_POSITION_QUANTUM
        c = IncrementalNudge(config, 10., self.arms, self.limits, [5]*6,
                             kinematics=ScaledKinematics())
        for i in range(100):
            now = 10.+i*.05
            advance(self.snapshot, now)
            q = np.asarray(self.arms['left']['position_rad'])
            points = self.snapshot['predictions']['gear']['keypoints_2d']
            points[1][0][1] = points[2][0][1] = 105+1000*q[2]
            command = c.tick(now, self.snapshot, self.arms, allowed=True)
            if command is not None:
                decoded = sdk_grid_target(command)[0]
                np.testing.assert_array_equal(sdk_position_ticks(command),
                                               np.rint(c.target/SDK_POSITION_QUANTUM))
                self.arms['left']['position_rad'] = decoded.tolist()
                self.arms['left']['tool_pose_base']['position_m'] = (decoded[:3]*.1).tolist()
            if not c.running:
                break
        self.assertEqual(c.reason, 'aligned_holding')
        self.assertGreater(c.steps, 0)
        self.assertEqual(c.status()['max_step_m'], .0005)

    def test_quantized_predicted_overshoot_is_rejected(self):
        class ScaledKinematics(LinearKinematics):
            def pose(self, q): return np.asarray(q[:3])*.1, np.eye(3)
        self.make()
        proposal = dict(target=np.array([0., 0., -2*SDK_POSITION_QUANTUM, 0., 0., 0.]),
                        dz_m=-.00012)
        with self.assertRaisesRegex(ValueError, 'motor_resolution_limited'):
            _quantized_candidate(proposal, np.zeros(6), [0., 4., 0., 4.],
                               self.matrix, ScaledKinematics(), self.limits)

    def test_lateral_component_counts_toward_next_step_budget(self):
        proposal = dict(target=np.array([.00008, 0., -.00008, 0., 0., 0.]),
                        dz_m=-.00008)
        for measured in (False, True):
            with self.subTest(measured=measured):
                c = self.make()
                if measured:
                    c.path_m = .0019
                else:
                    c.commanded_path_m = .0019
                with patch('incremental_nudge.incremental_proposal', return_value=proposal):
                    self.run_loop(iterations=10)
                self.assertEqual(c.reason, 'cumulative_travel_limit')
                self.assertEqual(c.steps, 0)

    def test_lattice_search_respects_joint_cap_after_rounding(self):
        class ScaledKinematics(LinearKinematics):
            def pose(self, q): return np.asarray(q[:3])*.1, np.eye(3)
        self.make()
        self.matrix[1, 2] = self.matrix[3, 2] = 1000.
        proposal = dict(target=np.array([0., 0., -.003, 0., 0., 0.]), dz_m=-.0003)
        self.assertGreater(np.linalg.norm(sdk_grid_target(proposal['target'])[0]), .003)
        accepted = quantized_proposal(proposal, np.zeros(6), [0., 10., 0., 10.],
                                     self.matrix, ScaledKinematics(), self.limits)
        self.assertLessEqual(np.linalg.norm(accepted['target']), .003)
        self.assertLess(accepted['dz_m'], 0)

    def test_stable_encoder_counts_ignore_contradictory_velocity(self):
        c = self.make()
        c.encoder_resolution = SDK_POSITION_QUANTUM
        self.arms['left']['velocity_rad_s'] = [.07, -.03, .05, -.07, .03, -.05]
        self.run_loop(initial=2.)
        self.assertEqual(c.reason, 'aligned_holding')
        self.assertEqual(c.steps, 0)

    def test_changed_encoder_count_resets_visual_dwell(self):
        c = self.make()
        c.encoder_resolution = SDK_POSITION_QUANTUM
        self.arms['left']['velocity_rad_s'] = [.07]*6
        advance(self.snapshot, 10.)
        c.tick(10., self.snapshot, self.arms, allowed=True)
        advance(self.snapshot, 10.2)
        c.tick(10.2, self.snapshot, self.arms, allowed=True)
        self.assertEqual(len(c.samples), 1)
        self.arms['left']['position_rad'][2] = SDK_POSITION_QUANTUM
        self.arms['left']['tool_pose_base']['position_m'][2] = SDK_POSITION_QUANTUM
        advance(self.snapshot, 10.25)
        c.tick(10.25, self.snapshot, self.arms, allowed=True)
        self.assertEqual(len(c.samples), 0)
        self.assertEqual(c.status()['wait_stage'], 'waiting_for_stable_encoders')
        self.assertEqual(c.steps, 0)

    def test_fresh_arrivals_with_latency_can_complete_three_sample_window(self):
        c = self.make()
        points = self.snapshot['predictions']['gear']['keypoints_2d']
        points[1][0][1] = points[2][0][1] = 102
        advance(self.snapshot, 10.)
        c.tick(10., self.snapshot, self.arms, allowed=True)
        for now in (10.35, 10.45, 10.55):
            advance(self.snapshot, now-.15)
            c.tick(now, self.snapshot, self.arms, allowed=True)
        self.assertEqual(c.reason, 'aligned_holding')
        self.assertEqual(c.status()['visual_sample_count'], 3)

    def test_nonfinite_velocity_still_blocks_hardware_observation(self):
        c = self.make()
        c.encoder_resolution = SDK_POSITION_QUANTUM
        self.arms['left']['velocity_rad_s'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid_feedback'):
            c.tick(10., self.snapshot, self.arms, allowed=True)

    def test_relaxed_profile_is_explicit_and_finite(self):
        c = self.make()
        self.assertEqual(c.status()['incremental_profile'], 'strict')
        self.assertEqual(c.status()['max_step_m'], .00025)
        self.assertEqual(nudge_bounds('supervised_relaxed')['travel_m'], .05)
        with self.assertRaisesRegex(ValueError, 'unknown_incremental_profile'):
            nudge_bounds('unlimited')

    def test_relaxed_trial_converges_beyond_strict_travel_budget(self):
        config, self.snapshot, self.arms, self.limits, _ = fixture()
        config['incremental_profile'] = 'supervised_relaxed'
        self.controller = IncrementalNudge(config, 10., self.arms, self.limits,
                                           [5]*6, kinematics=LinearKinematics())
        self.run_loop(initial=50.)
        c = self.controller
        self.assertEqual(c.reason, 'aligned_holding')
        self.assertGreater(c.path_m, .002)
        self.assertLess(c.path_m, .01)
        self.assertEqual(c.status()['max_step_m'], .005)
        self.assertIsNone(c.status()['step_norm_limit_rad'])
        self.assertIsNone(c.status()['horizontal_drift_limit_m'])
        self.assertIsNone(c.status()['orientation_limit_rad'])
        self.assertIsNone(c.status()['max_steps'])
        self.assertIsNone(c.status()['max_seconds'])

    def test_relaxed_quantized_step_keeps_explicit_limits(self):
        self.make()
        proposal = incremental_proposal(np.zeros(6), [0., 50., 0., 50.], self.matrix,
                         LinearKinematics(), self.limits, profile='supervised_relaxed')
        accepted = quantized_proposal(proposal, np.zeros(6), [0., 50., 0., 50.],
                         self.matrix, LinearKinematics(), self.limits, profile='supervised_relaxed')
        self.assertGreater(abs(accepted['dz_m']), .0005)
        self.assertLessEqual(abs(accepted['dz_m']), .005)
        self.assertLessEqual(np.linalg.norm(accepted['target']), .02)
        for position, _ in accepted['completion_corners']:
            self.assertLessEqual(np.linalg.norm(position[:2]), .002)

    def test_relaxed_requested_five_mm_step_and_fifty_mm_travel(self):
        self.make()
        bounds = nudge_bounds('supervised_relaxed', hardware=True)
        self.assertEqual(bounds['max_step_m'], .005)
        self.assertEqual(bounds['travel_m'], .05)
        proposal = incremental_proposal(np.zeros(6), [0., 100., 0., 100.], self.matrix,
                         LinearKinematics(), self.limits, profile='supervised_relaxed')
        accepted = quantized_proposal(proposal, np.zeros(6), [0., 100., 0., 100.],
                         self.matrix, LinearKinematics(), self.limits, profile='supervised_relaxed')
        self.assertGreater(abs(accepted['dz_m']), .002)
        self.assertLessEqual(abs(accepted['dz_m']), .005)
        self.assertLessEqual(np.linalg.norm(accepted['target']), .02)

    def test_relaxed_feedback_bounds_still_stop(self):
        config, self.snapshot, self.arms, self.limits, _ = fixture()
        config['incremental_profile'] = 'supervised_relaxed'
        c = IncrementalNudge(config, 10., self.arms, self.limits,
                             [5]*6, kinematics=LinearKinematics())
        self.arms['left']['position_rad'][0] = .0501
        self.arms['left']['tool_pose_base']['position_m'][0] = .0501
        with self.assertRaisesRegex(ValueError, 'measured_cartesian_bound'):
            c.tick(10., self.snapshot, self.arms, allowed=True)

    def test_relaxed_two_tick_target_offset_advances_after_cadence(self):
        config, self.snapshot, self.arms, self.limits, _ = fixture()
        config.update(incremental_profile='supervised_relaxed', encoder_resolution_rad=SDK_POSITION_QUANTUM)
        c = IncrementalNudge(config, 10., self.arms, self.limits, [5]*6, kinematics=LinearKinematics())
        for now in (10., 10.05, 10.1):
            advance(self.snapshot, now)
            c.tick(now, self.snapshot, self.arms, allowed=True)
        self.assertEqual(c.steps, 1)
        measured = c.target.copy()
        measured[2] += 2*SDK_POSITION_QUANTUM
        self.arms['left']['position_rad'] = measured.tolist()
        self.arms['left']['tool_pose_base']['position_m'] = measured[:3].tolist()
        self.arms['left']['velocity_rad_s'] = [.07]*6
        ready = c.next_observation_at
        for now in (ready+.01, ready+.06, ready+.11):
            advance(self.snapshot, now)
            c.tick(now, self.snapshot, self.arms, allowed=True)
        self.assertEqual(c.steps, 2)
        self.assertEqual(c.reason, 'incremental_vertical_step')
        self.assertLessEqual(np.linalg.norm(c.target-measured), .005)

    def test_recorded_failed_pose_has_relaxed_quantized_proposal(self):
        try:
            import pinocchio  # noqa: F401; available in the motor runtime environment
            import yaml
        except ImportError:
            self.skipTest('requires robot kinematics runtime')
        import json
        from pathlib import Path
        from cartesian_nudge import ToolKinematics
        root = Path(__file__).resolve().parents[1]
        robot = yaml.safe_load((root/'config/robot-configs/left.yaml').read_text())
        calibration = json.loads((root/'config/calibration/gear-supervised-trial.json').read_text())
        kin = ToolKinematics(robot)
        q = np.array([-.8155574798583984, 1.5343538522720337, .4674690067768097,
                       .18598228693008423, .463699072599411, .5171061754226685])
        error = [7.664337158203125, 32.115753173828125,
                 -6.2737274169921875, -8.678367614746094]
        matrix = calibration['image_jacobian_px_per_rad']
        limits = robot['robot']['joint_limits']
        continuous = incremental_proposal(q, error, matrix, kin, limits,
                                         profile='supervised_relaxed')
        accepted = quantized_proposal(continuous, q, error, matrix, kin, limits,
                                     profile='supervised_relaxed')
        self.assertLess(accepted['dz_m'], 0)
        self.assertLessEqual(abs(accepted['dz_m']), .002)
        self.assertLessEqual(np.linalg.norm(accepted['target']-q), .02)
        self.assertTrue(np.all(np.abs(accepted['predicted_height_error_px']) < np.abs(np.array(error)[[1, 3]])))
        origin, rotation = kin.pose(q)
        from cartesian_nudge import rotation_error
        for position, orientation in accepted['completion_corners']:
            self.assertLessEqual(np.linalg.norm(position-origin), .005)

    def test_relaxed_one_tick_floor_allows_bounded_near_target_crossing(self):
        self.make()
        error = [0., 3.1, 0., 3.1]
        continuous = incremental_proposal(np.zeros(6), error, self.matrix,
                    LinearKinematics(), self.limits, profile='supervised_relaxed')
        accepted = quantized_proposal(continuous, np.zeros(6), error, self.matrix,
                    LinearKinematics(), self.limits, profile='supervised_relaxed')
        self.assertEqual(max(abs(np.array(accepted['encoder_step_ticks']))), 1)
        self.assertTrue(np.all(accepted['predicted_height_error_px'] < -3))
        self.assertTrue(np.all(np.abs(accepted['predicted_height_error_px']) <= 6))

    def test_relaxed_submicrometer_request_can_issue_one_tick(self):
        self.make()
        self.matrix[1, 2] = self.matrix[3, 2] = 1e8
        result = incremental_proposal(np.zeros(6), [0., 4., 0., 4.], self.matrix,
                    LinearKinematics(), self.limits, profile='supervised_relaxed')
        accepted = quantized_proposal(result, np.zeros(6), [0., 4., 0., 4.], self.matrix,
                    LinearKinematics(), self.limits, profile='supervised_relaxed')
        self.assertEqual(max(abs(np.array(accepted['encoder_step_ticks']))), 1)
        self.assertLessEqual(np.linalg.norm(accepted['target']), .005)

    def test_unreached_tick_targets_do_not_stack_and_exhaust_distance_budget(self):
        config, self.snapshot, self.arms, self.limits, _ = fixture()
        config.update(incremental_profile='supervised_relaxed', encoder_resolution_rad=SDK_POSITION_QUANTUM)
        self.controller = IncrementalNudge(config, 10., self.arms, self.limits,
                                           [5]*6, kinematics=LinearKinematics())
        commands = self.run_loop(initial=3.1, follow=False)
        self.assertGreater(self.controller.steps, 1)
        self.assertEqual(self.controller.reason, 'cumulative_travel_limit')
        self.assertLessEqual(self.controller.commanded_path_m, .05)
        self.assertLessEqual(np.max(np.abs(commands[:, 2])), SDK_POSITION_QUANTUM*1.26)

    def test_relaxed_wrong_image_direction_does_not_stop_replanning(self):
        config, self.snapshot, self.arms, self.limits, _ = fixture()
        config['incremental_profile'] = 'supervised_relaxed'
        self.controller = IncrementalNudge(config, 10., self.arms, self.limits,
                                           [5]*6, kinematics=LinearKinematics())
        self.run_loop(gain=-1., iterations=100)
        c = self.controller
        self.assertGreater(c.steps, 1)
        self.assertEqual(c.reason, 'incremental_vertical_step')
        self.assertLessEqual(c.commanded_path_m, .05)

    def test_near_target_tick_reversals_remain_bounded_by_distance_budget(self):
        config, self.snapshot, self.arms, self.limits, _ = fixture()
        config.update(incremental_profile='supervised_relaxed', encoder_resolution_rad=SDK_POSITION_QUANTUM)
        c = IncrementalNudge(config, 10., self.arms, self.limits,
                             [5]*6, kinematics=LinearKinematics())
        directions = []
        for i in range(600):
            now = 10.+i*.05
            advance(self.snapshot, now)
            q = np.asarray(self.arms['left']['position_rad'])
            points = self.snapshot['predictions']['gear']['keypoints_2d']
            points[1][0][1] = points[2][0][1] = 103.1+10000*q[2]
            steps = c.steps
            command = c.tick(now, self.snapshot, self.arms, allowed=True)
            if command is not None:
                decoded = sdk_grid_target(command)[0]
                if c.steps > steps:
                    directions.append(np.sign(decoded[2]-q[2]))
                self.arms['left']['position_rad'] = decoded.tolist()
                self.arms['left']['tool_pose_base']['position_m'] = decoded[:3].tolist()
            if not c.running:
                break
        self.assertEqual(c.reason, 'cumulative_travel_limit')
        self.assertGreater(c.steps, 40)
        self.assertEqual(set(directions), {-1., 1.})
        self.assertTrue(all(a == -b for a, b in zip(directions, directions[1:])))
        self.assertLessEqual(max(abs(np.array(c.error_px)[[1, 3]])), 6)
        self.assertLessEqual(c.path_m, .05)

    def test_relaxed_requires_three_captures_after_command_cadence(self):
        config, self.snapshot, self.arms, self.limits, _ = fixture()
        config['incremental_profile'] = 'supervised_relaxed'
        c = IncrementalNudge(config, 10., self.arms, self.limits,
                             [5]*6, kinematics=LinearKinematics())
        for now in (10., 10.05, 10.1):
            advance(self.snapshot, now)
            c.tick(now, self.snapshot, self.arms, allowed=True)
        self.assertEqual(c.steps, 1)
        target = c.target.copy()
        ready = c.next_observation_at
        advance(self.snapshot, ready-.01)
        np.testing.assert_array_equal(c.tick(ready+.01, self.snapshot, self.arms, allowed=True), target)
        self.assertEqual(len(c.samples), 0)
        for index, now in enumerate((ready+.05, ready+.1, ready+.15)):
            advance(self.snapshot, now)
            c.tick(now, self.snapshot, self.arms, allowed=True)
            self.assertEqual(c.steps, 1 if index < 2 else 2)

    def test_distance_policy_evidence_keeps_identity_and_freshness_only(self):
        config, snapshot, arms, limits, _ = fixture()
        config.update(incremental_profile='supervised_relaxed', max_error_px=1)
        config['calibration']['reference_joint_position_rad'] = [1.]*6
        arms['left']['velocity_rad_s'] = [.7]*6
        arms['left']['gripper']['velocity'] = .7
        result = evaluate_incremental_evidence(config, 10., snapshot, arms, limits)
        self.assertEqual(result['reason'], 'nudging')
        self.assertEqual(evaluate_incremental_evidence(config, 11., snapshot, arms, limits)['reason'],
                         'stale_prediction')
        snapshot['predictions']['gear']['checkpoint_id'] = 'wrong-model'
        self.assertEqual(evaluate_incremental_evidence(config, 10., snapshot, arms, limits)['reason'],
                         'checkpoint_mismatch')

    def test_distance_policy_retains_hard_joint_and_3d_step_limits(self):
        for violation in ('joint', 'diagonal_step'):
            config, snapshot, arms, limits, _ = fixture()
            config['incremental_profile'] = 'supervised_relaxed'
            c = IncrementalNudge(config, 10., arms, limits, [5]*6, kinematics=LinearKinematics())
            if violation == 'joint':
                arms['left']['position_rad'][0] = limits['upper'][0]+.01
                reason = 'joint_absolute_limit'
            else:
                c.steps = 1
                arms['left']['position_rad'][:2] = [.004, .004]
                reason = 'measured_step_distance_limit'
            with self.assertRaisesRegex(ValueError, reason):
                c.tick(10., snapshot, arms, allowed=True)

    def test_unrepresentable_motor_step_never_commands(self):
        c = self.make()
        c.encoder_resolution = SDK_POSITION_QUANTUM
        self.run_loop(initial=5.)
        self.assertEqual(c.reason, 'motor_resolution_limited')
        self.assertEqual(c.steps, 0)
        self.assertIsNone(c.command())

    def test_feedback_bounds_remain_active(self):
        c = self.make()
        self.arms['left']['position_rad'][0] = .0003
        self.arms['left']['tool_pose_base']['position_m'][0] = .0003
        with self.assertRaisesRegex(ValueError, 'measured_cartesian_bound'):
            c.tick(10., self.snapshot, self.arms, allowed=True)


if __name__ == '__main__':
    unittest.main()
