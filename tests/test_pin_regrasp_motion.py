import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from pin_regrasp_motion import RegraspConfig, plan_regrasp, rotation_exp, retreat_axis


class Kinematics:
    def pose(self, q):
        return np.asarray(q[:3]), rotation_exp(np.asarray(q[3:]))

    def jacobian(self, q):
        # Numerical world-aligned angular Jacobian checks the real IK convention.
        from pin_regrasp_motion import rotation_log
        p, r = self.pose(q)
        j = np.zeros((6, 6))
        for i in range(6):
            step = np.zeros(6); step[i] = 1e-6
            pp, rr = self.pose(q+step)
            j[:, i] = np.r_[(pp-p)/1e-6, rotation_log(rr@r.T)/1e-6]
        return j


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.anchor = np.zeros(14); self.anchor[0] = .3; self.anchor[6] = .1; self.anchor[7:] = .2
        self.lower = np.full(14, -2.); self.upper = np.full(14, 2.)
        self.lower[6] = 0.

    def plan(self, **kw):
        return plan_regrasp(self.anchor, Kinematics(), self.lower, self.upper, **kw)

    def test_forward_order_full_close_and_stationary_carrier(self):
        plan = self.plan()
        phases = list(dict.fromkeys(phase for phase, _ in plan))
        self.assertEqual(phases, ['release', 'retreat', 'return', 'close', 'lift', 'lower', 'settle'])
        for phase, target in plan:
            np.testing.assert_array_equal(target[7:], self.anchor[7:])
        for phase in ('close', 'lift', 'lower', 'settle'):
            end = [q for name, q in plan if name == phase][-1]
            self.assertEqual(end[6], 0.)
        np.testing.assert_allclose(plan[-1][1][:6], self.anchor[:6], atol=2e-4)

    def test_seeded_variation_and_speed_bounds(self):
        a, b = self.plan(seed=1), self.plan(seed=2)
        pa = [q for phase, q in a if phase == 'retreat'][-1]
        pb = [q for phase, q in b if phase == 'retreat'][-1]
        self.assertGreater(np.linalg.norm(pa[:6]-pb[:6]), 1e-4)
        targets = np.vstack([self.anchor]+[q for _, q in a])
        self.assertLessEqual(np.max(np.abs(np.diff(targets[:, :6], axis=0)))*30, .15001)
        self.assertLessEqual(np.max(np.abs(np.diff(targets[:, 6])))*30, 4.00001)

    def test_tool_frame_retreat(self):
        self.anchor[4] = np.pi/2
        plan = self.plan(config=RegraspConfig(tool_axis='+z', cone_deg=0, angle_jitter_deg=0))
        retreat = [q for phase, q in plan if phase == 'retreat'][-1]
        self.assertGreater(retreat[0]-self.anchor[0], .01)
        self.assertLess(abs(retreat[2]), 1e-4)

    def test_infeasible_target_fails_preflight(self):
        self.upper[2] = .005
        with self.assertRaisesRegex(ValueError, 'IK joint limit'):
            self.plan(config=RegraspConfig(cone_deg=0, angle_jitter_deg=0))

    def test_continuous_cycles_start_at_current_and_place_around_initial(self):
        config = RegraspConfig(placement_jitter_m=.02)
        current = self.anchor.copy()
        for seed in range(3):
            plan = self.plan(seed=seed, config=config, current14=current)
            first = plan[0][1]
            self.assertLess(np.max(np.abs(first[:6]-current[:6])), 2e-4)
            self.assertLessEqual(abs(first[6]-current[6])*config.hz, config.opening_speed_rad_s)
            closed = [q for phase, q in plan if phase == 'close'][-1]
            np.testing.assert_allclose(closed[:3], current[:3], atol=2e-5)
            returned = [q for phase, q in plan if phase == 'return']
            self.assertGreater(returned[0][6], returned[-1][6])
            self.assertGreaterEqual(returned[-1][6], self.lower[6])
            current = plan[-1][1]
            self.assertTrue(np.all(np.abs(current[:2]-self.anchor[:2]) <= .02002))
            np.testing.assert_array_equal(current[7:], self.anchor[7:])
            self.assertEqual(current[6], self.lower[6])
            np.testing.assert_allclose(current[3:6], self.anchor[3:6], atol=2e-4)

    def test_carrier_current_must_match_anchor(self):
        current = self.anchor.copy(); current[7] += .01
        with self.assertRaisesRegex(ValueError, 'Carrier'):
            self.plan(current14=current)

    def test_base_retreat_is_up_40_degrees_independent_of_wrist(self):
        self.anchor[:3] = [.3, .2, .1]
        config = RegraspConfig(cone_deg=0, angle_jitter_deg=0)
        expected = np.r_[-self.anchor[:2]/np.linalg.norm(self.anchor[:2])*np.cos(np.deg2rad(40)),
                         np.sin(np.deg2rad(40))]
        for wrist in ([0, 0, 0], [.2, -.3, .4]):
            self.anchor[3:6] = wrist
            plan = self.plan(config=config)
            retreat = [q for phase, q in plan if phase == 'retreat'][-1]
            delta = retreat[:3]-self.anchor[:3]
            np.testing.assert_allclose(delta/np.linalg.norm(delta), expected, atol=.002)
            self.assertLess(np.linalg.norm(retreat[:2]), np.linalg.norm(self.anchor[:2]))
            self.assertGreater(retreat[2], self.anchor[2])

    def test_base_cone_bounds_and_initial_reference_across_placement(self):
        config = RegraspConfig(cone_deg=8)
        axis = retreat_axis(self.anchor[:3], np.eye(3), config)
        current = self.anchor.copy(); current[1] += .02
        for seed in range(5):
            plan = self.plan(seed=seed, config=config, current14=current)
            end = [q for phase, q in plan if phase == 'retreat'][-1]
            delta = end[:3]-current[:3]
            self.assertGreaterEqual(np.dot(delta/np.linalg.norm(delta), axis), np.cos(np.deg2rad(8.1)))
            self.assertGreater(delta[2], 0)

    def test_base_axis_degenerate_reference_rejected(self):
        self.anchor[:2] = 0
        with self.assertRaisesRegex(ValueError, 'directly above the base'):
            self.plan()

    def test_closing_spatial_gate_constant_rate_and_no_timing_stretch(self):
        config = RegraspConfig(retreat_m=.09, speed_m_s=.15, open_gripper_rad=2.,
                               angle_jitter_deg=0, cone_deg=0,
                               joint_speed_rad_s=.6, gripper_speed_rad_s=.4)
        plan = self.plan(config=config)
        returned = [q for phase, q in plan if phase == 'return']
        previous = [q for phase, q in plan if phase == 'retreat'][-1]
        for q in returned:
            distance = np.linalg.norm(q[:3]-self.anchor[:3])
            if distance > .02:
                self.assertEqual(q[6], config.open_gripper_rad)
            elif previous[6] > 0:
                self.assertAlmostEqual(previous[6]-q[6], min(previous[6], .4/30), places=10)
            previous = q
        self.assertGreater(returned[-1][6], 1.)  # Arm arrives without waiting for fingers.
        closes = [q for phase, q in plan if phase == 'close']
        self.assertAlmostEqual(returned[-1][6]-closes[0][6], .4/30, places=10)
        self.assertEqual(closes[-1][6], 0.)
        fast = self.plan(config=RegraspConfig(retreat_m=.09, speed_m_s=.15, open_gripper_rad=2.,
                          angle_jitter_deg=0, cone_deg=0, joint_speed_rad_s=.6,
                          gripper_speed_rad_s=5.))
        self.assertEqual(len(returned), sum(phase == 'return' for phase, _ in fast))
        closing = np.array([q[6] for phase, q in plan if phase in ('return', 'close')])
        self.assertTrue(np.all(np.diff(closing) <= 1e-12))
        changes = -np.diff(closing)
        self.assertTrue(np.all(changes <= .4/30+1e-12))
        active = changes[changes > 1e-10]
        np.testing.assert_allclose(active[:-1], .4/30, atol=1e-12)

    def test_slow_return_closes_at_arrival_at_a_steady_rate(self):
        config = RegraspConfig(retreat_m=.06, speed_m_s=.015, open_gripper_rad=2.,
                               angle_jitter_deg=0, cone_deg=0, gripper_speed_rad_s=2.)
        plan = self.plan(config=config)
        returned = [q for phase, q in plan if phase == 'return']
        previous = [q for phase, q in plan if phase == 'retreat'][-1]
        changes = []
        for q in returned:
            distance = np.linalg.norm(q[:3]-self.anchor[:3])
            if distance > 2e-5:
                self.assertGreater(q[6], self.lower[6])
            change = previous[6]-q[6]
            if change > 1e-10:
                changes.append(change)
            previous = q
        self.assertGreater(len(changes), 2)
        np.testing.assert_allclose(changes, changes[0], atol=1e-12)
        self.assertLessEqual(changes[0]*config.hz, config.gripper_speed_rad_s)
        self.assertEqual(returned[-1][6], self.lower[6])
        self.assertEqual(len([q for phase, q in plan if phase == 'close']), 1)

    def test_fast_opening_no_dwell_and_independent_closing(self):
        self.anchor[6] = 0.
        cfg = RegraspConfig(open_gripper_rad=2.)
        plan = self.plan(config=cfg)
        released = [q for phase, q in plan if phase == 'release']
        self.assertEqual(len(released), 15)  # Full 2-rad travel in .5s.
        values = np.r_[self.anchor[6], [q[6] for q in released]]
        np.testing.assert_allclose(np.diff(values), 4/30, atol=1e-12)
        self.assertEqual(released[-1][6], 2.)
        for q in released:
            np.testing.assert_array_equal(q[:6], self.anchor[:6])
        self.assertEqual(plan[len(released)][0], 'retreat')
        slow = self.plan(config=RegraspConfig(open_gripper_rad=2., opening_speed_rad_s=1.))
        self.assertEqual(sum(phase == 'release' for phase, _ in slow), 60)
        for phase in ('retreat', 'return', 'close'):
            np.testing.assert_allclose([q for name, q in plan if name == phase],
                                       [q for name, q in slow if name == phase])

    def test_extended_speed_and_placement_limits(self):
        RegraspConfig(speed_m_s=.15, joint_speed_rad_s=.6, gripper_speed_rad_s=5., placement_jitter_m=.1)
        for kwargs in ({'speed_m_s': .151}, {'joint_speed_rad_s': .601},
                       {'gripper_speed_rad_s': 5.01}, {'opening_speed_rad_s': 5.01}, {'opening_speed_rad_s': 0}, {'placement_jitter_m': .101},
                       {'closing_distance_m': .021}):
            with self.assertRaises(ValueError):
                RegraspConfig(**kwargs)

    def test_invalid_configuration(self):
        for kwargs in ({'retreat_m': 1.}, {'cone_deg': float('nan')}, {'hz': 0}, {'base_elevation_deg': -1}, {'base_elevation_deg': 91}, {'base_elevation_deg': float('nan')}):
            with self.assertRaises(ValueError):
                RegraspConfig(**kwargs)


class AlignmentTests(unittest.TestCase):
    setUp = PlannerTests.setUp
    plan = PlannerTests.plan

    def alignment(self, **kw):
        from pin_regrasp_motion import plan_alignment
        return plan_alignment(self.anchor, Kinematics(), self.lower, self.upper, **kw)

    def test_alignment_yaw_clockwise_preserves_height_and_restores_anchor(self):
        position = np.array([.3, .2, .1])
        regrasp_axis = retreat_axis(position, np.eye(3), RegraspConfig())
        zero = retreat_axis(position, np.eye(3),
                            RegraspConfig(task='alignment', alignment_yaw_deg=0))
        np.testing.assert_allclose(zero, regrasp_axis)
        for yaw in (40., -40.):
            config = RegraspConfig(task='alignment', alignment_yaw_deg=yaw, cone_deg=0)
            axis = retreat_axis(position, np.eye(3), config)
            signed_cross = zero[0]*axis[1]-zero[1]*axis[0]
            self.assertLess(signed_cross*yaw, 0.)
            cosine = np.dot(zero[:2], axis[:2])/(np.linalg.norm(zero[:2])*np.linalg.norm(axis[:2]))
            self.assertAlmostEqual(cosine, np.cos(np.deg2rad(40.)))
            self.assertAlmostEqual(axis[2], zero[2])
            self.assertAlmostEqual(np.linalg.norm(axis), 1.)
            np.testing.assert_allclose(retreat_axis(position, rotation_exp(np.array([.2, .3, -.4])), config), axis)
            plan = self.alignment(config=config)
            end_retreat = [q for phase, q in plan if phase == 'retreat'][-1]
            expected = retreat_axis(self.anchor[:3], np.eye(3), config)
            actual = end_retreat[:3]-self.anchor[:3]
            np.testing.assert_allclose(actual/np.linalg.norm(actual), expected, atol=.002)
            np.testing.assert_allclose(plan[-1][1], self.anchor, atol=2e-4)
        self.assertEqual(RegraspConfig(task='alignment').alignment_yaw_deg, 40.)

    def test_alignment_yaw_bounds_and_tool_axis_override(self):
        for yaw in (-180., 180.):
            RegraspConfig(alignment_yaw_deg=yaw)
        for yaw in (-180.1, 180.1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'alignment_yaw_deg'):
                RegraspConfig(alignment_yaw_deg=yaw)
        rotation = rotation_exp(np.array([.2, -.3, .1]))
        for task in ('alignment', 'regrasp'):
            a = retreat_axis(self.anchor[:3], rotation,
                             RegraspConfig(task=task, tool_axis='+z', alignment_yaw_deg=40))
            b = retreat_axis(self.anchor[:3], rotation,
                             RegraspConfig(task=task, tool_axis='+z', alignment_yaw_deg=-40))
            np.testing.assert_allclose(a, rotation[:, 2])
            np.testing.assert_array_equal(a, b)
        a = retreat_axis(self.anchor[:3], rotation, RegraspConfig(alignment_yaw_deg=40))
        b = retreat_axis(self.anchor[:3], rotation, RegraspConfig(alignment_yaw_deg=-40))
        np.testing.assert_array_equal(a, b)

    def test_alignment_fixed_grasps_and_hover(self):
        plan = self.alignment()
        self.assertEqual(list(dict.fromkeys(p for p, _ in plan)), ['retreat', 'return', 'verify'])
        for _, q in plan:
            np.testing.assert_array_equal(q[7:], self.anchor[7:])
            self.assertEqual(q[6], self.anchor[6])
            np.testing.assert_allclose(q[3:6], self.anchor[3:6], atol=2e-4)
            self.assertGreaterEqual(q[2], self.anchor[2]-2e-5)
        np.testing.assert_allclose(plan[-1][1], self.anchor, atol=2e-4)
        targets = np.vstack([self.anchor]+[q for _, q in plan])
        self.assertLessEqual(np.max(np.abs(np.diff(targets[:, :6], axis=0)))*30, .15001)

    def test_alignment_drag_calibrated_rotated_bases(self):
        transform = np.eye(4)
        transform[:3, :3] = rotation_exp(np.array([0., 0., np.pi/2]))
        config = RegraspConfig(task='alignment', carrier_jitter_m=.006)
        plan = self.alignment(config=config, right_kinematics=Kinematics(), T_left_base_right_base=transform)
        self.assertEqual(list(dict.fromkeys(p for p, _ in plan)),
                         ['retreat', 'return', 'verify'])
        end = plan[-1][1]
        displacement = transform[:3, :3] @ (end[7:10]-self.anchor[7:10])
        np.testing.assert_allclose(end[:3]-self.anchor[:3], displacement, atol=4e-5)
        self.assertLessEqual(np.linalg.norm(displacement), .00602)
        self.assertAlmostEqual(displacement[2], 0., places=5)
        for _, q in plan:
            np.testing.assert_array_equal(q[[6, 13]], self.anchor[[6, 13]])
        all_q = np.vstack([self.anchor]+[q for _, q in plan])
        self.assertLessEqual(np.max(np.abs(np.diff(all_q[:, 7:13], axis=0)))*30, .15001)

    def test_alignment_retreat_moves_both_arms_on_shared_relative_path(self):
        transform = np.eye(4)
        rotation = rotation_exp(np.array([.3, -.2, 1.2]))
        transform[:3, :3] = rotation
        config = RegraspConfig(task='alignment', carrier_jitter_m=.02,
                               cone_deg=0, angle_jitter_deg=0)
        kwargs = dict(config=config, right_kinematics=Kinematics(),
                      T_left_base_right_base=transform)
        current = self.anchor.copy()
        for seed in (4, 7):
            plan = self.alignment(current14=current, seed=seed, **kwargs)
            retreat = [q for phase, q in plan if phase == 'retreat']
            end = retreat[-1]
            left_delta = end[:3]-current[:3]
            right_delta = rotation @ (end[7:10]-current[7:10])
            relative_retreat = left_delta-right_delta
            self.assertGreater(np.linalg.norm(right_delta), .001)
            axis = retreat_axis(self.anchor[:3], np.eye(3), config)
            np.testing.assert_allclose(relative_retreat/np.linalg.norm(relative_retreat), axis, atol=.003)
            for tick, q in enumerate(retreat, 1):
                t = tick/len(retreat)
                u = t**3*(10+t*(-15+6*t))
                np.testing.assert_allclose(q[:3]-current[:3], u*left_delta, atol=4e-5)
                np.testing.assert_allclose(rotation @ (q[7:10]-current[7:10]), u*right_delta, atol=4e-5)
                relative = q[:3]-current[:3]-rotation @ (q[7:10]-current[7:10])
                np.testing.assert_allclose(relative, u*relative_retreat, atol=5e-5)
                np.testing.assert_allclose(q[3:6], self.anchor[3:6], atol=2e-4)
                np.testing.assert_allclose(q[10:13], self.anchor[10:13], atol=2e-4)
            middle = retreat[len(retreat)//2]
            self.assertGreater(np.linalg.norm(middle[:3]-current[:3]), .001)
            self.assertGreater(np.linalg.norm(middle[7:10]-current[7:10]), .0005)
            for phase, q in plan:
                if phase in ('return', 'verify'):
                    np.testing.assert_array_equal(q[7:], end[7:])
            current = plan[-1][1]

    def test_alignment_preflight_rejects_missing_frames_limits_and_changed_grasp(self):
        config = RegraspConfig(task='alignment', carrier_jitter_m=.006)
        with self.assertRaisesRegex(ValueError, 'calibrated'):
            self.alignment(config=config, right_kinematics=Kinematics())
        current = self.anchor.copy(); current[6] += .01
        with self.assertRaisesRegex(ValueError, 'grippers'):
            self.alignment(current14=current)
        self.upper[2] = .005
        with self.assertRaisesRegex(ValueError, 'IK joint limit'):
            self.alignment(config=RegraspConfig(task='alignment', cone_deg=0))
        with self.assertRaises(ValueError):
            RegraspConfig(carrier_jitter_m=.101)

    def test_alignment_right_ik_failure_is_preflighted(self):
        class Unreachable(Kinematics):
            def jacobian(self, q):
                return np.zeros((6, 6))
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            self.alignment(config=RegraspConfig(task='alignment', carrier_jitter_m=.006),
                           right_kinematics=Unreachable(), T_left_base_right_base=np.eye(4))

    def test_alignment_coordinated_retreat_honors_selected_speed(self):
        transform = np.eye(4)
        transform[:3, :3] = rotation_exp(np.array([0., 0., np.pi/2]))
        durations = []
        endpoints = []
        for speed in (.005, .05):
            config = RegraspConfig(task='alignment', carrier_jitter_m=.012,
                                   speed_m_s=speed, joint_speed_rad_s=.6)
            plan = self.alignment(config=config, seed=4, right_kinematics=Kinematics(),
                                  T_left_base_right_base=transform)
            drag = [q for phase, q in plan if phase == 'retreat']
            durations.append(len(drag)/config.hz)
            endpoints.append(drag[-1])
            previous = self.anchor.copy()
            for phase, q in plan:
                joint_speed = np.abs(q-previous)*config.hz
                self.assertLessEqual(np.max(joint_speed), config.joint_speed_rad_s*1.00001)
                for index in (0, 7):
                    before, _ = Kinematics().pose(previous[index:index+6])
                    after, _ = Kinematics().pose(q[index:index+6])
                    self.assertLessEqual(np.linalg.norm(after-before)*config.hz,
                                         config.speed_m_s*1.00001)
                self.assertEqual(q[6], self.anchor[6])
                self.assertEqual(q[13], self.anchor[13])
                previous = q
        self.assertLess(durations[1], durations[0]/2)
        np.testing.assert_allclose(endpoints[0][7:], endpoints[1][7:], atol=4e-5)

    def test_alignment_cycles_have_continuity_and_no_random_walk(self):
        transform = np.eye(4)
        transform[:3, :3] = rotation_exp(np.array([.3, -.2, 1.2]))
        # Translation must not enter displacement conversion.
        transform[:3, 3] = [.4, -.3, .1]
        config = RegraspConfig(task='alignment', carrier_jitter_m=.006)
        kwargs = dict(right_kinematics=Kinematics(), T_left_base_right_base=transform)
        current = self.anchor.copy()
        for seed in (1, 2, 3, 1):
            plan = self.alignment(config=config, current14=current, seed=seed, **kwargs)
            samples = np.vstack([current]+[q for _, q in plan])
            self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0)))*config.hz,
                                 config.joint_speed_rad_s*1.00001)
            for _, q in plan:
                self.assertTrue(np.all(q >= self.lower))
                self.assertTrue(np.all(q <= self.upper))
            end = plan[-1][1]
            drag = transform[:3, :3] @ (end[7:10]-self.anchor[7:10])
            self.assertLessEqual(np.linalg.norm(drag), config.carrier_jitter_m+2e-5)
            self.assertLess(abs(drag[2]), 2e-5)
            np.testing.assert_allclose(end[:3]-self.anchor[:3], drag, atol=4e-5)
            from_anchor = self.alignment(config=config, seed=seed, **kwargs)[-1][1]
            np.testing.assert_allclose(end, from_anchor, atol=4e-5)
            current = end
        # Reducing jitter to zero restores both arm references continuously.
        plan = self.alignment(config=RegraspConfig(task='alignment'), current14=current, **kwargs)
        samples = np.vstack([current]+[q for _, q in plan])
        self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0)))*config.hz,
                             config.joint_speed_rad_s*1.00001)
        np.testing.assert_allclose(plan[-1][1], self.anchor, atol=4e-5)

    def test_alignment_carrier_limits_fail_before_plan_return(self):
        # Both sides are independently bounded, including the optional drag.
        self.lower[7:10] = self.anchor[7:10]-1e-5
        self.upper[7:10] = self.anchor[7:10]+1e-5
        with self.assertRaisesRegex(ValueError, 'IK joint limit'):
            self.alignment(config=RegraspConfig(task='alignment', carrier_jitter_m=.006),
                           right_kinematics=Kinematics(), T_left_base_right_base=np.eye(4))

    def test_wrong_task_rejected(self):
        with self.assertRaisesRegex(ValueError, 'alignment task'):
            self.alignment(config=RegraspConfig())
        with self.assertRaisesRegex(ValueError, 'regrasp task'):
            self.plan(config=RegraspConfig(task='alignment'))

if __name__ == '__main__':
    unittest.main()
