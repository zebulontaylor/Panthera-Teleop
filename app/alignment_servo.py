"""Bounded, explicit-start gear visual servo with fixed position targets.

CNN corrections are sampled only after the previous step has settled. Joint
position control supplies the holding force; this is not a weak torque nudge.
"""
from collections import deque

import numpy as np

from alignment_assist import AlignmentAssist


class AlignmentServo:
    def __init__(self, config, now, arms, joint_limits, torque_limits):
        self.config = dict(config)
        self.config['servo_travel_bounded'] = True
        self.started = now
        self.initial = np.asarray(arms['left']['position_rad'], dtype=float)
        self.target = self.initial.copy()
        self.other_initial = np.asarray(arms['right']['position_rad'], dtype=float)
        self.initial_xyz = np.asarray(arms['left']['tool_pose_base']['position_m'], dtype=float)
        self.travel_m = 0.
        self.limits = joint_limits
        self.torque_limits = np.minimum(np.asarray(torque_limits, dtype=float),
            np.abs(np.asarray(arms['left']['gravity_model_torque_nm'], dtype=float))+.5)
        self.running = True
        self.reason = 'settling'
        self.steps = 0
        self.samples = deque(maxlen=3)
        self.last_stamp = None
        self.settled_since = None
        self.last_evidence = now
        self.last_step = now
        self.best_error = None
        self.bad_steps = 0
        self.no_progress = 0
        self.error_px = None

    def stop_steps(self, reason, q=None):
        self.running = False
        self.reason = reason
        if q is not None:
            self.target = q.copy()

    def tick(self, now, snapshot, arms, *, allowed):
        if not allowed:
            raise ValueError('collection_or_grasp_ended')
        q = np.asarray(arms['left']['position_rad'], dtype=float)
        other = np.asarray(arms['right']['position_rad'], dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(other).all():
            raise ValueError('invalid_feedback')
        if np.max(np.abs(q-self.target)) > .04:
            raise ValueError('tracking_deviation')
        xyz = np.asarray(arms['left']['tool_pose_base']['position_m'], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('invalid_tool_feedback')
        self.travel_m = float(np.linalg.norm(xyz-self.initial_xyz))
        if self.travel_m >= .03 or xyz[2] < self.initial_xyz[2]-.002:
            self.stop_steps('workspace_travel_limit', q)
        if np.max(np.abs(other-self.other_initial)) > .03:
            self.stop_steps('carrier_moved', q)
        if not self.running:
            return self.target.copy()
        if now-self.started >= 30 or self.steps >= 80:
            self.stop_steps('trial_budget_exhausted', q)
            return self.target.copy()
        if np.linalg.norm(q-self.initial) >= .2 or np.max(np.abs(q-self.initial)) >= .15:
            self.stop_steps('total_travel_limit', q)
            return self.target.copy()
        if np.max(np.abs(q-self.target)) > .0015:
            self.settled_since = None
            if now-self.last_step > 3:
                self.stop_steps('step_did_not_settle')
            return self.target.copy()
        if self.settled_since is None:
            self.settled_since = now
        if now-self.settled_since < .15:
            return self.target.copy()
        stamp = snapshot.get('monotonic_ns')
        if stamp is None or stamp == self.last_stamp or stamp/1e9 < self.settled_since:
            if now-self.last_evidence > 1:
                self.stop_steps('fresh_prediction_unavailable')
            return self.target.copy()
        candidate = AlignmentAssist(self.config).evaluate(now, snapshot, arms['left'], self.limits,
            other_arm=arms['right'], collecting=True, operator_confirmed=True, operator_press='visual-servo')
        if candidate['reason'] not in ('nudging', 'aligned'):
            self.stop_steps(candidate['reason'])
            return self.target.copy()
        self.last_stamp = stamp
        self.last_evidence = now
        self.samples.append(np.asarray(candidate['error_px'], dtype=float))
        if len(self.samples) < 3:
            return self.target.copy()
        error = np.median(np.asarray(self.samples), axis=0)
        self.error_px = error.tolist()
        height_error = error[[1, 3]]
        magnitude = float(np.linalg.norm(height_error))
        if np.max(np.abs(height_error)) <= 3:
            self.stop_steps('aligned_holding')
            return self.target.copy()
        if self.best_error is None or magnitude < self.best_error-1:
            self.best_error = magnitude
            self.bad_steps = self.no_progress = 0
        else:
            self.no_progress += 1
            self.bad_steps = self.bad_steps+1 if magnitude > self.best_error+6 else 0
        if self.bad_steps >= 2 or self.no_progress >= 8:
            self.stop_steps('image_error_not_improving')
            return self.target.copy()
        cal = self.config['calibration']
        j = np.asarray(cal['image_jacobian_px_per_rad'], dtype=float)[[1, 3]]
        delta = -j.T @ np.linalg.solve(j @ j.T+25*np.eye(2), height_error)
        delta *= min(1., .003/max(1e-12, float(np.linalg.norm(delta))))
        target = q+delta
        lower, upper = np.asarray(self.limits['lower']), np.asarray(self.limits['upper'])
        if (np.any(target <= lower+.03) or np.any(target >= upper-.03)
                or np.linalg.norm(target-self.initial) > .2 or np.max(np.abs(target-self.initial)) > .15):
            self.stop_steps('target_limit')
            return self.target.copy()
        self.target = target
        self.steps += 1
        self.last_step = now
        self.settled_since = None
        self.samples.clear()
        self.reason = 'correcting'
        return self.target.copy()

    def status(self):
        return dict(active=True, correcting=self.running, reason=self.reason, steps=self.steps,
                    target_position_rad=self.target.tolist(), error_px=self.error_px,
                    step_norm_limit_rad=.003, total_norm_limit_rad=.2, tool_travel_limit_m=.03,
                    tool_travel_m=self.travel_m, max_seconds=30,
                    speed_limit_rad_s=.08, total_motor_torque_limits_nm=self.torque_limits.tolist())
