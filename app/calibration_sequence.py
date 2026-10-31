"""Leased, physically gated +/- task-axis calibration sequence.

This module computes position targets only. The separate collector owns SDK
commands. No legacy visual Jacobian is used to choose movement directions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from alignment_assist import AlignmentAssist
from cartesian_nudge import rotation_error
from constrained_visual_nudge import vector, translation_inverse


def load_motion_gate(path):
    gate = json.loads(Path(path).read_text())
    if gate.get('operator_motion_verified') is not True:
        raise ValueError('independently_verified_motion_gate_required')
    axis = vector(gate.get('axis_base'), 3)
    if abs(np.linalg.norm(axis)-1) > 1e-6:
        raise ValueError('unit_task_axis_required')
    report_path = Path(gate['motion_audit_path'])
    if not report_path.is_absolute():
        report_path = Path(path).resolve().parent/report_path
    raw = report_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != gate.get('motion_audit_sha256'):
        raise ValueError('motion_audit_hash_mismatch')
    report = json.loads(raw)
    outcomes = report.get('outcomes', {})
    if report.get('method') != 'motion_first_audit' or outcomes.get('three_way_motion_agreement', 0) < 6:
        raise ValueError('six_independently_measured_motion_agreements_required')
    if any(value for key, value in outcomes.items() if key != 'three_way_motion_agreement'):
        raise ValueError('motion_audit_contains_failed_or_missing_measurements')
    measured = []
    for row in report.get('details', []):
        if row.get('outcome') != 'three_way_motion_agreement':
            continue
        delta = vector(row['encoder_fk_delta_m'], 3)
        along = float(delta @ axis)
        if np.linalg.norm(delta-along*axis) <= max(.0001, .1*abs(along)):
            measured.append(along)
    if sum(d >= .0002 for d in measured) < 3 or sum(d <= -.0002 for d in measured) < 3:
        raise ValueError('independent_positive_and_negative_axis_checks_required')
    reference = vector(gate.get('reference_q_rad'), 6)
    mapping = gate.get('camera_mapping')
    if not isinstance(mapping, dict) or not gate.get('checkpoint_id'):
        raise ValueError('motion_gate_perception_identity_required')
    return dict(gate, axis_base=axis.tolist(), reference_q_rad=reference.tolist())


class HoldToRunSequence:
    offsets = (.0005, 0., -.0005, 0., .0005, 0., -.0005, 0.)

    def __init__(self, config, now, arms, joint_limits, torque_limits, *, kinematics):
        self.config = config
        self.kinematics = kinematics
        self.limits = joint_limits
        raw_gate = Path(config['sequence_gate_path']).read_bytes()
        if hashlib.sha256(raw_gate).hexdigest() != config.get('sequence_gate_sha256'):
            raise ValueError('motion_gate_changed_restart_required')
        self.gate = load_motion_gate(config['sequence_gate_path'])
        if (config.get('task') != 'gear' or config.get('mode') != 'operator_supervised'
                or config.get('calibration', {}).get('side') != 'left'):
            raise ValueError('supervised_left_gear_only')
        for key in ('camera_mapping', 'checkpoint_id'):
            expected = config['calibration'][key]
            if self.gate[key] != expected:
                raise ValueError('motion_gate_'+key+'_mismatch')
        self.axis = np.asarray(self.gate['axis_base'])
        self.initial = vector(arms['left']['position_rad'], 6)
        if np.max(np.abs(self.initial-self.gate['reference_q_rad'])) > .1:
            raise ValueError('outside_verified_motion_region')
        self.target = self.initial.copy()
        self.xyz, self.rotation = kinematics.pose(self.initial)
        self.other = vector(arms['right']['position_rad'], 6)
        self.gripper = float(arms['left']['gripper']['position'])
        self.torque_limits = np.minimum(vector(torque_limits, 6),
            np.abs(vector(arms['left']['gravity_model_torque_nm'], 6))+.3)
        self.started = now
        self.renewed = None
        self.press_id = None
        self.running = True
        self.reason = 'hold_required'
        self.steps = 0
        self.settled_since = None
        self.last_stamp = None
        self.distinct = 0
        self.travel_m = 0.

    def renew(self, now, press_id):
        if self.press_id is not None and press_id != self.press_id:
            raise ValueError('different_press_cannot_renew_sequence')
        if self.renewed is not None and now-self.renewed > .2:
            raise ValueError('expired_press_cannot_rearm')
        self.press_id = press_id
        self.renewed = now

    def tick(self, now, snapshot, arms, *, allowed):
        if not allowed:
            raise ValueError('collection_or_grasp_ended')
        if self.renewed is None or not 0 <= now-self.renewed <= .2 or now-self.started > 20:
            raise ValueError('hold_released_expired_or_budget_exhausted')
        arm = arms['left']
        q = vector(arm['position_rad'], 6)
        velocity = vector(arm['velocity_rad_s'], 6)
        if np.linalg.norm(q-self.initial) > .006 or np.max(np.abs(q-self.target)) > .01:
            raise ValueError('joint_travel_or_tracking_bound')
        if np.max(np.abs(q-self.gate['reference_q_rad'])) > .1:
            raise ValueError('outside_verified_motion_region')
        position, rotation = self.kinematics.pose(q)
        if (np.linalg.norm(position-vector(arm['tool_pose_base']['position_m'], 3)) > .0001
                or rotation_error(rotation, arm['tool_pose_base']['rotation']) > .001):
            raise ValueError('feedback_model_mismatch')
        displacement = position-self.xyz
        self.travel_m = float(np.linalg.norm(displacement))
        off_axis = displacement-self.axis*np.dot(displacement, self.axis)
        if self.travel_m > .0008 or np.linalg.norm(off_axis) > .00015 or rotation_error(rotation, self.rotation) > .003:
            raise ValueError('measured_axis_path_bound')
        if np.max(np.abs(vector(arms['right']['position_rad'], 6)-self.other)) > .003:
            raise ValueError('carrier_moved')
        gp, gv = vector([arm['gripper']['position'], arm['gripper']['velocity']], 2)
        if abs(gp-self.gripper) > .02 or abs(gv) > .1:
            raise ValueError('gripper_changed')
        evidence = AlignmentAssist(self.config).evaluate(now, snapshot, arm, self.limits,
            other_arm=arms['right'], collecting=True, operator_confirmed=True, operator_press='calibration-sequence')
        if evidence['reason'] not in ('aligned', 'nudging'):
            raise ValueError(evidence['reason'])
        if not self.running:
            return self.target.copy()
        if np.max(np.abs(q-self.target)) > .0007 or np.max(np.abs(velocity)) > .02:
            self.settled_since = None
            self.distinct = 0
            return self.target.copy()
        if self.settled_since is None:
            self.settled_since = now
        stamp = int(snapshot['monotonic_ns'])
        if stamp != self.last_stamp and stamp/1e9 >= self.settled_since:
            self.distinct += 1
            self.last_stamp = stamp
        # Longer than endpoint capture, so the independent workbench can observe each hold.
        if now-self.settled_since < 1.2 or self.distinct < 5:
            self.reason = 'settling_and_capturing'
            return self.target.copy()
        if self.steps == len(self.offsets):
            self.running = False
            self.reason = 'sequence_complete_release_button'
            return self.target.copy()
        translation = self.xyz+self.offsets[self.steps]*self.axis-position
        delta = translation_inverse(q, self.kinematics) @ translation
        if np.linalg.norm(delta) > .003:
            raise ValueError('calibration_step_exceeds_joint_cap')
        target = q+delta
        lower, upper = vector(self.limits['lower'], 6), vector(self.limits['upper'], 6)
        if np.any(target <= lower+.03) or np.any(target >= upper-.03):
            raise ValueError('joint_target_limit')
        for a in np.linspace(0, 1, 11):
            p, r = self.kinematics.pose(q+a*delta)
            expected = position+a*translation
            if np.linalg.norm(p-expected) > .00005 or rotation_error(r, self.rotation) > .001:
                raise ValueError('predicted_axis_path_violation')
        self.target = target
        self.steps += 1
        self.settled_since = None
        self.distinct = 0
        self.reason = 'bounded_calibration_step'
        return target.copy()

    def status(self):
        return dict(active=True, correcting=self.running, reason=self.reason, steps=self.steps,
            controller='leased_calibration_sequence', tool_travel_m=self.travel_m,
            axis_base=self.axis.tolist(), target_position_rad=self.target.tolist(),
            lease_seconds=.2, max_seconds=20, max_offset_m=.0005,
            speed_limit_rad_s=.02, total_motor_torque_limits_nm=self.torque_limits.tolist())
