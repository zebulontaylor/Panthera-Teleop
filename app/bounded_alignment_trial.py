"""One-step position-hold runtime for the two additional experiments.

No motor methods are called here. The separate hand-guide entry points own
commands and exclusive controller locks. No automatic multi-step integration.
"""
from collections import deque
import numpy as np

from alignment_assist import AlignmentAssist
from cartesian_nudge import rotation_error
from constrained_visual_nudge import constrained_visual_proposal, forward_axis, vector
from stereo_pose_nudge import stereo_residual, stereo_pose_proposal, validate_calibration


class BoundedAlignmentTrial:
    mode = None

    def __init__(self, config, now, arms, joint_limits, torque_limits, *, kinematics):
        self.config = config
        self.settings = config.get('experiment', {})
        self.kinematics = kinematics
        self.limits = joint_limits
        self.initial = vector(arms['left']['position_rad'], 6)
        self.target = self.initial.copy()
        self.other = vector(arms['right']['position_rad'], 6)
        self.gripper = float(arms['left']['gripper']['position'])
        self.xyz, self.rotation = kinematics.pose(self.initial)
        self.torque_limits = np.minimum(vector(torque_limits, 6),
            np.abs(vector(arms['left']['gravity_model_torque_nm'], 6))+.3)
        self.started = now
        self.running = True
        self.steps = 0
        self.reason = 'waiting_for_fresh_evidence'
        self.target_position = None
        self.samples = deque(maxlen=3)
        self.last_stamp = None
        self.settled_since = None
        self.error_px = None
        self.residual = None
        self.before = None
        self.proposal = None
        self.travel_m = 0.

    def stop(self, reason, q):
        self.reason = reason
        self.running = False
        self.target = q.copy()
        return self.target.copy()

    def tick(self, now, snapshot, arms, *, allowed):
        if not allowed:
            raise ValueError('collection_or_grasp_ended')
        arm = arms['left']
        q = vector(arm['position_rad'], 6)
        velocity = vector(arm['velocity_rad_s'], 6)
        other = vector(arms['right']['position_rad'], 6)
        gp, gv = vector([arm['gripper']['position'], arm['gripper']['velocity']], 2)
        if abs(gp-self.gripper) > .04 or abs(gv) > .1:
            raise ValueError('gripper_changed')
        if np.linalg.norm(q-self.initial) > .006 or np.max(np.abs(q-self.target)) > .01:
            raise ValueError('measured_joint_or_tracking_bound')
        position, rotation = self.kinematics.pose(q)
        if not np.isfinite(position).all():
            raise ValueError('invalid_kinematic_position')
        observed = arm['tool_pose_base']
        if (np.linalg.norm(position-vector(observed['position_m'], 3)) > .0001
                or rotation_error(rotation, observed['rotation']) > .001):
            raise ValueError('feedback_model_mismatch')
        displacement = position-self.xyz
        self.travel_m = float(np.linalg.norm(displacement))
        if self.travel_m > .001 or rotation_error(rotation, self.rotation) > .005:
            raise ValueError('measured_cartesian_bound')
        if np.max(np.abs(other-self.other)) > .005:
            return self.stop('carrier_moved', q)
        if self.settings.get('axis_verified') is not True:
            return self.stop('measured_forward_axis_required', q)
        axis = forward_axis(self.settings)
        if np.dot(displacement, axis) < -.0001:
            raise ValueError('measured_retreat')
        if not self.running:
            return self.target.copy()
        if (self.config.get('task') != 'gear' or self.config.get('mode') != 'operator_supervised'
                or self.config.get('calibration', {}).get('side') != 'left'):
            return self.stop('supervised_left_gear_only', q)
        if self.mode == 'stereo':
            try:
                validate_calibration(self.settings.get('stereo_calibration', {}), self.config['calibration'])
            except (ValueError, KeyError, TypeError) as exc:
                return self.stop(str(exc), q)
        if now-self.started > 5:
            return self.stop('trial_timeout', q)
        if np.max(np.abs(q-self.target)) > .001 or np.max(np.abs(velocity)) > .02:
            self.settled_since = None
            return self.target.copy()
        if self.settled_since is None:
            self.settled_since = now
        if now-self.settled_since < .15:
            return self.target.copy()
        stamp = snapshot.get('monotonic_ns')
        if stamp is None or (self.last_stamp is not None and stamp <= self.last_stamp) or stamp/1e9 < self.settled_since:
            return self.target.copy()
        candidate = AlignmentAssist(self.config).evaluate(now, snapshot, arm, self.limits,
            other_arm=arms['right'], collecting=True, operator_confirmed=True,
            operator_press=self.mode+'-single-step')
        if candidate['reason'] not in ('nudging', 'aligned'):
            return self.stop(candidate['reason'], q)
        self.last_stamp = stamp
        self.error_px = candidate['error_px']
        try:
            measurement = (stereo_residual(snapshot, self.settings['stereo_calibration'])
                if self.mode == 'stereo' else np.asarray(candidate['error_px']))
            if not np.isfinite(measurement).all():
                raise ValueError('invalid_alignment_measurement')
        except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
            return self.stop(str(exc), q)
        self.samples.append(measurement)
        if len(self.samples) < 3:
            return self.target.copy()
        measurement = np.median(self.samples, axis=0)
        residual = measurement if self.mode == 'stereo' else measurement[[1, 3]]
        self.residual = residual.tolist()
        tolerance = .00005 if self.mode == 'stereo' else 2.
        if self.steps:
            wanted = self.proposal['delta_xyz_m']
            if np.dot(displacement, wanted) < .25*np.dot(wanted, wanted):
                return self.stop('physical_step_not_confirmed', q)
            if np.any(np.abs(residual) > np.abs(self.before)+tolerance):
                return self.stop('alignment_error_worsened', q)
            reduction = np.linalg.norm(self.before)-np.linalg.norm(residual)
            return self.stop('single_step_observed_holding' if reduction > tolerance
                else 'alignment_response_inconclusive', q)
        if np.max(np.abs(residual)) <= (.0003 if self.mode == 'stereo' else 3.):
            return self.stop('aligned_holding', q)
        if self.mode == 'stereo' and np.linalg.norm(residual) > .01:
            return self.stop('outside_3d_alignment_neighborhood', q)
        try:
            if self.mode == 'stereo':
                self.proposal = stereo_pose_proposal(q, residual, self.kinematics, self.limits, self.settings)
            else:
                self.proposal = constrained_visual_proposal(q, measurement,
                    self.config['calibration']['image_jacobian_px_per_rad'],
                    self.kinematics, self.limits, self.settings)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return self.stop(str(exc), q)
        self.before = residual.copy()
        self.target = self.proposal['target']
        self.steps = 1
        self.samples.clear()
        self.settled_since = None
        self.reason = 'one_'+self.mode+'_step'
        return self.target.copy()

    def status(self):
        return dict(active=True, correcting=self.running, reason=self.reason, steps=self.steps,
            target_position_rad=self.target.tolist(), error_px=self.error_px,
            alignment_residual=self.residual, residual_units='meters' if self.mode == 'stereo' else 'pixels',
            controller='experimental_'+self.mode, tool_travel_m=self.travel_m,
            tool_travel_limit_m=.001, max_step_m=.0005, max_seconds=5,
            speed_limit_rad_s=.02, total_motor_torque_limits_nm=self.torque_limits.tolist())


class ConstrainedVisualNudge(BoundedAlignmentTrial):
    mode = 'constrained_visual'


class StereoPoseNudge(BoundedAlignmentTrial):
    mode = 'stereo'
