"""Experimental one-step gear height correction; no device access in this module.

Uses the existing visual calibration only as a local prediction, not as proof
of Cartesian accuracy. Each start permits ONE step, followed by position hold.
"""
from collections import deque
import numpy as np

from alignment_assist import AlignmentAssist


def rotation_error(actual, reference):
    for rotation in (actual, reference):
        rotation = np.asarray(rotation)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or abs(np.linalg.det(rotation)-1) > 1e-5):
            raise ValueError('invalid_rotation')
    relative = np.asarray(actual) @ np.asarray(reference).T
    return float(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))


class ToolKinematics:
    """URDF-only kinematics. Constructing this never opens a motor connection."""
    def __init__(self, config):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(config['urdf']['file_path'])
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError('expected_six_joint_model')
        if list(self.model.names)[1:] != config['kinematics']['joint_names']:
            raise ValueError('kinematic_joint_order_mismatch')
        name = config['urdf']['end_effector_link']
        if not self.model.existFrame(name):
            raise ValueError('missing_tool_frame')
        self.frame = self.model.getFrameId(name)
        self.data = self.model.createData()

    def pose(self, q):
        self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
        pose = self.data.oMf[self.frame]
        return pose.translation.copy(), pose.rotation.copy()

    def jacobian(self, q):
        return self.pin.computeFrameJacobian(self.model, self.data, np.asarray(q),
            self.frame, self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()


def vertical_proposal(q, error, image_jacobian, kinematics, limits):
    """Constrain all six tool DOFs; reject infeasible or conflicting proposals."""
    q = np.asarray(q, dtype=float)
    error = np.asarray(error, dtype=float)[[1, 3]]
    j = kinematics.jacobian(q)
    singular = np.linalg.svd(j, compute_uv=False)
    if singular[-1] < 1e-4 or singular[0]/singular[-1] > 1000:
        raise ValueError('kinematic_singularity')
    # Translation is in base-frame meters; angular entries are radians.
    direction = np.linalg.solve(j, np.array([0., 0., 1., 0., 0., 0.]))
    slopes = np.asarray(image_jacobian, dtype=float)[[1, 3]] @ direction
    if not np.isfinite(slopes).all() or np.any(np.abs(slopes) < 100):
        raise ValueError('height_response_unobservable')
    requested = -error/slopes
    if np.any(np.abs(error) <= 3) or requested[0]*requested[1] <= 0:
        raise ValueError('camera_height_direction_disagreement')
    if max(abs(requested))/min(abs(requested)) > 3:
        raise ValueError('camera_height_magnitude_disagreement')
    dz = float(np.clip(-np.dot(slopes, error)/np.dot(slopes, slopes), -.0005, .0005))
    delta = direction*dz
    scale = min(1., .003/max(np.linalg.norm(delta), 1e-12))
    delta *= scale
    dz *= scale
    if abs(dz) < .00002:
        raise ValueError('step_too_small')
    target = q+delta
    if (np.any(target <= np.asarray(limits['lower'])+.03)
            or np.any(target >= np.asarray(limits['upper'])-.03)):
        raise ValueError('joint_target_limit')
    xyz, rotation = kinematics.pose(q)
    # Check the interpolated joint path, not only its endpoint.
    for fraction in np.linspace(0, 1, 11):
        p, r = kinematics.pose(q+fraction*delta)
        expected = xyz+np.array([0., 0., fraction*dz])
        if np.linalg.norm(p-expected) > .00005 or rotation_error(r, rotation) > .001:
            raise ValueError('predicted_path_not_vertical')
    return dict(target=target, dz_m=dz, slopes_px_per_m=slopes,
                predicted_height_error_px=error+slopes*dz)


class CartesianNudge:
    """Drop-in alternative to AlignmentServo, with a single step per start."""
    def __init__(self, config, now, arms, joint_limits, torque_limits, *, kinematics):
        self.config = config
        self.kinematics = kinematics
        self.limits = joint_limits
        self.initial = np.asarray(arms['left']['position_rad'], dtype=float)
        self.target = self.initial.copy()
        self.xyz, self.rotation = kinematics.pose(self.initial)
        self.other = np.asarray(arms['right']['position_rad'], dtype=float)
        self.gripper_position = float(arms['left']['gripper']['position'])
        self.torque_limits = np.minimum(np.asarray(torque_limits),
            np.abs(arms['left']['gravity_model_torque_nm'])+.3)
        self.started = now
        self.reason = 'waiting_for_fresh_evidence'
        self.running = True
        self.steps = 0
        self.error_px = None
        self.travel_m = 0.
        self.samples = deque(maxlen=3)
        self.last_stamp = None
        self.settled_since = None
        self.before_error = None
        self.step_time = None
        self.proposal = None
        # The supplied profile must retain the actual local calibration gates.
        if (config.get('task') != 'gear' or config.get('mode') != 'operator_supervised'
                or config.get('calibration', {}).get('side') != 'left'):
            raise ValueError('supervised_gear_only')

    def stop(self, reason, q):
        self.reason = reason
        self.running = False
        self.target = q.copy()
        return self.target.copy()

    def tick(self, now, snapshot, arms, *, allowed):
        if not allowed:
            raise ValueError('collection_or_grasp_ended')
        arm = arms['left']
        q = np.asarray(arm['position_rad'], dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('invalid_feedback')
        other = np.asarray(arms['right']['position_rad'], dtype=float)
        velocity = np.asarray(arm['velocity_rad_s'], dtype=float)
        gp = float(arm['gripper']['position'])
        gv = float(arm['gripper']['velocity'])
        if (other.shape != (6,) or velocity.shape != (6,)
                or not np.isfinite(np.r_[other, velocity, gp, gv]).all()):
            raise ValueError('invalid_feedback')
        if abs(gp-self.gripper_position) > .04 or abs(gv) > .1:
            raise ValueError('gripper_changed')
        if np.linalg.norm(q-self.initial) > .006:
            raise ValueError('measured_joint_travel_limit')
        p, r = self.kinematics.pose(q)
        self.travel_m = float(np.linalg.norm(p-self.xyz))
        # Feedback/FK agreement also catches a mismatched model/tool definition.
        observed = arm['tool_pose_base']
        observed_xyz = np.asarray(observed['position_m'])
        if observed_xyz.shape != (3,) or not np.isfinite(observed_xyz).all():
            raise ValueError('invalid_tool_feedback')
        if (np.linalg.norm(p-observed_xyz) > .0001
                or rotation_error(r, observed['rotation']) > .001):
            raise ValueError('feedback_model_mismatch')
        if np.max(np.abs(q-self.target)) > .01:
            raise ValueError('tracking_deviation')
        if (np.linalg.norm((p-self.xyz)[:2]) > .00025
                or rotation_error(r, self.rotation) > .005 or self.travel_m > .001):
            raise ValueError('measured_cartesian_bound')
        if np.max(np.abs(other-self.other)) > .005:
            return self.stop('carrier_moved', q)
        if not self.running:
            return self.target.copy()
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
            operator_press='cartesian-single-step')
        if candidate['reason'] not in ('nudging', 'aligned'):
            return self.stop(candidate['reason'], q)
        self.last_stamp = stamp
        self.samples.append(np.asarray(candidate['error_px']))
        if len(self.samples) < 3:
            return self.target.copy()
        error = np.median(self.samples, axis=0)
        self.error_px = error.tolist()
        if self.steps:
            change = error[[1, 3]]-self.before_error[[1, 3]]
            expected = self.proposal['slopes_px_per_m']*self.proposal['dz_m']
            moved = p[2]-self.xyz[2]
            if moved*self.proposal['dz_m'] <= 0 or abs(moved) < abs(self.proposal['dz_m'])*.25:
                reason = 'physical_step_not_confirmed'
            elif np.any(np.abs(error[[1, 3]]) > np.abs(self.before_error[[1, 3]])+2):
                reason = 'image_error_worsened'
            elif np.any(np.abs(expected) < 2) or np.linalg.norm(change-expected) > max(3., np.linalg.norm(expected)):
                reason = 'image_response_inconclusive'
            else:
                reason = 'single_step_observed_holding'
            return self.stop(reason, q)
        if np.max(np.abs(error[[1, 3]])) <= 3:
            return self.stop('aligned_holding', q)
        try:
            self.proposal = vertical_proposal(q, error,
                self.config['calibration']['image_jacobian_px_per_rad'], self.kinematics, self.limits)
        except ValueError as exc:
            return self.stop(str(exc), q)
        self.before_error = error.copy()
        self.target = self.proposal['target']
        self.steps = 1
        self.step_time = now
        self.settled_since = None
        self.samples.clear()
        self.reason = 'one_vertical_step'
        return self.target.copy()

    def status(self):
        return dict(active=True, correcting=self.running, reason=self.reason, steps=self.steps,
            target_position_rad=self.target.tolist(), error_px=self.error_px,
            step_norm_limit_rad=.003, total_norm_limit_rad=.003,
            tool_travel_limit_m=.001, tool_travel_m=self.travel_m, max_seconds=5,
            controller='experimental_cartesian_single_step', max_step_m=.0005,
            horizontal_drift_limit_m=.00025, orientation_limit_rad=.005,
            speed_limit_rad_s=.02, total_motor_torque_limits_nm=self.torque_limits.tolist())
