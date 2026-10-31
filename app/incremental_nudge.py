"""Bounded incremental height feedback using the recorded local image Jacobian.

No device access: each tick returns a joint position target to the host.
"""
from collections import deque
from itertools import product
import numpy as np

from alignment_assist import AlignmentAssist


from cartesian_nudge import rotation_error


SDK_TWO_PI = np.float32(6.28318530717)
SDK_POSITION_QUANTUM = float(SDK_TWO_PI)/10000.


def nudge_bounds(profile='strict', *, hardware=False):
    """Named, finite trial envelopes; arbitrary runtime limit overrides are unsupported."""
    if profile == 'strict':
        return dict(max_step_m=.0005 if hardware else .00025,
                    joint_step_rad=.003, joint_total_rad=.012, travel_m=.002,
                    horizontal_m=.00025, orientation_rad=.005,
                    endpoint_orientation_rad=.001, ideal_path_error_m=.00005,
                    excursion_padding_m=.000025, tracking_rad=.01,
                    max_steps=20, max_seconds=30, step_timeout_seconds=3, settle_ticks=0)
    if profile == 'supervised_relaxed':
        return dict(max_step_m=.005, travel_m=.05, speed_rad_s=.02)
    raise ValueError('unknown_incremental_profile')


def evaluate_incremental_evidence(config, now, snapshot, arms, joint_limits):
    """Fresh, finite, correctly identified image evidence; no experimental motion vetoes."""
    try:
        if not np.isfinite(now):
            raise ValueError('invalid_timestamp')
        cal = config['calibration']
        if config.get('task') != 'gear' or cal['side'] != 'left':
            raise ValueError('supervised_gear_only')
        cameras = cal['camera_names']
        if len(cameras) != 2 or set(cameras) != {'left_wrist', 'right_wrist'}:
            raise ValueError('invalid_camera_pair')
        mapping = cal['camera_mapping']
        if len(set(mapping[name] for name in cameras)) != 2:
            raise ValueError('invalid_camera_mapping')
        if snapshot.get('status') != 'ready':
            raise ValueError(snapshot.get('status', 'no_prediction'))
        if any(snapshot['camera_mapping'].get(name) != mapping[name] for name in cameras):
            raise ValueError('camera_mapping_mismatch')
        times = np.asarray([snapshot['monotonic_ns']]+[
            snapshot['frame_timestamps_ns'][mapping[name]] for name in cameras], dtype=float)/1e9
        age_limit = min(.25, float(config.get('max_age_seconds', .2)))
        if not np.isfinite(times).all() or np.any(now-times < -1e-9) or np.any(now-times > age_limit):
            raise ValueError('stale_prediction')
        if np.ptp(times[1:]) > min(.1, float(config.get('max_camera_skew_seconds', .06))):
            raise ValueError('unsynchronized_cameras')
        q = np.asarray(arms['left']['position_rad'], dtype=float)
        velocity = np.asarray(arms['left']['velocity_rad_s'], dtype=float)
        if q.shape != (6,) or velocity.shape != (6,) or not np.isfinite(np.r_[q, velocity]).all():
            raise ValueError('invalid_feedback')
        if np.any(q < np.asarray(joint_limits['lower'])) or np.any(q > np.asarray(joint_limits['upper'])):
            raise ValueError('joint_absolute_limit')
        jacobian = np.asarray(cal['image_jacobian_px_per_rad'], dtype=float)
        if jacobian.shape != (4, 6) or not np.isfinite(jacobian).all():
            raise ValueError('invalid_image_jacobian')
        prediction = snapshot['predictions']['gear']
        if not cal['checkpoint_id'] or prediction['checkpoint_id'] != cal['checkpoint_id']:
            raise ValueError('checkpoint_mismatch')
        if list(prediction['image_size']) != list(cal['image_size']):
            raise ValueError('image_size_mismatch')
        names = list(prediction['landmark_names'])
        moving, target = names.index('gear_center'), names.index('slot_center')
        if cal['target_landmark'] != 'slot_center':
            raise ValueError('invalid_target_landmark')
        desired = np.asarray(cal['target_error_px'], dtype=float)
        if desired.shape != (4,) or not np.isfinite(desired).all():
            raise ValueError('invalid_target_error')
        points = np.asarray(prediction['keypoints_2d'], dtype=float)
        errors = []
        for camera in cameras:
            index = list(prediction['camera_names']).index(camera)
            selected = points[index, [moving, target]]
            if selected.shape != (2, 2) or not np.isfinite(selected).all():
                raise ValueError('invalid_landmarks')
            if np.any(selected < 0) or np.any(selected >= cal['image_size']):
                raise ValueError('landmarks_out_of_frame')
            offset = len(errors)
            errors.extend(selected[0]-selected[1]-desired[offset:offset+2])
        reason = 'aligned' if np.max(np.abs(np.asarray(errors)[[1, 3]])) <= 3 else 'nudging'
        return dict(active=reason == 'nudging', reason=reason, error_px=np.asarray(errors).tolist())
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else 'invalid_evidence_or_calibration'
        return dict(active=False, reason=reason)


def distance_speed_proposal(q, error, image_jacobian, kinematics, limits):
    """Local proportional Z request; camera disagreements are handled by least squares."""
    q = np.asarray(q, dtype=float)
    jacobian = np.asarray(kinematics.jacobian(q), dtype=float)
    if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
        raise ValueError('invalid_kinematic_jacobian')
    direction = np.linalg.lstsq(jacobian, [0., 0., 1., 0., 0., 0.], rcond=1e-8)[0]
    slopes = np.asarray(image_jacobian, dtype=float)[[1, 3]] @ direction
    residual = np.asarray(error, dtype=float)[[1, 3]]
    denominator = float(slopes @ slopes)
    if not np.isfinite(denominator) or denominator < 1e-12:
        raise ValueError('height_response_unobservable')
    dz = float(np.clip(-.4*(slopes @ residual)/denominator, -.005, .005))
    if abs(dz) < 1e-15:
        raise ValueError('height_request_cancels')
    return dict(target=q+direction*dz, dz_m=dz, slopes_px_per_m=slopes,
                predicted_height_error_px=residual+slopes*dz)


def distance_speed_quantized_proposal(proposal, q, error, image_jacobian, kinematics, limits):
    """Select a representable correction constrained only by joint and 3D distances."""
    q = np.asarray(q, dtype=float)
    base_ticks = np.rint(q/SDK_POSITION_QUANTUM).astype(int)
    delta = np.asarray(proposal['target'])-q
    choices = set()
    for scale in (1., .8, .6, .4, .2):
        desired = (q+scale*delta)/SDK_POSITION_QUANTUM
        choices.update(product(*[tuple(set((int(np.floor(v)), int(np.ceil(v))))) for v in desired]))
    for joint in range(6):
        if abs(delta[joint]) > 1e-15:
            ticks = base_ticks.copy()
            ticks[joint] += int(np.sign(delta[joint]))
            choices.add(tuple(ticks))
    candidates = []
    image_j = np.asarray(image_jacobian)[[1, 3]]
    residual = np.asarray(error)[[1, 3]]
    origin, _ = kinematics.pose(q)
    for ticks in choices:
        step_ticks = np.asarray(ticks)-base_ticks
        if not np.any(step_ticks):
            continue
        target, command = sdk_grid_target(np.asarray(ticks)*SDK_POSITION_QUANTUM)
        if np.any(target < np.asarray(limits['lower'])) or np.any(target > np.asarray(limits['upper'])):
            continue
        endpoint, _ = kinematics.pose(target)
        dz = float(endpoint[2]-origin[2])
        if dz*proposal['dz_m'] <= 0 or np.linalg.norm(endpoint-origin) > .005+1e-12:
            continue
        change = image_j @ (target-q)
        candidates.append((float(np.linalg.norm(residual+change)), target, command, dz, change, step_ticks))
    for _, target, command, dz, change, step_ticks in sorted(candidates, key=lambda item: item[0]):
        corners = [kinematics.pose(q+(target-q)*np.asarray(mask)) for mask in product((0., 1.), repeat=6)]
        if any(np.linalg.norm(position-origin) > .005+1e-12 for position, _ in corners):
            continue
        return dict(proposal, target=target, command=command, dz_m=dz,
                    expected_change_px=change, predicted_height_error_px=residual+change,
                    completion_corners=corners, encoder_step_ticks=step_ticks.tolist())
    raise ValueError('motor_resolution_limited')


def sdk_position_ticks(command):
    """Mirror native float division then implicit int32 truncation."""
    return np.trunc((np.asarray(command, dtype=np.float32)/SDK_TWO_PI).astype(float)*10000).astype(np.int64)


def sdk_grid_target(target):
    """Return decoded motor endpoint and a command safely inside its wire bin.

    Sending decoded feedback back directly can truncate to the adjacent tick.
    The quarter-bin command bias affects encoding only, not the motor endpoint.
    """
    ticks = np.rint(np.asarray(target)/SDK_POSITION_QUANTUM).astype(np.int64)
    if np.any(np.abs(ticks) >= 32700):
        raise ValueError('motor_position_encoding_limit')
    command = (ticks+.25*np.sign(ticks))*SDK_POSITION_QUANTUM
    if not np.array_equal(sdk_position_ticks(command), ticks):
        raise ValueError('motor_position_encoding_mismatch')
    decoded = (ticks.astype(np.float32)*SDK_TWO_PI/np.float64(10000.)).astype(np.float32).astype(float)
    return decoded, command


def image_step_allowed(residual, change, step_ticks, profile):
    active = np.abs(residual) > 3
    if np.any(change[active]*residual[active] >= 0):
        return False
    predicted = residual+change
    ordinary = (not np.any(np.abs(predicted[active]) >= np.abs(residual[active]))
                and not np.any(np.abs(predicted[~active]) > 3))
    one_tick_near_target = (profile == 'supervised_relaxed'
                           and np.max(np.abs(step_ticks)) == 1
                           and np.max(np.abs(residual)) <= 6
                           and np.max(np.abs(predicted)) <= 6)
    return ordinary or one_tick_near_target


def _quantized_candidate(proposal, q, error, image_jacobian, kinematics, limits, *, profile='strict'):
    bounds = nudge_bounds(profile, hardware=True)
    target, command = sdk_grid_target(proposal['target'])
    delta = target-q
    origin, rotation = kinematics.pose(q)
    endpoint, end_rotation = kinematics.pose(target)
    dz = float(endpoint[2]-origin[2])
    if (dz*proposal['dz_m'] <= 0 or abs(dz) < .000001
            or abs(dz) > bounds['max_step_m'] or np.linalg.norm(delta) > bounds['joint_step_rad']
            or np.linalg.norm((endpoint-origin)[:2]) > bounds['horizontal_m']
            or rotation_error(end_rotation, rotation) > bounds['endpoint_orientation_rad']):
        raise ValueError('motor_resolution_limited')
    if (np.any(target <= np.asarray(limits['lower'])+.03)
            or np.any(target >= np.asarray(limits['upper'])-.03)):
        raise ValueError('joint_target_limit')
    # Joint_Pos_Vel drives joints independently. Check all completion corners,
    # rather than assuming synchronous joint interpolation.
    corners = []
    for completed in product((0., 1.), repeat=6):
        corner_q = q+delta*np.asarray(completed)
        corner_p, corner_r = kinematics.pose(corner_q)
        if (np.linalg.norm((corner_p-origin)[:2]) > bounds['horizontal_m']
                or abs(corner_p[2]-origin[2]) > bounds['max_step_m']
                or np.linalg.norm(corner_p-origin) > abs(dz)*1.5+bounds['excursion_padding_m']
                or rotation_error(corner_r, rotation) > bounds['orientation_rad']):
            raise ValueError('motor_resolution_limited')
        corners.append((corner_p, corner_r))
    change = np.asarray(image_jacobian)[[1, 3]] @ delta
    residual = np.asarray(error)[[1, 3]]
    active = np.abs(residual) > 3
    step_ticks = np.rint(target/SDK_POSITION_QUANTUM)-np.rint(q/SDK_POSITION_QUANTUM)
    if not np.any(step_ticks) or not image_step_allowed(residual, change, step_ticks, profile):
        raise ValueError('motor_resolution_limited')
    return dict(proposal, target=target, command=command, dz_m=dz,
                expected_change_px=change, predicted_height_error_px=residual+change,
                completion_corners=corners, encoder_step_ticks=step_ticks.tolist())


def quantized_proposal(proposal, q, error, image_jacobian, kinematics, limits, *, profile='strict'):
    """Search adjacent representable endpoints within the existing motion bounds.

    Rounding a continuous target at the joint-norm ceiling often exceeds that
    ceiling. Include floor/ceiling lattice neighbors of shorter proposals, then
    rank by predicted visual improvement before checking physical geometry.
    """
    bounds = nudge_bounds(profile, hardware=True)
    if profile == 'supervised_relaxed':
        return distance_speed_quantized_proposal(proposal, q, error, image_jacobian, kinematics, limits)
    q = np.asarray(q, dtype=float)
    delta = np.asarray(proposal['target'])-q
    residual = np.asarray(error)[[1, 3]]
    image_j = np.asarray(image_jacobian)[[1, 3]]
    candidates = {}
    tick_options = set()
    base_ticks = np.rint(q/SDK_POSITION_QUANTUM).astype(int)
    for scale in (1., .8, .6, .4, .2):
        tick_target = (q+scale*delta)/SDK_POSITION_QUANTUM
        choices = [tuple(set((int(np.floor(v)), int(np.ceil(v))))) for v in tick_target]
        tick_options.update(product(*choices))
    if profile == 'supervised_relaxed':
        # Explicit minimum actuator moves, including when proportional motion is
        # too small to survive rounding. Every candidate still passes geometry.
        for joint in range(6):
            if abs(delta[joint]) > 1e-15:
                ticks = base_ticks.copy()
                ticks[joint] += int(np.sign(delta[joint]))
                tick_options.add(tuple(ticks))
    for ticks in tick_options:
        step_ticks = np.asarray(ticks)-base_ticks
        if not np.any(step_ticks):
            continue
        endpoint, _ = sdk_grid_target(np.asarray(ticks)*SDK_POSITION_QUANTUM)
        step = endpoint-q
        if np.linalg.norm(step) > bounds['joint_step_rad'] or np.linalg.norm(step) < 1e-8:
            continue
        change = image_j @ step
        if not image_step_allowed(residual, change, step_ticks, profile):
            continue
        predicted = residual+change
        candidates[ticks] = (float(np.linalg.norm(predicted)), endpoint)
    for _, endpoint in sorted(candidates.values(), key=lambda item: item[0]):
        try:
            return _quantized_candidate(dict(proposal, target=endpoint), q, error,
                                        image_jacobian, kinematics, limits, profile=profile)
        except ValueError:
            continue
    raise ValueError('motor_resolution_limited')


def incremental_proposal(q, error, image_jacobian, kinematics, limits, *, max_step_m=None, profile='strict'):
    """Constrain all six tool DOFs; reject infeasible or conflicting proposals."""
    if profile == 'supervised_relaxed':
        return distance_speed_proposal(q, error, image_jacobian, kinematics, limits)
    bounds = nudge_bounds(profile)
    max_step_m = bounds['max_step_m'] if max_step_m is None else max_step_m
    if not np.isfinite(max_step_m) or not 0 < max_step_m <= nudge_bounds(profile, hardware=True)['max_step_m']:
        raise ValueError('invalid_step_limit')
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
    active = np.abs(error) > 3
    requested = -error[active]/slopes[active]
    if not np.any(active):
        raise ValueError('aligned_holding')
    if np.min(requested)*np.max(requested) <= 0:
        raise ValueError('camera_height_direction_disagreement')
    dz = float(np.clip(-.4*np.dot(slopes[active], error[active])/
                       np.dot(slopes[active], slopes[active]), -max_step_m, max_step_m))
    for residual, slope in zip(error[~active], slopes[~active]):
        predicted = residual+slope*dz
        if abs(predicted) > 3:
            dz *= max(0., (np.sign(predicted)*3-residual)/(slope*dz))
    delta = direction*dz
    scale = min(1., bounds['joint_step_rad']/max(np.linalg.norm(delta), 1e-12))
    delta *= scale
    dz *= scale
    if abs(dz) < (.000001 if profile == 'strict' else 1e-15):
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
        if np.linalg.norm(p-expected) > bounds['ideal_path_error_m'] or rotation_error(r, rotation) > bounds['endpoint_orientation_rad']:
            raise ValueError('predicted_path_not_vertical')
    return dict(target=target, dz_m=dz, slopes_px_per_m=slopes,
                predicted_height_error_px=error+slopes*dz)


class IncrementalNudge:
    """Repeat settled observations and conservative corrections, then hold."""
    def __init__(self, config, now, arms, joint_limits, torque_limits, *, kinematics):
        self.config = dict(config, servo_travel_bounded=False)
        self.kinematics = kinematics
        self.limits = joint_limits
        self.encoder_resolution = config.get('encoder_resolution_rad')
        self.profile = config.get('incremental_profile', 'strict')
        self.bounds = nudge_bounds(self.profile, hardware=self.encoder_resolution is not None)
        if self.encoder_resolution is not None and not np.isclose(
                self.encoder_resolution, SDK_POSITION_QUANTUM, rtol=1e-6, atol=0):
            raise ValueError('unsupported_motor_resolution')
        self.stable_q = None
        self.initial = np.asarray(arms['left']['position_rad'], dtype=float)
        self.target = self.initial.copy()
        self.xyz, self.rotation = kinematics.pose(self.initial)
        self.other = np.asarray(arms['right']['position_rad'], dtype=float)
        self.gripper_position = float(arms['left']['gripper']['position'])
        self.torque_limits = np.minimum(np.asarray(torque_limits),
            np.abs(arms['left']['gravity_model_torque_nm'])+.3)
        self.started = now
        self.reason = 'waiting_for_fresh_evidence'
        self.wait_stage = 'waiting_for_stable_encoders'
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
        self.step_origin = self.xyz.copy()
        self.path_m = 0.
        self.joint_path_rad = 0.
        self.previous_q = self.initial.copy()
        self.previous_p = self.xyz.copy()
        self.commanded_path_m = 0.
        self.commanded_joint_path_rad = 0.
        self.no_progress = 0
        self.last_camera_stamps = None
        self.response_accounted = False
        self.response_rechecks = 0
        self.next_observation_at = now
        # The supplied profile must retain the actual local calibration gates.
        if (config.get('task') != 'gear' or config.get('mode') != 'operator_supervised'
                or config.get('calibration', {}).get('side') != 'left'):
            raise ValueError('supervised_gear_only')

    def command(self):
        # Observe in gravity assist until there is an actual visual correction.
        if not self.steps:
            return None
        if self.encoder_resolution is not None:
            return sdk_grid_target(self.target)[1]
        return self.target.copy()

    def stop(self, reason, q):
        self.reason = reason
        self.running = False
        self.target = q.copy()
        return self.command()

    def tick(self, now, snapshot, arms, *, allowed):
        if self.profile == 'supervised_relaxed':
            return self.tick_distance_speed(now, snapshot, arms, allowed=allowed)
        if not allowed:
            raise ValueError('collection_or_grasp_ended')
        bounds = self.bounds
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
        if np.linalg.norm(q-self.initial) > bounds['joint_total_rad']:
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
        if np.max(np.abs(q-self.target)) > bounds['tracking_rad']:
            raise ValueError('tracking_deviation')
        if (np.linalg.norm((p-self.xyz)[:2]) > bounds['horizontal_m']
                or rotation_error(r, self.rotation) > bounds['orientation_rad'] or self.travel_m > bounds['travel_m'] or self.path_m > bounds['travel_m']):
            raise ValueError('measured_cartesian_bound')
        if (self.running and self.proposal is not None
                and np.linalg.norm(p-self.step_origin) > abs(self.proposal['dz_m'])*1.5+bounds['excursion_padding_m']):
            raise ValueError('measured_step_excursion_limit')
        if np.max(np.abs(other-self.other)) > .005:
            return self.stop('carrier_moved', q)
        if not self.running:
            return self.command()
        if now-self.started > bounds['max_seconds']:
            return self.stop('trial_timeout', q)
        target_p, target_r = self.kinematics.pose(self.target)
        position_tolerance = (min(.000025, abs(self.proposal['dz_m'])*.2)
                              if self.proposal is not None else .000025)
        at_target = (np.linalg.norm(p-target_p) <= position_tolerance
                     and rotation_error(r, target_r) <= bounds['endpoint_orientation_rad']
                     and np.max(np.abs(q-self.target)) <= .0002)
        speed_tolerance = .002
        if self.encoder_resolution is not None:
            current_ticks = np.rint(q/SDK_POSITION_QUANTUM)
            target_ticks = np.rint(self.target/SDK_POSITION_QUANTUM)
            at_target = self.steps == 0 or np.max(np.abs(current_ticks-target_ticks)) <= bounds['settle_ticks']
            if self.steps:
                measured_z = p[2]-self.step_origin[2]
                commanded_z = self.proposal['dz_m']
                at_target = (at_target and measured_z*commanded_z > 0
                             and abs(measured_z) >= abs(commanded_z)*.25)
            if self.stable_q is None or not np.array_equal(current_ticks, self.stable_q):
                self.settled_since = None
                self.samples.clear()
            self.stable_q = current_ticks
            # Firmware velocity can report motion while encoder counts remain
            # unchanged for many seconds. Require stable measured counts, not a
            # contradictory noisy velocity estimate. Finiteness is checked above.
            speed_tolerance = float('inf')
        if not at_target or np.max(np.abs(velocity)) > speed_tolerance:
            self.wait_stage = 'waiting_for_stable_encoders'
            self.settled_since = None
            self.samples.clear()
            if self.step_time is not None and now-self.step_time > bounds['step_timeout_seconds']:
                return self.stop('physical_step_not_confirmed', q)
            return self.command()
        if self.settled_since is None:
            self.settled_since = now
        if now-self.settled_since < .15:
            self.wait_stage = 'waiting_for_stable_encoders'
            return self.command()
        self.wait_stage = 'rechecking_image_response' if self.response_rechecks else 'waiting_for_fresh_camera_frames'
        stamp = snapshot.get('monotonic_ns')
        if stamp is None or (self.last_stamp is not None and stamp <= self.last_stamp) or stamp/1e9 < self.settled_since:
            return self.command()
        camera_stamps = tuple(snapshot.get('frame_timestamps_ns', {}).get(
            self.config['calibration']['camera_mapping'][name], -1)
            for name in self.config['calibration']['camera_names'])
        if (any(t/1e9 < self.settled_since for t in camera_stamps)
                or (self.last_camera_stamps is not None and
                    any(t <= old for t, old in zip(camera_stamps, self.last_camera_stamps)))):
            return self.command()
        candidate = AlignmentAssist(self.config).evaluate(now, snapshot, arm, self.limits,
            other_arm=arms['right'], collecting=True, operator_confirmed=True,
            operator_press='incremental-height')
        if candidate['reason'] not in ('nudging', 'aligned'):
            return self.stop(candidate['reason'], q)
        self.last_stamp = stamp
        self.last_camera_stamps = camera_stamps
        self.wait_stage = 'rechecking_image_response' if self.response_rechecks else 'collecting_visual_evidence'
        self.samples.append((min(camera_stamps)/1e9, np.asarray(candidate['error_px'])))
        # Each observation passed the individual age gate on arrival. Permit
        # transport latency plus three observations, but never multi-second reuse.
        window = .5
        while self.samples and now-self.samples[0][0] > window:
            self.samples.popleft()
        if len(self.samples) < 3:
            return self.command()
        error = np.median([sample[1] for sample in self.samples], axis=0)
        self.error_px = error.tolist()
        if self.proposal is not None:
            # Reobserving image response must not charge the same endpoint twice.
            if not self.response_accounted:
                self.path_m += float(np.linalg.norm(p-self.previous_p))
                self.joint_path_rad += float(np.linalg.norm(q-self.previous_q))
                self.previous_p, self.previous_q = p.copy(), q.copy()
                self.response_accounted = True
            if self.path_m > bounds['travel_m'] or self.joint_path_rad > bounds['joint_total_rad']:
                return self.stop('measured_path_limit', q)
            change = error[[1, 3]]-self.before_error[[1, 3]]
            expected = self.proposal.get('expected_change_px',
                self.proposal['slopes_px_per_m']*self.proposal['dz_m'])
            moved = p[2]-self.step_origin[2]
            if moved*self.proposal['dz_m'] <= 0 or abs(moved) < abs(self.proposal['dz_m'])*.25:
                return self.stop('physical_step_not_confirmed', q)
            if self.profile == 'supervised_relaxed' and np.max(np.abs(error[[1, 3]])) <= 3:
                return self.stop('aligned_holding', q)
            before_y = self.before_error[[1, 3]]
            after_y = error[[1, 3]]
            near_target = (self.profile == 'supervised_relaxed'
                           and np.max(np.abs(self.proposal.get('encoder_step_ticks', [2]))) <= 1
                           and np.max(np.abs(before_y)) <= 6 and np.max(np.abs(after_y)) <= 6)
            bad_direction = np.any((change*expected < 0) & (np.abs(change) > .5)
                                   & ((np.abs(before_y) > 3) if self.profile == 'supervised_relaxed' else True))
            worsened = np.any(np.abs(after_y) > np.abs(before_y)+1.) and not near_target
            if bad_direction or worsened:
                if self.profile == 'supervised_relaxed' and self.response_rechecks < 1:
                    self.response_rechecks += 1
                    self.samples.clear()
                    self.wait_stage = 'rechecking_image_response'
                    return self.command()
                return self.stop('image_response_wrong_direction' if bad_direction else 'image_error_worsened', q)
            improvement = np.linalg.norm(before_y)-np.linalg.norm(after_y)
            self.no_progress = self.no_progress+1 if improvement < .15 and not near_target else 0
            if self.no_progress >= 2:
                return self.stop('image_response_no_progress', q)
        if np.max(np.abs(error[[1, 3]])) <= 3:
            return self.stop('aligned_holding', q)
        if self.steps >= bounds['max_steps']:
            return self.stop('step_budget_exhausted', q)
        try:
            self.proposal = incremental_proposal(q, error,
                self.config['calibration']['image_jacobian_px_per_rad'], self.kinematics, self.limits,
                max_step_m=bounds['max_step_m'], profile=self.profile)
            if self.encoder_resolution is not None:
                self.proposal = quantized_proposal(self.proposal, q, error,
                    self.config['calibration']['image_jacobian_px_per_rad'], self.kinematics, self.limits,
                    profile=self.profile)
        except ValueError as exc:
            return self.stop(str(exc), q)
        calibration = self.config['calibration']
        if np.max(np.abs(self.proposal['target']-np.asarray(
                calibration['reference_joint_position_rad']))) > calibration['max_joint_distance_rad']:
            return self.stop('outside_calibration_envelope', q)
        delta = self.proposal['target']-q
        target_p, target_r = self.kinematics.pose(self.proposal['target'])
        step_distance = float(np.linalg.norm(target_p-p))
        next_path = self.commanded_path_m+step_distance
        next_joint_path = self.commanded_joint_path_rad+float(np.linalg.norm(delta))
        if (next_path > bounds['travel_m'] or next_joint_path > bounds['joint_total_rad']
                or self.path_m+step_distance > bounds['travel_m']
                or self.joint_path_rad+float(np.linalg.norm(delta)) > bounds['joint_total_rad']
                or np.linalg.norm(self.proposal['target']-self.initial) > bounds['joint_total_rad']
                or np.linalg.norm(target_p-self.xyz) > bounds['travel_m']
                or np.linalg.norm((target_p-self.xyz)[:2]) > bounds['horizontal_m']
                or rotation_error(target_r, self.rotation) > bounds['orientation_rad']):
            return self.stop('cumulative_travel_limit', q)
        for corner_p, corner_r in self.proposal.get('completion_corners', []):
            if (np.linalg.norm((corner_p-self.xyz)[:2]) > bounds['horizontal_m']
                    or np.linalg.norm(corner_p-self.xyz) > bounds['travel_m']
                    or rotation_error(corner_r, self.rotation) > bounds['orientation_rad']):
                return self.stop('cumulative_travel_limit', q)
        self.commanded_path_m = next_path
        self.commanded_joint_path_rad = next_joint_path
        self.step_origin = p.copy()
        self.response_accounted = False
        self.response_rechecks = 0
        self.before_error = error.copy()
        self.target = self.proposal['target']
        self.steps += 1
        self.step_time = now
        self.settled_since = None
        self.samples.clear()
        self.reason = 'incremental_vertical_step'
        self.wait_stage = 'waiting_for_stable_encoders'
        return self.command()

    def tick_distance_speed(self, now, snapshot, arms, *, allowed):
        if not allowed:
            raise ValueError('collection_or_grasp_ended')
        arm = arms['left']
        q = np.asarray(arm['position_rad'], dtype=float)
        velocity = np.asarray(arm['velocity_rad_s'], dtype=float)
        if q.shape != (6,) or velocity.shape != (6,) or not np.isfinite(np.r_[q, velocity]).all():
            raise ValueError('invalid_feedback')
        if np.any(q < np.asarray(self.limits['lower'])) or np.any(q > np.asarray(self.limits['upper'])):
            raise ValueError('joint_absolute_limit')
        p, rotation = self.kinematics.pose(q)
        if not np.isfinite(p).all() or not np.isfinite(rotation).all():
            raise ValueError('invalid_tool_feedback')
        self.travel_m = float(np.linalg.norm(p-self.xyz))
        if self.travel_m > .05:
            raise ValueError('measured_cartesian_bound')
        if self.steps and np.linalg.norm(p-self.step_origin) > .005+1e-12:
            raise ValueError('measured_step_distance_limit')
        if not self.running:
            return self.command()
        if now < self.next_observation_at:
            self.wait_stage = 'waiting_for_command_cadence'
            return self.command()
        self.wait_stage = 'waiting_for_fresh_camera_frames'
        evidence = evaluate_incremental_evidence(self.config, now, snapshot, arms, self.limits)
        if evidence['reason'] not in ('aligned', 'nudging'):
            self.wait_stage = evidence['reason']
            return self.command()
        stamp = snapshot['monotonic_ns']
        cal = self.config['calibration']
        camera_stamps = tuple(snapshot['frame_timestamps_ns'][cal['camera_mapping'][name]]
                              for name in cal['camera_names'])
        if (min(camera_stamps)/1e9 < self.next_observation_at
                or (self.last_stamp is not None and stamp <= self.last_stamp)
                or (self.last_camera_stamps is not None and
                    any(t <= old for t, old in zip(camera_stamps, self.last_camera_stamps)))):
            return self.command()
        self.last_stamp, self.last_camera_stamps = stamp, camera_stamps
        self.wait_stage = 'collecting_visual_evidence'
        self.samples.append((min(camera_stamps)/1e9, np.asarray(evidence['error_px'])))
        while self.samples and now-self.samples[0][0] > .5:
            self.samples.popleft()
        if len(self.samples) < 3:
            return self.command()
        error = np.median([item[1] for item in self.samples], axis=0)
        self.error_px = error.tolist()
        self.path_m += float(np.linalg.norm(p-self.previous_p))
        self.previous_p, self.previous_q = p.copy(), q.copy()
        if self.path_m > .05:
            return self.stop('measured_path_limit', q)
        if np.max(np.abs(error[[1, 3]])) <= 3:
            return self.stop('aligned_holding', q)
        try:
            proposal = distance_speed_proposal(q, error, cal['image_jacobian_px_per_rad'],
                                               self.kinematics, self.limits)
            if self.encoder_resolution is not None:
                proposal = distance_speed_quantized_proposal(proposal, q, error,
                    cal['image_jacobian_px_per_rad'], self.kinematics, self.limits)
            else:
                # Test/simulation path uses the same geometric checks without wire quantization.
                target = proposal['target']
                if np.any(target < np.asarray(self.limits['lower'])) or np.any(target > np.asarray(self.limits['upper'])):
                    raise ValueError('joint_absolute_limit')
                proposal['completion_corners'] = [self.kinematics.pose(q+(target-q)*np.asarray(mask))
                                                  for mask in product((0., 1.), repeat=6)]
                if any(np.linalg.norm(position-p) > .005+1e-12 for position, _ in proposal['completion_corners']):
                    raise ValueError('step_distance_limit')
        except (ValueError, np.linalg.LinAlgError) as exc:
            return self.stop(str(exc), q)
        target_p, _ = self.kinematics.pose(proposal['target'])
        distance = float(np.linalg.norm(target_p-p))
        if (distance > .005+1e-12 or self.commanded_path_m+distance > .05
                or self.path_m+distance > .05
                or any(np.linalg.norm(position-self.xyz) > .05
                       for position, _ in proposal['completion_corners'])):
            return self.stop('cumulative_travel_limit', q)
        self.proposal = proposal
        self.target = proposal['target'].copy()
        self.commanded_path_m += distance
        self.step_origin = p.copy()
        self.steps += 1
        self.step_time = now
        self.next_observation_at = now+float(np.max(np.abs(self.target-q)))/.02+.2
        self.samples.clear()
        self.reason = 'incremental_vertical_step'
        self.wait_stage = 'waiting_for_command_cadence'
        return self.command()

    def status(self):
        if self.profile == 'supervised_relaxed':
            return dict(active=True, correcting=self.running, reason=self.reason, steps=self.steps,
                motor_command_active=self.steps > 0, holding=self.steps > 0 and not self.running,
                encoder_resolution_rad=self.encoder_resolution, incremental_profile=self.profile,
                control_policy='distance_speed_only', wait_stage=self.wait_stage,
                visual_sample_count=len(self.samples), visual_sample_window_seconds=.5,
                target_position_rad=self.target.tolist(), error_px=self.error_px,
                tool_travel_limit_m=.05, tool_travel_m=self.travel_m, tool_path_m=self.path_m,
                commanded_path_m=self.commanded_path_m, max_total_path_m=.05, max_step_m=.005,
                speed_limit_rad_s=.02, next_observation_monotonic=self.next_observation_at,
                tolerance_px=3, step_norm_limit_rad=None, total_norm_limit_rad=None,
                max_steps=None, max_seconds=None, max_duration_s=None,
                horizontal_drift_limit_m=None, orientation_limit_rad=None,
                minimum_encoder_step_ticks=1 if self.encoder_resolution is not None else 0,
                controller='incremental_vertical_feedback',
                total_motor_torque_limits_nm=self.torque_limits.tolist())
        bounds = self.bounds
        return dict(active=True, correcting=self.running, reason=self.reason, steps=self.steps,
            motor_command_active=self.steps > 0, holding=self.steps > 0 and not self.running,
            encoder_resolution_rad=self.encoder_resolution,
            wait_stage=self.wait_stage, visual_sample_count=len(self.samples),
            visual_sample_window_seconds=.5,
            target_position_rad=self.target.tolist(), error_px=self.error_px,
            incremental_profile=self.profile,
            minimum_encoder_step_ticks=1 if self.profile == 'supervised_relaxed' and self.encoder_resolution is not None else 0,
            near_target_oscillation_px=6 if self.profile == 'supervised_relaxed' else 3,
            image_response_rechecks=self.response_rechecks,
            step_norm_limit_rad=bounds['joint_step_rad'], total_norm_limit_rad=bounds['joint_total_rad'],
            tool_travel_limit_m=bounds['travel_m'], tool_travel_m=self.travel_m, max_seconds=bounds['max_seconds'],
            controller='incremental_vertical_feedback',
            max_step_m=bounds['max_step_m'],
            max_steps=bounds['max_steps'], tool_path_m=self.path_m, joint_path_rad=self.joint_path_rad,
            max_total_path_m=bounds['travel_m'], max_duration_s=bounds['max_seconds'], tolerance_px=3,
            horizontal_drift_limit_m=bounds['horizontal_m'], orientation_limit_rad=bounds['orientation_rad'],
            endpoint_orientation_limit_rad=bounds['endpoint_orientation_rad'],
            settled_encoder_tolerance_ticks=bounds['settle_ticks'],
            speed_limit_rad_s=.02, total_motor_torque_limits_nm=self.torque_limits.tolist())
