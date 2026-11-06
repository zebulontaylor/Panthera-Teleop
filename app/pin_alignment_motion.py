"""Pure bounded left-arm Cartesian proposal; never opens devices or sends motion.

Backend must independently preflight collision/path safety and supervise execution.
Left and right configs share the native follower URDF and .165m tool offset.
"""
import numpy as np
from pin_alignment_geometry import right_link6_transform
from pin_regrasp_motion import rotation_log, solve_pose

HARD_MAX_STEP_M = .005
# Physical path envelopes, not micrometre straight-line accuracy requirements.
MAX_PATH_DEVIATION_M = .00001
MAX_HEIGHT_ERROR_M = .00001
CARTESIAN_WAYPOINT_SPACING_M = .00025
MAX_PATH_ORIENTATION_RAD = .01
FK_TRAVEL_MARGIN_M = .0001


class NativeToolKinematics:
    """Six active native joints only; gripper is not an IK coordinate."""
    def pose(self, q):
        t = right_link6_transform(q)
        return t[:3,3] + t[:3,:3] @ np.array([.165,0,0]), t[:3,:3]

    def jacobian(self, q):
        q = np.asarray(q, float)
        p, r = self.pose(q)
        columns = []
        for axis in range(6):
            step = q.copy()
            step[axis] += 1e-6
            pp, rr = self.pose(step)
            columns.append(np.r_[(pp-p)/1e-6, rotation_log(rr @ r.T)/1e-6])
        return np.array(columns).T


def _vector(value, size, name):
    array = np.asarray(value, float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError('invalid_' + name)
    return array


def _check_jacobian(kinematics, q):
    j = np.asarray(kinematics.jacobian(q), float)
    if j.shape != (6,6) or not np.isfinite(j).all():
        raise ValueError('invalid_kinematic_jacobian')
    singular = np.linalg.svd(j, compute_uv=False)
    if singular[-1] < 1e-4 or singular[0]/singular[-1] > 1000:
        raise ValueError('kinematic_singularity')
    return float(singular[0]/singular[-1])


def plan_left_nudge(state14, delta_left_m, lower14, upper14, *,
                    max_step_m=.005, max_joint_delta_rad=.2, kinematics=None):
    """Return a bounded target14 or a refusal; input state is never modified.

    Limits include all 14 channels. Left grip and all right channels remain
    exactly measured. A backend may freeze them to its held command separately.
    Success checks dense Cartesian waypoints and their joint-linear segments;
    this is not collision certification. Nonzero vertical requests are refused.
    """
    out = dict(valid=False, reason='', metadata={})
    try:
        state = _vector(state14,14,'state14')
        delta = _vector(delta_left_m,3,'delta_left_m')
        lower = _vector(lower14,14,'lower14')
        upper = _vector(upper14,14,'upper14')
        if np.any(lower >= upper):
            raise ValueError('invalid_joint_limits')
        if np.any(state < lower) or np.any(state > upper):
            raise ValueError('measured_joint_limit')
        if (not np.isfinite([max_step_m,max_joint_delta_rad]).all()
                or not 0 < max_step_m <= HARD_MAX_STEP_M
                or not 0 < max_joint_delta_rad <= .2):
            raise ValueError('invalid_motion_bounds')
        distance = float(np.linalg.norm(delta))
        if distance < 1e-7:
            raise ValueError('nudge_too_small')
        if distance > max_step_m + 1e-10:
            raise ValueError('cartesian_step_exceeds_bound')
        kin = kinematics if kinematics is not None else NativeToolKinematics()
        q = state[:6].copy()
        condition = _check_jacobian(kin,q)
        position, rotation = kin.pose(q)
        if abs(delta[2]) > 1e-12:
            raise ValueError('alignment_requires_zero_vertical_delta')
        position = _vector(position,3,'kinematic_position')
        rotation = np.asarray(rotation,float)
        if rotation.shape != (3,3) or not np.isfinite(rotation).all():
            raise ValueError('invalid_kinematic_rotation')
        anchor_position = position.copy()
        anchor_rotation = rotation.copy()
        intervals = max(1, int(np.ceil(distance/CARTESIAN_WAYPOINT_SPACING_M)))
        waypoints = [state.copy()]
        previous_q = q.copy()
        condition_max = condition
        # Each target is an independently solved Cartesian XY point at the
        # captured native TCP height and orientation. Never interpolate a
        # single endpoint across the full nudge.
        for fraction in np.linspace(0, 1, intervals+1)[1:]:
            wanted = position + fraction*delta
            target_q = solve_pose(kin,previous_q,wanted,rotation,lower[:6],upper[:6])
            for _ in range(12):
                p,r = kin.pose(target_q)
                error = np.r_[wanted-p,rotation_log(rotation @ r.T)]
                if np.linalg.norm(error[:3]) < 1e-7 and np.linalg.norm(error[3:]) < 1e-6:
                    break
                j = np.asarray(kin.jacobian(target_q),float)
                target_q += j.T @ np.linalg.solve(j@j.T+np.eye(6)*1e-10,error)
            target_q = _vector(target_q,6,'solution')
            if np.linalg.norm(target_q-q) > max_joint_delta_rad:
                raise ValueError('joint_change_exceeds_bound')
            if np.any(target_q < lower[:6]) or np.any(target_q > upper[:6]):
                raise ValueError('target_joint_limit')
            condition_max = max(condition_max,_check_jacobian(kin,target_q))
            waypoint = state.copy()
            waypoint[:6] = target_q
            waypoints.append(waypoint)
            previous_q = target_q.copy()
        path_translation_error = path_orientation_error = max_height_error = 0.
        path_samples = 0
        # Check the actual piecewise joint-linear execution envelope between
        # dense Cartesian waypoints, including intermediate native TCP Z.
        for segment,(start,end) in enumerate(zip(waypoints,waypoints[1:])):
            for local_fraction in np.linspace(0,1,5):
                fraction = (segment+local_fraction)/intervals
                sample = start[:6]+local_fraction*(end[:6]-start[:6])
                p,r = kin.pose(sample)
                translation_error = float(np.linalg.norm(p-(position+fraction*delta)))
                orientation_error = float(np.linalg.norm(rotation_log(rotation @ r.T)))
                height_error = abs(float(p[2]-position[2]))
                if not np.isfinite([translation_error,orientation_error,height_error]).all():
                    raise ValueError('invalid_predicted_pose')
                path_translation_error = max(path_translation_error,translation_error)
                path_orientation_error = max(path_orientation_error,orientation_error)
                max_height_error = max(max_height_error,height_error)
                path_samples += 1
                if (translation_error > MAX_PATH_DEVIATION_M
                        or height_error > MAX_HEIGHT_ERROR_M
                        or orientation_error > MAX_PATH_ORIENTATION_RAD
                        or np.linalg.norm(p-position) > max_step_m+FK_TRAVEL_MARGIN_M):
                    raise ValueError('predicted_cartesian_path_violation')
        target = waypoints[-1]
        dq = target[:6]-q
        out.update(valid=True,reason='ok',target14=target.tolist(),waypoints14=[w.tolist() for w in waypoints],metadata=dict(
            anchor_position_m=anchor_position.tolist(),anchor_rotation=anchor_rotation.tolist(),
            height_locked=True,max_path_height_error_m=max_height_error,
            max_height_error_limit_m=MAX_HEIGHT_ERROR_M,
            cartesian_waypoint_spacing_m=CARTESIAN_WAYPOINT_SPACING_M,
            waypoint_count=len(waypoints),
            requested_delta_left_m=delta.tolist(), translation_error_m=translation_error,
            orientation_error_rad=orientation_error,
            max_path_translation_error_m=path_translation_error,
            max_path_orientation_error_rad=path_orientation_error,
            joint_delta_norm_rad=float(np.linalg.norm(dq)),
            max_jacobian_condition=condition_max, path_samples=path_samples,
            max_path_deviation_limit_m=MAX_PATH_DEVIATION_M,
            max_path_orientation_limit_rad=MAX_PATH_ORIENTATION_RAD,
            requires_backend_path_validation=True))
    except (ValueError,TypeError,np.linalg.LinAlgError) as exc:
        out['reason'] = str(exc)
    return out
