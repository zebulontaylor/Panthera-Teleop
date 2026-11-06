"""Conditional image-to-face preview estimates. No device or motion interfaces.

The profile is a registration hypothesis, not validated metric calibration.
Positive deltas move the left pin toward the hole, parallel to the carrier face.
"""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np

PROFILE_DIR = Path(__file__).resolve().parents[1] / 'config/pin-insertion-photo'
GEOMETRY_DEFAULTS = dict(gain=0.5, max_step_m=0.001, deadband_px=3.0,
                         frame_yaw_deg=0.0)


def _rotation(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)


def right_link6_transform(q):
    """Native joint order; origins/axes from configured follower URDF."""
    q = np.asarray(q, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('invalid_right_joints')
    origins = [[0,0,.0584],[.018199,0,.053],[-.26,0,0],
               [.23,0,.06],[.07,0,.036319],[.02345,0,-.039]]
    axes = [[0,0,1],[0,1,0],[0,-1,0],[0,-1,0],[0,0,-1],[1,0,0]]
    t = np.eye(4)
    for p, a, angle in zip(origins, axes, q):
        step = np.eye(4)
        step[:3,3] = p
        step[:3,:3] = _rotation(a, angle)
        t = t @ step
    return t


def ray_plane_point(uv, K, camera_transform, plane_point, plane_normal):
    """Intersect an OpenCV optical ray with a plane in a shared base frame."""
    uv = np.asarray(uv, float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError('invalid_pixel')
    ray = camera_transform[:3,:3] @ np.linalg.solve(K, [*uv, 1.])
    origin = camera_transform[:3,3]
    denominator = float(np.dot(plane_normal, ray))
    if not np.isfinite(denominator) or abs(denominator) < 1e-8:
        raise ValueError('ray_parallel_to_carrier')
    distance = float(np.dot(plane_normal, plane_point-origin) / denominator)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('carrier_behind_camera')
    return origin + distance*ray


def preview_from_geometry(pin_uv, hole_uv, K, camera_transform, plane_point,
                          plane_normal, right_from_left_rotation, *, gain=.5,
                          max_step_m=.001, deadband_px=3., frame_yaw_deg=0.):
    """Project both pixels onto the face, then bound their lateral difference.

    ``unclamped_full_delta_*`` retain the complete face-plane correction before
    gain, pixel deadband and norm cap. Left vectors share the same calibrated
    base conversion and configured yaw; these remain conditional estimates.
    """
    result = dict(valid=False, reason='', label='preview estimated',
                  metric_accuracy_validated=False)
    try:
        params = np.asarray([gain, max_step_m, deadband_px, frame_yaw_deg], float)
        if not np.isfinite(params).all() or min(params[:3]) < 0:
            raise ValueError('invalid_preview_settings')
        camera_transform = np.asarray(camera_transform, float)
        K = np.asarray(K, float)
        plane_point = np.asarray(plane_point, float)
        normal = np.asarray(plane_normal, float)
        rotation = np.asarray(right_from_left_rotation, float)
        if (camera_transform.shape != (4,4) or K.shape != (3,3)
                or plane_point.shape != (3,) or normal.shape != (3,)
                or rotation.shape != (3,3) or np.linalg.norm(normal) < 1e-8
                or not all(np.isfinite(x).all() for x in [camera_transform,K,plane_point,normal,rotation])):
            raise ValueError('invalid_geometry')
        normal = normal / np.linalg.norm(normal)
        pin = ray_plane_point(pin_uv, K, camera_transform, plane_point, normal)
        hole = ray_plane_point(hole_uv, K, camera_transform, plane_point, normal)
        error = float(np.linalg.norm(np.asarray(hole_uv)-np.asarray(pin_uv)))
        full_delta = hole-pin
        full_delta -= normal*np.dot(normal, full_delta)
        delta = gain*full_delta
        if error <= deadband_px:
            delta[:] = 0
        norm = float(np.linalg.norm(delta))
        if norm > max_step_m:
            delta *= max_step_m/norm
        left = rotation.T @ delta
        # Optional display-frame correction only; zero preserves calibrated axes.
        yaw = _rotation([0,0,1], np.deg2rad(frame_yaw_deg))
        left = yaw @ left
        # Dashboard alignment holds height in the left native base.
        left[2] = 0.
        full_left = yaw @ rotation.T @ full_delta
        result.update(valid=True, reason='deadband' if error <= deadband_px else 'ok',
                      delta_right_m=delta.tolist(), delta_left_m=left.tolist(),
                      pixel_error_px=error, plane_normal_right=normal.tolist(),
                      unclamped_full_delta_right_m=full_delta.tolist(),
                      unclamped_full_delta_left_m=full_left.tolist())
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        result['reason'] = str(exc)
    return result


@lru_cache(maxsize=4)
def _profiles(directory):
    root = Path(directory)
    return tuple(json.loads((root/name).read_text()) for name in
                 ('camera.json','carrier-grasp-training.json','planar-base.json'))


def estimate_preview_nudge(state14, pin_uv, hole_uv, *, gain=.5,
                           max_step_m=.001, deadband_px=3., frame_yaw_deg=0.,
                           profile_dir=None):
    """Estimate a face-parallel nudge from current native 14-value state.

    Both points are assumed to lie on the carrier mouth plane. A pin above
    the face violates this assumption; this function does not estimate depth.
    """
    try:
        state = np.asarray(state14, float)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('invalid_state14')
        camera, carrier, base = _profiles(str(profile_dir or PROFILE_DIR))
        link = right_link6_transform(state[7:13])
        tool = link.copy()
        tool[:3,3] += link[:3,:3] @ np.array([.165,0,0])
        local_camera = np.eye(4)
        local_camera[:3,3] = camera['model_camera_local_position_m']
        local_camera[:3,:3] = np.asarray(camera['model_camera_local_rotation_mujoco']) @ np.diag([1,-1,-1])
        optical = link @ local_camera
        face = tool @ np.asarray(carrier['T_right_tool_carrier'])
        point = (face @ [0,0,.01725,1])[:3]
        height = int(base.get('image_height',480))
        width = int(base.get('image_width',640))
        focal = height/(2*np.tan(np.deg2rad(camera['fovy_deg'])/2))
        K = np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]])
        return preview_from_geometry(pin_uv,hole_uv,K,optical,point,face[:3,2],
            np.asarray(base['T_right_base_left_base'])[:3,:3],gain=gain,
            max_step_m=max_step_m,deadband_px=deadband_px,frame_yaw_deg=frame_yaw_deg)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return dict(valid=False, reason=str(exc), label='preview estimated',
                    metric_accuracy_validated=False)


@lru_cache(maxsize=4)
def _left_profiles(directory):
    root = Path(directory)
    return tuple(json.loads((root/name).read_text()) for name in
                 ('left-camera.json', 'carrier-grasp-training.json', 'planar-base.json'))


def left_preview_geometry(state14, *, profile_dir=None, image_width=None,
                          image_height=None):
    """Return left optical geometry in the right native base, without devices.

    The explicit LEFT mount is a nominal simulation hypothesis. Neither that
    mount nor the inter-base registration is validated real metric calibration.
    Native joint axes are shared by the two arms; their bases are distinct.
    """
    state = np.asarray(state14, float)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('invalid_state14')
    camera, carrier, base = _left_profiles(str(profile_dir or PROFILE_DIR))
    if camera.get('camera') != 'left_wrist':
        raise ValueError('left_camera_profile_required')
    right_from_left = np.asarray(base['T_right_base_left_base'], float)
    local = np.eye(4)
    local[:3,3] = camera['model_camera_local_position_m']
    local[:3,:3] = (np.asarray(camera['model_camera_local_rotation_mujoco'])
                       @ np.diag([1,-1,-1]))
    optical = right_from_left @ right_link6_transform(state[:6]) @ local
    tool = right_link6_transform(state[7:13])
    tool[:3,3] += tool[:3,:3] @ np.array([.165,0,0])
    face = tool @ np.asarray(carrier['T_right_tool_carrier'])
    width = float(image_width if image_width is not None else camera['image_width'])
    height = float(image_height if image_height is not None else camera['image_height'])
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError('invalid_image_dimensions')
    focal = height/(2*np.tan(np.deg2rad(camera['fovy_deg'])/2))
    K = np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]])
    return K, optical, (face @ [0,0,.01725,1])[:3], face[:3,2], right_from_left[:3,:3]


def estimate_left_preview_nudge(state14, pin_uv, hole_uv, *, gain=.5,
                                max_step_m=.001, deadband_px=3., frame_yaw_deg=0.,
                                profile_dir=None, image_width=None, image_height=None):
    """Apply the existing nudge rules to LEFT wrist pixels on the carrier plane.

    Pixel points must use the supplied image dimensions (default 640 by 480).
    A pin above the mouth plane violates the common-plane assumption. Face
    symmetry does not provide oriented hole identity; callers select the hole.
    """
    try:
        geometry = left_preview_geometry(state14, profile_dir=profile_dir,
            image_width=image_width, image_height=image_height)
        result = preview_from_geometry(pin_uv, hole_uv, *geometry, gain=gain,
            max_step_m=max_step_m, deadband_px=deadband_px, frame_yaw_deg=frame_yaw_deg)
        result.update(camera='left_wrist', geometry_source='nominal_left_wrist_photo_mount',
                      real_calibrated=False)
        return result
    except (ValueError, TypeError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        return dict(valid=False, reason=str(exc), label='preview estimated',
                    metric_accuracy_validated=False, real_calibrated=False, camera='left_wrist')
