"""Rigid grasp kinematic estimates in a caller-supplied shared metric frame.

Transforms map local coordinates into the named parent frame. This module does
not infer contacts, observe slip, or simulate dynamics.
"""
import numpy as np


def _pose(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite 4x4 transform')
    if (not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-6)):
        raise ValueError(f'{name} must be a proper rigid transform')
    return a.copy()


def estimate_tabletop_grasp(T_tool_pickup, object_height_m=0, object_yaw_rad=0,
                            offset_xy_m=(0, 0), table_z_m=0):
    """Return assumed object pose and tool offset at pickup.

    The object origin is its bottom surface center, at table_z_m +
    object_height_m. Set object_height_m to half the body height for a body-center
    origin. XY offset is expressed in the shared frame, not tool axes.
    """
    tool = _pose(T_tool_pickup, 'T_tool_pickup')
    xy = np.asarray(offset_xy_m, dtype=float)
    scalars = np.asarray([object_height_m, object_yaw_rad, table_z_m], dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(scalars).all():
        raise ValueError('tabletop parameters must be finite; offset_xy_m must have length 2')
    c, s = np.cos(object_yaw_rad), np.sin(object_yaw_rad)
    obj = np.eye(4)
    obj[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    obj[:2, 3] = tool[:2, 3] + xy
    obj[2, 3] = table_z_m + object_height_m
    return dict(object_pose_at_pickup=obj, tool_to_object=np.linalg.inv(tool) @ obj,
                pose_source='assumed_tabletop_pose', origin_assumption='caller-selected height above table',
                kinematic_estimate=True)


def reconstruct_grasp(timestamps, tcp_poses, pickup_index, release_index=None, *,
                      object_pose_at_pickup=None, tool_to_object=None):
    """Reconstruct one pickup/release, including release sample in rigid motion.

    Before pickup the object is stationary at its supplied pickup pose; after
    release it remains at its release pose. A missing release means held to end.
    Exactly one pose/offset must be supplied. Indices refer to recorded samples;
    no interpolation or contact detection is performed.
    """
    times = np.asarray(timestamps, dtype=float)
    poses = np.asarray(tcp_poses, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must be finite, nonempty, strictly increasing')
    if poses.shape != (len(times), 4, 4):
        raise ValueError('tcp_poses must have shape Nx4x4 matching timestamps')
    for pose in poses:
        _pose(pose, 'tcp_pose')
    def index(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < len(times):
            raise ValueError(f'{name} must be a valid sample index')
        return int(value)
    pickup = index(pickup_index, 'pickup_index')
    release = len(times)-1 if release_index is None else index(release_index, 'release_index')
    if release < pickup:
        raise ValueError('release_index must be at or after pickup_index')
    if (object_pose_at_pickup is None) == (tool_to_object is None):
        raise ValueError('supply exactly one of object_pose_at_pickup or tool_to_object')
    if tool_to_object is None:
        initial = _pose(object_pose_at_pickup, 'object_pose_at_pickup')
        offset = np.linalg.inv(poses[pickup]) @ initial
    else:
        offset = _pose(tool_to_object, 'tool_to_object')
        initial = poses[pickup] @ offset
    result = np.repeat(initial[None], len(times), axis=0)
    result[pickup:release+1] = poses[pickup:release+1] @ offset
    result[release+1:] = result[release]
    held = np.zeros(len(times), dtype=bool)
    held[pickup:release+1] = True
    return dict(timestamps=times.copy(), object_poses=result, held=held, tool_to_object=offset,
                metadata=dict(kinematic_estimate=True, physics_simulated=False,
                              pose_source='supplied_object_pose' if object_pose_at_pickup is not None else 'supplied_tool_offset',
                              assumptions=['shared metric coordinate frame', 'rigid grasp with no slip',
                                           'stationary before pickup and after release'],
                              slip_observed=False, release_sample_included=True))


def fit_planar_base_transform(right_points, left_points, *, heldout_right_points=None,
                              heldout_left_points=None):
    """Fit left = Rz(yaw) * right + translation, without scale.

    Held-out paired points only measure validation error, never enter the fit.
    Two distinct XY positions determine yaw; collinear XY points are supported.
    """
    def points(value, name):
        a = np.asarray(value, dtype=float)
        if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError(f'{name} must be finite Nx3 with at least two points')
        return a
    right, left = points(right_points, 'right_points'), points(left_points, 'left_points')
    if right.shape != left.shape:
        raise ValueError('paired points must have matching shapes')
    r, l = right - right.mean(0), left - left.mean(0)
    rank_r, rank_l = np.linalg.matrix_rank(r[:, :2]), np.linalg.matrix_rank(l[:, :2])
    dot = np.sum(r[:, 0]*l[:, 0] + r[:, 1]*l[:, 1])
    cross = np.sum(r[:, 0]*l[:, 1] - r[:, 1]*l[:, 0])
    if rank_r == 0 or rank_l == 0 or np.hypot(dot, cross) <= 1e-12:
        raise ValueError('degenerate XY correspondences cannot identify yaw')
    yaw = np.arctan2(cross, dot)
    c, s = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    transform[:3, 3] = left.mean(0) - transform[:3, :3] @ right.mean(0)
    def rms(a, b):
        return float(np.sqrt(np.mean(np.sum((a @ transform[:3, :3].T + transform[:3, 3] - b)**2, axis=1))))
    validation = None
    if (heldout_right_points is None) != (heldout_left_points is None):
        raise ValueError('supply both heldout point arrays')
    if heldout_right_points is not None:
        hr, hl = points(heldout_right_points, 'heldout_right_points'), points(heldout_left_points, 'heldout_left_points')
        if hr.shape != hl.shape:
            raise ValueError('heldout paired points must have matching shapes')
        validation = rms(hr, hl)
    return dict(right_to_left=transform, yaw_rad=float(yaw), translation_m=transform[:3, 3].copy(),
                fit_rms_m=rms(right, left), heldout_rms_m=validation,
                right_xy_rank=int(rank_r), left_xy_rank=int(rank_l), scale=1.0,
                heldout_used_in_fit=False)
