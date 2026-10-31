"""Calibrated two-view 3D landmark alignment with fixed tool orientation.

The current CNN has point landmarks, not enough geometry for a full object
orientation estimate. This implements translation-only position-based servoing.
No camera calibration or target geometry is invented.
"""
import copy
import threading
import time

import cv2
import numpy as np

from alignment_vision import AlignmentVision
from cartesian_nudge import rotation_error
from constrained_visual_nudge import checked_translation, vector


def transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError('invalid_rigid_transform')
    rotation_error(t[:3, :3], np.eye(3))
    return t


def pose_matrix(pose):
    t = np.eye(4)
    t[:3, 3] = vector(pose['position_m'], 3)
    t[:3, :3] = np.asarray(pose['rotation'])
    return transform(t)


class StereoAlignmentVision(AlignmentVision):
    """Associate measured wrist poses with the exact inference input stamps."""
    def __init__(self, config):
        super().__init__(config)
        self.geometry_lock = threading.Lock()
        self.geometry = {}

    def submit(self, frames, arms, state_monotonic_ns=None):
        try:
            stamps = tuple(int(frames[name]['monotonic_ns']) for name in self.camera_names)
            context = dict(state_monotonic_ns=time.monotonic_ns() if state_monotonic_ns is None else int(state_monotonic_ns),
                tool_poses={side: copy.deepcopy(arms[side]['tool_pose_base']) for side in ('left', 'right')})
            with self.geometry_lock:
                if stamps not in self.geometry:
                    self.geometry[stamps] = context
                while len(self.geometry) > 64:
                    self.geometry.pop(next(iter(self.geometry)))
        except (KeyError, ValueError, TypeError):
            return False
        return super().submit(frames, arms)

    def snapshot(self):
        snapshot = super().snapshot()
        try:
            stamps = tuple(snapshot['frame_timestamps_ns'][name] for name in self.camera_names)
        except KeyError:
            return snapshot
        with self.geometry_lock:
            context = self.geometry.get(stamps)
            if context is not None:
                snapshot['geometry_context'] = copy.deepcopy(context)
        return snapshot


def validate_calibration(calibration, runtime_calibration):
    if calibration.get('verified') is not True:
        raise ValueError('stereo_calibration_required')
    for key in ('checkpoint_id', 'camera_mapping', 'image_size'):
        if calibration.get(key) != runtime_calibration.get(key):
            raise ValueError('stereo_'+key+'_mismatch')
    vector(calibration.get('target_offset_left_base_m'), 3, 'demonstrated_3d_target_required')
    transform(calibration.get('T_left_base_right_base'))
    for name in ('left_wrist', 'right_wrist'):
        c = calibration['cameras'][name]
        if c['side'] not in ('left', 'right'):
            raise ValueError('invalid_camera_arm_assignment')
        k = np.asarray(c['intrinsics'], dtype=float)
        distortion = np.asarray(c['distortion'], dtype=float)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or distortion.ndim != 1 or len(distortion) not in (4, 5, 8, 12, 14)
                or not np.isfinite(distortion).all()):
            raise ValueError('invalid_camera_intrinsics')
        transform(c['T_tool_camera'])


def triangulate(observations, camera_poses, cameras):
    normalized, projections = [], []
    for point, pose, c in zip(observations, camera_poses, cameras):
        point = vector(point, 2)
        normalized.append(cv2.undistortPoints(point.reshape(1, 1, 2),
            np.asarray(c['intrinsics'], dtype=float), np.asarray(c['distortion'], dtype=float)).ravel())
        projections.append(np.linalg.inv(transform(pose))[:3])
    rays = [pose[:3, :3] @ np.r_[point, 1.] for pose, point in zip(camera_poses, normalized)]
    cosine = np.dot(*rays)/(np.linalg.norm(rays[0])*np.linalg.norm(rays[1]))
    angle = np.arccos(np.clip(cosine, -1, 1))
    if min(angle, np.pi-angle) < np.deg2rad(2):
        raise ValueError('triangulation_parallax_too_small')
    homogeneous = cv2.triangulatePoints(*projections,
        normalized[0].reshape(2, 1), normalized[1].reshape(2, 1)).ravel()
    if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-9:
        raise ValueError('invalid_triangulation')
    point = homogeneous[:3]/homogeneous[3]
    for observation, projection, c in zip(observations, projections, cameras):
        local = projection @ np.r_[point, 1.]
        if not .03 < local[2] < 2:
            raise ValueError('triangulated_depth_out_of_range')
        reprojected, _ = cv2.projectPoints(local.reshape(1, 3), np.zeros(3), np.zeros(3),
            np.asarray(c['intrinsics'], dtype=float), np.asarray(c['distortion'], dtype=float))
        if np.linalg.norm(reprojected.ravel()-observation) > 2:
            raise ValueError('stereo_reprojection_error')
    return point


def stereo_residual(snapshot, calibration):
    context = snapshot.get('geometry_context')
    if not isinstance(context, dict):
        raise ValueError('timestamped_camera_poses_required')
    p = snapshot['predictions']['gear']
    if p['checkpoint_id'] != calibration['checkpoint_id'] or p['image_size'] != calibration['image_size']:
        raise ValueError('stereo_prediction_identity_mismatch')
    poses, cameras, indices = [], [], []
    for name in ('left_wrist', 'right_wrist'):
        physical = calibration['camera_mapping'][name]
        if snapshot['camera_mapping'].get(name) != physical:
            raise ValueError('stereo_camera_mapping_mismatch')
        stamp = snapshot['frame_timestamps_ns'][physical]
        if abs(context['state_monotonic_ns']-stamp)/1e9 > .03:
            raise ValueError('camera_pose_timestamp_skew')
        c = calibration['cameras'][name]
        base_tool = pose_matrix(context['tool_poses'][c['side']])
        if c['side'] == 'right':
            base_tool = transform(calibration['T_left_base_right_base']) @ base_tool
        poses.append(base_tool @ transform(c['T_tool_camera']))
        cameras.append(c)
        indices.append(p['camera_names'].index(name))
    points = np.asarray(p['keypoints_2d'], dtype=float)
    world = []
    for name in ('gear_center', 'slot_center'):
        landmark = p['landmark_names'].index(name)
        world.append(triangulate([points[i, landmark] for i in indices], poses, cameras))
    return world[0]-world[1]-vector(calibration['target_offset_left_base_m'], 3)


def stereo_pose_proposal(q, residual_m, kinematics, limits, settings):
    residual = vector(residual_m, 3)
    translation = -residual
    translation *= min(1., .0005/max(np.linalg.norm(translation), 1e-12))
    proposal = checked_translation(q, translation, kinematics, limits, settings)
    proposal['predicted_residual'] = residual+proposal['delta_xyz_m']
    return proposal
