"""Versioned, deterministic measured-state features for offline and live Q inference.

Coordinates are never masked by visibility. History contains previous features and
finite differences per second; its sole mask is zero on an episode's first frame.
No timestamps, phase labels, goals, commands, or episode progress are features.
"""
from __future__ import annotations

import math

SCHEMA_VERSION = 1
CAMERA_ROLES = ('overhead', 'left_wrist', 'right_wrist')
CAMERA_NAMES = ('overhead', 'wrist_port3', 'wrist_port2')
IMAGE_SIZE = (640, 480)
LANDMARK_NAMES = {
    'gear': ('gear_center', 'slot_center'),
    'pin': ('pin_tip', 'pin_top', 'carrier_hole_1', 'carrier_hole_2',
            'carrier_hole_3', 'carrier_hole_4'),
}
CHECKPOINT_IDS = {
    'gear': '6c62fdb0dc069eb89eb27fbbfa300c3915fc2d5ab395e135f7ff9599c9e39273',
    'pin': '7fb9ddd55dc006c6c50d1e77fdf62726005d45d812916a54951de9689c1cfe7f',
}


def _array(value, shape, label):
    if not shape:
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(f'{label} must be a finite number')
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{label} must be a finite number') from exc
        if not math.isfinite(number):
            raise ValueError(f'{label} must be finite')
        return number
    if not hasattr(value, '__len__') or len(value) != shape[0]:
        raise ValueError(f'{label} requires shape {shape}')
    return [_array(v, shape[1:], label) for v in value]


def validate_prediction(prediction):
    """Reject changed calibration, model identities, ordering, or incomplete data."""
    try:
        if prediction['status'] != 'ready':
            raise ValueError('prediction must be ready')
        if tuple(prediction['camera_names']) != CAMERA_NAMES:
            raise ValueError('physical camera order does not match feature schema')
        if prediction['camera_mapping'] != dict(zip(CAMERA_ROLES, CAMERA_NAMES)):
            raise ValueError('camera mapping does not match feature schema')
        for task, landmarks in LANDMARK_NAMES.items():
            p = prediction['predictions'][task]
            if tuple(p['image_size']) != IMAGE_SIZE:
                raise ValueError('features require native 640x480 images')
            if tuple(p['camera_names']) != CAMERA_ROLES:
                raise ValueError('model camera role order does not match feature schema')
            if tuple(p['landmark_names']) != landmarks:
                raise ValueError(f'{task} landmark order does not match feature schema')
            if p['checkpoint_id'] != CHECKPOINT_IDS[task]:
                raise ValueError(f'{task} checkpoint identity does not match feature schema')
            _array(p['keypoints_2d'], (3, len(landmarks), 2), 'keypoints_2d')
            for key in ('visibility_probability', 'confidence', 'pixel_std'):
                _array(p[key], (3, len(landmarks)), key)
    except (KeyError, TypeError) as exc:
        raise ValueError(f'missing or invalid prediction field: {exc}') from exc


def _visual_names():
    names = []
    for task, landmarks in LANDMARK_NAMES.items():
        for role in CAMERA_ROLES:
            for landmark in landmarks:
                names.extend(f'{task}.{role}.{landmark}.{field}' for field in
                             ('x', 'y', 'visibility', 'confidence', 'std_x', 'std_y'))
            targets = landmarks[1:] if task == 'gear' else landmarks[2:]
            for target in targets:
                names.extend(f'{task}.{role}.{target}_minus_{landmarks[0]}.{axis}'
                             for axis in ('x', 'y'))
    return names


VISUAL_NAMES = tuple(_visual_names())
VISUAL_DIM = len(VISUAL_NAMES)
VISUAL_PREFIX_DIM = 3 * VISUAL_DIM + 1
PROPRIO_NAMES = tuple(
    [f'{side}.joint_{i}.position_rad' for side in ('left', 'right') for i in range(6)]
    + [f'{side}.gripper.position' for side in ('left', 'right')]
    + [f'{side}.tool.{axis}_m' for side in ('left', 'right') for axis in 'xyz']
    + [f'{side}.joint_{i}.velocity_rad_s' for side in ('left', 'right') for i in range(6)])
FEATURE_NAMES = tuple([f'current.{n}' for n in VISUAL_NAMES]
                      + [f'previous.{n}' for n in VISUAL_NAMES]
                      + [f'delta_per_s.{n}' for n in VISUAL_NAMES]
                      + ['history_valid'] + list(PROPRIO_NAMES))
OBSERVATION_DIM = len(FEATURE_NAMES)


def feature_spec():
    """JSON-serializable contract to persist alongside every Q checkpoint."""
    return dict(schema_version=SCHEMA_VERSION, visual_dim=VISUAL_DIM,
                visual_prefix_dim=VISUAL_PREFIX_DIM, observation_dim=OBSERVATION_DIM,
                names=list(FEATURE_NAMES), camera_roles=list(CAMERA_ROLES),
                camera_names=list(CAMERA_NAMES), camera_mapping=dict(zip(CAMERA_ROLES, CAMERA_NAMES)),
                image_size=list(IMAGE_SIZE), checkpoint_ids=dict(CHECKPOINT_IDS),
                landmark_names={k: list(v) for k, v in LANDMARK_NAMES.items()},
                coordinate_scale=[640, 480], pixel_std_scale=[640, 480],
                history='previous visual, (current - previous) / actual dt seconds, valid mask')


def _visual(prediction):
    validate_prediction(prediction)
    result = []
    for task, landmarks in LANDMARK_NAMES.items():
        p = prediction['predictions'][task]
        for camera in range(3):
            xy = p['keypoints_2d'][camera]
            for i in range(len(landmarks)):
                std = float(p['pixel_std'][camera][i])
                result.extend([float(xy[i][0]) / 640, float(xy[i][1]) / 480,
                               float(p['visibility_probability'][camera][i]),
                               float(p['confidence'][camera][i]), std / 640, std / 480])
            targets = range(1, 2) if task == 'gear' else range(2, 6)
            for i in targets:
                result.extend([(float(xy[i][0]) - float(xy[0][0])) / 640,
                               (float(xy[i][1]) - float(xy[0][1])) / 480])
    return result


def encode_observation(prediction, arms, previous_prediction=None, previous_dt=None):
    """Return current/history visual prefix followed by 32 measured arm values.

    ``previous_dt`` is actual elapsed seconds, mandatory exactly when history is
    supplied. Episode boundaries must reset history explicitly to None.
    """
    current = _visual(prediction)
    if previous_prediction is None:
        if previous_dt is not None:
            raise ValueError('previous_dt requires previous_prediction')
        previous, delta, mask = [0.] * VISUAL_DIM, [0.] * VISUAL_DIM, 0.
    else:
        dt = _array(previous_dt, (), 'previous_dt')
        if dt <= 0:
            raise ValueError('previous_dt must be positive elapsed seconds')
        previous = _visual(previous_prediction)
        delta, mask = [(c - p) / dt for c, p in zip(current, previous)], 1.
    try:
        proprio = []
        for side in ('left', 'right'):
            proprio.extend(_array(arms[side]['position_rad'], (6,), 'position_rad'))
        for side in ('left', 'right'):
            proprio.append(_array(arms[side]['gripper']['position'], (), 'gripper.position'))
        for side in ('left', 'right'):
            proprio.extend(_array(arms[side]['tool_pose_base']['position_m'], (3,), 'position_m'))
        for side in ('left', 'right'):
            proprio.extend(_array(arms[side]['velocity_rad_s'], (6,), 'velocity_rad_s'))
    except (KeyError, TypeError) as exc:
        raise ValueError(f'missing measured arm field: {exc}') from exc
    result = current + previous + delta + [mask] + proprio
    if not all(math.isfinite(v) for v in result):
        raise ValueError('derived features must be finite')
    return result
