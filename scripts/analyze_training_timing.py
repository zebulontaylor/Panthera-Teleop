#!/usr/bin/env python3
"""Audit next-pose training data and estimate retiming costs without editing it.

Run with work/postprocess-venv/bin/python. Reads numeric Parquet columns only.
Local velocity estimates are duration lower bounds, NOT acceleration-feasible
retiming plans. Uniform episode dilation is a conservative discrete reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

GROUPS = {'joints': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
          'grippers': [6, 13]}
# Sensitivity probes, not certified robot limits: joint v/a, gripper v/a.
PROBES = [(0.6, 2., 2., 8.), (0.8, 3., 2., 8.),
          (1., 4., 3., 12.), (1.5, 6., 3., 12.)]


def derivatives(states, actions, fps):
    states, actions = np.asarray(states, dtype=float), np.asarray(actions, dtype=float)
    if (states.ndim != 2 or states.shape[1] != 14 or actions.shape != states.shape
            or len(states) < 2 or not np.isfinite(fps) or fps <= 0
            or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError('Require at least two finite 14D next-pose rows and positive FPS')
    # This estimator is only valid for the exporter's measured-next-pose target.
    mismatch = float(np.max(np.abs(actions[:-1] - states[1:])))
    if mismatch > 1e-6:
        raise ValueError(f'Actions do not match next states: maximum error {mismatch} rad')
    poses = np.concatenate([states, actions[-1:]])
    velocity = np.diff(poses, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return velocity, acceleration, mismatch


def dilation_cost(velocity, acceleration, limits):
    joint_v, joint_a, gripper_v, gripper_a = limits
    caps_v = np.full(14, joint_v)
    caps_a = np.full(14, joint_a)
    caps_v[[6, 13]] = gripper_v
    caps_a[[6, 13]] = gripper_a
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError('All probe limits must be positive and finite')
    local = np.maximum(1., (np.abs(velocity) / caps_v).max(axis=1))
    # Uniform dilation r transforms finite-difference v to v/r and a to a/r².
    uniform = max(1., float((np.abs(velocity) / caps_v).max()),
                  float(np.sqrt((np.abs(acceleration) / caps_a).max())))
    return {'local_velocity_interval_equivalents': float(local.sum()),
            'velocity_exceedance_frame_percent': float((local > 1.).mean() * 100),
            'uniform_episode_scale': uniform}


def summarize(values):
    result = {}
    for group, axes in GROUPS.items():
        absolute = np.abs(values[:, axes])
        result[group] = {
            'frame_max_percentiles': dict(zip(['50', '90', '95', '99', '100'],
                np.percentile(absolute.max(axis=1), [50, 90, 95, 99, 100]).tolist())),
            'axis_sample_percentiles': dict(zip(['50', '90', '95', '99', '100'],
                np.percentile(absolute, [50, 90, 95, 99, 100]).tolist()))}
    return result


def analyze(root):
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    names = info['features']['action']['names']
    expected_names = [name for side in ('left', 'right')
                      for name in [*[f'{side}.joint{i}' for i in range(1, 7)],
                                   f'{side}.gripper_rad']]
    if names != expected_names or info['features']['observation.state']['names'] != names:
        raise ValueError('Require the Panthera 14D joint/gripper axis order')
    columns = ['episode_index', 'frame_index', 'observation.state', 'action']
    files = sorted((root / 'data').rglob('*.parquet'))
    if not files:
        raise ValueError('No numeric data found')
    parts = [pq.read_table(path, columns=columns).to_pydict() for path in files]
    data = {key: np.concatenate([np.asarray(part[key]) for part in parts])
            for key in columns}
    order = np.lexsort((data['frame_index'], data['episode_index']))
    data = {key: value[order] for key, value in data.items()}
    if len(order) != info['total_frames']:
        raise ValueError('Frame count differs from metadata')
    # Digest numeric inputs; do not read embedded camera bytes or hash huge media files.
    digest = hashlib.sha256()
    for key, value in data.items():
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    velocities, accelerations, episodes = [], [], []
    for episode in np.unique(data['episode_index']):
        mask = data['episode_index'] == episode
        frames = data['frame_index'][mask]
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f'Missing or duplicate frames in episode {episode}')
        v, a, mismatch = derivatives(data['observation.state'][mask],
                                     data['action'][mask], info['fps'])
        velocities.append(v)
        accelerations.append(a)
        episodes.append({'episode_index': int(episode), 'frames': len(v),
                         'next_state_error_rad': mismatch,
                         'probes': [dilation_cost(v, a, limits) for limits in PROBES]})
    if len(episodes) != info['total_episodes']:
        raise ValueError('Episode count differs from metadata')
    total = sum(e['frames'] for e in episodes)
    probes = []
    for index, limits in enumerate(PROBES):
        lower = sum(e['probes'][index]['local_velocity_interval_equivalents'] for e in episodes)
        uniform = sum(e['frames'] * e['probes'][index]['uniform_episode_scale'] for e in episodes)
        exceeded = sum(e['frames'] * e['probes'][index]['velocity_exceedance_frame_percent']
                       for e in episodes) / total
        probes.append(dict(zip(['joint_velocity_rad_s', 'joint_acceleration_rad_s2',
                               'gripper_velocity_rad_s', 'gripper_acceleration_rad_s2'], limits),
            local_velocity_only_duration_ratio_lower_bound=lower / total,
            local_velocity_only_added_seconds_lower_bound=(lower - total) / info['fps'],
            velocity_exceedance_frame_percent=exceeded,
            uniform_episode_discrete_velocity_acceleration_duration_ratio=uniform / total))
    return {'dataset_root': str(root.resolve()), 'episodes': len(episodes),
            'frames': total, 'fps': info['fps'], 'baseline_transition_duration_s': total / info['fps'],
            'numeric_input_sha256': digest.hexdigest(),
            'info_sha256': hashlib.sha256(info_path.read_bytes()).hexdigest(),
            'method': {'velocity': '(next_pose - current_pose) * fps, per episode',
                       'acceleration': 'diff(velocity) * fps, interior only, per episode',
                       'local_velocity_lower_bound': 'sum(max(1, max_axis(abs(v)/v_cap))) / frame_count',
                       'uniform_reference': 'per-episode max(1, max(abs(v)/v_cap), sqrt(max(abs(a)/a_cap)))',
                       'boundary_conditions': 'No assumed rest at episode boundaries; no cross-episode differences'},
            'limitations': ['Discrete acceleration includes encoder quantization and sampling artifacts.',
                           'Velocity-only local bound omits acceleration ramps and frame-grid rounding.',
                           'Uniform reference bounds original-grid finite differences, not continuous acceleration.',
                           'No local acceleration-constrained retimer or training-loss experiment was run.',
                           'Thresholds are sensitivity probes, not certified robot limits.'],
            'velocity_rad_s': summarize(np.concatenate(velocities)),
            'acceleration_rad_s2': summarize(np.concatenate(accelerations)),
            'probes': probes, 'per_episode': episodes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_episode'}, indent=2))
