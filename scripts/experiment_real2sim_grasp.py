#!/usr/bin/env python3
"""Reconstruct saved object motion from inspected grasps and explicit pose priors.

No cameras or motors are opened. Before-pickup tabletop poses are hypotheses;
held poses propagate by rigid-grasp FK. Pose after release is not reconstructed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from real2sim_data import load_episode
from real2sim_grasp import estimate_tabletop_grasp, reconstruct_grasp


def run(episode_path, config_path, output):
    ep = load_episode(episode_path)
    config = json.loads(config_path.read_text())
    if config['episode'] != ep.metadata['id']:
        raise ValueError('Grasp annotations belong to another episode')
    rows = {row['monotonic_ns']: row for row in
            (json.loads(line) for line in (episode_path / 'states.jsonl').read_text().splitlines() if line.strip())}
    tools = {}
    for side in ('left', 'right'):
        poses = np.tile(np.eye(4), (len(ep.state), 1, 1))
        for i, stamp in enumerate(ep.state_monotonic_ns):
            tcp = rows[int(stamp)]['arms'][side]['tool_pose_base']
            poses[i, :3, :3] = tcp['rotation']
            poses[i, :3, 3] = tcp['position_m']
        tools[side] = poses
    output.mkdir(parents=True, exist_ok=False)
    arrays = dict(time_s=ep.time_s, grid_monotonic_ns=ep.grid_monotonic_ns,
                  native_state_rad=ep.state)
    report = dict(episode=ep.metadata['id'], parts={}, common_world_frame_established=False,
                  object_pose_in_jaws_fitted_to_images=False, physics_simulated=False,
                  interpretation='grasp-anchored object trajectory hypotheses with explicit tabletop priors')
    for name, part in config['parts'].items():
        pickup = int(np.argmin(abs(ep.time_s - part['pickup_s'])))
        release = int(np.argmin(abs(ep.time_s - part['release_s'])))
        tcp = tools[part['side']]
        if 'tool_to_object' in part:
            offset = np.asarray(part['tool_to_object'], dtype=float)
            result = reconstruct_grasp(ep.time_s, tcp, pickup, release, tool_to_object=offset)
            prior = 'supplied grasp transform; inspect config provenance'
        else:
            guess = estimate_tabletop_grasp(tcp[pickup],
                object_yaw_rad=part.get('pickup_yaw_rad', 0),
                table_z_m=part.get('table_z_m', 0),
                offset_xy_m=part.get('pickup_center_offset_xy_m', [0, 0]))
            offset = guess['tool_to_object']
            result = reconstruct_grasp(ep.time_s, tcp, pickup, release, tool_to_object=offset)
            prior = 'initial body bottom at table z; centre XY at pickup TCP plus configured offset; upright yaw supplied'
        # Store no post-release object pose: support/gravity/assembly may move it.
        poses = result['object_poses'].copy()
        poses[release + 1:] = np.nan
        arrays[name + '_object_pose_native_base'] = poses
        arrays[name + '_held'] = result['held']
        arrays[name + '_tool_to_object'] = offset
        arrays[name + '_tcp_native_base'] = tcp
        report['parts'][name] = dict(side=part['side'], frame=part['side'] + '_native_base',
            pickup_s=float(ep.time_s[pickup]), release_s=float(ep.time_s[release]),
            observed_pickup_tcp_m=tcp[pickup, :3, 3].tolist(),
            assumed_initial_object_pose=poses[0].tolist(),
            tool_to_object=offset.tolist(), pose_prior=prior,
            pickup_interval_s=part['pickup_interval_s'], release_interval_s=part['release_interval_s'],
            held_frames=int(result['held'].sum()),
            post_release='unobserved; object pose stored as NaN',
            assumptions=['rigid no-slip grasp during annotated interval',
                         'object was stationary on table before pickup',
                         'object centre and yaw within jaws are priors unless a fitted transform is supplied'])
    arrays['left_tcp_native_base'] = tools['left']
    arrays['right_tcp_native_base'] = tools['right']
    np.savez_compressed(output / 'grasp_object_trajectories.npz', **arrays)
    report['config'] = config
    report['source_sha256'] = {n: hashlib.sha256((episode_path / n).read_bytes()).hexdigest()
                               for n in ('metadata.json', 'states.jsonl')}
    report['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.episode, args.config, args.output)
    print(json.dumps(report, indent=2))
