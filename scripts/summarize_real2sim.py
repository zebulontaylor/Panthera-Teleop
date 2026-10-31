#!/usr/bin/env python3
"""Compare fresh real2sim runs using stored arrays, including old report formats."""
import argparse
import json
from pathlib import Path

import numpy as np

ARM = [*range(6), *range(7, 13)]


def summarize(root):
    summary = json.loads((root / 'summary.json').read_text())
    episodes, squared, intersections = [], [], []
    for e in summary['episodes']:
        report = json.loads((root / e['episode'] / 'report.json').read_text())
        with np.load(root / e['episode'] / 'mapped.npz', allow_pickle=False) as a:
            expected = a['native_action_rad'][:, ARM]
            measured = a['dynamic_scene_contact_native'][:, ARM]
            error = measured - expected[:len(measured)]
            squared.append(np.square(error).ravel())
            rms = float(np.sqrt(np.square(error).mean())) if error.size else None
        contacts = report['kinematic_contacts']
        intersect = [c['counts']['robot_table'] > 0 for c in contacts]
        intersections.extend(intersect)
        episodes.append(dict(episode=e['episode'], frames=e['frames'],
                             arm_tracking_rmse_rad=rms,
                             arm_tracking_max_abs_rad=float(np.abs(error).max()) if error.size else None,
                             robot_table_frame_fraction=float(np.mean(intersect)),
                             robot_table_max_penetration_m=max((-c['distance_m'] for frame in contacts
                                 for c in frame['contacts'] if c['category'] == 'robot_table'), default=0.),
                             fk_max_position_error_m=e['fk_position_error_m']['max'],
                             arm_limit_violation_frame_fraction=e['arm_limit_violation_frame_fraction']))
    return dict(root=str(root.resolve()), table_offset_m=summary.get('table_offset_m', 0),
                frames=sum(e['frames'] for e in episodes), episodes=episodes,
                pooled_arm_tracking_rmse_rad=float(np.sqrt(np.concatenate(squared).mean())),
                robot_table_frame_fraction=float(np.mean(intersections)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a fresh file')
    result = dict(runs=[summarize(root) for root in args.runs],
                  interpretation='Oracle demonstration tracking and scene-registration ablation; no learned policy or task-success evaluation.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
