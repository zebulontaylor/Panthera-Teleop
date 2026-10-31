#!/usr/bin/env python3
"""Check whether old CNN caches contain near-pure task-axis motion pairs.

This does not assign train/validation labels or reinterpret old data as a new
controlled experiment. Results are geometric coverage diagnostics only.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np
from nudge_calibration_common import angle, finite, load_rows, write_report


def audit(rows, axis):
    axis = finite(axis, (3,))
    if abs(np.linalg.norm(axis)-1) > 1e-6:
        raise ValueError('axis_must_be_unit_vector')
    groups = defaultdict(list)
    for row in rows:
        groups[row['episode_id']].append(row)
    counts = Counter()
    displacement = []
    for episode in groups.values():
        episode.sort(key=lambda row: row['time_s'])
        i = 0
        while i < len(episode)-1:
            j = i+1
            while j < len(episode) and episode[j]['time_s']-episode[i]['time_s'] < .2:
                j += 1
            if j >= len(episode):
                break
            a, b = episode[i], episode[j]; i = j+1
            if b['time_s']-a['time_s'] > .5:
                counts['time_gap'] += 1
                continue
            left_a, left_b = a['arms']['left'], b['arms']['left']
            dx = finite(left_b['tool_pose_base']['position_m'], (3,))-finite(left_a['tool_pose_base']['position_m'], (3,))
            s = float(dx @ axis)
            displacement.append(float(np.linalg.norm(dx)))
            if not .0002 <= abs(s) <= .003:
                counts['axis_displacement_outside_range'] += 1
            elif np.linalg.norm(dx-s*axis) > max(.0001, .1*abs(s)):
                counts['off_axis_motion'] += 1
            elif angle(left_a['tool_pose_base']['rotation'], left_b['tool_pose_base']['rotation']) > .003:
                counts['orientation_changed'] += 1
            elif np.max(np.abs(np.asarray(b['q_right'])-a['q_right'])) > .003:
                counts['carrier_moved'] += 1
            else:
                counts['geometrically_eligible_positive' if s > 0 else 'geometrically_eligible_negative'] += 1
    return dict(scope='old-recording geometric coverage; not controlled calibration or corrective validation',
        observations=len(rows), episodes=len(groups), axis_base=axis.tolist(),
        pair_outcomes=dict(counts), total_pairs=sum(counts.values()),
        median_pair_tool_motion_mm=float(np.median(displacement)*1000) if displacement else None,
        new_protocol_annotations_present=False,
        statistical_candidate=False, reason='fresh marked bidirectional train/validation sessions still required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--axis', nargs=3, type=float, default=[0, 0, 1])
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = audit(load_rows(args.predictions), args.axis)
    write_report(args.output, result, [args.predictions, Path(__file__)])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
