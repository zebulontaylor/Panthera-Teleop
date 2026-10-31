#!/usr/bin/env python3
"""Assemble stable, operator-marked endpoints from synchronized CNN caches."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from nudge_calibration_common import angle, finite, load_rows, write_report


def plateau(rows, marker, window=.75):
    samples = sorted([r for r in rows if r['episode_id'] == marker['episode_id']
        and marker['time_s']-window <= r['time_s'] <= marker['time_s']], key=lambda r: r['time_s'])
    positions, errors, rotations, other_q, grippers, stamps = [], [], [], [], [], []
    identity = None
    last = None
    for row in samples:
        p = row['predictions']['gear']
        names = ('left_wrist', 'right_wrist')
        physical = [row['camera_mapping'][name] for name in names]
        pair_stamp = tuple(int(row['frame_timestamps_ns'][name]) for name in physical)
        if last is not None and any(a <= b for a, b in zip(pair_stamp, last)):
            continue
        if (max(pair_stamp)-min(pair_stamp) > 60_000_000
                or any(abs(t-row['state_monotonic_ns']) > 50_000_000 for t in pair_stamp)):
            raise ValueError('plateau_timestamp_skew')
        this_identity = (json.dumps(row['camera_mapping'], sort_keys=True), p['checkpoint_id'])
        if identity is not None and this_identity != identity:
            raise ValueError('plateau_perception_identity_changed')
        identity = this_identity
        arm, other = row['arms']['left'], row['arms']['right']
        if max(np.max(np.abs(finite(a['velocity_rad_s'], (6,)))) for a in (arm, other)) > .02:
            raise ValueError('plateau_arm_not_settled')
        for a in (arm, other):
            if any(a.get('motor_fault', [])):
                raise ValueError('plateau_motor_fault')
        points = np.asarray(p['keypoints_2d'], dtype=float)
        ci = [p['camera_names'].index(name) for name in names]
        moving, target = [p['landmark_names'].index(name) for name in ('gear_center', 'slot_center')]
        selected = points[np.ix_(ci, [moving, target])]
        if selected.shape != (2, 2, 2) or not np.isfinite(selected).all():
            raise ValueError('invalid_plateau_landmarks')
        if np.any(selected < 0) or np.any(selected >= p['image_size']):
            raise ValueError('plateau_landmark_out_of_frame')
        positions.append(finite(arm['tool_pose_base']['position_m'], (3,)))
        rotations.append(arm['tool_pose_base']['rotation'])
        errors.append((selected[:, 0]-selected[:, 1]).ravel())
        other_q.append(finite(other['position_rad'], (6,)))
        grippers.append(float(arm['gripper']['position']))
        stamps.append(row['monotonic_ns'])
        last = pair_stamp
    if len(positions) < 5:
        raise ValueError('fewer_than_five_distinct_stable_observations')
    center = np.median(positions, axis=0)
    if max(np.linalg.norm(p-center) for p in positions) > .00015:
        raise ValueError('plateau_position_drift')
    if max(angle(r, rotations[0]) for r in rotations) > .002:
        raise ValueError('plateau_orientation_drift')
    other_center = np.median(other_q, axis=0)
    if np.max(np.abs(np.asarray(other_q)-other_center)) > .002:
        raise ValueError('plateau_carrier_drift')
    if not np.isfinite(grippers).all() or np.ptp(grippers) > .01:
        raise ValueError('plateau_gripper_drift')
    error = np.median(errors, axis=0)
    if np.max(np.median(np.abs(np.asarray(errors)-error), axis=0)) > 3:
        raise ValueError('plateau_landmarks_unstable')
    return dict(position_m=center.tolist(), rotation=rotations[0], error_px=error.tolist(),
        other_q=other_center.tolist(), gripper_position=float(np.median(grippers)),
        camera_mapping=json.loads(identity[0]), checkpoint_id=identity[1],
        observations=len(positions), first_monotonic_ns=min(stamps), last_monotonic_ns=max(stamps))


def build(rows, markers):
    groups = defaultdict(list)
    for marker in markers:
        groups[(marker['session_id'], marker['episode_id'], marker['trial_id'])].append(marker)
    pairs, rejected, references = [], Counter(), []
    for (session, episode, trial), group in sorted(groups.items()):
        try:
            if len({m['split'] for m in group}) != 1:
                raise ValueError('mixed_trial_splits')
            for marker in group:
                if marker['label'] == 'reference':
                    references.append(dict(session_id=session, episode_id=episode, trial_id=trial,
                        time_s=marker['time_s'], **plateau(rows, marker)))
            endpoints = {label: [m for m in group if m['label'] == label] for label in ('start', 'end')}
            if not any(endpoints.values()):
                continue
            if any(len(v) != 1 for v in endpoints.values()):
                raise ValueError('exactly_one_start_and_end_required')
            start, end = (plateau(rows, endpoints[name][0]) for name in ('start', 'end'))
            if start['last_monotonic_ns'] >= end['first_monotonic_ns']:
                raise ValueError('overlapping_or_reversed_endpoint_windows')
            pair = dict(session_id=session, episode_id=episode, trial_id=trial,
                split=group[0]['split'], start=start, end=end)
            for key in ('requested_delta_m', 'external_delta_m'):
                if key in endpoints['end'][0]:
                    pair[key] = endpoints['end'][0][key]
            pairs.append(pair)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            rejected[str(exc)] += 1
    return pairs, dict(rejected), references


def reference_repeatability(references):
    groups = defaultdict(list)
    for row in references:
        groups[row['session_id']].append(row)
    report = {}
    for session, rows in groups.items():
        rows.sort(key=lambda r: (r['episode_id'], r['time_s']))
        if len(rows) < 2:
            report[session] = {'status': 'reference_return_missing'}
            continue
        first, last = rows[0], rows[-1]
        report[session] = dict(status='diagnostic_only', reference_count=len(rows),
            returned_position_difference_m=(np.asarray(last['position_m'])-first['position_m']).tolist(),
            returned_orientation_difference_rad=angle(last['rotation'], first['rotation']),
            returned_image_error_difference_px=(np.asarray(last['error_px'])-first['error_px']).tolist())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--markers', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.resolve() in (args.predictions.resolve(), args.markers.resolve()):
        p.error('output must differ from input files')
    pairs, rejected, references = build(load_rows(args.predictions), load_rows(args.markers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in pairs))
    report = dict(pairs=len(pairs), rejected=rejected, reference_plateaus=references,
        reference_repeatability=reference_repeatability(references),
        note='Operator-marked stable holds; no automatic grasp/geometry verification')
    write_report(args.output.with_suffix('.manifest.json'), report, [args.predictions, args.markers, Path(__file__)])
    print(json.dumps({k: v for k, v in report.items() if k != 'reference_plateaus'}, indent=2))


if __name__ == '__main__':
    main()
