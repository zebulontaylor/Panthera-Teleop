#!/usr/bin/env python3
"""Analyze a hand-guided, marked gear calibration recording, offline only.

No controller imports, runtime configuration edits, or motion authorization.
The temporal holdout is diagnostic; it is not an independent hardware trial.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from alignment_calibration_fit import _pairs, _regress


def errors(row):
    p = row['predictions']['gear']
    names = p['landmark_names']
    cameras = [p['camera_names'].index(c) for c in ('left_wrist', 'right_wrist')]
    points = np.asarray(p['keypoints_2d'], dtype=float)
    return (points[cameras, names.index('gear_center')] - points[cameras, names.index('slot_center')]).ravel()


def analyze(rows, markers):
    refs = [m for m in markers if m['label'] == 'reference']
    start = next(m['time_s'] for m in markers if m['label'] == 'motion-start')
    end = next(m['time_s'] for m in markers if m['label'] == 'motion-end')
    reference_rows = [[r for r in rows if m['time_s']-1 <= r['time_s'] <= m['time_s']]
                      for m in refs]
    if any(not group for group in reference_rows):
        raise ValueError('No synchronized predictions at a marked reference')
    targets = [np.median([errors(r) for r in group], axis=0) for group in reference_rows]
    qref = np.r_[refs[0]['arms']['left']['position_rad'], refs[0]['arms']['right']['position_rad']]
    motion = [r for r in rows if start <= r['time_s'] <= end]
    samples = [dict(episode_id=r['episode_id'], time_s=r['time_s'],
                    q_left=r['q_left'], q_right=r['q_right'], error_px=errors(r).tolist()) for r in motion]
    pairs, rejected = _pairs(samples, .2, .5)
    report = dict(verified=False, candidate_usable=False, scope='single hand-guided session; offline diagnostics',
                  reference_error_px=[v.tolist() for v in targets],
                  reference_return_difference_px=(targets[-1]-targets[0]).tolist(),
                  reference_joint_positions_rad=qref.tolist(), synchronized_motion_samples=len(samples),
                  motion_pair_count=len(pairs), rejected_samples=rejected,
                  remaining_requirements=['independent correction validation', 'validated runtime grasp and visibility evidence'])
    if not pairs:
        report['fit_status'] = 'no_motion_pairs'
        return report
    dx = np.asarray([p[2]-p[1] for p in pairs])
    dy = np.asarray([p[3] for p in pairs])
    q = np.asarray([r['q_left']+r['q_right'] for r in motion])
    report.update(max_joint_excursion_rad=np.max(np.abs(q-qref), axis=0).tolist(),
                  left_motion_rms_rad=float(np.sqrt(np.mean(dx[:,:6]**2))),
                  right_motion_rms_rad=float(np.sqrt(np.mean(dx[:,6:]**2))))
    quality = []
    for r in rows:
        p = r['predictions']['gear']
        cams = [p['camera_names'].index(c) for c in ('left_wrist', 'right_wrist')]
        ids = [p['landmark_names'].index(n) for n in ('gear_center', 'slot_center')]
        quality.append(all(np.all(np.asarray(p[field])[np.ix_(cams, ids)] >= threshold)
                           if field != 'pixel_std' else np.all(np.asarray(p[field])[np.ix_(cams, ids)] <= threshold)
                           for field, threshold in [('visibility_probability', .85), ('confidence', .5), ('pixel_std', 8.)]))
    report['runtime_quality_pass_count'] = int(sum(quality))
    report['evaluated_prediction_count'] = len(rows)
    report['left_gripper_abs_torque_range_nm'] = [float(v) for v in np.quantile(
        [abs(r['arms']['left']['gripper']['torque']) for r in motion], [0, 1])]
    if len(pairs) < 24:
        report['fit_status'] = 'insufficient_pairs_for_fit_and_temporal_holdout'
        return report
    cut = max(15, int(len(pairs)*.67))
    model = _regress(dx[:cut,:6], dx[:cut,6:], dy[:cut])
    prediction = dx[cut:,:6] @ model['coefficient'] + np.c_[dx[cut:,6:], np.ones(len(dx)-cut)] @ model['nuisance']
    actual = dy[cut:][:,[1,3]]
    predicted = prediction[:,[1,3]]
    rmse = float(np.sqrt(np.mean((actual-predicted)**2)))
    baseline = float(np.sqrt(np.mean(actual**2)))
    report.update(fit_status='diagnostic_fit_only', train_pairs=cut, temporal_holdout_pairs=len(dx)-cut,
                  independently_excited_rank=model['rank'],
                  conditional_excitation_rms_rad=model['excitation'].tolist(),
                  image_jacobian_px_per_rad=model['coefficient'].T.tolist(),
                  heldout_height_rmse_px=rmse, no_motion_height_rmse_px=baseline,
                  heldout_improvement_fraction=1-rmse/baseline if baseline else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--markers', required=True, type=Path)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['episode_id'] == args.episode]
    markers = [json.loads(line) for line in args.markers.read_text().splitlines() if line.strip()]
    markers = [m for m in markers if m['episode_id'] == args.episode]
    result = analyze(rows, markers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
