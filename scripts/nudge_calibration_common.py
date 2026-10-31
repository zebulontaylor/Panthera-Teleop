"""Offline measured-displacement fits. No robot control or live config writes."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def finite(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('invalid_shape_or_nonfinite')
    return a


def rotation(value):
    r = finite(value, (3, 3))
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r)-1) > 1e-5:
        raise ValueError('invalid_rotation')
    return r


def angle(a, b):
    return float(np.arccos(np.clip((np.trace(rotation(a) @ rotation(b).T)-1)/2, -1, 1)))


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_report(path, report, sources):
    report['verified'] = False
    report['motion_authorized'] = False
    report['source_sha256'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    destination = Path(path)
    if any(destination.resolve() == Path(p).resolve() for p in sources):
        raise ValueError('output_would_overwrite_input')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def clean_pairs(rows):
    kept, rejected = [], Counter()
    seen = set()
    for row in rows:
        try:
            identity = (row['session_id'], row['trial_id'])
            if identity in seen:
                raise ValueError('duplicate_trial')
            seen.add(identity)
            if row['split'] not in ('train', 'validation'):
                raise ValueError('explicit_split_required')
            start, end = row['start'], row['end']
            displacement = finite(end['position_m'], (3,))-finite(start['position_m'], (3,))
            change = finite(end['error_px'], (4,))-finite(start['error_px'], (4,))
            if angle(start['rotation'], end['rotation']) > .003:
                raise ValueError('tool_rotation_changed')
            if np.max(np.abs(finite(end['other_q'], (6,))-finite(start['other_q'], (6,)))) > .003:
                raise ValueError('carrier_moved')
            grippers = finite([start['gripper_position'], end['gripper_position']], (2,))
            if abs(grippers[1]-grippers[0]) > .02:
                raise ValueError('gripper_changed')
            if not .0002 <= np.linalg.norm(displacement) <= .003:
                raise ValueError('displacement_outside_0.2_to_3mm')
            if start['camera_mapping'] != end['camera_mapping'] or start['checkpoint_id'] != end['checkpoint_id']:
                raise ValueError('perception_identity_changed')
            kept.append(dict(row=row, dx=displacement, dy=change))
        except (KeyError, ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
    return kept, dict(rejected)


def robust_fit(x, y):
    weights = np.ones(len(x))
    for _ in range(10):
        root = np.sqrt(weights)[:, None]
        coefficient = np.linalg.lstsq(x*root, y*root, rcond=1e-6)[0]
        residual = np.linalg.norm(y-x @ coefficient, axis=1)
        scale = max(.25, float(np.median(residual))/1.54)
        weights = np.minimum(1., 2.5*scale/np.maximum(residual, 1e-12))
    return coefficient


def fit_response(rows, axis=None):
    pairs, rejected = clean_pairs(rows)
    if axis is not None:
        axis = finite(axis, (3,))
        if abs(np.linalg.norm(axis)-1) > 1e-6:
            raise ValueError('axis_must_be_unit_vector')
        filtered = []
        for p in pairs:
            along = float(np.dot(p['dx'], axis))
            if abs(along) < .0002 or np.linalg.norm(p['dx']-along*axis) > max(.0001, .1*abs(along)):
                rejected['off_axis_or_insufficient_axis_motion'] = rejected.get('off_axis_or_insufficient_axis_motion', 0)+1
            else:
                filtered.append(p)
        pairs = filtered
    report = dict(method='paired_task_axis' if axis is not None else 'cartesian_translation',
        statistical_candidate=False, accepted_pairs=len(pairs), rejected=rejected,
        reasons=[], limitations=['encoder/URDF displacement is not external tool metrology',
            'operator annotations do not verify object retention', 'independent hardware correction still required'])
    if not pairs:
        report['reasons'] = ['no_eligible_pairs']
        return report
    identities = {(json.dumps(p['row']['start']['camera_mapping'], sort_keys=True),
        p['row']['start']['checkpoint_id']) for p in pairs}
    if len(identities) != 1:
        report['reasons'] = ['mixed_camera_or_checkpoint_identities']
        return report
    endpoints = [p['row'][key] for p in pairs for key in ('start', 'end')]
    center = np.median([e['position_m'] for e in endpoints], axis=0)
    other_center = np.median([e['other_q'] for e in endpoints], axis=0)
    if (max(np.linalg.norm(np.asarray(e['position_m'])-center) for e in endpoints) > .01
            or max(angle(e['rotation'], endpoints[0]['rotation']) for e in endpoints) > .02
            or np.max(np.abs(np.asarray([e['other_q'] for e in endpoints])-other_center)) > .03
            or np.ptp([e['gripper_position'] for e in endpoints]) > .04):
        report['reasons'] = ['sessions_do_not_share_a_local_pose_and_grasp_setup']
        return report
    report['reference_position_m'] = center.tolist()
    report['other_arm_reference_q'] = other_center.tolist()
    groups = {split: {p['row']['session_id'] for p in pairs if p['row']['split'] == split}
        for split in ('train', 'validation')}
    if groups['train'] & groups['validation']:
        report['reasons'] = ['session_leakage_between_train_and_validation']
        return report
    episodes = {split: {p['row'].get('episode_id', p['row']['session_id']) for p in pairs
        if p['row']['split'] == split} for split in ('train', 'validation')}
    if episodes['train'] & episodes['validation']:
        report['reasons'] = ['episode_leakage_between_train_and_validation']
        return report
    train = [p for p in pairs if p['row']['split'] == 'train']
    test = [p for p in pairs if p['row']['split'] == 'validation']
    report.update(train_pairs=len(train), validation_pairs=len(test),
        train_sessions=sorted(groups['train']), validation_sessions=sorted(groups['validation']))
    minimum = (8, 4) if axis is not None else (12, 6)
    if len(train) < minimum[0] or len(test) < minimum[1] or len(groups['train']) < 2 or not groups['validation']:
        report['reasons'] = ['insufficient_independent_session_support']
        return report
    x = np.asarray([p['dx'] for p in train])*1000  # fit pixels/mm
    xt = np.asarray([p['dx'] for p in test])*1000
    if axis is not None:
        x, xt = (x @ axis)[:, None], (xt @ axis)[:, None]
        report['axis_base'] = axis.tolist()
        if any(np.sum(values[:, 0] > 0) < 2 or np.sum(values[:, 0] < 0) < 2 for values in (x, xt)):
            report['reasons'] = ['bidirectional_train_and_validation_required']
            return report
        for session in groups['train'] | groups['validation']:
            signed = [float(p['dx'] @ axis) for p in pairs if p['row']['session_id'] == session]
            if sum(s > 0 for s in signed) < 2 or sum(s < 0 for s in signed) < 2:
                report['reasons'] = ['each_session_requires_both_motion_directions']
                return report
    singular = np.linalg.svd(x, compute_uv=False)
    if singular[-1] < .1 or singular[0]/singular[-1] > 30:
        report['reasons'] = ['insufficient_independent_cartesian_excitation']
        return report
    y, yt = (np.asarray([p['dy'] for p in data]) for data in (train, test))
    coefficient = robust_fit(x, y)
    predicted = xt @ coefficient
    axes = [1, 3]  # gear height in both wrist views
    actual, prediction = yt[:, axes], predicted[:, axes]
    rmse = float(np.sqrt(np.mean((actual-prediction)**2)))
    baseline = float(np.sqrt(np.mean(actual**2)))
    eligible = (np.linalg.norm(actual, axis=1) >= 2) & (np.linalg.norm(prediction, axis=1) >= 1)
    direction = float(np.mean(np.sum(actual[eligible]*prediction[eligible], axis=1) > 0)) if eligible.any() else 0.
    report.update(response_px_per_m=(coefficient.T*1000).tolist(), condition=float(singular[0]/singular[-1]),
        heldout_rmse_px=rmse, no_motion_rmse_px=baseline, improvement_fraction=1-rmse/max(baseline, 1e-12),
        heldout_direction_accuracy=direction, eligible_direction_pairs=int(eligible.sum()),
        camera_mapping=train[0]['row']['start']['camera_mapping'], checkpoint_id=train[0]['row']['start']['checkpoint_id'])
    if report['improvement_fraction'] < .2 or direction < .8 or eligible.sum() < 4:
        report['reasons'].append('heldout_response_not_supported')
    if axis is not None:
        slopes = []
        for sign in (-1, 1):
            selected = x[:, 0]*sign > 0
            slopes.append(robust_fit(x[selected], y[selected])[:, axes].ravel())
        disagreement = float(np.linalg.norm(slopes[0]-slopes[1])/max(np.linalg.norm(coefficient[:, axes]), 1e-9))
        report['bidirectional_slope_disagreement_fraction'] = disagreement
        if disagreement > .3:
            report['reasons'].append('direction_dependent_response_or_hysteresis')
    report['statistical_candidate'] = not report['reasons']
    return report
