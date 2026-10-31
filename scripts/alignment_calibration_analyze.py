#!/usr/bin/env python3
"""Produce reproducible offline calibration diagnostics, never enable motion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .alignment_calibration_fit import fit_local
except ImportError:
    from alignment_calibration_fit import fit_local


def _quantiles(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), np.quantile(a, [0, .1, .5, .9, 1]).tolist())) if len(a) else None


def _fit(samples, side, task):
    # Pick the densest local region rather than a median between disjoint poses.
    reference = None
    if samples:
        q = np.asarray([s['q_left']+s['q_right'] for s in samples])
        references = [np.median(q, axis=0)]
        references.extend(np.median(q[[s['episode_id'] == ep for s in samples]], axis=0)
                          for ep in sorted({s['episode_id'] for s in samples}))
        reference = max(references, key=lambda center: int(np.sum(np.max(np.abs(q-center), axis=1) <= .3)))
    return fit_local(samples, side, task, local_radius_rad=.3,
                     reference_joint_positions=reference)


def analyze(predictions_path, audit_path):
    prediction_bytes = Path(predictions_path).read_bytes()
    audit_bytes = Path(audit_path).read_bytes()
    prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    rows = [json.loads(line) for line in prediction_bytes.decode().splitlines() if line.strip()]
    audit = json.loads(audit_bytes)
    episodes = {ep['id']: ep for ep in audit['episodes']}
    physical = {f'{arm}_wrist': camera for camera, arm in audit['mapping'].items() if arm in ('left', 'right')}
    side = audit['moving_arm']
    manifest_path = Path(predictions_path).with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    identity_errors = []
    if manifest.get('complete') is not True:
        identity_errors.append('cache_not_marked_complete')
    if manifest.get('output_sha256') != prediction_sha:
        identity_errors.append('cache_output_sha256_mismatch_or_missing')
    if manifest.get('samples') != len(rows):
        identity_errors.append('cache_sample_count_mismatch_or_missing')
    if {r['episode_id'] for r in rows} != set(episodes):
        identity_errors.append('cache_episode_set_does_not_match_audit')
    phase_selection = manifest.get('phase_selection')
    if phase_selection is not None and phase_selection.get('sha256') != audit_sha:
        identity_errors.append('phase_selection_audit_sha256_mismatch')
    if manifest.get('sampling_scope') == 'audited_phase_union' and not phase_selection:
        identity_errors.append('phase_selection_identity_missing')
    report = dict(format_version=1, verified=False, motion_enabled=False,
        analysis_complete=not identity_errors,
        input_identity_errors=identity_errors,
        analysis_complete_definition='Cache complete, bytes/count/episode set match its manifest and phase audit hash matches when applicable; not a claim of full-duration dense sampling.',
        task_analysis_scope='Only audited gear and pin phase intervals contribute to quality counts and regression.',
        input_sampling_scope=manifest.get('sampling_scope', manifest.get('sample_scope', 'See input manifest fps and sample counts.')),
        provenance=dict(predictions_path=str(Path(predictions_path).resolve()), predictions_sha256=prediction_sha,
                        audit_path=str(Path(audit_path).resolve()), audit_sha256=audit_sha,
                        manifest=manifest, row_count=len(rows), episode_count=len({r['episode_id'] for r in rows}),
                        physical_camera_mapping=physical, moving_arm=side,
                        original_recordings_modified=False),
        gate_thresholds=dict(min_visibility=.85, min_confidence=.5, max_pixel_std=8.,
                             max_target_distance_px=60., max_state_skew_s=.01,
                             max_camera_pair_skew_s=.06, gripper_contact_torque_nm=.1,
                             gripper_closed_position_rad=1.3, max_gripper_velocity_rad_s=.1),
        limitations=['Coarse task intervals are not per-frame grasp or successful insertion ground truth.',
                     'CNN confidence and uncertainty heads must pass existing runtime thresholds.',
                     'Diagnostic fits include rejected CNN outputs and cannot qualify robot motion.',
                     'Pin bore association is unverified; no user-selected fixed physical bore ROI exists.',
                     'Measured passive-gripper torque is not proof of retained object grasp.',
                     'Offline held-out prediction is not independent hardware validation.'], tasks={})
    for task in ('gear', 'pin'):
        diagnostic, strict = [], []
        counters = Counter()
        heads = defaultdict(list)
        selected = []
        for row in rows:
            episode = episodes.get(row['episode_id'])
            if not episode or not episode[task]['start_s'] <= row['time_s'] <= episode[task]['end_s']:
                continue
            prediction = row['predictions'][task]
            # camera_names in the network are slot labels. Associate each slot
            # through this cache's input mapping before assigning physical sides.
            slots = prediction['camera_names']
            indices = [next(i for i, name in enumerate(slots) if row['camera_mapping'][name] == physical[physical_name])
                       for physical_name in ('left_wrist', 'right_wrist')]
            selected.append((row, prediction, indices))
        # Fix each diagnostic pin target head near the audited phase endpoint.
        # This prevents nearest-target switching inside temporal regression pairs.
        pin_targets = {}
        if task == 'pin':
            for episode_id in sorted({r[0]['episode_id'] for r in selected}):
                choices = [r for r in selected if r[0]['episode_id'] == episode_id]
                endpoint = min(choices, key=lambda r: abs(r[0]['time_s']-episodes[episode_id][task]['reference_s']))
                points = np.asarray(endpoint[1]['keypoints_2d'])
                names = endpoint[1]['landmark_names']
                moving = names.index('pin_tip')
                candidates = [names.index(f'carrier_hole_{n}') for n in range(1, 5)]
                pin_targets[episode_id] = [min(candidates, key=lambda n: np.linalg.norm(points[i, moving]-points[i, n])) for i in endpoint[2]]
        for row, prediction, indices in selected:
            counters['phase_frames'] += 1
            names = prediction['landmark_names']
            moving = names.index('gear_center' if task == 'gear' else 'pin_tip')
            target = [names.index('slot_center')]*2 if task == 'gear' else pin_targets[row['episode_id']]
            points = np.asarray(prediction['keypoints_2d'], dtype=float)
            moving_px = points[indices, moving]
            target_px = points[indices, target]
            error = moving_px-target_px
            fields = {}
            for field in ('visibility_probability', 'confidence', 'pixel_std'):
                values = np.asarray(prediction[field], dtype=float)
                fields[field] = np.asarray([[values[i, moving], values[i, target[n]]] for n, i in enumerate(indices)])
                for n, camera in enumerate(('left_wrist', 'right_wrist')):
                    for landmark, value in zip(('moving', 'target'), fields[field][n]):
                        heads[f'{camera}.{landmark}.{field}'].append(value)
            finite = np.isfinite(np.r_[moving_px.ravel(), target_px.ravel()]).all()
            visible = bool(np.min(fields['visibility_probability']) >= .85)
            confident = bool(np.min(fields['confidence']) >= .5)
            precise = bool(np.max(fields['pixel_std']) <= 8.)
            close = bool(np.max(np.linalg.norm(error, axis=1)) <= 60.)
            gripper = row['arms'][side]['gripper']
            held = bool(gripper['position'] <= 1.3 and abs(gripper['torque']) >= .1 and abs(gripper['velocity']) <= .1)
            image_times = [row['frame_timestamps_ns'][physical[name]]/1e9 for name in ('left_wrist', 'right_wrist')]
            synchronized = bool(abs(row.get('state_skew_s', float('inf'))) <= .01 and max(image_times)-min(image_times) <= .06)
            for name, condition in (('finite_landmarks', finite), ('both_visible', visible), ('both_confident', confident),
                                    ('both_precise', precise), ('targets_close', close), ('gripper_contact_evidence', held),
                                    ('synchronized', synchronized)):
                counters[name] += int(condition)
            sample = dict(episode_id=row['episode_id'], time_s=row['time_s'], q_left=row['q_left'], q_right=row['q_right'],
                          error_px=error.ravel().tolist(), moving_px=moving_px.tolist(), target_px=target_px.tolist())
            if finite and synchronized:
                diagnostic.append(sample)
            if finite and synchronized and visible and confident and precise and close and held:
                strict.append(sample)
        counters['diagnostic_samples'] = len(diagnostic)
        counters['strict_samples'] = len(strict)
        strict_fit = _fit(strict, side, task)
        diagnostic_fit = _fit(diagnostic, side, task)
        strict_fit['statistical_candidate_usable'] = strict_fit['candidate_usable']
        strict_fit['candidate_usable'] = False
        strict_fit['reasons'].append('independent_grasp_jaw_roi_and_hardware_validation_missing')
        if task == 'pin':
            strict_fit['candidate_usable'] = False
            strict_fit['reasons'].append('physical_bore_association_unverified')
        # Explicitly disqualify the diagnostic model even if its regression fits.
        diagnostic_fit['statistical_candidate_usable'] = diagnostic_fit['candidate_usable']
        diagnostic_fit['candidate_usable'] = False
        diagnostic_fit['reasons'].append('diagnostic_ungated_landmarks_and_unverified_grasp')
        if task == 'pin':
            diagnostic_fit['reasons'].append('physical_bore_association_unverified')
        report['tasks'][task] = dict(counts=dict(counters),
            phase_episode_count=len({r[0]['episode_id'] for r in selected}),
            checkpoint_ids=sorted({r[1]['checkpoint_id'] for r in selected}),
            head_quantiles={name: _quantiles(values) for name, values in sorted(heads.items())},
            strict_fit=strict_fit, diagnostic_fit=diagnostic_fit,
            target_association='CNN slot_center head' if task == 'gear' else 'association_unverified: nearest endpoint head held fixed per episode/view',
            diagnostic_target_heads_by_episode=pin_targets)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.predictions, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({task: data['counts'] for task, data in report['tasks'].items()}, indent=2))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
