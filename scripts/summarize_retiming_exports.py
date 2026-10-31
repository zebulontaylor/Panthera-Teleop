#!/usr/bin/env python3
"""Compare completed verified exports; never changes training or dataset rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_training_timing import analyze

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def metrics(audit):
    result = dict(episodes=audit['episodes'], frames=audit['frames'],
                  duration_seconds=audit['baseline_transition_duration_s'])
    for group in ('joints', 'grippers'):
        for kind, key in [('velocity', 'velocity_rad_s'), ('acceleration', 'acceleration_rad_s2')]:
            values = audit[key][group]['frame_max_percentiles']
            result[group + '_' + kind] = {name: values[p] for name, p in [('p95','95'),('p99','99'),('max','100')]}
    return result


def summarize(previous, reference, retimed):
    original, new, transformed = (analyze(path) for path in (previous, reference, retimed))
    verification = read(retimed/'processing/verification.json')
    new_verification = read(reference/'verification.json')
    new_chunk_verification = read(reference/'processing/native-50-chunk-verification.json')
    new_stats_verification = read(reference/'processing/numeric-statistics-independent-verification.json')
    if any(proof.get('passed') is not True for proof in
           (verification, new_verification, new_chunk_verification, new_stats_verification)):
        raise ValueError('Both completed exports need successful independent verification')
    manifest = read(retimed/'processing/manifest.json')
    reports = [read(retimed / ep['retiming_report']) for ep in manifest['episodes']]
    interval_scales = np.concatenate([r['interval_scales'] for r in reports])
    stretched = interval_scales > 1 + 1e-8
    selected = read(ROOT/'docs/retiming-candidate-comparison.json')
    predicted = selected['candidates'][selected['selected_candidate_index']]
    if manifest['retiming']['config'] != predicted['config']:
        raise ValueError('Export differs from candidate configuration')
    actual = metrics(transformed)
    target = metrics(new)
    relative = {}
    for group in ('joints','grippers'):
        for kind in ('velocity','acceleration'):
            feature = group+'_'+kind
            relative[feature] = actual[feature]['p95']/target[feature]['p95']-1
            candidate_feature=('joint' if group=='joints' else 'gripper')+'_'+kind
            if abs(actual[feature]['p95']-predicted['stats'][candidate_feature]['p95']) > 1e-4:
                raise ValueError('Exported numeric percentile differs from candidate prediction')
    added=actual['duration_seconds']-original['baseline_transition_duration_s']
    if abs(actual['duration_seconds']-predicted['duration_seconds']) > 1e-6:
        raise ValueError('Export duration differs from candidate prediction')
    paths=[Path(__file__), ROOT/'scripts/trajectory_retiming.py', ROOT/'scripts/export_retimed_dataset.py',
           ROOT/'scripts/verify_retimed_dataset.py', ROOT/'scripts/prepare_retiming_reference.py',
           ROOT/'scripts/compare_retiming_candidates.py', ROOT/'config/retiming-oct28-reference.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(previous_original=metrics(original), new_reference=target, previous_retimed=actual,
        added_duration_seconds=added, added_duration_percent=100*added/original['baseline_transition_duration_s'],
        p95_relative_difference_from_new_reference=relative,
        protected_source_seconds=sum(r['metrics']['protected_source_duration_seconds'] for r in reports),
        stretched_source_intervals=int(stretched.sum()),
        stretched_source_interval_percent=float(stretched.mean() * 100),
        stretched_original_duration_seconds=float(stretched.sum() / original['fps']),
        max_protected_interval_duration_error_seconds=max(r['metrics']['protected_duration_error_seconds'] for r in reports),
        limits=manifest['retiming']['config'],
        residual_exceedances={name:{scope:sum(r['metrics']['violations'][name][scope] for r in reports)
                                   for scope in ('protected','editable_or_seam')}
                              for name in reports[0]['metrics']['violations']},
        intervals_at_iterative_scale_cap=sum(r['metrics']['intervals_at_scale_cap'] for r in reports),
        max_actual_interval_scale=max(r['metrics']['max_interval_scale'] for r in reports),
        max_camera_source_alignment_seconds=manifest.get('max_retimed_camera_source_alignment_s'),
        verification={label:{key:value for key,value in proof.items() if key in
            ('passed','checked_observations','checked_images','decoded_images','identical_source_jpeg_checks',
             'native_loader_50_action_chunk_checks','native_50_step_chunk_checks',
             'global_statistic_checks','numeric_metadata_verified',
             'episode_boundary_padding_verified','exact_protected_interval_durations_verified')}
            for label,proof in [('new_reference',new_verification),
                                ('new_reference_50_step_chunks',new_chunk_verification),
                                ('new_reference_numeric_statistics',new_stats_verification),
                                ('previous_retimed',verification)]},
        numeric_statistics={label:read(path/'processing/numeric_statistics_refresh.json')
                            for label,path in [('new_reference',reference),('previous_retimed',retimed)]},
        dataset_roots={label:str(path.resolve()) for label,path in [('previous_original',previous),('new_reference',reference),('previous_retimed',retimed)]},
        numeric_sha256={label:audit['numeric_input_sha256'] for label,audit in [('previous_original',original),('new_reference',new),('previous_retimed',transformed)]},
        code_and_config_sha256=hashes, actual_matches_selected_numeric_prediction=True,
        training_changed=False, published=False,
        limitations=['Acceleration is a 20Hz finite-difference estimate, not physical or continuously bounded acceleration.',
                    'Protected map durations are exact; output grid phase shifts by less than one frame.',
                    'Slow intervals and saved insertion windows are protected; other contact is not independently detected.',
                    'Residual protected/seam exceedances are retained and reported, not silently clipped.',
                    'Different task-phase composition limits interpretation of pooled percentile matching.',
                    'Cameras repeat existing source JPEGs; terminal image clamps can add up to one source tick of lag.'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--retimed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=summarize(a.previous,a.reference,a.retimed)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
