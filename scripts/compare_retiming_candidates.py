#!/usr/bin/env python3
"""Compare numeric-only retiming candidates against a new-method reference.

No images, source recordings, exports, or training configuration are changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from trajectory_retiming import retime, JOINTS, GRIPPERS
from analyze_training_timing import derivatives

CANDIDATES = [(.8, 6., 2.5, 19.), (.8, 10., 3., 40.),
              (.8, 20., 3., 40.), (.75, 20., 3., 40.),
              (.8, 20., 3.2, 50.), (.85, 20., 3.2, 50.)]


def pooled_summary(paths, fps):
    velocity = np.concatenate([np.diff(q, axis=0) * fps for q in paths])
    acceleration = np.concatenate([np.diff(q, axis=0, n=2) * fps**2 for q in paths])
    summary = {}
    for name, values in [('velocity', velocity), ('acceleration', acceleration)]:
        for group, axes in [('joint', JOINTS), ('gripper', GRIPPERS)]:
            x = np.abs(values[:, axes]).max(axis=1)
            summary[group + '_' + name] = {f'p{p}': float(np.percentile(x, p)) for p in [50, 90, 95, 99]}
            summary[group + '_' + name]['max'] = float(x.max())
    return summary


def compare(source, reference):
    manifest = json.loads((source/'processing/manifest.json').read_text())
    fps = manifest['fps']
    columns = ['episode_index', 'frame_index', 'observation.state', 'action']
    parts = [pq.read_table(p, columns=columns).to_pydict() for p in sorted((source/'data').rglob('*.parquet'))]
    data = {key: np.concatenate([np.asarray(t[key]) for t in parts]) for key in columns}
    episodes = []
    for e in manifest['episodes']:
        mask = data['episode_index'] == e['episode_index']
        order = np.argsort(data['frame_index'][mask])
        states, actions = data['observation.state'][mask][order], data['action'][mask][order]
        derivatives(states, actions, fps)
        poses = np.r_[states, actions[-1:]]
        alignment = pq.read_table(source/f"processing/alignment/episode_{e['episode_index']:06d}.parquet", columns=['source_episode_time_s']).to_pydict()
        times = np.r_[alignment['source_episode_time_s'], alignment['source_episode_time_s'][-1]+1/fps]
        protected = np.zeros(len(poses)-1, dtype=bool)
        for phase in e.get('precision_phases', []):
            protected |= (times[:-1] < phase['end_s']) & (times[1:] > phase['start_s'])
        episodes.append((poses, protected))
    target = {}
    new = json.loads(reference.read_text())['new_reference']
    for feature in ['joint_velocity', 'joint_acceleration', 'gripper_velocity', 'gripper_acceleration']:
        group, kind = feature.split('_')
        target[feature] = new['velocity_rad_s' if kind == 'velocity' else 'acceleration_rad_s2']['joints' if group=='joint' else 'grippers']['frame_max_percentiles']['95']
    baseline_duration = sum((len(q)-1)/fps for q,_ in episodes)
    results = []
    for limits in CANDIDATES:
        config = dict(zip(['joint_velocity','joint_acceleration','gripper_velocity','gripper_acceleration'],limits),
                      protect_joint_speed=.15, protect_gripper_speed=.3, max_iterations=40, max_scale=4.)
        outputs = [retime(q, fps=fps, protected_mask=protected, **config) for q,protected in episodes]
        stats = pooled_summary([o['poses'] for o in outputs], fps)
        duration = sum(o['metrics']['after']['duration_seconds'] for o in outputs)
        relative = {feature: stats[feature]['p95']/target[feature]-1 for feature in target}
        # Choose approximate similarity rather than treating reference p95 as a
        # hard acceleration ceiling (resampling itself changes finite differences).
        score = sum(abs(np.log(1+x)) for x in relative.values())/4 + .1*(duration/baseline_duration-1)
        result = dict(config=config, stats=stats, duration_seconds=duration,
            added_duration_seconds=duration-baseline_duration, added_duration_percent=100*(duration/baseline_duration-1),
            p95_relative_error_to_reference=relative, comparison_score=float(score),
            protected_source_seconds=sum(o['metrics']['protected_source_duration_seconds'] for o in outputs),
            max_protected_duration_error_seconds=max(o['metrics']['protected_duration_error_seconds'] for o in outputs),
            editable_or_seam_violations={key:sum(o['metrics']['violations'][key]['editable_or_seam'] for o in outputs) for key in target},
            protected_violations={key:sum(o['metrics']['violations'][key]['protected'] for o in outputs) for key in target},
            intervals_at_scale_cap=sum(o['metrics']['intervals_at_scale_cap'] for o in outputs))
        results.append(result)
        print(json.dumps(result), flush=True)
    selected = min(range(len(results)), key=lambda i:results[i]['comparison_score'])
    return dict(source=str(source.resolve()), reference=str(reference.resolve()),
        reference_p95=target, source_duration_seconds=baseline_duration,
        original=pooled_summary([q for q,_ in episodes], fps), candidates=results,
        selected_candidate_index=selected, selected_config=results[selected]['config'],
        selection_method='Minimum mean absolute log ratio of four pooled frame-max p95 values to reference, plus 0.1 times fractional added duration. Six fixed exploratory candidates; no training-loss optimization.',
        limitations=['20Hz finite differences, not physical acceleration.',
                     'Reference and old episodes differ in phase composition; matching p95 does not establish behavioral equivalence.',
                     'Slow intervals and annotated precision windows fixed exactly in source clock; protected/seam violations reported.',
                     '4x per-interval cap bounds pathological slowdown; this is not a hard-limit trajectory planner.'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selected-config',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.selected_config.exists():raise FileExistsError('Choose fresh outputs')
    report=compare(a.source,a.reference)
    for path,content in [(a.output,report),(a.selected_config,report['selected_config'])]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(json.dumps(content,indent=2,allow_nan=False)+'\n')
