#!/usr/bin/env python3
"""Prepare offline phase transitions with weak demonstration milestones.

Closure is a measured gripper event, not grasp success. Insertion endpoints are
reviewed approach milestones, not physical seating. Actions are measured-pose
deltas, never logged motor commands. All Oct22 holdouts remain within-session.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.alignment_q_features import encode_observation, feature_spec


def pose14(row):
    arms = row['arms']
    values = [float(v) for side in ('left', 'right') for v in arms[side]['position_rad']]
    values += [float(arms[side]['gripper']['position']) for side in ('left', 'right')]
    if len(values) != 14 or not all(math.isfinite(v) for v in values):
        raise ValueError('expected finite measured 14-dimensional pose')
    return values


def episode_splits(ids, seed=19):
    ids = sorted(set(ids)); random.Random(seed).shuffle(ids)
    # 17 episodes -> 11/3/3. Never describe this as a session split.
    held = max(1, round(len(ids) * .18)) if len(ids) >= 3 else 0
    return {eid: ('test' if i < held else 'val' if i < 2 * held else 'train')
            for i, eid in enumerate(ids)}


def pickup_bounds(rows, lower, insertion_start, max_gap=.15):
    """Last stable downward closure after a genuinely open prior observation."""
    candidates = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        t = float(b['time_s'])
        if not lower <= t <= insertion_start or t - float(a['time_s']) > max_gap + 1e-8:
            continue
        ga = float(a['arms']['left']['gripper']['position'])
        gb = float(b['arms']['left']['gripper']['position'])
        if not ga >= .7 > gb:
            continue
        start = max(lower, t - 2.)
        prior = [r for r in rows[:i] if start <= float(r['time_s']) < t]
        after = []
        for future in rows[i:]:
            after.append(future)
            if float(future['time_s']) >= t + .2 - 1e-8:
                break
        if not prior or not any(float(r['arms']['left']['gripper']['position']) > .9 for r in prior):
            continue
        if not after or float(after[-1]['time_s']) < t + .2 - 1e-6:
            continue
        if any(float(r['arms']['left']['gripper']['position']) >= .7 for r in after):
            continue
        if any(float(y['time_s']) - float(x['time_s']) > max_gap + 1e-8 for x, y in zip(after, after[1:])):
            continue
        candidates.append((start, t))
    return candidates[-1] if candidates else None


def build_transitions(predictions, manifest, *, seed=19, max_gap=.15, endpoint_tolerance=.15):
    """Return transitions and an audit; inputs remain untouched.

    Reject an entire phase if its endpoint is missing or any internal frame gap
    exceeds the limit. Consequently truncated/gapped segments never get rewards.
    """
    if not 0 < max_gap <= .15 or not 0 < endpoint_tolerance <= .15:
        raise ValueError('gap and endpoint tolerance must be in (0,.15]')
    grouped = defaultdict(list)
    for row in predictions:
        if not math.isfinite(float(row['time_s'])):
            raise ValueError('nonfinite sample time')
        grouped[row['episode_id']].append(row)
    selections = manifest['episodes']
    ids = [s['id'] for s in selections]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate manifest episode')
    splits = episode_splits(ids, seed)
    rejected, bounds, output = Counter(), [], []
    for selection in selections:
        eid = selection['id']; rows = sorted(grouped[eid], key=lambda r: float(r['time_s']))
        phase_bounds = {}
        for task in ('gear', 'pin'):
            interval = selection[task]
            start, end = float(interval['start_s']), float(interval['end_s'])
            if not 0 <= start < end:
                raise ValueError('invalid reviewed bounds')
            phase_bounds[task + '_insertion'] = (start, end)
            lower = 0. if task == 'gear' else float(selection['gear']['end_s'])
            pickup = pickup_bounds(rows, lower, start, max_gap)
            if pickup is None:
                rejected[task + '_pickup:missing_stable_closure'] += 1
            else:
                phase_bounds[task + '_pickup'] = pickup
        for phase, (start, end) in phase_bounds.items():
            seq = [r for r in rows if start - 1e-8 <= float(r['time_s']) <= end + 1e-8]
            record = dict(episode_id=eid, phase=phase, start_s=start, end_s=end,
                          milestone_kind='weak_gripper_closure' if phase.endswith('pickup') else 'weak_reviewed_approach_endpoint')
            bounds.append(record)
            if len(seq) < 2:
                rejected[phase + ':insufficient_frames'] += 1; continue
            if end - float(seq[-1]['time_s']) > endpoint_tolerance + 1e-8:
                rejected[phase + ':missing_endpoint'] += 1; continue
            times = [float(r['time_s']) for r in seq]
            if any(not 0 < b-a <= max_gap + 1e-8 for a, b in zip(times, times[1:])):
                rejected[phase + ':gap_or_duplicate'] += 1; continue
            if any(abs((b-a)-.1) > 1e-6 for a, b in zip(times, times[1:])):
                rejected[phase + ':irregular_action_cadence'] += 1; continue
            try:
                poses = [pose14(r) for r in seq]
                features = [encode_observation(r, r['arms'], seq[i-1] if i else None,
                            times[i]-times[i-1] if i else None) for i, r in enumerate(seq)]
            except (ValueError, KeyError, TypeError) as exc:
                rejected[phase + ':invalid_features'] += 1
                record['rejection_detail'] = str(exc)
                continue
            for i in range(len(seq)-1):
                terminal = i == len(seq)-2
                output.append(dict(episode_id=eid, session_id=eid[:8], split=splits[eid],
                    trajectory_id=eid+':'+phase, phase=phase, t=times[i],
                    observation=features[i], next_observation=features[i+1],
                    action=[b-a for a,b in zip(poses[i],poses[i+1])], reward=float(terminal),
                    terminal=terminal, dt=times[i+1]-times[i],
                    action_provenance='measured_next_pose_delta_proxy',
                    reward_kind=record['milestone_kind']))
            record['accepted_frames'] = len(seq)
    audit = dict(schema_version=1, samples=len(output), counts_by_split=dict(Counter(r['split'] for r in output)),
        counts_by_phase=dict(Counter(r['phase'] for r in output)), rejected=dict(rejected), phase_bounds=bounds,
        episode_splits=splits, split_seed=seed, split_unit='episode',
        evaluation_scope='Within-session episode holdout; no cross-session generalization estimate.',
        session_id_provenance='First 8 recording ID characters: coarse UTC recording day; not independent sessions.',
        reward_semantics='Weak demonstrated milestone attainment, never verified grasp or insertion success.',
        action_semantics='14 measured next-pose minus current-pose radians: left joints, right joints, left gripper, right gripper; not sent commands.',
        max_gap_s=max_gap, endpoint_tolerance_s=endpoint_tolerance, action_horizon_s=.1,
        action_horizon_tolerance_s=1e-6,
        feature_spec=feature_spec(), camera_mapping=feature_spec()['camera_mapping'],
        checkpoint_ids=feature_spec()['checkpoint_ids'])
    return output, audit


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):h.update(block)
    return h.hexdigest()



def validate_source_manifest(predictions_path):
    """Require a finished, content-addressed extraction with expected models."""
    predictions_path = Path(predictions_path)
    sidecar = predictions_path.with_suffix('.manifest.json')
    manifest = json.loads(sidecar.read_text())
    if manifest.get('complete') is not True:
        raise ValueError('prediction extraction manifest must be complete')
    if manifest.get('output_sha256') != sha256(predictions_path):
        raise ValueError('prediction extraction output SHA256 mismatch')
    spec = feature_spec()
    if manifest.get('camera_mapping') != spec['camera_mapping']:
        raise ValueError('prediction extraction camera mapping mismatch')
    for task, expected in spec['checkpoint_ids'].items():
        if manifest.get('checkpoints', {}).get(task, {}).get('sha256') != expected:
            raise ValueError('prediction extraction checkpoint SHA256 mismatch: ' + task)
    return sidecar


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,default=ROOT/'work/alignment-q-20261030/predictions.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'work/alignment-q-20261030/transitions.jsonl')
    parser.add_argument('--phase-manifest',type=Path,default=ROOT/'config/calibration/recorded-alignment-near-trials.json')
    args=parser.parse_args()
    source_sidecar=validate_source_manifest(args.predictions)
    predictions=[json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    rows,audit=build_transitions(predictions,json.loads(args.phase_manifest.read_text()))
    audit['sources']={str(p.resolve()):sha256(p) for p in (args.predictions,args.phase_manifest,source_sidecar)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    audit['output_sha256']=sha256(args.output)
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:audit[k] for k in ('samples','counts_by_split','counts_by_phase','rejected')},indent=2))

if __name__=='__main__':main()
