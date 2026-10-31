#!/usr/bin/env python3
"""Offline controller gate replay. Never imports hardware or changes calibration.

The TEST fixture deliberately marks calibration verified only inside this process
so recorded sensor gates can be inspected. Its geometry is incomplete and is NOT
motion authorization, deployment calibration, or measured grasp evidence.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from alignment_assist import AlignmentAssist


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tick(controller, snapshot, arms, limits, mode='recording', now=None):
    """Mirror hand_guide caller's mode exclusion; no hardware commands exist here."""
    side = controller.config['calibration'].get('side', 'left')
    return controller.evaluate(now if now is not None else snapshot['monotonic_ns']/1e9,
        snapshot, arms[side], limits, other_arm=arms['right' if side == 'left' else 'left'],
        collecting=mode == 'recording')


def synthetic_fixture(task='gear'):
    mapping = {'left_wrist': 'wrist_port3', 'right_wrist': 'wrist_port2'}
    names = (['gear_center', 'slot_center'] if task == 'gear' else
             ['pin_tip', 'pin_top'] + [f'carrier_hole_{i}' for i in range(1, 5)])
    cal = dict(verified=True, side='left', camera_names=list(mapping), camera_mapping=mapping,
        checkpoint_id='SYNTHETIC-TEST-ONLY', image_size=[640, 480],
        reference_joint_position_rad=[0.]*6, other_arm_reference_joint_position_rad=[0.]*6,
        max_joint_distance_rad=.3, image_jacobian_px_per_rad=(np.eye(4, 6)*100).tolist(),
        jaw_rois_px={c: [80, 80, 140, 140] for c in mapping},
        target_rois_px={c: [95, 95, 105, 105] for c in mapping},
        gripper_closed_position_rad=1.3, gripper_contact_torque_nm=.1,
        target_landmark='slot_center' if task == 'gear' else 'carrier_bore',
        target_error_px=[0.]*4, residual_axes=[1, 3] if task == 'gear' else [0, 1, 2, 3])
    points = np.full((3, len(names), 2), 100.)
    points[:, 0] = [108, 112]
    points[2, 0] = [112, 108]
    if task == 'pin':
        points[:, 3:] = [300, 300]
    prediction = dict(checkpoint_id=cal['checkpoint_id'], image_size=[640, 480],
        camera_names=['overhead', *mapping], landmark_names=names, keypoints_2d=points.tolist(),
        visibility_probability=np.ones((3, len(names))).tolist(),
        confidence=np.ones((3, len(names))).tolist(), pixel_std=np.ones((3, len(names))).tolist())
    snapshot = dict(status='ready', camera_mapping=mapping, predictions={task: prediction})
    arm = dict(position_rad=[0.]*6, velocity_rad_s=[0.]*6,
               gripper=dict(position=1., torque=.2, velocity=0.))
    arms = {'left': deepcopy(arm), 'right': deepcopy(arm)}
    limits = dict(lower=[-2.]*6, upper=[2.]*6)
    advance(snapshot, 10.)
    return AlignmentAssist(dict(enabled=True, task=task, calibration=cal)), snapshot, arms, limits


def advance(snapshot, now):
    snapshot['monotonic_ns'] = round(now*1e9)
    snapshot['frame_timestamps_ns'] = {p: round(now*1e9) for p in snapshot['camera_mapping'].values()}


def activate(task):
    controller, snapshot, arms, limits = synthetic_fixture(task)
    assert tick(controller, snapshot, arms, limits)['reason'] == 'confirming_grasp'
    advance(snapshot, 10.21)
    result = tick(controller, snapshot, arms, limits)
    assert result['active'] and np.linalg.norm(result['torque_nm']) > 0, result
    return controller, snapshot, arms, limits, result


def fault_checks():
    results = []
    cases = ['stale', 'skewed', 'missing', 'nan', 'visibility_loss', 'checkpoint_swap',
             'camera_order_changed', 'no_grasp', 'near_joint_limits', 'return', 'hold',
             'gripper_reset', 'nonrecording', 'disable', 'budget_exhausted']
    for task in ('gear', 'pin'):
        for case in cases:
            c, s, a, limits, positive = activate(task)
            mode, now = 'recording', 10.22
            advance(s, now)
            p = s['predictions'][task]
            expected = {
                'stale': 'stale_prediction', 'skewed': 'unsynchronized_cameras',
                'missing': 'invalid_evidence_or_calibration', 'nan': 'invalid_landmarks',
                'visibility_loss': 'targets_not_visible_or_confident', 'checkpoint_swap': 'checkpoint_mismatch',
                'camera_order_changed': 'camera_mapping_mismatch', 'no_grasp': 'gripper_not_holding',
                'near_joint_limits': 'joint_limit_margin', 'return': 'not_collecting', 'hold': 'not_collecting',
                'gripper_reset': 'not_collecting', 'nonrecording': 'not_collecting', 'disable': 'disabled',
                'budget_exhausted': 'nudge_budget_exhausted'}[case]
            if case == 'stale':
                now += .3
            elif case == 'skewed':
                s['frame_timestamps_ns']['wrist_port3'] -= 80_000_000
            elif case == 'missing':
                del p['keypoints_2d']
            elif case == 'nan':
                p['keypoints_2d'][1][0][0] = float('nan')
            elif case == 'visibility_loss':
                p['visibility_probability'][1][0] = 0.
            elif case == 'checkpoint_swap':
                p['checkpoint_id'] = 'changed'
            elif case == 'camera_order_changed':
                s['camera_mapping'] = {'left_wrist': 'wrist_port2', 'right_wrist': 'wrist_port3'}
            elif case == 'no_grasp':
                a['left']['gripper']['torque'] = 0.
            elif case == 'near_joint_limits':
                a['left']['position_rad'][0] = 1.99
            elif case in ('return', 'hold', 'gripper_reset', 'nonrecording'):
                mode = case
            elif case == 'disable':
                c.set_enabled(False)
            elif case == 'budget_exhausted':
                now = 10.511
                advance(s, now)
            result = tick(c, s, a, limits, mode, now)
            assert result['reason'] == expected, (task, case, result)
            assert not result['active'] and result['torque_nm'] == [0.]*6, result
            assert result['delta_rad'] == [0.]*6
            assert np.linalg.norm(positive['delta_rad']) <= .006000001
            assert max(abs(x) for x in positive['torque_nm']) <= .15
            results.append(dict(task=task, fault=case, positive_baseline_active=True,
                reason=result['reason'], zero_torque=True))
    return results


def geometry_fixture(task, report, observations):
    """Partial diagnostic geometry. Unavailable manual fields stay documented."""
    fit = report['tasks'][task]['diagnostic_fit']
    manifest = report['provenance']['manifest']
    mapping = {k: manifest['camera_mapping'][k] for k in ('left_wrist', 'right_wrist')}
    moving = 'gear_center' if task == 'gear' else 'pin_tip'
    phase = 'gear' if task == 'gear' else 'pin_approach'
    train = [o for o in observations if o['split'] == 'train' and o['phase'] == phase]
    jaw, source_ids, missing = {}, {}, []
    for camera, port in mapping.items():
        pts = [(o['observation_id'], o['labels']['views'][port]['landmarks'][moving]['xy_px'])
               for o in train if o['labels']['views'][port]['landmarks'][moving]['xy_px'] is not None]
        source_ids[camera] = [v[0] for v in pts]
        if pts:
            xy = np.array([p[1] for p in pts])
            jaw[camera] = [*np.maximum(0, xy.min(0)-15), *np.minimum([640, 480], xy.max(0)+15)]
        else:
            jaw[camera] = [0, 0, 640, 480]
            missing.append(f'{camera}: no visible {moving} jaw labels; FULL-IMAGE TEST PLACEHOLDER')
    # No slot-center labels and no dual-view bore labels exist. Never infer true
    # insertion alignment from the CNN's own coordinate output.
    missing.append('target_error_px unmeasured: ZERO TEST PLACEHOLDER, not calibrated alignment')
    if task == 'pin':
        missing.append('target_rois_px unmeasured: FULL-IMAGE TEST PLACEHOLDERS; physical bore identity unverified')
    cal = {k: deepcopy(fit[k]) for k in ('side', 'camera_names', 'reference_joint_position_rad',
        'other_arm_reference_joint_position_rad', 'max_joint_distance_rad', 'image_jacobian_px_per_rad', 'residual_axes')}
    cal.update(verified=True, camera_mapping=mapping, checkpoint_id=report['tasks'][task]['checkpoint_ids'][0],
        image_size=[640, 480], jaw_rois_px={k: list(map(float, v)) for k, v in jaw.items()},
        gripper_closed_position_rad=1.3, gripper_contact_torque_nm=.1,
        target_landmark='slot_center' if task == 'gear' else 'carrier_bore',
        target_error_px=[0.]*4, target_rois_px={c: [0, 0, 640, 480] for c in mapping})
    return cal, dict(label_source_ids=source_ids, unavailable_geometry=missing,
        warning='ISOLATED TEST FIXTURE ONLY. verified=True is process-local gate reachability, not verification.',
        gripper_threshold_source='synthetic controller contract fixture; passive torque has no validated grasp threshold',
        jacobian_source='prior all-trial diagnostic fit, not held-out calibration evaluation')


def replay(rows, config, limits):
    controllers, counts, trials = {}, Counter(), defaultdict(Counter)
    active, per_row = 0, []
    for row in rows:
        episode = row['episode_id']
        controller = controllers.setdefault(episode, AlignmentAssist(config))
        # Replay at evidence availability, after all actual source frames/state.
        # No live inference latency was recorded, so this is ideal offline age.
        now = max([row['monotonic_ns'], row['state_monotonic_ns'], *row['frame_timestamps_ns'].values()])/1e9
        result = tick(controller, row, row['arms'], limits, now=now)
        assert not result['active'] or config['calibration']['verified'] is True
        if not result['active']:
            assert result['torque_nm'] == [0.]*6
        counts[result['reason']] += 1
        trials[episode][result['reason']] += 1
        active += int(result['active'])
        per_row.append(dict(episode_id=episode, time_s=row['time_s'], reason=result['reason'], active=result['active']))
    return dict(samples=len(rows), active_samples=active, reasons=dict(counts),
        per_episode={k: dict(v) for k, v in trials.items()},
        activation_coverage='NONE: cannot validate real nudge behavior' if active == 0 else 'observed'), per_row


def independent_evidence_counts(rows, task, report):
    """Count necessary gates independently; never replace recorded evidence."""
    counts = Counter()
    for row in rows:
        p = row['predictions'][task]
        names = p['landmark_names']
        moving = names.index('gear_center' if task == 'gear' else 'pin_tip')
        indices = [p['camera_names'].index(c) for c in ('left_wrist', 'right_wrist')]
        heads = ([names.index('slot_center')]*2 if task == 'gear' else
                 report['tasks'][task]['diagnostic_target_heads_by_episode'][row['episode_id']])
        visibility = all(p['visibility_probability'][i][j] >= .85
                         for i, target in zip(indices, heads) for j in (moving, target))
        counts['two_view_visibility_pass'] += int(visibility)
        confidence = all(p['confidence'][i][j] >= .5 and p['pixel_std'][i][j] <= 8.
                         for i, target in zip(indices, heads) for j in (moving, target))
        counts['two_view_confidence_std_pass'] += int(confidence)
        gripper = row['arms']['left']['gripper']
        counts['recorded_torque_grasp_predicate_pass'] += int(gripper['position'] <= 1.3
            and abs(gripper['torque']) >= .1 and abs(gripper['velocity']) <= .1)
        counts['recorded_passive_gripper'] += int(gripper.get('control_mode') == 'passive')
    return dict(samples=len(rows), counts=dict(counts),
        note='Independent necessary-gate counts; pin heads are diagnostic, physical bore association unverified. '
             'Torque predicate is existing controller contract, not a validated real grasp detector.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'logs/alignment-validation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'logs/alignment-calibration/predictions.phases10hz.corrected-order.jsonl'
    labels_path = ROOT/'data/calibration/manual-labels/assembled/labels.json'
    report_path = ROOT/'logs/alignment-calibration/calibration-report.corrected-order.json'
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    observations = read_json(labels_path)['observations']
    report = read_json(report_path)
    config = read_json(ROOT/'config/recording.json')['alignment_assist']
    # robot-config joint limits are identical for both sides; parse only YAML
    # lists without introducing a runtime dependency on the robot package.
    limit_lines = (ROOT/'config/robot-configs/left.yaml').read_text().split('  joint_limits:\n', 1)[1].split('  gripper_limits:', 1)[0]
    limits = {}
    for line in limit_lines.splitlines():
        if line.strip() in ('lower:', 'upper:'):
            key = line.strip()[:-1]
            limits[key] = []
        elif line.strip().startswith('- '):
            limits[key].append(float(line.strip()[2:]))
    assert all(len(limits[k]) == 6 for k in ('lower', 'upper'))
    output = dict(scope='OFFLINE ONLY; no hardware imports/commands/runtime config changes',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (cache, labels_path, report_path, ROOT/'app/alignment_assist.py', ROOT/'config/recording.json')},
        limitations=['Ideal replay evidence age excludes unrecorded live inference/transport latency.',
            'Recorded first-stop counts mask later gates; zero activation is not a validated motion pass.',
            'Synthetic positive/fault tests establish controller logic only, not perception or calibration accuracy.',
            'Return/hold/reset checks mirror collecting=False caller contract; no hardware loop is run.'],
        tasks={})
    for task in ('gear', 'pin'):
        selected = [r for r in rows if task in r['phases']]
        original = deepcopy(config)
        original['task'] = task
        default_counts, _ = replay(selected, original, limits)
        enabled = deepcopy(original)
        enabled['enabled'] = True
        enabled_counts, _ = replay(selected, enabled, limits)
        cal, provenance = geometry_fixture(task, report, observations)
        fixture = deepcopy(enabled)
        fixture['calibration'] = cal
        diagnostic_counts, detail = replay(selected, fixture, limits)
        heldout = {o['episode_id'] for o in observations if o['split'] != 'train'}
        validation_rows = [r for r in selected if r['episode_id'] in heldout]
        validation_counts, _ = replay(validation_rows, fixture, limits)
        (args.output/f'gates-{task}-rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detail))
        output['tasks'][task] = dict(production_unchanged=default_counts,
            production_enabled_only=enabled_counts, isolated_partial_fixture=diagnostic_counts,
            reserved_four_trials_fixture=validation_counts, fixture_provenance=provenance,
            independent_recorded_evidence=independent_evidence_counts(selected, task, report))
    output['synthetic_fault_tests'] = fault_checks()
    (args.output/'gates.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = ['# Offline recorded controller gate replay', '', output['scope'], '',
        'The deployed controller does **not** activate on these recordings. This is a blocker, not a passing motion-validation result.', '']
    for task, data in output['tasks'].items():
        lines += [f'## {task}', '', *[f'- {name}: {data[name]["samples"]} samples; {data[name]["active_samples"]} active; {data[name]["reasons"]}'
            for name in ('production_unchanged', 'production_enabled_only', 'isolated_partial_fixture', 'reserved_four_trials_fixture')], '',
            'Missing geometry: '+ '; '.join(data['fixture_provenance']['unavailable_geometry']), '',
            'Independent recorded necessary-gate counts: '+str(data['independent_recorded_evidence']), '']
    lines += [f'{len(output["synthetic_fault_tests"])} fault tests passed after independently proving a positive baseline for each case. Both tasks cover stale/skewed/missing/NaN evidence, visibility loss, checkpoint/camera change, absent grasp, joint limits, all excluded modes, disable, and the 0.3 s budget. Delta norm <=0.006 rad; torque component <=0.15 Nm.', '', *output['limitations'], '']
    (args.output/'gates.md').write_text('\n'.join(lines))
    print(json.dumps({t: d['isolated_partial_fixture'] for t, d in output['tasks'].items()}, indent=2))
    print(f'{len(output["synthetic_fault_tests"])} synthetic fault cases passed; report {args.output}/gates.md')


if __name__ == '__main__':
    main()
