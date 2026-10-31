#!/usr/bin/env python3
"""Audit isolated proposals on recordings; no robot interfaces or commands.

Recorded trajectories are not responses to these counterfactual proposals.
This audit cannot measure closed-loop convergence or validate hardware motion.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from cartesian_nudge import ToolKinematics, rotation_error


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_recordings(path, profile):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('complete') is not True
            or manifest.get('output_sha256') != hashlib.sha256(data).hexdigest()
            or manifest.get('samples') != len(rows)):
        raise ValueError('prediction_manifest_integrity_failed')
    if manifest['checkpoints']['gear']['sha256'] != profile['checkpoint_id']:
        raise ValueError('checkpoint_mismatch')
    if manifest['camera_mapping'] != profile['camera_mapping']:
        raise ValueError('camera_mapping_mismatch')
    return rows, manifest_path


def main():
    from incremental_nudge import incremental_proposal, quantized_proposal, nudge_bounds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path,
        default=ROOT / 'logs/alignment-calibration/live-predictions.20261022T214847.jsonl')
    parser.add_argument('--profile', type=Path,
        default=ROOT / 'config/calibration/gear-supervised-trial.json')
    parser.add_argument('--output', type=Path,
        default=ROOT / 'logs/alignment-calibration/incremental-nudge-offline.json')
    parser.add_argument('--controller-profile', choices=('strict', 'supervised_relaxed'), default='strict')
    parser.add_argument('--failed-states', type=Path)
    args = parser.parse_args()
    bounds = nudge_bounds(args.controller_profile, hardware=True)
    profile = json.loads(args.profile.read_text())
    if args.controller_profile == 'supervised_relaxed':
        profile['max_joint_distance_rad'] = .4
    rows, manifest_path = read_recordings(args.predictions, profile)
    robot_path = ROOT / 'config/robot-configs/left.yaml'
    robot = yaml.safe_load(robot_path.read_text())
    kin = ToolKinematics(robot)
    jac = np.asarray(profile['image_jacobian_px_per_rad'])
    outcomes = Counter()
    eligible_outcomes = Counter()
    proposals = []
    eligible_count = 0
    for row in rows:
        p = row['predictions']['gear']
        if (row['camera_mapping'] != profile['camera_mapping']
                or p['checkpoint_id'] != profile['checkpoint_id']
                or list(p['image_size']) != list(profile['image_size'])):
            raise ValueError('row_identity_mismatch')
        points = np.asarray(p['keypoints_2d'])
        ci = [p['camera_names'].index(name) for name in profile['camera_names']]
        moving = p['landmark_names'].index('gear_center')
        target = p['landmark_names'].index(profile['target_landmark'])
        error = (points[ci, moving] - points[ci, target]).ravel() - profile['target_error_px']
        q = np.asarray(row['q_left'])
        eligible = bool(
            np.max(np.abs(q - profile['reference_joint_position_rad'])) <= profile['max_joint_distance_rad']
            and np.max(np.abs(np.asarray(row['q_right']) - profile['other_arm_reference_joint_position_rad'])) <= profile['max_joint_distance_rad']
            and all(np.linalg.norm(error[i:i+2]) <= 100 for i in (0, 2)))
        eligible_count += int(eligible)
        if np.max(np.abs(error[[1, 3]])) <= 3:
            outcome = 'already_within_image_tolerance'
        else:
            try:
                result = incremental_proposal(q, error, jac, kin, robot['robot']['joint_limits'],
                    max_step_m=bounds['max_step_m'], profile=args.controller_profile)
                result = quantized_proposal(result, q, error, jac, kin, robot['robot']['joint_limits'],
                    profile=args.controller_profile)
                xyz, rotation = kin.pose(q)
                new_xyz, new_rotation = kin.pose(result['target'])
                proposals.append(dict(episode_id=row['episode_id'], time_s=row['time_s'],
                    within_pose_and_proximity_bounds=eligible,
                    requested_dz_mm=float(result['dz_m'] * 1000),
                    horizontal_mm=float(np.linalg.norm((new_xyz-xyz)[:2]) * 1000),
                    rotation_rad=rotation_error(new_rotation, rotation)))
                outcome = 'proposal'
            except ValueError as exc:
                outcome = str(exc)
        outcomes[outcome] += 1
        if eligible:
            eligible_outcomes[outcome] += 1
    sources = [args.predictions, manifest_path, args.profile, robot_path,
        Path(robot['urdf']['file_path']), ROOT / 'app/incremental_nudge.py',
        ROOT / 'app/cartesian_nudge.py', Path(__file__).resolve()]
    failed_case = None
    if args.failed_states:
        sources.append(args.failed_states)
        with args.failed_states.open() as stream:
            for line in stream:
                row = json.loads(line)
                servo = row.get('alignment_servo') or {}
                if servo.get('reason') != 'motor_resolution_limited' or servo.get('error_px') is None:
                    continue
                q = np.asarray(row['arms']['left']['position_rad'])
                error = np.asarray(servo['error_px'])
                failed_case = dict(episode_time_s=row['episode_time_s'], q_left=q.tolist(),
                    error_px=error.tolist(), original_reason=servo['reason'], original_steps=servo['steps'])
                started = time.perf_counter()
                try:
                    result = incremental_proposal(q, error, jac, kin, robot['robot']['joint_limits'],
                        max_step_m=bounds['max_step_m'], profile=args.controller_profile)
                    result = quantized_proposal(result, q, error, jac, kin, robot['robot']['joint_limits'],
                        profile=args.controller_profile)
                    p0, r0 = kin.pose(q)
                    p1, r1 = kin.pose(result['target'])
                    corners = result['completion_corners']
                    failed_case.update(outcome='proposal', target_position_rad=result['target'].tolist(),
                        endpoint_delta_mm=((p1-p0)*1000).tolist(),
                        endpoint_rotation_rad=rotation_error(r1, r0),
                        predicted_height_error_px=result['predicted_height_error_px'].tolist(),
                        completion_corner_count=len(corners),
                        max_corner_horizontal_mm=max(float(np.linalg.norm((p-p0)[:2])*1000) for p, r in corners),
                        max_corner_rotation_rad=max(rotation_error(r, r0) for p, r in corners),
                        nominal_minimum_motion_time_s=float(np.max(np.abs(result['target']-q))/.02))
                except ValueError as exc:
                    failed_case['outcome'] = str(exc)
                failed_case['offline_proposal_compute_ms'] = (time.perf_counter()-started)*1000
                break
        if failed_case is None:
            raise ValueError('failed_case_not_found')
    report = dict(
        scope='isolated model-space proposals at recorded poses; not closed-loop replay',
        proposal_mode='hardware_sdk_quantized_with_independent_joint_completion_checks',
        controller_profile=args.controller_profile, bounds=bounds,
        effective_profile_envelope_rad=profile['max_joint_distance_rad'], failed_live_case=failed_case,
        hardware_commanded=False, calibration_verified=False, convergence_validated=False,
        runtime_camera_and_grasp_gates_evaluated=False,
        eligibility_definition='Only both-arm profile pose envelopes and 100px per-view residual bounds; not full runtime eligibility.',
        observations=len(rows), observations_within_pose_and_proximity_bounds=eligible_count,
        all_observation_outcomes=dict(outcomes), eligible_observation_outcomes=dict(eligible_outcomes),
        proposals=len(proposals), proposals_within_pose_and_proximity_bounds=sum(
            p['within_pose_and_proximity_bounds'] for p in proposals),
        max_proposed_horizontal_mm=max((p['horizontal_mm'] for p in proposals), default=None),
        max_proposed_rotation_rad=max((p['rotation_rad'] for p in proposals), default=None),
        limitations=[
            'Each observation is independently evaluated using the unchanged demonstration seed.',
            'No measured response to a proposed command exists in this audit; online adaptation is not exercised.',
            'Recorded samples and seed share one episode; proposal counts are not held-out accuracy.',
            'An image-tolerance outcome is not verified physical alignment or insertion.'],
        source_sha256={str(path.resolve()): digest(path) for path in sources}, proposal_details=proposals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'proposal_details')}, indent=2))


if __name__ == '__main__':
    main()
