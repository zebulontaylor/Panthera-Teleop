#!/usr/bin/env python3
"""Read-only recorded-pose comparison; never imports robot-control interfaces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from cartesian_nudge import ToolKinematics, vertical_proposal, rotation_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT/'logs/alignment-calibration/live-predictions.20261022T214847.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/alignment-calibration/cartesian-nudge-offline.json')
    args = parser.parse_args()
    profile_path = ROOT/'config/calibration/gear-supervised-trial.json'
    robot_path = ROOT/'config/robot-configs/left.yaml'
    profile = json.loads(profile_path.read_text())
    robot = yaml.safe_load(robot_path.read_text())
    kin = ToolKinematics(robot)
    jac = np.asarray(profile['image_jacobian_px_per_rad'])
    blocked = Counter()
    accepted = []
    legacy = []
    gate_eligible = 0
    for line in args.predictions.read_text().splitlines():
        row = json.loads(line)
        p = row['predictions']['gear']
        points = np.asarray(p['keypoints_2d'])
        ci = [p['camera_names'].index(n) for n in ('left_wrist', 'right_wrist')]
        moving, target = [p['landmark_names'].index(n) for n in ('gear_center', 'slot_center')]
        error = (points[ci, moving]-points[ci, target]).ravel()-profile['target_error_px']
        q = np.asarray(row['q_left'])
        xyz, rotation = kin.pose(q)
        j = jac[[1, 3]]
        delta = -j.T @ np.linalg.solve(j @ j.T+25*np.eye(2), error[[1, 3]])
        delta *= min(1., .003/max(np.linalg.norm(delta), 1e-12))
        old_xyz, old_rotation = kin.pose(q+delta)
        legacy.append(dict(horizontal_mm=float(np.linalg.norm((old_xyz-xyz)[:2])*1000),
            rotation_rad=rotation_error(old_rotation, rotation)))
        try:
            result = vertical_proposal(q, error, jac, kin, robot['robot']['joint_limits'])
        except ValueError as exc:
            blocked[str(exc)] += 1
            continue
        new_xyz, new_rotation = kin.pose(result['target'])
        eligible = (np.max(np.abs(q-profile['reference_joint_position_rad'])) <= profile['max_joint_distance_rad']
            and np.max(np.abs(np.asarray(row['q_right'])-profile['other_arm_reference_joint_position_rad'])) <= profile['max_joint_distance_rad']
            and all(np.linalg.norm(error[i:i+2]) <= 100 for i in (0, 2)))
        gate_eligible += int(eligible)
        accepted.append(dict(time_s=row['time_s'], within_pose_and_proximity_bounds=bool(eligible),
            requested_dz_mm=result['dz_m']*1000,
            horizontal_mm=float(np.linalg.norm((new_xyz-xyz)[:2])*1000),
            rotation_rad=rotation_error(new_rotation, rotation)))
    # Independent finite-difference check of the model Jacobian's vertical axis.
    q = np.asarray(profile['reference_joint_position_rad'])
    direction = np.linalg.solve(kin.jacobian(q), [0., 0., 1., 0., 0., 0.])
    a, _ = kin.pose(q-direction*1e-6)
    b, _ = kin.pose(q+direction*1e-6)
    derivative = (b-a)/2e-6
    if not np.allclose(derivative, [0, 0, 1], atol=1e-5):
        raise ValueError('model_jacobian_finite_difference_failed')
    result = dict(scope='offline model-space proposals only; no physical response validation',
        hardware_commanded=False, calibration_verified=False,
        observations=len(legacy), blocked=dict(blocked), proposals=len(accepted),
        proposals_within_pose_and_proximity_bounds=gate_eligible,
        runtime_camera_and_grasp_gates_evaluated=False,
        jacobian_vertical_finite_difference=derivative.tolist(),
        original_unconstrained_max_horizontal_mm=max(v['horizontal_mm'] for v in legacy),
        constrained_max_horizontal_mm=max((v['horizontal_mm'] for v in accepted), default=None),
        constrained_max_rotation_rad=max((v['rotation_rad'] for v in accepted), default=None),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.predictions, profile_path, robot_path, Path(robot['urdf']['file_path']))},
        proposal_details=accepted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('proposal_details', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
