#!/usr/bin/env python3
"""Offline proposals on recorded poses; no motor or robot SDK imports.

The +X forward axis below is a hypothetical coordinate constraint for this
comparison only. It does not populate or verify either runtime configuration.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from cartesian_nudge import ToolKinematics, rotation_error
from constrained_visual_nudge import constrained_visual_proposal


def main():
    source = ROOT/'logs/alignment-calibration/live-predictions.20261022T214847.jsonl'
    profile_path = ROOT/'config/calibration/gear-supervised-trial.json'
    robot_path = ROOT/'config/robot-configs/left.yaml'
    profile = json.loads(profile_path.read_text())
    robot = yaml.safe_load(robot_path.read_text())
    kin = ToolKinematics(robot)
    blocked = Counter()
    proposals = []
    count = 0
    for line in source.read_text().splitlines():
        row = json.loads(line); count += 1
        p = row['predictions']['gear']
        points = np.asarray(p['keypoints_2d'])
        ci = [p['camera_names'].index(name) for name in ('left_wrist', 'right_wrist')]
        moving, target = [p['landmark_names'].index(name) for name in ('gear_center', 'slot_center')]
        error = (points[ci, moving]-points[ci, target]).ravel()-profile['target_error_px']
        q = np.asarray(row['q_left'])
        if np.max(np.abs(error[[1, 3]])) <= 3:
            blocked['already_aligned'] += 1
            continue
        if (np.max(np.abs(q-profile['reference_joint_position_rad'])) > profile['max_joint_distance_rad']
                or np.max(np.abs(np.asarray(row['q_right'])-profile['other_arm_reference_joint_position_rad'])) > profile['max_joint_distance_rad']
                or any(np.linalg.norm(error[i:i+2]) > 100 for i in (0, 2))):
            blocked['outside_original_pose_or_proximity_bounds'] += 1
            continue
        try:
            result = constrained_visual_proposal(q, error, profile['image_jacobian_px_per_rad'],
                kin, robot['robot']['joint_limits'], {'forward_axis_base': [1, 0, 0]})
        except ValueError as exc:
            blocked[str(exc)] += 1
            continue
        xyz, rotation = kin.pose(q)
        new_xyz, new_rotation = kin.pose(result['target'])
        displacement = new_xyz-xyz
        proposals.append(dict(time_s=row['time_s'], tool_delta_mm=(displacement*1000).tolist(),
            step_norm_mm=float(np.linalg.norm(displacement)*1000),
            rotation_rad=rotation_error(new_rotation, rotation),
            predicted_image_error_before_px=error[[1, 3]].tolist(),
            predicted_image_error_after_px=result['predicted_residual'].tolist()))
    report = dict(scope='offline geometric feasibility only; hypothetical +X forward axis',
        hardware_commanded=False, runtime_configuration_changed=False,
        runtime_camera_and_grasp_gates_evaluated=False,
        observations=count, blocked=dict(blocked), proposals=len(proposals),
        maximum_modeled_step_mm=max((p['step_norm_mm'] for p in proposals), default=None),
        minimum_modeled_forward_mm=min((p['tool_delta_mm'][0] for p in proposals), default=None),
        maximum_modeled_rotation_rad=max((p['rotation_rad'] for p in proposals), default=None),
        stereo_status='blocked: measured intrinsics, hand-eye, base transform and 3D target offset missing',
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source, profile_path, robot_path, Path(robot['urdf']['file_path']),
                ROOT/'app/constrained_visual_nudge.py', ROOT/'app/stereo_pose_nudge.py',
                ROOT/'app/bounded_alignment_trial.py', Path(__file__).resolve())},
        proposal_details=proposals)
    destination = ROOT/'logs/alignment-calibration/additional-nudges-offline.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'proposal_details')}, indent=2))


if __name__ == '__main__':
    main()
