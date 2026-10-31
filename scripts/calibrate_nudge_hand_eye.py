#!/usr/bin/env python3
"""Offline eye-in-hand calibration against a stationary calibration target.

Input JSONL: split, T_base_tool, T_camera_target. T_A_B maps B into A.
Target poses must come from measured intrinsics and known metric board geometry.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from nudge_calibration_common import angle, finite, load_rows, rotation, write_report


def rigid(value):
    t = finite(value, (4, 4))
    rotation(t[:3, :3])
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('invalid_transform_bottom_row')
    return t


def fit_hand_eye(rows):
    report = dict(method='eye_in_hand_stationary_target', statistical_candidate=False, reasons=[])
    train = [row for row in rows if row['split'] == 'train']
    test = [row for row in rows if row['split'] == 'validation']
    if len(train) < 8 or len(test) < 4:
        report['reasons'] = ['at_least_8_train_and_4_reserved_poses_required']
        return report
    identities = [(row['session_id'], row['pose_id']) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate_pose_id')
    if {r['session_id'] for r in train} & {r['session_id'] for r in test}:
        report['reasons'] = ['session_leakage_between_train_and_validation']
        return report
    if any(row['split'] not in ('train', 'validation') for row in rows):
        raise ValueError('explicit_split_required')
    tools = [rigid(row['T_base_tool']) for row in train]
    targets = [rigid(row['T_camera_target']) for row in train]
    # Hand-eye rotation needs nonparallel rotation axes; translations alone do not suffice.
    rotvecs = [Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
        for i, a in enumerate(tools) for b in tools[i+1:]]
    singular = np.linalg.svd(rotvecs, compute_uv=False)
    if singular[1] < .15 or singular[1]/singular[0] < .1:
        report['reasons'] = ['insufficient_independent_rotation_axes']
        return report
    r, t = cv2.calibrateHandEye([x[:3, :3] for x in tools], [x[:3, 3] for x in tools],
        [x[:3, :3] for x in targets], [x[:3, 3] for x in targets], method=cv2.CALIB_HAND_EYE_PARK)
    camera = np.eye(4); camera[:3, :3] = r; camera[:3, 3] = t.ravel()
    rigid(camera)
    boards = [a @ camera @ b for a, b in zip(tools, targets)]
    center = np.eye(4)
    center[:3, 3] = np.median([x[:3, 3] for x in boards], axis=0)
    center[:3, :3] = Rotation.from_matrix(np.asarray([x[:3, :3] for x in boards])).mean().as_matrix()
    errors = []
    for row in test:
        board = rigid(row['T_base_tool']) @ camera @ rigid(row['T_camera_target'])
        errors.append(dict(session_id=row['session_id'], pose_id=row['pose_id'],
            translation_m=float(np.linalg.norm(board[:3, 3]-center[:3, 3])),
            rotation_rad=angle(board[:3, :3], center[:3, :3])))
    report.update(T_tool_camera=camera.tolist(), train_poses=len(train), validation_poses=len(test),
        heldout_board_constancy=errors,
        max_heldout_translation_m=max(x['translation_m'] for x in errors),
        max_heldout_rotation_rad=max(x['rotation_rad'] for x in errors))
    if report['max_heldout_translation_m'] > .002 or report['max_heldout_rotation_rad'] > np.deg2rad(.5):
        report['reasons'].append('reserved_board_pose_not_constant')
    report['statistical_candidate'] = not report['reasons']
    report['limitations'] = ['Requires the same target to remain fixed in the same base frame across all poses',
        'Does not estimate camera intrinsics, inter-arm base transform or assembly target offset',
        'Board constancy is not independent tool-motion validation']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = fit_hand_eye(load_rows(args.poses))
    write_report(args.output, result, [args.poses, Path(__file__)])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
