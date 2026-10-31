#!/usr/bin/env python3
"""Calibrate one camera from a metric checkerboard, reserving whole sessions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from nudge_calibration_common import write_report


def fit_intrinsics(observations, image_size):
    train = [r for r in observations if r['split'] == 'train']
    test = [r for r in observations if r['split'] == 'validation']
    report = dict(method='metric_checkerboard_intrinsics', statistical_candidate=False, reasons=[])
    if len(train) < 10 or len(test) < 4:
        report['reasons'] = ['at_least_10_train_and_4_reserved_images_required']
        return report
    if {r['session_id'] for r in train} & {r['session_id'] for r in test}:
        report['reasons'] = ['session_leakage_between_train_and_validation']
        return report
    if len({r['path'] for r in observations}) != len(observations):
        raise ValueError('duplicate_source_image')
    objects = [np.asarray(r['object_points'], np.float32) for r in train]
    points = [np.asarray(r['corners'], np.float32).reshape(-1, 1, 2) for r in train]
    rms, k, distortion, rotations, _ = cv2.calibrateCamera(objects, points, tuple(image_size), None, None)
    normals = [cv2.Rodrigues(r)[0][:, 2] for r in rotations]
    normal_span = max(np.arccos(np.clip(a @ b, -1, 1)) for a in normals for b in normals)
    errors = []
    for row in test:
        objects = np.asarray(row['object_points'], np.float32)
        points = np.asarray(row['corners'], np.float32).reshape(-1, 1, 2)
        success, r, t = cv2.solvePnP(objects, points, k, distortion)
        if not success:
            report['reasons'].append('reserved_board_pose_failed')
            continue
        predicted, _ = cv2.projectPoints(objects, r, t, k, distortion)
        errors.append(dict(path=row['path'], rmse_px=float(np.sqrt(np.mean(np.sum((predicted-points)**2, axis=2))))))
    report.update(intrinsics=k.tolist(), distortion=distortion.ravel().tolist(), image_size=list(image_size),
        training_reprojection_rms_px=float(rms), training_board_normal_span_deg=float(np.rad2deg(normal_span)),
        reserved_reprojection=errors)
    if normal_span < np.deg2rad(15):
        report['reasons'].append('insufficient_board_tilt_diversity')
    if len(errors) != len(test) or max((r['rmse_px'] for r in errors), default=999) > 1 or rms > 1:
        report['reasons'].append('reprojection_error_above_1px')
    report['statistical_candidate'] = not report['reasons']
    report['limitations'] = ['Reprojection alone does not certify metric accuracy or hand-eye geometry',
        'Intrinsics apply only to the measured camera, lens settings and image resolution']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path, help='JSON list: path, session_id, split')
    p.add_argument('--corners', required=True, nargs=2, type=int, metavar=('COLUMNS', 'ROWS'))
    p.add_argument('--square-size-mm', required=True, type=float)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if not np.isfinite(args.square_size_mm) or args.square_size_mm <= 0 or min(args.corners) < 3:
        p.error('positive measured square size and at least 3x3 internal corners required')
    manifest = json.loads(args.manifest.read_text())
    pattern = np.zeros((np.prod(args.corners), 3), np.float32)
    pattern[:, :2] = np.mgrid[0:args.corners[0], 0:args.corners[1]].T.reshape(-1, 2)*args.square_size_mm/1000
    observations, sources, rejected = [], [args.manifest, Path(__file__)], []
    image_size = None
    for row in manifest:
        if row['split'] not in ('train', 'validation'):
            raise ValueError('explicit_split_required')
        path = (args.manifest.parent/row['path']).resolve()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('unreadable_image: '+str(path))
        sources.append(path)
        size = (image.shape[1], image.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError('mixed_image_resolutions')
        image_size = size
        success, corners = cv2.findChessboardCornersSB(image, tuple(args.corners))
        if not success:
            rejected.append(str(path))
            continue
        observations.append(dict(row, path=str(path), object_points=pattern, corners=corners))
    result = fit_intrinsics(observations, image_size)
    result['checkerboard_not_detected'] = rejected
    result['square_size_m'] = args.square_size_mm/1000
    write_report(args.output, result, sources)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
