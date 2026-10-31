"""Experimental camera-local carrier reconstruction; never a world calibration.

The four indistinguishable bores leave a 90 degree yaw ambiguity. A measured
camera matrix is required for metric PnP; passing an approximate matrix does not
make the reconstruction calibrated. The default origin is the bore plane.
"""
from itertools import permutations

import cv2
import numpy as np


CARRIER_BORE_RADIUS_M = 0.0179625
CARRIER_BORE_HEIGHT_M = 0.01725


def carrier_bore_points(radius_m=CARRIER_BORE_RADIUS_M):
    """Four bore centers on a circle, in counterclockwise order, meters."""
    r = float(radius_m)
    if not np.isfinite(r) or r <= 0:
        raise ValueError('radius_m must be positive')
    return np.array([[r, 0, 0], [0, r, 0], [-r, 0, 0], [0, -r, 0]], dtype=np.float64)


def _points(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return a


def _quad(points, name):
    p = _points(points, (4, 2), name)
    hull = cv2.convexHull(p.astype(np.float32))
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 1e-6:
        raise ValueError(f'{name} must contain four distinct nondegenerate convex points')
    return p


def fit_carrier_pose(image_points, camera_matrix, distortion=None, *, calibrated=False,
                     object_points=None, max_reprojection_error_px=3.0):
    """Fit unordered bore pixels, returning all distinct acceptable IPPE poses.

    ``object_points`` are CAD bore coordinates in meters (default bore-plane
    origin). Positive depth is enforced for every bore. RMS error is in pixels.
    Calibration is a caller assertion, never inferred from numeric intrinsics.
    Reflections of the unlabeled square also leave its normal sign ambiguous;
    no visible-face prior is applied. Camera coordinates follow OpenCV: +x right,
    +y down, +z forward. No camera-to-world transform is estimated.
    """
    pixels = _quad(image_points, 'image_points')
    k = _points(camera_matrix, (3, 3), 'camera_matrix')
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('camera_matrix must have positive focal lengths and last row [0,0,1]')
    obj = carrier_bore_points() if object_points is None else _points(object_points, (4, 3), 'object_points')
    if np.ptp(obj[:, 2]) > 1e-9:
        raise ValueError('IPPE requires a constant-z bore plane')
    _quad(obj[:, :2], 'object_points XY')
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    if dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise ValueError('invalid distortion coefficients')
    if max_reprojection_error_px < 0 or not np.isfinite(max_reprojection_error_px):
        raise ValueError('max_reprojection_error_px must be finite and nonnegative')
    hypotheses = []
    for perm in permutations(range(4)):
        matched = pixels[list(perm)]
        result = cv2.solvePnPGeneric(obj, matched, k, dist, flags=cv2.SOLVEPNP_IPPE)
        for rv, tv in zip(result[1], result[2]):
            rotation = cv2.Rodrigues(rv)[0]
            translation = tv.reshape(3)
            camera_points = obj @ rotation.T + translation
            if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 1e-9):
                continue
            projected = cv2.projectPoints(obj, rv, tv, k, dist)[0].reshape(4, 2)
            residual = float(np.sqrt(np.mean(np.sum((projected - matched)**2, axis=1))))
            if not np.isfinite(residual) or residual > max_reprojection_error_px:
                continue
            if any(np.linalg.norm(rotation - np.asarray(h['rotation_matrix'])) < 1e-5 and
                   np.linalg.norm(translation - np.asarray(h['translation_m'])) < 1e-7 for h in hypotheses):
                continue
            hypotheses.append(dict(rotation_matrix=rotation.tolist(), translation_m=translation.tolist(),
                                   rvec=rv.reshape(3).tolist(), reprojection_rms_px=residual,
                                   image_indices_for_object_points=list(perm)))
    hypotheses.sort(key=lambda h: h['reprojection_rms_px'])
    return dict(hypotheses=hypotheses, calibrated=bool(calibrated), coordinate_frame='opencv_camera',
                units='meters', yaw_symmetry_degrees=90, normal_sign_ambiguous=True,
                world_ready=False, intrinsics_source='caller_provided',
                distortion_assumption='zero' if distortion is None else 'caller_provided',
                object_points_m=obj.tolist(),
                caveat='Metric accuracy depends on supplied camera calibration and CAD correspondences; no world extrinsics are available.')


def localize_on_carrier_plane(image_points, measured_bore_pixels, cad_bore_xy_m):
    """Map pixels to metric XY using four explicitly ordered correspondences.

    This is a plane homography, not depth reconstruction. All queried points must
    lie in the same physical plane as the measured bores. Undistort both inputs
    first if lens distortion is significant. Unordered bores cannot define axes.
    """
    src = _quad(measured_bore_pixels, 'measured_bore_pixels')
    dst = _quad(cad_bore_xy_m, 'cad_bore_xy_m')
    query = np.asarray(image_points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2 or not np.isfinite(query).all():
        raise ValueError('image_points must be finite Nx2')
    h, _ = cv2.findHomography(src, dst, method=0)
    if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
        raise ValueError('degenerate homography')
    homogeneous = np.column_stack([query, np.ones(len(query))]) @ h.T
    if np.any(abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError('query maps to infinity')
    return dict(xy_m=(homogeneous[:, :2] / homogeneous[:, 2:]).tolist(),
                pixel_to_plane_homography=h.tolist(), coordinate_frame='carrier_bore_plane',
                world_ready=False, depth_estimated=False,
                caveat='Assumes every query is on the bore plane; distortion must already be corrected.')
