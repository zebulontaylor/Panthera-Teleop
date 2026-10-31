import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

if cv2 is not None and np is not None:
    from scripts.real2sim_object_pose import (
        carrier_bore_points, fit_carrier_pose, localize_on_carrier_plane,
    )

K = np.array([[900., 0, 320], [0, 910., 240], [0, 0, 1]]) if np is not None else None


def projected(t=(.03, -.02, .45)):
    return cv2.projectPoints(carrier_bore_points(), np.array([.3, -.2, .1]),
                             np.array(t), K, np.zeros(5))[0].reshape(4, 2)


@unittest.skipIf(cv2 is None or np is None, "NumPy and OpenCV required")
class CarrierPoseTests(unittest.TestCase):
    def test_unordered_camera_local_pose(self):
        result = fit_carrier_pose(projected()[[2, 0, 3, 1]], K, max_reprojection_error_px=.001)
        assert not result['calibrated'] and not result['world_ready']
        assert result['yaw_symmetry_degrees'] == 90
        assert len(result['hypotheses']) >= 4
        for pose in result['hypotheses']:
            assert pose['reprojection_rms_px'] < .001
            np.testing.assert_allclose(pose['translation_m'], [.03, -.02, .45], atol=1e-5)
            camera = carrier_bore_points() @ np.array(pose['rotation_matrix']).T + pose['translation_m']
            assert np.all(camera[:, 2] > 0)


    def test_calibration_is_explicit(self):
        assert fit_carrier_pose(projected(), K, calibrated=True)['calibrated']


    def test_degenerate_rejected(self):
        for pixels in [np.zeros((4, 2)), np.array([[0, 0], [1, 0], [2, 0], [3, 0]])]:
            with self.assertRaises(ValueError):
                fit_carrier_pose(pixels, K)


    def test_behind_camera_never_returned(self):
        # Pixels alone cannot reveal which side generated them; positive-depth
        # alternatives may exist, but a returned solution must never be behind.
        result = fit_carrier_pose(projected((.03, -.02, -.45)), K)
        for pose in result['hypotheses']:
            xyz = carrier_bore_points() @ np.array(pose['rotation_matrix']).T + pose['translation_m']
            assert (xyz[:, 2] > 0).all()


    def test_homography_recovers_plane_coordinates(self):
        xy = carrier_bore_points()[:, :2]
        pixels = projected()
        result = localize_on_carrier_plane(pixels, pixels, xy)
        np.testing.assert_allclose(result['xy_m'], xy, atol=1e-8)
        assert not result['depth_estimated']
        assert not result['world_ready']


if __name__ == "__main__":
    unittest.main()
