"""Known-response and geometry tests for the new offline calibration tools."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from nudge_calibration_common import fit_response, write_report
from build_nudge_calibration_pairs import build, plateau, reference_repeatability
from audit_nudge_motion import audit
from calibrate_nudge_hand_eye import fit_hand_eye
from calibrate_nudge_camera import fit_intrinsics


def endpoint(position, error):
    return dict(position_m=np.asarray(position).tolist(), error_px=np.asarray(error).tolist(),
        rotation=np.eye(3).tolist(), other_q=[0.]*6, gripper_position=.5,
        camera_mapping={'left_wrist': 'port3', 'right_wrist': 'port2'}, checkpoint_id='test-checkpoint')


def measured_pairs(cartesian=False):
    response = np.array([[1000, 2000, 1000], [2000, -1000, 20000],
                         [-1000, 1000, 2000], [3000, 2000, -15000.]])
    pairs = []
    for i in range(36 if cartesian else 24):
        train = i < (24 if cartesian else 16)
        session = ('a' if i < (12 if cartesian else 8) else 'b') if train else 'c'
        dx = np.zeros(3)
        dx[(i//2)%3 if cartesian else 2] = (.0005 if i%2 == 0 else -.0005)
        pairs.append(dict(session_id=session, episode_id=session, trial_id=str(i),
            split='train' if train else 'validation', requested_delta_m=dx.tolist(), external_delta_m=dx.tolist(),
            start=endpoint([0, 0, .1], [2, 40, 10, -50]),
            end=endpoint(np.array([0, 0, .1])+dx, np.array([2, 40, 10, -50])+response @ dx)))
    return pairs, response


class CalibrationProtocolTests(unittest.TestCase):
    def test_bidirectional_axis_recovers_known_sign_and_units(self):
        rows, response = measured_pairs()
        report = fit_response(rows, axis=[0, 0, 1])
        self.assertTrue(report['statistical_candidate'], report)
        np.testing.assert_allclose(np.asarray(report['response_px_per_m']).ravel(), response[:, 2], atol=1e-7)
        self.assertGreater(report['improvement_fraction'], .999)

    def test_cartesian_fit_recovers_three_independent_columns(self):
        rows, response = measured_pairs(cartesian=True)
        report = fit_response(rows)
        self.assertTrue(report['statistical_candidate'], report)
        np.testing.assert_allclose(report['response_px_per_m'], response, atol=1e-7)

    def test_session_and_episode_leakage_are_rejected(self):
        rows, _ = measured_pairs()
        rows[-1]['session_id'] = 'a'
        self.assertIn('session_leakage_between_train_and_validation', fit_response(rows, [0, 0, 1])['reasons'])
        rows, _ = measured_pairs()
        rows[-1]['episode_id'] = 'a'
        self.assertIn('episode_leakage_between_train_and_validation', fit_response(rows, [0, 0, 1])['reasons'])

    def test_rotating_and_off_axis_data_are_rejected(self):
        rows, _ = measured_pairs()
        rows[0]['end']['rotation'] = Rotation.from_rotvec([.1, 0, 0]).as_matrix().tolist()
        rows[1]['end']['position_m'][0] = .001
        report = fit_response(rows, [0, 0, 1])
        self.assertEqual(report['rejected']['tool_rotation_changed'], 1)
        self.assertEqual(report['rejected']['off_axis_or_insufficient_axis_motion'], 1)

    def test_hysteresis_and_wrong_reserved_response_fail(self):
        rows, _ = measured_pairs()
        for row in rows:
            if row['end']['position_m'][2] > row['start']['position_m'][2]:
                a = np.asarray(row['start']['error_px']); b = np.asarray(row['end']['error_px'])
                row['end']['error_px'] = (a+2*(b-a)).tolist()
        report = fit_response(rows, [0, 0, 1])
        self.assertIn('direction_dependent_response_or_hysteresis', report['reasons'])
        rows, _ = measured_pairs()
        for row in rows:
            if row['split'] == 'validation':
                a = np.asarray(row['start']['error_px']); b = np.asarray(row['end']['error_px'])
                row['end']['error_px'] = (2*a-b).tolist()
        self.assertFalse(fit_response(rows, [0, 0, 1])['statistical_candidate'])

    def test_motion_audit_catches_physical_disagreement(self):
        rows, _ = measured_pairs()
        rows[0]['external_delta_m'] = (-np.asarray(rows[0]['external_delta_m'])).tolist()
        rows[1].pop('external_delta_m')
        report = audit(rows)
        self.assertEqual(report['outcomes']['external_motion_disagrees_with_encoder_fk'], 1)
        self.assertEqual(report['outcomes']['external_measurement_missing'], 1)
        self.assertEqual(report['outcomes']['three_way_motion_agreement'], 22)

    def test_marked_plateaus_use_distinct_frames_and_do_not_overlap(self):
        from test_cartesian_nudge import fixture
        _, snapshot, arms, _, _ = fixture()
        rows = []
        for i in range(31):
            t = i*.1
            arm = copy.deepcopy(arms)
            arm['left']['tool_pose_base']['position_m'][2] = 0 if t < 1.5 else .0005
            rows.append(dict(episode_id='episode', time_s=t, monotonic_ns=int(t*1e9),
                state_monotonic_ns=int(t*1e9), frame_timestamps_ns={name: int(t*1e9)
                    for name in snapshot['camera_mapping'].values()}, camera_mapping=snapshot['camera_mapping'],
                predictions=snapshot['predictions'], arms=arm))
        markers = [dict(session_id='s', episode_id='episode', trial_id='one', split='train', label=label, time_s=t)
            for label, t in [('start', 1.), ('end', 3.)]]
        pairs, rejected, _ = build(rows, markers)
        self.assertEqual(len(pairs), 1, rejected)
        self.assertAlmostEqual(pairs[0]['end']['position_m'][2]-pairs[0]['start']['position_m'][2], .0005)
        references = [dict(session_id='s', episode_id='episode', time_s=t, **pairs[0][key])
            for key, t in [('start', 1.), ('end', 3.)]]
        self.assertAlmostEqual(reference_repeatability(references)['s']['returned_position_difference_m'][2], .0005)
        duplicates = [copy.deepcopy(rows[0]) for _ in range(10)]
        with self.assertRaisesRegex(ValueError, 'five_distinct'):
            plateau(duplicates, dict(episode_id='episode', time_s=.1))
        markers[-1]['time_s'] = 1.1
        self.assertIn('overlapping_or_reversed_endpoint_windows', build(rows, markers)[1])

    def test_hand_eye_recovers_camera_transform_on_reserved_poses(self):
        rng = np.random.default_rng(73)
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_rotvec([.1, -.2, .05]).as_matrix()
        camera[:3, 3] = [.03, .02, .05]
        board = np.eye(4); board[:3, 3] = [.1, .05, .7]
        rows = []
        for i in range(22):
            tool = np.eye(4)
            tool[:3, :3] = Rotation.from_rotvec(rng.normal(0, .35, 3)).as_matrix()
            tool[:3, 3] = rng.uniform(-.1, .1, 3)
            rows.append(dict(session_id='train' if i < 16 else 'test', pose_id=str(i),
                split='train' if i < 16 else 'validation', T_base_tool=tool.tolist(),
                T_camera_target=(np.linalg.inv(tool @ camera) @ board).tolist()))
        report = fit_hand_eye(rows)
        self.assertTrue(report['statistical_candidate'], report)
        np.testing.assert_allclose(report['T_tool_camera'], camera, atol=1e-8)
        for row in rows:
            row['T_base_tool'] = np.eye(4).tolist()
        self.assertIn('insufficient_independent_rotation_axes', fit_hand_eye(rows)['reasons'])

    def test_camera_intrinsics_recover_known_geometry(self):
        rng = np.random.default_rng(7)
        k = np.array([[600., 0, 320], [0, 610., 240], [0, 0, 1.]])
        points = np.zeros((54, 3), np.float32)
        points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*.02
        rows = []
        for i in range(20):
            r = rng.uniform(-.35, .35, 3)
            t = np.r_[rng.uniform(-.1, -.03, 2), rng.uniform(.5, .8)]
            corners, _ = cv2.projectPoints(points, r, t, k, np.zeros(5))
            rows.append(dict(path=str(i), session_id='train' if i < 16 else 'test',
                split='train' if i < 16 else 'validation', object_points=points, corners=corners))
        report = fit_intrinsics(rows, [640, 480])
        self.assertTrue(report['statistical_candidate'], report)
        np.testing.assert_allclose(report['intrinsics'], k, atol=.05)

    def test_reports_never_verify_or_overwrite_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.json'; source.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'overwrite_input'):
                write_report(source, {}, [source])
            report = {'verified': True, 'motion_authorized': True}
            write_report(Path(directory)/'report.json', report, [source])
            self.assertFalse(report['verified'])
            self.assertFalse(report['motion_authorized'])


if __name__ == '__main__':
    unittest.main()
