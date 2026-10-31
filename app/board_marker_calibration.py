"""Guided image/pose capture and offline camera plus hand-eye candidate fitting."""
import base64
import copy
import hashlib
import json
import sys

import cv2
import numpy as np

from calibration_workflow_common import Workflow, ROOT, angle, write_report

sys.path.insert(0, str(ROOT/'scripts'))
from calibrate_nudge_camera import fit_intrinsics
from calibrate_nudge_hand_eye import fit_hand_eye


class BoardMarkerCalibration(Workflow):
    name = 'board'

    def __init__(self, output):
        super().__init__(output)
        self.observations = []
        self.capture_requested = False
        self.settings = None
        self.detection = None
        self.image_size = None

    def begin(self, settings):
        pattern = settings.get('pattern', 'checkerboard')
        size_mm = float(settings.get('size_mm', 0))
        columns, rows = int(settings.get('columns', 9)), int(settings.get('rows', 6))
        camera, side = settings.get('camera'), settings.get('side')
        marker_id = int(settings.get('marker_id', 0))
        if (pattern not in ('checkerboard', 'aruco') or not np.isfinite(size_mm) or not 0 < size_mm <= 200
                or not 3 <= columns <= 30 or not 3 <= rows <= 30 or not 0 <= marker_id < 50
                or camera not in ('left_wrist', 'right_wrist', 'overhead') or side not in ('left', 'right')):
            raise ValueError('valid_metric_board_size_camera_and_physical_tool_assignment_required')
        if camera == 'overhead':
            raise ValueError('this_hand_eye_workflow_requires_an_eye_in_hand_camera')
        geometry = dict(pattern=pattern, size_m=size_mm/1000, columns=columns, rows=rows,
            camera=camera, side=side, marker_id=marker_id)
        if self.settings is not None and self.settings != geometry:
            raise ValueError('use_a_separate_workbench_run_for_each_camera_and_pattern')
        super().begin(settings)
        self.settings = geometry
        self.phase = 'board_pose'
        self.message = 'Keep the board fixed. Vary the wrist pose, hold still, then Capture endpoint.'

    def command(self, action, settings=None):
        if action == 'capture':
            if self.phase != 'board_pose':
                raise ValueError('begin_a_board_session_first')
            self.capture_requested = True
            self.rows = []
            return
        if action == 'analyze':
            return self.analyze()
        return super().command(action, settings)

    def stable(self, endpoint, frames):
        if self.phase != 'board_pose' or not self.capture_requested:
            return
        logical = self.settings['camera']
        if self.observations and self.last['camera_mapping'] != self.observations[0]['camera_mapping']:
            raise ValueError('camera_mapping_changed_start_a_separate_workbench_run')
        physical = self.last['camera_mapping'][logical]
        stamp = self.last['frame_timestamps_ns'][physical]
        if abs(stamp-self.last['state_monotonic_ns']) > 30_000_000:
            self.observation_reason = 'camera_tool_pose_timestamp_skew'
            return
        image_bytes = base64.b64decode(frames[physical], validate=True)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('invalid_camera_jpeg')
        h, w = image.shape
        if self.image_size is not None and [w, h] != self.image_size:
            raise ValueError('camera_resolution_changed')
        self.image_size = [w, h]
        s = self.settings
        if s['pattern'] == 'checkerboard':
            success, corners = cv2.findChessboardCornersSB(image, (s['columns'], s['rows']))
            if not success:
                self.observation_reason = 'checkerboard_not_detected'
                return
            objects = np.zeros((s['columns']*s['rows'], 3), np.float32)
            objects[:, :2] = np.mgrid[0:s['columns'], 0:s['rows']].T.reshape(-1, 2)*s['size_m']
        else:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            detected, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
            matches = [] if ids is None else [i for i, n in enumerate(ids.ravel()) if n == s['marker_id']]
            if len(matches) != 1:
                self.observation_reason = 'selected_marker_missing_or_ambiguous'
                return
            corners = detected[matches[0]].reshape(-1, 1, 2)
            objects = np.array([[0, 0, 0], [s['size_m'], 0, 0], [s['size_m'], s['size_m'], 0], [0, s['size_m'], 0]], np.float32)
        pose = copy.deepcopy(self.last['arms'][s['side']]['tool_pose_base'])
        if any(abs(row['frame_stamp']-stamp) < 1_000_000 for row in self.observations):
            self.observation_reason = 'frame_already_captured'
            return
        previous = [row for row in self.observations if row['session_id'] == self.session]
        if previous and any(np.linalg.norm(np.asarray(pose['position_m'])-r['tool_pose']['position_m']) < .01
                and angle(pose['rotation'], r['tool_pose']['rotation']) < np.deg2rad(5) for r in previous):
            self.observation_reason = 'vary_wrist_position_or_rotation_before_capturing_again'
            return
        path = self.output/('board-'+str(len(self.observations)+1)+'.jpg')
        path.write_bytes(image_bytes)
        self.observations.append(dict(path=str(path.resolve()), session_id=self.session,
            episode_id=self.episode, pose_id=str(len(self.observations)+1), split=self.split,
            object_points=objects.tolist(), corners=np.asarray(corners).tolist(),
            tool_pose=pose, frame_stamp=stamp, state_stamp=self.last['state_monotonic_ns'],
            camera_mapping=self.last['camera_mapping'], image_sha256=hashlib.sha256(image_bytes).hexdigest()))
        self.capture_requested = False
        self.detection = dict(corners=np.asarray(corners).reshape(-1, 2).tolist(), image_size=[w, h],
            image_base64=frames[physical])
        (self.output/'board-observations.json').write_text(json.dumps(self.observations, indent=2)+'\n')
        self.message = 'Pose captured. Vary board coverage and wrist tilt, or finish this session.'

    def observe(self, row, frames=None, collector_status=None):
        self.latest_status = collector_status or {}
        if self.last is not None and row['monotonic_ns'] <= self.last['monotonic_ns']:
            return
        self.last = row
        if self.phase != 'board_pose' or not self.capture_requested:
            return
        if row['episode_id'] != self.episode:
            self.phase = 'cancelled'
            self.message = 'Recording changed. Begin a separate board session.'
            return
        self.rows.append(row)
        self.rows = [r for r in self.rows if row['monotonic_ns']-r['monotonic_ns'] <= 750_000_000]
        if len(self.rows) < 5 or self.rows[-1]['monotonic_ns']-self.rows[0]['monotonic_ns'] < 600_000_000:
            self.observation_reason = 'Hold the wrist still for at least 0.6 seconds.'
            return
        side = self.settings['side']
        arms = [r['arms'][side] for r in self.rows]
        center = np.median([a['tool_pose_base']['position_m'] for a in arms], axis=0)
        reference = arms[0]['tool_pose_base']['rotation']
        if (any(np.linalg.norm(np.asarray(a['tool_pose_base']['position_m'])-center) > .00015
                or angle(a['tool_pose_base']['rotation'], reference) > .002
                or np.max(np.abs(a['velocity_rad_s'])) > .02 or any(a.get('motor_fault', [])) for a in arms)):
            self.observation_reason = 'Wrist is not settled.'
            return
        self.observation_reason = None
        self.stable({}, frames or {})

    def analyze(self):
        self.report = fit_intrinsics(self.observations, self.image_size)
        images = [row['path'] for row in self.observations]
        write_report(self.output/'camera-candidate.json', self.report, images)
        if not self.report.get('statistical_candidate'):
            return
        k, distortion = np.asarray(self.report['intrinsics']), np.asarray(self.report['distortion'])
        poses = []
        for row in self.observations:
            objects = np.asarray(row['object_points'], np.float32)
            corners = np.asarray(row['corners'], np.float32)
            success, r, t = cv2.solvePnP(objects, corners, k, distortion)
            if not success:
                raise ValueError('board_pose_estimation_failed')
            target = np.eye(4); target[:3, :3] = cv2.Rodrigues(r)[0]; target[:3, 3] = t.ravel()
            tool = np.eye(4); tool[:3, :3] = row['tool_pose']['rotation']; tool[:3, 3] = row['tool_pose']['position_m']
            poses.append(dict(session_id=row['session_id'], pose_id=row['pose_id'], split=row['split'],
                T_base_tool=tool.tolist(), T_camera_target=target.tolist()))
        pose_path = self.output/'board-and-tool-poses.jsonl'
        pose_path.write_text(''.join(json.dumps(p)+'\n' for p in poses))
        hand_eye = fit_hand_eye(poses)
        write_report(self.output/'hand-eye-candidate.json', hand_eye, [pose_path])
        self.report = dict(camera=self.report, hand_eye=hand_eye, verified=False, motion_authorized=False,
            note='Inter-arm base transform and demonstrated assembly offset remain separate measurements.')

    def status(self):
        return dict(super().status(), board_captures=len(self.observations), detection=self.detection)
