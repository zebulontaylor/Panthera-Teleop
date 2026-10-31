"""Collect operator-demonstrated corrections and an unverified local library."""
import json
import numpy as np
from calibration_workflow_common import Workflow, angle


class DemonstratedCorrectionCalibration(Workflow):
    name = 'demonstration'

    def __init__(self, output):
        super().__init__(output)
        self.capture_requested = False
        self.library = []

    def command(self, action, settings=None):
        if action == 'capture':
            if self.phase not in ('bad_pose', 'corrected_pose'):
                raise ValueError('capture_not_available_in_this_phase')
            self.capture_requested = True
            self.rows = []
            self.message = 'Hold still while the endpoint is captured.'
            return
        if action == 'next':
            if self.phase != 'pair_complete':
                raise ValueError('finish_the_current_pair_first')
            self.phase = 'bad_pose'
            self.message = 'Show a nearby misalignment, then select Capture endpoint.'
            self.capture_requested = False
            return
        return super().command(action, settings)

    def stable(self, endpoint, frames):
        if self.phase == 'reference':
            self.reference = endpoint
            self.event('demonstrated_good_reference', endpoint)
            self.phase = 'bad_pose'
            self.message = 'Show a nearby misalignment, then select Capture endpoint.'
            return
        if not self.capture_requested:
            return
        self.capture_requested = False
        if self.phase == 'bad_pose':
            self.before = endpoint
            self.phase = 'corrected_pose'
            self.message = 'Manually correct the height, then select Capture endpoint.'
            return
        if self.phase != 'corrected_pose':
            return
        if any(endpoint[key] != self.reference[key] or endpoint[key] != self.before[key]
                for key in ('camera_mapping', 'checkpoint_id')):
            self.observation_reason = 'Camera mapping or checkpoint changed; cancel and restart.'
            return
        delta = np.asarray(endpoint['position_m'])-self.before['position_m']
        before = (np.asarray(self.before['error_px'])-self.reference['error_px'])[[1, 3]]
        after = (np.asarray(endpoint['error_px'])-self.reference['error_px'])[[1, 3]]
        if (not .0002 <= np.linalg.norm(delta) <= .003 or angle(endpoint['rotation'], self.before['rotation']) > .003
                or np.max(np.abs(np.asarray(endpoint['other_q'])-self.before['other_q'])) > .003
                or abs(endpoint['gripper_position']-self.before['gripper_position']) > .02):
            self.observation_reason = 'Correction changed orientation, carrier, grasp or exceeded the local displacement range.'
            self.message = 'Retake the corrected endpoint or cancel this session.'
            return
        if np.linalg.norm(before)-np.linalg.norm(after) < 2 or np.any(np.abs(after) > np.abs(before)+1):
            self.observation_reason = 'No clear two-view improvement over the demonstrated reference.'
            return
        if not self.pair(self.before, endpoint):
            return
        self.library.append(dict(session_id=self.session, split=self.split,
            start_position_m=self.before['position_m'], start_rotation=self.before['rotation'],
            start_error_px=self.before['error_px'], target_error_px=self.reference['error_px'],
            delta_xyz_m=delta.tolist(), delta_q_rad=(np.asarray(endpoint['q'])-self.before['q']).tolist(),
            before_height_error_px=before.tolist(), after_height_error_px=after.tolist(),
            camera_mapping=endpoint['camera_mapping'], checkpoint_id=endpoint['checkpoint_id']))
        (self.output/'demonstrated-library.json').write_text(json.dumps(dict(verified=False, motion_authorized=False,
            scope='local demonstrated translations; not an autonomous correction policy', examples=self.library), indent=2)+'\n')
        self.phase = 'pair_complete'
        self.message = 'Demonstrated improvement captured. Capture another example or finish the session.'

    def status(self):
        return dict(super().status(), demonstrated_examples=len(self.library))
