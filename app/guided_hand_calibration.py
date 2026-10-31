"""Automatically mark a reference/up/return/down/return hand-guided protocol."""
import numpy as np
from calibration_workflow_common import Workflow, angle


class GuidedHandCalibration(Workflow):
    name = 'guided'
    sequence = (1, 0, -1, 0, 1, 0, -1, 0)

    def stable(self, endpoint, frames):
        if self.phase == 'reference':
            self.reference = self.before = endpoint
            self.references.append(endpoint)
            self.event('reference', endpoint)
            self.phase = 'move'
            self.message = 'Move slightly in the positive task-axis direction, then hold still.'
            return
        if self.phase != 'move':
            return
        if any(endpoint[key] != self.reference[key] for key in ('camera_mapping', 'checkpoint_id')):
            self.observation_reason = 'Camera mapping or checkpoint changed; cancel and restart.'
            return
        if endpoint['first_monotonic_ns'] <= self.before['last_monotonic_ns']:
            return
        displacement = np.asarray(endpoint['position_m'])-self.reference['position_m']
        signed = float(displacement @ self.axis)
        expected = self.sequence[self.counter]
        if angle(endpoint['rotation'], self.reference['rotation']) > .003:
            self.observation_reason = 'Keep tool orientation unchanged.'
            return
        if np.linalg.norm(displacement-signed*self.axis) > max(.0001, .1*abs(signed)):
            self.observation_reason = 'Movement is too far off the task axis.'
            return
        if expected == 0:
            if np.linalg.norm(displacement) > .00015:
                return
            if np.max(np.abs(np.asarray(endpoint['error_px'])-self.reference['error_px'])) > 3:
                self.observation_reason = 'Reference image offset changed; check grasp and target.'
                return
            self.event('reference_return', endpoint)
        elif not .0002 <= signed*expected <= .001:
            return
        if not self.pair(self.before, endpoint):
            return
        self.before = endpoint
        self.counter += 1
        if self.counter == len(self.sequence):
            self.phase = 'complete'
            self.message = 'Eight movement pairs captured. Record another session or analyze reserved evidence.'
        else:
            next_direction = self.sequence[self.counter]
            self.message = ('Return to the demonstrated reference and hold still.' if next_direction == 0 else
                'Move slightly in the '+('positive' if next_direction > 0 else 'negative')+' task-axis direction, then hold still.')
        self.persist()
