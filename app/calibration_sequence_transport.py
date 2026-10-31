"""Sequence-specific request queue and command validation for the new collector."""
import queue
import time
from recording import EpisodeStore
from robot_commands import RobotCommands


class SequenceStore(EpisodeStore):
    def __init__(self, *args, **kwargs):
        self.calibration_sequence_requests = queue.SimpleQueue()
        super().__init__(*args, **kwargs)

    def handle(self, kind, payload):
        if kind == 'calibration_sequence':
            self.calibration_sequence_requests.put(payload)
            return
        return super().handle(kind, payload)


class SequenceCommands(RobotCommands):
    @staticmethod
    def capabilities():
        result = RobotCommands.capabilities()
        result['commands']['calibration.sequence'] = {'fields': ['held', 'press_id', 'episode_id']}
        result['motion_gate_required'] = True
        return result

    def submit(self, request):
        if isinstance(request, dict) and request.get('command') == 'calibration.sequence':
            if set(request) != {'command', 'parameters'}:
                raise ValueError('explicit_sequence_parameters_required')
            payload = request['parameters']
            if not isinstance(payload, dict) or set(payload) != {'held', 'press_id', 'episode_id'}:
                raise ValueError('held_press_id_and_episode_id_required')
            if type(payload['held']) is not bool or not isinstance(payload['press_id'], str) or not 1 <= len(payload['press_id']) <= 80:
                raise ValueError('invalid_sequence_press')
            state = self.store.snapshot()
            if payload['held'] and (state.get('demo') or not state.get('control_active')
                    or state.get('phase') != 'recording' or payload['episode_id'] != state.get('active')
                    or not state.get('gear_grasp', {}).get('settled')
                    or not state.get('alignment_assist', {}).get('enabled')):
                raise ValueError('sequence_requires_current_real_recording_and_settled_grasp')
            self.store.submit('calibration_sequence', dict(payload, received_monotonic_ns=time.monotonic_ns()))
            return dict(accepted=True, command='calibration.sequence', status='queued_not_completed')
        return super().submit(request)

    def dispatch(self, command, payload):
        if command in ('alignment.nudge', 'alignment.servo', 'alignment.settings'):
            raise ValueError('legacy_motion_and_envelope_widening_disabled_in_sequence_collector')
        return super().dispatch(command, payload)
