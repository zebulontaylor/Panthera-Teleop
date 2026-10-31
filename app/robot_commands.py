"""Transport-independent robot intents; only the control loop accesses motors.

Natural language is an explicit vocabulary, not an autonomous task planner.
"""
import math
import time


LEGACY_ROUTES = {'/api/alignment-assist': 'alignment.assist', '/api/alignment-nudge': 'alignment.nudge', '/api/gear-grasp': 'gear.grasp', '/api/alignment-trial-settings': 'alignment.settings', '/api/alignment-servo': 'alignment.servo', '/api/decision': 'episode.finish'}

# Each phrase has exactly one meaning. Episode/press identifiers remain explicit.
PHRASES = {
    'enable alignment assistance': ('alignment.assist', {'enabled': True}),
    'disable alignment assistance': ('alignment.assist', {'enabled': False}),
    'close the gear gripper': ('gear.grasp', {'action': 'close'}),
    'release the gear gripper': ('gear.grasp', {'action': 'release'}),
    'start visual alignment': ('alignment.servo', {'enabled': True}),
    'stop visual alignment': ('alignment.servo', {'enabled': False}),
    'keep this episode': ('episode.finish', {'label': 'kept'}),
    'flag this episode for review': ('episode.finish', {'label': 'review'}),
    'discard this episode': ('episode.finish', {'label': 'discarded'}),
}
FIELDS = {
    'alignment.assist': ('enabled',),
    'alignment.nudge': ('held', 'press_id', 'episode_id'),
    'gear.grasp': ('action', 'episode_id'),
    'alignment.settings': ('settings',),
    'alignment.servo': ('enabled',),
    'episode.finish': ('label', 'episode_id'),
}


class RobotCommands:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def capabilities():
        return dict(version=1, commands={name: {'fields': list(fields)}
                    for name, fields in FIELDS.items()}, phrases=list(PHRASES))

    def submit(self, request):
        """Validate one explicit intent and enqueue it; acceptance is not completion."""
        if not isinstance(request, dict):
            raise ValueError('Expected a JSON object')
        if not request.keys() <= {'command', 'instruction', 'parameters'}:
            raise ValueError('Unknown request fields')
        if ('command' in request) == ('instruction' in request):
            raise ValueError('Provide exactly one command or instruction')
        parameters = request.get('parameters', {})
        if not isinstance(parameters, dict):
            raise ValueError('parameters must be an object')
        parameters = dict(parameters)
        if 'instruction' in request:
            instruction = request['instruction']
            if not isinstance(instruction, str):
                raise ValueError('instruction must be a string')
            phrase = ' '.join(instruction.lower().strip().rstrip('.!').split())
            if phrase not in PHRASES:
                raise ValueError('Unsupported instruction; see GET /api/commands for supported phrases')
            command, inferred = PHRASES[phrase]
            if parameters.keys() & inferred.keys():
                raise ValueError('Instruction parameters cannot override its meaning')
            parameters.update(inferred)
        else:
            command = request['command']
        if not isinstance(command, str) or command not in FIELDS:
            raise ValueError('Unknown command')
        if not parameters.keys() <= set(FIELDS[command]):
            raise ValueError('Unknown command parameters')
        self.dispatch(command, parameters)
        return dict(accepted=True, command=command, status='queued')

    def dispatch(self, command, payload):
        store = self.store
        if command == 'alignment.assist':
            if type(payload.get('enabled')) is not bool:
                raise ValueError('enabled must be a boolean')
            store.submit('alignment_assist', {'enabled': payload['enabled']})
            return
        if command == 'alignment.nudge':
            state = store.snapshot()
            if state.get('alignment_servo', {}).get('active'):
                raise ValueError('Release the visual servo before requesting a manual pulse')
            press = payload.get('press_id')
            if (type(payload.get('held')) is not bool or not isinstance(press, str)
                    or not 1 <= len(press) <= 80):
                raise ValueError('held must be boolean and press_id must identify the press')
            if (state.get('alignment_assist', {}).get('mode') != 'operator_supervised'
                    or not payload.get('episode_id') or payload['episode_id'] != state.get('active')
                    or state.get('phase') != 'recording'):
                raise ValueError('Supervised trial requires the current active recording')
            store.submit('alignment_operator_hold', dict(held=payload['held'], press_id=press,
                episode_id=payload['episode_id'], received_monotonic_ns=time.monotonic_ns()))
            return
        if command == 'gear.grasp':
            state = store.snapshot()
            if (payload.get('action') not in ('close', 'release')
                    or not payload.get('episode_id') or payload['episode_id'] != state.get('active')
                    or state.get('phase') != 'recording' or not state.get('control_active')
                    or state.get('alignment_assist', {}).get('mode') != 'operator_supervised'):
                raise ValueError('Gear grasp requires a current supervised recording and close/release action')
            store.submit('gear_grasp', dict(action=payload['action'], episode_id=payload['episode_id'],
                received_monotonic_ns=time.monotonic_ns()))
            return
        if command == 'alignment.settings':
            state = store.snapshot()
            settings = payload.get('settings')
            bounds = {'max_joint_distance_rad': (.05, .4), 'max_error_px': (10., 100.)}
            if (state.get('alignment_assist', {}).get('mode') != 'operator_supervised'
                    or not isinstance(settings, dict) or not settings or not settings.keys() <= bounds.keys()
                    or state.get('alignment_assist', {}).get('active')):
                raise ValueError('Settings require an inactive supervised trial')
            for name, value in settings.items():
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not bounds[name][0] <= value <= bounds[name][1]):
                    raise ValueError('Setting outside supervised trial bounds')
            store.submit('alignment_trial_settings', settings)
            return
        if command == 'alignment.servo':
            state = store.snapshot()
            if type(payload.get('enabled')) is not bool:
                raise ValueError('enabled must be boolean')
            if payload['enabled'] and (state.get('phase') != 'recording'
                    or not state.get('control_active')
                    or state.get('alignment_assist', {}).get('mode') != 'operator_supervised'
                    or not state.get('gear_grasp', {}).get('settled')):
                raise ValueError('Visual servo needs a recording and settled gear grasp')
            store.submit('alignment_servo', dict(enabled=payload['enabled'],
                episode_id=state.get('active'), received_monotonic_ns=time.monotonic_ns()))
            return
        if command != 'episode.finish' or payload.get('label') not in ('kept', 'review', 'discarded'):
            raise ValueError('Unknown action')
        if payload.get('episode_id') != store.snapshot().get('active') or not payload.get('episode_id'):
            raise ValueError('Episode changed; action was not applied')
        store.submit('decision', payload)
