"""Isolation of legacy torque pulses from the experimental dashboard."""
import time

from alignment_assist import AlignmentAssist
from robot_commands import RobotCommands


class PreviewOnlyAlignmentAssist(AlignmentAssist):
    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        result.update(active=False, torque_nm=[0.]*6, delta_rad=[0.]*6)
        if result['reason'] == 'nudging':
            result['reason'] = 'use_cartesian_single_step'
        return result


class CartesianCommands(RobotCommands):
    def __init__(self, store, incremental=False, relaxed=False):
        super().__init__(store)
        self.incremental = incremental
        self.relaxed = incremental and relaxed

    def dispatch(self, command, payload):
        if self.relaxed and command == 'alignment.servo' and payload.get('enabled') is True:
            state = self.store.snapshot()
            if (state.get('phase') != 'recording' or not state.get('active')
                    or not state.get('control_active')
                    or state.get('alignment_assist', {}).get('mode') != 'operator_supervised'
                    or not state.get('gear_grasp', {}).get('active')):
                raise ValueError('Visual servo needs a recording and an engaged gear grasp')
            self.store.submit('alignment_servo', dict(enabled=True, episode_id=state['active'],
                received_monotonic_ns=time.monotonic_ns()))
            return
        if command == 'alignment.nudge':
            action = 'Align vertically with repeated nudges' if self.incremental else 'One vertical nudge and hold'
            raise ValueError(f'Legacy torque pulse disabled here; use {action}')
        if command == 'alignment.settings':
            raise ValueError('Calibration bounds cannot be widened in Cartesian nudge mode')
        return super().dispatch(command, payload)
