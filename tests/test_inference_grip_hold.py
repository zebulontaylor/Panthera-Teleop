"""Hardware-free regressions for preserving a loaded grip during correction."""
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import inference_dashboard as dashboard
from hitl_demo import DemoHardware


class LoadedGripHardware(DemoHardware):
    """An object prevents the measured fingers reaching their squeeze target."""
    def __init__(self):
        super().__init__()
        self.auto_pull = False
        self.config['takeover_hold'].update(baseline_seconds=.03)
        self.config['motion_confirm_seconds'] = .03
        self.gripper_torque = dict(left=.35, right=.35)
        self.grip_error = .04
        self.position[[6, 13]] = .5
        self.commands = []
        self.guides = []
        self.reads = 0

    def open(self):
        # No camera, recorder, pedal, network inference, or physical devices.
        pass

    def read(self):
        self.reads += 1
        return super().read()

    def command(self, target):
        super().command(target)
        self.commands.append(target.copy())
        self.position[[6, 13]] += self.grip_error

    def guide(self, arms, target, released):
        super().guide(arms, target, released)
        for side, index in (('left', 6), ('right', 13)):
            if not released[side]:
                self.position[index] += self.grip_error
        self.guides.append((target.copy(), dict(released),
                            {side: arm['gripper']['control_mode'] for side, arm in arms.items()}))


def health_request(base, path, body=None, timeout=3):
    if path != '/health':
        raise AssertionError('These tests must never request policy playback')
    return dict(status='ready', contract=dashboard.CONTRACT, checkpoint='loaded-grip-test')


class InferenceGripHoldTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.hardware = LoadedGripHardware()
        self.controller = dashboard.Controller('unused', lambda: self.hardware,
            log_path=Path(directory.name)/'inference.jsonl', data_dir=Path(directory.name)/'recordings')
        request_patch = patch.object(dashboard, 'request', health_request)
        request_patch.start()
        self.addCleanup(request_patch.stop)
        self.addCleanup(self.shutdown)
        self.controller.connect(reset_pose=False)
        self.wait(lambda state: state['mode'] == 'holding' and self.hardware.commands)
        self.squeeze_target = self.hardware.commands[-1][[6, 13]].copy()

    def shutdown(self):
        self.controller.shutdown()
        if self.controller.worker:
            self.controller.worker.join(2)

    def wait(self, predicate, timeout=2):
        deadline = time.monotonic()+timeout
        while True:
            self.controller.heartbeat()
            state = self.controller.snapshot()
            if predicate(state):
                return state
            if time.monotonic() >= deadline:
                self.fail(f'Controller did not reach expected state: {state}')
            time.sleep(.003)

    def intervene(self):
        self.controller.collection('intervene', {})
        return self.wait(lambda state: state['mode'] == 'takeover_hold' and
                         state['handoff_baseline_ready'] and not state['collection_pending'])

    def assert_waiting_grip(self, side):
        target, released, modes = self.hardware.guides[-1]
        index = 6 if side == 'left' else 13
        expected = self.squeeze_target[0 if side == 'left' else 1]
        self.assertFalse(released[side])
        self.assertEqual(modes[side], 'position_hold')
        self.assertAlmostEqual(target[index], expected)
        self.assertAlmostEqual(self.hardware.position[index], expected+self.hardware.grip_error)

    def test_loaded_grippers_exceeding_arm_threshold_keep_squeeze_without_fault(self):
        self.hardware.grip_error = .4
        reads = self.hardware.reads
        state = self.wait(lambda state: self.hardware.reads >= reads+15)
        self.assertEqual(state['mode'], 'holding')
        self.assertFalse(state['error'])
        np.testing.assert_allclose(self.hardware.commands[-1][[6, 13]], self.squeeze_target)
        np.testing.assert_allclose(self.hardware.position[[6, 13]], self.squeeze_target+.4)

    def test_constant_preload_holds_and_resume_and_stop_preserve_squeeze(self):
        self.intervene()
        reads = self.hardware.reads
        state = self.wait(lambda state: self.hardware.reads >= reads+15)
        self.assertEqual(state['handoff_ready'], dict(left=False, right=False))
        self.assert_waiting_grip('left')
        self.assert_waiting_grip('right')
        self.controller.collection('resume', {})
        self.wait(lambda state: state['mode'] == 'holding' and not state['collection_pending'])
        np.testing.assert_allclose(self.hardware.commands[-1][[6, 13]], self.squeeze_target)
        self.intervene()
        self.controller.stop()
        self.wait(lambda state: state['mode'] == 'holding' and not self.controller.pause_event.is_set())
        np.testing.assert_allclose(self.hardware.commands[-1][[6, 13]], self.squeeze_target)

    def test_force_change_releases_one_arm_and_stop_holds_other_grip(self):
        self.intervene()
        self.hardware.gripper_torque['left'] += .2
        self.wait(lambda state: state['mode'] == 'human' and
                  state['handoff_ready'] == dict(left=True, right=False) and
                  self.hardware.guides[-1][1] == dict(left=True, right=False))
        self.assertEqual(self.hardware.guides[-1][2]['left'], 'passive')
        self.assert_waiting_grip('right')
        self.controller.stop()
        self.wait(lambda state: state['mode'] == 'holding' and not self.controller.pause_event.is_set())
        self.assertAlmostEqual(self.hardware.commands[-1][13], self.squeeze_target[1])

    def test_resume_after_one_arm_releases_preserves_other_grip(self):
        self.intervene()
        self.hardware.gripper_torque['right'] -= .2
        self.wait(lambda state: state['handoff_ready'] == dict(left=False, right=True) and
                  self.hardware.guides[-1][1] == dict(left=False, right=True))
        self.assert_waiting_grip('left')
        self.controller.collection('resume', {})
        self.wait(lambda state: state['mode'] == 'holding' and not state['collection_pending'])
        self.assertAlmostEqual(self.hardware.commands[-1][6], self.squeeze_target[0])
        self.assertFalse(self.controller.snapshot()['playing'])
