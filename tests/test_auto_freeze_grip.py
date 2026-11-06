"""Automatic freeze adds bounded squeeze once, preserving manual freeze behavior."""
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

import numpy as np
import pytest

from test_inference_dashboard import dashboard, FakeHardware, fake_request, wait_for


@pytest.mark.parametrize('initial, lower, manual', [(.3, 0., False), (.02, 0., False), (.3, 0., True)])
def test_freeze_squeeze_uses_existing_command_once_and_manual_request_wins(initial, lower, manual):
    class ContactHardware(FakeHardware):
        measured_gripper = None

        def read(self):
            measured = self.position.copy()
            if self.measured_gripper is not None:
                measured[13] = self.measured_gripper
            return measured, {}, {}

    hardware = ContactHardware()
    hardware.lower[[6, 13]] = lower
    hardware.position[[6, 13]] = initial
    trigger = threading.Event()
    with tempfile.TemporaryDirectory() as directory, patch.object(dashboard, 'request', side_effect=fake_request):
        controller = dashboard.Controller('unused', lambda: hardware,
            log_path=Path(directory)/'steps.jsonl', data_dir=directory)
        # Keep returning a trigger after freezing: the controller must never
        # ratchet the preload farther closed on subsequent control ticks.
        def detect(*args, **kwargs):
            if trigger.is_set() and manual:
                controller._freeze_requested = True
                trigger.clear()
                return True
            return trigger.is_set()

        with patch.object(controller.right_settle, 'update', side_effect=detect):
            controller.connect(reset_pose=False)
            try:
                wait_for(lambda: controller.snapshot()['mode'] == 'holding')
                controller.heartbeat()
                before = np.asarray(controller.snapshot()['target'])
                hardware.measured_gripper = initial + .15
                trigger.set()
                wait_for(lambda: controller.snapshot()['right_arm_frozen'])
                expected = initial if manual else max(lower, initial-.05)
                held = np.asarray(controller.snapshot()['right_arm_freeze_target'])
                np.testing.assert_allclose(held[:6], before[7:13])
                assert held[6] == pytest.approx(expected)
                # Projection must preserve squeeze through future policy chunks,
                # leaving the other gripper and the source action array alone.
                actions = np.full((3, 14), .8)
                projected = controller.project_right_hold(actions)
                np.testing.assert_allclose(projected[:, 13], expected)
                np.testing.assert_allclose(projected[:, :7], .8)
                np.testing.assert_allclose(actions, .8)
                start = len(hardware.commands)
                wait_for(lambda: len(hardware.commands) >= start+5)
                np.testing.assert_allclose([q[13] for q in hardware.commands[start:]], expected)
                assert controller.snapshot()['target'][6] == pytest.approx(initial)
            finally:
                controller.shutdown()
                controller.worker.join(3)
        events = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
        freezes = [event for event in events if event.get('event') == 'right_arm_freeze']
        assert len(freezes) == 1
        assert freezes[0]['automatic'] is (not manual)
