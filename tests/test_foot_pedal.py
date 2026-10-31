"""Pedal decoding, event loss detection and device cleanup without USB hardware."""
import errno
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from foot_pedal import EVENT, FootPedal


class PedalTests(unittest.TestCase):
    def open_pedal(self, ioctl=None):
        with patch('foot_pedal.glob.glob', return_value=['pedal']), \
             patch('foot_pedal.os.open', return_value=42), \
             patch('foot_pedal.fcntl.ioctl', side_effect=ioctl):
            return FootPedal()

    def test_press_hold_repeat_and_release(self):
        pedal = self.open_pedal()
        with patch('foot_pedal.os.read', side_effect=[EVENT.pack(10, 123, 1, 48, 1), BlockingIOError()]):
            state = pedal.snapshot()
        self.assertTrue(state['physical_pressed'])
        self.assertEqual(state['events'], [dict(monotonic_ns=10_000_123_000, code=48, pressed=True)])
        with patch('foot_pedal.os.read', side_effect=[EVENT.pack(11, 0, 1, 48, 2), BlockingIOError()]):
            state = pedal.snapshot()
        self.assertTrue(state['physical_pressed'])
        self.assertEqual(state['events'], [])
        with patch('foot_pedal.os.read', side_effect=[EVENT.pack(12, 0, 1, 48, 0), BlockingIOError()]):
            self.assertFalse(pedal.snapshot()['physical_pressed'])

    def test_short_tap_preserves_both_edges(self):
        pedal = self.open_pedal()
        data = EVENT.pack(10, 1, 1, 48, 1) + EVENT.pack(10, 2, 1, 48, 0)
        with patch('foot_pedal.os.read', side_effect=[data, BlockingIOError()]):
            state = pedal.snapshot()
        self.assertFalse(state['physical_pressed'])
        self.assertEqual([e['pressed'] for e in state['events']], [True, False])

    def test_already_held_at_startup(self):
        def ioctl(fd, operation, argument, *args):
            if operation == 0x80604518:
                argument[6] = 1  # key 48
        pedal = self.open_pedal(ioctl)
        with patch('foot_pedal.os.read', side_effect=BlockingIOError()):
            self.assertTrue(pedal.snapshot()['physical_pressed'])

    def test_event_loss_and_disconnect_are_errors(self):
        pedal = self.open_pedal()
        for outcome in [EVENT.pack(1, 0, 0, 3, 0), OSError(errno.ENODEV, 'unplugged'), b'']:
            with self.subTest(outcome=outcome), patch('foot_pedal.os.read', side_effect=[outcome]):
                with self.assertRaises(RuntimeError):
                    pedal.snapshot()

    def test_missing_is_unknown_not_released(self):
        with patch('foot_pedal.glob.glob', return_value=[]):
            state = FootPedal().snapshot()
        self.assertFalse(state['connected'])
        self.assertIsNone(state['physical_pressed'])

    def test_permission_failure_cleans_up_grabs(self):
        with patch('foot_pedal.glob.glob', return_value=['pedal']), \
             patch('foot_pedal.os.open', return_value=42), \
             patch('foot_pedal.fcntl.ioctl', side_effect=PermissionError()), \
             patch('foot_pedal.os.close') as close:
            with self.assertRaisesRegex(RuntimeError, 'enable-foot-pedal'):
                FootPedal()
            close.assert_called_once_with(42)


if __name__ == '__main__':
    unittest.main()
