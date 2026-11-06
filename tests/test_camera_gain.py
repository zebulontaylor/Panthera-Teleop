"""Hardware gain protocol and validation without opening physical cameras."""
import errno
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import camera_gain as gain


class CameraGainTests(unittest.TestCase):
    def setUp(self):
        self.value = 20
        self.flags = 0
        self.writes = []
        self.open = patch.object(gain.os, 'open', return_value=42).start()
        self.close = patch.object(gain.os, 'close').start()
        self.ioctl = patch.object(gain.fcntl, 'ioctl', side_effect=self.control).start()
        self.addCleanup(patch.stopall)

    def control(self, fd, command, buffer, mutate):
        self.assertEqual(fd, 42)
        if command == gain.VIDIOC_QUERYCTRL:
            buffer[:] = gain._QUERY.pack(gain.V4L2_CID_GAIN, 1, b'Gain', 10, 100, 5, 25, self.flags, 0, 0)
        elif command == gain.VIDIOC_G_CTRL:
            buffer[:] = gain._CONTROL.pack(gain.V4L2_CID_GAIN, self.value)
        elif command == gain.VIDIOC_S_CTRL:
            cid, self.value = gain._CONTROL.unpack(buffer)
            self.assertEqual(cid, gain.V4L2_CID_GAIN)
            self.writes.append(self.value)
        else:
            self.fail(f'Unexpected ioctl {command}')

    def test_query_layout_and_metadata(self):
        self.assertEqual(gain._QUERY.size, 68)
        self.assertEqual(gain.VIDIOC_QUERYCTRL, 0xc0445624)
        status = gain.gain_status('/camera')
        self.assertEqual(status, dict(supported=True, writable=True, min=10, max=100,
                                     step=5, default=25, flags=0, value=20))
        self.assertFalse(self.writes)
        self.close.assert_called_once_with(42)

    def test_set_reads_back_and_closes(self):
        self.assertEqual(gain.set_gain('/camera', 35)['value'], 35)
        self.assertEqual(self.writes, [35])
        self.close.assert_called_once_with(42)

    def test_invalid_values_never_write(self):
        for value in (True, 30.0, '30', None, 9, 101, 32):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gain.set_gain('/camera', value)
        self.assertFalse(self.writes)

    def test_unavailable_flags_never_write(self):
        for flags in (gain.DISABLED, gain.READ_ONLY, gain.INACTIVE, gain.GRABBED):
            self.flags = flags
            with self.subTest(flags=flags), self.assertRaises(RuntimeError):
                gain.set_gain('/camera', 30)
        self.assertFalse(self.writes)

    def test_unsupported_query(self):
        self.ioctl.side_effect = OSError(errno.EINVAL, 'unsupported')
        self.assertFalse(gain.gain_status('/camera')['supported'])
        with self.assertRaises(RuntimeError):
            gain.set_gain('/camera', 30)
        self.assertFalse(self.writes)

    def test_io_failures_surface_and_close(self):
        self.ioctl.side_effect = OSError(errno.EIO, 'device gone')
        self.assertIn('device gone', gain.gain_status('/camera')['error'])
        with self.assertRaisesRegex(RuntimeError, 'device gone'):
            gain.set_gain('/camera', 30)
        self.assertEqual(self.close.call_count, 2)

    def test_open_failure_status_and_mutation(self):
        self.open.side_effect = OSError(errno.ENOENT, 'missing')
        self.assertFalse(gain.gain_status('/camera')['supported'])
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            gain.set_gain('/camera', 30)
        self.close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
