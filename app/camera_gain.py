"""Linux V4L2 hardware gain controls, independent of OpenCV capture handles.

Layouts and ioctl numbers follow linux/videodev2.h and v4l2-controls.h.
No exposure or automatic gain controls are changed implicitly.
"""
import errno
import fcntl
import os
import struct

V4L2_CID_GAIN = 0x00980913
_QUERY = struct.Struct('=II32siiiiI2I')
_CONTROL = struct.Struct('=Ii')
# Linux _IOWR('V', number, struct): read/write, size, type, number.
VIDIOC_QUERYCTRL = (3 << 30) | (_QUERY.size << 16) | (ord('V') << 8) | 36
VIDIOC_G_CTRL = (3 << 30) | (_CONTROL.size << 16) | (ord('V') << 8) | 27
VIDIOC_S_CTRL = (3 << 30) | (_CONTROL.size << 16) | (ord('V') << 8) | 28
DISABLED, GRABBED, READ_ONLY, INACTIVE = 1, 2, 4, 16


def _query(fd):
    buffer = bytearray(_QUERY.pack(V4L2_CID_GAIN, 0, b'', 0, 0, 0, 0, 0, 0, 0))
    try:
        fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buffer, True)
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.ENOTTY):
            return dict(supported=False, writable=False, error='Camera does not support hardware gain')
        raise
    _, kind, _, minimum, maximum, step, default, flags, _, _ = _QUERY.unpack(buffer)
    supported = not bool(flags & DISABLED) and kind == 1
    result = dict(supported=supported, writable=supported and not bool(flags & (GRABBED | READ_ONLY | INACTIVE)),
                  min=minimum, max=maximum, step=step, default=default, flags=flags)
    if not supported:
        result['error'] = 'Camera does not support an enabled integer gain control'
        return result
    if flags & READ_ONLY:
        result['error'] = 'Camera gain is read-only'
    elif flags & INACTIVE:
        result['error'] = 'Camera gain is inactive; check camera automatic controls'
    elif flags & GRABBED:
        result['error'] = 'Camera gain is currently locked by the device'
    value = bytearray(_CONTROL.pack(V4L2_CID_GAIN, 0))
    fcntl.ioctl(fd, VIDIOC_G_CTRL, value, True)
    result['value'] = _CONTROL.unpack(value)[1]
    return result


def gain_status(device):
    """Read gain capability and current value; failures do not break capture."""
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        try:
            return _query(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        return dict(supported=False, writable=False, error=f'Cannot read camera gain: {exc}')


def set_gain(device, value):
    """Set a valid integer hardware gain and report the device readback."""
    if type(value) is not int:
        raise ValueError('Camera gain must be an integer')
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        try:
            status = _query(fd)
            if not status['supported'] or not status['writable']:
                raise RuntimeError(status.get('error', 'Camera gain is unavailable'))
            minimum, maximum, step = status['min'], status['max'], status['step']
            if not minimum <= value <= maximum or (value-minimum) % max(1, step):
                raise ValueError(f'Camera gain must be from {minimum} to {maximum} in steps of {max(1, step)}')
            buffer = bytearray(_CONTROL.pack(V4L2_CID_GAIN, value))
            fcntl.ioctl(fd, VIDIOC_S_CTRL, buffer, True)
            return _query(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise RuntimeError(f'Cannot set camera gain: {exc}') from exc
