"""Read only the USB foot switch, using Linux evdev without extra dependencies."""
import fcntl
import glob
import os
import struct
import time


EVENT = struct.Struct('@llHHi')
DEFAULT_GLOB = '/dev/input/by-id/usb-PCsensor_FootSwitch-event*'
DESCRIPTION = dict(signal='physical_pressed', type='boolean',
                   meaning='Pedal taps end and classify the episode',
                   classification='1 tap kept, 2 review, 3 discarded',
                   timestamps='Linux input event timestamps on host CLOCK_MONOTONIC')


class FootPedal:
    def __init__(self, settings=None, enabled=True):
        settings = settings or {}
        self.devices = {}
        self.paths = []
        if not enabled or not settings.get('enabled', True):
            return
        try:
            for path in sorted(glob.glob(settings.get('device_glob', DEFAULT_GLOB))):
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                self.devices[fd] = set()
                self.paths.append(path)
                # Keep pedal keystrokes out of browser episode decisions and terminals.
                fcntl.ioctl(fd, 0x40044590, 1)  # EVIOCGRAB
                fcntl.ioctl(fd, 0x400445a0, struct.pack('@i', time.CLOCK_MONOTONIC))
                keys = bytearray(96)  # KEY_MAX = 0x2ff
                fcntl.ioctl(fd, 0x80604518, keys, True)  # EVIOCGKEY(96)
                self.devices[fd] = {code for code in range(768)
                                    if keys[code // 8] & (1 << (code % 8))}
        except Exception as exc:
            self.close()
            raise RuntimeError('Cannot read/grab USB foot pedal. Run '
                               'sudo ./scripts/enable-foot-pedal.sh, then restart recording. '
                               f'Details: {exc}') from exc

    def snapshot(self):
        events = []
        for fd, held in self.devices.items():
            while True:
                try:
                    data = os.read(fd, EVENT.size * 64)
                except BlockingIOError:
                    break
                except OSError as exc:
                    raise RuntimeError(f'Foot pedal disconnected or unreadable: {exc}') from exc
                if not data or len(data) % EVENT.size:
                    raise RuntimeError('Foot pedal input stream ended or returned a partial event')
                for sec, usec, kind, code, value in EVENT.iter_unpack(data):
                    if kind == 0 and code == 3:  # SYN_DROPPED
                        raise RuntimeError('Foot pedal events lost; recording stopped to avoid incorrect episode classification')
                    if kind != 1 or value not in (0, 1):  # EV_KEY; ignore autorepeat
                        continue
                    if value:
                        held.add(code)
                    else:
                        held.discard(code)
                    events.append(dict(monotonic_ns=sec * 1_000_000_000 + usec * 1000,
                                       code=code, pressed=bool(value)))
        return dict(connected=bool(self.devices),
                    physical_pressed=any(self.devices.values()) if self.devices else None,
                    sampled_monotonic_ns=time.monotonic_ns(),
                    events=sorted(events, key=lambda event: event['monotonic_ns']))

    def close(self):
        for fd in self.devices:
            os.close(fd)  # Also releases the exclusive grab.
        self.devices.clear()


if __name__ == '__main__':
    import json
    pedal = FootPedal()
    try:
        if not pedal.devices:
            raise SystemExit('No PCsensor foot pedal found')
        print('Pedal monitor: press, hold, release. Ctrl+C exits.', flush=True)
        previous = None
        while True:
            state = pedal.snapshot()
            if state['events'] or state['physical_pressed'] != previous:
                print(json.dumps(state), flush=True)
                previous = state['physical_pressed']
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        pedal.close()
