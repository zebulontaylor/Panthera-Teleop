"""Conservative temporal visual advice. No contact inference or motor access.

Call update with the latest monitor proposal and a monotonic nanosecond clock.
The caller supplies policy eligibility and the actual right-arm frozen state.
Both flags must stay true throughout evidence collection.
"""
import json
import math


class PinNudgeDetector:
    DEFAULTS = dict(enabled=False, min_frames=3, min_span_s=1.,
                    max_frame_age_ms=1250., deadband_px=3.)
    SUPPORTED_HOLES = frozenset(('demo_reference_tracked', 'demo_reference_held',
        'captured_demo_reference', 'captured_operator_reference',
        'left_carrier_cnn_observed', 'left_carrier_opencv_observed',
        'left_carrier_layout_inferred', 'left_carrier_stationary_accumulated'))

    def __init__(self, config=None):
        self.config = dict(self.DEFAULTS)
        self.reset()
        self.configure(config)

    def reset(self):
        self._key = self._pose = self._direction = None
        self._count = 0
        self._decision = None
        self._first = self._last = None
        self._last_frame = None

    def configure(self, config=None, **values):
        updated = dict(self.config)
        for key, value in {**(config or {}), **values}.items():
            if key not in updated:
                raise ValueError('unknown nudge detection config: ' + key)
            if key == 'enabled':
                if type(value) is not bool:
                    raise ValueError('enabled must be boolean')
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value)):
                raise ValueError(key + ' must be finite numeric')
            elif key == 'min_frames' and (int(value) != value or value < 3):
                raise ValueError('at least three distinct frames required')
            elif key == 'min_span_s' and value < 1:
                raise ValueError('at least one second required')
            elif key == 'max_frame_age_ms' and not 0 < value <= 2000:
                raise ValueError('frame age limit must be in (0, 2000] ms')
            elif key == 'deadband_px' and not 0 <= value <= 100:
                raise ValueError('deadband must be in [0, 100] pixels')
            updated[key] = value
        if updated != self.config:
            self.config = updated
            self.reset()
        return dict(self.config)

    def set_enabled(self, enabled):
        return self.configure(enabled=enabled)

    def _result(self, status, reason, needed=None):
        return dict(status=status, state=status, reason=reason, needs_nudge=needed,
                    enabled=self.config['enabled'], frame_count=self._count,
                    span_s=(self._last-self._first)/1e9 if self._first is not None else 0.,
                    advisory_only=True, motor_permission=False)

    def _unknown(self, reason):
        self.reset()
        return self._result('unknown', reason)

    @staticmethod
    def _vector(value, length):
        return (isinstance(value, (list, tuple)) and len(value) == length
                and all(not isinstance(x, bool) and isinstance(x, (int, float))
                        and math.isfinite(x) for x in value))

    def update(self, proposal, now_ns, right_frozen, eligible=True):
        if not self.config['enabled']:
            return self._unknown('disabled')
        if right_frozen is not True or eligible is not True:
            return self._unknown('requires frozen right arm and eligible policy context')
        p = proposal if isinstance(proposal, dict) else {}
        stamp = p.get('analyzed_timestamp_ns')
        if (type(now_ns) is not int or type(stamp) is not int or stamp <= 0
                or not 0 <= now_ns-stamp <= self.config['max_frame_age_ms']*1e6
                or p.get('stale_result') or p.get('status') != 'ready'):
            return self._unknown('fresh analyzed frame required')
        if (p.get('localization_ready') is not True or p.get('nudge_valid') is not True
                or p.get('hole_source') not in self.SUPPORTED_HOLES
                or p.get('pin_source') not in ('cnn', 'operator_selected')):
            return self._unknown('supported pin and hole localization required')
        pose, pin, hole = p.get('measured_state14'), p.get('pin_px'), p.get('hole_px')
        frame = p.get('frame_id')
        if (type(frame) is not int or not self._vector(pose, 14)
                or not self._vector(pin, 2) or not self._vector(hole, 2)
                or not all(0 <= v < limit for point in (pin, hole)
                           for v, limit in zip(point, (640, 480)))):
            return self._unknown('valid frame identity, native measured pose and image points required')
        # Capture timestamps identify explicit target changes; omit changing age fields.
        if any(p.get(k) is not None and not isinstance(p.get(k), dict)
               for k in ('captured_hole', 'captured_pin')):
            return self._unknown('invalid captured reference')
        captures = tuple((p.get(k) or {}).get('captured_timestamp_ns')
                         for k in ('captured_hole', 'captured_pin'))
        key = json.dumps([p.get(k) for k in ('generation', 'camera_id', 'camera_role',
            'grasp_id', 'context_id', 'target_hole_index')] + [captures], sort_keys=True)
        joints = tuple(pose[:6]) + tuple(pose[7:13])
        changed = self._key != key or (self._pose is not None and
                    any(abs(a-b) > .01 for a, b in zip(joints, self._pose)))
        if self._last is not None and stamp-self._last > self.config['max_frame_age_ms']*1e6:
            changed = True
        if changed:
            self.reset()
            self._key, self._pose = key, joints
        if self._last is not None:
            if stamp == self._last and frame == self._last_frame:
                if self._decision is not None:
                    return self._result(*self._decision)
                return self._result('waiting', 'distinct increasing fresh frames required')
            if stamp <= self._last or frame <= self._last_frame:
                return self._unknown('regressing or inconsistent frame identity')
        error = tuple(h-p for h, p in zip(hole, pin))
        aligned = math.hypot(*error) <= self.config['deadband_px']
        direction = tuple(0 if abs(v) <= self.config['deadband_px'] else (1 if v > 0 else -1)
                          for v in error)
        if self._direction is not None and (aligned != self._aligned or any(
                a*b < 0 for a, b in zip(direction, self._direction))):
            self.reset()
            self._key, self._pose = key, joints
        self._direction, self._aligned = direction, aligned
        self._count += 1
        self._last_frame = frame
        if self._first is None:
            self._first = stamp
        self._last = stamp
        if (self._count < self.config['min_frames'] or
                stamp-self._first < self.config['min_span_s']*1e9):
            return self._result('waiting', 'collecting persistent visual evidence')
        self._decision = ('aligned' if aligned else 'needed',
            'persistent visual alignment within pixel deadband' if aligned else
            'persistent visual misalignment', not aligned)
        return self._result(*self._decision)
