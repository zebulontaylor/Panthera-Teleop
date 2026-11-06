"""Causal pin pickup and carrier-region handoff for guarded contact approach.

This module never commands hardware or confirms physical contact. XYZ inputs
are native follower TCP coordinates from the caller's FK, in metres. Relative
carrier windows encode the fixed installation's base separation; they are not
portable calibration. Gripper coordinates are native radians.
"""
import math
from collections import deque


class PinContactDetector:
    DEFAULTS = dict(enabled=False, pickup_dwell_s=.15, contact_dwell_s=.15,
                    max_gap_s=.1, pickup_z_max_m=.05, pickup_x_min_m=.21, lift_m=.015,
                    descent_m=.01, stall_travel_m=.0005,
                    open_grip_rad=.5, narrow_grip_min_rad=-.02,
                    narrow_grip_max_rad=.25)
    REGION = ((-.07, .02), (-.64, -.55), (.015, .04))

    def __init__(self, config=None):
        self.config = dict(self.DEFAULTS)
        self.reset()
        self.configure(config)

    @property
    def enabled(self):
        return self.config['enabled']

    @property
    def latched(self):
        return self._latched

    def configure(self, config=None, **values):
        if type(config) is bool:
            config = dict(enabled=config)
        updated = dict(self.config)
        for key, value in {**(config or {}), **values}.items():
            if key not in updated:
                raise ValueError('unknown pin contact setting: '+key)
            if key == 'enabled':
                if type(value) is not bool:
                    raise ValueError('enabled must be boolean')
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or (value <= 0 and key != 'narrow_grip_min_rad')):
                raise ValueError(key+' must be finite and positive')
            updated[key] = value
        if updated['narrow_grip_min_rad'] >= updated['narrow_grip_max_rad']:
            raise ValueError('invalid narrow grip range')
        if updated != self.config or 'enabled' in (config or {}) or 'enabled' in values:
            self.config = updated
            self.reset()
        return dict(self.config)

    def set_enabled(self, enabled):
        return self.configure(enabled=enabled)

    def reset(self):
        self.phase = 'waiting_for_open_grip'
        self.last_time = self.pickup_since = self.pickup_z = self.peak_z = None
        self._pickup_grip = self._pickup_anchor_z = self._pickup_low_z = None
        self._stall = deque(maxlen=256)  # Region dwell; legacy name retained internally.
        self._last_result = None
        self._latched = False
        self.result = dict(enabled=self.config['enabled'], status='observing' if self.config['enabled'] else 'disabled', phase=self.phase, reason='waiting for open left grip evidence', triggered=False, contact_candidate=False, contact_confirmed=False, motor_permission=False)

    def complete(self):
        """Latch completion; explicit reset/rearm is required for another attempt."""
        self._latched = True
        self.phase = 'complete'
        self._stall.clear()
        return self._result('attempt complete; explicit rearm required')

    rearm = reset

    @staticmethod
    def _vector(value, length):
        try:
            return len(value) == length and all(not isinstance(v, bool)
                and isinstance(v, (int, float)) and math.isfinite(v) for v in value)
        except TypeError:
            return False

    def _result(self, reason, *, triggered=False, candidate=False, **diagnostics):
        result = dict(enabled=self.config['enabled'], status='disabled' if not self.enabled else 'triggered' if triggered
            else 'complete' if self.phase == 'complete' else 'assisting' if self._latched else 'observing', phase=self.phase,
            reason=reason, triggered=triggered, contact_candidate=candidate,
            contact_confirmed=False, heuristic='pin_pickup_lift_carrier_region_v2',
            motor_permission=False, **diagnostics)
        self._last_result = result
        self.result = result
        return result

    def record(self):
        """JSON-safe latest evidence for recording; triggering never repeats here."""
        return dict(self._last_result or {}, triggered=False,
                    config=dict(self.config), carrier_relative_region_m=[list(v) for v in self.REGION])

    def update(self, now, measured14, left_xyz, right_xyz, eligible,
               right_frozen, commanded_left_xyz=None, *, allow_trigger=True):
        if not self.config['enabled']:
            self.reset()
            return self._result('disabled')
        if self._latched:
            return self._result('attempt latched; rearm required', candidate=self.phase == 'contact_candidate')
        if eligible is not True:
            self.reset()
            return self._result('active policy context required')
        if (isinstance(now, bool) or not isinstance(now, (int, float))
                or not math.isfinite(now) or now < 0
                or not self._vector(measured14, 14)
                or not self._vector(left_xyz, 3) or not self._vector(right_xyz, 3)):
            self._stall.clear();self.pickup_since = None
            return self._result('fresh finite measured state and native FK required')
        if self.last_time is not None and now <= self.last_time:
            self._stall.clear();self.pickup_since = None
            return self._result('strictly increasing measurement time required')
        if self.last_time is not None and now-self.last_time > self.config['max_gap_s']:
            self._stall.clear();self.pickup_since = None
        self.last_time = now
        if self._latched:
            return self._result('attempt latched; rearm required')
        c = self.config
        grip, z = measured14[6], left_xyz[2]
        relative = [a-b for a,b in zip(left_xyz,right_xyz)]
        in_region = all(lo <= v <= hi for v,(lo,hi) in zip(relative,self.REGION))
        downward_intent = (self._vector(commanded_left_xyz,3)
                           and commanded_left_xyz[2] < z-.0005)
        diagnostic = dict(relative_xyz_m=relative, in_carrier_region=in_region,
                          commanded_downward_intent=bool(downward_intent))
        if grip > c['open_grip_rad']:
            self.phase = 'waiting_for_pin_grasp'
            self.pickup_since = self.pickup_z = self.peak_z = None
            self._stall.clear()
            return self._result('open left grip observed', **diagnostic)
        narrow = c['narrow_grip_min_rad'] < grip < c['narrow_grip_max_rad']
        if self.phase == 'waiting_for_open_grip':
            return self._result('waiting for open left grip evidence', **diagnostic)
        if self.phase == 'waiting_for_pin_grasp':
            # Remember the causal low-table closure while the smooth pickup
            # starts lifting. Grip settling need not force a pause on the table.
            if not narrow:
                self.pickup_since = self._pickup_low_z = None
            else:
                if z < c['pickup_z_max_m'] and left_xyz[0] > c['pickup_x_min_m']:
                    self._pickup_low_z = z if self._pickup_low_z is None else min(z, self._pickup_low_z)
                if self._pickup_low_z is not None:
                    if self.pickup_since is None or abs(grip-self._pickup_grip) > .02:
                        self.pickup_since, self._pickup_grip = now, grip
                    elif now-self.pickup_since >= c['pickup_dwell_s']:
                        self.phase, self.pickup_z, self.peak_z = 'pin_grasped', self._pickup_low_z, z
            return self._result('collecting narrow-grip pickup and early lift evidence', **diagnostic)
        if not narrow:
            self.reset()
            return self._result('pin grasp evidence lost', **diagnostic)
        self.peak_z = max(self.peak_z, z)
        if self.phase == 'pin_grasped':
            if self.peak_z-self.pickup_z >= c['lift_m']:
                self.phase = 'transporting_pin'
            return self._result('waiting for pin lift and carrier transport', **diagnostic)
        if not in_region or right_frozen is not True:
            self._stall.clear()
            return self._result('waiting for carrier approach region and frozen right arm', **diagnostic)
        self.phase = 'checking_carrier_region'
        self._stall.append((now, z))
        while len(self._stall)>1 and now-self._stall[1][0] >= c['contact_dwell_s']:
            self._stall.popleft()
        window = now-self._stall[0][0]
        diagnostic.update(region_dwell_s=window,
                          approach_descent_m=self.peak_z-z)
        if window >= c['contact_dwell_s']:
            if allow_trigger is not True:
                return self._result('carrier region ready; waiting for active policy handoff', **diagnostic)
            self._latched = True
            self.phase = 'contact_candidate'
            return self._result('sustained carrier region entry; guarded descent required to infer contact',
                                triggered=True, candidate=True, **diagnostic)
        return self._result('checking sustained carrier region entry', **diagnostic)
