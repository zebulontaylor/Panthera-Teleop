"""Explicit, episode-bound small closing step for the left gear gripper only."""
import math


class GearGrasp:
    def __init__(self):
        self.cancel()

    def cancel(self):
        self.episode = None
        self.started = None
        self.initial = self.target = None
        self.settled = False

    def start(self, now, episode, gripper, limits):
        if self.episode == episode:
            return  # Repeated requests never accumulate closing steps.
        p, v = float(gripper['position']), float(gripper['velocity'])
        lo, hi = float(limits['lower']), float(limits['upper'])
        if (not all(math.isfinite(x) for x in (now, p, v, lo, hi))
                or not lo < hi or not lo-.03 <= p <= hi or abs(v) > .1 or not episode):
            raise ValueError('Gear gripper must have valid, stationary feedback')
        self.episode, self.started, self.initial = episode, now, p
        self.target = max(lo, p-.05)
        self.settled = False

    def tick(self, now, episode, gripper, allowed):
        if self.episode is None:
            return None
        if not allowed or episode != self.episode:
            self.cancel()
            return None
        p, v = float(gripper['position']), float(gripper['velocity'])
        if not all(math.isfinite(x) for x in (p, v)) or p > self.initial+.04:
            self.cancel()
            return None
        self.settled = now-self.started >= .6 and abs(v) <= .1
        return dict(mode='position_velocity', control_mode='gear_hold', position_rad=self.target,
                    speed_limit_rad_s=.1, torque_limit_nm=.15)

    def status(self):
        return dict(active=self.episode is not None, settled=self.settled,
                    side='left', target_position_rad=self.target, max_closing_step_rad=.05,
                    torque_limit_nm=.15, speed_limit_rad_s=.1)
