"""Causal, task-specific carrier settling heuristic; no motor commands."""
from collections import deque
import numpy as np


class RightArmSettleDetector:
    settle_s = 1.0
    tolerance_rad = .01

    def __init__(self):
        self.enabled = False
        self.reset()

    def reset(self):
        self.origin = None
        self.motion_seen = False
        self.samples = deque()
        self.last_time = None
        self.inhibited = False
        self.result = self.status('waiting', 'Waiting for policy motion')

    def status(self, status, reason, stable_s=0.):
        return dict(enabled=self.enabled, status=status if self.enabled else 'off',
                    reason=reason if self.enabled else 'Automatic freeze disabled',
                    stable_s=float(stable_s), settle_s=self.settle_s)

    def configure(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('enabled must be a boolean')
        self.enabled = enabled
        self.reset()
        return self.result

    def update(self, now, measured, target, eligible, frozen=False):
        q = np.asarray(measured, dtype=float)
        command = np.asarray(target, dtype=float)
        if q.shape != (14,) or command.shape != (14,) or not np.isfinite(q).all() or not np.isfinite(command).all() or not np.isfinite(now):
            self.samples.clear()
            self.result = self.status('waiting', 'Waiting for finite measured feedback')
            return False
        gap = self.last_time is not None and (now <= self.last_time or now-self.last_time > .1)
        self.last_time = now
        if gap:
            self.samples.clear()
        if not self.enabled or frozen or self.inhibited or not eligible:
            self.samples.clear()
            self.result = self.status('frozen' if frozen else 'waiting',
                'Right arm frozen' if frozen else 'Manual unfreeze: toggle auto freeze to rearm' if self.inhibited else 'Waiting for active policy playback')
            return False
        if self.origin is None:
            self.origin = q[7:13].copy()
        self.motion_seen |= bool(np.max(np.abs(q[7:13]-self.origin)) >= .1)
        # Oct30/31 native carrier-holding region. Never freeze the initial
        # neutral pose or the open-gripper pickup approach just because it pauses.
        plausible = q[7] > .35 and q[8] > 1.2 and q[13] < 1.3
        tracking = np.max(np.abs(q[7:13]-command[7:13])) <= .05
        if not self.motion_seen or not plausible or not tracking:
            self.samples.clear()
            self.result = self.status('waiting', 'Waiting for carrier holding pose' if not plausible else 'Waiting for carrier motion' if not self.motion_seen else 'Right arm has not reached its commanded pose')
            return False
        self.samples.append((now, q[7:14].copy()))
        while self.samples and np.max(np.ptp(np.stack([s[1] for s in self.samples]), axis=0)) > self.tolerance_rad:
            self.samples.popleft()
        duration = now-self.samples[0][0]
        ready = duration >= self.settle_s
        self.result = self.status('settled' if ready else 'settling',
            'Carrier holding pose stable for one second' if ready else 'Checking one second of carrier stability', duration)
        # Keep the window bounded at control frequency, including the boundary.
        while len(self.samples) > 1 and now-self.samples[1][0] >= self.settle_s:
            self.samples.popleft()
        return ready
