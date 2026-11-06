"""Bounded loaded descent before inferred carrier contact; no device access."""
import numpy as np

APPROACH_DOWN_M = .020
APPROACH_XY_M = .005
APPROACH_TIMEOUT_S = 30.
APPROACH_FORCE_N = .5
APPROACH_SPEED_M_S = .001
APPROACH_SPEED_RAMP_S = 1.
APPROACH_FORCE_SLEW_N_S = .15
# Stop a gravity/model mismatch rather than recovering a fast downward sag.
APPROACH_TRACKING_LAG_M = .0026
# Native FK can change about .54 mm from combined encoder quantization.
CONTACT_HEIGHT_RANGE_M = .0006


class ContactApproach:
    """Infer contact from stable height under established downward load.

    This is a motion heuristic, not a force measurement. Keep one fixed origin
    for the whole approach. Free descent never counts as an insertion drop.
    """
    def __init__(self, now, position):
        self.started = self.last_time = float(now)
        self.start = np.asarray(position, dtype=float).copy()
        if not np.isfinite(now) or self.start.shape != (3,) or not np.isfinite(self.start).all():
            raise ValueError('invalid_contact_approach_origin')
        self.stable = []
        self.stop_reason = None

    def update(self, now, position, *, load_established):
        p = np.asarray(position, dtype=float)
        if (not np.isfinite(now) or now < self.last_time or p.shape != (3,)
                or not np.isfinite(p).all()):
            raise ValueError('invalid_contact_approach_measurement')
        gap = now-self.last_time
        self.last_time = float(now)
        down = float(self.start[2]-p[2])
        if gap > .1:
            self.stop_reason = self.stop_reason or 'measurement_gap_stop'
        elif down >= APPROACH_DOWN_M:
            self.stop_reason = self.stop_reason or 'contact_approach_travel_limit'
        elif now-self.started >= APPROACH_TIMEOUT_S:
            self.stop_reason = self.stop_reason or 'contact_approach_timeout'
        ready = False
        if not self.stop_reason:
            # Let load ramp and initial settling finish before collecting the
            # half-second stable window. Initial stillness alone is insufficient.
            if not load_established or now-self.started < 1.:
                self.stable.clear()
            else:
                self.stable.append((now, float(p[2])))
                while len(self.stable)>1 and self.stable[1][0] <= now-.5:
                    self.stable.pop(0)
                heights = [v for _,v in self.stable]
                if max(heights)-min(heights) > CONTACT_HEIGHT_RANGE_M+1e-12:
                    self.stable = [(now, float(p[2]))]
                else:
                    if len(self.stable)>=3 and now-self.stable[0][0]>=.5:
                        times = np.asarray([t for t,_ in self.stable])
                        centered = times-times.mean()
                        slope = float(np.dot(centered, np.asarray(heights)-np.mean(heights))
                                      / np.dot(centered, centered))
                        # A slowly moving free pin can fit inside the encoder
                        # noise band, especially with a small effort setting.
                        ready = abs(slope) <= .00025
        return dict(contact_ready=ready, stop_reason=self.stop_reason,
                    phase='approaching_contact', baseline_z_m=None,
                    baseline_authority='inferred_contact_candidate',
                    downward_from_start_m=down, insertion_confirmed=False)
