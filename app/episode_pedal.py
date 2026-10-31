"""Release-delimited keep/review/delete gestures for active teleop episodes."""


class EpisodePedal:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.reset()

    def reset(self):
        self.ready_ns = None
        self.released = None
        self.pressed_ns = None
        self.last_release_ns = None
        self.taps = 0
        self.invalid = False
        self.started = False

    def update(self, sample, trigger_ns):
        cfg = self.settings
        now = sample['monotonic_ns']
        pedal = sample.get('foot_pedal', {})
        cues = []
        if not cfg.get('enabled', True):
            return False, None, cues
        for event in pedal.get('events', []):
            timestamp = event['monotonic_ns']
            if timestamp <= trigger_ns:
                continue
            if event['pressed']:
                if self.released is False:
                    continue
                if self.pressed_ns is None:
                    # Settle a previous group before consuming a late press.
                    if self.last_release_ns is not None and timestamp - self.last_release_ns > cfg.get('tap_gap_seconds', 0.65) * 1e9:
                        break
                    self.pressed_ns = timestamp
                    if not self.started:
                        self.started = True
                        self.ready_ns = timestamp
                        cues.extend(('end', 'decision_start'))
            else:
                self.released = True
                if self.pressed_ns is not None:
                    duration = (timestamp - self.pressed_ns) / 1e9
                    self.pressed_ns = None
                    if cfg.get('minimum_tap_seconds', 0.04) <= duration <= cfg.get('maximum_tap_seconds', 0.5):
                        self.taps += 1
                        self.invalid |= self.taps > 3
                    else:
                        self.invalid = True
                    self.last_release_ns = timestamp
        if pedal.get('connected') and self.pressed_ns is None:
            self.released = not pedal.get('physical_pressed', pedal.get('downward_force_requested', False))
        label = None
        if (self.pressed_ns is None and self.last_release_ns is not None and
                now - self.last_release_ns >= cfg.get('tap_gap_seconds', 0.65) * 1e9):
            if not self.invalid and self.taps:
                label = ('kept', 'review', 'discarded')[self.taps - 1]
            else:
                cues.append('cancel')
            self.taps = 0
            self.invalid = False
            self.last_release_ns = None
        return True, label, cues
