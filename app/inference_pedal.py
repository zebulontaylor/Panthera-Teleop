"""Tap and hold inference gestures without owning physical input devices."""


class InferencePedal:
    def __init__(self, gap_seconds=.35, minimum_tap_seconds=.04,
                 maximum_tap_seconds=.5, review_hold_seconds=.6):
        self.gap_ns = int(gap_seconds * 1e9)
        self.minimum_ns = int(minimum_tap_seconds * 1e9)
        self.maximum_ns = int(maximum_tap_seconds * 1e9)
        self.review_hold_ns = int(review_hold_seconds * 1e9)
        self.reset()

    def _clear_taps(self):
        self.last_release_ns = None
        self.taps = 0
        self.invalid = False

    def reset(self):
        self.initialized = False
        self.pressed = False
        self.press_ns = None
        self._clear_taps()

    def update(self, snapshot):
        if not snapshot.get('connected'):
            self.reset()
            return None
        now = snapshot['sampled_monotonic_ns']
        if not self.initialized:
            self.initialized = True
            # Initial samples establish state only: queued input may predate ownership.
            # A pedal already held must be released before any gesture can begin.
            self.pressed = bool(snapshot.get('physical_pressed', snapshot.get('downward_force_requested')))
            return None
        review = False
        for event in sorted(snapshot.get('events', []), key=lambda item: item['monotonic_ns']):
            pressed = event['pressed']
            timestamp = event['monotonic_ns']
            if pressed == self.pressed:
                continue
            self.pressed = pressed
            if pressed:
                # Discard a sequence that could not be delivered before this press.
                if self.last_release_ns is not None and timestamp - self.last_release_ns > self.gap_ns:
                    self._clear_taps()
                self.press_ns = timestamp
            elif self.press_ns is not None:
                duration = timestamp - self.press_ns
                self.press_ns = None
                if duration >= self.review_hold_ns:
                    # Also recognize a hold entirely contained in a batch of edges.
                    self._clear_taps()
                    review = True
                else:
                    self.last_release_ns = timestamp
                    if self.minimum_ns <= duration <= self.maximum_ns:
                        self.taps += 1
                        self.invalid |= self.taps > 2
                    else:
                        self.invalid = True
        if (self.pressed and self.press_ns is not None and
                now - self.press_ns >= self.review_hold_ns):
            self._clear_taps()
            # Consuming the press suppresses repeat review and the release's tap.
            self.press_ns = None
            review = True
        if review:
            return 'review'
        if (not self.pressed and self.last_release_ns is not None and
                now - self.last_release_ns >= self.gap_ns):
            result = ('single' if self.taps == 1 else 'double' if self.taps == 2 else None)
            if self.invalid:
                result = None
            self._clear_taps()
            return result
        return None
