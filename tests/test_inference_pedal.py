"""Deterministic tap and hold inference gestures."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from inference_pedal import InferencePedal


class InferencePedalTests(unittest.TestCase):
    def setUp(self):
        self.pedal = InferencePedal()
        self.update(0)

    def update(self, seconds, events=(), pressed=False, connected=True):
        return self.pedal.update(dict(connected=connected,
            downward_force_requested=pressed, sampled_monotonic_ns=round(seconds * 1e9),
            events=[dict(monotonic_ns=round(t * 1e9), pressed=p) for t, p in events]))

    def test_single_waits_for_gap_and_only_emits_once(self):
        self.assertIsNone(self.update(.2, [(.1, True), (.2, False)]))
        self.assertIsNone(self.update(.549))
        self.assertEqual(self.update(.55), 'single')
        self.assertIsNone(self.update(1))

    def test_double_never_emits_single_and_waits_second_release_gap(self):
        self.update(.2, [(.1, True), (.2, False)])
        self.assertIsNone(self.update(.5, [(.4, True), (.5, False)]))
        self.assertIsNone(self.update(.849))
        self.assertEqual(self.update(.85), 'double')

    def test_batched_double_and_duplicate_edges(self):
        events = [(.1, True), (.1, True), (.2, False), (.2, False),
                  (.3, True), (.4, False)]
        self.assertEqual(self.update(.75, events), 'double')

    def test_three_taps_rejected(self):
        self.update(.6, [(.1, True), (.2, False), (.3, True), (.4, False),
                         (.5, True), (.6, False)])
        self.assertIsNone(self.update(.95))

    def test_bounce_and_long_holds_invalidate_whole_sequence(self):
        for release in (.11, .65):
            with self.subTest(release=release):
                self.setUp()
                self.update(release, [(.1, True), (release, False)])
                self.assertIsNone(self.update(release + .35))
                self.update(release + .6, [(release + .5, True), (release + .6, False)])
                self.assertEqual(self.update(release + 1), 'single')

    def test_startup_held_release_is_ignored(self):
        self.pedal.reset()
        self.update(0, pressed=True)
        self.update(.1, [(.1, False)])
        self.assertIsNone(self.update(.5))
        self.update(.7, [(.6, True), (.7, False)])
        self.assertEqual(self.update(1.05), 'single')

    def test_disconnect_and_reset_clear_pending_gestures(self):
        for disconnect in (False, True):
            self.setUp()
            self.update(.2, [(.1, True), (.2, False)])
            if disconnect:
                self.update(.3, connected=False)
            else:
                self.pedal.reset()
            self.assertIsNone(self.update(1))
            self.assertIsNone(self.update(2))

    def test_new_press_after_expired_gap_discards_stale_sequence(self):
        self.update(.2, [(.1, True), (.2, False)])
        self.assertIsNone(self.update(.8, [(.7, True), (.8, False)]))
        self.assertEqual(self.update(1.15), 'single')

    def test_duration_boundaries_and_held_second_tap(self):
        self.update(.14, [(.1, True), (.14, False)])
        self.update(.4, [(.3, True)], pressed=True)
        self.assertIsNone(self.update(.7, pressed=True))
        self.update(.8, [(.8, False)])
        self.assertEqual(self.update(1.15), 'double')

    def test_review_fires_at_point_six_seconds_once_without_release_tap(self):
        self.assertIsNone(self.update(.1, [(.1, True)], pressed=True))
        self.assertIsNone(self.update(.699999999, pressed=True))
        self.assertEqual(self.update(.7, pressed=True), 'review')
        self.assertIsNone(self.update(2, pressed=True))
        self.assertIsNone(self.update(2.1, [(2.1, False)]))
        self.assertIsNone(self.update(3))

    def test_review_recognizes_hold_released_in_same_batch(self):
        for release in (.7, 1.5):
            with self.subTest(release=release):
                self.setUp()
                self.assertEqual(self.update(release, [(.1, True), (release, False)]), 'review')
                self.assertIsNone(self.update(release + 1))

    def test_review_supersedes_pending_taps(self):
        self.update(.2, [(.1, True), (.2, False)])
        self.update(.3, [(.3, True)], pressed=True)
        self.assertIsNone(self.update(.8, pressed=True))
        self.assertEqual(self.update(.9, pressed=True), 'review')
        self.assertIsNone(self.update(1.4, [(1.4, False)]))
        self.assertIsNone(self.update(2))

    def test_hold_between_tap_and_review_threshold_is_invalid(self):
        for duration in (.500000001, .55, .599999999):
            with self.subTest(duration=duration):
                self.setUp()
                release = .1 + duration
                self.assertIsNone(self.update(release, [(.1, True), (release, False)]))
                self.assertIsNone(self.update(release + 1))

    def test_long_hold_then_double_works(self):
        self.update(.1, [(.1, True)], pressed=True)
        self.assertEqual(self.update(.7, pressed=True), 'review')
        self.update(1.2, [(1.2, False)])
        self.update(1.6, [(1.3, True), (1.4, False), (1.5, True), (1.6, False)])
        self.assertEqual(self.update(1.95), 'double')

    def test_startup_held_without_edges_never_reviews(self):
        self.pedal.reset()
        self.update(0, pressed=True)
        self.assertIsNone(self.update(5, pressed=True))
        self.assertIsNone(self.update(6, [(6, False)]))
        self.assertIsNone(self.update(7))
        self.update(7.1, [(7.1, True)], pressed=True)
        self.assertEqual(self.update(7.7, pressed=True), 'review')

    def test_reset_while_held_requires_release_and_new_press(self):
        self.update(.1, [(.1, True)], pressed=True)
        self.pedal.reset()
        self.update(.5, pressed=True)
        self.assertIsNone(self.update(2, pressed=True))
        self.update(2.1, [(2.1, False)])
        self.assertIsNone(self.update(3))
        self.update(3.1, [(3.1, True)], pressed=True)
        self.assertEqual(self.update(3.7, pressed=True), 'review')

    def test_idle_without_events_does_not_review(self):
        self.assertIsNone(self.update(10))
        self.assertIsNone(self.update(100))


if __name__ == '__main__':
    unittest.main()
