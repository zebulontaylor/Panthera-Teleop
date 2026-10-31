"""Pedal-only episode decisions without motor or audio devices."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from episode_pedal import EpisodePedal


class EpisodePedalTests(unittest.TestCase):
    def setUp(self):
        self.detector = EpisodePedal()
        self.trigger = 10_000_000_000
        self.now = self.trigger

    def tick(self, pressed=False, dt=.1, events=(), arms=None):
        self.now += round(dt * 1e9)
        sample = dict(monotonic_ns=self.now,
                      foot_pedal=dict(connected=True, physical_pressed=pressed,
                                      events=list(events)))
        if arms is not None:
            sample['arms'] = arms
        return self.detector.update(sample, self.trigger)

    def tap(self, duration=.1):
        press = self.now + 10_000_000
        release = press + round(duration * 1e9)
        return self.tick(dt=duration + .02, events=[
            dict(monotonic_ns=press, pressed=True),
            dict(monotonic_ns=release, pressed=False)])

    def test_first_tap_ends_and_classifies_without_arm_data_or_minimum_duration(self):
        for count, label in [(1, 'kept'), (2, 'review'), (3, 'discarded')]:
            with self.subTest(count=count):
                self.setUp()
                self.detector.settings['minimum_episode_seconds'] = 100
                self.assertEqual(self.tap(), (True, None, ['end', 'decision_start']))
                self.assertLess(self.now - self.trigger, 1_000_000_000)
                for _ in range(count - 1):
                    self.assertEqual(self.tap(), (True, None, []))
                self.assertIsNone(self.tick(dt=.5)[1])
                self.assertEqual(self.tick(dt=.2), (True, label, []))
                self.assertEqual(self.tick(dt=1), (True, None, []))

    def test_gripper_movement_cannot_end_episode(self):
        for position in [0] * 10 + [1.7] * 10 + [0, 1.7] * 5:
            arms = {'right': {'gripper': {'position': position}}}
            self.assertEqual(self.tick(arms=arms), (True, None, []))
        self.assertFalse(self.detector.started)
        self.assertIsNone(self.detector.ready_ns)

    def test_initially_held_pedal_requires_release_and_fresh_press(self):
        self.assertEqual(self.tick(pressed=True), (True, None, []))
        self.assertEqual(self.tick(pressed=True, events=[
            dict(monotonic_ns=self.now + 1, pressed=True)]), (True, None, []))
        self.assertEqual(self.tick(events=[
            dict(monotonic_ns=self.now + 1, pressed=False)]), (True, None, []))
        self.assertEqual(self.detector.taps, 0)
        self.assertFalse(self.detector.started)
        self.assertEqual(self.tap()[2], ['end', 'decision_start'])
        self.assertEqual(self.tick(dt=.7)[1], 'kept')

    def test_events_at_or_before_trigger_are_ignored(self):
        events = [dict(monotonic_ns=self.trigger - 100_000_000, pressed=True),
                  dict(monotonic_ns=self.trigger, pressed=False)]
        self.assertEqual(self.tick(events=events), (True, None, []))
        self.assertFalse(self.detector.started)
        self.assertEqual(self.detector.taps, 0)
        self.assertIsNone(self.tick(dt=1)[1])
        self.assertEqual(self.tap()[2], ['end', 'decision_start'])
        self.assertEqual(self.tick(dt=.7)[1], 'kept')

    def test_classification_waits_for_release(self):
        press = self.now + 1
        self.assertEqual(self.tick(pressed=True, events=[
            dict(monotonic_ns=press, pressed=True)])[2], ['end', 'decision_start'])
        self.assertIsNone(self.tick(pressed=True, dt=.3)[1])
        self.assertIsNone(self.tick(events=[
            dict(monotonic_ns=self.now + 1, pressed=False)])[1])
        self.assertEqual(self.tick(dt=.7)[1], 'kept')

    def test_bounce_hold_and_overflow_cancel_and_allow_fresh_gesture(self):
        for durations in [[.01], [.8], [.1] * 4]:
            with self.subTest(durations=durations):
                self.setUp()
                for duration in durations:
                    self.tap(duration)
                self.assertEqual(self.tick(dt=.7), (True, None, ['cancel']))
                self.tap()
                self.assertEqual(self.tick(dt=.7)[1], 'kept')

    def test_reset_clears_previous_episode_gesture(self):
        self.tap()
        self.detector.reset()
        self.assertEqual(self.tick(dt=1), (True, None, []))
        self.assertFalse(self.detector.started)
        self.assertEqual(self.detector.taps, 0)
        self.assertIsNone(self.detector.ready_ns)
        self.assertEqual(self.tap()[2], ['end', 'decision_start'])
        self.assertEqual(self.tick(dt=.7)[1], 'kept')

    def test_disabled_mode_ignores_pedal(self):
        self.detector.settings['enabled'] = False
        self.assertEqual(self.tap(), (False, None, []))
        self.assertEqual(self.tick(dt=1), (False, None, []))
        self.assertFalse(self.detector.started)
        self.assertEqual(self.detector.taps, 0)
