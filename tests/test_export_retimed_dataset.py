"""Regression checks for the export's shared-clock image alignment contract."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from export_retimed_dataset import camera_alignment_extra_bound, exact_numeric_statistics


class CameraAlignmentBoundTests(unittest.TestCase):
    def test_nearest_observation_uses_half_source_tick(self):
        self.assertEqual(camera_alignment_extra_bound(2.4, 6, 20), .025)
        self.assertEqual(camera_alignment_extra_bound(5.49, 6, 20), .025)

    def test_fractional_terminal_observation_clamps_to_last_image(self):
        # Six observations have images at 0..5; the terminal action knot is 6.
        # Retiming can emit a state at 5.8. Its image must stay at 5, adding 40ms
        # source-clock lag, which is legal but exceeds the ordinary 25ms bound.
        extra = (5.8 - 5) / 20
        bound = camera_alignment_extra_bound(5.8, 6, 20)
        self.assertGreater(extra, .5 / 20)
        self.assertLessEqual(extra, bound)
        self.assertEqual(bound, 1 / 20)
        self.assertEqual(camera_alignment_extra_bound(5.5, 6, 20), 1 / 20)

    def test_out_of_source_clock_rejected(self):
        with self.assertRaises(ValueError):
            camera_alignment_extra_bound(6.01, 6, 20)


class NumericStatisticsPrecisionTests(unittest.TestCase):
    def test_low_variance_float32_effort_keeps_variance(self):
        values = np.array([1000. + i * .001 for i in range(1000)], dtype=np.float32)[:, None]
        expected = values.astype(np.float64).std(axis=0)
        native_cancellation = np.maximum(0, (values ** 2).mean(axis=0) - values.mean(axis=0) ** 2) ** .5
        self.assertGreater(abs(float(native_cancellation[0] - expected[0])), .01)
        stats = exact_numeric_statistics(values)
        np.testing.assert_array_equal(stats['std'], expected)
        np.testing.assert_array_equal(stats['mean'], values.astype(np.float64).mean(axis=0))
        np.testing.assert_array_equal(stats['q50'], np.quantile(values.astype(np.float64), .5, axis=0))
        self.assertEqual(stats['count'], [1000])

    def test_scalar_index_stats_keep_native_shapes_and_integer_extrema(self):
        stats = exact_numeric_statistics(np.arange(10, 16, dtype=np.int64))
        self.assertEqual(stats['min'], [10])
        self.assertIsInstance(stats['min'][0], int)
        self.assertEqual(stats['max'], [15])
        self.assertEqual(stats['mean'], [12.5])
        self.assertEqual(stats['count'], [6])


if __name__ == '__main__':
    unittest.main()
