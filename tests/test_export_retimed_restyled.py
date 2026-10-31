"""Focused paired mapping, multiplicity, and numerical stability tests."""
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
import pyarrow as pa
from PIL import Image

spec = importlib.util.spec_from_file_location('paired', Path(__file__).parents[1]/'scripts/export_retimed_restyled.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PairedExportTests(unittest.TestCase):
    def test_weighted_hist_matches_repeated_decodes(self):
        cells = []
        for color in ((0, 10, 20), (200, 210, 220)):
            stream = io.BytesIO()
            Image.new('RGB', (256, 256), color).save(stream, format='JPEG')
            cells.append({'bytes': stream.getvalue(), 'path': None})
        selected = np.array([0, 1, 1, 1])
        hist = m.weighted_image_hist(cells, selected)
        self.assertTrue(np.all(hist.sum(1) == 4*256*256))
        pixels = np.stack([np.asarray(Image.open(io.BytesIO(cells[i]['bytes']))) for i in selected])
        np.testing.assert_allclose(np.array(m.image_stats(hist, 4)['mean']).ravel(), pixels.mean((0, 1, 2))/255)

    def test_float64_centered_small_variance(self):
        values = np.array([[10000., 2.], [10000.001, 4.]], dtype=np.float32)
        actual = m.numeric_stats(values)
        np.testing.assert_array_equal(actual['std'], values.astype(np.float64).std(0))
        self.assertGreater(actual['std'][0], 0)

    def test_pair_requires_exact_saved_image_map_and_identity(self):
        report = dict(source_id='source-a', source_frames=2, output_frames=3)
        mapping = pa.table({'source_image_frame': [0, 1, 1], 'source_state_frame': [0., .6, 1.9]})
        numeric = pa.table({key: [[0.]*14]*3 for key in m.NUMERIC})
        np.testing.assert_array_equal(m.validate_pair(report, mapping, numeric, [{}, {}], 'source-a'), [0, 1, 1])
        with self.assertRaises(ValueError):
            m.validate_pair(report, mapping, numeric, [{}, {}], 'wrong-id')
        with self.assertRaises(ValueError):
            m.validate_pair(report, pa.table({'source_image_frame': [0, 0, 1], 'source_state_frame': [0., .6, 1.9]}), numeric, [{}, {}], 'source-a')


if __name__ == '__main__':
    unittest.main()
