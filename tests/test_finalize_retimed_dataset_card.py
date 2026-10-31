"""Viewer patterns select training data while leaving provenance accessible."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from finalize_retimed_dataset_card import viewer_card


class DatasetCardTests(unittest.TestCase):
    def test_inserts_explicit_training_pattern_and_preserves_body(self):
        text='---\ntags:\n- lerobot\n---\n# Title\nBody\n'
        result=viewer_card(text)
        self.assertIn('path: data/chunk-*/*.parquet',result)
        self.assertTrue(result.endswith('# Title\nBody\n'))
        self.assertEqual(viewer_card(result),result)

    def test_requires_frontmatter(self):
        with self.assertRaises(ValueError):viewer_card('# Missing header\n')


if __name__=='__main__':unittest.main()
