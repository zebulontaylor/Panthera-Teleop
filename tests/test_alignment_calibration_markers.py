"""Live markers must not mistake partial/stale feedback for a reference pose."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('markers', Path(__file__).resolve().parents[1]/'scripts/mark_alignment_calibration.py')
markers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(markers)


class MarkerTests(unittest.TestCase):
    def test_partial_state_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'states.jsonl'
            path.write_bytes(b'{"monotonic_ns": 10}\n{"monotonic_ns": 20')
            self.assertEqual(markers.latest_sample(path)['monotonic_ns'], 10)

    def setup_recording(self, directory):
        root = Path(directory)
        episode = root/'pending'/'trial'
        episode.mkdir(parents=True)
        (episode/'metadata.json').write_text(json.dumps(dict(start_monotonic_ns=0, config={})))
        (episode/'states.jsonl').write_text(json.dumps(dict(monotonic_ns=2_000_000_000, arms={}))+'\n')
        state = dict(phase='recording', active='trial', cameras={name:dict(ok=True) for name in ('a','b','c')})
        return root, state

    def test_stale_feedback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, state = self.setup_recording(directory)
            with patch.object(markers, 'status', return_value=state), patch.object(markers.time, 'monotonic_ns', return_value=4_000_000_000):
                with self.assertRaisesRegex(ValueError, 'stale'):
                    markers.mark('', root, 'reference')

    def test_changed_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, state = self.setup_recording(directory)
            with patch.object(markers, 'status', side_effect=[state, dict(state, active='next')]), patch.object(markers.time, 'monotonic_ns', return_value=2_100_000_000):
                with self.assertRaisesRegex(ValueError, 'Episode changed'):
                    markers.mark('', root, 'reference')

    def test_marker_preserves_recorded_time_and_is_not_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root, state = self.setup_recording(directory)
            with patch.object(markers, 'status', return_value=state), patch.object(markers.time, 'monotonic_ns', return_value=2_100_000_000):
                row = markers.mark('', root, 'reference')
            self.assertEqual(row['time_s'], 2.)
            self.assertFalse(row['verified'])


if __name__ == '__main__':
    unittest.main()
