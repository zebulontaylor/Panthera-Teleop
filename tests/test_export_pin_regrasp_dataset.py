"""Source selection, timing, AVI, and paired donor provenance without robot access."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_pin_regrasp_dataset import (regular_grid, nearest_rows, paired_donor_choices,
    retained_episodes, validate_samples, left_images, donor_rows, camera_aligned_grid)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_grid_uses_nearest_measured_next_state_and_drops_terminal(self):
        base = 9_000_000_000
        times = np.array([base, base+33_000_000, base+67_000_000, base+101_000_000])
        ticks, indices = regular_grid(times)
        self.assertEqual(ticks.tolist(), [base, base+50_000_000, base+100_000_000])
        self.assertEqual(indices.tolist(), [0, 1, 3])  # midpoint tie uses earlier row
        states = np.arange(4*14).reshape(4, 14)
        observations, actions = states[indices[:-1]], states[indices[1:]]
        self.assertEqual(len(observations), 2)
        np.testing.assert_array_equal(actions[0], observations[1])
        np.testing.assert_array_equal(actions[-1], states[-1])

    def test_camera_grid_is_independent_with_duplicate_capture_ties(self):
        captures = np.array([0, 0, 40, 80]) * 1_000_000
        ticks = np.array([0, 50, 60]) * 1_000_000
        self.assertEqual(nearest_rows(captures, ticks).tolist(), [0, 2, 2])
        self.assertEqual(nearest_rows([0, 40_000_000, 80_000_000, 120_000_000], ticks).tolist(), [0, 1, 1])
        for times in ([1, 1], [3, 2], [0], [0, 10]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                regular_grid(times)

    def test_stale_camera_tail_trims_grid_but_preserves_next_measured_action(self):
        times = np.arange(11, dtype=np.int64) * 50_000_000
        captures = np.minimum(times, 250_000_000)
        ticks, states, cameras, trim = camera_aligned_grid(times, captures)
        self.assertEqual(ticks.tolist(), times[:8].tolist())
        self.assertEqual(states.tolist(), list(range(8)))
        self.assertEqual(len(cameras), 7)
        self.assertEqual(trim['original_grid_frames'], 10)
        self.assertEqual(trim['trailing_grid_rows'], 3)
        self.assertEqual(trim['dropped_grid_rows'], 3)
        self.assertEqual(trim['leading_grid_rows'], 0)
        # Last retained observation is 300ms, with its next measured action at 350ms.
        self.assertEqual(ticks[-2], 300_000_000)
        self.assertEqual(ticks[-1], 350_000_000)
        with self.assertRaisesRegex(ValueError, 'Internal camera'):
            camera_aligned_grid(times, [0, 0, 0, 0, 0, 0, 0, 400_000_000, 450_000_000,
                                        500_000_000, 500_000_000])

    def test_only_complete_unverified_immediate_retained_episodes(self):
        entries = {'valid': dict(label='unverified', complete=True),
                   'partial': dict(label='unverified', complete=False),
                   'bad': dict(label='unverified', complete=True, error='camera'),
                   'discarded_label': dict(label='discarded', complete=True),
                   'discard_flag': dict(label='unverified', complete=True, discarded=True)}
        for name, metadata in entries.items():
            path = self.root / name
            path.mkdir()
            (path / 'metadata.json').write_text(json.dumps(metadata))
        discarded = self.root / 'discarded'
        discarded.mkdir()
        (discarded / 'metadata.json').write_text(json.dumps(entries['valid']))
        selected, excluded = retained_episodes(self.root)
        self.assertEqual([path.name for path, _ in selected], ['valid'])
        self.assertEqual(len(excluded), 5)
        with self.assertRaises(ValueError):
            retained_episodes(discarded)

    def test_alignment_excluded_and_legacy_regrasp_still_retained(self):
        for name, task in [('alignment', 'alignment'), ('other', 'other'),
                           ('regrasp', 'regrasp'), ('pin', 'pin'), ('legacy', None)]:
            path = self.root / name
            path.mkdir()
            metadata = dict(label='unverified', complete=True)
            if task is not None:
                metadata['task'] = task
            (path / 'metadata.json').write_text(json.dumps(metadata))
        selected, excluded = retained_episodes(self.root)
        self.assertEqual([path.name for path, _ in selected], ['legacy', 'pin', 'regrasp'])
        self.assertEqual(excluded, [dict(source_id=name, reason='not_regrasp_task')
                                    for name in ['alignment', 'other']])

    def test_donor_draw_is_repeatable_and_always_one_paired_row(self):
        candidates = [dict(episode_index=episode, frame_index=frame)
                      for episode in (2, 4) for frame in range(3)]
        first = paired_donor_choices(candidates, 30, 31, 0)
        self.assertEqual(first, paired_donor_choices(candidates, 30, 31, 0))
        self.assertNotEqual(first, paired_donor_choices(candidates, 30, 31, 1))
        self.assertTrue(all(candidate in candidates for candidate in first))
        with self.assertRaises(ValueError):
            paired_donor_choices([dict(episode_index=1, frame_index=-1)], 1, 31, 0)

    def test_source_samples_preserve_raw_gripper_and_validate_forward_clock(self):
        rows = []
        for i in range(4):
            timestamp = 1_000_000_000 + i*34_000_000
            rows.append(dict(index=i, video_frame=i, phase='return', monotonic_ns=timestamp,
                unix_ns=timestamp+1_000_000, state=[2.002]*14, action=[.4]*14,
                camera=dict(camera='left_wrist', sequence=i, monotonic_ns=timestamp-4_000_000,
                            unix_ns=timestamp+996_000, width=640, height=480)))
        path = self.root / 'samples.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        validated, ticks, indices = validate_samples(self.root, {'samples': 4})
        self.assertEqual(validated[0]['state'][6], 2.002)
        rows[2]['phase'] = 'human'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaises(ValueError):
            validate_samples(self.root, {'samples': 4})

    def test_actual_avi_count_and_crop_dimensions(self):
        import av
        from PIL import Image
        video = self.root / 'left_wrist.avi'
        with av.open(str(video), 'w') as container:
            stream = container.add_stream('mjpeg', rate=30)
            stream.width, stream.height, stream.pix_fmt = 640, 480, 'yuvj420p'
            for i in range(4):
                array = np.zeros((480, 640, 3), dtype=np.uint8)
                array[:, 80:560] = [180+i*5, 60, 20]
                frame = av.VideoFrame.from_ndarray(array, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        images = left_images(video, 4, {0, 2})
        self.assertEqual(set(images), {0, 2})
        with Image.open(io.BytesIO(images[0])) as image:
            self.assertEqual(image.size, (256, 256))
            self.assertEqual(image.format, 'JPEG')
            self.assertGreater(np.asarray(image)[128, 128, 0], 150)
        with self.assertRaisesRegex(ValueError, 'expected 5'):
            left_images(video, 5, {0})

    def test_donor_loader_preserves_exact_paired_bytes_and_hashes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        from export_pin_regrasp_dataset import AUXILIARY, sha256
        (self.root / 'data').mkdir()
        values = []
        for i in range(2):
            stream = io.BytesIO()
            Image.new('RGB', (256, 256), (20+i*60, 20, 30)).save(stream, 'JPEG')
            values.append(stream.getvalue())
        source = self.root / 'data/chunk.parquet'
        pq.write_table(pa.Table.from_pylist([dict(episode_index=3, frame_index=4,
            **{AUXILIARY[0]: {'bytes': values[0], 'path': None},
               AUXILIARY[1]: {'bytes': values[1], 'path': None}})]), source)
        loaded, hashes = donor_rows(self.root, [dict(episode_index=3, frame_index=4)])
        pair = loaded[(3, 4)]['row']
        self.assertEqual(pair[AUXILIARY[0]]['bytes'], values[0])
        self.assertEqual(pair[AUXILIARY[1]]['bytes'], values[1])
        self.assertEqual(hashes['data/chunk.parquet'], sha256(source))
        with self.assertRaisesRegex(ValueError, 'Missing donor'):
            donor_rows(self.root, [dict(episode_index=9, frame_index=9)])


if __name__ == '__main__':
    unittest.main()
