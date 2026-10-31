#!/usr/bin/env python3
"""Verify every exported pin-regrasp state, action and image against its sources."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from export_pin_regrasp_dataset import (AUXILIARY, LEFT, FPS, PERIOD_NS,
    read_json, sha256, left_images, donor_rows, write_json)


def verify(root):
    root = Path(root).resolve()
    report = read_json(root/'processing/export_report.json')
    info = read_json(root/'meta/info.json')
    source = Path(report['source_root'])
    assert info['fps'] == FPS and info['codebase_version'] == 'v3.0'
    assert info['total_episodes'] == len(report['episodes'])
    assert info['total_frames'] == report['totals']['frames']
    selected = {ep['source_id'] for ep in report['episodes']}
    discarded = source.parent/'pin-regrasp-discarded'
    if discarded.exists():
        assert not selected.intersection(p.name for p in discarded.iterdir() if p.is_dir())
    assert not selected.intersection(entry['source_id'] for entry in report['excluded'])
    manifest = report['donor_manifest']
    donors, hashes = donor_rows(Path(manifest['dataset_root']), manifest['candidates'])
    assert hashes == report['donor_data_files_sha256']
    data = ds.dataset(root/'data', format='parquet')
    checked = 0
    unique_donors, source_episodes = set(), set()
    max_camera_offset = max_state_offset = max_action_offset = 0.
    preview = []
    for ep in report['episodes']:
        index, identifier = ep['episode_index'], ep['source_id']
        raw_root = source/identifier
        for name, digest in ep['source_hashes'].items():
            assert sha256(raw_root/name) == digest, (identifier, name)
        metadata = read_json(raw_root/'metadata.json')
        assert metadata['complete'] and metadata['label'] == 'unverified'
        assert not metadata.get('discarded')
        raw = [json.loads(line) for line in (raw_root/'samples.jsonl').read_text().splitlines()]
        alignment = pq.read_table(root/ep['alignment']).to_pylist()
        table = data.to_table(filter=ds.field('episode_index') == index).sort_by('frame_index')
        rows = table.to_pylist()
        assert len(rows) == len(alignment) == ep['exported_frames']
        expected_images = left_images(raw_root/'left_wrist.avi', len(raw),
                                      {a['source_video_frame'] for a in alignment})
        for f, (row, a) in enumerate(zip(rows, alignment)):
            assert row['index'] == a['index'] == checked
            assert row['episode_index'] == a['episode_index'] == index
            assert row['frame_index'] == a['frame_index'] == f
            assert abs(row['timestamp']-f/FPS) < 1e-5
            assert row['task_index'] == 0
            assert a['source_episode_id'] == identifier
            assert a['source_phase'] in ('return', 'close')
            assert a['next_grid_monotonic_ns']-a['source_grid_monotonic_ns'] == PERIOD_NS
            original, following = raw[a['source_sample_index']], raw[a['next_state_sample_index']]
            assert following['monotonic_ns'] > original['monotonic_ns']
            np.testing.assert_array_equal(np.asarray(row['observation.state'], np.float32),
                                          np.asarray(original['state'], np.float32))
            np.testing.assert_array_equal(np.asarray(row['action'], np.float32),
                                          np.asarray(following['state'], np.float32))
            assert original['action'] == a['source_commanded_action']
            assert following['action'] == a['next_source_commanded_action']
            assert np.isfinite(row['observation.state']).all() and np.isfinite(row['action']).all()
            max_state_offset = max(max_state_offset, abs(a['measured_grid_offset_s']))
            max_action_offset = max(max_action_offset, abs(following['monotonic_ns']-a['next_grid_monotonic_ns'])/1e9)
            max_camera_offset = max(max_camera_offset, abs(a['camera_grid_offset_s']))
            assert abs(a['camera_grid_offset_s']) <= .06
            key = (a['donor_episode_index'], a['donor_frame_index'])
            unique_donors.add(key); source_episodes.add(key[0])
            assert a['auxiliary_camera_substitution'] and a['success'] == 'unverified'
            assert a['synthetic_camera_names'] == list(AUXILIARY)
            for camera in (*AUXILIARY, LEFT):
                payload = row[camera]['bytes']
                assert hashlib.sha256(payload).hexdigest() == a[camera+'.sha256']
                if camera == LEFT:
                    assert payload == expected_images[a['source_video_frame']]
                else:
                    assert payload == donors[key]['row'][camera]['bytes']
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    assert image.size == (256,256) and image.mode == 'RGB'
            if index in (0, len(report['episodes'])-1) and f in (0,len(rows)//2,len(rows)-1):
                preview.append((f'Episode {index}, frame {f}; auxiliary donor {key[0]}:{key[1]}',row))
            checked += 1
    assert checked == info['total_frames']
    assert max_state_offset < .018 and max_action_offset < .018
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    native = LeRobotDataset(report['repo_id'], root=root, video_backend='pyav',
                            delta_timestamps={'action':[i/FPS for i in range(10)]})
    assert len(native) == checked and native.num_episodes == len(report['episodes'])
    for i in (0, checked//2, checked-1):
        sample = native[i]
        assert tuple(sample['action'].shape) == (10,14)
        assert tuple(sample['observation.state'].shape) == (14,)
        for camera in (*AUXILIARY,LEFT):
            assert tuple(sample[camera].shape) == (3,256,256)
    canvas = Image.new('RGB',(768,len(preview)*284),'#161b19')
    draw = ImageDraw.Draw(canvas)
    for y,(label,row) in enumerate(preview):
        draw.text((6,y*284+2),label,fill='white')
        for x,camera in enumerate((*AUXILIARY,LEFT)):
            draw.text((x*256+6,y*284+14), 'REAL left wrist' if camera==LEFT else 'DONOR '+camera.split('.')[-1], fill='white')
            with Image.open(io.BytesIO(row[camera]['bytes'])) as image:
                canvas.paste(image,(x*256,y*284+28))
    canvas.save(root/'processing/preview.jpg',quality=95)
    result=dict(passed=True, episodes=len(report['episodes']), frames=checked, images=checked*3,
                exact_source_image_checks=checked*3, native_loader_checked=True,
                action_chunk_shape=[10,14], donor_episodes_used=len(source_episodes),
                distinct_donor_frames_used=len(unique_donors), max_state_offset_s=max_state_offset,
                max_action_offset_s=max_action_offset, max_left_camera_offset_s=max_camera_offset,
                discarded_sources_included=0, source_hashes_verified=True,
                stale_camera_grid_rows_trimmed=report['totals']['dropped_grid_rows'])
    write_json(root/'processing/verification.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    verify(parser.parse_args().dataset)
