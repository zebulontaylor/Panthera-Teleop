#!/usr/bin/env python3
"""Independently verify an alignment append; run in work/postprocess-venv.

Re-decodes every selected source camera unless --skip-image-redecode is given.
Never changes data or sources; writes only the requested verification report.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image
from export_pin_regrasp_dataset import left_images, read_json, sha256, write_json

CAMERAS = {'left_wrist': 'observation.images.wrist_port3',
           'right_wrist': 'observation.images.wrist_port2',
           'overhead': 'observation.images.overhead'}
TASKS = {'align the pin with the carrier hole': (192, 20, 172),
         'align the gear with the carrier': (86, 9, 77)}


def nearest(times, tick):
    # Independent implementation, stable first duplicate and earlier-time tie.
    return min(range(len(times)), key=lambda i: (abs(times[i] - tick), times[i], i))


def verify(args):
    root, base = args.dataset.resolve(), args.base.resolve()
    report = read_json(root/'processing/alignment_append_report.json')
    selection = read_json(args.selection)
    assert report['selection'] == selection
    assert report['selection_sha256'] == sha256(args.selection)
    selected = selection['episodes']
    paths = [str(Path(e['source_path']).resolve()) for e in selected]
    assert len(paths) == len(set(paths)) == 249
    excluded = {str(Path(e['source_path']).resolve()) for e in selection['excluded']}
    assert not set(paths) & excluded
    assert all('pickup2' not in p.lower() and 'discarded' not in p.lower() for p in paths)
    for task, (original, dropped, kept) in TASKS.items():
        chosen = [e for e in selected if e['task'] == task]
        tuning = [e for e in selection['excluded'] if e['task'] == task and e['reason'] == 'earliest_10_percent_tuning']
        assert len(chosen) == kept and len(tuning) == dropped == math.ceil(original*.1)
        start = lambda e: read_json(Path(e['source_path'])/'metadata.json')['started_unix_ns']
        assert max(map(start, tuning)) < min(map(start, chosen))
    assert set(e['task'] for e in selected) == set(TASKS)
    protected = {str(p.relative_to(base)): sha256(p) for directory in ('data', 'meta/episodes', 'processing')
                 for p in (base/directory).rglob('*') if p.is_file()}
    assert report['base_file_hashes'] == protected
    for name, digest in protected.items():
        assert sha256(root/name) == digest, ('changed base file', name)
    base_info, info = read_json(base/'meta/info.json'), read_json(root/'meta/info.json')
    assert info['fps'] == 20 and info['codebase_version'] == 'v3.0'
    assert info['features'] == base_info['features']
    assert info['total_episodes'] == base_info['total_episodes'] + len(selected)
    assert info['total_tasks'] == 3
    tasks = pq.read_table(root/'meta/tasks.parquet').to_pandas()['task_index'].to_dict()
    data = ds.dataset(root/'data', format='parquet')
    metadata = ds.dataset(root/'meta/episodes', format='parquet').to_table().to_pylist()
    assert len(metadata) == info['total_episodes']
    assert sorted(e['episode_index'] for e in metadata) == list(range(info['total_episodes']))
    assert sum(e['length'] for e in metadata) == info['total_frames']
    meta_by_index = {e['episode_index']: e for e in metadata}
    checked, images = base_info['total_frames'], 0
    for offset, (entry, ep) in enumerate(zip(selected, report['episodes'])):
        index = base_info['total_episodes'] + offset
        source = Path(entry['source_path'])
        assert ep['source_path'] == str(source) and ep['task'] == entry['task'] and ep['episode_index'] == index
        for name, digest in ep['source_hashes'].items():
            assert sha256(source/name) == digest, (source, name)
        raw = [json.loads(line) for line in (source/'samples.jsonl').read_text().splitlines() if line.strip()]
        sm = read_json(source/'metadata.json')
        assert sm['complete'] and sm['label'] == 'unverified' and not sm.get('error') and not sm.get('discarded')
        assert len(raw) == sm['samples']
        ticks = list(range(raw[0]['monotonic_ns'], raw[-1]['monotonic_ns']+1, 50_000_000))
        capture_times = {c: [r['cameras'][c]['monotonic_ns'] for r in raw] for c in CAMERAS}
        good = [t for t in ticks[:-1] if all(min(abs(v-t) for v in capture_times[c]) <= 60_000_000 for c in CAMERAS)]
        assert good and good == list(range(good[0], good[-1]+1, 50_000_000))
        alignment = pq.read_table(root/f'processing/alignment/episode_{index:06d}.parquet').to_pylist()
        rows = data.to_table(filter=ds.field('episode_index') == index).sort_by('frame_index').to_pylist()
        assert len(rows) == len(alignment) == ep['frames'] == len(good)
        em = meta_by_index[index]
        assert em['tasks'] == [entry['task']] and em['length'] == len(rows)
        assert em['dataset_from_index'] == checked and em['dataset_to_index'] == checked+len(rows)
        decoded = {} if args.skip_image_redecode else {c: left_images(source/f'{c}.avi', len(raw),
                   {nearest(capture_times[c], t) for t in good}) for c in CAMERAS}
        for frame, (row, a, tick) in enumerate(zip(rows, alignment, good)):
            current = nearest([r['monotonic_ns'] for r in raw], tick)
            following = nearest([r['monotonic_ns'] for r in raw], tick+50_000_000)
            assert row['index'] == a['index'] == checked
            assert row['frame_index'] == a['frame_index'] == frame
            assert row['episode_index'] == a['episode_index'] == index
            assert abs(row['timestamp']-frame/20) < 1e-5 and a['timestamp'] == frame/20
            assert row['task_index'] == tasks[entry['task']] and a['task'] == entry['task']
            assert a['source_grid_monotonic_ns'] == tick and a['next_grid_monotonic_ns'] == tick+50_000_000
            assert a['source_sample_index'] == current and a['next_state_sample_index'] == following
            assert a['success'] == 'unverified' and not a['auxiliary_camera_substitution']
            for key, source_row in [('observation.state', current), ('action', following)]:
                np.testing.assert_array_equal(np.asarray(row[key], np.float32), np.asarray(raw[source_row]['state'], np.float32))
                assert np.isfinite(row[key]).all()
            assert a['source_commanded_action'] == raw[current]['action']
            for camera, feature in CAMERAS.items():
                picked = nearest(capture_times[camera], tick)
                p = a[camera]
                assert p['source_sample_index'] == picked and p['source_video_frame'] == raw[picked]['video_frame']
                for key, value in raw[picked]['cameras'][camera].items():
                    assert p[key] == value
                assert abs(p['grid_offset_s']) <= .06
                assert p['grid_offset_s'] == (capture_times[camera][picked]-tick)/1e9
                payload = row[feature]['bytes']
                assert hashlib.sha256(payload).hexdigest() == p['image_sha256']
                if decoded:
                    assert payload == decoded[camera][picked], (index, frame, camera)
                with Image.open(io.BytesIO(payload)) as im:
                    im.load()
                    assert im.size == (256,256) and im.mode == 'RGB'
                images += 1
            checked += 1
        print(f'verified {offset+1}/{len(selected)} episode {index}', flush=True)
    assert len(report['episodes']) == len(selected)
    assert checked == info['total_frames'] == report['totals']['frames']
    # Recompute numeric stats directly from parquet, without exporter utilities.
    stats = read_json(root/'meta/stats.json')
    for key in info['features']:
        assert stats[key]['count'] == [checked], ('global stats count', key)
    for em in metadata:
        for key in info['features']:
            assert em[f'stats/{key}/count'] == [em['length']], ('episode stats count', em['episode_index'], key)
    for key, feature in info['features'].items():
        if feature['dtype'] in ('image', 'video', 'string'):
            continue
        values = np.asarray(data.to_table(columns=[key])[key].to_pylist(), dtype=np.float64)
        assert stats[key]['count'] == [checked]
        expected = [('min', values.min(axis=0)), ('max', values.max(axis=0)),
                    ('mean', values.mean(axis=0)), ('std', values.std(axis=0))]
        expected += [(f'q{int(q*100):02d}', np.quantile(values, q, axis=0)) for q in (.01,.10,.50,.90,.99)]
        for name, value in expected:
            np.testing.assert_allclose(np.asarray(stats[key][name]).reshape(-1), np.asarray(value).reshape(-1), rtol=1e-5, atol=1e-6)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    native = LeRobotDataset(report['repo_id'], root=root, video_backend='pyav',
                            delta_timestamps={'action': [i/20 for i in range(10)]})
    assert len(native) == checked and native.num_episodes == info['total_episodes']
    for i in (0, base_info['total_frames']-1, base_info['total_frames'], checked//2, checked-1):
        sample = native[i]
        assert tuple(sample['action'].shape) == (10,14)
        assert tuple(sample['observation.state'].shape) == (14,)
        for feature in CAMERAS.values():
            assert tuple(sample[feature].shape) == (3,256,256)
    result = dict(passed=True, total_frames=checked, total_episodes=info['total_episodes'],
                  appended_episodes=len(selected), appended_images_verified=images,
                  source_images_redecoded=not args.skip_image_redecode,
                  base_files_unchanged=len(protected), native_loader_checked=True,
                  numeric_stats_checked=True, earliest_ten_percent_checked=True)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path('data/processed/panthera-pin-regrasp-100-20hz'))
    parser.add_argument('--selection', type=Path, default=Path('work/alignment-upload-20261031/selection.json'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--skip-image-redecode', action='store_true')
    verify(parser.parse_args())
