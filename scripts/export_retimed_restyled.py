#!/usr/bin/env python3
"""Pair pinned restyled JPEGs with already verified real retiming maps and numerics.

This never reruns a timing optimizer or image generator. Input data files are
verified against the pinned Hub LFS hashes before and after the export. Each
output episode is a standalone Arrow parquet; all statistics describe its actual
weighted frame distribution. Run with the repository's LeRobot environment.
"""
import argparse
import copy
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

REPO = 'ZebTaylor/panthera-gear-carrier-pin-latest200-restyled-20hz'
REVISION = '95c9576e09682582611636eebb683304e9eebf4b'
NUMERIC = ('observation.state', 'action', 'observation.velocity', 'observation.effort')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def numeric_stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Expected finite nonempty numeric data')
    result = dict(min=a.min(0), max=a.max(0), mean=a.mean(0), std=a.std(0), count=np.array([len(a)]))
    result.update({f'q{int(q*100):02d}': np.quantile(a, q, axis=0) for q in (.01, .1, .5, .9, .99)})
    return {k: v.tolist() for k, v in result.items()}


def image_stats(hist, frames):
    counts = hist.sum(1)
    levels = np.arange(256) / 255.
    mean = (hist * levels).sum(1) / counts
    var = (hist * (levels[None, :] - mean[:, None]) ** 2).sum(1) / counts
    result = dict(mean=mean, std=np.sqrt(var), min=np.argmax(hist > 0, axis=1)/255.,
                  max=(255-np.argmax(hist[:, ::-1] > 0, axis=1))/255.)
    result.update({f'q{int(q*100):02d}': np.array([np.searchsorted(np.cumsum(h), q*h.sum())/255. for h in hist])
                   for q in (.01, .1, .5, .9, .99)})
    return {**{k: v.reshape(3, 1, 1).tolist() for k, v in result.items()}, 'count': [frames]}


def weighted_image_hist(cells, selected):
    """Decode each selected source image once, weighting by output multiplicity."""
    hist = np.zeros((3, 256), dtype=np.int64)
    unique, weights = np.unique(selected, return_counts=True)
    for frame, weight in zip(unique, weights):
        payload = cells[int(frame)]['bytes']
        with Image.open(io.BytesIO(payload)) as im:
            if im.format != 'JPEG' or im.size != (256, 256):
                raise ValueError('Expected 256px embedded JPEG')
            a = np.asarray(im.convert('RGB'))
        for channel in range(3):
            hist[channel] += int(weight) * np.bincount(a[:, :, channel].ravel(), minlength=256)
    return hist


def validate_pair(report, mapping, numeric, source_rows, source_id):
    if report['source_id'] != source_id or report['source_frames'] != len(source_rows):
        raise ValueError('Source identity/frame count differs from real timing map')
    count = report['output_frames']
    if len(mapping) != count or numeric.num_rows != count:
        raise ValueError('Paired output frame counts differ')
    selected = np.asarray(mapping['source_image_frame'].to_pylist(), dtype=np.int64)
    sf = np.asarray(mapping['source_state_frame'].to_pylist())
    expected = np.minimum(np.floor(sf + .5).astype(np.int64), len(source_rows)-1)
    if not np.array_equal(selected, expected) or np.any(selected < 0):
        raise ValueError('Image rows differ from saved nearest-row timing map')
    for key in NUMERIC:
        if len(numeric[key]) != count:
            raise ValueError('Missing paired real numeric rows')
    return selected


def verify_sources(source, files):
    result = {}
    for record in files:
        path = source / record['path']
        actual = sha256(path)
        if actual != record['lfs']['oid'] or path.stat().st_size != record['size']:
            raise ValueError(f'Pinned source file differs: {path}')
        result[record['path']] = actual
    return result


def native_checks(out, repo_id, metadata):
    import torch
    import datasets
    import lerobot.datasets.lerobot_dataset as native_module
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(1)
    try:
        from lerobot.datasets.utils import get_hf_features_from_features, hf_transform_to_torch
    except ImportError:
        from lerobot.datasets.feature_utils import get_hf_features_from_features
        from lerobot.datasets.io_utils import hf_transform_to_torch

    def bounded_table(episode, features):
        table = pq.read_table(out / f'data/chunk-000/file-{episode:03d}.parquet')
        dataset = datasets.Dataset(table, info=datasets.DatasetInfo(features=get_hf_features_from_features(features)))
        dataset.set_transform(hf_transform_to_torch)
        return dataset

    class BoundedNativeDataset(LeRobotDataset):
        def load_hf_dataset(self):
            return bounded_table(self.episodes[0], self.features)

    # New LeRobot revisions delegate the identical native sampling logic to a
    # reader. Override only its Arrow table materialization in this process.
    original_reader = getattr(native_module, 'DatasetReader', None)
    if original_reader is not None:
        class BoundedReader(original_reader):
            def _load_hf_dataset(self):
                return bounded_table(self.episodes[0], self._meta.features)
        native_module.DatasetReader = BoundedReader
    checks = 0
    try:
        for ep in metadata:
            index = ep['episode_index']
            expected = pq.read_table(out / f'data/chunk-000/file-{index:03d}.parquet', columns=['action']).to_pydict()['action']
            native = BoundedNativeDataset(repo_id, root=out, episodes=[index], video_backend='pyav',
                delta_timestamps={'action': [i / 20 for i in range(50)]})
            assert len(native) == ep['length']
            for row in (0, ep['length']-1):
                loaded = native[row]
                assert tuple(loaded['action'].shape) == (50, 14)
                for camera in ('overhead', 'wrist_port2', 'wrist_port3'):
                    assert tuple(loaded[f'observation.images.{camera}'].shape) == (3, 256, 256)
                target = np.asarray([expected[min(row+i, len(expected)-1)] for i in range(50)], dtype=np.float32)
                np.testing.assert_array_equal(loaded['action'].numpy(), target)
                checks += 1
            del native
            if (index+1) % 25 == 0:
                print(f'Native50 verified {index+1}/{len(metadata)} episodes', flush=True)
    finally:
        if original_reader is not None:
            native_module.DatasetReader = original_reader
    return checks


def export(args):
    source, inputs, out = args.source.resolve(), args.inputs.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    if source == out or source in out.parents:
        raise ValueError('Output must be separate from source')
    contract = read_json(inputs / 'pinned_source.json')
    assert contract['repo_id'] == REPO and contract['revision'] == REVISION
    hashes = verify_sources(source, contract['files'])
    selection = read_json(inputs / 'selection.json')
    provenance = read_json(inputs / 'provenance.json')
    source_ids = {e['episode_index']: e['source_id'] for e in selection['selected_episodes']}
    source_info = read_json(inputs / 'info.json')
    assert source_info['total_frames'] == 124480 and source_info['total_episodes'] == 400
    assert len(provenance['episodes']) == 400
    assert read_json(source / 'selection.json') == selection
    actual_provenance = source / 'meta/restyled_provenance.json'
    if not actual_provenance.exists():
        actual_provenance = source / 'meta/provenance.json'
    assert read_json(actual_provenance) == provenance
    # Index only the 84 authenticated source files; the real tail of mixed600
    # remains outside the read set. Packed files may contain several episodes.
    files = {i: [] for i in range(400)}
    for record in contract['files']:
        path = source / record['path']
        for episode in np.unique(pq.read_table(path, columns=['episode_index'])['episode_index'].to_numpy()):
            assert int(episode) in files
            files[int(episode)].append(path)
    cameras = [k for k, v in source_info['features'].items() if v['dtype'] == 'image']
    all_numeric = {k: [] for k in source_info['features'] if k not in cameras}
    global_hist = {k: np.zeros((3, 256), dtype=np.int64) for k in cameras}
    out.mkdir(parents=True)
    metadata, reports, output_hashes = [], [], {}
    offset = 0
    source_rows_checked = 0
    image_checks = 0
    for episode, (name, style) in enumerate(provenance['episodes'].items()):
        original = style['source_episode']
        table = pa.concat_tables([pq.read_table(p, filters=[('episode_index', '=', episode)]) for p in files[episode]])
        table = table.take(pa.compute.sort_indices(table, sort_keys=[('frame_index', 'ascending')]))
        source_rows = table.to_pylist()
        assert len(source_rows) == style['frames']
        assert [r['frame_index'] for r in source_rows] == list(range(len(source_rows)))
        report_path = inputs / f'processing/retiming/episode_{original:06d}.json'
        map_path = report_path.with_suffix('.parquet')
        num_path = inputs / f'numeric/episode_{original:06d}.parquet'
        report, mapping, numeric = read_json(report_path), pq.read_table(map_path), pq.read_table(num_path)
        selected = validate_pair(report, mapping, numeric, source_rows, source_ids[original])
        n = len(selected)
        # Compare the two source restyles to the exact real motion (source numerics
        # transferred independently) before replacing the output motion.
        base = pq.read_table(inputs / f'original_numeric/episode_{original:06d}.parquet')
        for key in NUMERIC:
            np.testing.assert_array_equal(np.asarray(table[key].to_pylist()), np.asarray(base[key].to_pylist()))
        rows = [copy.copy(source_rows[int(i)]) for i in selected]
        numeric_values = numeric.to_pydict()
        for frame, row in enumerate(rows):
            for key in NUMERIC:
                row[key] = numeric_values[key][frame]
            row.update(timestamp=frame / 20, frame_index=frame, episode_index=episode, index=offset+frame)
        path = out / f'data/chunk-000/file-{episode:03d}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        result = pa.Table.from_pylist(rows, schema=table.schema)
        pq.write_table(result, path, compression='zstd', row_group_size=32)
        saved = pq.read_table(path)
        # Disk round trip proves all numeric coordinates and every encoded camera.
        for key in NUMERIC:
            np.testing.assert_array_equal(np.asarray(saved[key].to_pylist()), np.asarray(numeric_values[key]))
        for camera in cameras:
            expected = [source_rows[int(i)][camera] for i in selected]
            assert saved[camera].to_pylist() == expected
            image_checks += n
        ep = dict(episode_index=episode, tasks=['Assemble the gear, carrier and pin.'], length=n,
                  dataset_from_index=offset, dataset_to_index=offset+n,
                  **{'data/chunk_index': 0, 'data/file_index': episode,
                     'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0})
        # Use the actual task text from canonical metadata rather than guessing.
        tasks = pq.read_table(inputs / 'tasks.parquet').to_pandas().index.tolist()
        ep['tasks'] = tasks
        for key in all_numeric:
            values = saved[key].to_pylist()
            all_numeric[key].extend(values)
            ep.update({f'stats/{key}/{stat}': value for stat, value in numeric_stats(values).items()})
        for camera in cameras:
            hist = weighted_image_hist(table[camera].to_pylist(), selected)
            global_hist[camera] += hist
            ep.update({f'stats/{camera}/{stat}': value for stat, value in image_stats(hist, n).items()})
        metadata.append(ep)
        reports.append(dict(episode_index=episode, source_episode_index=original, source_id=source_ids[original],
            style_name=name, style_copy=int(name.rsplit('restyled', 1)[1]), source_frames=len(source_rows),
            output_frames=n, real_numeric_sha256=sha256(num_path), timing_map_sha256=sha256(map_path),
            timing_report_sha256=sha256(report_path), source_files=[str(p.relative_to(source)) for p in files[episode]]))
        output_hashes[str(path.relative_to(out))] = sha256(path)
        offset += n
        source_rows_checked += len(source_rows)
        print(f'{episode+1}/400 {name}: {len(source_rows)} -> {n}; rows={offset}', flush=True)
    assert source_rows_checked == 124480 and offset == 129892
    info = copy.deepcopy(source_info)
    info.update(total_frames=offset, total_episodes=400, splits={'train': '0:400'})
    write_json(out / 'meta/info.json', info)
    shutil.copy2(inputs / 'tasks.parquet', out / 'meta/tasks.parquet')
    ep_path = out / 'meta/episodes/chunk-000/file-000.parquet'
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(metadata), ep_path, compression='zstd')
    stats = {key: numeric_stats(values) for key, values in all_numeric.items()}
    stats.update({key: image_stats(hist, offset) for key, hist in global_hist.items()})
    write_json(out / 'meta/stats.json', stats)
    for name in ('selection.json', 'pinned_source.json'):
        shutil.copy2(inputs / name, out / name)
    shutil.copy2(inputs / 'provenance.json', out / 'meta/source_restyled_provenance.json')
    shutil.copytree(inputs / 'processing/retiming', out / 'processing/source_real_timing')
    shutil.copytree(inputs / 'numeric', out / 'processing/paired_real_numeric')
    write_json(out / 'processing/manifest.json', dict(schema='panthera-paired-retimed-restyled-v1',
        repo_id=args.repo_id, source_repo=REPO, source_revision=REVISION,
        real_source_revision=provenance['revision'], source_parquet_sha256=hashes,
        exact_source_jpeg_bytes=True, paired_real_numeric_exact=True,
        timing_policy='Reuse saved real227 source maps unchanged for both style copies',
        image_statistics='Exact decoded JPEG pixel histograms, weighted by repeated output row multiplicity',
        numeric_statistics='Float64 population moments and exact linear quantiles of all stored values',
        fps=20, episodes=reports, totals=dict(episodes=400, source_frames=124480, frames=offset, observations=offset)))
    checks = native_checks(out, args.repo_id, metadata)
    assert verify_sources(source, contract['files']) == hashes
    write_json(out / 'processing/verification.json', dict(passed=True, source_unchanged=True,
        pinned_source_files_checked=len(hashes), all_source_numeric_rows_checked=source_rows_checked,
        exact_paired_numeric_rows=offset, exact_encoded_image_cells=image_checks,
        native_50_step_first_last_checks=checks, native_threads=1,
        native_backend='Inherited LeRobot per-episode Arrow backend without full dataset cache',
        output_parquet_sha256=output_hashes, input_identity_sha256=sha256(inputs / 'pinned_source.json')))
    shutil.copy2(out / 'processing/verification.json', out / 'verification.json')
    reproduction = out / 'processing/reproduction'
    reproduction.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), reproduction / 'export_retimed_restyled.py')
    (out / 'COMPLETE').write_text(f'400 episodes, {offset} frames; verified\n')


def verify_existing(args):
    """Resume/repeat verification of a fully written export without rewriting it."""
    source, inputs, out = args.source.resolve(), args.inputs.resolve(), args.output.resolve()
    contract = read_json(inputs / 'pinned_source.json')
    assert contract['repo_id'] == REPO and contract['revision'] == REVISION
    hashes = verify_sources(source, contract['files'])
    manifest = read_json(out / 'processing/manifest.json')
    assert manifest['source_parquet_sha256'] == hashes and len(manifest['episodes']) == 400
    metadata = pq.read_table(out / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    output_hashes, image_checks, frames = {}, 0, 0
    for report in manifest['episodes']:
        ep, original = report['episode_index'], report['source_episode_index']
        paths = [source / p for p in report['source_files']]
        base = pa.concat_tables([pq.read_table(p, filters=[('episode_index', '=', ep)]) for p in paths])
        base = base.take(pa.compute.sort_indices(base, sort_keys=[('frame_index', 'ascending')]))
        mapping = pq.read_table(inputs / f'processing/retiming/episode_{original:06d}.parquet')
        path = out / f'data/chunk-000/file-{ep:03d}.parquet'
        saved = pq.read_table(path)
        numeric_path = inputs / f'numeric/episode_{original:06d}.parquet'
        assert sha256(numeric_path) == report['real_numeric_sha256']
        numeric = pq.read_table(numeric_path)
        assert saved.num_rows == report['output_frames']
        for key in NUMERIC:
            np.testing.assert_array_equal(np.asarray(saved[key].to_pylist()), np.asarray(numeric[key].to_pylist()))
        selected = mapping['source_image_frame'].to_pylist()
        for key in base.column_names:
            if key.startswith('observation.images.'):
                cells = base[key].to_pylist()
                assert saved[key].to_pylist() == [cells[i] for i in selected]
                image_checks += saved.num_rows
        assert saved['index'].to_pylist() == list(range(frames, frames+saved.num_rows))
        assert saved['frame_index'].to_pylist() == list(range(saved.num_rows))
        assert saved['episode_index'].to_pylist() == [ep]*saved.num_rows
        output_hashes[str(path.relative_to(out))] = sha256(path)
        frames += saved.num_rows
    assert frames == 129892 and image_checks == frames*3
    checks = native_checks(out, args.repo_id, metadata)
    assert verify_sources(source, contract['files']) == hashes
    manifest['repo_id'] = args.repo_id
    manifest['totals']['observations'] = frames
    write_json(out / 'processing/manifest.json', manifest)
    proof = dict(passed=True, source_unchanged=True, pinned_source_files_checked=len(hashes),
        all_source_numeric_rows_checked=124480, exact_paired_numeric_rows=frames,
        exact_encoded_image_cells=image_checks, native_50_step_first_last_checks=checks, native_threads=1,
        native_backend='Inherited LeRobot sampling with per-episode Arrow materialization',
        output_parquet_sha256=output_hashes, input_identity_sha256=sha256(inputs / 'pinned_source.json'))
    write_json(out / 'processing/verification.json', proof)
    write_json(out / 'verification.json', proof)
    reproduction = out / 'processing/reproduction'
    reproduction.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), reproduction / 'export_retimed_restyled.py')
    (out / 'COMPLETE').write_text(f'400 episodes, {frames} frames; verified\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo-id', default='ZebTaylor/panthera-gear-carrier-pin-retimed-restyled-20hz')
    p.add_argument('--verify-existing', action='store_true')
    args = p.parse_args()
    (verify_existing if args.verify_existing else export)(args)
