#!/usr/bin/env python3
"""Non-destructive pin-forward export. Run with work/postprocess-venv/bin/python."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

FPS = 20
PERIOD_NS = 50_000_000
TASK = 'grasp the pin'
AUXILIARY = ('observation.images.overhead', 'observation.images.wrist_port2')
LEFT = 'observation.images.wrist_port3'
NAMES = [name for side in ('left', 'right') for name in
         [*(f'{side}.joint{i}' for i in range(1, 7)), f'{side}.gripper_rad']]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def retained_episodes(source):
    """Only immediate retained episode directories; never traverse discarded trees."""
    source = Path(source).resolve()
    if 'discarded' in source.parts:
        raise ValueError('Discarded recordings cannot be exported')
    selected, excluded = [], []
    for path in sorted(source.iterdir()):
        if not path.is_dir() or path.name.startswith('.'):
            continue
        if path.is_symlink() or path.name == 'discarded':
            excluded.append(dict(source_id=path.name, reason='discarded_or_symlink'))
            continue
        metadata_path = path / 'metadata.json'
        if not metadata_path.is_file():
            continue
        metadata = read_json(metadata_path)
        if (metadata.get('label') != 'unverified' or metadata.get('complete') is not True
                or metadata.get('discarded') or metadata.get('error')):
            excluded.append(dict(source_id=path.name, reason='not_complete_unverified_retained'))
            continue
        selected.append((path, metadata))
    return selected, excluded


def nearest_rows(times, ticks):
    times = np.asarray(times, dtype=np.int64)
    if times.ndim != 1 or not len(times) or np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError('Nearest-row timestamps must be nonnegative and nondecreasing')
    # First row wins duplicate capture timestamps; equal-distance ties choose earlier time.
    unique, first = np.unique(times, return_index=True)
    right = np.minimum(np.searchsorted(unique, ticks), len(unique) - 1)
    left = np.maximum(right - 1, 0)
    return first[np.where(ticks - unique[left] <= unique[right] - ticks, left, right)]


def regular_grid(times):
    """Return host-time ticks and nearest source rows, including terminal state knot."""
    times = np.asarray(times, dtype=np.int64)
    if times.ndim != 1 or len(times) < 2 or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError('At least two strictly increasing nonnegative host timestamps required')
    ticks = times[0] + np.arange((int(times[-1]) - int(times[0])) // PERIOD_NS + 1,
                               dtype=np.int64) * PERIOD_NS
    if len(ticks) < 2:
        raise ValueError('Episode is shorter than one 20 Hz action interval')
    return ticks, nearest_rows(times, ticks)


def camera_aligned_grid(times, captures, max_alignment_s=.06):
    """Trim only terminal unavailable camera grids; never join an internal gap."""
    ticks, indices = regular_grid(times)
    camera_indices = nearest_rows(captures, ticks[:-1])
    captures = np.asarray(captures, dtype=np.int64)
    valid = np.abs(captures[camera_indices] - ticks[:-1]) <= round(max_alignment_s * 1e9)
    kept = np.flatnonzero(valid)
    if not len(kept):
        raise ValueError('No camera-aligned 20Hz observations')
    first, last = int(kept[0]), int(kept[-1])
    if not valid[first:last+1].all():
        raise ValueError('Internal camera alignment gap cannot be joined')
    trimming = dict(original_grid_frames=len(ticks)-1, leading_grid_rows=first,
        trailing_grid_rows=len(valid)-last-1, dropped_grid_rows=int((~valid).sum()),
        max_camera_alignment_s=max_alignment_s,
        reason='unavailable_camera_tail_or_head_removed' if not valid.all() else None)
    return ticks[first:last+2], indices[first:last+2], camera_indices[first:last+1], trimming


def paired_donor_choices(candidates, count, seed, episode_index):
    if not candidates:
        raise ValueError('Donor manifest needs candidates')
    for candidate in candidates:
        if any(type(candidate.get(k)) is not int or candidate[k] < 0
               for k in ('episode_index', 'frame_index')):
            raise ValueError('Donor episode/frame indices must be nonnegative integers')
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode_index]))
    # One draw selects both auxiliary cameras from the very same recorded donor row.
    return [candidates[int(index)] for index in rng.integers(0, len(candidates), size=count)]


def validate_samples(path, metadata):
    rows = [json.loads(line) for line in (path / 'samples.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != metadata.get('samples'):
        raise ValueError(f'{path.name}: metadata sample count differs')
    last_sequence = last_camera_ns = -1
    for index, row in enumerate(rows):
        camera = row.get('camera', {})
        if (row.get('index') != index or row.get('video_frame') != index
                or row.get('phase') not in ('return', 'close') or camera.get('camera') != 'left_wrist'):
            raise ValueError(f'{path.name}: invalid forward sample/video mapping')
        for field in ('state', 'action'):
            vector = np.asarray(row.get(field), dtype=float)
            if vector.shape != (14,) or not np.isfinite(vector).all():
                raise ValueError(f'{path.name}: invalid {field}')
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ('monotonic_ns', 'unix_ns')):
            raise ValueError('Missing host sample timestamps')
        if any(type(camera.get(k)) is not int or camera[k] < 0
               for k in ('sequence', 'monotonic_ns', 'unix_ns')):
            raise ValueError('Missing camera sequence/capture timestamps')
        if (camera['sequence'] < last_sequence or camera['monotonic_ns'] < last_camera_ns
                or camera['monotonic_ns'] > row['monotonic_ns']
                or (camera.get('width'), camera.get('height')) != (640, 480)):
            raise ValueError('Invalid left wrist camera clock or dimensions')
        last_sequence, last_camera_ns = camera['sequence'], camera['monotonic_ns']
    ticks, indices = regular_grid([row['monotonic_ns'] for row in rows])
    return rows, ticks, indices


def left_images(video, count, needed):
    """Decode the entire AVI to verify actual count, encoding selected crops once."""
    import av
    from PIL import Image
    result, decoded = {}, 0
    with av.open(str(video)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Expected exactly one left wrist video stream')
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (640, 480):
                raise ValueError('Source wrist video must be 640x480')
            if decoded in needed:
                image = frame.to_image().convert('RGB')
                image = image.crop((80, 0, 560, 480)).resize((256, 256), Image.Resampling.LANCZOS)
                stream = io.BytesIO()
                image.save(stream, format='JPEG', quality=95, subsampling=0)
                result[decoded] = stream.getvalue()
            decoded += 1
    if decoded != count or set(result) != set(needed):
        raise ValueError(f'AVI decoded {decoded} frames, expected {count}: {video}')
    return result


def donor_rows(root, candidates):
    import pyarrow.parquet as pq
    from PIL import Image
    wanted = {(row['episode_index'], row['frame_index']) for row in candidates}
    found, hashes = {}, {}
    for path in sorted((root / 'data').rglob('*.parquet')):
        # Predicate filtering avoids decoding unselected donor images into Python.
        table = pq.read_table(path, columns=['episode_index', 'frame_index', *AUXILIARY],
                              filters=[[('episode_index', '=', episode),
                                        ('frame_index', 'in', sorted(frame for ep, frame in wanted if ep == episode))]
                                       for episode in sorted({key[0] for key in wanted})])
        used = False
        for row in table.to_pylist():
            key = row['episode_index'], row['frame_index']
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f'Duplicate donor row {key}')
            for camera in AUXILIARY:
                payload = row[camera].get('bytes')
                if not payload:
                    raise ValueError('Donor cameras must contain embedded JPEG bytes')
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != 'JPEG' or image.size != (256, 256):
                        raise ValueError('Donor JPEG dimensions/format must be 256x256')
                    image.load()
            found[key] = dict(row=row, source_file=str(path.relative_to(root)))
            used = True
        if used:
            hashes[str(path.relative_to(root))] = sha256(path)
    if set(found) != wanted:
        raise ValueError(f'Missing donor candidates: {sorted(wanted - set(found))}')
    return found, hashes


def export(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Destination exists; choose a new output directory')
    if output == source or source in output.parents:
        raise ValueError('Output must be separate from source recordings')
    manifest_path = Path(args.donor_manifest).resolve()
    donor_manifest = read_json(manifest_path)
    donor_root = Path(donor_manifest['dataset_root'])
    if not donor_root.is_absolute() or not donor_root.is_dir():
        raise ValueError('Donor dataset_root must be an existing absolute directory')
    if not isinstance(donor_manifest.get('selection_method'), str):
        raise ValueError('Donor selection_method is required')
    candidates = donor_manifest.get('candidates', [])
    paired_donor_choices(candidates, 0, args.seed, 0)
    selected, excluded = retained_episodes(source)
    if not selected:
        raise ValueError('No complete retained unverified forward episodes')
    prepared = []
    for path, metadata in selected:
        source_hashes = {name: sha256(path / name) for name in
                         ('metadata.json', 'samples.jsonl', 'left_wrist.avi')}
        if read_json(path / 'metadata.json') != metadata:
            raise ValueError('Source metadata changed during preflight')
        rows, ticks, indices = validate_samples(path, metadata)
        ticks, indices, camera_indices, trimming = camera_aligned_grid(
            [row['monotonic_ns'] for row in rows], [row['camera']['monotonic_ns'] for row in rows])
        images = left_images(path / 'left_wrist.avi', len(rows), set(camera_indices.tolist()))
        prepared.append((path, metadata, rows, ticks, indices, camera_indices, images, trimming, source_hashes))
    donors, donor_hashes = donor_rows(donor_root, candidates)
    # Heavy runtime imports are intentionally deferred so timing/selection helpers
    # can be tested without loading the training framework.
    import pyarrow as pa
    import pyarrow.parquet as pq
    from export_retimed_dataset import ExactJpegDataset, encoded_image, refresh_numeric_statistics
    features = {key: dict(dtype='float32', shape=(14,), names=NAMES)
                for key in ('observation.state', 'action')}
    features.update({key: dict(dtype='image', shape=(256, 256, 3),
                               names=['height', 'width', 'channels']) for key in (*AUXILIARY, LEFT)})
    dataset = ExactJpegDataset.create(repo_id=args.repo_id, root=output, fps=FPS,
        robot_type='panthera_ht_bimanual_real', features=features, use_videos=False,
        metadata_buffer_size=1)
    report = dict(schema_version=1, repo_id=args.repo_id, fps=FPS, task=TASK, seed=args.seed,
        source_root=str(source), source_kind='autonomous_pin_regrasp_forward',
        label='unverified', success='unverified', auxiliary_camera_substitution=True,
        real_camera=LEFT, synthetic_camera_names=list(AUXILIARY),
        auxiliary_camera_semantics='Cross-episode paired donor views; not synchronized with robot state or real left wrist',
        action_semantics='Raw measured state at next 20Hz grid tick; not commanded targets',
        donor_manifest=donor_manifest, donor_manifest_sha256=sha256(manifest_path),
        donor_data_files_sha256=donor_hashes, excluded=excluded, episodes=[])
    alignment_dir = output / 'processing/alignment'
    alignment_dir.mkdir(parents=True)
    global_index = 0
    for episode_index, (path, metadata, rows, ticks, indices, camera_indices, images, trimming, source_hashes) in enumerate(prepared):
        count = len(ticks) - 1
        choices = paired_donor_choices(candidates, count, args.seed, episode_index)
        mapping = []
        for frame, candidate in enumerate(choices):
            source_index, action_index = int(indices[frame]), int(indices[frame + 1])
            observation, following = rows[source_index], rows[action_index]
            image_index = int(camera_indices[frame])
            camera_row = rows[image_index]
            donor = donors[(candidate['episode_index'], candidate['frame_index'])]
            encoded = {key: donor['row'][key] for key in AUXILIARY}
            encoded[LEFT] = dict(bytes=images[image_index])
            opened = {key: encoded_image(value) for key, value in encoded.items()}
            try:
                dataset.add_frame(dict(task=TASK,
                    **{'observation.state': np.asarray(observation['state'], dtype=np.float32),
                       'action': np.asarray(following['state'], dtype=np.float32)}, **opened))
            finally:
                for image in opened.values():
                    image.close()
            provenance = dict(index=global_index, episode_index=episode_index, frame_index=frame,
                timestamp=frame/FPS, source_episode_id=path.name,
                original_grid_index=frame+trimming['leading_grid_rows'],
                source_grid_monotonic_ns=int(ticks[frame]), next_grid_monotonic_ns=int(ticks[frame+1]),
                source_sample_index=source_index, next_state_sample_index=action_index,
                source_monotonic_ns=observation['monotonic_ns'], source_unix_ns=observation['unix_ns'],
                next_state_monotonic_ns=following['monotonic_ns'], next_state_unix_ns=following['unix_ns'],
                source_phase=observation['phase'], source_camera_sample_index=image_index,
                source_video_frame=camera_row['video_frame'],
                source_camera_sequence=camera_row['camera']['sequence'],
                source_camera_monotonic_ns=camera_row['camera']['monotonic_ns'],
                source_camera_unix_ns=camera_row['camera']['unix_ns'],
                camera_grid_offset_s=(camera_row['camera']['monotonic_ns']-int(ticks[frame]))/1e9,
                camera_age_at_source_sample_s=(camera_row['monotonic_ns']-camera_row['camera']['monotonic_ns'])/1e9,
                measured_grid_offset_s=(observation['monotonic_ns']-int(ticks[frame]))/1e9,
                source_commanded_action=observation['action'], next_source_commanded_action=following['action'],
                donor_episode_index=candidate['episode_index'], donor_frame_index=candidate['frame_index'],
                donor_source_file=donor['source_file'], donor_candidate_json=json.dumps(candidate),
                auxiliary_camera_substitution=True, synthetic_camera_names=list(AUXILIARY), success='unverified')
            for camera, value in encoded.items():
                provenance[camera + '.sha256'] = hashlib.sha256(value['bytes']).hexdigest()
            mapping.append(provenance)
            global_index += 1
        dataset.save_episode()
        pq.write_table(pa.Table.from_pylist(mapping), alignment_dir / f'episode_{episode_index:06d}.parquet',
                       compression='zstd')
        report['episodes'].append(dict(episode_index=episode_index, source_id=path.name,
            source_metadata=metadata, source_samples=len(rows), exported_frames=count, camera_trimming=trimming,
            source_hashes=source_hashes,
            alignment=f'processing/alignment/episode_{episode_index:06d}.parquet',
            success='unverified', auxiliary_camera_substitution=True))
        print(f'{episode_index+1}/{len(prepared)} {path.name}: {count} frames', flush=True)
    for path, *_, original_hashes in prepared:
        if any(sha256(path / name) != digest for name, digest in original_hashes.items()):
            raise ValueError(f'Source files changed during export: {path}')
    if sha256(manifest_path) != report['donor_manifest_sha256']:
        raise ValueError('Donor manifest changed during export')
    if any(sha256(donor_root / name) != digest for name, digest in donor_hashes.items()):
        raise ValueError('Donor source files changed during export')
    dataset.finalize()
    refresh_numeric_statistics(output)
    report['totals'] = dict(episodes=len(prepared), frames=global_index, duration_s=global_index/FPS,
        original_grid_frames=sum(entry['camera_trimming']['original_grid_frames'] for entry in report['episodes']),
        dropped_grid_rows=sum(entry['camera_trimming']['dropped_grid_rows'] for entry in report['episodes']),
        camera_trimmed_episodes=sum(entry['camera_trimming']['dropped_grid_rows'] > 0 for entry in report['episodes']))
    write_json(output / 'processing/export_report.json', report)
    write_json(output / 'processing/manifest.json', report)
    (output / 'README.md').write_text(f'''---
task_categories:
- robotics
tags:
- lerobot
- real-robot
- imitation-learning
configs:
- config_name: default
  data_files:
  - split: train
    path: data/chunk-*/*.parquet
---
# Pin regrasp forward collection

Task: **{TASK}**. {len(prepared)} episodes, {global_index} frames, 20 Hz, embedded 256×256 JPEGs.
Only retained complete return/close recordings are included. Every grasp outcome is **unverified**;
completion means the trajectory finished and does not establish grasp success.

`observation.state` is the raw measured 14-vector at the nearest original host timestamp.
`action` is the measured 14-vector at the **next 20 Hz grid tick**, consistent with the
measured-next training convention. The final observation without a next grid state is omitted.
Left wrist images independently use the nearest original camera capture timestamp; duplicate
captures select the first row and equal-distance ties select the earlier capture.
Observations require camera/grid alignment within 60 ms. Leading/trailing grid observations
with stale camera captures are removed, preserving the next measured-state action knot;
internal camera gaps reject export rather than joining disconnected intervals.
{report['totals']['camera_trimmed_episodes']} episode(s) had camera-unavailable heads/tails trimmed
({report['totals']['dropped_grid_rows']} grid observations); original files remain unchanged.
Trimming does not establish grasp success. Per-episode trimming details are in the export report.
Original commanded targets are retained only in processing/alignment parquet, with source row,
video frame, capture sequence, timestamps, donor identities, and image SHA-256 hashes.
No velocity or effort features are fabricated.

`observation.images.wrist_port3` is the real left wrist, decoded from original 640×480 MJPG,
cropped [80, 0, 560, 480], resized with Lanczos, and encoded JPEG quality 95, subsampling 0.
**Auxiliary camera substitution is enabled.** `observation.images.overhead` and
`observation.images.wrist_port2` are exact JPEG bytes from paired rows of another dataset.
They are cross-episode views, **not synchronized** with these states or left-wrist images.
One seeded random manifest candidate is selected independently per output frame; both auxiliary
cameras always share the same donor episode/frame. Do not treat these views as physical evidence
of the recorded grasp or as calibrated multi-view observations.

`processing/export_report.json` records the full donor selection manifest, original episode
metadata, all source hashes, excluded episodes, and measured-next alignment semantics.
Native LeRobot v3 finalization writes episode metadata and image statistics; numeric statistics
are refreshed from the actual stored float32 values. Original recordings are never modified.
''')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--donor-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', default='local/panthera-pin-regrasp-20hz')
    parser.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    export(args)


if __name__ == '__main__':
    main()
