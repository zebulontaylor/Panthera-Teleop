#!/usr/bin/env python3
"""Copy a pin dataset unchanged and append explicitly selected alignment episodes.

Run with work/postprocess-venv/bin/python. Never uploads or changes sources.
Selection: {"episodes": [{"source_path": "/absolute/episode", "task":
"align the gear"}], "excluded": [...]}. Task labels are supplied by selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from export_pin_regrasp_dataset import (FPS, NAMES, nearest_rows, regular_grid,
    left_images, read_json, write_json, sha256)

CAMERAS = {'left_wrist': 'observation.images.wrist_port3',
           'right_wrist': 'observation.images.wrist_port2',
           'overhead': 'observation.images.overhead'}


def prepare(entry):
    path = Path(entry['source_path']).resolve()
    if not path.is_dir() or any('discarded' in part.lower() for part in path.parts):
        raise ValueError(f'Invalid retained source: {path}')
    task = entry['task']
    if not isinstance(task, str) or not task.strip():
        raise ValueError('Each selected episode needs a nonempty task label')
    metadata = read_json(path / 'metadata.json')
    if (metadata.get('complete') is not True or metadata.get('discarded') or
            metadata.get('error') or metadata.get('label') != 'unverified'):
        raise ValueError(f'Source must be retained complete and unverified: {path}')
    names = ['metadata.json', 'samples.jsonl', *(f'{camera}.avi' for camera in CAMERAS)]
    hashes = {name: sha256(path / name) for name in names}
    rows = [json.loads(line) for line in (path / 'samples.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != metadata.get('samples'):
        raise ValueError(f'Sample count mismatch: {path}')
    for index, row in enumerate(rows):
        if row.get('index') != index or row.get('video_frame') != index:
            raise ValueError(f'Invalid source/video index: {path}:{index}')
        if row.get('phase') not in ('return', 'close'):
            raise ValueError(f'Unexpected source phase: {path}:{index}')
        for key in ('state', 'action'):
            vector = np.asarray(row.get(key), dtype=float)
            if vector.shape != (14,) or not np.isfinite(vector).all():
                raise ValueError(f'Invalid {key}: {path}:{index}')
        for clock in ('monotonic_ns', 'unix_ns'):
            if type(row.get(clock)) is not int or row[clock] < 0:
                raise ValueError(f'Invalid host clock: {path}:{index}')
    ticks, indices = regular_grid([row['monotonic_ns'] for row in rows])
    selected, valid = {}, np.ones(len(ticks)-1, dtype=bool)
    for camera in CAMERAS:
        captures = []
        previous_sequence = -1
        for row in rows:
            capture = row.get('cameras', {}).get(camera, {})
            for clock in ('monotonic_ns', 'unix_ns', 'sequence'):
                if type(capture.get(clock)) is not int or capture[clock] < 0:
                    raise ValueError(f'Missing camera clock: {path}:{camera}')
            if (capture['sequence'] < previous_sequence or
                    capture['monotonic_ns'] > row['monotonic_ns'] or
                    (capture.get('width'), capture.get('height')) != (640, 480)):
                raise ValueError(f'Invalid camera capture: {path}:{camera}')
            previous_sequence = capture['sequence']
            captures.append(capture['monotonic_ns'])
        selected[camera] = nearest_rows(captures, ticks[:-1])
        valid &= np.abs(np.asarray(captures)[selected[camera]]-ticks[:-1]) <= 60_000_000
    kept = np.flatnonzero(valid)
    if not len(kept) or not valid[kept[0]:kept[-1]+1].all():
        raise ValueError(f'No contiguous camera-aligned observations, or internal gap: {path}')
    first, last = int(kept[0]), int(kept[-1])
    trimming = dict(original_grid_frames=len(valid), leading_grid_rows=first,
        trailing_grid_rows=len(valid)-last-1, dropped_grid_rows=int((~valid).sum()),
        max_camera_alignment_s=.06)
    selected = {camera: values[first:last+1] for camera, values in selected.items()}
    return path, task, metadata, hashes, rows, ticks[first:last+2], indices[first:last+2], selected, trimming


def export(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from export_retimed_dataset import ExactJpegDataset, encoded_image, refresh_numeric_statistics
    base, output, selection_path = args.base.resolve(), args.output.resolve(), args.selection.resolve()
    if output.exists() or output == base or base in output.parents:
        raise ValueError('Choose a fresh output outside the base dataset')
    selection = read_json(selection_path)
    entries = selection.get('episodes', [])
    if not entries or len({str(Path(e['source_path']).resolve()) for e in entries}) != len(entries):
        raise ValueError('Selection must contain nonempty unique source episodes')
    info = read_json(base / 'meta/info.json')
    if info['fps'] != FPS or set(k for k, v in info['features'].items() if v['dtype'] == 'image') != set(CAMERAS.values()):
        raise ValueError('Expected a 20Hz three-image base dataset')
    selection_hash = sha256(selection_path)
    # Preflight all clocks before creating the output. Image payloads stay per-episode.
    prepared = [prepare(entry) for entry in entries]
    for path, *_ in prepared:
        if output == path or path in output.parents:
            raise ValueError('Output must not be inside source recordings')
    protected = {str(path.relative_to(base)): sha256(path)
        for directory in ('data', 'meta/episodes', 'processing')
        for path in (base / directory).rglob('*') if path.is_file()}
    shutil.copytree(base, output, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    # Native LeRobot resume starts new data AND episode-metadata parquet files.
    # Original parquet files are never opened for writing by the append writer.
    for receipt in ('verification.json', 'publication.json', 'COMPLETE'):
        (output / receipt).unlink(missing_ok=True)
    dataset = ExactJpegDataset(args.repo_id, root=output)
    dataset.meta.metadata_buffer_size = 1
    initial_frames, initial_episodes = info['total_frames'], info['total_episodes']
    report = dict(schema_version=1, repo_id=args.repo_id, base_root=str(base),
        base_episodes=initial_episodes, base_frames=initial_frames,
        base_file_hashes=protected, selection=selection, selection_sha256=selection_hash,
        fps=FPS, action_semantics='Raw measured state at next 20Hz grid tick; not commanded targets',
        camera_semantics='Each camera independently nearest capture within 60ms; earlier tie and first duplicate',
        crop=[80, 0, 560, 480], image_size=[256, 256], jpeg_quality=95,
        jpeg_subsampling=0, success='unverified', episodes=[])
    global_index = initial_frames
    alignment = output / 'processing/alignment'
    alignment.mkdir(parents=True, exist_ok=True)
    for offset, prepared_episode in enumerate(prepared):
        path, task, metadata, hashes, rows, ticks, indices, cameras, trimming = prepared_episode
        episode = initial_episodes + offset
        images = {camera: left_images(path / f'{camera}.avi', len(rows), set(values.tolist()))
                  for camera, values in cameras.items()}
        provenance = []
        for frame in range(len(ticks)-1):
            current, following = int(indices[frame]), int(indices[frame+1])
            encoded = {CAMERAS[camera]: dict(bytes=images[camera][int(values[frame])])
                       for camera, values in cameras.items()}
            opened = {key: encoded_image(value) for key, value in encoded.items()}
            try:
                dataset.add_frame(dict(task=task, **opened,
                    **{'observation.state': np.asarray(rows[current]['state'], dtype=np.float32),
                       'action': np.asarray(rows[following]['state'], dtype=np.float32)}))
            finally:
                for image in opened.values():
                    image.close()
            record = dict(index=global_index, episode_index=episode, frame_index=frame,
                timestamp=frame/FPS, source_path=str(path), task=task,
                source_sample_index=current, next_state_sample_index=following,
                source_grid_monotonic_ns=int(ticks[frame]), next_grid_monotonic_ns=int(ticks[frame+1]),
                source_monotonic_ns=rows[current]['monotonic_ns'],
                next_state_monotonic_ns=rows[following]['monotonic_ns'],
                source_commanded_action=rows[current]['action'], source_phase=rows[current]['phase'],
                success='unverified', auxiliary_camera_substitution=False)
            for camera, values in cameras.items():
                image_index = int(values[frame])
                capture = rows[image_index]['cameras'][camera]
                record[camera] = dict(source_sample_index=image_index,
                    source_video_frame=rows[image_index]['video_frame'], **capture,
                    grid_offset_s=(capture['monotonic_ns']-int(ticks[frame]))/1e9,
                    image_sha256=hashlib.sha256(encoded[CAMERAS[camera]]['bytes']).hexdigest())
            provenance.append(record)
            global_index += 1
        dataset.save_episode()
        pq.write_table(pa.Table.from_pylist(provenance), alignment / f'episode_{episode:06d}.parquet', compression='zstd')
        report['episodes'].append(dict(episode_index=episode, source_path=str(path), task=task,
            frames=len(provenance), source_metadata=metadata, source_hashes=hashes, camera_trimming=trimming))
        print(f'{offset+1}/{len(prepared)} {task} {path.name}: {len(provenance)} frames', flush=True)
    dataset.finalize()
    refresh = refresh_numeric_statistics(output)
    write_json(output / 'processing/alignment_append_numeric_statistics.json', refresh)
    # Original per-episode stats already describe unchanged original rows exactly.
    # Restore their original files to preserve all existing episode metadata bytes.
    for name, digest in protected.items():
        if name.startswith(('meta/episodes/', 'processing/')):
            shutil.copy2(base / name, output / name)
        if sha256(output / name) != digest or sha256(base / name) != digest:
            raise ValueError(f'Original dataset file changed: {name}')
    for path, _, _, hashes, *_ in prepared:
        if any(sha256(path / name) != digest for name, digest in hashes.items()):
            raise ValueError(f'Source changed during export: {path}')
    if sha256(selection_path) != selection_hash:
        raise ValueError('Selection changed during export')
    report['totals'] = dict(episodes=initial_episodes+len(prepared), frames=global_index,
        appended_episodes=len(prepared), appended_frames=global_index-initial_frames)
    write_json(output / 'processing/alignment_append_report.json', report)
    original_readme = (output / 'README.md').read_text()
    counts = {}
    for entry in report['episodes']:
        counts[entry['task']] = counts.get(entry['task'], 0) + 1
    frontmatter = ''
    if original_readme.startswith('---\n'):
        end = original_readme.find('\n---', 4) + 4
        frontmatter, original_readme = original_readme[:end] + '\n', original_readme[end:]
    (output / 'README.md').write_text(frontmatter + '\n# Pin pickup and alignment collection\n\n'
        + f"{report['totals']['episodes']} episodes, {global_index} frames, 20 Hz. Original pin pickup: {initial_episodes} episodes. "
        + '; '.join(f'{task}: {count} episodes' for task, count in counts.items()) + '.\n\n'
        + 'Existing `processing/manifest.json` and `processing/export_report.json` audit only the original '
        + f'{initial_episodes} pickup episodes. `processing/alignment_append_report.json` describes the current additions and totals.\n\n'
        + '## Original pin pickup subset documentation\n' + original_readme)
    with (output / 'README.md').open('a') as stream:
        stream.write('\n\n## Alignment additions\n\n'
            f'Appended {len(prepared)} alignment episodes ({global_index-initial_frames} frames) at 20 Hz. '
            'Original pin episodes and embedded JPEG bytes are unchanged. Added episodes use all three '
            'recorded cameras, each selected independently by nearest capture time within 60 ms. '
            'Only unavailable heads/tails are trimmed; internal camera gaps reject export. '
            'States are measured joint positions and actions are measured positions at the next grid tick. '
            'All outcomes remain unverified. Added images use the same centered square crop, Lanczos '
            '256×256 resize and JPEG quality 95 as original wrist images. Explicit selection, exclusions, '
            'task labels, source hashes and per-frame alignment are in processing/alignment_append_report.json '
            'and processing/alignment/. Original pin auxiliary-camera substitution semantics remain applicable '
            'only to the original pin episodes.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    print(json.dumps(export(parser.parse_args())['totals'], indent=2))


if __name__ == '__main__':
    main()
