#!/usr/bin/env python3
"""Export a separate, provenance-linked LeRobot dataset with slower motion timing.

Run with work/postprocess-venv/bin/python. Original data and training configuration
are never modified. Embedded JPEG bytes are copied exactly, with one source clock
for every camera, state, and next-pose target. Source sampling is only 20 Hz: this
cannot recover unobserved paths or new camera views. Measured effort is interpolated
as historical context; it is not an estimate of retimed dynamics.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from add_precision_metadata import frame_label
from trajectory_retiming import retime

DEFAULT_KEYS = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ExactJpegDataset(LeRobotDataset):
    """Use native metadata/stats writer while preserving existing encoded pixels."""
    def _save_image(self, image, fpath, compress_level=1):
        # PIL is accepted by native frame validation. Opening is lazy: stats decode
        # these exact bytes from disk after save_episode, just as in a normal export.
        payload = getattr(image, '_retiming_encoded_bytes', None)
        if payload is None:
            raise ValueError('All retimed images must have original embedded bytes')
        Path(fpath).write_bytes(payload)


def encoded_image(value):
    payload = value.get('bytes')
    if not payload:
        raise ValueError('Source must contain embedded images, not external paths')
    image = Image.open(io.BytesIO(payload))
    if image.format != 'JPEG':
        raise ValueError('Expected embedded JPEG source')
    image._retiming_encoded_bytes = payload
    return image


def interpolation(frames, count):
    frames = np.clip(np.asarray(frames, dtype=np.float64), 0, count - 1)
    left = np.floor(frames).astype(int)
    right = np.minimum(left + 1, count - 1)
    return left, right, frames - left


def interpolate(values, frames):
    left, right, weight = interpolation(frames, len(values))
    return values[left] * (1 - weight[:, None]) + values[right] * weight[:, None]



def camera_alignment_extra_bound(source_frame, source_count, fps):
    """Additional image lag from nearest observation selection, in source seconds.

    There is a terminal action knot at source_count, but images exist only through
    source_count - 1. Beyond the last half-tick, nearest-knot selection therefore
    clamps to the final observation and can lag by nearly a full source tick.
    """
    if (source_count < 1 or fps <= 0 or not np.isfinite(source_frame) or
            source_frame < -1e-7 or source_frame > source_count + 1e-7):
        raise ValueError('Invalid source coordinate for camera selection')
    terminal_clamped = int(np.floor(source_frame + .5)) >= source_count
    return (1. if terminal_clamped else .5) / fps


def episode_tables(source, files, episode, columns):
    tables = [pq.read_table(path, columns=columns,
                           filters=[('episode_index', '=', episode)]) for path in files]
    table = pa.concat_tables(tables)
    return table.take(pa.compute.sort_indices(table, sort_keys=[('frame_index', 'ascending')]))


def source_clock(source, episode, count, fps, annotation):
    path = source / f'processing/alignment/episode_{episode:06d}.parquet'
    if path.exists():
        alignments = pq.read_table(path).to_pylist()
        if len(alignments) != count:
            raise ValueError(f'Alignment count differs for episode {episode}')
        times = np.asarray([row['source_episode_time_s'] for row in alignments], dtype=float)
        # Alignment uses source-grid time, preserving annotation clock even when
        # motor measurements were selected a few milliseconds from each grid tick.
        times = np.r_[times, times[-1] + 1 / fps]
        return times, alignments, sha256(path)
    if annotation.get('precision_phases'):
        raise ValueError('Precision protection requires source alignment clock')
    start = float(annotation.get('first_source_grid_s', 0.))
    return start + np.arange(count + 1) / fps, None, None


def precision_protection(times, phases):
    mask = np.zeros(len(times) - 1, dtype=bool)
    for phase in phases:
        mask |= (times[:-1] < phase['end_s']) & (times[1:] > phase['start_s'])
    return mask



def exact_numeric_statistics(values):
    """Native-shaped population statistics with float64 centered variance.

    The native writer computes E[x**2]-E[x]**2 in the input float32 dtype;
    low-variance measured torques can lose precision through cancellation.
    Read stored values exactly, then center and reduce in float64 instead.
    """
    raw = np.asarray(values)
    if raw.ndim == 1:
        raw = raw[:, None]
    values64 = raw.astype(np.float64)
    if not len(values64) or not np.isfinite(values64).all():
        raise ValueError('Numeric statistics require nonempty finite values')
    result = dict(min=raw.min(axis=0).tolist(), max=raw.max(axis=0).tolist(),
        mean=values64.mean(axis=0).tolist(), std=values64.std(axis=0).tolist(),
        count=[len(values64)])
    for quantile in (.01, .10, .50, .90, .99):
        result[f'q{int(quantile * 100):02d}'] = np.quantile(values64, quantile, axis=0).tolist()
    return result


def refresh_numeric_statistics(root):
    """Recompute exact numeric metadata only; preserve all media stats/data rows.

    Loads numeric columns only. The exact global quantiles are calculated from
    all exported numeric samples, rather than averages of episode quantiles.
    Each episode parquet keeps its native schema and every nonnumeric field.
    """
    root = Path(root)
    info = read_json(root / 'meta/info.json')
    features = [key for key, feature in info['features'].items()
                if feature['dtype'] not in ('image', 'video', 'string')]
    tables = [pq.read_table(path, columns=features)
              for path in sorted((root / 'data').rglob('*.parquet'))]
    numeric = pa.concat_tables(tables).to_pydict()
    arrays = {key: np.asarray(values) for key, values in numeric.items()}
    episodes = arrays['episode_index']
    if len(episodes) != info['total_frames']:
        raise ValueError('Numeric sample count differs from dataset metadata')
    by_episode = {}
    for episode in np.unique(episodes):
        selected = episodes == episode
        by_episode[int(episode)] = {
            key: exact_numeric_statistics(values[selected]) for key, values in arrays.items()}
    stats_path = root / 'meta/stats.json'
    stats = read_json(stats_path)
    image_stats = {key: copy.deepcopy(value) for key, value in stats.items() if key not in features}
    stats.update({key: exact_numeric_statistics(values) for key, values in arrays.items()})
    updated_episodes = 0
    for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
        original = pq.read_table(path)
        rows = original.to_pylist()
        for row in rows:
            for key, feature_stats in by_episode[int(row['episode_index'])].items():
                for statistic, value in feature_stats.items():
                    name = f'stats/{key}/{statistic}'
                    if name not in row:
                        raise ValueError(f'Missing native numeric statistic {name}')
                    row[name] = value
            updated_episodes += 1
        temporary = path.with_suffix('.parquet.refresh-tmp')
        pq.write_table(pa.Table.from_pylist(rows, schema=original.schema), temporary, compression='zstd')
        temporary.replace(path)
    if updated_episodes != info['total_episodes']:
        raise ValueError('Episode metadata count differs from dataset info')
    write_json(stats_path, stats)
    if {key: stats[key] for key in image_stats} != image_stats:
        raise AssertionError('Image statistics changed during numeric refresh')
    report = dict(version=1, numeric_features=features, episodes=updated_episodes,
        frames=len(episodes), method='Exact min/max and float64 centered population mean/std; exact numpy linear quantiles from stored values, per episode and globally.',
        image_statistics_preserved=True, data_rows_modified=False)
    write_json(root / 'processing/numeric_statistics_refresh.json', report)
    return report


def export(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    if source == output or source in output.parents:
        raise ValueError('Output must be separate from source dataset')
    info = read_json(source / 'meta/info.json')
    manifest_path = source / 'processing/manifest.json'
    original_manifest = read_json(manifest_path)
    config = read_json(args.config)
    fps = float(info['fps'])
    if fps != 20:
        raise ValueError('This exporter currently requires a 20 Hz source')
    features = {key: copy.deepcopy(value) for key, value in info['features'].items()
                if key not in DEFAULT_KEYS}
    for feature in features.values():
        feature['shape'] = tuple(feature['shape'])
    cameras = [key for key, feature in features.items() if feature['dtype'] == 'image']
    required = {'observation.state', 'action', 'observation.velocity', 'observation.effort'}
    if set(features) != required | set(cameras):
        raise ValueError(f'Unsupported source features: {set(features) - required - set(cameras)}')
    if not cameras:
        raise ValueError('No embedded camera features')
    files = sorted((source / 'data').rglob('*.parquet'))
    if not files:
        raise ValueError(f'No source parquet data: {source}')
    # Numeric-only first pass discovers file boundaries without reading any media.
    episode_files = {}
    numeric_cache = {}
    numeric_columns = ['observation.state', 'action', 'observation.effort',
                       'episode_index', 'frame_index', 'task_index']
    for path in files:
        table = pq.read_table(path, columns=numeric_columns)
        episodes = np.asarray(table['episode_index'])
        for episode in np.unique(episodes):
            episode = int(episode)
            episode_files.setdefault(episode, []).append(path)
            numeric_cache.setdefault(episode, []).append(table.filter(pa.array(episodes == episode)))
    if sorted(episode_files) != list(range(info['total_episodes'])):
        raise ValueError('Source episode indices must be contiguous')
    tasks = {row['task_index']: row['__index_level_0__']
             for row in pq.read_table(source / 'meta/tasks.parquet').to_pylist()}
    annotations = {int(row['episode_index']): row for row in original_manifest['episodes']}
    source_hashes = {str(p.relative_to(source)): sha256(p)
                     for p in [source / 'meta/info.json', manifest_path, source / 'meta/tasks.parquet']}
    # Hash once per data file, never once per episode in a shared file.
    source_data_hashes = {str(p.relative_to(source)): sha256(p) for p in files}
    dataset = ExactJpegDataset.create(repo_id=args.repo_id, root=output, fps=int(fps),
                robot_type=info['robot_type'], features=features, use_videos=False,
                metadata_buffer_size=1)
    manifest = copy.deepcopy(original_manifest)
    manifest.update(repo_id=args.repo_id, episodes=[], source_kind='retimed_processed_real_recordings')
    manifest['retiming'] = dict(version=1, source_root=str(source), source_hashes=source_hashes,
        source_data_files_sha256=source_data_hashes, config=config, cohort_label=args.cohort_label,
        source_sampling_hz=fps, uniform_output_hz=fps, image_encoding='Original embedded JPEG bytes copied exactly',
        state_semantics='Linear interpolation of source state knots plus original final action terminal knot',
        action_semantics='Next retimed pose at 1/fps seconds, with the original terminal action preserved',
        velocity_semantics='Forward finite difference of retimed pose trajectory, matching next-pose action demand',
        effort_semantics='Interpolated recorded measured effort; historical context, not physical retimed torque',
        camera_semantics='All cameras use the same nearest source state row; added source-clock alignment error is at most half a tick, or one tick when terminal image clamps to final observation',
        precision_semantics='All annotated precision phase overlaps are protected in addition to automatic slow intervals',
        training_changed=False)
    source_contract_keys = ('sampling', 'timing', 'state_semantics', 'action_semantics',
        'max_state_alignment_s', 'max_camera_alignment_s', 'source_lookup_root', 'source_lookup_note',
        'rebuild_contract', 'reused_processed_batches', 'curation_summaries', 'overhead_crop_audit',
        'cohort_summary')
    manifest['retiming']['source_processing_contract'] = {
        key: copy.deepcopy(original_manifest[key]) for key in source_contract_keys
        if key in original_manifest}
    for key in source_contract_keys:
        manifest.pop(key, None)
    manifest.update(
        sampling='Uniform 20 Hz output clock; one monotone retimed source map drives linear pose interpolation and nearest original image-row selection for all cameras.',
        timing='Output timestamps are retimed trajectory time. Original host receive/image timestamps remain in the per-row retiming sidecars; they are not hardware exposure timestamps.',
        state_semantics=manifest['retiming']['state_semantics'] + '; all 14 joint and gripper coordinates remain native radians.',
        action_semantics=manifest['retiming']['action_semantics'] + '; absolute next-pose proxy, not sent motor commands.')
    precision_path = source / 'meta/precision.json'
    precision = read_json(precision_path) if precision_path.exists() else {}
    multiplier = float(precision.get('suggested_precision_multiplier', 3.))
    precision_rows, reports = [], []
    global_index = 0
    for episode in sorted(episode_files):
        table = pa.concat_tables(numeric_cache.pop(episode))
        table = table.take(pa.compute.sort_indices(table, sort_keys=[('frame_index', 'ascending')]))
        numeric = table.to_pydict()
        count = len(numeric['frame_index'])
        if numeric['frame_index'] != list(range(count)):
            raise ValueError(f'Non-contiguous source frames in episode {episode}')
        states = np.asarray(numeric['observation.state'], dtype=float)
        actions = np.asarray(numeric['action'], dtype=float)
        if states.shape != (count, 14) or not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f'Invalid source poses in episode {episode}')
        if not np.allclose(states[1:], actions[:-1], atol=1e-6, rtol=0):
            raise ValueError(f'Next-pose source continuity fails in episode {episode}')
        poses = np.concatenate([states, actions[-1:]])
        annotation = annotations[episode]
        times, alignments, alignment_hash = source_clock(source, episode, count, fps, annotation)
        phases = annotation.get('precision_phases', [])
        phase_mask = precision_protection(times, phases)
        result = retime(poses, fps=fps, protected_mask=phase_mask, **config)
        target_poses = np.asarray(result['poses'], dtype=np.float32)
        source_frames = np.asarray(result['source_times']) * fps
        if not np.allclose(source_frames[[0, -1]], [0, count], atol=1e-6):
            raise ValueError('Retiming must preserve both terminal source knots')
        if not np.allclose(np.diff(result['output_times']), 1 / fps, atol=1e-8):
            raise ValueError('Retiming output must have a uniform 20 Hz clock')
        length = len(target_poses) - 1
        velocities = np.diff(target_poses, axis=0) * fps
        # Source effort has no terminal measurement: use the final observed effort.
        effort = np.asarray(numeric['observation.effort'], dtype=float)
        effort = np.concatenate([effort, effort[-1:]])
        target_effort = interpolate(effort, source_frames[:-1]).astype(np.float32)
        observation_times = np.interp(source_frames[:-1], np.arange(count + 1), times)
        action_times = np.interp(source_frames[1:], np.arange(count + 1), times)
        left, right, weights = interpolation(source_frames[:-1], count + 1)
        action_left, action_right, action_weights = interpolation(source_frames[1:], count + 1)
        image_frames = np.minimum(np.floor(source_frames[:-1] + .5).astype(int), count - 1)
        image_table = episode_tables(source, episode_files[episode], episode,
                                    ['episode_index', 'frame_index'] + cameras)
        images = image_table.to_pydict()
        mapping = []
        for frame in range(length):
            image_frame = int(image_frames[frame])
            row = {'task': tasks[numeric['task_index'][image_frame]],
                   'observation.state': target_poses[frame], 'action': target_poses[frame + 1],
                   'observation.velocity': velocities[frame], 'observation.effort': target_effort[frame]}
            opened = []
            for camera in cameras:
                image = encoded_image(images[camera][image_frame])
                opened.append(image)
                row[camera] = image
            dataset.add_frame(row)
            for image in opened:
                image.close()
            provenance = dict(index=global_index, episode_index=episode, frame_index=frame,
                source_episode_index=episode, source_episode_id=annotation['id'],
                source_state_frame=float(source_frames[frame]), source_action_frame=float(source_frames[frame + 1]),
                source_image_frame=image_frame,
                image_source_terminal_clamped=bool(np.floor(source_frames[frame] + .5) >= count),
                camera_alignment_extra_bound_s=camera_alignment_extra_bound(source_frames[frame], count, fps),
                source_state_left=int(left[frame]),
                source_state_right=int(right[frame]), source_state_weight=float(weights[frame]),
                source_action_left=int(action_left[frame]), source_action_right=int(action_right[frame]),
                source_action_weight=float(action_weights[frame]),
                source_episode_time_s=float(observation_times[frame]),
                action_source_episode_time_s=float(action_times[frame]))
            if alignments is not None:
                inherited = alignments[image_frame]
                for camera in cameras:
                    name = camera.removeprefix('observation.images.')
                    provenance.update({key: value for key, value in inherited.items()
                                       if key.startswith(name + '.') and key != name + '.offset_s'})
                    # Keep original alignment evidence and report the actual offset
                    # relative to the fractional source tick selected by retiming.
                    original_offset = float(inherited[name + '.offset_s'])
                    provenance[name + '.original_offset_s'] = original_offset
                    provenance[name + '.offset_s'] = (float(inherited['source_episode_time_s']) +
                        original_offset - float(observation_times[frame]))
                    bound = provenance['camera_alignment_extra_bound_s']
                    if abs(provenance[name + '.offset_s']) > abs(original_offset) + bound + 1e-7:
                        raise ValueError('Camera alignment exceeds nearest-row or terminal-clamp bound')
            mapping.append(provenance)
            precision_rows.append(dict(index=global_index, episode_index=episode, frame_index=frame,
                timestamp=frame / fps, action_source_time_s=float(action_times[frame]),
                observation_source_time_s=float(observation_times[frame]),
                **frame_label(phases, action_times[frame], multiplier)))
            global_index += 1
        dataset.save_episode()
        directory = output / 'processing/retiming'
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(mapping), directory / f'episode_{episode:06d}.parquet', compression='zstd')
        report = dict(source_episode_index=episode, source_id=annotation['id'], source_frames=count,
            output_frames=length, source_duration_s=count / fps, output_duration_s=length / fps,
            added_duration_s=(length - count) / fps, source_alignment_sha256=alignment_hash,
            source_hashes=annotation.get('source_hashes', {}),
            source_data_files_sha256={str(p.relative_to(source)): source_data_hashes[str(p.relative_to(source))]
                                      for p in episode_files[episode]},
            source_knot_episode_times_s=times.tolist(), precision_protected_mask=phase_mask.tolist(),
            protected_mask=np.asarray(result['protected_mask']).tolist(),
            interval_scales=np.asarray(result['interval_scales']).tolist(), metrics=result['metrics'])
        write_json(directory / f'episode_{episode:06d}.json', report)
        reports.append(report)
        entry = copy.deepcopy(annotation)
        entry.pop('max_camera_alignment_s', None)
        entry['max_retimed_camera_source_alignment_s'] = max(
            (abs(row[camera.removeprefix('observation.images.') + '.offset_s'])
             for row in mapping for camera in cameras), default=0.) if alignments is not None else None
        entry.update(episode_index=episode, exported_observations=length,
                     retiming_report=f'processing/retiming/episode_{episode:06d}.json',
                     source_exported_observations=count)
        manifest['episodes'].append(entry)
        print(f'{episode + 1}/{len(episode_files)} {annotation["id"]}: {count} -> {length} frames '
              f'(+{(length - count) / fps:.2f}s)', flush=True)
    dataset.finalize()
    refresh_numeric_statistics(output)
    source_duration = sum(report['source_duration_s'] for report in reports)
    output_duration = global_index / fps
    manifest['totals'] = dict(episodes=len(reports), observations=global_index,
        source_observations=sum(report['source_frames'] for report in reports),
        source_duration_s=source_duration, output_duration_s=output_duration,
        added_duration_s=output_duration - source_duration,
        added_duration_percent=100 * (output_duration / source_duration - 1))
    manifest['max_retimed_camera_source_alignment_s'] = max(
        (entry['max_retimed_camera_source_alignment_s'] or 0. for entry in manifest['episodes']), default=0.)
    manifest['retiming']['verification'] = dict(terminal_source_knots_preserved=True,
        source_next_pose_continuity_checked=True, uniform_output_clock_checked=True,
        original_encoded_jpeg_bytes_written=True, all_camera_source_rows_shared=True,
        report='processing/retiming/summary.json')
    write_json(output / 'processing/manifest.json', manifest)
    write_json(output / 'processing/retiming/summary.json', dict(totals=manifest['totals'],
        config=config, source_root=str(source), training_changed=False,
        episodes=[{key: value for key, value in report.items()
                   if key not in {'source_knot_episode_times_s', 'protected_mask', 'precision_protected_mask', 'interval_scales'}}
                  for report in reports]))
    pq.write_table(pa.Table.from_pylist(precision_rows), output / 'meta/precision_weights.parquet', compression='zstd')
    if precision:
        precision['weight_time_basis'] = 'Retimed next-pose action source episode time from shared source map'
        precision['retimed_from'] = str(source)
        precision['coverage'] = dict(total_action_targets=global_index,
            weighted_action_targets=sum(row['loss_weight'] > 1 for row in precision_rows),
            gear_action_targets=sum(row['gear_insertion'] for row in precision_rows),
            pin_action_targets=sum(row['pin_insertion'] for row in precision_rows),
            episodes=len(reports))
        write_json(output / 'meta/precision.json', precision)
        native_info = read_json(output / 'meta/info.json')
        native_info['precision_loss_metadata'] = dict(schema='meta/precision.json',
            table='meta/precision_weights.parquet', join_key='index')
        write_json(output / 'meta/info.json', native_info)
        manifest['precision_loss_metadata'] = {key: value for key, value in precision.items()
            if key in {'table', 'join_key', 'default_loss_weight', 'suggested_precision_multiplier',
                       'weight_time_basis', 'coverage'}}
        write_json(output / 'processing/manifest.json', manifest)
    write_json(output / 'processing/retiming/config.json', config)
    print(json.dumps(manifest['totals'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cohort-label')
    export(parser.parse_args())
