#!/usr/bin/env python3
"""Cache native assembly CNN measurements from complete operator-kept recordings.

Offline only: imports no robot interface and never changes source recordings.
Run with RobotArmTraining/.venv-cnns/bin/python. Host receive timestamps are
matched within explicit limits; camera assignment is supplied, not inferred.
The default image order restores the adapted checkpoints' camera roles for the
October 22 setup: overhead, port3 (left gear/pin), port2 (right carrier).
Historical adaptation recordings used opposite USB port names. Model slots
must follow camera content; audit the mapping when hardware connections change.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT.parent / 'RobotArmTraining/outputs/panthera-assembly-cnns'
DEFAULT_CAMERA_NAMES = ('overhead', 'wrist_port3', 'wrist_port2')
CHECKPOINTS = {'gear': 'gear_alignment/real_v4/adapted.pt',
               'pin': 'pin_perception/carrier_v4/adapted.pt'}
PREDICTION_KEYS = ('keypoints_2d', 'visibility_probability', 'confidence', 'pixel_std',
                   'peak_probability', 'camera_names', 'landmark_names')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def nearest(rows, times, timestamp, max_skew_s):
    index = bisect_left(times, timestamp)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    if not candidates:
        raise ValueError('empty observation stream')
    index = min(candidates, key=lambda i: (abs(times[i] - timestamp), times[i]))
    if abs(times[index] - timestamp) > max_skew_s * 1e9:
        raise ValueError('timestamp skew exceeds limit')
    return rows[index]


def episode_samples(path, camera_names, fps, camera_max_skew_s=.04, state_max_skew_s=.01, phases=None,
                    allow_non_kept=False):
    """Yield synchronized samples or rejection records across the full recording."""
    metadata = json.loads((path / 'metadata.json').read_text())
    if (metadata.get('label') not in (('kept', 'review', 'discarded') if allow_non_kept else ('kept',))
            or not metadata.get('complete')):
        raise ValueError(f'{path}: recording must be complete and operator-kept')
    if metadata.get('source_kind') == 'hitl' or metadata.get('context', {}).get('mode') == 'hitl':
        raise ValueError(f'{path}: HITL clocks need an explicit synchronization contract')
    states = sorted(read_jsonl(path / 'states.jsonl'), key=lambda x: x['monotonic_ns'])
    times = [x['monotonic_ns'] for x in states]
    all_frames = read_jsonl(path / 'frames.jsonl')
    frames = {c: sorted((x for x in all_frames if x['camera'] == c),
                        key=lambda x: x['monotonic_ns']) for c in camera_names}
    frame_times = {c: [x['monotonic_ns'] for x in rows] for c, rows in frames.items()}
    origin = int(metadata['start_monotonic_ns'])
    end = min(int(metadata['end_monotonic_ns']), times[-1])
    for tick in range(math.floor((end - origin) / 1e9 * fps) + 1):
        time_s = tick / fps
        phase_names = [name for name, bounds in (phases or {}).items()
                       if bounds['start_s'] - 1e-9 <= time_s <= bounds['end_s'] + 1e-9]
        if phases is not None and not phase_names:
            continue
        timestamp = origin + round(time_s * 1e9)
        try:
            state = nearest(states, times, timestamp, state_max_skew_s)
            selected = {c: nearest(frames[c], frame_times[c], timestamp, camera_max_skew_s)
                        for c in camera_names}
        except ValueError as exc:
            yield {'episode_id': metadata['id'], 'time_s': tick / fps, 'rejection': str(exc)}
            continue
        yield {'episode_id': metadata['id'], 'episode_path': str(path.resolve()),
               'time_s': tick / fps, 'phases': phase_names, 'monotonic_ns': timestamp,
               'state_monotonic_ns': state['monotonic_ns'],
               'state_skew_s': (state['monotonic_ns'] - timestamp) / 1e9,
               'q_left': state['arms']['left']['position_rad'],
               'q_right': state['arms']['right']['position_rad'],
               'arms': state['arms'],
               'images': {c: {'path': str((path / row['path']).resolve()),
                               'monotonic_ns': row['monotonic_ns'],
                               'skew_s': (row['monotonic_ns'] - timestamp) / 1e9,
                               'image_size_wh': [row['width'], row['height']]}
                          for c, row in selected.items()},
               'frame_timestamps_ns': {c: row['monotonic_ns'] for c, row in selected.items()},
               'camera_names': list(camera_names),
               'camera_mapping': dict(zip(('overhead', 'left_wrist', 'right_wrist'), camera_names))}


def summarize_predictions(path):
    """Report visibility gates without relaxing any runtime quality threshold."""
    import numpy as np
    rows = read_jsonl(path)
    report = {'samples': len(rows), 'thresholds': {'min_visibility': .85, 'min_confidence': .5,
                                                  'max_pixel_std': 8., 'max_error_px': 60.},
              'quantile_levels': [0, .25, .5, .75, .95, 1], 'tasks': {}}
    if not rows:
        return report
    for task in CHECKPOINTS:
        predictions = [row['predictions'][task] for row in rows]
        names = predictions[0]['landmark_names']
        fields = {key: np.asarray([pred[key] for pred in predictions])
                  for key in ('visibility_probability', 'confidence', 'pixel_std', 'keypoints_2d')}
        quality = ((fields['visibility_probability'] >= .85) & (fields['confidence'] >= .5)
                   & (fields['pixel_std'] <= 8.))
        task_report = {'by_camera': {}}
        for camera_index, camera in enumerate(predictions[0]['camera_names']):
            task_report['by_camera'][camera] = {name: {
                field: np.quantile(fields[field][:, camera_index, landmark_index],
                                    report['quantile_levels']).tolist()
                for field in ('visibility_probability', 'confidence', 'pixel_std')}
                for landmark_index, name in enumerate(names)}
        moving = names.index('gear_center' if task == 'gear' else 'pin_tip')
        wrist_indices = [predictions[0]['camera_names'].index(c) for c in ('left_wrist', 'right_wrist')]
        moving_visible = (fields['visibility_probability'][:, wrist_indices, moving] >= .5).all(axis=1)
        moving_quality = quality[:, wrist_indices, moving].all(axis=1)
        pair_quality = []
        close = []
        for index in wrist_indices:
            targets = [names.index('slot_center')] if task == 'gear' else [names.index(f'carrier_hole_{n}') for n in range(1, 5)]
            delta = fields['keypoints_2d'][:, index, targets] - fields['keypoints_2d'][:, index, moving][:, None]
            target_quality = quality[:, index, targets]
            pair_quality.append(quality[:, index, moving] & target_quality.any(axis=1))
            close.append(quality[:, index, moving] & (target_quality & (np.linalg.norm(delta, axis=-1) <= 60.)).any(axis=1))
        task_report.update(moving_visible_both_wrists_at_0_5=int(moving_visible.sum()),
                           moving_passes_quality_both_wrists=int(moving_quality.sum()),
                           pair_passes_quality_both_wrists=int(np.logical_and.reduce(pair_quality).sum()),
                           pair_passes_quality_and_close_both_wrists=int(np.logical_and.reduce(close).sum()))
        if task == 'pin':
            task_report['target_matching_note'] = 'Any qualifying bore is an optimistic bound; shared physical bore identity and jaw/contact gates are not established.'
        report['tasks'][task] = task_report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/kept')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/alignment-calibration/predictions.jsonl')
    parser.add_argument('--model-root', type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument('--camera-names', nargs=3, default=list(DEFAULT_CAMERA_NAMES),
                        metavar=('OVERHEAD', 'LEFT_WRIST', 'RIGHT_WRIST'))
    parser.add_argument('--fps', type=float, default=3.)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--camera-max-skew-s', type=float, default=.04)
    parser.add_argument('--state-max-skew-s', type=float, default=.01)
    parser.add_argument('--episode', action='append', help='Restrict to named episodes (repeatable)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-non-kept', action='store_true',
                        help='Diagnose complete review/discarded episodes without relabeling or training')
    parser.add_argument('--phase-manifest', type=Path, help='Restrict to union of audited gear/pin intervals')
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be finite and positive')
    if len(set(args.camera_names)) != 3:
        parser.error('camera names must be distinct')
    for bound in (args.camera_max_skew_s, args.state_max_skew_s):
        if not math.isfinite(bound) or bound < 0:
            parser.error('skew limits must be finite and nonnegative')
    paths = sorted(p for p in args.data_root.iterdir() if p.is_dir() and (p / 'metadata.json').is_file()
                   and (not args.episode or p.name in args.episode))
    if not paths or (args.episode and set(args.episode) != {p.name for p in paths}):
        parser.error('requested episodes not found')
    phase_manifest = None
    selections = {}
    if args.phase_manifest:
        phase_manifest = json.loads(args.phase_manifest.read_text())
        selections = {row['id']: {task: row[task] for task in CHECKPOINTS if task in row}
                      for row in phase_manifest['episodes']}
        if len(selections) != len(phase_manifest['episodes']) or not {p.name for p in paths} <= selections.keys():
            parser.error('phase manifest must identify every selected episode exactly once')
        if phase_manifest.get('time_origin') != 'metadata.start_monotonic_ns':
            parser.error('phase manifest time origin must be metadata.start_monotonic_ns')
        for selection in selections.values():
            for bounds in selection.values():
                start_s, end_s = (float(bounds[key]) for key in ('start_s', 'end_s'))
                if not all(math.isfinite(v) for v in (start_s, end_s)) or not 0 <= start_s <= end_s:
                    parser.error('invalid phase interval')
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(max(1, args.cpu_threads))
    sys.path.insert(0, str(args.model_root.resolve()))
    from tools.gear_alignment_model import GearAlignmentPredictor
    from tools.pin_perception_model import PinPerceptionPredictor
    checkpoints = {task: {'path': str((args.model_root / path).resolve()),
                          'sha256': sha256(args.model_root / path)} for task, path in CHECKPOINTS.items()}
    models = {'gear': GearAlignmentPredictor(checkpoints['gear']['path'], device=args.device),
              'pin': PinPerceptionPredictor(checkpoints['pin']['path'], device=args.device)}
    manifest_path = args.output.with_suffix('.manifest.json')
    identity = {'schema_version': 1, 'fps': args.fps,
                'camera_mapping': dict(zip(('overhead', 'left_wrist', 'right_wrist'), args.camera_names)),
                'camera_max_skew_s': args.camera_max_skew_s, 'state_max_skew_s': args.state_max_skew_s,
                'checkpoints': checkpoints,
                'phase_selection': ({'path': str(args.phase_manifest.resolve()), 'sha256': sha256(args.phase_manifest)}
                                    if args.phase_manifest else None),
                'sources': {p.name: {name: sha256(p / name)
                                     for name in ('metadata.json', 'states.jsonl', 'frames.jsonl')}
                            for p in paths}}
    if args.allow_non_kept:
        identity['diagnostic_non_kept_allowed'] = True
    done = set()
    counts, rejected = Counter(), Counter()
    if args.output.exists():
        if not args.resume:
            parser.error('output exists; use --resume or choose a new output')
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            parser.error('resume identity mismatch: parameters, source files or checkpoint changed')
        for row in read_jsonl(args.output):
            done.add((row['episode_id'], row['monotonic_ns']))
            counts[row['episode_id']] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(identity, complete=False, device=str(models['gear'].device),
                    sampling_scope='audited_phase_union' if args.phase_manifest else 'full_recordings',
                    torch_version=torch.__version__, source_timing='Host monotonic receive; not hardware synchronized',
                    camera_mapping_verified=False, original_recordings_modified=False,
                    camera_mapping_semantics='Model slot to recorded stream; not physical arm identity',
                    checkpoint_real_input_camera_order={task: model.metadata.get('real_input_camera_order')
                                                       for task, model in models.items()},
                    physical_camera_arm_mapping=None,
                    checkpoint_contracts={task: {key: model.metadata[key] for key in
                        ('landmark_contract', 'landmark_semantics', 'visible_fraction',
                         'adaptation_supervision', 'real_annotation_landmark_contract',
                         'real_position_supervision_counts', 'optical_limitation')
                        if key in model.metadata} for task, model in models.items()})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    start = time.monotonic()
    with args.output.open('a' if args.resume else 'w') as stream:
        for path in paths:
            for row in episode_samples(path, args.camera_names, args.fps,
                                       args.camera_max_skew_s, args.state_max_skew_s, selections.get(path.name),
                                       allow_non_kept=args.allow_non_kept):
                if 'rejection' in row:
                    rejected[row['episode_id']] += 1
                    continue
                if (row['episode_id'], row['monotonic_ns']) in done:
                    continue
                images = []
                for camera in args.camera_names:
                    info = row['images'][camera]
                    with Image.open(info['path']) as image:
                        if image.size != (640, 480):
                            raise ValueError('CNN calibration requires native 640x480 images')
                        images.append(np.asarray(image.convert('RGB')))
                    info['sha256'] = sha256(info['path'])
                images = np.stack(images)
                xyz = np.asarray([value for side in ('left', 'right')
                                  for value in row['arms'][side]['tool_pose_base']['position_m']], dtype=np.float32)
                if xyz.shape != (6,) or not np.isfinite(xyz).all():
                    raise ValueError('Pin CNN requires finite recorded tool XYZ [6]')
                row.update(status='ready', predictions={}, errors={})
                for task, model in models.items():
                    result = model.predict(images, **({'end_effector_positions': xyz} if task == 'pin' else {}))
                    row['predictions'][task] = {k: result[k].tolist() if hasattr(result[k], 'tolist') else list(result[k])
                                                for k in PREDICTION_KEYS}
                    row['predictions'][task].update(checkpoint=checkpoints[task]['path'],
                                                    checkpoint_id=checkpoints[task]['sha256'], image_size=[640, 480])
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                counts[row['episode_id']] += 1
            print(json.dumps({'episode': path.name, 'samples': counts[path.name],
                              'rejected': rejected[path.name], 'elapsed_s': round(time.monotonic() - start, 1)}), flush=True)
            manifest.update(samples_by_episode=dict(counts), rejected_by_episode=dict(rejected))
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest.update(complete=True, samples=sum(counts.values()), elapsed_s=time.monotonic() - start,
                    output_sha256=sha256(args.output))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    args.output.with_suffix('.diagnostics.json').write_text(json.dumps(summarize_predictions(args.output), indent=2) + '\n')
    print(json.dumps({'complete': True, 'samples': manifest['samples'], 'manifest': str(manifest_path)}), flush=True)


if __name__ == '__main__':
    main()
