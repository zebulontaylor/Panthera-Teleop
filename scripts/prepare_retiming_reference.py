#!/usr/bin/env python3
"""Discover local-date kept recordings and audit a numeric 20 Hz reference.

Reads state/frame indexes only; never edits recordings or exports camera data.
The generated annotation is a separate, reproducible postprocessing input.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from hitl_recording import data_timestamp, data_origin
from analyze_training_timing import analyze, derivatives, summarize

CAMERAS = ('overhead', 'wrist_port2', 'wrist_port3')


def read_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def nearest(times, tick):
    right = int(np.searchsorted(times, tick))
    return min((max(0, right - 1), min(len(times) - 1, right)),
               key=lambda i: abs(int(times[i]) - tick))


def pose(row):
    return np.asarray([v for side in ('left', 'right') for v in
                       row['arms'][side]['position_rad'] +
                       [row['arms'][side]['gripper']['position']]], dtype=np.float32)


def prepare(source, local_date, timezone, template, previous):
    spec = json.loads(template.read_text())
    current_manifest = json.loads((previous / 'processing/manifest.json').read_text())
    for key in ('image_crops_xyxy', 'image_resize', 'crop_rationale', 'overhead_crop_audit'):
        if key in current_manifest:
            spec[key] = current_manifest[key]
    spec.update(recording_session_local_date=local_date, recording_timezone=timezone,
                source_kind='operator_kept_new_timing_reference', episodes=[],
                trim_method='Stop at earliest terminal-opening cue or classification-state timestamp; preserve all earlier motion and pauses. Metadata/pedal boundary only; no visual task-success claim.',
                annotation_source='Complete operator-kept recordings selected by created_utc converted to local date; validated source indexes and image availability.')
    velocities, accelerations, episodes, excluded = [], [], [], []
    digest = hashlib.sha256()
    fps = spec['fps']
    period = round(1e9 / fps)
    for path in sorted(source.glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        if datetime.fromisoformat(metadata['created_utc']).astimezone(ZoneInfo(timezone)).date().isoformat() != local_date:
            continue
        if metadata.get('complete') is not True or metadata.get('label') != 'kept':
            excluded.append(dict(id=path.parent.name, reason='Not complete and operator-kept'))
            continue
        if metadata['context']['mode'] != 'independent_gravity_assist':
            raise ValueError(f'Need explicit control-boundary selection: {path.parent.name}')
        states, frames = read_rows(path.with_name('states.jsonl')), read_rows(path.with_name('frames.jsonl'))
        origin = data_origin(metadata)
        times = np.asarray([data_timestamp(s) for s in states], dtype=np.int64)
        if len(states) != metadata['samples'] or not np.all(np.diff(times) > 0):
            raise ValueError(f'Invalid state sequence: {path.parent.name}')
        candidates = [s['monotonic_ns'] for s in states
                      if s.get('foot_pedal', {}).get('interpretation') == 'episode_classification']
        detection = metadata.get('episode_end_detection', {}).get('monotonic_ns')
        if detection is not None:
            candidates.append(detection)
        if not candidates:
            raise ValueError(f'Missing terminal classification boundary: {path.parent.name}')
        # Independent recordings have host and data time equal. Fail if this changes.
        if any(data_timestamp(s) != s['monotonic_ns'] for s in states):
            raise ValueError('Need explicit clock conversion for non-host data clock')
        end_ns = min(candidates)
        keep = (times >= origin) & (times <= end_ns)
        local_states = [s for s, selected in zip(states, keep) if selected]
        local_times = times[keep]
        groups = {c: [f for f in frames if f['camera'] == c and origin <= data_timestamp(f) <= end_ns] for c in CAMERAS}
        camera_times = {c: np.asarray([data_timestamp(f) for f in groups[c]], dtype=np.int64) for c in CAMERAS}
        for camera, group in groups.items():
            if len(group) < 2 or not np.all(np.diff(camera_times[camera]) > 0):
                raise ValueError(f'Invalid camera sequence: {path.parent.name}/{camera}')
            if any(not (path.parent / f['path']).is_file() for f in group):
                raise FileNotFoundError(f'Missing source images: {path.parent.name}/{camera}')
        first = max([int(local_times[0])] + [int(t[0]) for t in camera_times.values()])
        start = origin + math.ceil((first - origin) / period) * period
        stop = min(end_ns, int(local_times[-1])) - period
        ticks = np.arange(start, stop + 1, period, dtype=np.int64)
        if len(ticks) < 2:
            raise ValueError('Insufficient retained reference data')
        poses = np.asarray([pose(local_states[nearest(local_times, int(t))]) for t in np.r_[ticks, ticks[-1] + period]])
        state_error = max(abs(int(local_times[nearest(local_times, int(t))]) - int(t)) / 1e9 for t in np.r_[ticks, ticks[-1] + period])
        camera_error = max(abs(int(ct[nearest(ct, int(t))]) - int(t)) / 1e9 for ct in camera_times.values() for t in ticks)
        if state_error > spec['max_state_alignment_s'] or camera_error > spec['max_camera_alignment_s']:
            raise ValueError(f'Alignment tolerance exceeded: {path.parent.name}: {state_error}, {camera_error}')
        v, a, _ = derivatives(poses[:-1], poses[1:], fps)
        velocities.append(v)
        accelerations.append(a)
        digest.update(path.parent.name.encode())
        digest.update(poses.tobytes())
        end_s = (end_ns - origin) / 1e9
        spec['episodes'].append(dict(id=path.parent.name, source_category='kept', start_s=0,
            end_s=end_s, original_end_s=(metadata['end_monotonic_ns'] - origin) / 1e9,
            reason='Operator-kept complete new-method reference; exclude terminal classification tail only.',
            annotation_source=spec['annotation_source']))
        episodes.append(dict(id=path.parent.name, frames=len(v), duration_s=len(v)/fps,
            trimmed_tail_s=(metadata['end_monotonic_ns']-end_ns)/1e9,
            max_state_alignment_s=state_error, max_camera_alignment_s=camera_error,
            velocity_rad_s=summarize(v), acceleration_rad_s2=summarize(a)))
    if not episodes:
        raise ValueError('No matching complete kept recordings')
    velocity, acceleration = np.concatenate(velocities), np.concatenate(accelerations)
    new = dict(episodes=len(episodes), frames=len(velocity), fps=fps,
        baseline_transition_duration_s=len(velocity)/fps,
        numeric_pose_sha256=digest.hexdigest(), velocity_rad_s=summarize(velocity),
        acceleration_rad_s2=summarize(acceleration), per_episode=episodes)
    old = analyze(previous)
    # Rounded upper-tail targets preserve the reference's ordinary fast movements.
    def cap(values, axes, step):
        return math.ceil(float(np.percentile(np.abs(values[:, axes]).max(axis=1), 95)) / step) * step
    joints = [0,1,2,3,4,5,7,8,9,10,11,12]
    recommendation = dict(joint_velocity_rad_s=cap(velocity,joints,.05),
        joint_acceleration_rad_s2=cap(acceleration,joints,.5),
        gripper_velocity_rad_s=cap(velocity,[6,13],.1),
        gripper_acceleration_rad_s2=cap(acceleration,[6,13],1.))
    return spec, dict(local_date=local_date, timezone=timezone, new_reference=new,
        previous=old, excluded=excluded, recommended_initial_caps=recommendation,
        recommendation_method='Round new-reference frame-maximum p95 upward (joint v .05, joint a .5, gripper v .1, gripper a 1). Offline dataset-shaping targets only; compare derived export before training.',
        method='Same previous-export nearest measured state, common three-camera availability, regular 20 Hz source-time ticks, next-pose action, per-episode differences; no cross-episode differences.',
        limitations=['New kept labels reflect operator selection; task success/contact boundaries were not visually certified.',
            'Terminal cue/classification trims are metadata based; no earlier contact or pause edits.',
            'Finite-difference acceleration includes quantization and sampling noise and is not physical acceleration.',
            'New cohort contains 28 recordings and may differ in task phases/durations from previous curation.',
            'Camera crops match the current curated manifest (400-square overhead, 480-square wrists); new camera setup has not been visually revalidated.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/kept'))
    parser.add_argument('--local-date', default='2026-10-28')
    parser.add_argument('--timezone', default='America/Chicago')
    parser.add_argument('--template', type=Path, default=Path('config/postprocessing-oct25-new-curated.json'))
    parser.add_argument('--previous', type=Path, default=Path('data/processed/panthera-oct22-onward-curated-20hz'))
    parser.add_argument('--annotations', type=Path, default=Path('config/postprocessing-oct28-retiming-reference.json'))
    parser.add_argument('--report', type=Path, default=Path('docs/retiming-new-cohort-baseline.json'))
    args = parser.parse_args()
    if args.annotations.exists() or args.report.exists():
        raise FileExistsError('Choose fresh outputs to preserve existing reports')
    spec, report = prepare(args.source, args.local_date, args.timezone, args.template, args.previous)
    for path, content in ((args.annotations,spec),(args.report,report)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('recommended_initial_caps','limitations')},indent=2))
    for label, audit in (('new_reference',report['new_reference']),('previous',report['previous'])):
        print(label,json.dumps({key:audit[key] for key in ('episodes','frames','baseline_transition_duration_s','velocity_rad_s','acceleration_rad_s2')}))


if __name__ == '__main__':
    main()
