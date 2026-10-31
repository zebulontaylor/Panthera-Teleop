#!/usr/bin/env python3
"""Attach insertion phase provenance and action-aligned loss weights to an export.

The table is a sidecar; stock LeRobot trainers must explicitly consume its weights.
No image, measured state, action, or original recording is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def phase_at(phases, time_s):
    # Half-open intervals prevent double attribution at adjacent boundaries.
    return [p for p in phases if p['start_s'] <= time_s < p['end_s']]


def eligible(phase):
    confidence = str(phase.get('confidence', '')).lower()
    if confidence == 'visual':
        confidence = str(phase.get('visual_reviewer_confidence', 'medium')).lower()
    return confidence in ('high', 'medium', 'medium-high')


def frame_label(phases, time_s, multiplier):
    active = phase_at(phases, time_s)
    return dict(gear_insertion=any(p['phase'] == 'gear_insertion' for p in active),
                pin_insertion=any(p['phase'] == 'pin_insertion' for p in active),
                loss_weight=float(multiplier if any(eligible(p) for p in active) else 1.),
                annotation_scope=';'.join(sorted({p.get('scope', 'visual_phase_window') for p in active})),
                confidence=';'.join(sorted({str(p.get('confidence', 'unknown')) for p in active})))


def attach(root, motor_path, review_paths, multiplier=3.):
    assert multiplier > 1
    manifest = read(root / 'processing/manifest.json')
    motor = read(motor_path)
    labels = {e['id']: e for e in motor['episodes']}
    provenance_files = [motor_path] + review_paths
    for path in review_paths:
        for e in read(path):
            assert e['id'] not in labels
            labels[e['id']] = dict(e, status='visual_partial_recovery',
                                  annotation_source=str(path))
    rows, metadata_episodes = [], []
    for e in manifest['episodes']:
        label = labels.get(e['id'], dict(status='no_insertion_retained_or_unlocalized', phases=[]))
        # Motor labels must describe the exact source used for this export.
        for key, digest in label.get('source_sha256', {}).items():
            assert e['source_hashes'][key] == digest, (e['id'], key)
        phases = []
        for p in label.get('phases', []):
            start = max(float(e['start_s']), float(p['start_s']))
            end = min(float(e['end_s']), float(p['end_s']))
            if end <= start:
                continue
            assert p['phase'] in ('gear_insertion', 'pin_insertion')
            phases.append(dict(p, start_s=start, end_s=end,
                clipped_to_retained_interval=(start != p['start_s'] or end != p['end_s']),
                suggested_loss_multiplier=float(multiplier if eligible(p) else 1.),
                contact_verified=False))
        epmeta = dict(id=e['id'], episode_index=e['episode_index'], status=label['status'],
                      phases=phases, phase_abstentions=label.get('phase_abstentions', {}),
                      note='Insertion approach/seating proxies; not verified contact or success.')
        metadata_episodes.append(epmeta)
        e['precision_phases'] = phases
        e['precision_annotation_status'] = label['status']
        source_meta = read(root / f"processing/source_metadata/{e['id']}.json")
        origin = source_meta['start_monotonic_ns']
        alignment = pq.read_table(root / f"processing/alignment/episode_{e['episode_index']:06d}.parquet").to_pylist()
        for a in alignment:
            target_time = (a['action_data_monotonic_ns'] - origin) / 1e9
            observation_time = (a['state_data_monotonic_ns'] - origin) / 1e9
            row = dict(index=a['index'], episode_index=a['episode_index'], frame_index=a['frame_index'],
                       timestamp=a['frame_index'] / manifest['fps'],
                       action_source_time_s=target_time, observation_source_time_s=observation_time,
                       **frame_label(phases, target_time, multiplier))
            rows.append(row)
    assert len(rows) == manifest['totals']['observations']
    assert [r['index'] for r in rows] == list(range(len(rows)))
    table = root / 'meta/precision_weights.parquet'
    pq.write_table(pa.Table.from_pylist(rows), table, compression='zstd')
    counts = dict(episodes=len(metadata_episodes),
                  episodes_with_precision_phases=sum(bool(e['phases']) for e in metadata_episodes),
                  gear_phase_intervals=sum(p['phase'] == 'gear_insertion' for e in metadata_episodes for p in e['phases']),
                  pin_phase_intervals=sum(p['phase'] == 'pin_insertion' for e in metadata_episodes for p in e['phases']),
                  weighted_action_targets=sum(r['loss_weight'] > 1 for r in rows),
                  gear_action_targets=sum(r['gear_insertion'] for r in rows),
                  pin_action_targets=sum(r['pin_insertion'] for r in rows),
                  total_action_targets=len(rows))
    config = dict(version=1, table='meta/precision_weights.parquet', join_key='index',
                  default_loss_weight=1., suggested_precision_multiplier=float(multiplier),
                  weight_time_basis='Exact next-pose action measurement timestamp from alignment; not observation timestamp.',
                  interval_convention='start inclusive, end exclusive, original episode seconds',
                  action_chunk_rule='For H-step chunks, gather H per-target weights within this episode; pad weights using its last target, never cross an episode boundary.',
                  application='Explicit trainer opt-in required: multiply each action-step loss by loss_weight before reduction. Stored weights do not modify a stock trainer automatically.',
                  uncertain_labels='Missing, low-confidence, or ambiguous phases keep weight1. Visual medium-low windows are recorded but not upweighted.',
                  limits=['Motor labels cover final pre-release approach only; they do not exhaust all seating movements.',
                          'Visually reviewed windows can include post-release seating nudges.',
                          'No calibrated force/contact or independently verified success labels.'],
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files},
                  coverage=counts, episodes=metadata_episodes)
    write(root / 'meta/precision.json', config)
    manifest['precision_loss_metadata'] = {k: config[k] for k in ['table', 'join_key', 'default_loss_weight', 'suggested_precision_multiplier', 'weight_time_basis', 'coverage']}
    write(root / 'processing/manifest.json', manifest)
    info = read(root / 'meta/info.json')
    info['precision_loss_metadata'] = dict(schema='meta/precision.json', table=config['table'], join_key='index')
    write(root / 'meta/info.json', info)
    # Read back every stored label against the exact action timestamp and intervals.
    stored = pq.read_table(table).to_pylist()
    assert stored == rows
    by_ep = {e['episode_index']: e['phases'] for e in metadata_episodes}
    for row in stored:
        expected = frame_label(by_ep[row['episode_index']], row['action_source_time_s'], multiplier)
        assert all(row[k] == v for k, v in expected.items())
    write(root / 'processing/precision_verification.json', dict(passed=True,
          checked_action_targets=len(stored), exact_action_timestamp_alignment=True,
          retained_interval_clipping=True, source_hashes_verified=True,
          uncertain_phases_not_upweighted=True, **counts))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--motor-phases', type=Path, required=True)
    p.add_argument('--review-phases', nargs='*', type=Path, default=[])
    p.add_argument('--multiplier', type=float, default=3.)
    a = p.parse_args()
    attach(a.root, a.motor_phases, a.review_phases, a.multiplier)
