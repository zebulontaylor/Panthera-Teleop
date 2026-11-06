#!/usr/bin/env python3
"""Merge reviewed contact-stage annotations without changing the frozen approach set.

Labels are supplied independently; this utility never estimates image landmarks
or infers contact. It preserves the original episode/session split and attaches
the actual recorded state nearest each newly annotated source frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(base_path, audit_paths, output, data_root):
    base = json.loads(base_path.read_text())
    rows = {}
    episodes = {r['episode_id']: (r['split'], r['session_id']) for r in base['frames']}
    for r in base['frames']:
        rows[(r['episode_id'], r.get('video_frame', r.get('frame_index')))] = r
    sources = []
    cache = {}
    for path in audit_paths:
        audit = json.loads(path.read_text())
        sources.append({'path': str(path.resolve()), 'sha256': digest(path)})
        for annotation in audit['frames']:
            r = dict(annotation)
            eid = r.get('episode_id', r.get('source_episode_id'))
            frame = r.get('video_frame', r.get('frame_index'))
            if not eid or frame is None:
                raise ValueError('Each audited frame requires episode and frame index')
            key = (eid, frame)
            if eid in episodes:
                split, session = episodes[eid]
                if r['split'] != split:
                    raise ValueError('Audit changes frozen episode split')
            else:
                # New episodes must be explicitly assigned before merging.
                split = r['split']
                session = r.get('session_id', 'full_grasp_' + eid)
                episodes[eid] = (split, session)
            prior = rows.get(key, {})
            r = {**prior, **r, 'id': f'{eid}__{frame:07d}', 'episode_id': eid,
                 'session_id': session, 'split': split, 'video_frame': frame,
                 'contact_audit': True, 'contact_annotation_source': str(path.resolve())}
            image = Path(r['image_path'])
            if digest(image) != r['image_sha256']:
                raise ValueError('Audited source image has changed')
            with Image.open(image) as im:
                r['image_width'], r['image_height'] = im.size
            r['uncertainty_px'] = [float(x or 0) for x in r['uncertainty_px']]
            if 'state14' not in r:
                forward = data_root / 'pin-alignment-forward' / eid
                if forward.exists():
                    if eid not in cache:
                        cache[eid] = [json.loads(x) for x in (forward / 'samples.jsonl').read_text().splitlines() if x.strip()]
                    matches = [s for s in cache[eid] if s['video_frame'] == frame]
                    if len(matches) != 1:
                        raise ValueError('Expected one exact saved sample per video frame')
                    sample = matches[0]
                    r['state14'] = sample['state']
                    r['camera_sample_skew_ms'] = (sample['cameras']['right_wrist']['monotonic_ns'] - sample['monotonic_ns']) / 1e6
                    r['source_episode_path'] = str(forward.resolve())
                else:
                    ep = data_root / 'kept' / eid
                    if eid not in cache:
                        cache[eid] = [json.loads(x) for x in (ep / 'states.jsonl').read_text().splitlines() if x.strip()]
                    sample = min(cache[eid], key=lambda s: abs(s['monotonic_ns'] - r['monotonic_ns']))
                    skew = (r['monotonic_ns'] - sample['monotonic_ns']) / 1e6
                    if abs(skew) > 25:
                        raise ValueError('Full-demo state is more than 25 ms from labeled frame')
                    r['state14'] = []
                    for side in ('left', 'right'):
                        arm = sample['arms'][side]
                        r['state14'] += arm['position_rad'] + [arm['gripper']['position']]
                    r['camera_sample_skew_ms'] = skew
                    r['source_episode_path'] = str(ep.resolve())
            if np.asarray(r['state14']).shape != (14,) or not np.isfinite(r['state14']).all():
                raise ValueError('Native recorded state14 required')
            if r.get('gate_label') == 1:
                raise ValueError('This contact audit has no verified outside-hole positive gate labels')
            rows[key] = r
    result = {**base, 'frames': list(rows.values()), 'purpose': 'contact-stage localization adaptation; physical contact and outside-hole status unverified',
              'base_manifest': {'path': str(base_path.resolve()), 'sha256': digest(base_path)},
              'contact_annotation_sources': sources,
              'landmark_definition': 'Free distal pin contact-face center when evidence-supported. Target: outer carrier pin bore; bore-center pitch radius 17.9625 mm, actual bore diameter approximately 8.34 mm. Amodal temporal hole labels are explicitly flagged.',
              'episodes': [{'episode_id': e, 'split': s, 'session_id': g} for e, (s, g) in episodes.items()]}
    result['summary'] = {}
    for split in ('train', 'validation', 'test'):
        selected = [r for r in result['frames'] if r['split'] == split]
        result['summary'][split] = {'frames': len(selected), 'episodes': len({r['episode_id'] for r in selected}),
            'contact_audit_frames': sum(bool(r.get('contact_audit')) for r in selected),
            'landmark_counts': np.sum([r['valid'] for r in selected], axis=0).tolist(),
            'positive_gates': sum(r.get('gate_label') == 1 for r in selected)}
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result['summary']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--audits', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    a = p.parse_args()
    if a.output.resolve() in {a.base.resolve(), *(p.resolve() for p in a.audits)}:
        raise ValueError('Preserve frozen manifest and independent audit files')
    print(json.dumps(prepare(a.base, a.audits, a.output, a.data_root), indent=2))
