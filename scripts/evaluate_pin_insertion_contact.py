#!/usr/bin/env python3
"""Evaluate audited near-contact and untouched failed-attempt challenge images.

Pixel error vectors are image-space diagnostics, never robot displacement labels.
No fitting, threshold selection or checkpoint selection occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch

from app.pin_insertion_cnn import PinInsertionPredictor
from scripts.adapt_pin_insertion_cnn import source_groups
from scripts.evaluate_pin_insertion_adaptation import distribution, landmark_metrics, validate_test_row


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_heldout(rows, metadata):
    forbidden = source_groups(metadata)
    overlap = {r['episode_id'] for r in rows} & forbidden
    if overlap:
        raise ValueError('Contact evaluation overlaps training/selection: ' + ','.join(sorted(overlap)))
    provenance = metadata.get('adaptation', {}).get('manifest_metadata', {})
    sessions = {e['session_id'] for e in provenance.get('episodes', []) if e['split'] in ('train', 'validation')}
    if {r['session_id'] for r in rows} & sessions:
        raise ValueError('Contact evaluation overlaps training/selection grasp group')


def error_vectors(predictions, targets, valid, wh):
    mask = valid.all(axis=1)
    predicted = (predictions[:, 1] - predictions[:, 0]) * wh
    expected = (targets[:, 1] - targets[:, 0]) * wh
    pn = np.linalg.norm(predicted, axis=1)
    tn = np.linalg.norm(expected, axis=1)
    directional = mask & (pn > 1e-9) & (tn > 1e-9)
    cosine = np.sum(predicted[directional] * expected[directional], axis=1) / (pn[directional] * tn[directional])
    return {'paired_labels': int(mask.sum()),
            'pin_to_hole_vector_error_px': distribution(np.linalg.norm(predicted[mask] - expected[mask], axis=1)),
            'direction_cosine': distribution(np.clip(cosine, -1, 1)),
            'meaning': 'Image-plane pin-to-hole vector only; no metric nudge, surface homography or physical contact established'}


def evaluate_rows(predictor, rows):
    check_heldout(rows, predictor.metadata)
    prepared = []; predictions = []; records = []
    for source in rows:
        r = dict(source)
        r['image_width'] = r.get('image_width', r.get('width'))
        r['image_height'] = r.get('image_height', r.get('height'))
        r['uncertainty_px'] = [float(x or 0) for x in r['uncertainty_px']]
        prepared.append(validate_test_row(r))
        if sha256(r['image_path']) != r['image_sha256']:
            raise ValueError('Reviewed image has changed')
        with Image.open(r['image_path']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        if rgb.shape[:2] != (r['image_height'], r['image_width']):
            raise ValueError('Annotated dimensions disagree with source')
        out = predictor.predict(rgb, r['state14'])
        kp = np.asarray(out['keypoints']).reshape(2, 2)
        if not np.isfinite(kp).all():
            raise ValueError('Nonfinite prediction')
        predictions.append(kp)
        records.append({'id': r['id'], 'episode_id': r['episode_id'],
                        'prediction_px': (kp * [r['image_width'], r['image_height']]).reshape(4).tolist(),
                        'labels_px': r['keypoints_px'], 'valid': r['valid'],
                        'contact_status': r.get('contact_status'),
                        'hole_visibility': r.get('hole_visibility', r.get('target_visibility')),
                        'gate_probability': float(1 / (1 + np.exp(-np.clip(np.asarray(out['active_logit']).item(), -60, 60))))})
    targets, valid, wh, uncertainty = map(np.stack, zip(*prepared))
    predictions = np.stack(predictions)
    result = {'frames': len(rows), 'episodes': len({r['episode_id'] for r in rows}),
              'landmarks': landmark_metrics(predictions, targets, valid, wh, uncertainty),
              'image_error_vector': error_vectors(predictions, targets, valid, wh),
              'predictions': records}
    for held_object in sorted({r.get('held_object', 'unspecified') for r in rows}):
        indices = [i for i, r in enumerate(rows) if r.get('held_object', 'unspecified') == held_object]
        result.setdefault('by_held_object', {})[held_object] = {
            'frames': len(indices),
            'episodes': len({rows[i]['episode_id'] for i in indices}),
            'landmarks': landmark_metrics(predictions[indices], targets[indices], valid[indices], wh[indices], uncertainty[indices]),
            'image_error_vector': error_vectors(predictions[indices], targets[indices], valid[indices], wh[indices])}
    for visibility in ('hidden', 'fully_hidden', 'crescent', 'mostly_visible'):
        indices = [i for i, r in enumerate(rows) if r.get('hole_visibility', r.get('target_visibility')) == visibility]
        if indices:
            result.setdefault('by_hole_visibility', {})[visibility] = {
                'frames': len(indices),
                'landmarks': landmark_metrics(predictions[indices], targets[indices], valid[indices], wh[indices], uncertainty[indices])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--challenge', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    challenge = json.loads(args.challenge.read_text())
    contact = [r for r in manifest['frames'] if r['split'] == 'test' and r.get('contact_audit')]
    if any(r.get('split') != 'challenge_test' or not r.get('never_use_for_training_or_calibration') for r in challenge['frames']):
        raise ValueError('Challenge requires explicit whole-episode heldout policy')
    report = {'manifest_sha256': sha256(args.manifest), 'challenge_sha256': sha256(args.challenge),
              'selection': 'Checkpoints selected beforehand using validation only',
              'limitations': ['Contact remains visual/likely, not force-confirmed.',
                  'Adjacent frames are correlated; challenge labels cover one failed trial.',
                  'No positive gate labels verified; gate recall and insertion success unmeasured.',
                  'AI manual labels and uncertainty are not human-certified metrology.',
                  'Pixel direction is not a metric robot command.'], 'results': {}}
    for path in args.checkpoints:
        predictor = PinInsertionPredictor(path, args.device)
        report['results'][path.stem] = {'checkpoint': str(path.resolve()), 'sha256': sha256(path),
            'near_contact_test': evaluate_rows(predictor, contact),
            'failed_attempt_challenge': evaluate_rows(predictor, challenge['frames'])}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {k: {x: v[x] for x in ('frames', 'episodes', 'landmarks', 'image_error_vector')} for k, v in d.items() if k.endswith('test') or k.endswith('challenge')} for name, d in report['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
