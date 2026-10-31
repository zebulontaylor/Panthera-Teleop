#!/usr/bin/env python3
"""Validate and assemble AI manual image annotations without touching recordings.

Ambiguity is never a visibility-negative label. Occluded points never supply
amodal coordinates. All coordinates refer to original 640x480 camera images.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

VISIBILITIES = {'visible', 'occluded', 'out_of_frame', 'absent', 'ambiguous', 'not_reviewed'}
CONFIDENCES = {'high', 'medium', 'low'}


def validate_point(point):
    if point.get('visibility') not in VISIBILITIES or point.get('confidence') not in CONFIDENCES:
        raise ValueError('Invalid visibility/confidence')
    xy = point.get('xy_px')
    if point['visibility'] == 'visible':
        if not isinstance(xy, list) or len(xy) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in xy):
            raise ValueError('Visible points require finite native xy coordinates')
        if not (0 <= xy[0] < 640 and 0 <= xy[1] < 480):
            raise ValueError('Point outside native image')
        uncertainty = point.get('uncertainty_px')
        if not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty) or uncertainty <= 0:
            raise ValueError('Visible point requires positive uncertainty')
    elif xy is not None:
        raise ValueError('Hidden/ambiguous/unreviewed points cannot have invented coordinates')


def supervision(point):
    validate_point(point)
    trusted = point['confidence'] in {'high', 'medium'}
    visibility_known = point['visibility'] in {'visible', 'occluded', 'out_of_frame', 'absent'}
    return dict(point, position_supervision=trusted and point['visibility'] == 'visible',
                visibility_supervision=trusted and visibility_known,
                visibility_target=(point['visibility'] == 'visible') if trusted and visibility_known else None)


def assemble(manifest_path, label_paths, output, *, verify_images=True, review_path=None):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    sources = {row['observation_id']: row for row in manifest['observations']}
    if len(sources) != len(manifest['observations']):
        raise ValueError('Duplicate source observation')
    splits = {}
    for source in sources.values():
        ep = source['episode_id']
        if ep in splits and splits[ep] != source['split']:
            raise ValueError('An episode cannot cross training/validation splits')
        splits[ep] = source['split']
    labels, provenance = {}, []
    for path in map(Path, label_paths):
        data = json.loads(path.read_text())
        provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               annotator=data['annotator']))
        for label in data['observations']:
            oid = label['observation_id']
            if oid not in sources or oid in labels:
                raise ValueError(f'Unknown or duplicate label: {oid}')
            for side in ('left', 'right'):
                grasp = label['grasp'][side]
                if grasp['object'] not in {'gear', 'pin', 'carrier', 'empty', 'unknown'} or grasp['confidence'] not in CONFIDENCES:
                    raise ValueError('Invalid grasp label')
            if set(label['views']) != {'wrist_port2', 'wrist_port3'}:
                raise ValueError('Both wrist views must be reviewed')
            required = ({'gear_center', 'slot_center'} if sources[oid]['phase'] == 'gear' else
                        {'pin_tip', 'pin_top', 'target_bore'} if sources[oid]['phase'].startswith('pin') else set())
            for camera, view in label['views'].items():
                if not required <= view['landmarks'].keys():
                    raise ValueError(f'Missing required landmarks for {oid}/{camera}')
                for point in view['landmarks'].values():
                    validate_point(point)
                for point in view.get('carrier_bores', []):
                    validate_point(point)
            labels[oid] = dict(label, annotator=data['annotator'])
    if set(labels) != set(sources):
        raise ValueError(f'Missing {len(set(sources)-set(labels))} observations')
    review = None
    if review_path:
        review_path = Path(review_path)
        review = json.loads(review_path.read_text())
        provenance.append(dict(path=str(review_path.resolve()),
            sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(), annotator=review['reviewer']))
        for override in review.get('overrides', []):
            landmarks = labels[override['observation_id']]['views'][override['camera']]['landmarks']
            if landmarks[override['landmark']] != override['previous']:
                raise ValueError('Review override refers to a changed annotation')
            validate_point(override['replacement'])
            landmarks[override['landmark']] = override['replacement']
    observations, training, counts, grasp_counts = [], [], Counter(), Counter()
    for oid, source in sources.items():
        label = labels[oid]
        for camera, image in source['images'].items():
            if verify_images and hashlib.sha256(Path(image['path']).read_bytes()).hexdigest() != image['sha256']:
                raise ValueError(f'Source image changed: {image["path"]}')
            view = label['views'][camera]
            landmarks = {name: supervision(point) for name, point in view['landmarks'].items()}
            for name, point in landmarks.items():
                counts[f'{name}.{point["visibility"]}'] += 1
            training.append(dict(observation_id=oid, episode_id=source['episode_id'], split=source['split'],
                phase=source['phase'], camera=camera, image=image, landmarks=landmarks,
                carrier_bores=[supervision(point) for point in view.get('carrier_bores', [])],
                bore_association_supervision=label['cross_view_target_identity'] == 'confirmed',
                grasp=label['grasp'], annotation_kind='AI manual visual annotation; not human-certified'))
        for side, grasp in label['grasp'].items():
            grasp_counts[f'{side}.{grasp["object"]}.{grasp["confidence"]}'] += 1
        observations.append(dict(source, labels=label))
    output.mkdir(parents=True, exist_ok=True)
    assembled = dict(schema_version=1, annotation_kind='AI manual visual annotation; not human-certified',
                     manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                     label_provenance=provenance, review=review, observations=observations)
    (output/'labels.json').write_text(json.dumps(assembled, indent=2, allow_nan=False)+'\n')
    (output/'training.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in training))
    summary = dict(observations=len(observations), camera_views=len(training), episodes=len(splits),
                   episode_splits=dict(Counter(splits.values())), landmark_counts=dict(sorted(counts.items())),
                   grasp_counts=dict(sorted(grasp_counts.items())), source_images_verified=verify_images,
                   annotation_kind=assembled['annotation_kind'], automatic_motion_authorized=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(args.manifest, args.labels, args.output, review_path=args.review), indent=2))


if __name__ == '__main__':
    main()
