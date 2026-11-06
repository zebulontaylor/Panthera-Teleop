#!/usr/bin/env python3
"""Read-only real-video prediction audit. No offline insertion-success inference.

Select held-out episode IDs explicitly. Predictions are unlabeled unless independent
manual image annotations are supplied; recorded actions/anchors are never labels.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def eligible_samples(episode, camera='right_wrist', stride=5, max_skew_ms=100):
    if stride < 1 or not math.isfinite(max_skew_ms) or max_skew_ms < 0:
        raise ValueError('Require positive stride and finite nonnegative skew limit')
    meta = json.loads((episode/'metadata.json').read_text())
    if not meta.get('complete'):
        raise ValueError('Incomplete episode')
    mapping = meta.get('camera_mapping', {})
    if camera not in mapping:
        raise ValueError('Explicit saved camera mapping is required')
    accepted, rejected = [], 0
    for row in read_jsonl(episode/meta.get('samples_file', 'samples.jsonl'))[::stride]:
        state = np.asarray(row['state'], dtype=float)
        cam = row.get('cameras', {}).get(camera)
        if state.shape != (14,) or not np.isfinite(state).all() or cam is None:
            rejected += 1
            continue
        skew = (int(cam['monotonic_ns'])-int(row['monotonic_ns']))/1e6
        if abs(skew) > max_skew_ms:
            rejected += 1
            continue
        if cam.get('source_camera') != mapping[camera]:
            raise ValueError('Per-frame camera source disagrees with metadata')
        accepted.append((row, skew))
    return meta, accepted, rejected


def prediction_stats(rows):
    if not rows:
        return {'samples': 0, 'accuracy': None, 'insertion_success': None}
    delta = np.asarray([r['prediction']['delta_xyz_m'] for r in rows])
    gate = np.asarray([r['active_probability'] for r in rows])
    return {'samples': len(rows), 'accuracy': None, 'insertion_success': None,
            'label_kind': 'unlabeled_real_video',
            'mean_active_probability': float(gate.mean()),
            'active_fraction_at_0_5': float((gate >= .5).mean()),
            'delta_xyz_mean_m': delta.mean(0).tolist(),
            'delta_xyz_std_m': delta.std(0).tolist(),
            'max_camera_sample_skew_ms': max(abs(r['camera_sample_skew_ms']) for r in rows)}


def annotated_error(prediction, landmarks, image_wh):
    """Only compare independently annotated, position-supervised visible landmarks."""
    kp = np.asarray(prediction['keypoints']).reshape(2, 2)
    errors = {}
    for i, name in enumerate(('pin_tip', 'target_bore')):
        label = landmarks.get(name, {})
        if label.get('position_supervision') and label.get('xy_px') is not None:
            errors[name] = float(np.linalg.norm(kp[i]*np.asarray(image_wh)-label['xy_px']))
    return errors


def used_episode_ids(metadata):
    """Include source recordings and both optimization/model-selection groups."""
    keys = {'source_episode_id','source_episode_ids','source_id','source_ids',
            'episode_id','episode_ids','train_groups','val_groups'}
    found = set()
    def visit(value, collect=False):
        if isinstance(value, dict):
            for key, child in value.items(): visit(child, key in keys)
        elif isinstance(value, (list,tuple)):
            for child in value: visit(child, collect)
        elif collect and isinstance(value, (str,int)):
            found.add(str(value))
    visit(metadata)
    for source in metadata.get('dataset_metadata',{}).get('sources',[]):
        if isinstance(source,dict) and source.get('id') is not None: found.add(str(source['id']))
    return found


def assert_disjoint(ids, metadata):
    overlap = sorted(set(ids) & used_episode_ids(metadata))
    if overlap:
        raise ValueError('Evaluation overlaps training/source/model-selection episodes: '+', '.join(overlap))
    return {'evaluation_episode_ids':sorted(set(ids)), 'overlap':[],
            'checked_used_id_count':len(used_episode_ids(metadata))}


def validate_camera(camera, metadata):
    declared = metadata.get('input_camera',metadata.get('camera','right_wrist'))
    dataset = metadata.get('dataset_metadata',{})
    declared = dataset.get('input_camera',dataset.get('camera',declared))
    aliases = {'wrist_port2':'right_wrist','right_wrist':'right_wrist',
               'wrist_port3':'left_wrist','left_wrist':'left_wrist','overhead':'overhead'}
    if camera != aliases.get(declared,declared):
        raise ValueError(f'Checkpoint camera {declared!r} disagrees with requested {camera!r}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=ROOT/'data/pin-alignment-forward')
    p.add_argument('--episodes', nargs='+', required=True, help='Explicit held-out episode IDs')
    p.add_argument('--camera', default='right_wrist', choices=['right_wrist','left_wrist','overhead'])
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--max-skew-ms', type=float, default=100)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--manual-labels', type=Path, help='Independent assembled training.jsonl; evaluates validation split only')
    args = p.parse_args()
    import cv2
    from PIL import Image, ImageDraw
    from app.pin_insertion_cnn import PinInsertionPredictor
    import torch
    if args.device == 'cpu': torch.set_num_threads(4)
    predictor = PinInsertionPredictor(args.checkpoint, device=args.device)
    validate_camera(args.camera,predictor.metadata)
    disjointness = assert_disjoint(args.episodes,predictor.metadata)
    manual_rows = read_jsonl(args.manual_labels) if args.manual_labels else []
    manual_ids = [r['episode_id'] for r in manual_rows if r['split']=='validation' and r['camera']=='wrist_port2']
    manual_disjointness = assert_disjoint(manual_ids,predictor.metadata)
    rows, tiles, episodes = [], [], []
    selected_tile_episodes = set(np.linspace(0,len(args.episodes)-1,min(48,len(args.episodes)),dtype=int).tolist())
    for episode_number, episode_id in enumerate(args.episodes):
        ep = args.data_root/episode_id
        meta, samples, rejected = eligible_samples(ep, args.camera, args.stride, args.max_skew_ms)
        episodes.append({'id': episode_id, 'source_camera': meta['camera_mapping'][args.camera],
                         'label': meta.get('label'), 'accepted': len(samples), 'rejected': rejected})
        tile_budget = max(1,48//min(48,len(args.episodes))) if episode_number in selected_tile_episodes else 0
        tile_indices = set(np.linspace(0,len(samples)-1,min(tile_budget,len(samples)),dtype=int).tolist()) if samples else set()
        cap = cv2.VideoCapture(str(ep/meta['videos'][args.camera]))
        try:
            for sample_number, (sample, skew) in enumerate(samples):
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample['video_frame'])
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode {episode_id} frame {sample["video_frame"]}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pred = predictor.predict(rgb, sample['state'])
                logit = float(np.asarray(pred['active_logit']).reshape(-1)[0])
                prob = 1/(1+math.exp(-max(-700,min(700,logit))))
                row = {'episode_id': episode_id, 'video_frame': sample['video_frame'],
                       'camera_sample_skew_ms': skew, 'prediction': pred,
                       'active_probability': prob, 'label_kind': 'unlabeled_real_video'}
                rows.append(row)
                if sample_number in tile_indices:
                    im = Image.fromarray(rgb).resize((320,240)); draw = ImageDraw.Draw(im)
                    for xy,color in zip(np.asarray(pred['keypoints']).reshape(2,2), ('red','lime')):
                        x,y = xy*np.array([320,240]); draw.ellipse((x-4,y-4,x+4,y+4), outline=color,width=2)
                    draw.rectangle((0,0,320,48),fill='black')
                    draw.text((3,2),f'{episode_id[:8]} frame {sample["video_frame"]} UNLABELED',fill='white')
                    draw.text((3,16),f'gate={prob:.2f} xyz mm='+','.join(f'{v*1000:.1f}' for v in pred['delta_xyz_m']),fill='white')
                    tiles.append(im)
        finally:
            cap.release()
    report = {'checkpoint': str(args.checkpoint.resolve()), 'camera':args.camera,
              'checkpoint_metadata': predictor.metadata,
              'evaluation_scope':'Independent real recording evaluation',
              'disjointness':disjointness, 'manual_disjointness':manual_disjointness,
              'episodes': episodes, 'summary': prediction_stats(rows), 'predictions':rows,
              'limitations':['Recorded forward_complete is not insertion success.',
                            'Forward camera_sample_skew_ms compares camera receipt with sample recording time; measured joint timestamps are unavailable, so actual state age is not guaranteed.',
                            'Manual annotation joins use recorded state receipt timestamps, not device measurement timestamps.',
                            'XYZ and gate have no independent physical labels in these videos.',
                            'Saved camera mapping is used; no calibrated camera-to-robot transform is inferred.']}
    if args.manual_labels:
        errors = []
        for label in manual_rows:
            if label['split'] != 'validation' or label['camera'] != 'wrist_port2':
                continue
            bgr = cv2.imread(label['image']['path'])
            if bgr is None: raise ValueError('Missing manually annotated image')
            states = read_jsonl(ROOT/'data/kept'/label['episode_id']/'states.jsonl')
            timestamp = label['image']['monotonic_ns']
            state = min(states, key=lambda r: abs(r['monotonic_ns']-timestamp))
            skew_ms = (state['monotonic_ns']-timestamp)/1e6
            if abs(skew_ms) > 40:
                errors.append({'observation_id':label['observation_id'], 'status':'state_skew_rejected'})
                continue
            q = []
            for arm in ('left','right'):
                q.extend(state['arms'][arm]['position_rad'])
                q.append(state['arms'][arm]['gripper']['position'])
            pred = predictor.predict(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),q)
            errors.append({'observation_id':label['observation_id'], 'state_record_receipt_skew_ms':skew_ms,
                           'errors_px':annotated_error(pred,label['landmarks'],[bgr.shape[1],bgr.shape[0]])})
        report['manual_annotation_evaluation'] = {
            'provenance':'Independent AI manual visual annotations; not human-certified',
            'split':'validation', 'camera':'wrist_port2', 'observations':errors,
            'landmarks':{name:{'count':len(values), 'mean_error_px':float(np.mean(values)),
                               'median_error_px':float(np.median(values))}
                         for name in ('pin_tip','target_bore')
                         if (values := [r['errors_px'][name] for r in errors if name in r.get('errors_px',{})])}}

    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if tiles:
        cols=4; montage=Image.new('RGB',(cols*320,math.ceil(len(tiles)/cols)*240))
        for i,tile in enumerate(tiles): montage.paste(tile,((i%cols)*320,(i//cols)*240))
        montage.save(args.output.with_suffix('.jpg'))
    print(json.dumps(report['summary'],indent=2))

if __name__ == '__main__': main()
