#!/usr/bin/env python3
"""Truthful within-recording block evaluation for the purposeful pin recording.

Validation can compare candidates. Test evaluation requires a previously frozen
selection artifact. Different blocks from the same recording/grasp are never
reported as independent recordings or sessions. Future insertion references may
support annotations, but never enter predictions or initialize hole memory here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from app.pin_insertion_cnn import PinInsertionPredictor
from scripts.adapt_pin_insertion_cnn import read_manifest,sha256,source_groups,BLOCK_SPLIT_UNITS
from scripts.evaluate_pin_insertion_adaptation import (validate_test_row,landmark_metrics,gate_confusion,train_mean_baseline)


def balanced_score(landmarks):
    errors=[value['error_native_px']['mean'] for value in landmarks.values() if value['error_native_px']['mean'] is not None]
    return float(np.mean(errors)) if errors else None


def check_candidate(metadata,manifest,manifest_hash,baseline_hash):
    adaptation=metadata.get('adaptation',{})
    if adaptation.get('manifest_sha256')!=manifest_hash:raise ValueError('candidate manifest SHA mismatch')
    if adaptation.get('pretrained_sha256')!=baseline_hash:raise ValueError('candidate baseline ancestor SHA mismatch')
    for split,key in (('train','train_episode_ids'),('validation','validation_episode_ids'),('test','reserved_test_episode_ids')):
        expected={r['episode_id'] for r in manifest['frames'] if r['split']==split}
        if set(adaptation.get(key,[]))!=expected:raise ValueError('candidate frozen partition identities mismatch')
    if adaptation.get('test_used_for_training_or_selection') is not False:raise ValueError('test use not explicitly excluded')
    audit=adaptation.get('split_audit',{})
    if audit.get('session_disjoint') is not False or audit.get('independent_recordings')!=1:raise ValueError('candidate does not truthfully declare within-recording holdout')


def validate_selection(selection_path,checkpoint,manifest_path):
    if not selection_path:raise ValueError('test evaluation requires frozen validation-only selection artifact')
    selection=json.loads(Path(selection_path).read_text())
    if selection.get('selected_checkpoint_sha256')!=sha256(checkpoint) or selection.get('manifest_sha256')!=sha256(manifest_path):raise ValueError('selection artifact checkpoint/manifest mismatch')
    if selection.get('selection_split')!='validation' or selection.get('test_used_for_selection') is not False:raise ValueError('selection must use validation only')
    return selection


def evaluate(args):
    if args.device=='cpu':torch.set_num_threads(4)
    if args.split=='test':selection=validate_selection(args.selection,args.checkpoint,args.manifest)
    else:selection=None
    baseline=PinInsertionPredictor(args.baseline_checkpoint,args.device)
    candidate=PinInsertionPredictor(args.checkpoint,args.device)
    manifest=read_manifest(args.manifest,baseline.metadata)
    if manifest.get('split_unit') not in BLOCK_SPLIT_UNITS:raise ValueError('this evaluator requires explicit within-recording block contract')
    manifest_hash=sha256(args.manifest);baseline_hash=sha256(args.baseline_checkpoint)
    baseline_only=Path(args.checkpoint).resolve()==Path(args.baseline_checkpoint).resolve()
    if not baseline_only:check_candidate(candidate.metadata,manifest,manifest_hash,baseline_hash)
    elif args.split=='test':raise ValueError('freeze a selected candidate before test evaluation')
    rows=[r for r in manifest['frames'] if r['split']==args.split]
    if not rows:raise ValueError('selected split is empty')
    # Only selected evaluation labels are inspected; validation mode never opens
    # test images or consumes their coordinate/gate supervision.
    prepared=[validate_test_row(r) for r in rows]
    target=np.stack([x[0] for x in prepared]);valid=np.stack([x[1] for x in prepared]);wh=np.stack([x[2] for x in prepared]);uncertainty=np.stack([x[3] for x in prepared])
    images=[]
    for row,size in zip(rows,wh):
        if sha256(row['image_path'])!=row['image_sha256']:raise ValueError('evaluation image SHA mismatch')
        with Image.open(row['image_path']) as source:image=np.asarray(source.convert('RGB')).copy()
        if tuple(image.shape[1::-1])!=tuple(size.astype(int)):raise ValueError('decoded frame dimensions disagree with labels')
        images.append(image)
    if len({x.shape for x in images})!=1:raise ValueError('ablations require one full-frame image shape')
    permutation=np.random.default_rng(2026).permutation(len(rows));blocks=np.array([r['block_id'] for r in rows])
    results={};predictions={}
    models={'baseline':baseline} if baseline_only else {'baseline':baseline,'candidate':candidate}
    for model_name,predictor in models.items():
        results[model_name]={};predictions[model_name]={}
        for mode in ('normal','black_images','shuffled_images'):
            output=[];probability=[]
            for i,row in enumerate(rows):
                image=images[permutation[i]] if mode=='shuffled_images' else images[i]
                if mode=='black_images':image=np.zeros_like(image)
                prediction=predictor.predict(image,row['state14'])
                kp=np.asarray(prediction['keypoints'],float).reshape(2,2);logit=np.asarray(prediction['active_logit'],float)
                if not np.isfinite(kp).all() or not np.isfinite(logit).all() or logit.size!=1:raise ValueError('nonfinite prediction')
                output.append(kp);probability.append(float(1/(1+np.exp(-np.clip(logit.item(),-60,60)))))
            output=np.stack(output)
            def metrics(mask):
                landmarks=landmark_metrics(output[mask],target[mask],valid[mask],wh[mask],uncertainty[mask])
                return {'landmarks':landmarks,'balanced_landmark_mean_error_px':balanced_score(landmarks),
                    'gate':{str(t):gate_confusion(np.asarray(probability)[mask],[rows[i].get('gate_label') for i in np.flatnonzero(mask)],t) for t in (.5,.8)}}
            results[model_name][mode]=metrics(np.ones(len(rows),bool))
            results[model_name][mode]['per_block']={str(block):metrics(blocks==block) for block in np.unique(blocks)}
            predictions[model_name][mode]=[{'id':r['id'],'block_id':r['block_id'],'source_recording_id':r['source_recording_id'],
                'source_frame_index':r['source_frame_index'],'source_image_monotonic_ns':r.get('source_image_monotonic_ns'),
                'keypoints_normalized':k.reshape(4).tolist(),'active_probability':p} for r,k,p in zip(rows,output,probability)]
    mean,counts=train_mean_baseline(manifest['frames'])
    mean_valid=valid.copy()
    for j,count in enumerate(counts):
        if not count:mean_valid[:,j]=False
    constant_landmarks=landmark_metrics(np.broadcast_to(mean,target.shape),target,mean_valid,wh,uncertainty)
    report={'manifest_sha256':manifest_hash,'evaluation_split':args.split,'split_audit':manifest['_split_audit'],
        'baseline_checkpoint_sha256':baseline_hash,'checkpoint_sha256':sha256(args.checkpoint),'checkpoint':str(Path(args.checkpoint).resolve()),
        'baseline_checkpoint':str(Path(args.baseline_checkpoint).resolve()),'selection_artifact':selection,
        'test_images_or_supervision_used':args.split=='test','test_used_for_selection':False,
        'train_mean_baseline':{'fit_split':'train','train_label_counts':counts,'mean_normalized':mean.tolist(),
            'landmarks':constant_landmarks,'balanced_landmark_mean_error_px':balanced_score(constant_landmarks)},
        'results':results,'predictions':predictions,
        'limitations':['Held-out carrier-pose blocks share one recording and likely the same grasp; not independent-recording/session tests.',
            'Coordinates are AI annotations with stated uncertainty, not human-certified geometric calibration.',
            'Gate metrics use only explicit known annotation labels; they are not force-certified contact, success, or safety probabilities.',
            'Future insertion reference images are annotation provenance only; no reference, timestamp, block ID or labels enter the CNN.',
            'No temporal hole-memory seed or camera calibration is fitted here; video memory requires an explicit early-visible CNN-only protocol.',
            'Old separate-recording tests and the earlier failed-occlusion challenge remain separate evaluation domains.']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'split':args.split,'normal':{k:v['normal'] for k,v in results.items()},'constant_train_mean':report['train_mean_baseline']},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--baseline-checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--selection')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    evaluate(p.parse_args())

if __name__=='__main__':main()
