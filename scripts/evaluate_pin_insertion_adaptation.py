#!/usr/bin/env python3
"""Final independent test audit after adaptation checkpoint selection.

Labels can be uncertain and are not human-certified. This evaluates localization
and explicitly annotated gates only; no XYZ, safety, or insertion-success claims.
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
from app.pin_insertion_cnn import PinInsertionPredictor,joint_features
from scripts.adapt_pin_insertion_cnn import sha256,source_groups,read_manifest


def verify_split(manifest,adapted_metadata,baseline_metadata):
    frames=manifest['frames']
    by_split={name:[r for r in frames if r['split']==name] for name in ('train','validation','test')}
    if any(not rows for rows in by_split.values()): raise ValueError('nonempty train/validation/test required')
    episodes={name:{r['episode_id'] for r in rows} for name,rows in by_split.items()}
    sessions={name:{r.get('session_id',r['episode_id']) for r in rows} for name,rows in by_split.items()}
    images={name:{str(Path(r['image_path']).resolve()) for r in rows} for name,rows in by_split.items()}
    for mapping in (episodes,sessions,images):
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            if mapping[a]&mapping[b]: raise ValueError('split episode/session/image leakage')
    adaptation=adapted_metadata.get('adaptation',{})
    expected={'train':'train_episode_ids','validation':'validation_episode_ids','test':'reserved_test_episode_ids'}
    for split,key in expected.items():
        if set(adaptation.get(key,[]))!=episodes[split]: raise ValueError('adaptation split IDs disagree with final manifest')
    if adaptation.get('test_used_for_training_or_selection') is not False: raise ValueError('checkpoint does not explicitly reserve test data')
    forbidden=source_groups(baseline_metadata)|episodes['train']|episodes['validation']
    if episodes['test']&forbidden: raise ValueError('test overlaps optimization/model-selection/pretrained sources')
    return {'episodes':{k:sorted(v) for k,v in episodes.items()},'sessions':{k:sorted(v) for k,v in sessions.items()},
        'frames':{k:len(v) for k,v in by_split.items()},'test_overlap':[],'test_used_for_training_or_selection':False}


def validate_test_row(row):
    joint_features(row.get('state14'))
    wh=np.asarray([row.get('image_width'),row.get('image_height')],dtype=float)
    if not np.isfinite(wh).all() or (wh<16).any(): raise ValueError('invalid test image dimensions')
    valid=row.get('valid')
    if not isinstance(valid,list) or len(valid)!=2 or any(type(x) is not bool for x in valid): raise ValueError('test validity requires two booleans')
    kp=row.get('keypoints_px')
    if kp is None and not any(valid): kp=[None]*4
    if not isinstance(kp,list) or len(kp)!=4: raise ValueError('test requires four nullable coordinates')
    normalized=np.full((2,2),.5)
    for j in range(2):
        if valid[j]:
            xy=np.asarray(kp[j*2:j*2+2],dtype=float)
            if not np.isfinite(xy).all() or (xy<0).any() or (xy>=wh).any(): raise ValueError('test landmark outside full image')
            normalized[j]=xy/wh
    gate=row.get('gate_label')
    if gate is not None and (type(gate) not in (int,float) or gate not in (0,1)): raise ValueError('test gate must be null/zero/one')
    uncertainty=np.asarray(row.get('uncertainty_px',0.),dtype=float)
    if uncertainty.ndim==0: uncertainty=np.repeat(uncertainty,2)
    if uncertainty.shape!=(2,) or not np.isfinite(uncertainty).all() or (uncertainty<0).any(): raise ValueError('invalid test uncertainty')
    return normalized,np.asarray(valid,bool),wh,uncertainty


def gate_confusion(probability,labels,threshold):
    known=np.array([x is not None for x in labels])
    truth=np.array([0 if x is None else x for x in labels],dtype=bool)[known]
    positive=np.asarray(probability)[known]>=threshold
    tp=int((positive&truth).sum());fp=int((positive&~truth).sum());tn=int((~positive&~truth).sum());fn=int((~positive&truth).sum())
    return {'known_labels':int(known.sum()),'unknown_labels':int((~known).sum()),'known_positive_labels':int(truth.sum()),
        'known_negative_labels':int((~truth).sum()),'true_positive':tp,'false_positive':fp,'true_negative':tn,'false_negative':fn,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'accuracy':(tp+tn)/len(truth) if len(truth) else None}


def distribution(values):
    values=np.asarray(values,dtype=float)
    return {'count':len(values),'mean':float(values.mean()) if len(values) else None,
        'median':float(np.median(values)) if len(values) else None,'p90':float(np.quantile(values,.9)) if len(values) else None}


def landmark_metrics(predictions,targets,valid,wh,uncertainty):
    error_norm=np.linalg.norm(predictions-targets,axis=-1)
    error_px=np.linalg.norm((predictions-targets)*wh[:,None,:],axis=-1)
    result={}
    for i,name in enumerate(('pin_tip','target_hole')):
        mask=valid[:,i]
        result[name]={'valid_labels':int(mask.sum()),'invalid_or_unknown_labels':int((~mask).sum()),
            'error_native_px':distribution(error_px[mask,i]),'error_normalized_image_coordinates':distribution(error_norm[mask,i]),
            'annotation_uncertainty_px':distribution(uncertainty[mask,i])}
    return result


def train_mean_baseline(frames):
    values=[[],[]]
    for row in frames:
        if row['split']!='train': continue
        normalized,valid,_,_=validate_test_row(row)
        for j in range(2):
            if valid[j]: values[j].append(normalized[j])
    means=np.array([np.mean(v,axis=0) if v else [.5,.5] for v in values])
    return means,[len(v) for v in values]


def evaluate(args):
    if args.device=='cpu':torch.set_num_threads(4)
    adapted=PinInsertionPredictor(args.checkpoint,args.device)
    baseline=PinInsertionPredictor(args.baseline_checkpoint,args.device)
    adaptation=adapted.metadata.get('adaptation',{})
    if adaptation.get('manifest_sha256')!=sha256(args.manifest): raise ValueError('final manifest SHA differs from adaptation training manifest')
    if adaptation.get('pretrained_sha256')!=sha256(args.baseline_checkpoint): raise ValueError('baseline checkpoint differs from adaptation ancestor')
    manifest=read_manifest(args.manifest,baseline.metadata)
    split_audit=verify_split(manifest,adapted.metadata,baseline.metadata)
    test=[r for r in manifest['frames'] if r['split']=='test']
    prepared=[validate_test_row(r) for r in test]
    targets=np.stack([r[0] for r in prepared]);valid=np.stack([r[1] for r in prepared]);wh=np.stack([r[2] for r in prepared]);uncertainty=np.stack([r[3] for r in prepared])
    images=[]
    for row,size in zip(test,wh):
        with Image.open(row['image_path']) as source:rgb=np.asarray(source.convert('RGB')).copy()
        if tuple(rgb.shape[1::-1])!=tuple(size.astype(int)):raise ValueError('test decoded dimensions disagree with annotation')
        images.append(rgb)
    permutation=np.random.default_rng(2026).permutation(len(test))
    # Different dimensions would make random-image substitution change frame geometry.
    if len({tuple(x.shape) for x in images})!=1: raise ValueError('ablation requires one full-frame test image shape')
    results={};predictions={}
    for name,predictor in (('baseline',baseline),('adapted',adapted)):
        results[name]={};predictions[name]={}
        for mode in ('normal','black_images','shuffled_images'):
            output=[];probability=[]
            for i,row in enumerate(test):
                image=images[permutation[i]] if mode=='shuffled_images' else images[i]
                if mode=='black_images':image=np.zeros_like(image)
                pred=predictor.predict(image,row['state14'])
                kp=np.asarray(pred['keypoints'],dtype=float).reshape(2,2)
                logit=np.asarray(pred['active_logit'],dtype=float)
                if not np.isfinite(kp).all() or not np.isfinite(logit).all() or logit.size!=1:raise ValueError('nonfinite/invalid prediction')
                output.append(kp);probability.append(float(1/(1+np.exp(-np.clip(logit.item(),-60,60)))))
            output=np.stack(output)
            results[name][mode]={'landmarks':landmark_metrics(output,targets,valid,wh,uncertainty),
                'gate':{str(t):gate_confusion(probability,[r.get('gate_label') for r in test],t) for t in (.5,.8)}}
            predictions[name][mode]=[{'id':row['id'],'episode_id':row['episode_id'],'keypoints_normalized':kp.reshape(4).tolist(),
                'active_probability':p} for row,kp,p in zip(test,output,probability)]
    means,counts=train_mean_baseline(manifest['frames'])
    mean_valid=valid.copy()
    for j,count in enumerate(counts):
        if not count:mean_valid[:,j]=False
    report={'checkpoint':str(Path(args.checkpoint).resolve()),'baseline_checkpoint':str(Path(args.baseline_checkpoint).resolve()),
        'manifest_sha256':sha256(args.manifest),'split_audit':split_audit,'test_supervision_used_only_for_evaluation':True,
        'checkpoint_selection':'prior validation only; test metrics do not select checkpoints',
        'annotation_provenance':{k:v for k,v in manifest.items() if k!='frames'},
        'train_mean_baseline':{'fit_split':'train','train_label_counts':counts,'mean_normalized':means.tolist(),
            'landmarks':landmark_metrics(np.broadcast_to(means,targets.shape),targets,mean_valid,wh,uncertainty)},
        'results':results,'predictions':predictions,
        'limitations':['Annotations are not human-certified; reported uncertainty is annotation metadata, not calibrated model confidence.',
            'Gate precision/recall use only explicit known labels; absent positive labels yield no recall estimate.',
            'Test is independent by recorded episode and session/grasp group; no physical insertion-success, XYZ, or safety assessment.',
            'Black/shuffled image ablations probe image dependence and cannot establish physical visual grounding.',
            'No calibration is fitted to validation or test labels.']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'split_audit':split_audit,'normal':{k:v['normal'] for k,v in results.items()}},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--baseline-checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    evaluate(p.parse_args())

if __name__=='__main__':main()
