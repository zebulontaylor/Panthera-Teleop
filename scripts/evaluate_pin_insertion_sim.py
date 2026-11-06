#!/usr/bin/env python3
"""Held-out synthetic geometry audit; no physics rollout or real success estimate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from app.pin_insertion_cnn import PinInsertionPredictor
from scripts.train_pin_insertion_cnn import validate_dataset_metadata


def confusion(probability, active, threshold):
    positive=probability>=threshold
    truth=active.astype(bool)
    tp=int((positive&truth).sum()); fp=int((positive&~truth).sum())
    tn=int((~positive&~truth).sum()); fn=int((~positive&truth).sum())
    return {'samples':len(truth),'true_positive':tp,'false_positive':fp,
        'true_negative':tn,'false_negative':fn,
        'precision':tp/(tp+fp) if tp+fp else None,
        'recall':tp/(tp+fn) if tp+fn else None,
        'accuracy':(tp+tn)/len(truth) if len(truth) else None}


def summarize(pred, keypoints, delta, active, valid, regimes, width, height):
    kp=pred['keypoints']; pdelta=pred['delta_xyz_m']
    probability=1/(1+np.exp(-np.clip(pred['active_logit'].reshape(-1),-60,60)))
    mask=active.reshape(-1).astype(bool)
    pairs=valid.reshape(-1,2,2).min(axis=-1)>0
    error_px=np.linalg.norm((kp-keypoints).reshape(-1,2,2)*[width,height],axis=-1)
    keypoint_metrics={}
    for j,name in enumerate(('pin_tip','target_hole')):
        values=error_px[:,j][pairs[:,j]]
        keypoint_metrics[name]={'valid_samples':len(values),'mean_error_px':float(values.mean()) if len(values) else None,
            'median_error_px':float(np.median(values)) if len(values) else None,
            'p90_error_px':float(np.quantile(values,.9)) if len(values) else None}
    target=delta[mask]; prediction=pdelta[mask]
    norms=np.linalg.norm(prediction,axis=1)
    capped=prediction*np.minimum(1.,.0005/np.maximum(norms,1e-30))[:,None]
    before=np.linalg.norm(target,axis=1)
    def proxy(step):
        after=np.linalg.norm(target-step,axis=1)
        return {'active_samples':len(target),'fraction_strictly_improved':float((after<before-1e-9).mean()) if len(target) else None,
            'mean_residual_before_mm':float(before.mean()*1000) if len(target) else None,
            'mean_residual_after_mm':float(after.mean()*1000) if len(target) else None}
    gate={}
    for threshold in (.5,.8):
        gate[str(threshold)]={'overall':confusion(probability,mask,threshold),
            'per_regime':{str(r):confusion(probability[regimes==r],mask[regimes==r],threshold) for r in np.unique(regimes)}}
    denominator=np.linalg.norm(prediction,axis=1)*before
    cosine_mask=denominator>1e-12
    return {'keypoints_native_pixel_errors':keypoint_metrics,
        'active_delta_mae_per_axis_mm':(np.abs(prediction-target).mean(axis=0)*1000).tolist() if len(target) else None,
        'active_direction_cosine_mean':float(((prediction*target).sum(axis=1)[cosine_mask]/denominator[cosine_mask]).mean()) if cosine_mask.any() else None,
        'active_direction_cosine_valid_samples':int(cosine_mask.sum()),
        'geometric_proxy_ungated_capped_step':proxy(capped),
        'geometric_proxy_gate_0_8_capped_step':proxy(capped*(probability[mask]>=.8)[:,None]),
        'gate_confusion':gate,
        'active_gate_probability_mean':float(probability[mask].mean()) if mask.any() else None}


def evaluate(args):
    if args.device=='cpu': torch.set_num_threads(4)
    predictor=PinInsertionPredictor(args.checkpoint,args.device)
    metadata=predictor.metadata
    digest=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    if metadata.get('dataset_sha256')!=digest: raise ValueError('Dataset SHA256 differs from checkpoint training dataset')
    sidecar=Path(args.dataset).with_suffix('.json')
    dataset_metadata=validate_dataset_metadata(json.loads(sidecar.read_text()))
    if metadata.get('dataset_metadata')!=dataset_metadata: raise ValueError('Dataset sidecar differs from checkpoint metadata')
    with np.load(args.dataset,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    if 'metadata_json' in arrays and json.loads(str(arrays['metadata_json'].item()))!=dataset_metadata:
        raise ValueError('Embedded metadata differs from sidecar')
    groups=arrays.get('group_ids',arrays.get('groups'))
    if groups is None: raise ValueError('Missing group IDs')
    val_groups=metadata.get('val_groups'); train_groups=metadata.get('train_groups')
    if not val_groups or not train_groups or set(val_groups)&set(train_groups): raise ValueError('Invalid checkpoint split groups')
    if set(np.unique(groups).tolist()) != set(val_groups)|set(train_groups): raise ValueError('Checkpoint split groups do not cover dataset groups exactly')
    indices=np.flatnonzero(np.isin(groups,val_groups))
    if not len(indices): raise ValueError('Empty validation group selection')
    images=arrays['images'][indices]; n,h,w,c=images.shape
    state=arrays['state'][indices]
    if state.shape!=(n,14) or not np.isfinite(state).all(): raise ValueError('Invalid measured state14')
    joints=np.concatenate((state[:,:6],state[:,7:13]),axis=1)
    keypoints=arrays['keypoints'][indices]; delta=arrays['delta_xyz_m'][indices]
    active=arrays['active'][indices]
    valid=arrays.get('keypoints_valid',np.ones((len(groups),4)))[indices]
    if valid.shape==(n,2): valid=np.repeat(valid,2,axis=1)
    if valid.shape==(n,): valid=np.repeat(valid[:,None],4,axis=1)
    regimes=arrays.get('regime',np.full(len(groups),'unknown'))[indices]
    if keypoints.shape!=(n,4) or delta.shape!=(n,3) or valid.shape!=(n,4) or active.size!=n: raise ValueError('Invalid target dimensions')
    if any(not np.isfinite(x).all() for x in (keypoints,delta,valid,active)): raise ValueError('Nonfinite targets')
    if (keypoints<0).any() or (keypoints>1).any() or (valid<0).any() or (valid>1).any() or not np.isin(active,[0,1]).all(): raise ValueError('Invalid labels')
    if images.dtype!=np.uint8 or c!=3: raise ValueError('Images must be uint8 NHWC RGB')
    rng=np.random.default_rng(2026); shuffled=rng.permutation(n)
    summaries={}
    for mode in ('normal','black_images','shuffled_images'):
        predictions={k:[] for k in ('keypoints','delta_xyz_m','active_logit')}
        for start in range(0,n,64):
            stop=min(n,start+64)
            rgb=images[shuffled[start:stop]] if mode=='shuffled_images' else images[start:stop]
            x=torch.from_numpy(rgb.copy()).permute(0,3,1,2).float().to(predictor.device)/255
            if mode=='black_images': x.zero_()
            size=metadata.get('image_size',96)
            x=F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)
            q=torch.from_numpy(joints[start:stop].astype(np.float32)).to(predictor.device)
            with torch.inference_mode(): out=predictor.model(x,q)
            for key in predictions: predictions[key].append(out[key].cpu().numpy())
        predictions={key:np.concatenate(value) for key,value in predictions.items()}
        if any(not np.isfinite(value).all() for value in predictions.values()): raise ValueError('Nonfinite model output')
        summaries[mode]=summarize(predictions,keypoints,delta,active,valid,regimes,w,h)
    report={'checkpoint':str(Path(args.checkpoint).resolve()),'dataset':str(Path(args.dataset).resolve()),
        'dataset_sha256':digest,'validation_groups':val_groups,'validation_samples':n,'native_image_width':w,'native_image_height':h,
        'controlled_arm':'left','coordinate_frame':'right_base','results':summaries,
        'limitations':['Synthetic held-out geometric evaluation; no physical rollout, collision, insertion success, or real-world accuracy estimate.',
            'Residual proxy subtracts a capped 0.5mm proposal from a training delta already clipped at 4mm; it is not a true untruncated geometric residual.',
            'Gate 0.8 proxy evaluates isolated frames without runtime persistence, timing, or hysteresis.',
            'Black/shuffled images test dependence on image content; either degradation alone does not establish real-world visual grounding.',
            'Pin-tip supervision may be amodal where the CAD tip center is hidden by its own pin surface; validity semantics come from dataset metadata.']}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'validation_samples':n,'normal':summaries['normal']},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    evaluate(p.parse_args())

if __name__=='__main__': main()
