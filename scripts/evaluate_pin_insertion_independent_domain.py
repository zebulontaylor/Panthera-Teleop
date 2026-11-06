#!/usr/bin/env python3
"""Evaluate a selected block-adapted model on a separate recording domain.

The new training manifest and validation-only selection remain SHA-bound. The
separate evaluation manifest supplies TEST rows only; used recording and grasp
identities are excluded using both checkpoint ancestry and its baseline source.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from app.pin_insertion_cnn import PinInsertionPredictor
from scripts.adapt_pin_insertion_cnn import read_manifest,sha256,source_groups
from scripts.evaluate_pin_insertion_block_model import validate_selection,check_candidate,balanced_score
from scripts.evaluate_pin_insertion_adaptation import validate_test_row,landmark_metrics,gate_confusion,train_mean_baseline


def audit_domain(rows,training,baseline_metadata,candidate_metadata):
    forbidden=source_groups(candidate_metadata)|source_groups(baseline_metadata)
    forbidden.update(training['_split_audit'].get('source_recording_ids',[]))
    sessions={r.get('session_id',r['episode_id']) for r in training['frames'] if r['split'] in ('train','validation')}
    old=baseline_metadata.get('adaptation',{}).get('manifest_metadata',{}).get('episodes',[])
    sessions.update(r.get('session_id',r.get('episode_id')) for r in old if r.get('split') in ('train','validation'))
    episodes=set();source_ids=set();test_sessions=set()
    for row in rows:
        episode=row.get('episode_id');source=row.get('source_recording_id',episode);session=row.get('session_id',episode)
        if not all(isinstance(x,str) and x for x in (episode,source,session)):raise ValueError('explicit evaluation recording/grasp identities required')
        if episode in forbidden or source in forbidden:raise ValueError('independent evaluation overlaps optimization/model-selection recording source')
        if session in sessions:raise ValueError('independent evaluation shares optimization/model-selection session/grasp')
        episodes.add(episode);source_ids.add(source);test_sessions.add(session)
    return {'test_episode_ids':sorted(episodes),'test_source_recording_ids':sorted(source_ids),'test_session_ids':sorted(test_sessions),
        'optimization_recording_overlap':[],'available_grasp_identity_overlap':[],
        'scope':'separate recording domain; checked stored episode/source/session identities against model ancestry'}


def evaluate(args):
    if args.device=='cpu':torch.set_num_threads(4)
    selection=validate_selection(args.selection,args.checkpoint,args.training_manifest)
    baseline=PinInsertionPredictor(args.baseline_checkpoint,args.device);candidate=PinInsertionPredictor(args.checkpoint,args.device)
    training=read_manifest(args.training_manifest,baseline.metadata)
    check_candidate(candidate.metadata,training,sha256(args.training_manifest),sha256(args.baseline_checkpoint))
    evaluation=json.loads(Path(args.evaluation_manifest).read_text())
    rows=[dict(r) for r in evaluation['frames'] if r['split']=='test']
    if not rows:raise ValueError('independent evaluation requires TEST frames')
    audit=audit_domain(rows,training,baseline.metadata,candidate.metadata)
    for row in rows:
        image=Path(row['image_path'])
        if not image.is_absolute():image=Path(args.evaluation_manifest).resolve().parent/image
        row['image_path']=str(image)
    prepared=[validate_test_row(r) for r in rows]
    target=np.stack([r[0] for r in prepared]);valid=np.stack([r[1] for r in prepared]);wh=np.stack([r[2] for r in prepared]);uncertainty=np.stack([r[3] for r in prepared])
    images=[]
    for row,size in zip(rows,wh):
        if row.get('image_sha256') and sha256(row['image_path'])!=row['image_sha256']:raise ValueError('independent image SHA mismatch')
        with Image.open(row['image_path']) as source:rgb=np.asarray(source.convert('RGB')).copy()
        if tuple(rgb.shape[1::-1])!=tuple(size.astype(int)):raise ValueError('independent decoded image dimensions mismatch')
        images.append(rgb)
    if len({r.shape for r in images})!=1:raise ValueError('ablation requires same image dimensions')
    shuffled=np.random.default_rng(2026).permutation(len(rows));results={};predictions={}
    contact=np.array([r.get('contact_audit',False) is True for r in rows])
    for name,predictor in (('baseline',baseline),('candidate',candidate)):
        results[name]={};predictions[name]={}
        for mode in ('normal','black_images','shuffled_images'):
            out=[];probs=[]
            for i,row in enumerate(rows):
                image=images[shuffled[i]] if mode=='shuffled_images' else images[i]
                if mode=='black_images':image=np.zeros_like(image)
                pred=predictor.predict(image,row['state14']);kp=np.asarray(pred['keypoints'],float).reshape(2,2);logit=np.asarray(pred['active_logit'],float)
                if not np.isfinite(kp).all() or not np.isfinite(logit).all() or logit.size!=1:raise ValueError('invalid independent prediction')
                out.append(kp);probs.append(float(1/(1+np.exp(-np.clip(logit.item(),-60,60)))))
            out=np.stack(out)
            def metric(mask):
                points=landmark_metrics(out[mask],target[mask],valid[mask],wh[mask],uncertainty[mask])
                return {'landmarks':points,'balanced_landmark_mean_error_px':balanced_score(points),
                    'gate':{str(t):gate_confusion(np.asarray(probs)[mask],[rows[i].get('gate_label') for i in np.flatnonzero(mask)],t) for t in (.5,.8)}}
            results[name][mode]=metric(np.ones(len(rows),bool));results[name][mode]['strata']={'approach':metric(~contact),'contact':metric(contact)}
            predictions[name][mode]=[{'id':r['id'],'episode_id':r['episode_id'],'keypoints_normalized':k.flatten().tolist(),'active_probability':p} for r,k,p in zip(rows,out,probs)]
    mean,counts=train_mean_baseline(training['frames']);mean_valid=valid.copy()
    for j,count in enumerate(counts):
        if not count:mean_valid[:,j]=False
    report={'training_manifest_sha256':sha256(args.training_manifest),'evaluation_manifest_sha256':sha256(args.evaluation_manifest),
        'selection':selection,'independent_domain_audit':audit,'test_used_for_selection':False,'results':results,'predictions':predictions,
        'constant_new_train_mean':{'counts':counts,'mean_normalized':mean.tolist(),
            'landmarks':landmark_metrics(np.broadcast_to(mean,target.shape),target,mean_valid,wh,uncertainty)},
        'limitations':['This is separate-recording transfer after checkpoint selection; results cannot be used to retune the selected checkpoint.',
            'Coordinate annotations and inferred amodal landmarks carry stated uncertainty and are not human-certified.',
            'Only explicit known gate labels are scored; unknown positives do not become negative labels.',
            'No contact force, insertion success, motor action, or metric XYZ is evaluated.']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'audit':audit,'normal':{k:v['normal'] for k,v in results.items()}},indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','baseline-checkpoint','training-manifest','evaluation-manifest','selection','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');evaluate(p.parse_args())

if __name__=='__main__':main()
