#!/usr/bin/env python3
"""Selected model versus its baseline on the frozen three-frame Oct27 challenge."""
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
from scripts.adapt_pin_insertion_cnn import sha256,source_groups
from scripts.evaluate_pin_insertion_block_model import validate_selection
from scripts.evaluate_pin_insertion_adaptation import validate_test_row,landmark_metrics,gate_confusion


def image_plane_directions(predictions,targets,valid,wh,uncertainty):
    records=[]
    for pred,truth,mask,size,u in zip(predictions,targets,valid,wh,uncertainty):
        if not mask.all():continue
        predicted=(pred[1]-pred[0])*size;target=(truth[1]-truth[0])*size
        distance=float(np.linalg.norm(target));norm=float(np.linalg.norm(predicted));bound=float(u.sum())
        cosine=float(np.dot(predicted,target)/(norm*distance)) if norm>1e-8 and distance>1e-8 else None
        reliable=distance>bound
        records.append({'cosine':cosine,'annotated_pin_hole_separation_px':distance,'sum_stated_coordinate_uncertainty_px':bound,
            'separation_exceeds_sum_stated_uncertainty':reliable,
            'annotation_only_direction_cone_half_angle_deg':float(np.rad2deg(np.arcsin(min(1.,bound/distance)))) if reliable else None})
    usable=[r['cosine'] for r in records if r['cosine'] is not None]
    robust=[r['cosine'] for r in records if r['cosine'] is not None and r['separation_exceeds_sum_stated_uncertainty']]
    return {'both_landmarks_valid_count':len(records),'nonzero_direction_count':len(usable),'mean_cosine_all_nondegenerate':float(np.mean(usable)) if usable else None,
        'separation_exceeds_uncertainty_count':len(robust),'mean_cosine_separation_exceeds_uncertainty':float(np.mean(robust)) if robust else None,
        'records':records,'limitations':['Image-plane hole-minus-pin direction only; no metric XYZ, depth, base-frame rotation, or motor authority.',
            'Manual uncertainty estimates are uncalibrated; separation filtering is a conservative diagnostic, not a statistical confidence interval.']}


def evaluate(args):
    torch.set_num_threads(4)
    selection=validate_selection(args.selection,args.checkpoint,args.training_manifest)
    predictors={'baseline':PinInsertionPredictor(args.baseline_checkpoint,args.device),'candidate':PinInsertionPredictor(args.checkpoint,args.device)}
    raw=json.loads(Path(args.labels).read_text());rows=[]
    for original in raw['frames']:
        row=dict(original);row['image_width']=row.get('image_width',row['width']);row['image_height']=row.get('image_height',row['height'])
        if row.get('split')!='challenge_test' or row.get('never_use_for_training_or_calibration') is not True:raise ValueError('requires explicitly reserved challenge labels')
        for predictor in predictors.values():
            if row['episode_id'] in source_groups(predictor.metadata):raise ValueError('challenge overlaps model optimization/selection ancestry')
        if sha256(row['image_path'])!=row['image_sha256']:raise ValueError('challenge image SHA mismatch')
        rows.append(row)
    prepared=[validate_test_row(r) for r in rows]
    targets=np.stack([r[0] for r in prepared]);valid=np.stack([r[1] for r in prepared]);wh=np.stack([r[2] for r in prepared]);uncertainty=np.stack([r[3] for r in prepared])
    results={};predictions={}
    for name,predictor in predictors.items():
        output=[];probability=[]
        for row in rows:
            with Image.open(row['image_path']) as image:rgb=np.asarray(image.convert('RGB')).copy()
            pred=predictor.predict(rgb,row['state14']);output.append(np.asarray(pred['keypoints']).reshape(2,2))
            probability.append(float(1/(1+np.exp(-np.clip(float(pred['active_logit'][0]),-60,60)))))
        output=np.stack(output)
        results[name]={'landmarks':landmark_metrics(output,targets,valid,wh,uncertainty),
            'image_plane_direction':image_plane_directions(output,targets,valid,wh,uncertainty),
            'gate':{str(t):gate_confusion(probability,[r.get('gate_label') for r in rows],t) for t in (.5,.8)}}
        predictions[name]=[{'id':r['id'],'episode_id':r['episode_id'],'frame_index':r['frame_index'],'keypoints_normalized':p.flatten().tolist(),'active_probability':g} for r,p,g in zip(rows,output,probability)]
    report={'selection':selection,'labels_sha256':sha256(args.labels),'domain':'Oct27 failed insertion/occlusion challenge; one recording, three correlated labeled frames',
        'checkpoint_sha256':sha256(args.checkpoint),'baseline_checkpoint_sha256':sha256(args.baseline_checkpoint),'test_used_for_selection':False,
        'results':results,'predictions':predictions,'limitations':['No tuning, calibration, or seed choice uses these challenge labels.',
            'Hole labels can be inferred amodal references with stated uncertainty; no independent contact force or success labels.',
            'This report evaluates current CNN predictions; temporal memory is separately assessed under a fixed causal protocol.']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','baseline-checkpoint','training-manifest','selection','labels','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');evaluate(p.parse_args())

if __name__=='__main__':main()
