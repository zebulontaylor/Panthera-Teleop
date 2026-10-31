#!/usr/bin/env python3
"""Score frozen checkpoints on TRAIN/VAL-only semantic preference pairs.

This small candidate-comparison entry point calls the exact preference loss used
by state_value_train.py. It rejects test pair labels and never updates parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from state_value_train import VisualValue, evaluate_preferences, preference_batch, read_manifest, read_preferences


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--preference-manifest',type=Path,required=True)
    ap.add_argument('--preference-image-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--cpu-threads',type=int,default=4)
    args=ap.parse_args()
    if min(args.batch_size,args.cpu_threads)<1:ap.error('batch size and cpu threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows,episodes=read_manifest(args.manifest)
    prefs=read_preferences(args.preference_manifest,episodes)
    val=[r for r in prefs if r['split']=='val']
    if not val:raise ValueError('no known validation preference pairs')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    conf=ck['config'];model=VisualValue(pretrained=False,backbone=conf.get('backbone','resnet50'))
    model.load_state_dict(ck['model']);model.to(device).eval()
    eval_args=SimpleNamespace(preference_image_root=args.preference_image_root,
        image_size=int(conf.get('image_size',224)),preference_batch_size=args.batch_size)
    summary=evaluate_preferences(model,val,eval_args,device)
    details=[]
    with torch.inference_mode():
        for i,pair in enumerate(val):
            loss=preference_batch(model,[pair],eval_args,device)
            details.append({'pair_id':pair.get('pair_id',str(i)),'episode_id_a':pair.get('episode_id_a',pair.get('episode_id')),
                'episode_id_b':pair.get('episode_id_b',pair.get('episode_id')),'session_id':pair.get('session_id'),
                'split':'val','preference':pair['preference'],'progress_component':pair.get('progress_component'),
                'mean_hinge':float(loss.mean()),'rationale':pair.get('rationale')})
    output={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'primary_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'preference_manifest_sha256':hashlib.sha256(args.preference_manifest.read_bytes()).hexdigest(),
        'selection_scope':'validation-only semantic preference loss; test pair labels are forbidden and no parameters are fit',
        'objective':'exact state_value_train.preference_objective / evaluate_preferences implementation',
        'warning':'Visual pair judgments describe local appearance changes; they are not verified assembly success, retention, or calibrated physical progress.',
        **summary,'details':details}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
