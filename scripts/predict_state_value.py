#!/usr/bin/env python3
"""Run an inference-only state-value checkpoint on one overhead/wrist/wrist triplet.

This tool does not accept timestamps, state, episode IDs, ownership or labels. The
returned weak-demo progress and discounted-value outputs are uncalibrated model
scores, not task-success probabilities or verified physical progress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from state_value_train import VisualValue


def load_bundle(path):
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    for key in ('model','config'):
        if key not in bundle:raise ValueError(f'checkpoint missing required {key}')
    return bundle


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--images',nargs=3,required=True,metavar=('OVERHEAD','WRIST_PORT2','WRIST_PORT3'))
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    paths=[Path(x) for x in args.images]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'image files missing: {missing}')
    bundle=load_bundle(args.checkpoint);config=bundle['config']
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    size=int(config.get('image_size',224));backbone=config.get('backbone','resnet50')
    model=VisualValue(pretrained=False,backbone=backbone)
    model.load_state_dict(bundle['model']);model.goal_bank=bundle.get('goal_bank');model.goal_bank_ids=bundle.get('goal_bank_ids',[])
    model.to(device).eval()
    tf=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    with torch.inference_mode():
        views=[]
        for p in paths:
            with Image.open(p) as im:views.append(tf(im.convert('RGB')))
        encoded=model.encode(torch.stack(views)[None].to(device))
        progress=float(torch.sigmoid(model.progress(encoded)).flatten()[0].cpu())
        value=float(torch.sigmoid(model.value(encoded)).flatten()[0].cpu())
        goal_proximity=None
        if model.goal_bank is not None and len(model.goal_bank):
            bank=model.goal_bank.to(device);distances=[]
            for goal in bank:
                g=goal[None].expand_as(encoded)
                distances.append(model.distance(torch.cat([encoded,g,g-encoded],-1)).flatten())
            goal_proximity=float((1-torch.stack(distances,-1).clamp(0,1).median(-1).values)[0].cpu())
    result={'scores':{'weak_demo_progress_uncalibrated':progress,'weak_demo_discounted_value_uncalibrated':value,
                      'train_goal_bank_proximity_uncalibrated':goal_proximity},
        'model':{'source_checkpoint_sha256':bundle.get('source_checkpoint_sha256',hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()),
                 'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 'manifest_sha256':bundle.get('manifest_sha256'),'backbone':backbone,'image_size':size,
                 'camera_order':['overhead','wrist_port2','wrist_port3'],'goal_bank_size':len(bundle.get('goal_bank_ids',[]))},
        'images':[str(p) for p in paths],
        'warning':'Scores are uncalibrated weak demonstration-relative model outputs. They are not task-success probabilities, verified assembly progress, or safe-control signals.'}
    encoded_result=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded_result)
    print(encoded_result,end='')


if __name__=='__main__':main()
