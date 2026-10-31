#!/usr/bin/env python3
"""Correct fusion/value layers with cached frozen backbone features, never test data."""
import argparse
import copy
import json
from pathlib import Path
import random
import torch
from torch import nn
from torch.nn import functional as F
from state_value_correct import anchors,digest,features,load_model,metrics,objective,read_pairs,tensors
from state_value_train import atomic_save


class BackboneEncoder(nn.Module):
    def __init__(self,model):
        super().__init__();self.backbone=model.backbone;self.feature_dim=model.feature_dim
    def encode(self,x):
        b,v,c,h,w=x.shape
        return self.backbone(x.reshape(b*v,c,h,w)).reshape(b,v*self.feature_dim)


class FusionValue(nn.Module):
    def __init__(self,model):
        super().__init__();self.fuse=model.fuse;self.value=model.value
    def forward(self,x):return self.value(self.fuse(x))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--teacher-checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--anchors',type=Path);ap.add_argument('--anchors-per-episode',type=int,default=12);ap.add_argument('--anchor-weight',type=float,default=2.);ap.add_argument('--anchor-val-weight',type=float,default=.2)
    ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--lr',type=float,default=1e-5);ap.add_argument('--value-lr',type=float,default=1e-4);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--anchor-batch-size',type=int,default=256)
    ap.add_argument('--margin',type=float,default=.15);ap.add_argument('--hinge-weight',type=float,default=1.);ap.add_argument('--seed',type=int,default=19);ap.add_argument('--reset-value-head',action='store_true');ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    if min(args.epochs,args.batch_size,args.anchor_batch_size,args.anchors_per_episode)<1:raise ValueError('positive counts required')
    import math
    for key in ('anchor_weight','anchor_val_weight','lr','value_lr','margin','hinge_weight'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0:raise ValueError('invalid '+key)
    random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device);model,ck=load_model(args.teacher_checkpoint,device);teacher=FusionValue(copy.deepcopy(model)).eval()
    rows=read_pairs(args.pairs);selected=[r for r in rows if r['split'] in ('train','val')];train=[r for r in selected if r['split']=='train'];val=[r for r in selected if r['split']=='val']
    if not train or not val:raise ValueError('train/val pairs required')
    ar=anchors(args.anchors,rows,args.anchors_per_episode);args.output.mkdir(parents=True,exist_ok=True)
    cache=features(BackboneEncoder(model),[r[s] for r in selected for s in ('a','b')]+[r['images'] for r in ar],ck['config']['image_size'],device,args.batch_size,args.output/('backbone-features-'+digest(args.teacher_checkpoint)[:16]+'.pt'))
    head=FusionValue(model)
    report={'teacher_checkpoint_sha256':digest(args.teacher_checkpoint),'pairs_sha256':digest(args.pairs),'anchors_sha256':digest(args.anchors) if args.anchors else None,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'selection':'validation reviewed preference loss + weighted normal validation anchor drift; no test encoding or selection','teacher':{s:metrics(teacher,rs,cache,device,args.margin,args.hinge_weight) for s,rs in [('train',train),('val',val)]},'anchor_counts':{s:sum(r['split']==s for r in ar) for s in ('train','val')},'history':[]}
    az={};targets={}
    with torch.no_grad():
        for split in ('train','val'):
            seq=[r for r in ar if r['split']==split]
            if seq:
                az[split]=torch.stack([cache[tuple(r['images'])] for r in seq]).to(device)
                targets[split]=teacher(az[split]).sigmoid().detach()
    if args.reset_value_head:
        for module in model.value.modules():
            if hasattr(module,'reset_parameters'):module.reset_parameters()
    for p in head.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW([{'params':model.fuse.parameters(),'lr':args.lr},{'params':model.value.parameters(),'lr':args.value_lr}],weight_decay=1e-4)
    best=float('inf')
    for epoch in range(args.epochs):
        random.shuffle(train)
        for start in range(0,len(train),args.batch_size):
            a,b,y=tensors(train[start:start+args.batch_size],cache,device);loss=objective(head(a).flatten(),head(b).flatten(),y,args.margin,args.hinge_weight)
            if 'train' in az:
                ix=torch.randperm(len(az['train']),device=device)[:args.anchor_batch_size]
                loss=loss+args.anchor_weight*F.mse_loss(head(az['train'][ix]).sigmoid(),targets['train'][ix])
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);opt.step()
        result={s:metrics(head,rs,cache,device,args.margin,args.hinge_weight) for s,rs in [('train',train),('val',val)]}
        with torch.no_grad():drift=float(F.mse_loss(head(az['val']).sigmoid(),targets['val'])) if 'val' in az else 0.
        score=result['val']['loss']+args.anchor_val_weight*drift
        entry={'epoch':epoch+1,**result,'anchor_val_mse':drift,'selection_score':score};report['history'].append(entry)
        if score<best:
            best=score;state={k:v for k,v in ck.items() if k not in ('optimizer','scheduler','scaler')};state.update(model=model.state_dict(),epoch=epoch+1,best=best,correction_provenance={k:v for k,v in report.items() if k!='history'},selection_objective=report['selection']);atomic_save(state,args.output/'best.pt');report['best_epoch']=epoch+1;report['corrected']=result;report['best_anchor_val_mse']=drift
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        if epoch%10==0:print(json.dumps(entry),flush=True)

if __name__=='__main__':main()
