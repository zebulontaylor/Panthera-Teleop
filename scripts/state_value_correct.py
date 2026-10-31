#!/usr/bin/env python3
"""Frozen-encoder visual value correction using independently reviewed preferences.

Train never encodes test observations. Evaluate the selected bundle separately with
`eval --split test`. Time metadata is provenance only, never an input or target.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
from state_value_train import VisualValue, atomic_save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pairs(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids=set(); episodes={}; sessions={}; observations={}
    for r in rows:
        if r['id'] in ids: raise ValueError('duplicate pair id')
        ids.add(r['id']); split=r['split']
        if split not in ('train','val','test'): raise ValueError('invalid split')
        if isinstance(r['label'],bool) or r['label'] not in (0,1): raise ValueError('label must be binary preference (1 means a better)')
        for key,seen in [('episode_id',episodes),('session_id',sessions)]:
            identity=str(r[key])
            if not identity: raise ValueError('empty identity')
            if identity in seen and seen[identity]!=split: raise ValueError(key+' split leakage')
            seen[identity]=split
        for key in ('t_a','t_b'):
            if key in r and not math.isfinite(float(r[key])): raise ValueError('nonfinite timestamp')
        for side in ('a','b'):
            if len(r[side])!=3: raise ValueError('three camera paths required')
            for p in r[side]:
                if not Path(p).is_absolute() or not Path(p).is_file(): raise ValueError('missing or nonabsolute image: '+str(p))
                p=str(Path(p).resolve())
                if p in observations and observations[p]!=split: raise ValueError('image split leakage')
                observations[p]=split
        if r['a']==r['b']: raise ValueError('identical preference observations')
    return rows


def objective(a,b,labels,margin=.15,hinge_weight=1.):
    sign=labels*2-1
    return F.binary_cross_entropy_with_logits(a-b,labels)+hinge_weight*F.relu(margin-sign*(a.sigmoid()-b.sigmoid())).mean()


def load_model(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=VisualValue(pretrained=False,backbone=ck['config'].get('backbone','resnet50'))
    model.load_state_dict(ck['model']);model.goal_bank=ck.get('goal_bank');model.goal_bank_ids=ck.get('goal_bank_ids',[])
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,ck


def anchors(path,pairs,per_episode):
    if path is None:return []
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    identities={key:{str(r[key]):r['split'] for r in pairs} for key in ('episode_id','session_id')}
    groups=collections.defaultdict(list)
    observations={str(Path(p).resolve()):r['split'] for r in pairs for side in ('a','b') for p in r[side]}
    for r in rows:
        for key in identities:
            identity=str(r.get(key,r['episode_id']))
            if identity in identities[key] and identities[key][identity]!=r['split']:raise ValueError('anchor '+key+' split leakage')
        if r['split'] not in ('train','val','test'):raise ValueError('invalid anchor split')
        for key in identities:identities[key][str(r.get(key,r['episode_id']))]=r['split']
        if r['split'] not in ('train','val') or r.get('source_category')!='kept':continue
        if not math.isfinite(float(r['t'])):raise ValueError('nonfinite anchor timestamp')
        refs=[str((Path(r.get('image_root',path.parent))/p).resolve()) for p in r['images']]
        if len(refs)!=3 or not all(Path(p).is_file() for p in refs):raise ValueError('invalid anchor images')
        for p in refs:
            if p in observations and observations[p]!=r['split']:raise ValueError('anchor image split leakage')
            observations[p]=r['split']
        groups[(r['split'],str(r['episode_id']))].append({'images':refs,'split':r['split'],'t':float(r['t'])})
    selected=[]
    for seq in groups.values():
        seq.sort(key=lambda r:r['t']);n=min(per_episode,len(seq))
        selected.extend(seq[round(i*(len(seq)-1)/max(1,n-1))] for i in range(n))
    return selected


def features(model,observations,size,device,batch_size,cache):
    keys=sorted(set(tuple(x) for x in observations)); fingerprint=hashlib.sha256(json.dumps({'images':[(p,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for x in keys for p in x],'size':size,'teacher':cache.name},sort_keys=True).encode()).hexdigest()
    if cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=False)
        if saved['fingerprint']==fingerprint:return saved['features']
    tf=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]);result={}
    with torch.inference_mode():
        for start in range(0,len(keys),batch_size):
            batch=keys[start:start+batch_size];ims=[]
            for views in batch:
                values=[]
                for p in views:
                    with Image.open(p) as im:values.append(tf(im.convert('RGB')))
                ims.append(torch.stack(values))
            z=model.encode(torch.stack(ims).to(device)).cpu()
            if not torch.isfinite(z).all():raise ValueError('nonfinite embeddings')
            result.update(zip(batch,z))
    atomic_save({'fingerprint':fingerprint,'features':result},cache)
    return result


def tensors(rows,cache,device):
    return (torch.stack([cache[tuple(r['a'])] for r in rows]).to(device),torch.stack([cache[tuple(r['b'])] for r in rows]).to(device),torch.tensor([r['label'] for r in rows],dtype=torch.float32,device=device))


def metrics(head,rows,cache,device,margin,hinge_weight):
    with torch.no_grad():
        za,zb,y=tensors(rows,cache,device);a=head(za).flatten();b=head(zb).flatten();gap=(a.sigmoid()-b.sigmoid())*(2*y-1)
        result={'pairs':len(rows),'loss':float(objective(a,b,y,margin,hinge_weight)),'directional_accuracy':float((gap>0).float().mean()),'mean_signed_margin':float(gap.mean()),'margin_satisfied':float((gap>=margin).float().mean())}
        result['per_pair']=[{'id':r['id'],'event':r.get('event','unspecified'),'value_a':float(a[i].sigmoid()),'value_b':float(b[i].sigmoid()),'signed_margin':float(gap[i])} for i,r in enumerate(rows)]
        result['per_event']={}
        for event in sorted(set(r.get('event','unspecified') for r in rows)):
            mask=torch.tensor([r.get('event','unspecified')==event for r in rows],device=device);g=gap[mask]
            result['per_event'][event]={'pairs':int(mask.sum()),'directional_accuracy':float((g>0).float().mean()),'mean_signed_margin':float(g.mean()),'margin_satisfied':float((g>=margin).float().mean())}
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    for name in ('train','eval'):
        p=sub.add_parser(name);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--margin',type=float,default=.15);p.add_argument('--hinge-weight',type=float,default=1.)
    p=sub.choices['train'];p.add_argument('--teacher-checkpoint',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=19);p.add_argument('--anchors',type=Path);p.add_argument('--anchors-per-episode',type=int,default=12);p.add_argument('--anchor-weight',type=float,default=.2);p.add_argument('--anchor-val-weight',type=float,default=.2)
    p=sub.choices['eval'];p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--split',choices=['train','val','test'],default='test')
    args=ap.parse_args()
    for k in ('margin','hinge_weight','lr','anchor_weight','anchor_val_weight'):
        if hasattr(args,k) and (not math.isfinite(getattr(args,k)) or getattr(args,k)<0):raise ValueError('invalid '+k)
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    rows=read_pairs(args.pairs);device=torch.device(args.device)
    checkpoint=args.teacher_checkpoint if args.command=='train' else args.checkpoint
    model,ck=load_model(checkpoint,device);size=ck['config']['image_size']
    output=args.output if args.command=='train' else args.output.parent;output.mkdir(parents=True,exist_ok=True)
    selected=[r for r in rows if r['split'] in (('train','val') if args.command=='train' else (args.split,))]
    if not selected:raise ValueError('no selected pairs')
    anchor_rows=anchors(args.anchors,rows,args.anchors_per_episode) if args.command=='train' else []
    cache=features(model,[r[s] for r in selected for s in ('a','b')]+[r['images'] for r in anchor_rows],size,device,args.batch_size,output/('features-'+digest(checkpoint)[:16]+'-'+args.command+'.pt'))
    if args.command=='eval':
        report=metrics(model.value,selected,cache,device,args.margin,args.hinge_weight);report.update(checkpoint_sha256=digest(checkpoint),pairs_sha256=digest(args.pairs),split=args.split)
        args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    if args.epochs<1 or args.anchors_per_episode<1:raise ValueError('positive epochs/anchors-per-episode required')
    train=[r for r in selected if r['split']=='train'];val=[r for r in selected if r['split']=='val']
    if not train or not val:raise ValueError('train and val pairs required')
    random.seed(args.seed);torch.manual_seed(args.seed)
    report={'teacher_checkpoint_sha256':digest(checkpoint),'pairs_sha256':digest(args.pairs),'anchors_sha256':digest(args.anchors) if args.anchors else None,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'teacher':{s:metrics(model.value,rs,cache,device,args.margin,args.hinge_weight) for s,rs in [('train',train),('val',val)]},'anchor_counts':{s:sum(r['split']==s for r in anchor_rows) for s in ('train','val')},'selection':'validation preference BCE + sigmoid hinge + weighted validation anchor drift; test observations never encoded','history':[]}
    az={};targets={}
    for split in ('train','val'):
        ar=[r for r in anchor_rows if r['split']==split]
        if ar:
            az[split]=torch.stack([cache[tuple(r['images'])] for r in ar]).to(device)
            with torch.no_grad():targets[split]=model.value(az[split]).sigmoid().detach()
    for p in model.value.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(model.value.parameters(),lr=args.lr,weight_decay=1e-4);best=float('inf')
    for epoch in range(args.epochs):
        random.shuffle(train)
        for start in range(0,len(train),args.batch_size):
            a,b,y=tensors(train[start:start+args.batch_size],cache,device);loss=objective(model.value(a).flatten(),model.value(b).flatten(),y,args.margin,args.hinge_weight)
            if 'train' in az:loss=loss+args.anchor_weight*F.mse_loss(model.value(az['train']).sigmoid(),targets['train'])
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.value.parameters(),5.);opt.step()
        result=metrics(model.value,val,cache,device,args.margin,args.hinge_weight)
        with torch.no_grad():drift=float(F.mse_loss(model.value(az['val']).sigmoid(),targets['val'])) if 'val' in az else 0.
        score=result['loss']+args.anchor_val_weight*drift
        report['history'].append({'epoch':epoch+1,'val':result,'anchor_val_mse':drift,'selection_score':score})
        if score<best:
            best=score;state={k:v for k,v in ck.items() if k not in ('optimizer','scheduler','scaler')};state.update(model=model.state_dict(),epoch=epoch+1,best=best,correction_provenance={k:v for k,v in report.items() if k!='history'},selection_objective=report['selection']);atomic_save(state,output/'best.pt');report['best_epoch']=epoch+1
        (output/'report.json').write_text(json.dumps(report,indent=2))
        if epoch%10==0:print(json.dumps({'epoch':epoch+1,'selection_score':score}),flush=True)
    corrected,_=load_model(output/'best.pt',device)
    report['corrected']={s:metrics(corrected.value,rs,cache,device,args.margin,args.hinge_weight) for s,rs in [('train',train),('val',val)]};(output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
