#!/usr/bin/env python3
"""End-to-end multi-view trajectory representation and WEAK demonstration progress value.

Frame JSONL: episode_id, session_id, split, t, images (three paths), episode_end_s,
source_category and/or weak_demo. All TRAIN trajectories train representation/temporal
objectives. Only kept weak demonstrations train progress/value; curation is never success.
No timestamp, state, episode ID or curation is input to the visual network.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from PIL import Image


def read_manifest(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    episodes=collections.defaultdict(list); sessions={}; splits={}
    for r in rows:
        ep=str(r['episode_id']); sp=r['split']; se=str(r.get('session_id',ep))
        if sp not in ('train','val','test'): raise ValueError(f'invalid split {sp}')
        if ep in splits and splits[ep]!=sp: raise ValueError(f'episode leakage: {ep}')
        if se in sessions and sessions[se]!=sp: raise ValueError(f'session leakage: {se}')
        splits[ep]=sp; sessions[se]=sp
        if len(r['images'])!=3: raise ValueError('exactly three camera images required')
        episodes[ep].append(r)
    for seq in episodes.values():
        seq.sort(key=lambda r:float(r['t']))
        start=float(seq[0].get('episode_start_s',0)); end=float(seq[-1].get('episode_end_s',seq[-1]['t']))
        if end<=start: raise ValueError('nonpositive episode duration')
        weak=bool(seq[0].get('weak_demo',seq[0].get('source_category')=='kept'))
        for r in seq:
            if bool(r.get('weak_demo',r.get('source_category')=='kept'))!=weak:raise ValueError('inconsistent episode weak_demo')
            if r.get('source_category')!=seq[0].get('source_category'):raise ValueError('inconsistent episode source_category')
            if r.get('episode_end_s') is not None and abs(float(r['episode_end_s'])-end)>1e-6:raise ValueError('inconsistent episode end time')
        for r in seq:
            label=r.get('terminal_outcome')
            if label is not None:
                if label not in (0,1) or r is not seq[-1]:raise ValueError('terminal_outcome requires binary label on exact last observed frame only')
                if 'terminal_outcome_confidence' not in r or not 0<float(r['terminal_outcome_confidence'])<=1:raise ValueError('terminal_outcome requires confidence in (0,1]')
            r['_phase']=min(1.,max(0.,(float(r['t'])-start)/(end-start)));r['_remaining']=max(0.,end-float(r['t']));r['_weak']=weak
    return rows,episodes


class Trajectories(Dataset):
    def __init__(self, episodes, root, split, length, size, seed=19):
        self.seqs=[v for v in episodes.values() if v[0]['split']==split and len(v)>1]
        self.root=Path(root); self.length=length; self.split=split; self.seed=seed
        self.reviewed_terminal=[v for v in self.seqs if v[-1].get('terminal_outcome') in (0,1)]
        # Modest crops retain scene geometry; no horizontal flips or camera permutation.
        pre=[T.Resize((size,size))]
        if split=='train': pre += [T.RandomResizedCrop(size,scale=(.85,1.),ratio=(.95,1.05)),T.ColorJitter(.15,.15,.12,.025)]
        self.tf=T.Compose(pre+[T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self): return self.length if self.seqs else 0
    def images(self,r):
        out=[]
        for ref in r['images']:
            with Image.open(Path(r.get('image_root',self.root))/ref) as im: out.append(self.tf(im.convert('RGB')))
        return torch.stack(out)
    def __getitem__(self,index):
        rng=random if self.split=='train' else random.Random(self.seed+index)
        # Adjacent training samples share an episode, providing within-episode negatives
        # that reduce the incentive to solve instance SSL using episode identity alone.
        chooser=random.Random(torch.initial_seed()+index//2) if self.split=='train' else rng
        seq=self.seqs[chooser.randrange(len(self.seqs))];i=rng.randrange(len(seq))
        if self.split=='train' and self.reviewed_terminal and rng.random()<.25:
            seq=self.reviewed_terminal[rng.randrange(len(self.reviewed_terminal))];i=len(seq)-1
        j=rng.randrange(len(seq));a,b=seq[i],seq[j]
        # Every anchor receives two independent visual augmentations for instance SSL.
        return {'a':self.images(a),'aug':self.images(a),'b':self.images(b),
                'terminal_outcome':torch.tensor([a.get('terminal_outcome',0),b.get('terminal_outcome',0)],dtype=torch.float32),
                'terminal_confidence':torch.tensor([a.get('terminal_outcome_confidence',0) if a.get('terminal_outcome') in (0,1) else 0,b.get('terminal_outcome_confidence',0) if b.get('terminal_outcome') in (0,1) else 0],dtype=torch.float32),
                'delta':torch.tensor(b['_phase']-a['_phase'],dtype=torch.float32),
                'phase':torch.tensor([a['_phase'],b['_phase']],dtype=torch.float32),
                'remaining':torch.tensor([a['_remaining'],b['_remaining']],dtype=torch.float32),
                'anchor_t':torch.tensor(float(a['t']),dtype=torch.float32),
                'episode_key':torch.tensor(int(hashlib.sha256(str(a['episode_id']).encode()).hexdigest()[:15],16)),
                'weak':torch.tensor(float(a['_weak']))}


class VisualValue(nn.Module):
    def __init__(self,pretrained=True,backbone='resnet50'):
        super().__init__();self.backbone_name=backbone;self.feature_dim=2048 if backbone=='resnet50' else 512
        factory,weights=(resnet50,ResNet50_Weights.IMAGENET1K_V2) if backbone=='resnet50' else (resnet18,ResNet18_Weights.IMAGENET1K_V1)
        self.backbone=factory(weights=weights if pretrained else None);self.backbone.fc=nn.Identity();self.goal_bank=None;self.goal_bank_ids=[]
        self.fuse=nn.Sequential(nn.Linear(3*self.feature_dim,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,256))
        self.project=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,128))
        self.distance=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,1),nn.Tanh())
        self.progress=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
        self.value=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
    def encode(self,x):
        b,v,c,h,w=x.shape
        return self.fuse(self.backbone(x.reshape(b*v,c,h,w)).reshape(b,v*self.feature_dim))
    def forward(self,a,aug,b):
        # Single pass minimizes dispatch overhead. All backbone weights are trained.
        n=len(a);z=self.encode(torch.cat([a,aug,b]));za,zaug,zb=z.split(n)
        return za,zaug,zb,self.distance(torch.cat([za,zb,zb-za],-1)).squeeze(-1),torch.stack([self.progress(za).squeeze(-1),self.progress(zb).squeeze(-1)],-1),torch.stack([self.value(za).squeeze(-1),self.value(zb).squeeze(-1)],-1)


def losses(model,batch,args):
    za,zaug,zb,d,p,v=model(batch['a'],batch['aug'],batch['b'])
    # Same-frame augmentations are positives. Suppress only near same-episode
    # negatives (<1s); distant within-episode frames remain useful negatives.
    # Cross-episode clock proximity is not meaningful, so it does not define a mask.
    q=F.normalize(model.project(za),dim=-1);k=F.normalize(model.project(zaug),dim=-1)
    logits=q@k.T/args.temperature; labels=torch.arange(len(q),device=q.device)
    same=batch['episode_key'][:,None]==batch['episode_key'][None,:]
    near=(batch['anchor_t'][:,None]-batch['anchor_t'][None,:]).abs()<1.
    logits=logits.masked_fill(same & near & ~torch.eye(len(q),device=q.device,dtype=torch.bool),float('-inf'))
    ssl=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
    temporal=F.smooth_l1_loss(d,batch['delta'])
    mask=batch['weak'];denom=mask.sum().clamp_min(1)
    progress=((torch.sigmoid(p)-batch['phase']).square().mean(-1)*mask).sum()/denom
    target=torch.exp(-batch['remaining']/args.value_horizon)
    terminal_conf=batch.get('terminal_confidence',torch.zeros_like(v))
    # Exact reviewed endpoint overrides its weak clock target, never the other frame.
    value=(((torch.sigmoid(v)-target).square()*(terminal_conf==0)).mean(-1)*mask).sum()/denom
    terminal_loss=(F.binary_cross_entropy_with_logits(v,batch.get('terminal_outcome',torch.zeros_like(v)),reduction='none')*terminal_conf).sum()/terminal_conf.sum().clamp_min(1)
    margin=batch['delta'].abs().clamp(max=.2)
    rank=(F.relu(margin-(torch.sigmoid(p[:,1])-torch.sigmoid(p[:,0]))*batch['delta'].sign())*mask).sum()/denom
    total=args.ssl_weight*ssl+args.temporal_weight*temporal+args.progress_weight*(progress+rank)+args.value_weight*value+getattr(args,'terminal_weight',1.)*terminal_loss
    stats={'loss':total,'ssl':ssl,'temporal':temporal,'weak_progress_mse':progress,'weak_value_mse':value,'weak_ranking':rank,'reviewed_terminal_bce':terminal_loss,'reviewed_terminal_confidence_sum':terminal_conf.sum(),
           'temporal_mae':(d-batch['delta']).abs().mean(),'pair_order_accuracy':((d.sign()==batch['delta'].sign()).float()*(batch['delta']!=0)).sum()/(batch['delta']!=0).sum().clamp_min(1),'weak_pairs':mask.sum()}
    return total,stats


def read_preferences(path,episodes):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    for r in rows:
        split=r.get('split')
        if split not in ('train','val'):raise ValueError('preference training manifest must contain only train/val; test preference rows are forbidden')
        if r.get('preference') not in ('a_better','b_better','equal','unknown'):raise ValueError('invalid preference')
        for side in ('a','b'):
            ep=str(r.get('episode_id_'+side,r.get('episode_id','')))
            if ep not in episodes or episodes[ep][0]['split']!=split:raise ValueError('preference episode/split leakage: '+ep)
            refs=r.get('observation_'+side,{}).get('current',[])
            if isinstance(refs,dict):
                refs=[refs[k] for k in ('overhead','wrist_port2','wrist_port3')];r['observation_'+side]['current']=refs
            if len(refs)!=3:raise ValueError('preference observation.current requires three camera paths')
    rows=[r for r in rows if r['preference']!='unknown']
    if not rows:raise ValueError('no known preference labels remain')
    return rows


def preference_objective(progress_logits,value_logits,labels,margin=.05):
    """Relative visual quality on two heads, without claiming success probabilities."""
    per_head=[]
    for logits in (progress_logits,value_logits):
        difference=torch.sigmoid(logits[:,0])-torch.sigmoid(logits[:,1])
        loss=torch.where(labels==0,F.relu(difference.abs()-margin),F.relu(margin-difference*labels))
        per_head.append(loss)
    return (per_head[0]+per_head[1])/2


def preference_batch(model,rows,args,device,augment=False):
    pre=[T.Resize((args.image_size,args.image_size))]
    if augment:pre+=[T.ColorJitter(.1,.1,.08,.015)]
    tf=T.Compose(pre+[T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]);images=[]
    for r in rows:
        for side in ('a','b'):
            views=[]
            for ref in r['observation_'+side]['current']:
                with Image.open(args.preference_image_root/ref) as im:views.append(tf(im.convert('RGB')))
            images.append(torch.stack(views))
    z=model.encode(torch.stack(images).to(device));p=model.progress(z).reshape(len(rows),2);v=model.value(z).reshape(len(rows),2)
    labels=torch.tensor([{'a_better':1.,'b_better':-1.,'equal':0.}[r['preference']] for r in rows],device=device)
    return preference_objective(p,v,labels)


def evaluate_preferences(model,rows,args,device):
    selected=[r for r in rows if r['split']=='val'];total=0.;model.eval()
    with torch.inference_mode():
        for start in range(0,len(selected),args.preference_batch_size):total+=float(preference_batch(model,selected[start:start+args.preference_batch_size],args,device).sum())
    return {'pairs':len(selected),'mean_hinge':total/len(selected) if selected else 0.,'scope':'All validation preference pairs once; test pairs are forbidden.'}


def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def evaluate_reviewed_terminals(model,episodes,args,device):
    """Deterministically cover every reviewed validation terminal once per epoch."""
    helper=Trajectories(episodes,args.dataset_root,'val',1,args.image_size,args.seed)
    rows=[v[-1] for v in episodes.values() if v[0]['split']=='val' and v[-1].get('terminal_outcome') in (0,1)]
    model.eval();numerator=0.;denominator=0.
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            sub=rows[start:start+args.batch_size];z=model.encode(torch.stack([helper.images(r) for r in sub]).to(device));logits=model.value(z).flatten()
            y=torch.tensor([r['terminal_outcome'] for r in sub],device=device,dtype=torch.float32);w=torch.tensor([r['terminal_outcome_confidence'] for r in sub],device=device,dtype=torch.float32)
            numerator+=float((F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum());denominator+=float(w.sum())
    return {'bce':numerator/denominator if denominator else 0.,'states':len(rows),'confidence_sum':denominator,'negative_states':sum(r['terminal_outcome']==0 for r in rows),'positive_states':sum(r['terminal_outcome']==1 for r in rows)}


def build_goal_bank(model,episodes,args,device):
    """One unverified kept endpoint per TRAIN episode only; never include heldout goals."""
    helper=Trajectories(episodes,args.dataset_root,'val',1,args.image_size,args.seed)
    goals=[v[-1] for v in episodes.values() if v[0]['split']=='train' and v[0]['_weak']]
    model.eval();zs=[]
    with torch.inference_mode():
        for start in range(0,len(goals),args.batch_size):
            zs.append(model.encode(torch.stack([helper.images(r) for r in goals[start:start+args.batch_size]]).to(device)).cpu())
    model.goal_bank=torch.cat(zs) if zs else None;model.goal_bank_ids=[r['episode_id'] for r in goals]


def predict(model,rows,args,device):
    tf=T.Compose([T.Resize((args.image_size,args.image_size)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]);results=[]
    model.eval()
    goal=None
    if args.goal_images:
        ims=[]
        for ref in args.goal_images:
            with Image.open(args.dataset_root/ref) as im:ims.append(tf(im.convert('RGB')))
        with torch.inference_mode():goal=model.encode(torch.stack(ims)[None].to(device))
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            subset=rows[start:start+args.batch_size];images=[]
            for r in subset:
                views=[]
                for ref in r['images']:
                    with Image.open(Path(r.get('image_root',args.dataset_root))/ref) as im: views.append(tf(im.convert('RGB')))
                images.append(torch.stack(views))
            z=model.encode(torch.stack(images).to(device));p=torch.sigmoid(model.progress(z)).flatten().cpu().tolist();v=torch.sigmoid(model.value(z)).flatten().cpu().tolist()
            pooled=None
            if model.goal_bank is not None:
                bank=model.goal_bank.to(device);ds=[]
                for ref in bank:
                    g=ref[None].expand_as(z);ds.append(model.distance(torch.cat([z,g,g-z],-1)).flatten())
                pooled=(1-torch.stack(ds,-1).clamp(0,1).median(-1).values).cpu().tolist()
            dist=None if goal is None else model.distance(torch.cat([z,goal.expand_as(z),goal.expand_as(z)-z],-1)).flatten().cpu().tolist()
            for j,(r,pp,vv) in enumerate(zip(subset,p,v)):results.append({'episode_id':r['episode_id'],'split':r['split'],'t':r['t'],'uncalibrated_train_goal_bank_proximity':None if pooled is None else pooled[j],'goal_bank_size':len(model.goal_bank_ids),'goal_signed_relative_temporal_distance':None if dist is None else dist[j],'weak_demo_progress':pp,'weak_demo_discounted_value':vv,'weak_progress_target':r['_phase'] if r['_weak'] else None,'weak_value_target':math.exp(-r['_remaining']/args.value_horizon) if r['_weak'] else None,'source_category':r.get('source_category'),'terminal_outcome':r.get('terminal_outcome'),'terminal_outcome_confidence':r.get('terminal_outcome_confidence'),'outcome':r.get('outcome',-1),'session_id':r.get('session_id')})
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    return results


def diagnostics(rows,predictions,args):
    """Train-only fitted shortcut baselines; curation AUC is diagnostic, never selection."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    result={'normalized_time_oracle_non_deployable':{'definition':'(t-episode_start)/(recorded_episode_end-episode_start); privileged recording-duration metadata equals the weak target by construction','weak_progress_rmse':0.},'interpretation':'Progress targets are normalized recorded time in kept demonstrations. Time baseline sees elapsed seconds, never episode duration. Curation labels are not success.'}
    split=np.asarray([r['split'] for r in rows]);weak=np.asarray([r['_weak'] for r in rows]);target=np.asarray([r['_phase'] for r in rows]);train=(split=='train')&weak
    def ep_rmse(mask,pred):
        groups=collections.defaultdict(list)
        for i in np.flatnonzero(mask):groups[str(rows[i]['episode_id'])].append((float(pred[i])-target[i])**2)
        return float(np.sqrt(np.mean([np.mean(v) for v in groups.values()]))) if groups else None
    if train.sum():
        result['visual_weak_progress']={sp:ep_rmse((split==sp)&weak,np.asarray([r['weak_demo_progress'] for r in predictions])) for sp in ('train','val','test')}
        features={'elapsed_time':np.asarray([[float(r['t'])] for r in rows])}
        dims={len(r['state']) for r in rows if r.get('state') is not None}
        if len(dims)==1 and all(r.get('state') is not None for r in rows):features['state']=np.asarray([r['state'] for r in rows])
        for name,x in features.items():
            clf=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(x[train],target[train]);pr=clf.predict(x)
            result[name+'_weak_progress']={sp:ep_rmse((split==sp)&weak,pr) for sp in ('train','val','test')}
    # Aggregate final-window predictions once per trajectory. Discarded is solely curation.
    endpoints=collections.defaultdict(list)
    for r,p in zip(rows,predictions):
        if r['_phase']>=.9:endpoints[str(r['episode_id'])].append((r,p))
    result['reviewed_terminal_value_diagnostic']={}
    for sp in ('train','val','test'):
        known=[(r,p) for r,p in zip(rows,predictions) if r['split']==sp and r.get('terminal_outcome') in (0,1)]
        weights=np.asarray([float(r['terminal_outcome_confidence']) for r,p in known]);ys=np.asarray([r['terminal_outcome'] for r,p in known]);ps=np.asarray([p['weak_demo_discounted_value'] for r,p in known])
        result['reviewed_terminal_value_diagnostic'][sp]={'reviewed_terminal_states':len(known),'confidence_weighted_brier':float(np.average((ps-ys)**2,weights=weights)) if known else None,'mean_value':float(ps.mean()) if known else None,'interpretation':'Sparse reviewed terminal outcomes only, mixed with weak value objective; not calibrated success probability.'}
    result['endpoint_curation_diagnostic']={}
    for sp in ('train','val','test'):
        y=[];scores=[]
        for seq in endpoints.values():
            r=seq[0][0]
            if r['split']!=sp or r.get('source_category') not in ('kept','discarded'):continue
            y.append(int(r['source_category']=='kept'));scores.append(float(np.mean([p['weak_demo_discounted_value'] for _,p in seq])))
        result['endpoint_curation_diagnostic'][sp]={'episodes':len(y),'auroc':float(roc_auc_score(y,scores)) if len(set(y))==2 else None}
    (args.output/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')


def trajectory_review(rows,predictions,args):
    """Raw heldout trajectories for human review; do not impose monotonic scores."""
    import html
    grouped=collections.defaultdict(list)
    for r,p in zip(rows,predictions):
        if r['split'] in ('val','test'):grouped[str(r['episode_id'])].append((r,p))
    cards=[];summaries=[]
    for ep,seq in grouped.items():
        seq.sort(key=lambda item:float(item[0]['t']));ts=np.asarray([float(r['t']) for r,_ in seq]);duration=max(1e-6,ts[-1]-ts[0]);lines=[]
        keys=[('weak_demo_progress','#2563eb'),('weak_demo_discounted_value','#db2777'),('uncalibrated_train_goal_bank_proximity','#059669')]
        for key,color in keys:
            if any(p.get(key) is None for _,p in seq):continue
            vs=np.asarray([p[key] for _,p in seq]);coords=' '.join(f'{30+640*(t-ts[0])/duration:.1f},{180-150*v:.1f}' for t,v in zip(ts,vs));lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        vals=np.asarray([p['weak_demo_progress'] for _,p in seq]);changes=np.diff(vals)
        summaries.append({'episode_id':ep,'split':seq[0][0]['split'],'frames':len(seq),'progress_start':float(vals[0]),'progress_end':float(vals[-1]),'negative_score_steps':int((changes<-.01).sum()),'near_flat_score_steps':int((abs(changes)<.005).sum()),'interpretation':'Score changes only; stalls/recoveries need frame review and do not establish physical task progress.'})
        positions=sorted(set([0,len(seq)//2,len(seq)-1,int(np.argmin(changes))+1 if len(changes) else 0]));thumbs=[]
        for i in positions:
            r=seq[i][0];ref=(Path(r.get('image_root',args.dataset_root))/r['images'][0]).resolve().as_uri();thumbs.append(f'<figure><img width="200" src="{html.escape(ref)}"><figcaption>t={r["t"]}s</figcaption></figure>')
        cards.append(f'<section><h2>{html.escape(ep)} ({seq[0][0]["split"]})</h2><svg viewBox="0 0 700 205" width="700"><path d="M30 20V180H680" fill="none" stroke="#888"/>'+''.join(lines)+f'<text x="30" y="200">{ts[0]:.1f}s</text><text x="630" y="200">{ts[-1]:.1f}s</text></svg><div style="display:flex">'+''.join(thumbs)+'</div></section>')
    (args.output/'trajectory_review.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (args.output/'trajectory_review.html').write_text('<!doctype html><meta charset="utf-8"><title>Heldout trajectory review</title><style>body{font:15px sans-serif;max-width:1000px;margin:30px auto}section{border-top:1px solid #ccc}figure{margin:8px}</style><h1>Uncalibrated heldout visual trajectory scores</h1><p>Blue: weak progress; pink: weak discounted value; green: train goal-bank proximity. Raw predictions may fall or stall. Weak time supervision assumes demonstration advancement, but is not verified progress. Inspect the frames to distinguish actual stalls, retries, resets and recovery; no outcome label forces discarded trajectories monotonic.</p>'+''.join(cards))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--dataset-root',type=Path,default=Path('.'));ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--preference-manifest',type=Path);ap.add_argument('--preference-image-root',type=Path,default=Path('.'));ap.add_argument('--preference-weight',type=float,default=.2);ap.add_argument('--preference-batch-size',type=int,default=4)
    ap.add_argument('--backbone',choices=('resnet18','resnet50'),default='resnet50')
    ap.add_argument('--goal-images',nargs=3,help='optional three goal-reference image paths for inference temporal distance');ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--steps-per-epoch',type=int,default=1000);ap.add_argument('--validation-steps',type=int,default=100)
    ap.add_argument('--batch-size',type=int,default=12);ap.add_argument('--image-size',type=int,default=224);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--cpu-threads',type=int,default=8)
    ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--seed',type=int,default=19)
    ap.add_argument('--temperature',type=float,default=.1);ap.add_argument('--value-horizon',type=float,default=30.)
    ap.add_argument('--terminal-weight',type=float,default=1.)
    ap.add_argument('--ssl-weight',type=float,default=.25);ap.add_argument('--temporal-weight',type=float,default=1.);ap.add_argument('--progress-weight',type=float,default=1.);ap.add_argument('--value-weight',type=float,default=1.)
    ap.add_argument('--pretrained',action=argparse.BooleanOptionalAction,default=True);ap.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--gpu-memory-gb',type=float,default=40.);ap.add_argument('--resume',type=Path);ap.add_argument('--reset-selection',action='store_true',help='Resume with a fresh best-score comparison after changing the validation objective');ap.add_argument('--predict',action='store_true');ap.add_argument('--checkpoint',type=Path);ap.add_argument('--export-predictions',action='store_true')
    args=ap.parse_args()
    if min(args.batch_size,args.steps_per_epoch,args.validation_steps,args.epochs)<1:ap.error('batch/steps/epochs must be positive')
    if args.preference_batch_size<1:ap.error('preference batch size must be positive')
    if args.reset_selection and (not args.resume or args.predict):ap.error('--reset-selection requires training with --resume')
    if args.value_horizon<=0:ap.error('value horizon must be positive')
    torch.set_num_threads(args.cpu_threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(min(.95,args.gpu_memory_gb*1024**3/total));torch.backends.cudnn.benchmark=True
    rows,episodes=read_manifest(args.manifest);digest=hashlib.sha256(args.manifest.read_bytes()).hexdigest();args.output.mkdir(parents=True,exist_ok=True)
    preferences=read_preferences(args.preference_manifest,episodes) if args.preference_manifest else []
    training_preferences=[r for r in preferences if r['split']=='train']
    checkpoint=args.checkpoint if args.predict else args.resume;ck=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint else None
    if ck and ck['manifest_sha256']!=digest and not args.predict:raise ValueError('resume manifest changed')
    if ck:
        for name in ('image_size','value_horizon','backbone'):setattr(args,name,ck['config'][name])
    selection_objective={name:getattr(args,name) for name in ('temporal_weight','progress_weight','value_weight','terminal_weight','preference_weight','value_horizon')}
    selection_objective['preference_manifest_sha256']=hashlib.sha256(args.preference_manifest.read_bytes()).hexdigest() if args.preference_manifest else None
    if ck and not args.predict:
        old_objective=ck.get('selection_objective')
        if old_objective is None:
            old_objective={name:ck['config'].get(name,1. if name!='preference_weight' else .2) for name in ('temporal_weight','progress_weight','value_weight','terminal_weight','preference_weight','value_horizon')}
            old_objective['preference_manifest_sha256']=None if not ck['config'].get('preference_manifest') else 'unknown_legacy_manifest_hash'
        if old_objective!=selection_objective and not args.reset_selection:raise ValueError('selection objective changed on resume; use --reset-selection for a new validation comparison')
    model=VisualValue(args.pretrained and not ck,args.backbone).to(device)
    if ck:
        model.load_state_dict(ck['model']);model.goal_bank=ck.get('goal_bank');model.goal_bank_ids=ck.get('goal_bank_ids',[])
    if args.predict:
        if not ck:raise ValueError('--predict requires --checkpoint')
        predictions=predict(model,rows,args,device);trajectory_review(rows,predictions,args);return
    train=Trajectories(episodes,args.dataset_root,'train',args.steps_per_epoch*args.batch_size,args.image_size,args.seed)
    val=Trajectories(episodes,args.dataset_root,'val',args.validation_steps*args.batch_size,args.image_size,args.seed)
    if not len(train) or not len(val):raise ValueError('train and heldout validation trajectories required')
    for sp in ('train','val'):
        if not any(v[0]['split']==sp and v[0]['_weak'] for v in episodes.values()):raise ValueError(f'{sp} requires kept weak demonstration trajectories to train/validate weak value')
    loader_args=dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=False)
    tr=DataLoader(train,**loader_args);vl=DataLoader(val,**loader_args)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda');best=float('inf');start=0
    if ck:
        opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scheduler.T_max=args.epochs;scaler.load_state_dict(ck['scaler']);start=ck['epoch'];best=ck['best']
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch'])
        if device.type=='cuda' and ck.get('rng_cuda'):torch.cuda.set_rng_state_all(ck['rng_cuda'])
    if args.reset_selection:best=float('inf')
    report={'selection_objective':selection_objective,'selection_reset':{'requested':args.reset_selection,'previous_best':ck.get('best') if ck and args.reset_selection else None,'reason':'Fresh validation comparison; previous score is not comparable after objective changes. Compare stage checkpoints on the same heldout validation objective.' if args.reset_selection else None},'preference_supervision':{'manifest_sha256':hashlib.sha256(args.preference_manifest.read_bytes()).hexdigest() if args.preference_manifest else None,'train_pairs':len(training_preferences),'val_pairs':sum(r['split']=='val' for r in preferences),'weight':args.preference_weight,'margin':.05,'scope':'Optional relative visual quality labels train progress/value heads every four optimizer steps; never test pairs.'},'warning':'Weak demonstration-relative progress and discounted demonstration endpoint proximity, NOT verified task success or calibrated control value. Kept endpoints are an explicit unverified goal assumption. Discarded/review trajectories are unlabeled representation data.', 'manifest_sha256':digest,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'split_counts':{s:{'episodes':sum(v[0]['split']==s for v in episodes.values()),'frames':sum(r['split']==s for r in rows),'weak_demo_episodes':sum(v[0]['split']==s and v[0]['_weak'] for v in episodes.values())} for s in ('train','val','test')},'reviewed_terminal_supervision':{'rows':sum(r.get('terminal_outcome') in (0,1) for r in rows),'negative_rows':sum(r.get('terminal_outcome')==0 for r in rows),'positive_rows':sum(r.get('terminal_outcome')==1 for r in rows),'scope':'Exact reviewed final state only; 25% training pair oversampling when available. Never propagate failure backward or force progress to zero.'},'model_selection':'heldout temporal + weak progress/value + full-coverage reviewed terminal + optional full-coverage preference loss; endpoint curation is never model selection','representation_caveats':['Instance SSL may still encode episode/background identity; adjacent sample pairs share episodes to include distant within-episode negatives.','Signed temporal distance is trained within episodes; goal-bank cross-episode extrapolation is unvalidated and never used for selection.','Kept normalized timestamp phase represents advancement along recording, not verified physical task progress; monotonic weak ranking is an explicit imperfect assumption.'],'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    for epoch in range(start,args.epochs):
        # Recompute absolute schedule so extending a completed pilot does not resume at zero LR.
        for group in opt.param_groups:group['lr']=args.lr*.5*(1+math.cos(math.pi*epoch/args.epochs))
        record={'epoch':epoch+1}
        for name,loader in [('train',tr),('val',vl)]:
            model.train(name=='train');acc=collections.defaultdict(float);n=0
            for batch in loader:
                batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
                with torch.set_grad_enabled(name=='train'),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=args.amp and device.type=='cuda'):
                    loss,stats=losses(model,batch,args)
                    if name=='train' and training_preferences and (n+1)%4==0:
                        chosen=random.choices(training_preferences,k=min(args.preference_batch_size,len(training_preferences)))
                        pref=preference_batch(model,chosen,args,device,augment=True).mean();loss=loss+args.preference_weight*pref;stats['preference_hinge']=pref
                if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite {name} loss')
                if name=='train':
                    opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(opt);scaler.update()
                for k,v in stats.items():acc[k]+=float(v.detach())
                n+=1
                if name=='train' and n%100==0:print(json.dumps({'epoch':epoch+1,'step':n,'loss':float(loss.detach())}),flush=True)
            record[name]={k:v/n for k,v in acc.items()}
        pref_val=evaluate_preferences(model,preferences,args,device);record['val']['preference_full_coverage']=pref_val
        reviewed_val=evaluate_reviewed_terminals(model,episodes,args,device)
        record['val']['sampled_reviewed_terminal_bce']=record['val']['reviewed_terminal_bce']
        record['val']['reviewed_terminal_bce']=reviewed_val['bce'];record['val']['reviewed_terminal_full_coverage']=reviewed_val
        scheduler.step();score=args.temporal_weight*record['val']['temporal']+args.progress_weight*(record['val']['weak_progress_mse']+record['val']['weak_ranking'])+args.value_weight*record['val']['weak_value_mse']+args.terminal_weight*record['val']['reviewed_terminal_bce']+args.preference_weight*pref_val['mean_hinge'];improved=score<best;best=min(best,score)
        record['selection_score']=score;record['best']=best;record['gpu_peak_gb']=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0
        build_goal_bank(model,episodes,args,device)
        state={'selection_objective':selection_objective,'goal_bank':model.goal_bank,'goal_bank_ids':model.goal_bank_ids,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'best':best,'config':report['config'],'manifest_sha256':digest,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        atomic_save(state,args.output/'last.pt')
        if improved:atomic_save(state,args.output/'best.pt')
        with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    if args.export_predictions:
        best_ck=torch.load(args.output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best_ck['model']);model.goal_bank=best_ck['goal_bank'];model.goal_bank_ids=best_ck['goal_bank_ids'];predictions=predict(model,rows,args,device);diagnostics(rows,predictions,args);trajectory_review(rows,predictions,args)


if __name__=='__main__':main()
