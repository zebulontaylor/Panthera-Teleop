#!/usr/bin/env python3
"""Uncalibrated visual state goodness from demonstrated states and reviewed mistakes.

No elapsed-time, phase, episode-end distance, curation ID or state-vector inputs.
Confirmed clean kept states are demonstrated solvable states, not guaranteed current
policy completions. Reviewed bad endpoints can be recoverable; negatives express
relative visual quality rather than impossible completion. Original bundles untouched.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, random, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
from PIL import Image
from state_value_train import VisualValue,read_preferences,atomic_save


def load_frames(path,clean_kept_confirmed):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()];episodes=collections.defaultdict(list);sessions={};ep_splits={}
    for row in rows:
        ep=str(row['episode_id']);session=str(row.get('session_id',ep));split=row['split']
        if split not in ('train','val','test'):raise ValueError('invalid frame split')
        if ep in ep_splits and ep_splits[ep]!=split:raise ValueError('episode split leakage')
        if session in sessions and sessions[session]!=split:raise ValueError('session split leakage')
        ep_splits[ep]=split;sessions[session]=split
        if len(row['images'])!=3:raise ValueError('three ordered views required')
        row['_label']=None;row['_confidence']=0.
        # Explicit reviewed terminal annotations always override clean-demo support.
        if row.get('terminal_outcome') in (0,1):
            row['_label']=float(row['terminal_outcome']);row['_confidence']=float(row['terminal_outcome_confidence']);row['_evidence']='reviewed_terminal_relative_quality'
        elif clean_kept_confirmed and row.get('source_category')=='kept':
            row['_label']=1.;row['_confidence']=1.;row['_evidence']='user_confirmed_clean_demonstration_support'
        if row['_label'] is not None and not 0<row['_confidence']<=1:raise ValueError('label confidence must be in (0,1]')
        episodes[ep].append(row)
    # No numerical timestamp or duration is read in label construction.
    return rows,episodes


class Images:
    def __init__(self,root,size,augment):
        self.root=Path(root);steps=[T.Resize((size,size))]
        if augment:steps += [T.RandomResizedCrop(size,scale=(.92,1.),ratio=(.97,1.03)),T.ColorJitter(.15,.15,.1,.02)]
        self.transform=T.Compose(steps+[T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __call__(self,refs,root=None):
        views=[]
        for ref in refs:
            with Image.open(Path(root or self.root)/ref) as im:views.append(self.transform(im.convert('RGB')))
        return torch.stack(views)


class BalancedStates(Dataset):
    """Alternating evidence classes; uniform sessions, then episodes, then frames."""
    def __init__(self,episodes,root,size,length):
        self.groups={0:collections.defaultdict(lambda:collections.defaultdict(list)),1:collections.defaultdict(lambda:collections.defaultdict(list))};self.length=length;self.images=Images(root,size,True)
        for ep,seq in episodes.items():
            for row in seq:
                if row['split']=='train' and row['_label'] is not None:self.groups[int(row['_label'])][str(row.get('session_id',ep))][ep].append(row)
        if any(not x for x in self.groups.values()):raise ValueError('both positive support and reviewed negative training states required')
    def __len__(self):return self.length
    def __getitem__(self,index):
        groups=self.groups[index%2];session=random.choice(list(groups));eps=groups[session];ep=random.choice(list(eps));row=random.choice(eps[ep])
        return self.images(row['images'],row.get('image_root')),torch.tensor(row['_label']),torch.tensor(row['_confidence'])


class Goodness(nn.Module):
    def __init__(self,backbone='resnet50',pretrained=True,freeze_encoder=False,camera_mode='all'):
        super().__init__();
        if camera_mode not in ('all','wrists-only'):raise ValueError('invalid camera mode')
        self.camera_mode=camera_mode;base=VisualValue(pretrained,backbone);self.backbone=base.backbone;self.fuse=base.fuse
        # Always fresh head: prior elapsed-time/progress/value heads are not imported.
        self.goodness=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1));self.freeze_encoder=freeze_encoder
        if freeze_encoder:
            # Freeze the pretrained feature extractor; learn camera fusion and fresh head.
            for module in (self.backbone,):
                for p in module.parameters():p.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode)
        if self.freeze_encoder:self.backbone.eval()
        return self
    def forward(self,images):
        # Remove overhead pixels structurally, before any learned computation.
        # A constant normalized-zero view preserves transfer-compatible fusion shape.
        if self.camera_mode=='wrists-only':images=torch.cat([torch.zeros_like(images[:,:1]),images[:,1:]],dim=1)
        b,v,c,h,w=images.shape;z=self.fuse(self.backbone(images.reshape(b*v,c,h,w)).reshape(b,-1));return self.goodness(z).flatten()


def read_goodness_preferences(path,episodes):
    """Preserve unknown reviews for coverage reporting, never use them as targets."""
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    for row in rows:
        if row.get('split') not in ('train','val'):raise ValueError('test preference rows are forbidden in training')
        if row.get('preference') not in ('a_better','b_better','equal','unknown'):raise ValueError('invalid preference')
        for side in ('a','b'):
            ep=str(row['episode_id_'+side])
            if ep not in episodes or episodes[ep][0]['split']!=row['split']:raise ValueError('preference episode/split leakage')
            refs=row['observation_'+side]['current']
            if isinstance(refs,dict):refs=[refs[k] for k in ('overhead','wrist_port2','wrist_port3')];row['observation_'+side]['current']=refs
            if len(refs)!=3:raise ValueError('three ordered preference views required')
        if row['preference']!='unknown' and not 0<float(row.get('confidence',1.))<=1:raise ValueError('preference confidence must be in (0,1]')
    return rows


def support_loss(logits,labels,confidence,smoothing):
    target=labels*(1-2*smoothing)+smoothing
    return (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*confidence).sum()/confidence.sum().clamp_min(1e-6)


def preference_loss(logits,labels,margin=.05):
    delta=torch.sigmoid(logits[:,0])-torch.sigmoid(logits[:,1])
    return torch.where(labels==0,F.relu(delta.abs()-margin),F.relu(margin-delta*labels))


def confidence_preference_loss(logits,labels,confidence,margin=.05):
    # Absolute trust multiplier; no normalization by confidence that would cancel it.
    return (preference_loss(logits,labels,margin)*confidence).mean()


def pair_forward(model,rows,reader,device):
    images=[]
    for row in rows:
        for side in ('a','b'):images.append(reader(row['observation_'+side]['current']))
    logits=model(torch.stack(images).to(device)).reshape(len(rows),2)
    labels=torch.tensor([{'a_better':1.,'b_better':-1.,'equal':0.}[r['preference']] for r in rows],device=device)
    weights=torch.tensor([float(r.get('confidence',1.)) for r in rows],device=device)
    return logits,labels,weights


def sample_pairs(rows,count):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:groups[str(r.get('session_id',r['episode_id_a']))][str(r['episode_id_a'])].append(r)
    chosen=[]
    for _ in range(count):
        eps=groups[random.choice(list(groups))];chosen.append(random.choice(eps[random.choice(list(eps))]))
    return chosen


def evaluation_rows(episodes,split,frames_per_episode):
    selected=[]
    for seq in episodes.values():
        known=[r for r in seq if r['split']==split and r['_label'] is not None]
        if not known:continue
        # Include every reviewed state and index-uniform clean support examples.
        reviewed=[r for r in known if r['_evidence'].startswith('reviewed')];support=[r for r in known if not r['_evidence'].startswith('reviewed')]
        selected+=reviewed
        if support:selected += [support[i] for i in np.unique(np.linspace(0,len(support)-1,min(frames_per_episode,len(support))).astype(int))]
    return selected


def macro_preference_hinge(pair_stats):
    """Mean confidence times hinge per episode, then equal episodes and sessions.

    Divide by pair count, not summed confidence: uniformly weak annotations
    must have proportionately less influence than high-confidence annotations.
    """
    grouped=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in pair_stats:grouped[str(row['session_id'])][str(row['episode_id_a'])].append(row)
    if not grouped:return 0.
    return float(np.mean([np.mean([np.mean([r['loss']*r['confidence'] for r in rows]) for rows in episodes.values()]) for episodes in grouped.values()]))


def pair_evidence_group(row):
    source=row.get('label_source') or 'reviewed_local_pair'
    if source=='user_workflow_weak_intent_prior':
        return 'visually_corroborated_weak_intent' if row.get('visual_review_status') else 'weak_intent_prior'
    return source


def grouped_pair_metrics(pair_stats,raw_pairs,split):
    groups=collections.defaultdict(list);unknown=collections.Counter()
    for row in pair_stats:groups[pair_evidence_group(row)].append(row)
    for row in raw_pairs:
        if row['split']==split and row['preference']=='unknown':unknown[pair_evidence_group(row)]+=1
    output={}
    for name in sorted(set(groups)|set(unknown)):
        rows=groups[name];directional=[r for r in rows if r['preference']!='equal'];equal=[r for r in rows if r['preference']=='equal'];sessions=collections.defaultdict(list);equal_sessions=collections.defaultdict(list)
        for row in directional:sessions[str(row['session_id'])].append(row['correct'])
        for row in equal:equal_sessions[str(row['session_id'])].append(abs(row['score_delta_a_minus_b']))
        output[name]={'known_pairs':len(rows),'unknown_pairs':unknown[name],'directional_pairs':len(directional),'directional_accuracy':float(np.mean([r['correct'] for r in directional])) if directional else None,'session_macro_directional_accuracy':float(np.mean([np.mean(v) for v in sessions.values()])) if sessions else None,'directional_sessions':len(sessions),'equal_pairs':len(equal),'equal_mean_absolute_score_gap':float(np.mean([abs(r['score_delta_a_minus_b']) for r in equal])) if equal else None,'equal_session_macro_mean_absolute_score_gap':float(np.mean([np.mean(v) for v in equal_sessions.values()])) if equal_sessions else None,'macro_confidence_scaled_hinge':macro_preference_hinge(rows)}
    return output


def evaluate(model,episodes,pairs,args,device,split='val'):
    model.eval();reader=Images(args.dataset_root,args.image_size,False);rows=evaluation_rows(episodes,split,args.val_frames_per_episode);records=[]
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            sub=rows[start:start+args.batch_size];scores=torch.sigmoid(model(torch.stack([reader(r['images'],r.get('image_root')) for r in sub]).to(device))).cpu().tolist()
            for row,p in zip(sub,scores):records.append((row,p))
        unknown_pairs=sum(r['split']==split and r['preference']=='unknown' for r in pairs)
        val_pairs=[r for r in pairs if r['split']==split and r['preference']!='unknown'];pair_reader=Images(args.preference_image_root,args.image_size,False);pair_stats=[]
        for start in range(0,len(val_pairs),args.preference_batch_size):
            sub=val_pairs[start:start+args.preference_batch_size];logits,labels,weights=pair_forward(model,sub,pair_reader,device);loss=preference_loss(logits,labels,args.preference_margin);delta=(torch.sigmoid(logits[:,0])-torch.sigmoid(logits[:,1]));correct=torch.where(labels==0,delta.abs()<=args.preference_margin,delta*labels>0)
            for r,l,c,w,d in zip(sub,loss.cpu().tolist(),correct.cpu().tolist(),weights.cpu().tolist(),delta.cpu().tolist()):pair_stats.append({'pair_id':r.get('pair_id'),'label_source':r.get('label_source') or 'reviewed_local_pair','visual_review_status':r.get('visual_review_status'),'review_status':r.get('review_status'),'human_visible_risk':r.get('human_visible_risk'),'preference':r['preference'],'session_id':r.get('session_id',r['episode_id_a']),'episode_id_a':r['episode_id_a'],'episode_id_b':r['episode_id_b'],'loss':l,'correct':c,'confidence':float(r.get('confidence',1.)),'annotated_confidence':r.get('confidence'),'score_delta_a_minus_b':d,'within_episode':r['episode_id_a']==r['episode_id_b'],'progress_component':r.get('progress_component')})
    # Macro average within episode, session, evidence class; one long demo never dominates.
    grouped=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
    for row,p in records:
        target=row['_label']*(1-2*args.label_smoothing)+args.label_smoothing;p=min(1-1e-7,max(1e-7,p));loss=-(target*math.log(p)+(1-target)*math.log(1-p));grouped[int(row['_label'])][str(row.get('session_id',row['episode_id']))][str(row['episode_id'])].append((loss,row['_confidence']))
    class_losses={str(label):float(np.mean([np.mean([np.average([x[0] for x in ls],weights=[x[1] for x in ls]) for ls in eps.values()]) for eps in sessions.values()])) for label,sessions in grouped.items()}
    if len(class_losses)!=2:raise ValueError('heldout validation must contain both support and reviewed negative states')
    pair_hinge=macro_preference_hinge(pair_stats)
    within=[r for r in pair_stats if r['within_episode']]
    directional=[r for r in pair_stats if r['preference']!='equal'];equals=[r for r in pair_stats if r['preference']=='equal'];by_session=collections.defaultdict(list)
    for row in directional:by_session[str(row['session_id'])].append(row['correct'])
    equal_by_session=collections.defaultdict(list)
    for row in equals:equal_by_session[str(row['session_id'])].append(abs(row['score_delta_a_minus_b']))
    return {'balanced_support_bce':float(np.mean(list(class_losses.values()))),'class_bce':class_losses,'state_examples':len(records),'state_episodes':len(set(str(r['episode_id']) for r,p in records)),'preference_hinge':pair_hinge,'preference_aggregation':'mean(confidence * hinge) within episode_id_a; episode mean within session; session mean','preference_pairs':len(pair_stats),'within_episode_preference_accuracy':float(np.mean([r['correct'] for r in within])) if within else None,'directional_preference_pairs':len(directional),'directional_preference_accuracy':float(np.mean([r['correct'] for r in directional])) if directional else None,'session_macro_directional_accuracy':float(np.mean([np.mean(v) for v in by_session.values()])) if by_session else None,'directional_sessions':len(by_session),'equal_preference_pairs':len(equals),'equal_mean_absolute_score_gap':float(np.mean([abs(r['score_delta_a_minus_b']) for r in equals])) if equals else None,'equal_session_macro_mean_absolute_score_gap':float(np.mean([np.mean(v) for v in equal_by_session.values()])) if equal_by_session else None,'equal_sessions':len(equal_by_session),'unknown_preference_pairs':unknown_pairs,'evidence_source_metrics':grouped_pair_metrics(pair_stats,pairs,split),'pair_details':pair_stats}


def validation_selection_score(validation,args):
    weight=getattr(args,'selection_preference_weight',None)
    if weight is None:weight=args.preference_weight
    return validation['balanced_support_bce']+weight*validation['preference_hinge']


def shortcut_baselines(episodes,args,split='val'):
    """Fit on train evidence only; diagnostic metadata is never a network input."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    train=evaluation_rows(episodes,'train',args.val_frames_per_episode);val=evaluation_rows(episodes,split,args.val_frames_per_episode);result={}
    def features(rows,name):
        if name=='raw_state':return np.asarray([r['state'] for r in rows],dtype=float)
        if name=='elapsed_time':return np.asarray([[float(r.get('t',0))] for r in rows])
        return np.asarray([[float(r.get('source_category')=='kept')] for r in rows])
    labels=np.asarray([r['_label'] for r in train]);yv=np.asarray([r['_label'] for r in val])
    if len(set(labels))<2:return {'warning':'train evidence needs both classes'}
    for name in ('raw_state','elapsed_time','curation_membership'):
        if name=='raw_state' and not all(r.get('state') is not None for r in train+val):continue
        clf=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000,random_state=args.seed));clf.fit(features(train,name),labels);p=clf.predict_proba(features(val,name))[:,1];target=yv*(1-2*args.label_smoothing)+args.label_smoothing
        result[name]={'auroc':float(roc_auc_score(yv,p)) if len(set(yv))==2 else None,'smoothed_bce':float(np.mean(-(target*np.log(np.clip(p,1e-7,1))+(1-target)*np.log(np.clip(1-p,1e-7,1))))),'n':len(val),'warning':'Diagnostic shortcut only; curation membership constructs much of the support label, so its performance is circular.' if name=='curation_membership' else 'Metadata baseline; not an input to the visual model.'}
    return result


def validate_encoder_provenance(encoder,current_episodes,current_digest,source_manifest):
    if encoder.get('manifest_sha256')==current_digest:return 'identical_frozen_manifest'
    if not source_manifest:raise ValueError('encoder checkpoint evidence differs; provide --encoder-manifest to verify heldout split isolation')
    if hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest()!=encoder.get('manifest_sha256'):raise ValueError('encoder manifest hash mismatch')
    _,old=load_frames(source_manifest,False);heldout_eps={ep for ep,v in current_episodes.items() if v[0]['split']!='train'};heldout_sessions={str(v[0].get('session_id',ep)) for ep,v in current_episodes.items() if v[0]['split']!='train'}
    for ep,v in old.items():
        if v[0]['split']=='train' and (ep in heldout_eps or str(v[0].get('session_id',ep)) in heldout_sessions):raise ValueError('encoder exposure leaks current heldout episode/session')
    return 'source_manifest_verified_train_disjoint_current_holdouts'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--dataset-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--camera-mode',choices=('all','wrists-only'),default='all');ap.add_argument('--clean-kept-confirmed',action='store_true',help='User confirms kept demonstrations are clean; enables positive support labels')
    ap.add_argument('--preference-manifest',type=Path);ap.add_argument('--preference-image-root',type=Path,default=Path('.'));ap.add_argument('--preference-batch-size',type=int,default=8);ap.add_argument('--preference-weight',type=float,default=1.);ap.add_argument('--selection-preference-weight',type=float,default=None,help='Validation hinge multiplier; defaults to training preference weight for compatibility');ap.add_argument('--preference-margin',type=float,default=.05)
    ap.add_argument('--encoder-manifest',type=Path,help='Source manifest for split verification if transfer checkpoint has a different manifest hash');ap.add_argument('--encoder-checkpoint',type=Path,help='Transfer backbone/fusion only from stage1; fresh goodness head');ap.add_argument('--freeze-encoder','--freeze-backbone',dest='freeze_encoder',action='store_true',help='Freeze ResNet features; camera fusion and new goodness head remain trainable');ap.add_argument('--backbone',choices=('resnet18','resnet50'),default='resnet50');ap.add_argument('--pretrained',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--epochs',type=int,default=4);ap.add_argument('--steps-per-epoch',type=int,default=500);ap.add_argument('--batch-size',type=int,default=24);ap.add_argument('--image-size',type=int,default=224);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--label-smoothing',type=float,default=.1);ap.add_argument('--val-frames-per-episode',type=int,default=8)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--cpu-threads',type=int,default=8);ap.add_argument('--seed',type=int,default=29);ap.add_argument('--gpu-memory-gb',type=float,default=40.);ap.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);ap.add_argument('--resume',type=Path);ap.add_argument('--wandb-mode',choices=('offline','disabled'),default='offline')
    args=ap.parse_args()
    if args.selection_preference_weight is None:args.selection_preference_weight=args.preference_weight
    if args.batch_size<2 or args.batch_size%2 or min(args.epochs,args.steps_per_epoch,args.preference_batch_size,args.val_frames_per_episode)<1:ap.error('positive steps/epochs/pair batches required; state batch size must be even and >=2')
    if not 0<=args.label_smoothing<.5:ap.error('label smoothing must be in [0,.5)')
    torch.set_num_threads(args.cpu_threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(.95,args.gpu_memory_gb*1024**3/torch.cuda.get_device_properties(0).total_memory))
    args.output.mkdir(parents=True,exist_ok=True);rows,episodes=load_frames(args.manifest,args.clean_kept_confirmed);pairs=read_goodness_preferences(args.preference_manifest,episodes) if args.preference_manifest else [];train_pairs=[r for r in pairs if r['split']=='train' and r['preference']!='unknown']
    digest=hashlib.sha256(args.manifest.read_bytes()).hexdigest();pref_digest=hashlib.sha256(args.preference_manifest.read_bytes()).hexdigest() if args.preference_manifest else None
    ck=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None;enc=torch.load(args.encoder_checkpoint,map_location='cpu',weights_only=False) if args.encoder_checkpoint and not ck else None
    if ck:
        for key in ('clean_kept_confirmed','label_smoothing','preference_weight','preference_margin','val_frames_per_episode','selection_preference_weight','camera_mode'):
            if ck['config'].get(key,ck['config']['preference_weight'] if key=='selection_preference_weight' else 'all' if key=='camera_mode' else None)!=getattr(args,key):raise ValueError('resume label/selection semantics changed: '+key)
        if ck['manifest_sha256']!=digest or ck['preference_manifest_sha256']!=pref_digest:raise ValueError('resume evidence manifest changed')
        for key in ('backbone','image_size','freeze_encoder'):setattr(args,key,ck['config'][key])
    transfer_provenance=validate_encoder_provenance(enc,episodes,digest,args.encoder_manifest) if enc else None
    if enc:args.backbone=enc['config']['backbone']
    model=Goodness(args.backbone,args.pretrained and not(enc or ck),args.freeze_encoder,args.camera_mode).to(device)
    if enc:
        encoder_state={k:v for k,v in enc['model'].items() if k.startswith(('backbone.','fuse.'))};missing,unexpected=model.load_state_dict(encoder_state,strict=False)
        if unexpected or any(not k.startswith('goodness.') for k in missing):raise ValueError('incompatible encoder checkpoint')
    if ck:model.load_state_dict(ck['model'])
    dataset=BalancedStates(episodes,args.dataset_root,args.image_size,args.steps_per_epoch*args.batch_size);loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=args.weight_decay);scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda');start=0;best=float('inf')
    if ck:
        opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);start=ck['epoch'];best=ck['best'];random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch'])
        if device.type=='cuda' and ck.get('rng_cuda'):torch.cuda.set_rng_state_all(ck['rng_cuda'])
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};cfg['preference_aggregation']='mean(confidence * hinge) within episode_id_a, denominator=pair count; equal-episode mean within session; equal-session mean';report={'encoder_split_verification':transfer_provenance,'preference_trust_boundary':'Declared pair episode/split IDs are checked. Pair image provenance is trusted audited reviewer input; raw-frame references may differ from sampled-frame paths.','config':cfg,'manifest_sha256':digest,'preference_manifest_sha256':pref_digest,'interpretation':'Uncalibrated demonstrated-state support / relative visual quality. Clean demo state means demonstrated solvability, not guaranteed current-policy completion. Reviewed mistakes may be recoverable. No timestamp/phase/end-distance targets or inputs.','selection_predeclared':'Macro class/session/episode validation smoothed BCE + selection_preference_weight * validation preference hinge (training preference_weight is separate) (mean(confidence * hinge) within episode_id_a (divide by pair count, not summed confidence), equal episodes within session, equal sessions). Every known validation pair evaluated once; unknown reviews counted but excluded. Test frames and fresh six heldout review cases are never used for tuning.','encoder_initialization':'stage1_backbone_and_fusion_fresh_head' if enc else 'resume' if ck else 'imagenet' if args.pretrained else 'random','camera_input':'Overhead replaced with normalized-zero constant before encoder; no overhead pixel affects output.' if args.camera_mode=='wrists-only' else 'All three camera views used.','shortcut_limits':'Image backgrounds and curation cohorts can still confound state labels; compare fixed ImageNet vs stage1 transfer and inspect same-episode mistake/recovery preference accuracy.','evidence_counts':{s:{'positive_frames':sum(r['split']==s and r['_label']==1 for r in rows),'negative_frames':sum(r['split']==s and r['_label']==0 for r in rows),'episodes':sum(v[0]['split']==s for v in episodes.values())} for s in ('train','val','test')}}
    (args.output/'shortcut_baselines.json').write_text(json.dumps(shortcut_baselines(episodes,args),indent=2)+'\n')
    (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    wb=None
    if args.wandb_mode=='offline':
        import wandb
        wb=wandb.init(project='panthera-state-goodness',mode='offline',dir=str(args.output),config=report,name=args.output.name)
    pair_reader=Images(args.preference_image_root,args.image_size,True)
    for epoch in range(start,args.epochs):
        model.train();totals=collections.defaultdict(float)
        for group in opt.param_groups:group['lr']=args.lr*.5*(1+math.cos(math.pi*epoch/args.epochs))
        for step,(images,labels,confidence) in enumerate(loader,1):
            images=images.to(device);labels=labels.to(device);confidence=confidence.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=args.amp and device.type=='cuda'):
                logits=model(images);state_loss=support_loss(logits,labels,confidence,args.label_smoothing);pair_loss=logits.sum()*0
                if train_pairs:
                    chosen=sample_pairs(train_pairs,args.preference_batch_size);pl,py,pw=pair_forward(model,chosen,pair_reader,device);pair_loss=confidence_preference_loss(pl,py,pw,args.preference_margin)
                loss=state_loss+args.preference_weight*pair_loss
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite goodness loss')
            opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(opt);scaler.update()
            step_record={'step':epoch*args.steps_per_epoch+step,'epoch':epoch+1,'train':{'loss':float(loss.detach()),'rank_loss':float(pair_loss.detach()),'outcome_loss':float(state_loss.detach())}}
            if step<args.steps_per_epoch:
                with (args.output/'history.jsonl').open('a') as out:out.write(json.dumps(step_record)+'\n')
            if wb and step<args.steps_per_epoch:wb.log({'train/loss':float(loss.detach()),'train/rank_loss':float(pair_loss.detach()),'train/outcome_loss':float(state_loss.detach())},step=epoch*args.steps_per_epoch+step)
            totals['state_bce']+=float(state_loss.detach());totals['preference_hinge']+=float(pair_loss.detach())
            if step%50==0:
                progress={'epoch':epoch+1,'step':step,'train_state_bce':float(state_loss.detach()),'train_preference_hinge':float(pair_loss.detach())};print(json.dumps(progress),flush=True)

        val=evaluate(model,episodes,pairs,args,device);score=validation_selection_score(val,args);improved=score<best;best=min(best,score)
        record={'step':(epoch+1)*args.steps_per_epoch,'epoch':epoch+1,'epoch_summary':True,'train':step_record['train'],'epoch_train_mean':{k:v/args.steps_per_epoch for k,v in totals.items()},'val':val,'selection_score':score,'best':best,'gpu_peak_gb':torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0.}
        state={'kind':'visual_state_goodness','model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'best':best,'config':cfg,'manifest_sha256':digest,'preference_manifest_sha256':pref_digest,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None};atomic_save(state,args.output/'last.pt')
        if improved:atomic_save(state,args.output/'best.pt')
        with (args.output/'history.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if wb:wb.log({'train/loss':step_record['train']['loss'],'train/rank_loss':step_record['train']['rank_loss'],'train/outcome_loss':step_record['train']['outcome_loss'],'epoch':epoch+1,'val/balanced_support_bce':val['balanced_support_bce'],'val/preference_hinge':val['preference_hinge'],'selection_score':score},step=(epoch+1)*args.steps_per_epoch)
    if wb:wb.finish()


if __name__=='__main__':main()
