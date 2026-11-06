#!/usr/bin/env python3
"""Train center-axis top Y from real+sim, distal bottom offset from sim only."""
from __future__ import annotations
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from app.held_pin_cnn import HeldPinCNN,SCHEMA,heatmap_loss
from app.held_pin_augmentation import augment_batch, apply_shadows, AUGMENTATION_CONFIG

def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def load_real(path):
    value=json.loads(Path(path).read_text());rows=value['frames'];groups={}
    usable=[]
    for r in rows:
        if not r.get('top_supervision_valid',False):continue
        s='validation' if r['split']=='val' else r['split'];r['split']=s
        if groups.setdefault(r['group_id'],s)!=s:raise ValueError('real pose group leakage')
        if r.get('bottom_supervision_valid',False):raise ValueError('real bottom ground truth unavailable in this task')
        usable.append(r)
    return value,usable

def prepare(real_manifest,sim_path):
    info,rr=load_real(real_manifest);rows=[]
    axis=float(info.get('axis_x_fraction',.5234375))
    for r in rr:
        im=np.asarray(Image.open(r['image_path']).convert('RGB')).copy();h,w=im.shape[:2]
        top=r.get('top_px');top=[axis*w,r['top_y_px']] if top is None else top
        rows.append({'id':r['id'],'domain':'real','split':r['split'],'group_id':r['group_id'],
            'rgb':torch.from_numpy(im).permute(2,0,1),'top':np.asarray(top,np.float32)/[w,h],
            'bottom':np.array([axis,.5],np.float32),'top_visible':r.get('top_visible',True),'bottom_valid':False})
    with np.load(sim_path) as archive:
        data={key:archive[key] for key in archive.files}
    for i in range(len(data['images'])):
        split=str(data['splits'][i]);split='validation' if split=='val' else split
        rows.append({'id':f'sim_{i}','domain':'sim','split':split,'group_id':str(data['groups'][i]),
            'rgb':torch.from_numpy(data['images'][i].copy()).permute(2,0,1),
            'top':data['top_uv'][i].astype(np.float32),'bottom':data['bottom_uv'][i].astype(np.float32),
            'top_visible':bool(data['top_visible'][i]),'bottom_valid':bool(np.isfinite(data['bottom_uv'][i]).all() and np.isfinite(data['top_uv'][i]).all())})
    return info,rows,axis

def batch(rr,device):
    # Match camera angular crop scale; only real frames contain original640detail.
    size=(240,320) if all(r['domain']=='sim' for r in rr) else (480,640)
    x=torch.cat([F.interpolate(r['rgb'][None].to(device).float()/255,size,mode='bilinear',align_corners=False) for r in rr])
    top=torch.tensor(np.stack([r['top'] for r in rr]),device=device,dtype=torch.float32)
    bottom=torch.tensor(np.stack([r['bottom'] for r in rr]),device=device,dtype=torch.float32)
    visible=torch.tensor([r['top_visible'] for r in rr],device=device)
    bv=torch.tensor([r['bottom_valid'] for r in rr],device=device)
    return x,top,bottom,visible,bv

def summarize(values):
    return {'count':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p90':float(np.quantile(values,.9)),'max':float(np.max(values))} if values else {'count':0}

def evaluate(model,rr,device,axis,threshold=.6,shadow_seed=None):
    records=[];model.eval()
    shadow_generator=None if shadow_seed is None else torch.Generator(device=device).manual_seed(shadow_seed)
    with torch.inference_mode():
        for start in range(0,len(rr),12):
            part=rr[start:start+12];x,t,b,v,bv=batch(part,device)
            if shadow_generator is not None:x=apply_shadows(x,shadow_generator,extreme=True)
            out=model(x,axis_x=axis,roi_size_px=160*x.shape[-1]/640)
            conf=torch.minimum(out['coarse']['visibility_logits'].sigmoid(),out['fine']['visibility_logits'].sigmoid())
            for i,r in enumerate(part):
                records.append({'id':r['id'],'domain':r['domain'],'group_id':r['group_id'],'top_visible':bool(v[i]),
                    'top_truth_y_480':float(t[i,1]*480),'top_pred_y_480':float(out['top_y'][i]*480),
                    'bottom_truth_y_480':float(b[i,1]*480) if bv[i] else None,'bottom_pred_y_480':float(out['bottom_y'][i]*480),
                    'split':r['split'],'top_error_y_px_480':float(abs(out['top_y'][i]-t[i,1])*480) if v[i] else None,
                    'bottom_error_y_px_480':float(abs(out['bottom_y'][i]-b[i,1])*480) if bv[i] else None,
                    'coarse_top_error_y_px_480':float(abs(out['coarse']['y'][i]-t[i,1])*480) if v[i] else None,
                    'confidence':float(conf[i]),'accepted':bool(conf[i]>=threshold and torch.isfinite(out['top_y'][i]) and torch.isfinite(out['bottom_y'][i]) and 0<=out['top_y'][i]<=1 and 0<=out['bottom_y'][i]<=1)})
    result={'records':records}
    for domain in ('real','sim'):
        d=[r for r in records if r['domain']==domain]
        result[domain]={'top_y_error_px_480':summarize([r['top_error_y_px_480'] for r in d if r['top_error_y_px_480'] is not None]),
            'accepted_top_y_error_px_480':summarize([r['top_error_y_px_480'] for r in d if r['top_error_y_px_480'] is not None and r['accepted']]),
            'bottom_y_error_px_480':summarize([r['bottom_error_y_px_480'] for r in d if r['bottom_error_y_px_480'] is not None]),
            'coarse_top_y_error_px_480':summarize([r['coarse_top_error_y_px_480'] for r in d if r['coarse_top_error_y_px_480'] is not None]),
            'top_visible_count':sum(r['top_visible'] for r in d),'top_hidden_count':sum(not r['top_visible'] for r in d),
            'accepted_visible_count':sum(r['top_visible'] and r['accepted'] for r in d),
            'false_visible_count':sum(not r['top_visible'] and r['accepted'] for r in d)}
    return result

def train(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed);device=torch.device(args.device)
    info,rows,axis=prepare(args.real_manifest,args.sim_dataset)
    train_rows=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation'];test=[r for r in rows if r['split']=='test']
    # Check combined source-group partition before fitting; variants stay with their acquisition.
    for domain in ('real','sim'):
        partitions={}
        for row in rows:
            if row['domain']!=domain:continue
            if partitions.setdefault(row['group_id'],row['split'])!=row['split']:
                raise ValueError(f'{domain} pose group leakage: {row["group_id"]}')
    real_train=[r for r in train_rows if r['domain']=='real'];sim_train=[r for r in train_rows if r['domain']=='sim']
    if not real_train or not sim_train:raise ValueError('real TOP and simulation endpoint training data required')
    model=HeldPinCNN().to(device)
    if args.visual_initialization:
        old=torch.load(args.visual_initialization,map_location=device,weights_only=True)['model']
        backbone={k[len('vision.'):]:v for k,v in old.items() if k.startswith('vision.')}
        for part in (model.coarse,model.refiner):part.vision.load_state_dict(backbone)
    if args.resume:
        model.load_state_dict(torch.load(args.resume,map_location=device,weights_only=True)['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    generator=torch.Generator(device=device).manual_seed(args.seed+1);rng=np.random.default_rng(args.seed)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    metadata={'schema':SCHEMA,'camera':'left_wrist','axis_x_fraction':axis,'roi_size_px':160,'visibility_threshold':.6,
        'top_target':'near/proximal cap face center','bottom_target':'distal endpoint inferred under center-axis assumption',
        'trainer_sha256':file_sha256(__file__),'augmentation_sha256':file_sha256(ROOT/'app/held_pin_augmentation.py'),
        'generator_sha256':file_sha256(ROOT/'scripts/generate_held_pin_axis_sim.py'),
        'real_bottom_supervision':False,'real_top_labels':len(real_train),'real_validation_labels':sum(r['domain']=='real' for r in validation),
        'real_manifest':str(Path(args.real_manifest).resolve()),'real_manifest_sha256':hashlib.sha256(Path(args.real_manifest).read_bytes()).hexdigest(),
        'sim_dataset':str(Path(args.sim_dataset).resolve()),'sim_dataset_sha256':file_sha256(args.sim_dataset),'augmentation':AUGMENTATION_CONFIG,'seed':args.seed,
        'assumptions':['pin lies on configured gripper center axis','simulation nominal grasp and 17.88mm pin length determine projected bottom offset','real cap labels account for observed top displacement; real bottom remains unvalidated'],
        'pose_group_holdout':True,'selection_protocol':'validation only: ordinary real TOP + .2 sim BOTTOM + .5 shadow real TOP + .1 shadow sim BOTTOM',
        'shadow_validation_seed':args.seed+9000,'shadow_test_seed':args.seed+19000,
        'training_epochs':args.epochs,'real_repeat':args.real_repeat,'learning_rate':args.lr,'native_real_resolution':[640,480],'sim_roi_scaling':'native simulation RGB; crop width scaled proportionally to640-reference160px','sim_native_resolution':[int(sim_train[0]['rgb'].shape[2]),int(sim_train[0]['rgb'].shape[1])],
        'real_label_uncertainty_px':3,'visual_initialization':args.visual_initialization,'resume_checkpoint':args.resume}
    best=float('inf');history=[];variants=0;start_time=time.time()
    for epoch in range(args.epochs):
        model.train();losses=[]
        # Real TOP observations are repeated with new random augmentation, not new independent labels.
        # Homogeneous batches retain native real detail and avoid upsampling simulation.
        minibatches=[]
        for pool in (sim_train.copy(),real_train*args.real_repeat):
            rng.shuffle(pool)
            minibatches.extend(pool[start:start+args.batch_size] for start in range(0,len(pool),args.batch_size))
        rng.shuffle(minibatches)
        for rr in minibatches:
            x,t,b,v,bv=batch(rr,device)
            points=t[:,None].expand(-1,10,-1).clone();points[:,1]=b
            masks=v[:,None].expand(-1,10).clone();masks[:,1]=bv
            x,points,vm=augment_batch(x,points,masks,generator)
            t=points[:,0];b=points[:,1];v=vm[:,0];bv=bv & torch.isfinite(t).all(-1)&torch.isfinite(b).all(-1)
            variants+=len(rr)
            with torch.no_grad():coarse=model.coarse(F.interpolate(x,(120,160),mode='bilinear',align_corners=False))
            jitter=(torch.rand((len(rr),),device=device,generator=generator)-.5)*.15
            teacher=t[:,1]+jitter
            # Real hidden-top examples use the image proposal; sim labels may guide a crop even amodally.
            choose=torch.rand((len(rr),),device=device,generator=generator)<(.35 if epoch>=15 else 0)
            centers=torch.where(choose,coarse['y'],teacher)
            out=model(x,roi_y=centers,axis_x=axis,roi_size_px=160*x.shape[-1]/640)
            fine_target=(t[:,1]-out['origin'][:,1])/out['scale'][:,1]
            fine_valid=v&(fine_target>=0)&(fine_target<=1)
            top_loss=heatmap_loss(out['coarse'],t[:,1],v)+heatmap_loss(out['fine'],fine_target,fine_valid,sigma=.025)
            vis_loss=F.binary_cross_entropy_with_logits(out['coarse']['visibility_logits'],v.float())+F.binary_cross_entropy_with_logits(out['fine']['visibility_logits'],fine_valid.float())
            offset=b[:,1]-t[:,1]
            offset_loss=(F.smooth_l1_loss(out['fine']['offset']*480,offset*480,reduction='none')*bv).sum()/bv.sum().clamp_min(1)
            loss=top_loss+vis_loss+.15*offset_loss
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        if epoch%5==0 or epoch==args.epochs-1:
            val=evaluate(model,validation,device,axis)
            real_error=val['real']['top_y_error_px_480'].get('mean',100)
            sim_error=val['sim']['bottom_y_error_px_480'].get('mean',100)
            shadow_val=evaluate(model,validation,device,axis,shadow_seed=args.seed+9000)
            shadow_real=shadow_val['real']['top_y_error_px_480'].get('mean',100)
            shadow_sim=shadow_val['sim']['bottom_y_error_px_480'].get('mean',100)
            score=real_error+.2*sim_error+.5*shadow_real+.1*shadow_sim
            summary={'epoch':epoch+1,'loss':float(np.mean(losses)),'real_validation_top_y_px':real_error,'sim_validation_bottom_y_px':sim_error,'selection_score':score,'shadow_real_validation_top_y_px':shadow_real,'shadow_sim_validation_bottom_y_px':shadow_sim}
            history.append(summary);print(json.dumps(summary),flush=True)
            if score<best:
                best=score;metadata['selected_epoch']=epoch+1;metadata['selection_score']=score
                torch.save({'model':model.state_dict(),'metadata':metadata},dest/'best.pt')
            (dest/'history.json').write_text(json.dumps(history,indent=2))
    payload=torch.load(dest/'best.pt',map_location=device,weights_only=True);model.load_state_dict(payload['model'])
    result={'metadata':payload['metadata'],'history':history,'training_seconds':time.time()-start_time,'augmentation_sample_variants':variants,
            'unique_real_train':len(real_train),'sim_training_samples':len(sim_train),'sim_training_pose_groups':sorted(set(r['group_id'] for r in sim_train)),'validation':evaluate(model,validation,device,axis),'test':evaluate(model,test,device,axis),
            'shadow_validation':evaluate(model,validation,device,axis,shadow_seed=args.seed+9000),
            'shadow_test':evaluate(model,test,device,axis,shadow_seed=args.seed+19000)}
    mean=float(np.mean([r['top'][1] for r in real_train if r['top_visible']]))
    for split,rr in [('validation',validation),('test',test)]:
        result[split]['constant_real_training_top_y_baseline_px_480']=summarize([abs(r['top'][1]-mean)*480 for r in rr if r['domain']=='real' and r['top_visible']])
    (dest/'metrics.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--real-manifest',required=True);p.add_argument('--sim-dataset',required=True);p.add_argument('--output',default='work/held-pin-cnn')
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--lr',type=float,default=.0006);p.add_argument('--seed',type=int,default=721);p.add_argument('--real-repeat',type=int,default=8)
    p.add_argument('--visual-initialization');p.add_argument('--resume');train(p.parse_args())
