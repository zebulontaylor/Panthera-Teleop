#!/usr/bin/env python3
"""Duration-bounded offline paired-reference training; never opens test image pixels."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from app.pin_insertion_cnn import PinInsertionPredictor,joint_features
from app.pin_reference_cnn import PinReferenceCNN,SCHEMA,paired_tensor
from scripts.adapt_pin_insertion_cnn import read_manifest
from scripts.train_pin_insertion_cnn import heatmap_supervision_loss


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def atomic_save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)

def continuation_metadata(manifest_path,ancestor_metadata):
    paired=json.loads(Path(manifest_path).read_text())
    source=Path(paired['source_manifest_path'])
    expected=ancestor_metadata.get('adaptation',{}).get('manifest_sha256')
    if sha(source)!=expected or paired.get('source_manifest_sha256')!=expected:
        raise ValueError('continued adaptation requires exact ancestor source manifest SHA')
    original=json.loads(source.read_text())
    roles={r['id']:(r['episode_id'],r['split']) for r in original['frames']}
    if roles!={r['id']:(r['episode_id'],r['split']) for r in paired['frames']}:
        raise ValueError('continued adaptation must preserve every frozen frame/episode role')
    source_rows={r['id']:r for r in original['frames']}
    for row in paired['frames']:
        previous=source_rows[row['id']]
        for key in ('image_path','image_sha256','state14'):
            if row.get(key)!=previous.get(key):raise ValueError('continued adaptation source image/state identity changed')
    adjusted=copy.deepcopy(ancestor_metadata)
    # Only explicitly verified same-partition development IDs may be exempted.
    # Other inherited optimization/selection identities retain leakage protection.
    permitted={r['episode_id'] for r in original['frames'] if r['split'] in ('train','validation')}
    for key in ('train_episode_ids','validation_episode_ids','ancestor_optimization_and_selection_ids'):
        adaptation=adjusted.get('adaptation',{})
        adaptation[key]=[v for v in adaptation.get(key,[]) if v not in permitted]
    return adjusted

def load_pairs(manifest_path, ancestor_metadata, weak_manifest=None):
    manifest=read_manifest(manifest_path,continuation_metadata(manifest_path,ancestor_metadata))
    if weak_manifest:
        extra=json.loads(Path(weak_manifest).read_text())['frames']
        reserved={r['episode_id'] for r in manifest['frames'] if r['split']!='train'}
        ids={r['id'] for r in manifest['frames']}
        for row in extra:
            if row.get('split')!='train' or row.get('episode_id') in reserved or row.get('id') in ids or row.get('weak_pin_box_px') is None:
                raise ValueError('weak manifest must contain distinct explicitly train-only boxed frames')
            joint_features(row['state14'])
            if sha(row['image_path'])!=row['image_sha256']:raise ValueError('weak current image SHA mismatch')
        manifest['frames'].extend(extra)
    rows=[]
    # Contract validation above reads no held-out test pixels. Reference validation
    # is deliberately inside this filtered loop for the same guarantee.
    for row in manifest['frames']:
        if row['split'] not in ('train','validation') or row.get('excluded_from_paired_training',False):continue
        if row.get('weak_pin_box_px') is not None:
            if row['split']!='train' or row['valid'][0]:raise ValueError('weak pin boxes require train-only masked exact pin supervision')
            box=np.asarray(row['weak_pin_box_px'],float)
            if box.shape!=(4,) or not np.isfinite(box).all() or not (0<=box[0]<box[2]<=row['image_width'] and 0<=box[1]<box[3]<=row['image_height']):raise ValueError('invalid native-pixel weak box')
        path=Path(row['reference_image_path'])
        if not path.is_absolute():path=Path(manifest_path).resolve().parent/path
        if sha(path)!=row['reference_image_sha256']:raise ValueError('reference image SHA mismatch')
        provenance=row.get('reference_provenance',{})
        if provenance.get('causal_reference') is not True:raise ValueError('explicit causal clean reference provenance required')
        current=np.asarray(Image.open(row['image_path']).convert('RGB'));reference=np.asarray(Image.open(path).convert('RGB'))
        if current.shape!=reference.shape:raise ValueError('reference/current full-frame dimensions mismatch')
        points=np.array([0 if x is None else x for x in row['keypoints_px']],np.float32).reshape(2,2)/[current.shape[1],current.shape[0]]
        rows.append(dict(row,current=current,reference=reference,points=points.astype(np.float32)))
    for split in ('train','validation'):
        selected=[r for r in rows if r['split']==split]
        if not selected or not all(any(r['valid'][j] for r in selected) for j in (0,1)):
            raise ValueError('each split needs valid pin and hole supervision')
    return manifest,rows

def geometric_pair(current,reference,points,valid,rng):
    import cv2
    h,w=current.shape[:2]
    matrix=cv2.getRotationMatrix2D((w/2,h/2),float(rng.uniform(-4,4)),float(rng.uniform(.95,1.05)))
    matrix[:,2]+=rng.uniform(-.025,.025,2)*[w,h]
    def warp(image,transform):return cv2.warpAffine(image,transform,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    current=warp(current,matrix);reference=warp(reference,matrix)
    xy=np.asarray(points)*[w,h];xy=xy@matrix[:,:2].T+matrix[:,2]
    points=xy/[w,h];valid=np.asarray(valid)&((points>=0)&(points<=1)).all(1)
    jitter=np.array([[1,0,rng.uniform(-1.5,1.5)],[0,1,rng.uniform(-1.5,1.5)]],np.float32)
    reference=warp(reference,jitter)
    return current,reference,points.clip(0,1).astype(np.float32),valid

def batch(rows,size,device,rng=None):
    xs=[];qs=[];ks=[];vs=[];boxes=[]
    for row in rows:
        current=row['current'];reference=row['reference'];points=row['points'].copy();valid=np.array(row['valid'],bool)
        if rng is not None:
            from scripts.pin_reference_augmentation import augment_pair
            if row.get('weak_pin_box_px') is None:
                current,reference,points,valid=geometric_pair(current,reference,points,valid,rng)
            current,reference,points,valid=augment_pair(current,reference,points,valid,rng)
        box=row.get('weak_pin_box_px')
        boxes.append(np.asarray(box,np.float32)/np.array([current.shape[1],current.shape[0]]*2) if box is not None else np.full(4,np.nan,np.float32))
        xs.append(paired_tensor(current,reference,size));qs.append(joint_features(row['state14']));ks.append(np.asarray(points).reshape(4));vs.append(np.repeat(valid/(1+(np.asarray(row.get('_uncertainty',row.get('uncertainty_px',[0.,0.])),dtype=float)/4)**2),2))
    return torch.cat(xs).to(device),torch.tensor(np.asarray(qs),device=device),torch.tensor(np.asarray(ks),device=device,dtype=torch.float32),torch.tensor(np.asarray(vs),device=device,dtype=torch.float32),torch.tensor(np.asarray(boxes),device=device,dtype=torch.float32)

def weak_box_loss(logits,boxes):
    """Weak region probability mass, without fabricating an exact endpoint."""
    pin=logits[:,0];n,h,w=pin.shape;losses=[]
    for i,box in enumerate(boxes):
        if not torch.isfinite(box).all():continue
        x0,y0,x1,y1=box
        if not (0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('invalid normalized weak pin box')
        # Include every score cell intersecting the broad annotated region.
        left=max(0,min(w-1,int(torch.floor(x0*w))));right=max(left+1,min(w,int(torch.ceil(x1*w))))
        top=max(0,min(h-1,int(torch.floor(y0*h))));bottom=max(top+1,min(h,int(torch.ceil(y1*h))))
        losses.append(torch.logsumexp(pin[i].flatten(),0)-torch.logsumexp(pin[i,top:bottom,left:right].flatten(),0))
    return torch.stack(losses).mean() if losses else logits.sum()*0

def evaluate(model,rows,args,device):
    model.eval();errors=[[],[]]
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            selected=rows[start:start+args.batch_size];x,q,k,v,_=batch(selected,args.image_size,device)
            predicted=model(x,q)['keypoints'].cpu().numpy().reshape(-1,2,2)
            for i,row in enumerate(selected):
                wh=np.array([row['current'].shape[1],row['current'].shape[0]])
                for j in (0,1):
                    if row['valid'][j]:errors[j].append(float(np.linalg.norm((predicted[i,j]-row['points'][j])*wh)))
    means=[float(np.mean(e)) for e in errors]
    return {'balanced_mean_error_px':float(np.mean(means)),'pin_mean_error_px':means[0],'hole_mean_error_px':means[1],'label_counts':[len(e) for e in errors]}

def train(args):
    if args.duration_seconds<=0 or min(args.image_size,args.batch_size,args.validation_epochs)<1 or args.lr<=0:raise ValueError('positive training settings required')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);device=torch.device(args.device)
    ancestor=PinInsertionPredictor(args.checkpoint,'cpu')
    if ancestor.metadata.get('model_variant')!='heatmap':raise ValueError('heatmap RGB ancestor required')
    manifest,rows=load_pairs(args.manifest,ancestor.metadata,args.weak_manifest)
    train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='validation']
    model=PinReferenceCNN.from_rgb_model(ancestor.model).to(device)
    # Neither gate nor XYZ is supervised or eligible for runtime use.
    for head in (model.gate,model.delta):
        for parameter in head.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    metadata=copy.deepcopy(ancestor.metadata);metadata.update(schema=SCHEMA,model_variant='paired_reference_heatmap',image_size=args.image_size,localization_only=True,direct_xyz_disabled=True,gate_unverified=True,input_channels=['current_rgb','aligned_clean_reference_rgb','signed_current_minus_reference'],ancestor_checkpoint_sha256=sha(args.checkpoint),paired_manifest_sha256=sha(args.manifest),test_used_for_training_or_selection=False)
    metadata['paired_training']={'arguments':vars(args),'selection':'validation balanced pin/hole native pixel error','split_audit':manifest['_split_audit'],'ancestor_development_partitions_reused':True,'weak_manifest_sha256':sha(args.weak_manifest) if args.weak_manifest else None,'uncertainty_weighting':'valid/(1+(uncertainty_px/4)^2)','weak_pin_box_count':sum(r.get('weak_pin_box_px') is not None for r in train_rows),'weak_box_frames_disable_geometric_augmentation':True,'validation_label_counts':[sum(r['valid'][j] for r in val_rows) for j in (0,1)],'limitation':'Small within-recording validation does not establish independent-session generalization. Endpoint labels remain AI annotations unless individually human verified.'}
    epoch=0;best=float('inf');steps=0
    if args.resume:
        payload=torch.load(args.resume,map_location=device,weights_only=True)
        if payload['metadata']['paired_manifest_sha256']!=metadata['paired_manifest_sha256']:raise ValueError('resume manifest mismatch')
        model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer']);epoch=payload['epoch'];steps=payload['steps'];best=payload['best_validation_score']
    stop={'requested':False,'signal':None}
    def handle(signum,frame):stop.update(requested=True,signal=signum)
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    started=time.monotonic();deadline=started+args.duration_seconds;wall_started=time.time();last_validation=started
    status={'state':'running','pid':os.getpid(),'started_unix_s':wall_started,'deadline_unix_s':wall_started+args.duration_seconds,'duration_seconds':args.duration_seconds,'test_images_read':False,'output_dir':str(output.resolve())}
    atomic_json(output/'run-status.json',status)
    def save(name):
        atomic_save(output/name,{'model':model.state_dict(),'metadata':metadata,'optimizer':optimizer.state_dict(),'epoch':epoch,'steps':steps,'best_validation_score':best,'elapsed_seconds':time.monotonic()-started})
    try:
        with (output/'metrics.jsonl').open('a',buffering=1) as log:
            initial=evaluate(model,val_rows,args,device)
            if initial['balanced_mean_error_px']<=best or not (output/'best.pt').exists():
                best=initial['balanced_mean_error_px'];save('best.pt')
            save('last.pt')
            log.write(json.dumps(dict(initial,epoch=epoch,steps=steps,elapsed_seconds=0,kind='initial_validation'))+'\n')
            while time.monotonic()<deadline and not stop['requested']:
                model.train();losses=[];order=rng.permutation(len(train_rows))
                for start in range(0,len(order),args.batch_size):
                    if time.monotonic()>=deadline or stop['requested']:break
                    elapsed=time.monotonic()-started;lr=1e-6+.5*(args.lr-1e-6)*(1+math.cos(math.pi*min(elapsed/args.duration_seconds,1)))
                    for group in optimizer.param_groups:group['lr']=lr
                    selected=[train_rows[i] for i in order[start:start+args.batch_size]];x,q,k,v,boxes=batch(selected,args.image_size,device,rng)
                    out=model(x,q,return_logits=True);coordinate=((out['keypoints']-k).square()*v).sum()/v.sum().clamp_min(1)
                    loss=1000*coordinate+10*heatmap_supervision_loss(out['keypoint_logits'],k,v)+args.weak_pin_weight*weak_box_loss(out['keypoint_logits'],boxes)
                    if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();steps+=1;losses.append(float(loss.detach()))
                epoch+=1
                due=epoch%args.validation_epochs==0 or time.monotonic()-last_validation>=30 or time.monotonic()>=deadline or stop['requested']
                if due:
                    metric=evaluate(model,val_rows,args,device);last_validation=time.monotonic();score=metric['balanced_mean_error_px'];improved=score<best
                    if improved:best=score
                    record=dict(metric,epoch=epoch,steps=steps,elapsed_seconds=time.monotonic()-started,train_loss=float(np.mean(losses)) if losses else None,lr=optimizer.param_groups[0]['lr'],best_validation_score=best)
                    log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                    if improved:save('best.pt')
                    save('last.pt');status.update(epoch=epoch,steps=steps,latest_validation=metric,elapsed_seconds=record['elapsed_seconds']);atomic_json(output/'run-status.json',status)
            save('last.pt');status.update(state='interrupted' if stop['requested'] else 'completed',signal=stop['signal'],completed_unix_s=time.time(),elapsed_seconds=time.monotonic()-started,epoch=epoch,steps=steps,best_validation_score=best);atomic_json(output/'run-status.json',status)
    except BaseException as exc:
        save('last.pt');status.update(state='failed',error=str(exc),completed_unix_s=time.time());atomic_json(output/'run-status.json',status);raise
    return status

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--checkpoint',default=str(ROOT/'work/pin-insertion-newdemo/proprio.pt'));p.add_argument('--duration-seconds',type=float,default=3600);p.add_argument('--device',default='cuda');p.add_argument('--image-size',type=int,default=320);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--validation-epochs',type=int,default=20);p.add_argument('--weak-manifest');p.add_argument('--weak-pin-weight',type=float,default=5.);p.add_argument('--resume');train(p.parse_args())
if __name__=='__main__':main()
