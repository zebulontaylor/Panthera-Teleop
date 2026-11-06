#!/usr/bin/env python3
"""Train a synthetic pin alignment CNN with seed-group-disjoint validation."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from app.pin_insertion_cnn import build_model, SCHEMA


def validate_dataset_metadata(metadata):
    expected = {'schema_version': 1, 'generator': 'pin_insertion_sim.py',
        'camera': 'right_wrist', 'controlled_object': 'pin', 'controlled_arm': 'left',
        'delta_frame': 'right_native_base', 'delta_units': 'meters',
        'state_order': 'left joints1..6,gripper,right joints1..6,gripper (radians)',
        'keypoints_order': ['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v']}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'dataset semantic mismatch: {key} must be {value!r}')
    return metadata


def group_split(groups, seed=0, fraction=.2, labels=None):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('at least two independent seed groups required')
    rng = np.random.default_rng(seed)
    fallback = None
    for _ in range(512 if labels is not None else 1):
        order = rng.permutation(unique)
        val = order[:max(1, min(len(unique)-1, round(len(unique)*fraction)))]
        mask = np.isin(groups, val)
        indices = np.flatnonzero(~mask), np.flatnonzero(mask)
        if labels is None: return indices
        if len(np.unique(labels[indices[0]])) == 2:
            fallback = indices
            if len(np.unique(labels[indices[1]])) == 2: return indices
    if fallback is not None: return fallback
    raise ValueError('no group-disjoint split can provide both training gate classes')


def augment(images, keypoints=None):
    """Photometric changes only; no geometric label transformations required."""
    n, _, h, w = images.shape
    images = images * torch.empty(n, 1, 1, 1, device=images.device).uniform_(.65, 1.35)
    images = images + torch.empty(n, 3, 1, 1, device=images.device).uniform_(-.12, .12)
    images = images + torch.randn_like(images) * .025
    if torch.rand(()).item() < .35:
        images = F.avg_pool2d(images, 3, stride=1, padding=1)
    for i in range(n):
        if torch.rand(()).item() < .25:
            y, x = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
            y2, x2 = min(h, y+h//6), min(w, x+w//6)
            points = keypoints[i].reshape(2,2) if keypoints is not None else None
            # Retain target visibility and labels: never cover either landmark or its local context.
            covers = points is None or bool((((points[:,0]*w >= x-6) & (points[:,0]*w <= x2+6)) & ((points[:,1]*h >= y-6) & (points[:,1]*h <= y2+6))).any())
            if not covers:
                images[i, :, y:y2, x:x2] = torch.rand(3, 1, 1, device=images.device)
    return images.clamp(0, 1)


def heatmap_supervision_loss(logits, keypoints, validity, sigma=1.5):
    """Cross entropy to normalized Gaussian landmark targets on score-map centers."""
    n,_,h,w=logits.shape
    target_xy=keypoints.reshape(n,2,2)
    xs=(torch.arange(w,device=logits.device,dtype=logits.dtype)+.5)/w
    ys=(torch.arange(h,device=logits.device,dtype=logits.dtype)+.5)/h
    distance=((xs[None,None,None,:]-target_xy[:,:,0,None,None])*w).square()+((ys[None,None,:,None]-target_xy[:,:,1,None,None])*h).square()
    target=torch.exp(-distance/(2*sigma*sigma)).flatten(2)
    target=target/target.sum(dim=-1,keepdim=True).clamp_min(1e-20)
    loss=-(target*logits.flatten(2).log_softmax(dim=-1)).sum(dim=-1)
    mask=validity.reshape(n,2,2).amin(dim=-1)
    return (loss*mask).sum()/mask.sum().clamp_min(1)


def train(args):
    if not np.isfinite([args.keypoint_weight,args.delta_weight]).all() or args.keypoint_weight <= 0 or args.delta_weight < 0:
        raise ValueError('invalid keypoint/delta loss weights')
    if not np.isfinite(args.heatmap_weight) or args.heatmap_weight < 0:
        raise ValueError('heatmap weight must be nonnegative and finite')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    with np.load(args.dataset, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    sidecar = Path(args.dataset).with_suffix('.json')
    if not sidecar.exists(): raise ValueError('dataset semantic metadata .json sidecar required')
    dataset_metadata = validate_dataset_metadata(json.loads(sidecar.read_text()))
    if 'metadata_json' in arrays and json.loads(str(arrays['metadata_json'].item())) != dataset_metadata:
        raise ValueError('embedded and sidecar dataset metadata disagree')
    images = arrays['images']
    n = len(images)
    state = arrays['state']
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('images must be uint8 NHWC RGB')
    if state.shape != (n,14) or not np.isfinite(state).all():
        raise ValueError('finite Nx14 state with both measured arms required')
    state = np.concatenate((state[:,:6],state[:,7:13]),axis=1)
    kp, delta = arrays['keypoints'], arrays['delta_xyz_m']
    active = arrays['active'].reshape(-1,1)
    valid = arrays.get('keypoints_valid', np.ones((n,4), dtype=np.float32))
    if valid.shape == (n,): valid = np.repeat(valid[:,None], 4, axis=1)
    if valid.shape == (n,2): valid = np.repeat(valid, 2, axis=1)
    if kp.shape != (n,4) or delta.shape != (n,3) or active.shape != (n,1) or valid.shape != (n,4):
        raise ValueError('invalid target dimensions')
    for value in (state,kp,delta,active,valid):
        if not np.isfinite(value).all(): raise ValueError('nonfinite dataset')
    if (kp < 0).any() or (kp > 1).any() or not np.isin(active,[0,1]).all() or (valid < 0).any() or (valid > 1).any():
        raise ValueError('invalid normalized keypoints or active labels')
    groups = arrays.get('group_ids', arrays.get('groups', arrays.get('group_id')))
    if groups is None or groups.shape != (n,): raise ValueError('group_ids Nx1 flattened required')
    train_idx,val_idx = group_split(groups,args.seed,labels=active[:,0])
    image = torch.from_numpy(images.copy()).permute(0,3,1,2).float()/255
    image = F.interpolate(image,size=(args.image_size,args.image_size),mode='bilinear',align_corners=False)
    tensors = [torch.from_numpy(np.asarray(v,dtype=np.float32)) for v in (state,kp,delta,active,valid)]
    model = build_model(args.model_variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    labels = active[train_idx]
    if labels.sum() == 0 or labels.sum() == len(labels): raise ValueError('training requires both active and inactive samples')
    pos_weight = torch.tensor([(len(labels)-labels.sum())/labels.sum()],device=device)
    def batch(indices, training):
        x = image[indices].to(device)
        q,k,d,a,v = [t[indices].to(device) for t in tensors]
        out = model(augment(x,k) if training else x,q,**({'return_logits':True} if args.model_variant=='heatmap' else {}))
        kmask = v
        kloss = ((out['keypoints']-k).square()*kmask).sum()/kmask.sum().clamp_min(1)
        dloss = (((out['delta_xyz_m']-d)*1000).square()*a).sum()/(a.sum()*3).clamp_min(1)
        gate = F.binary_cross_entropy_with_logits(out['active_logit'],a,pos_weight=pos_weight)
        spatial_loss=heatmap_supervision_loss(out['keypoint_logits'],k,v) if args.model_variant=='heatmap' else kloss.new_zeros(())
        return args.keypoint_weight*kloss+args.heatmap_weight*spatial_loss+args.delta_weight*dloss+gate,out,(k,d,a,v)
    train_valid=valid[train_idx]
    train_mean_keypoints=(kp[train_idx]*train_valid).sum(axis=0)/np.maximum(train_valid.sum(axis=0),1)
    history=[]; best=float('inf')
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    metadata={'schema':SCHEMA,'model_variant':args.model_variant,'seed':args.seed,'image_size':args.image_size,
        'state_convention':'native14: left6,left_gripper,right6,right_gripper; right slice 7:13',
        'delta_frame':'hole minus pin; right arm base; metres; LEFT pin correction only',
        'controlled_arm':'left','coordinate_frame':'right_base','requires_calibrated_left_base_transform':True,
        'dataset_sha256':digest,'train_groups':np.unique(groups[train_idx]).tolist(),
        'val_groups':np.unique(groups[val_idx]).tolist(),'arguments':vars(args),
        'torch_version':str(torch.__version__),'synthetic_only':True}
    metadata['heatmap_supervision']='Gaussian cross entropy sigma1.5 score-map pixels plus coordinate MSE' if args.model_variant=='heatmap' else None
    metadata['train_mean_keypoints']=train_mean_keypoints.tolist()
    metadata['checkpoint_selection']=args.select_by
    metadata['keypoint_supervision']='all position-valid landmarks, including inactive aligned/hover/away states'
    metadata['delta_supervision']='active gate samples only; no correction supervision for inactive states'
    metadata['geometry_supervision']='synthetic CAD hypotheses; visibility and amodal pin-tip semantics from dataset_metadata; not real geometric calibration'
    metadata['dataset_metadata']=dataset_metadata
    metadata['validation_gate_classes']=np.unique(active[val_idx]).tolist()
    for epoch in range(args.epochs):
        model.train(); losses=[]
        order=np.random.permutation(train_idx)
        for start in range(0,len(order),args.batch_size):
            loss,_,_=batch(order[start:start+args.batch_size],True)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); predictions=[]; truth=[]; validation=[]
        with torch.inference_mode():
            for start in range(0,len(val_idx),args.batch_size):
                loss,out,target=batch(val_idx[start:start+args.batch_size],False)
                validation.append((float(loss),len(target[0])))
                predictions.append({k:v.cpu() for k,v in out.items() if k!='keypoint_logits'}); truth.append(tuple(t.cpu() for t in target))
        pred={k:torch.cat([p[k] for p in predictions]) for k in predictions[0]}
        k,d,a,v=[torch.cat([t[j] for t in truth]) for j in range(4)]
        mask=a[:,0].bool(); gate=pred['active_logit'][:,0]>=0
        metrics={'epoch':epoch+1,'train_loss':float(np.mean(losses)),
            'val_loss':sum(l*n for l,n in validation)/len(val_idx),
            'gate_accuracy':float((gate==mask).float().mean()),
            'gate_precision':float((gate&mask).sum()/gate.sum().clamp_min(1)),
            'gate_recall':float((gate&mask).sum()/mask.sum().clamp_min(1)),
            'active_delta_mae_mm':float((pred['delta_xyz_m'][mask]-d[mask]).abs().mean()*1000) if mask.any() else None,
            'active_zero_delta_baseline_mae_mm':float(d[mask].abs().mean()*1000) if mask.any() else None,
            'active_direction_cosine':float(F.cosine_similarity(pred['delta_xyz_m'][mask],d[mask],dim=1).mean()) if mask.any() else None,
            'active_center_keypoint_baseline_mae':float(((.5-k).abs()*a*v).sum()/(a*v).sum().clamp_min(1)),
            'all_visible_keypoint_mae':float(((pred['keypoints']-k).abs()*v).sum()/v.sum().clamp_min(1)),
            'all_visible_center_keypoint_baseline_mae':float(((.5-k).abs()*v).sum()/v.sum().clamp_min(1)),
            'active_keypoint_mae':float(((pred['keypoints']-k).abs()*a*v).sum()/(a*v).sum().clamp_min(1))}
        mean_baseline=torch.tensor(train_mean_keypoints,dtype=k.dtype)
        metrics['all_visible_train_mean_keypoint_baseline_mae']=float(((mean_baseline-k).abs()*v).sum()/v.sum().clamp_min(1))
        for landmark,j in (('pin_tip',0),('target_hole',1)):
            sl=slice(2*j,2*j+2)
            metrics[f'{landmark}_train_mean_keypoint_baseline_mae']=float(((mean_baseline[sl]-k[:,sl]).abs()*v[:,sl]).sum()/v[:,sl].sum().clamp_min(1))
        history.append(metrics); print(json.dumps(metrics),flush=True)
        selection=metrics['all_visible_keypoint_mae'] if args.select_by=='keypoint' else metrics['val_loss']
        if selection < best:
            best=selection; metadata['best_metrics']=metrics
            temporary=output.with_suffix(output.suffix+'.tmp')
            torch.save({'model':model.state_dict(),'metadata':metadata},temporary); temporary.replace(output)
    report={'metadata':metadata,'history':history,'train_samples':len(train_idx),'val_samples':len(val_idx)}
    output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=20); parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--image-size',type=int,default=96); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--keypoint-weight',type=float,default=300.)
    parser.add_argument('--delta-weight',type=float,default=1.)
    parser.add_argument('--select-by',choices=['val_loss','keypoint'],default='val_loss')
    parser.add_argument('--model-variant',choices=['regression','heatmap'],default='regression')
    parser.add_argument('--heatmap-weight',type=float,default=10.)
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.image_size,args.threads)<1: parser.error('positive training settings required')
    if not np.isfinite([args.keypoint_weight,args.delta_weight]).all() or args.keypoint_weight <= 0 or args.delta_weight < 0:
        parser.error('keypoint weight must be positive and delta weight nonnegative and finite')
    train(args)

if __name__=='__main__': main()
