#!/usr/bin/env python3
"""Adapt a pretrained heatmap localizer using explicitly declared real-data splits.

Test image pixels and labels are never used for optimization or selection. XYZ is
unsupervised and disabled in metadata. This script does not evaluate held-out test
performance; use a separate evaluator after model selection.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from app.pin_insertion_cnn import PinInsertionPredictor, joint_features
from scripts.train_pin_insertion_cnn import heatmap_supervision_loss, validate_dataset_metadata


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_groups(metadata):
    ids=set(map(str,metadata.get('train_groups',[])+metadata.get('val_groups',[])))
    adaptation=metadata.get('adaptation',{})
    ids.update(map(str,adaptation.get('train_episode_ids',[])+adaptation.get('validation_episode_ids',[])+adaptation.get('ancestor_optimization_and_selection_ids',[])))
    ids.update(str(x['id']) for x in metadata.get('dataset_metadata',{}).get('sources',[]) if 'id' in x)
    return ids


BLOCK_SPLIT_UNITS={'carrier_pose_block','stationary_carrier_pose_block_within_one_recording'}


def validate_block_contract(manifest,manifest_path):
    """Verify a frozen within-recording block split without claiming session independence."""
    unit=manifest.get('split_unit')
    if unit not in BLOCK_SPLIT_UNITS:return None
    definition=manifest.get('split_definition_path')
    expected_hash=manifest.get('split_definition_sha256')
    if not isinstance(definition,str) or not definition:raise ValueError('block split requires frozen split_definition_path')
    definition=Path(definition)
    if not definition.is_absolute():definition=Path(manifest_path).resolve().parent/definition
    if not isinstance(expected_hash,str) or sha256(definition)!=expected_hash:raise ValueError('frozen block split definition SHA mismatch')
    frozen=json.loads(definition.read_text())
    if frozen.get('split_unit') not in BLOCK_SPLIT_UNITS or frozen.get('split_frozen_before_coordinate_annotation_or_fitting') is not True:
        raise ValueError('block split must be explicitly frozen before labels/fitting')
    recording=frozen.get('source_recording_id')
    if not isinstance(recording,str) or not recording:raise ValueError('frozen source recording ID required')
    if frozen.get('independent_recordings')!=1:raise ValueError('within-recording split must declare exactly one recording')
    definitions=frozen.get('blocks')
    if not isinstance(definitions,list) or not definitions:raise ValueError('frozen carrier-pose blocks required')
    blocks={};ranges=[]
    for block in definitions:
        bid=block.get('block_id');split=block.get('split');first=block.get('first_frame');last=block.get('last_frame')
        if not isinstance(bid,str) or not bid or bid in blocks or split not in ('train','validation','test'):raise ValueError('invalid frozen block identity/split')
        if type(first) is not int or type(last) is not int or not 0<=first<=last:raise ValueError('invalid frozen frame interval')
        start,end=block.get('start_s'),block.get('end_s')
        if not isinstance(start,(int,float)) or not isinstance(end,(int,float)) or not np.isfinite([start,end]).all() or not 0<=start<end:raise ValueError('invalid frozen block time interval')
        for oldfirst,oldlast,oldstart,oldend in ranges:
            if max(first,oldfirst)<=min(last,oldlast) or max(start,oldstart)<min(end,oldend):raise ValueError('frozen block intervals overlap')
        ranges.append((first,last,start,end));blocks[bid]=block
    frame_ids=set();hash_splits={};sessions={};seen_splits=set()
    for row in manifest['frames']:
        bid=row.get('block_id');block=blocks.get(bid);index=row.get('source_frame_index');split=row.get('split')
        if block is None or row.get('source_recording_id')!=recording or row.get('episode_id')!=recording+'::'+bid:
            raise ValueError('block frame requires truthful source_recording_id::block_id unit')
        if split!=block['split'] or type(index) is not int or not block['first_frame']<=index<=block['last_frame']:raise ValueError('frame violates frozen block split/bounds')
        identity=(recording,index)
        if identity in frame_ids:raise ValueError('duplicate source frame in block manifest')
        frame_ids.add(identity);seen_splits.add(split)
        digest=row.get('image_sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('block frames require image_sha256')
        if digest in hash_splits and hash_splits[digest]!=split:raise ValueError('image hash leakage across block splits')
        hash_splits[digest]=split
        image=Path(row.get('image_path',''))
        if not image.is_absolute():image=Path(manifest_path).resolve().parent/image
        if split!='test' and sha256(image)!=digest:raise ValueError('block train/validation image SHA mismatch')
        source_time=row.get('actual_episode_time_s',row.get('source_time_s',row.get('data_time_s')))
        if source_time is not None and (not isinstance(source_time,(int,float)) or not np.isfinite(source_time) or not block['start_s']<=source_time<=block['end_s']):raise ValueError('frame violates frozen block time bounds')
        session=row.get('session_id',recording)
        if not isinstance(session,str) or not session:raise ValueError('actual shared session identifier required')
        sessions.setdefault(session,set()).add(split)
    if not {'train','validation'}<=seen_splits:raise ValueError('block split requires train and validation frames')
    return {'split_unit':'stationary_carrier_pose_block_within_one_recording','definition_path':str(definition.resolve()),
        'definition_sha256':expected_hash,'source_recording_ids':[recording],'independent_recordings':1,
        'episode_id_semantics':'source recording ID plus carrier-pose block ID; these are NOT independent episodes',
        'session_disjoint':False,'independent_episode_holdout':False,
        'shared_sessions_across_splits':sorted(k for k,v in sessions.items() if len(v)>1),
        'train_validation_image_hashes_verified':True,'test_image_pixels_read_by_trainer':False,
        'limitations':['Within one recording and likely the same grasp; held-out blocks do not establish independent-recording/session generalization.',
            'Separate recordings remain necessary for an independent test; block IDs are partition units, not fabricated recording identities.']}


def read_manifest(path, pretrained_metadata):
    manifest=json.loads(Path(path).read_text())
    rows=manifest.get('frames')
    if not isinstance(rows,list) or not rows: raise ValueError('manifest requires nonempty frames')
    if manifest.get('split_unit') not in (None,'episode','episode_session',*BLOCK_SPLIT_UNITS):raise ValueError('unsupported manifest split unit')
    block_audit=validate_block_contract(manifest,path)
    ids=set(); episode_splits={}; session_splits={}; image_splits={}
    for row in rows:
        if not isinstance(row.get('id'),str) or not row['id'] or row['id'] in ids: raise ValueError('frame IDs must be unique nonempty strings')
        ids.add(row['id'])
        split=row.get('split')
        if split not in ('train','validation','test'): raise ValueError('invalid frame split')
        episode=row.get('episode_id')
        if not isinstance(episode,str) or not episode: raise ValueError('episode_id required')
        identities=((episode,episode_splits),) if block_audit is not None else ((episode,episode_splits),(row.get('session_id',episode),session_splits))
        for identity,mapping in identities:
            if not isinstance(identity,str) or not identity: raise ValueError('session/episode identity required')
            if identity in mapping and mapping[identity]!=split: raise ValueError('episode/session leakage across splits')
            mapping[identity]=split
        image=Path(row.get('image_path',''))
        if not image.is_absolute(): image=Path(path).resolve().parent/image
        image=str(image.resolve())
        if image in image_splits and image_splits[image]!=split: raise ValueError('image path leakage across splits')
        image_splits[image]=split
        row['image_path']=image
        # Test rows are reserved, not inspected for their supervision or pixels.
        if split=='test': continue
        joint_features(row.get('state14'))
        width,height=row.get('image_width'),row.get('image_height')
        if any(isinstance(x,bool) or not isinstance(x,int) or x<16 for x in (width,height)): raise ValueError('valid image dimensions required')
        valid=row.get('valid')
        if not isinstance(valid,list) or len(valid)!=2 or any(type(x) is not bool for x in valid): raise ValueError('valid must be two booleans')
        kp=row.get('keypoints_px')
        if kp is None and not any(valid):
            kp=[None]*4
            row['keypoints_px']=kp
        if not isinstance(kp,list) or len(kp)!=4: raise ValueError('keypoints_px requires four nullable coordinates')
        for j in range(2):
            if valid[j]:
                xy=np.asarray(kp[2*j:2*j+2],dtype=float)
                if not np.isfinite(xy).all() or (xy<0).any() or (xy>=[width,height]).any(): raise ValueError('valid landmarks must lie in the image')
        gate=row.get('gate_label')
        if gate is not None and (type(gate) not in (int,float) or gate not in (0,1)): raise ValueError('gate_label must be null, zero, or one')
        uncertainty=np.asarray(row.get('uncertainty_px',0.),dtype=float)
        if uncertainty.ndim==0: uncertainty=np.repeat(uncertainty,2)
        if uncertainty.shape!=(2,) or not np.isfinite(uncertainty).all() or (uncertainty<0).any(): raise ValueError('uncertainty_px requires nonnegative scalar or two values')
        radius=row.get('hole_radius_px')
        if radius is not None and (isinstance(radius,bool) or not isinstance(radius,(int,float)) or not np.isfinite(radius) or radius<=0): raise ValueError('hole_radius_px must be a positive audited pixel radius')
        row['_uncertainty']=uncertainty.tolist()
    if not any(r['split']=='train' for r in rows) or not any(r['split']=='validation' for r in rows): raise ValueError('both train and validation frames required')
    reserved={episode for episode,split in episode_splits.items() if split in ('validation','test')}
    if block_audit is not None: reserved.update(block_audit['source_recording_ids'])
    overlap=reserved & source_groups(pretrained_metadata)
    if overlap: raise ValueError('validation/test overlaps pretrained optimization/model-selection sources: '+','.join(sorted(overlap)))
    for split in ('train','validation'):
        if not any(r['split']==split and any(r['valid']) for r in rows): raise ValueError(f'{split} requires valid landmark supervision')
    manifest['_split_audit']=block_audit or {'split_unit':'episode_session','session_disjoint':True,'independent_episode_holdout':True}
    return manifest


def affine_sample(image, keypoints, valid, rng):
    """Apply the same modest affine map to RGB pixels and normalized landmarks."""
    import cv2
    h,w=image.shape[:2]
    matrix=cv2.getRotationMatrix2D((w/2,h/2),float(rng.uniform(-4,4)),float(rng.uniform(.95,1.05)))
    matrix[:,2]+=rng.uniform(-.03,.03,2)*[w,h]
    transformed=cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    xy=keypoints.reshape(2,2)*[w,h]
    xy=np.c_[xy,np.ones(2)]@matrix.T
    valid=valid.copy()*np.all((xy>=0)&(xy<[w,h]),axis=1)
    return transformed,(xy/[w,h]).clip(0,1).reshape(4).astype(np.float32),valid


def estimate_bore_radius(image, center_xy):
    """Uncalibrated visual rim scale, not a physical bore/contact measurement.

    Use robust radial contrast around the independently annotated center. Ambiguous
    or low-contrast estimates are rejected; audited per-frame radii take priority.
    """
    import cv2
    h,w=image.shape[:2]
    gray=cv2.GaussianBlur(image.astype(np.float32).mean(axis=2),(3,3),0)
    scale=w/640
    radii=np.linspace(max(3.,5*scale),max(6.,22*scale),36)
    angle=np.linspace(0,2*np.pi,64,endpoint=False)
    direction=np.stack((np.cos(angle),np.sin(angle)),axis=1)
    center=np.asarray(center_xy,dtype=float)
    if (center-radii[-1]-2<0).any() or (center+radii[-1]+2>=[w,h]).any(): return None
    scores=[]
    for radius in radii:
        outer=np.rint(center+direction*(radius+1.5)).astype(int)
        inner=np.rint(center+direction*(radius-1.5)).astype(int)
        contrast=np.abs(gray[outer[:,1],outer[:,0]]-gray[inner[:,1],inner[:,0]])
        scores.append(float(np.quantile(contrast,.65)))
    scores=np.asarray(scores)
    index=int(scores.argmax())
    if scores[index]<7 or scores[index]<np.median(scores)*1.2: return None
    return float(radii[index])


def partial_hole_occlusion(image,keypoints,valid,rng,radius_px=None):
    """Generic partial-hole occlusion with known synthetic amodal coordinates.

    This is not a second reconstructed pin, a contact render, or a contact label.
    Original landmark coordinates remain known. A small observed pin-tip region
    is protected; at least 45% of the nominal hole disk remains uncovered.
    """
    import cv2
    if not valid[1]: return image,None
    h,w=image.shape[:2];xy=keypoints.reshape(2,2)*[w,h];hole=xy[1]
    radius=float(radius_px) if radius_px is not None else estimate_bore_radius(image,hole)
    if radius is None or not np.isfinite(radius) or radius<2: return image,None
    yy,xx=np.ogrid[:h,:w]
    disk=(xx-hole[0])**2+(yy-hole[1])**2<=radius**2
    if disk.sum()<12:return image,None
    guard=(xx-xy[0,0])**2+(yy-xy[0,1])**2<=max(2.,radius*.18)**2 if valid[0] else np.zeros((h,w),bool)
    for _ in range(12):
        angle=rng.uniform(-np.pi,np.pi);direction=np.array([np.cos(angle),np.sin(angle)])
        side=np.array([-direction[1],direction[0]])*rng.uniform(-.6,.6)*radius
        begin=hole+side+direction*radius*1.6
        end=hole+side+direction*rng.uniform(-.15,.55)*radius
        mask=np.zeros((h,w),np.uint8)
        cv2.line(mask,tuple(np.rint(begin).astype(int)),tuple(np.rint(end).astype(int)),1,max(2,int(radius*rng.uniform(.35,.8))),cv2.LINE_8)
        mask=mask.astype(bool)&~guard
        coverage=float((mask&disk).sum()/disk.sum())
        if .12<=coverage<=.55:break
    else:return image,None
    # Sample neighboring scene material; no claim that this texture is a real pin.
    source=np.rint(hole-direction*radius*1.7).astype(int)
    sx,sy=np.clip(source,[3,3],[w-4,h-4])
    color=np.median(image[sy-3:sy+4,sx-3:sx+4],axis=(0,1))
    output=image.copy()
    material=np.clip(color+rng.normal(0,4,(int(mask.sum()),3)),0,255).astype(np.uint8)
    output[mask]=material
    return output,{'radius_px':radius,'nominal_hole_covered_fraction':coverage,'gate_supervision_masked':True}


def load_rows(rows,size):
    result=[]
    for row in rows:
        with Image.open(row['image_path']) as source: image=np.asarray(source.convert('RGB')).copy()
        h,w=image.shape[:2]
        if (w,h)!=(row['image_width'],row['image_height']): raise ValueError('decoded image dimensions disagree with annotation')
        kp=np.array([.5 if value is None else float(value) for value in row['keypoints_px']],dtype=np.float32)
        kp=(kp/np.array([w,h,w,h],dtype=np.float32)).clip(0,1)
        valid=np.asarray(row['valid'],dtype=np.float32)
        weight=valid/(1+(np.asarray(row['_uncertainty'],dtype=np.float32)/4)**2)
        result.append({'image':image,'keypoints':kp,'valid':valid,'weight':weight,'joints':joint_features(row['state14']),
            'gate':-1. if row.get('gate_label') is None else float(row['gate_label']),'contact_audit':row.get('contact_audit') is True,'hole_radius_px':row.get('hole_radius_px'),'wh':np.array([w,h],dtype=np.float32),'id':row['id']})
    return result


def make_batch(rows,size,device,rng=None,affine=False,image_only=False,hole_occlusion_probability=0.,augmentation_audit=None):
    images=[]; kp=[]; valid=[]; weights=[]; gates=[]; joints=[]
    for row in rows:
        rgb=row['image']; k=row['keypoints'].copy(); v=row['valid'].copy(); gate=row['gate']
        input_gate=gate
        if rng is not None and rng.random()<hole_occlusion_probability:
            if augmentation_audit is not None: augmentation_audit['attempted']+=1
            rgb,audit=partial_hole_occlusion(rgb,k,v,rng,row.get('hole_radius_px'))
            if audit is not None:
                gate=-1.
                if augmentation_audit is not None:
                    augmentation_audit['radii'].append(audit['radius_px'])
                    augmentation_audit['coverage'].append(audit['nominal_hole_covered_fraction'])
        if affine and rng is not None:
            before_valid=v.copy()
            rgb,k,v=affine_sample(rgb,k,v,rng)
            # Occluded landmarks can accompany protocol-confirmed contact. Only
            # newly cropped supervised evidence masks an otherwise valid label.
            if gate==1 and np.any((before_valid>0)&(v<=0)): gate=-1.
        if augmentation_audit is not None:
            for prefix,value in (('gate_input',input_gate),('gate_used',gate)):
                name=prefix+('_positive' if value==1 else '_negative' if value==0 else '_unknown')
                augmentation_audit[name]=augmentation_audit.get(name,0)+1
        x=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255
        x=F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)[0]
        images.append(x);kp.append(k);valid.append(v);weights.append(row['weight']*v);gates.append(gate);joints.append(row['joints'])
    x=torch.stack(images).to(device)
    if rng is not None:
        gain=torch.tensor(rng.uniform(.8,1.2,(len(rows),1,1,1)),dtype=x.dtype,device=device)
        bias=torch.tensor(rng.uniform(-.07,.07,(len(rows),3,1,1)),dtype=x.dtype,device=device)
        x=(x*gain+bias+torch.randn_like(x)*.012).clamp(0,1)
    q=torch.tensor(np.asarray(joints),device=device,dtype=torch.float32)
    if image_only: q.zero_()
    return x,q,torch.tensor(np.asarray(kp),device=device),torch.tensor(np.repeat(np.asarray(weights),2,axis=1),device=device),torch.tensor(gates,device=device,dtype=torch.float32)[:,None]


def localization_loss(out,keypoints,weights,gates,args):
    coordinate=((out['keypoints']-keypoints).square()*weights).sum()/weights.sum().clamp_min(1)
    heatmap=heatmap_supervision_loss(out['keypoint_logits'],keypoints,weights)
    gate_mask=gates>=0
    gate=F.binary_cross_entropy_with_logits(out['active_logit'][gate_mask],gates[gate_mask]) if gate_mask.any() else coordinate.new_zeros(())
    return args.keypoint_weight*coordinate+args.heatmap_weight*heatmap+args.gate_weight*gate


def validation(model,rows,args,device):
    model.eval();totals={name:[np.zeros(2),np.zeros(2,dtype=int)] for name in ('all','approach','contact')}
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            selected=rows[start:start+args.batch_size]
            x,q,k,_,_=make_batch(selected,args.image_size,device,image_only=args.image_only)
            out=model(x,q)['keypoints'].cpu().numpy().reshape(-1,2,2)
            target=k.cpu().numpy().reshape(-1,2,2)
            wh=np.stack([r['wh'] for r in selected])
            errors=np.linalg.norm((out-target)*wh[:,None,:],axis=-1)
            valid=np.stack([r['valid'] for r in selected]).astype(bool)
            contact=np.asarray([r.get('contact_audit',False) for r in selected],bool)
            for name,subset in (('all',np.ones(len(selected),bool)),('approach',~contact),('contact',contact)):
                mask=valid&subset[:,None]
                totals[name][0]+=(errors*mask).sum(axis=0);totals[name][1]+=mask.sum(axis=0)
    strata={}
    for name,(sums,counts) in totals.items():
        landmark_means=[float(sums[i]/counts[i]) for i in range(2) if counts[i]]
        strata[name]={'mean_error_px':float(sums.sum()/counts.sum()) if counts.sum() else None,
            'balanced_landmark_mean_error_px':float(np.mean(landmark_means)) if landmark_means else None,
            'landmarks':{landmark:{'valid_count':int(counts[i]),'mean_error_px':float(sums[i]/counts[i]) if counts[i] else None} for i,landmark in enumerate(('pin_tip','target_hole'))}}
    contact_only=getattr(args,'select_contact_only',False)
    selected='contact' if contact_only else 'all'
    score=strata[selected]['balanced_landmark_mean_error_px'] if contact_only else strata[selected]['mean_error_px']
    if score is None:raise ValueError('selected validation stratum has no valid landmark labels')
    return {'selection_mean_error_px':score,'selection_stratum':selected,
        'selection_metric':'balanced visible landmark means' if contact_only else 'pooled visible landmark mean',
        'landmarks':strata['all']['landmarks'],'strata':strata}


def load_contact_replay(path,manifest):
    """Accept only geometry-hypothesis renders sourced from this real TRAIN split."""
    sidecar=Path(path).with_suffix('.json')
    metadata=json.loads(sidecar.read_text())
    if metadata.get('diagnostic_only') or metadata.get('training_use_approved') is False:
        raise ValueError('contact replay is diagnostic only or failed geometry review')
    expected={'schema_version':1,'camera':'right_wrist','controlled_object':'pin','controlled_arm':'left',
        'delta_frame':'right_native_base','delta_units':'meters','physics_simulated':False,'real_calibrated':False,
        'state_order':'left joints1..6,gripper,right joints1..6,gripper (radians)',
        'keypoints_order':['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v']}
    for key,value in expected.items():
        if key not in metadata or metadata[key]!=value:raise ValueError(f'contact replay semantic mismatch: {key}')
    if not isinstance(metadata.get('generator'),str) or not metadata['generator']:raise ValueError('contact replay generator provenance required')
    train=[r for r in manifest['frames'] if r['split']=='train']
    episodes={r['episode_id'] for r in train};sessions={r.get('session_id',r['episode_id']) for r in train}
    association={r['episode_id']:r.get('session_id',r['episode_id']) for r in train}
    sources=metadata.get('sources')
    if not isinstance(sources,list) or not sources:raise ValueError('contact replay sources required')
    source_ids=set()
    for source in sources:
        episode=source.get('id');session=source.get('session_id')
        if episode not in episodes or session not in sessions or association.get(episode)!=session:
            raise ValueError('contact replay source episode/session is not exclusively real train')
        if source.get('split','train')!='train':raise ValueError('contact replay source split is not train')
        source_ids.add(episode)
    with np.load(path,allow_pickle=False) as archive:
        if 'metadata_json' not in archive or json.loads(str(archive['metadata_json'].item()))!=metadata:raise ValueError('contact replay embedded/sidecar metadata mismatch')
        arrays={k:archive[k] for k in ('images','state','keypoints','active','keypoints_valid','group_ids')}
        internal_split=archive['split'].copy() if 'split' in archive else None
    if metadata.get('sim_validation_session_id') is not None and internal_split is None:
        raise ValueError('contact replay declares validation session but omits internal split')
    n=len(arrays['images']);image=arrays['images'];state=arrays['state'];kp=arrays['keypoints'];active=arrays['active'];valid=arrays['keypoints_valid'];groups=arrays['group_ids']
    if not n or image.dtype!=np.uint8 or image.ndim!=4 or image.shape[-1]!=3:raise ValueError('contact replay RGB images invalid')
    if state.shape!=(n,14) or kp.shape!=(n,4) or active.shape!=(n,1) or valid.shape!=(n,4) or groups.shape!=(n,):raise ValueError('contact replay target dimensions invalid')
    if any(not np.isfinite(value).all() for value in (state,kp,active,valid)):raise ValueError('contact replay nonfinite values')
    if (kp<0).any() or (kp>1).any() or (valid<0).any() or (valid>1).any() or not np.isin(active,[0,1]).all():raise ValueError('contact replay label range invalid')
    if not set(np.unique(groups).tolist())<=source_ids or not set(np.unique(groups).tolist())<=episodes:raise ValueError('contact replay groups contain non-train sources')
    internal_audit={'provided':internal_split is not None,'train_samples':n,'validation_samples':0}
    if internal_split is not None:
        if internal_split.shape!=(n,) or not np.isin(internal_split,['train','validation']).all():raise ValueError('invalid contact replay internal split')
        train_mask=internal_split=='train';val_mask=internal_split=='validation'
        if not train_mask.any():raise ValueError('contact replay has no internal train samples')
        train_ids=set(groups[train_mask].tolist());val_ids=set(groups[val_mask].tolist())
        if train_ids&val_ids:raise ValueError('contact replay internal episode split leakage')
        train_sessions={association[g] for g in train_ids};val_sessions={association[g] for g in val_ids}
        if train_sessions&val_sessions:raise ValueError('contact replay internal session split leakage')
        declared=metadata.get('sim_validation_session_id')
        if val_sessions and val_sessions!={declared}:raise ValueError('contact replay validation session metadata mismatch')
        internal_audit.update(train_samples=int(train_mask.sum()),validation_samples=int(val_mask.sum()),
            train_sessions=sorted(train_sessions),validation_sessions=sorted(val_sessions),validation_used_for_optimization=False)
        arrays={k:v[train_mask] for k,v in arrays.items()}
    return arrays,{'sha256':sha256(path),'metadata':metadata,'groups':np.unique(arrays['group_ids']).tolist(),'internal_split':internal_audit,
        'semantics':'synthetic geometry/contact hypotheses only; no real contact positives or physical success evidence'}


def train(args):
    if Path(args.output).resolve()==Path(args.checkpoint).resolve(): raise ValueError('adaptation output must preserve pretrained checkpoint')
    if min(args.epochs,args.batch_size,args.image_size,args.threads)<1: raise ValueError('positive training configuration required')
    if not np.isfinite([args.lr,args.sim_weight,args.keypoint_weight,args.heatmap_weight,args.gate_weight]).all() or args.lr<=0 or min(args.sim_weight,args.keypoint_weight,args.heatmap_weight,args.gate_weight)<0: raise ValueError('invalid loss weights/learning rate')
    if not np.isfinite(getattr(args,'contact_sim_weight',.5)) or getattr(args,'contact_sim_weight',.5)<0:
        raise ValueError('contact replay weight must be nonnegative and finite')
    if not np.isfinite(args.hole_occlusion_probability) or not 0<=args.hole_occlusion_probability<=1:
        raise ValueError('hole occlusion probability must lie in [0,1]')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(args.seed);device=torch.device(args.device)
    predictor=PinInsertionPredictor(args.checkpoint,args.device)
    if predictor.metadata.get('model_variant')!='heatmap': raise ValueError('adaptation requires pretrained heatmap model')
    manifest=read_manifest(args.manifest,predictor.metadata)
    train_frames=[r for r in manifest['frames'] if r['split']=='train'];val_frames=[r for r in manifest['frames'] if r['split']=='validation']
    train_rows=load_rows(train_frames,args.image_size);val_rows=load_rows(val_frames,args.image_size)
    if getattr(args,'select_contact_only',False) and not any(r.get('contact_audit') and r['valid'].any() for r in val_rows):
        raise ValueError('contact-only selection requires independently audited validation landmark labels')
    expected=predictor.metadata.get('dataset_metadata',{})
    aspect=expected.get('image_width',4)/expected.get('image_height',3)
    if any(abs(r['wh'][0]/r['wh'][1]/aspect-1)>.02 for r in train_rows+val_rows): raise ValueError('real annotations must use full-frame source aspect ratio')
    model=predictor.model
    if args.image_only:
        # Zero all direct proprioceptive paths. Since training also supplies zeros,
        # keep them exactly zero (cos(0)=1 still needs explicit gradient masking).
        with torch.no_grad():
            model.condition.weight.zero_()
            model.fusion[0].weight[:,1024:].zero_()
        model.condition.weight.requires_grad_(False)
        model.fusion[0].weight.register_hook(lambda gradient: torch.cat((gradient[:,:1024],torch.zeros_like(gradient[:,1024:])),dim=1))
    for parameter in model.delta.parameters(): parameter.requires_grad_(False)
    sim=None
    if args.sim_dataset:
        if sha256(args.sim_dataset)!=predictor.metadata.get('dataset_sha256'): raise ValueError('sim replay SHA differs from pretrained source dataset')
        sim_meta=validate_dataset_metadata(json.loads(Path(args.sim_dataset).with_suffix('.json').read_text()))
        if sim_meta!=predictor.metadata.get('dataset_metadata'): raise ValueError('sim replay metadata mismatch')
        with np.load(args.sim_dataset,allow_pickle=False) as archive: sim={k:archive[k] for k in ('images','state','keypoints','active','keypoints_valid','group_ids')}
        keep=np.isin(sim['group_ids'],predictor.metadata['train_groups'])
        sim={k:v[keep] for k,v in sim.items()}
        if not len(sim['images']): raise ValueError('no eligible synthetic train replay')
    contact_sim=None;contact_audit=None
    if getattr(args,'contact_sim_dataset',None):
        contact_sim,contact_audit=load_contact_replay(args.contact_sim_dataset,manifest)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    metadata=copy.deepcopy(predictor.metadata)
    metadata.update(synthetic_only=False,direct_xyz_disabled=True,delta_supervision='none during real adaptation; frozen head; outputs invalid for execution',
        gate_unverified=True,real_gate_supervised=any(r['gate']>=0 for r in train_rows),localization_only=True,
        adaptation={'ancestor_optimization_and_selection_ids':sorted(source_groups(predictor.metadata)),'pretrained_checkpoint':str(Path(args.checkpoint).resolve()),'pretrained_sha256':sha256(args.checkpoint),
            'manifest_sha256':sha256(args.manifest),'manifest_metadata':{k:v for k,v in manifest.items() if k!='frames'},'split_audit':manifest['_split_audit'],'arguments':vars(args),'selection':'contact-only balanced landmark validation pixel error' if getattr(args,'select_contact_only',False) else 'all-validation mean visible landmark pixel error',
            'train_episode_ids':sorted({r['episode_id'] for r in train_frames}),'validation_episode_ids':sorted({r['episode_id'] for r in val_frames}),
            'reserved_test_episode_ids':sorted({r['episode_id'] for r in manifest['frames'] if r['split']=='test'}),
            'partition_identity_semantics':'within-recording carrier-pose blocks' if manifest['_split_audit']['split_unit'] in BLOCK_SPLIT_UNITS else 'recorded episodes',
            'future_insertion_reference_use':'annotation supervision only; never model inputs or memory seeds',
            'gate_masking_semantics':'generic partial-hole augmentation masks all gate labels; affine masks positives only when previously valid evidence is newly cropped; originally hidden pin does not itself discard protocol label',
            'model_input_features':['current RGB full frame','current measured left/right six joint radians'],
            'test_used_for_training_or_selection':False,'image_only':args.image_only,
            'partial_hole_occlusion':{'probability':args.hole_occlusion_probability,'training_only':True,
                'radius_source':'audited hole_radius_px if supplied; otherwise uncalibrated robust radial image contrast; skip ambiguous estimates',
                'semantics':'generic partial-hole occlusion robustness; not a reconstructed pin or physical contact simulation',
                'labels':'known original coordinates retained as synthetic amodal labels; no invented contact or positive gate labels',
                'nominal_coverage_fraction':[.12,.55],'gate_supervision':'masked on every applied sample','pin_tip_guard':'max(2px,0.18*radius) protected'}})
    metadata['adaptation']['contact_sim_replay']=contact_audit
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf');history=[]
    for epoch in range(args.epochs):
        model.train();losses=[];order=rng.permutation(len(train_rows));augmentation_audit={'attempted':0,'radii':[],'coverage':[]}
        for start in range(0,len(order),args.batch_size):
            batch=[train_rows[i] for i in order[start:start+args.batch_size]]
            x,q,k,v,g=make_batch(batch,args.image_size,device,rng,args.affine,args.image_only,args.hole_occlusion_probability,augmentation_audit)
            loss=localization_loss(model(x,q,return_logits=True),k,v,g,args)
            for replay,replay_weight in ((sim,args.sim_weight),(contact_sim,getattr(args,'contact_sim_weight',.5))):
                if replay is None or not replay_weight:continue
                chosen=rng.integers(0,len(replay['images']),size=len(batch))
                sx=torch.from_numpy(replay['images'][chosen].copy()).permute(0,3,1,2).float().to(device)/255
                sx=F.interpolate(sx,size=(args.image_size,args.image_size),mode='bilinear',align_corners=False)
                sq=torch.tensor(np.stack([joint_features(s) for s in replay['state'][chosen]]),device=device)
                if args.image_only: sq.zero_()
                sk=torch.tensor(replay['keypoints'][chosen],device=device);sv=torch.tensor(replay['keypoints_valid'][chosen],device=device)
                sg=torch.tensor(replay['active'][chosen],device=device)
                loss=loss+replay_weight*localization_loss(model(sx,sq,return_logits=True),sk,sv,sg,args)
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        metrics=validation(model,val_rows,args,device);metrics.update(epoch=epoch+1,train_loss=float(np.mean(losses)),
            hole_occlusion_attempted=augmentation_audit['attempted'],hole_occlusion_applied=len(augmentation_audit['radii']),
            hole_occlusion_mean_radius_px=float(np.mean(augmentation_audit['radii'])) if augmentation_audit['radii'] else None,
            hole_occlusion_mean_coverage=float(np.mean(augmentation_audit['coverage'])) if augmentation_audit['coverage'] else None,
            real_gate_exposure={key:augmentation_audit.get(key,0) for key in ('gate_input_positive','gate_input_negative','gate_input_unknown','gate_used_positive','gate_used_negative','gate_used_unknown')})
        history.append(metrics);print(json.dumps(metrics),flush=True)
        if metrics['selection_mean_error_px']<best:
            best=metrics['selection_mean_error_px'];metadata['adaptation']['best_validation']=metrics;metadata['image_size']=args.image_size
            temporary=output.with_suffix(output.suffix+'.tmp');torch.save({'model':model.state_dict(),'metadata':metadata},temporary);temporary.replace(output)
    report={'metadata':metadata,'history':history,'train_frames':len(train_rows),'validation_frames':len(val_rows),
        'limitations':[*(manifest['_split_audit'].get('limitations',[])),
            'Real XYZ has no supervision and is disabled; freezing its head does not preserve predictions when shared features change.',
            'Gate validity is limited by annotation quality; no physical success or safety claim.',
            'Held-out test metrics must be computed separately after checkpoint selection.']}
    output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--contact-sim-dataset',help='new contact geometry replay, exclusively sourced from real TRAIN episodes/sessions')
    p.add_argument('--contact-sim-weight',type=float,default=.5)
    p.add_argument('--sim-dataset');p.add_argument('--sim-weight',type=float,default=.25)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--image-size',type=int,default=160)
    p.add_argument('--lr',type=float,default=.0001);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--keypoint-weight',type=float,default=1000.);p.add_argument('--heatmap-weight',type=float,default=10.);p.add_argument('--gate-weight',type=float,default=1.)
    p.add_argument('--select-contact-only',action='store_true',help='select only on independently contact_audit=true validation rows; report all strata')
    p.add_argument('--hole-occlusion-probability',type=float,default=0.,help='training-only generic partial-hole occlusion; keeps known amodal labels and masks gate loss')
    p.add_argument('--affine',action='store_true');p.add_argument('--image-only',action='store_true')
    train(p.parse_args())

if __name__=='__main__':main()
