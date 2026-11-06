#!/usr/bin/env python3
"""Conservative RGB-only bore pseudo labels from reviewed real LEFT seed frames.

Never makes hidden labels: every propagated center needs bidirectional flow AND
fresh local dark-ellipse/rim evidence. Heldout manual seeds are copied unchanged,
never tracked or used to tune parameters. Slot order is seed-local, not CAD identity.
"""
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np


def refine_bore(gray, center, radius, max_shift=5):
    """Verify full large-bore rim, rejecting occlusion, tiny holes and scale drift."""
    x,y=map(float,center); r=float(radius)
    if r<6 or r>28:return None
    margin=int(np.ceil(r*2+max_shift)); ix,iy=int(round(x)),int(round(y))
    if ix-margin<0 or iy-margin<0 or ix+margin>=gray.shape[1] or iy+margin>=gray.shape[0]:return None
    crop=gray[iy-margin:iy+margin+1,ix-margin:ix+margin+1]
    smooth=cv2.GaussianBlur(crop,(3,3),0)
    edges=cv2.Canny(smooth,35,95)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    best=None
    for contour in contours:
        if len(contour)<16:continue
        (cx,cy),(a,b),angle=cv2.fitEllipse(contour)
        lo,hi=min(a,b)/2,max(a,b)/2
        q=np.array([cx+ix-margin,cy+iy-margin]); shift=np.linalg.norm(q-[x,y])
        if shift>max_shift or not .66*r<np.sqrt(lo*hi)<1.4*r or lo/hi<.55:continue
        theta=np.linspace(0,2*np.pi,72,endpoint=False); ar=np.deg2rad(angle)
        # Actual fitted ellipse, with points sampled on both sides of rim.
        def samples(scale):
            xx=cx+scale*(a/2*np.cos(theta)*np.cos(ar)-b/2*np.sin(theta)*np.sin(ar))
            yy=cy+scale*(a/2*np.cos(theta)*np.sin(ar)+b/2*np.sin(theta)*np.cos(ar))
            return cv2.remap(smooth,xx.astype('float32')[None],yy.astype('float32')[None],cv2.INTER_LINEAR)[0].astype(float)
        inner=samples(.68); outer=samples(1.25); contrast=outer-inner
        coverage=float(np.mean(contrast>12)); mean_contrast=float(np.mean(contrast))
        if coverage<.68 or mean_contrast<18:continue
        score=coverage+min(mean_contrast/100,.7)-shift/15
        item=dict(center=q.tolist(),radius=float(np.sqrt(lo*hi)),ellipse_axes_px=[float(a),float(b)],ellipse_angle_deg=float(angle),confidence=float(min(1,coverage*.65+mean_contrast/160)),rim_coverage=coverage,contrast=mean_contrast)
        if best is None or score>best[0]:best=(score,item)
    return None if best is None else best[1]


def track_step(previous,current,center,radius):
    """Track textured rim points, not the textureless dark bore center."""
    p=np.asarray(center,np.float32); theta=np.linspace(0,2*np.pi,24,endpoint=False)
    features=(p+np.column_stack([np.cos(theta),np.sin(theta)])*radius*1.05).astype(np.float32).reshape(-1,1,2)
    kw=dict(winSize=(21,21),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    forward,status,error=cv2.calcOpticalFlowPyrLK(previous,current,features,None,**kw)
    if forward is None:return None
    backward,status2,error2=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,**kw)
    if backward is None:return None
    fb=np.linalg.norm(backward-features,axis=2).ravel(); good=(status.ravel()>0)&(status2.ravel()>0)&(fb<.8)&(error.ravel()<22)
    if good.sum()<16:return None
    motion=np.median((forward-features).reshape(-1,2)[good],axis=0)
    spread=np.linalg.norm((forward-features).reshape(-1,2)[good]-motion,axis=1)
    if np.quantile(spread,.8)>1.7 or np.linalg.norm(motion)>18:return None
    result=refine_bore(current,p+motion,radius,max_shift=3.5)
    if result is not None:result['flow_fb_p90']=float(np.quantile(fb[good],.9))
    return result


def generate(seeds_path,split_path,out,max_window=100,eval_path=None):
    seeds=json.loads(Path(seeds_path).read_text()); split=json.loads(Path(split_path).read_text()); out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frames={}; rejected=0
    for seed in seeds['frames']:
        if seed['split']!='train':continue
        block=next(b for b in split['blocks'] if b['block_id']==seed['group_id'].split('::')[-1])
        if block['split']!='train':raise ValueError('Frozen heldout block cannot generate train labels')
        source=Path(seed['image_path']); index=seed['source_frame_index']; gray=cv2.imread(str(source),0)
        points=seed['unordered_visible_bores_px']; radii=seed.get('bore_radii_px',[11]*len(points))
        for direction in (-1,1):
            previous=gray; active=[refine_bore(gray,p,r,max_shift=5) for p,r in zip(points,radii)]
            reference_radii=[a['radius'] if a else None for a in active]
            for step in range(max_window+1):
                n=index+direction*step
                if not block['first_frame']<=n<=block['last_frame']:break
                path=source.with_name(f'{n:07d}.jpg'); current=cv2.imread(str(path),0)
                if current is None:break
                if step:
                    active=[track_step(previous,current,a['center'],a['radius']) if a else None for a in active]
                    active=[a if a and .75*rr<a['radius']<1.3*rr else None for a,rr in zip(active,reference_radii)]
                previous=current
                if not any(active):rejected+=1;break
                # Conservative stop: missing points never silently reappear without manual seed.
                key=f'{seed["source_recording_id"]}::{n}'
                record=dict(id=key,source_recording_id=seed['source_recording_id'],source_frame_index=n,image_path=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera='left_wrist',camera_source='wrist_port3',group_id=seed['group_id'],split='train',unordered_visible_bores_px=[a['center'] for a in active if a],bore_evidence=[a for a in active if a],seed_id=seed['id'],seed_frame_index=index,tracking_steps=step,label_type='reviewed_seed_local_opencv_rim' if step==0 else 'opencv_pseudo_label',canonical_physical_slot_identity_verified=False,hidden_ground_truth=False,training_eligible=True,state_used_for_labels=False,visibility_unlabeled_is_unknown=True,seed_annotation_uncertainty_px=seed.get('annotation_uncertainty_px',5))
                old=frames.get(key)
                if old is None or len(record['bore_evidence'])>len(old['bore_evidence']) or (len(record['bore_evidence'])==len(old['bore_evidence']) and step<old['tracking_steps']):frames[key]=record
    records=sorted(frames.values(),key=lambda x:x['source_frame_index'])
    manifest=dict(schema_version=1,method='real RGB only; reviewed train seeds + bidirectional LK and per-frame ellipse/rim verification',seed_manifest=str(Path(seeds_path).resolve()),seed_sha256=hashlib.sha256(Path(seeds_path).read_bytes()).hexdigest(),frozen_split_sha256=hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),max_tracking_steps=max_window,heldout_tracking=False,canonical_physical_slot_identity_verified=False,limitations=['Pseudo labels are not human ground truth.','Unordered bore identities cannot train canonical CAD slot channels without explicit identity annotation.','No coordinates or negative visibility targets assigned to unobserved bores.'],counts=dict(train_frames=len(records),visible_bore_labels=sum(len(f['bore_evidence']) for f in records),stopped_tracks=rejected),frames=records)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    evidence=[a for f in records for a in f['bore_evidence']]
    audit=dict(manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),per_group={g:sum(f['group_id']==g for f in records) for g in sorted(set(f['group_id'] for f in records))},checks=dict(rgb_only=True,no_fk_projection=True,heldout_frames_in_train=0,hidden_labels=0,unlabeled_visibility='unknown'),confidence_min=float(min([a['confidence'] for a in evidence],default=0)),rim_coverage_min=float(min([a['rim_coverage'] for a in evidence],default=0)),contrast_min=float(min([a['contrast'] for a in evidence],default=0)),max_seed_distance=max([f['tracking_steps'] for f in records],default=0),visual_audit='Review tracked-contact-sheet.jpg; seed and temporal labels remain AI/OpenCV pseudo labels, not human certified.')
    (out/'quality-audit.json').write_text(json.dumps(audit,indent=2))
    if records:
        chosen=[records[i] for i in np.linspace(0,len(records)-1,min(36,len(records))).astype(int)];tiles=[]
        for f in chosen:
            im=cv2.imread(f['image_path'])
            for a in f['bore_evidence']:
                p=tuple(np.round(a['center']).astype(int));cv2.circle(im,p,round(a['radius']),(0,255,0),2);cv2.circle(im,p,2,(0,0,255),-1)
            cv2.putText(im,f'{f["source_frame_index"]} seed {f["seed_frame_index"]} d{f["tracking_steps"]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2);tiles.append(cv2.resize(im,(320,240)))
        canvas=np.zeros((int(np.ceil(len(tiles)/6))*240,6*320,3),np.uint8)
        for i,t in enumerate(tiles):canvas[i//6*240:(i//6+1)*240,i%6*320:(i%6+1)*320]=t
        cv2.imwrite(str(out/'tracked-contact-sheet.jpg'),canvas)
    if eval_path:
        evaluation=json.loads(Path(eval_path).read_text())
        assert all(f['split'] in ('validation','test') and not f.get('training_eligible',False) for f in evaluation['frames'])
        (out/'evaluation-manifest.json').write_text(json.dumps(evaluation,indent=2))
        (out/'dataset-sources.json').write_text(json.dumps(dict(train_manifest=str((out/'manifest.json').resolve()),evaluation_manifest=str((out/'evaluation-manifest.json').resolve()),evaluation_source_sha256=hashlib.sha256(Path(eval_path).read_bytes()).hexdigest(),evaluation_labels_unchanged=True,heldout_tracking=False),indent=2))
    print(json.dumps(manifest['counts']))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',required=True);p.add_argument('--split',default='work/pin-insertion-newdemo/split.json');p.add_argument('--out',default='work/left-carrier-opencv');p.add_argument('--max-window',type=int,default=100);p.add_argument('--evaluation');a=p.parse_args();generate(a.seeds,a.split,a.out,a.max_window,a.evaluation)
