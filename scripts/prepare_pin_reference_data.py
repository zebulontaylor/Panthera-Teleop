#!/usr/bin/env python3
"""Prepare causal empty-carrier RGB references without changing frozen labels.

Reference selection is an explicit visual audit, not automatic contact detection.
Pin coordinate supervision suspected of stationary-bore confusion is masked;
original coordinates remain in audit metadata and the source stays immutable.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(manifest_path, output):
    manifest_path=Path(manifest_path).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    source=json.loads(manifest_path.read_text()); result=copy.deepcopy(source)
    result['source_manifest_path']=str(manifest_path)
    result['source_manifest_sha256']=digest(manifest_path)
    episode=Path(source['frames'][0]['source_episode_path'])
    frames=[json.loads(x) for x in (episode/'frames.jsonl').read_text().splitlines()]
    frames={f['frame_index']:f for f in frames if f['camera']=='wrist_port2'}
    states=[json.loads(x) for x in (episode/'states.jsonl').read_text().splitlines()]
    state_hash=digest(episode/'states.jsonl')
    # Frames in transition guards are online observations, never labeled targets.
    # The held pin is withdrawn outside the carrier in these audited references.
    reference_frames={'B1':[0,1,2],'B2':[280,281,282],
                      'B3':[498,499,500],'B4':[950,951,952],
                      'B5':[935,936,937]}
    refs={}; records=[]
    for block,indices in reference_frames.items():
        paths=[episode/frames[i]['path'] for i in indices]
        images=[cv2.imread(str(p)) for p in paths]
        if any(im is None or im.shape!=(480,640,3) for im in images):
            raise ValueError('reference must decode full 640x480 RGB')
        reference=np.median(np.stack(images),axis=0).astype(np.uint8)
        path=output/f'{block}-causal-reference.png'; cv2.imwrite(str(path),reference)
        rec={'block_id':block,'image_path':str(path),'image_sha256':digest(path),
             'source_recording_id':source['source_recording_id'],
             'source_images':[{'image_path':str(p.resolve()),'image_sha256':digest(p),
                               'frame':frames[i]} for i,p in zip(indices,paths)],
             'latest_source_monotonic_ns':max(frames[i]['monotonic_ns'] for i in indices),
             'construction':'per-channel pixel median of three prior frames',
             'visual_audit':'Carrier surface clear of held pin; withdrawn pin may remain elsewhere in full image. No claim of pin absence outside carrier.',
             'scope':'causal visual context only; no reference landmark coordinates used'}
        refs[block]=(reference,rec);records.append(rec)
        for item in rec['source_images']:
            index=min(range(len(states)),key=lambda k:abs(states[k]['monotonic_ns']-item['frame']['monotonic_ns']))
            item['source_states_path']=str(episode/'states.jsonl')
            item['source_states_sha256']=state_hash
            item['nearest_state_jsonl_line_one_based']=index+1
            item['nearest_measured_state']=states[index]
    masked=[];excluded=[];registration=[]
    # These four points visibly terminate an exposed withdrawn pin. Other TRAIN
    # point labels require re-review after the user identified stationary bore
    # reflections being mistaken for distal endpoints; no replacement is invented.
    retained_train_tips={73,221,608,678}
    for row in result['frames']:
        raw,ref=refs[row['block_id']]
        row['reference_provenance']=copy.deepcopy(ref)
        causal=ref['latest_source_monotonic_ns']<row['monotonic_ns']
        row['reference_is_causal']=causal
        row['reference_provenance']['causal_reference']=causal
        if not causal:
            row['excluded_from_paired_training']=True
            row['paired_exclusion_reason']='Reference median includes present/future frame; excluded rather than silently using future context.'
            excluded.append(row['id'])
        if row['split']=='train' and row['valid'][0] and row['frame_index'] not in retained_train_tips:
            row['original_pin_annotation_for_audit']={'pin_tip_px':row.get('pin_tip_px'),'keypoints_px':row['keypoints_px'][:2],
                'valid':True,'reason':'Conservative exclusion pending temporal physical-subfeature review; user confirmed stationary-bore reflection failure. This is suspicion, not relabeling as wrong.'}
            row['valid'][0]=False; row['keypoints_px'][:2]=[None,None];row['pin_tip_px']=None
            masked.append(row['id'])
        # ECC is causal per-frame preprocessing using current image only. No
        # annotations, model predictions, state calibration, or future pixels.
        current=cv2.imread(row['image_path']); gray=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        refgray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        mask=np.zeros((480,640),np.uint8)
        cv2.ellipse(mask,(325,355),(82,80),0,0,360,255,-1)
        mask[:320,:320]=0
        warp=np.eye(2,3,dtype=np.float32);score=None;error=None
        try:
            score,warp=cv2.findTransformECC(gray,refgray,warp,cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),mask,5)
            angle=float(np.arctan2(warp[1,0],warp[0,0]))
            if abs(angle)>.35 or np.linalg.norm(warp[:,2])>100 or score<.5:
                raise ValueError('ECC registration outside audited bounds')
            aligned=cv2.warpAffine(raw,warp,(640,480),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT_101)
        except (cv2.error,ValueError) as exc:
            aligned=raw;warp=np.eye(2,3,dtype=np.float32);error=str(exc)
        aligned_path=output/'references'/f"{row['id']}.png"
        aligned_path.parent.mkdir(exist_ok=True);cv2.imwrite(str(aligned_path),aligned)
        row['reference_image_path']=str(aligned_path);row['reference_image_sha256']=digest(aligned_path)
        row['reference_registration']={'method':'causal current-image ECC Euclidean on lower/right carrier mask; no labels',
            'template_to_source_warp':warp.tolist(),'ecc_score':float(score) if score is not None else None,
            'fallback_unregistered':error is not None,'error':error}
        registration.append({'id':row['id'],**row['reference_registration']})
    result['paired_reference_data']={'source_manifest_path':str(manifest_path),'source_manifest_sha256':digest(manifest_path),
        'reference_frames':reference_frames,'causal':True,'unlabeled_guard_images_used_as_online_context':True,
        'reference_split_sharing':'Distinct source reference image sets for every block; single original recording remains non-session-independent.',
        'pin_label_policy':'Conservatively mask suspected TRAIN near-bore subfeature points; retain original validation/test annotations unchanged; no new point ground truth.',
        'limitations':['Reference carrier clear; withdrawn pin can remain elsewhere in full frame.',
            'Median of three frames does not establish metric registration or physical contact.',
            'Validation contains only one original valid pin label; it is not human certified.',
            'Test pixels used only to prepare their own causal inference input; never training/selection targets.']}
    counts={s:{'rows':sum(r['split']==s for r in result['frames']),
               'eligible_rows':sum(r['split']==s and not r.get('excluded_from_paired_training') for r in result['frames']),
               'eligible_pin_labels':sum(r['split']==s and r['valid'][0] and not r.get('excluded_from_paired_training') for r in result['frames'])} for s in ('train','validation','test')}
    report={'source_manifest_sha256':digest(manifest_path),'reference_audits':records,'masked_train_pin_rows':masked,
        'noncausal_excluded_rows':excluded,'counts':counts,'registration':registration,'limitations':result['paired_reference_data']['limitations']}
    (output/'paired-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'data-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(counts,indent=2))
    return result,report


def prepare_weak(output, episode):
    """Three reviewed adjacent frames; weak box is user directional evidence."""
    output=Path(output).resolve();episode=Path(episode).resolve()
    frames=[json.loads(x) for x in (episode/'frames.jsonl').read_text().splitlines()]
    frames={f['frame_index']:f for f in frames if f['camera']=='wrist_port2'}
    states=[json.loads(x) for x in (episode/'states.jsonl').read_text().splitlines()]
    reference_indices=[0,1,2]
    paths=[episode/frames[i]['path'] for i in reference_indices]
    images=[cv2.imread(str(p)) for p in paths]
    ref=np.median(np.stack(images),axis=0).astype(np.uint8)
    refpath=output/'weak-causal-reference.png';cv2.imwrite(str(refpath),ref)
    rows=[]
    for i in [54,55,56]:
        f=frames[i];path=episode/f['path']
        k=min(range(len(states)),key=lambda k:abs(states[k]['monotonic_ns']-f['monotonic_ns']))
        state=states[k]
        state14=[]
        for arm in ('left','right'):
            state14+=state['arms'][arm]['position_rad']+[state['arms'][arm]['gripper']['position']]
        rows.append({'id':episode.name+f'__weak_{i:04d}','split':'train','episode_id':episode.name,
            'source_recording_id':episode.name,'session_id':episode.name,'source_episode_path':str(episode),
            'image_path':str(path),'image_sha256':digest(path),'image_width':640,'image_height':480,
            'camera':'right_wrist','camera_source':'wrist_port2','frame_index':i,'monotonic_ns':f['monotonic_ns'],
            'state14':state14,'source_state_monotonic_ns':state['monotonic_ns'],
            'source_states_path':str(episode/'states.jsonl'),'source_state_jsonl_line_one_based':k+1,
            'image_state_skew_ms':(f['monotonic_ns']-state['monotonic_ns'])/1e6,
            'valid':[False,False],'keypoints_px':[None,None,None,None],'uncertainty_px':[8.,8.],
            'gate_label':None,'weak_pin_box_px':[267,274,282,289],
            'weak_pin_evidence':'User states physical bottom is approximately10–15px up-left of cyan marker; independently inspected consecutive raw frames54–56 show same moving shaft/end above-left of static bore. Box is broad directional supervision, not an exact center label.',
            'weak_pin_box_not_point_ground_truth':True,'development_training_source':True,
            'reference_image_path':str(refpath),'reference_image_sha256':digest(refpath),'reference_is_causal':True,
            'reference_provenance':{'causal_reference':True,'source_recording_id':episode.name,'source_frame_indices':reference_indices,
                'source_images':[{'image_path':str(p),'image_sha256':digest(p),'frame':frames[j]} for j,p in zip(reference_indices,paths)],
                'latest_source_monotonic_ns':max(frames[j]['monotonic_ns'] for j in reference_indices),
                'construction':'median3 raw early frames with visibly empty carrier; no future data',
                'scope':'development TRAIN source; cannot be reported as independent test'}})
    manifest={'schema_version':1,'split_unit':'episode_session','frames':rows,
        'development_train_only':True,'user_annotation_kind':'approximate directional physical-subfeature guidance, not exact coordinates',
        'source_metadata_sha256':digest(episode/'metadata.json'),'source_frames_sha256':digest(episode/'frames.jsonl'),
        'source_states_sha256':digest(episode/'states.jsonl'),
        'limitations':['Three highly correlated frames, not three independent physical examples.',
            'Acquisition is now development training and must never be called held-out test.',
            'No exact point, hole, contact-force or successful-insertion label invented.']}
    (output/'weak-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default='work/pin-insertion-newdemo/manifest.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--weak-episode');a=parser.parse_args();prepare(a.manifest,a.output)
    if a.weak_episode:prepare_weak(a.output,a.weak_episode)
