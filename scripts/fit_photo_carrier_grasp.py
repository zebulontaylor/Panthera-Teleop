#!/usr/bin/env python3
"""TRAIN-only CAD-bore/right-tool rigid hypothesis under fixed own-wrist camera."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from real2sim_object_pose import fit_carrier_pose,carrier_bore_points,CARRIER_BORE_HEIGHT_M


def rigid(r,t):
    a=np.eye(4);a[:3,:3]=r;a[:3,3]=t;return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--camera',type=Path,required=True)
    p.add_argument('--reset-provenance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=json.loads(args.manifest.read_text());camera=json.loads(args.camera.read_text())
    prior=np.asarray(json.loads(args.reset_provenance.read_text())['carrier_tool_grasp_from_existing_physical_reset'])
    k=np.asarray(camera['K']);dist=np.asarray(camera['distortion']);tool_camera=np.asarray(camera['T_right_tool_camera_opencv'])
    selected=[];audits=[]
    for row in manifest['frames']:
        pixels=row.get('carrier_bores_px')
        if row['split']!='train' or pixels is None or np.asarray(pixels).shape!=(4,2):continue
        hypotheses=fit_carrier_pose(pixels,k,dist,calibrated=False,max_reprojection_error_px=8)['hypotheses']
        candidates=[]
        right_r=np.asarray(row['T_right_base_tool'])[:3,:3]
        for hypothesis in hypotheses:
            tc=rigid(np.asarray(hypothesis['rotation_matrix']),np.asarray(hypothesis['translation_m']))@rigid(np.eye(3),[0,0,-CARRIER_BORE_HEIGHT_M])
            tool_carrier=tool_camera@tc;t=tool_carrier[:3,3];normal_up=float((right_r@tool_carrier[:3,:3])[:,2]@[0,0,1])
            # Bounded jaw contact envelope is physical prior, not fitted depth.
            envelope_penalty=max(-.045-t[0],0,t[0])/.01+max(abs(t[1])-.012,0)/.01
            tilt_penalty=max(.65-normal_up,0)*20
            rotation_distance=float(Rotation.from_matrix(prior[:3,:3].T@tool_carrier[:3,:3]).magnitude())
            cost=float(hypothesis['reprojection_rms_px'])+8*envelope_penalty+tilt_penalty+rotation_distance
            candidates.append({'T_right_tool_carrier':tool_carrier.tolist(),'cost':cost,'normal_up':normal_up,
                               'envelope_penalty':envelope_penalty,'rotation_distance_to_reset_rad':rotation_distance,
                               'reprojection_rms_px':hypothesis['reprojection_rms_px']})
        if not candidates:continue
        candidates.sort(key=lambda c:c['cost']);winner=candidates[0]
        selected.append(np.asarray(winner['T_right_tool_carrier']));audits.append({'id':row['id'],'episode_id':row['episode_id'],'selected':winner,'candidate_count':len(candidates)})
    if not selected:raise ValueError('No train bore pose hypotheses')
    positions=np.array([a[:3,3] for a in selected]);center=np.median(positions,axis=0)
    distance=np.linalg.norm(positions-center,axis=1);mad=float(np.median(np.abs(distance-np.median(distance))))
    keep=distance<=np.median(distance)+max(3*mad,.002)
    rotation=Rotation.from_matrix(np.array([a[:3,:3] for a in selected])[keep]).mean().as_matrix()
    transform=rigid(rotation,np.median(positions[keep],axis=0))
    from scipy.optimize import linear_sum_assignment
    def evaluate(split):
        reports=[]
        for row in manifest['frames']:
            if row['split']!=split or row.get('carrier_bores_px') is None:continue
            tc=np.linalg.inv(tool_camera)@transform
            points=carrier_bore_points()+[0,0,CARRIER_BORE_HEIGHT_M]
            rv=cv2.Rodrigues(tc[:3,:3])[0];uv=cv2.projectPoints(points,rv,tc[:3,3],k,dist)[0].reshape(4,2)
            target=np.asarray(row['carrier_bores_px']);matrix=np.linalg.norm(uv[:,None]-target[None],axis=-1);a,b=linear_sum_assignment(matrix)
            reports.append({'id':row['id'],'episode_id':row['episode_id'],'errors_px':matrix[a,b].tolist(),'predicted_bores_px':uv.tolist()})
        values=[e for r in reports for e in r['errors_px']]
        return {'frames':len(reports),'mean_px':float(np.mean(values)) if values else None,'median_px':float(np.median(values)) if values else None,'rows':reports}
    result={'kind':'train_only_constant_right_tool_carrier_hypothesis','T_right_tool_carrier':transform.tolist(),
      'camera_sha256':hashlib.sha256(args.camera.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
      'selection':'TRAIN PnP candidates: upward normal, pad X/Y envelope, nearest existing reset rotation; no val/test selection',
      'training_candidate_frames':len(selected),'training_retained_frames':int(keep.sum()),'training_translation_std_m':positions[keep].std(0).tolist(),
      'train':evaluate('train'),'validation':evaluate('validation'),'training_hypotheses':audits,
      'annotation_coverage':{split:{'frames':len(rows),'episodes':len({r['episode_id'] for r in rows}),'exact_distinct_coordinate_sets':len({json.dumps(r['carrier_bores_px']) for r in rows})} for split in ('train','validation') if (rows := [r for r in manifest['frames'] if r['split']==split and r.get('carrier_bores_px') is not None])},
      'real_calibrated':False,'metric_accuracy_validated':False,'physics_validated':False,
      'limitations':['Repeated annotations provide only two exact coordinate sets in each split; small fitted dispersion is not independent submillimeter accuracy or broad viewpoint validation.',
       'Unordered fourfold planar bores retain90deg yaw ambiguity and possible normal/depth ambiguity; physical priors select one branch.',
       'Constant carrier-tool rigid grasp across train and heldout is a hypothesis; no per-test-frame object fitting.',
       'Camera itself is conditional own-finger image registration with opening-scale prior, not metric calibration.',
       'CAD mesh complete assembly will be attached render-only, not claimed contact-valid physical reset.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('training_hypotheses','train','validation')},indent=2));print('Train',result['train']['mean_px'],'Validation',result['validation']['mean_px'])

if __name__=='__main__':main()
