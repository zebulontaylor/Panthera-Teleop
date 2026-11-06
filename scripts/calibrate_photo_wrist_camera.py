#!/usr/bin/env python3
"""Fit own-right-finger image silhouettes in the existing photo environment.

No pin/carrier/base residuals, no changes to robot meshes. Pixel registration
is conditional on CAD and a gripper endpoint scaling, not metric metrology.
"""
import argparse,json,os,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import cv2,mujoco,numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from pin_insertion_environment import PhotoSetupBridge


def fill_envelope(mask):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=np.zeros_like(mask,dtype=np.uint8);cv2.drawContours(result,contours,-1,255,cv2.FILLED)
    return result>0


def boundary(mask):return cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)


def silhouette_metrics(prediction,target):
    intersection=np.count_nonzero(prediction&target);union=np.count_nonzero(prediction|target)
    iou=intersection/max(union,1);pe,te=boundary(prediction),boundary(target)
    if not pe.any() or not te.any():return dict(iou=iou,chamfer_px=200.,score=220.)
    pd=cv2.distanceTransform((~pe).astype(np.uint8),cv2.DIST_L2,3);td=cv2.distanceTransform((~te).astype(np.uint8),cv2.DIST_L2,3)
    chamfer=float((td[pe].mean()+pd[te].mean())*.5)
    return dict(iou=iou,chamfer_px=chamfer,score=chamfer+20*(1-iou))


def apply_registration(bridge,fit,*,adjust_gripper=True):
    """Apply after each native pose; only camera and approximate gripper opening."""
    model,data=bridge.model,bridge.data;cid=model.camera('right_wrist').id
    model.cam_pos[cid]=fit['model_camera_local_position_m']
    mujoco.mju_mat2Quat(model.cam_quat[cid],np.asarray(fit['model_camera_local_rotation_mujoco']).ravel())
    model.cam_fovy[cid]=fit['fovy_deg']
    if adjust_gripper:
        opening=data.qpos[bridge.finger_qadr[1,0]]*fit['right_gripper_opening_scale']+fit.get('right_gripper_opening_offset_m',0.)
        data.qpos[bridge.finger_qadr[1]]=[opening,-opening]
    mujoco.mj_forward(model,data)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-gripper-scale',type=float);p.add_argument('--front-x-min-m',type=float,default=-.060);p.add_argument('--initial-profile',type=Path);p.add_argument('--starts',type=int,default=3);p.add_argument('--width',type=int,default=320);p.add_argument('--max-evaluations',type=int,default=2200)
    a=p.parse_args();raw=json.loads(a.labels.read_text());rows=raw['frames'];width=a.width;height=round(width*.75)
    bridge=PhotoSetupBridge();model,data=bridge.model,bridge.data;model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    cid=model.camera('right_wrist').id;pos0=model.cam_pos[cid].copy();rot0=np.empty(9);mujoco.mju_quat2Mat(rot0,model.cam_quat[cid]);rot0=rot0.reshape(3,3)
    fov0=float(model.cam_fovy[cid]);groups=model.geom_group.copy();model.geom_group[:]=0
    white_ids=[]
    for gid in range(model.ngeom):
        body=model.body(int(model.geom_bodyid[gid])).name
        if body.startswith('right_') and groups[gid]==2:model.geom_group[gid]=2
        if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH and 'white_digit' in model.mesh(int(model.geom_dataid[gid])).name and body.startswith('right_'):white_ids.append(gid)
    renderer=mujoco.Renderer(model,height=height,width=width);renderer.enable_segmentation_rendering();opt=mujoco.MjvOption();opt.geomgroup[:]=[0,0,1,0,0,0];opt.sitegroup[:]=0
    prepared=[]
    for row in rows:
        mask=np.zeros((height,width),np.uint8)
        for polygon in row['right_finger_polygons']:cv2.fillPoly(mask,[(np.asarray(polygon)*[width/row['width'],height/row['height']]).astype(np.int32)],255)
        prepared.append(dict(row=row,target=fill_envelope(mask)))
    train=[r for r in prepared if r['row']['split']=='train'];validation=[r for r in prepared if r['row']['split']=='validation']
    if not train or not validation:raise ValueError('Train and independent validation masks required')
    scales=np.array([.02,.02,.02,.25,.25,.25,12.]+([] if a.fixed_gripper_scale is not None else [.2]))
    bounds=[(-1.25,3),(-2,2),(-3.75,.5)]+[(-3,3)]*3+[(-1.2,2.8)]+([] if a.fixed_gripper_scale is not None else [(-2,3)])
    def decode(z):
        x=z*scales
        return dict(model_camera_local_position_m=(pos0+x[:3]).tolist(),model_camera_local_rotation_mujoco=(Rotation.from_rotvec(x[3:6]).as_matrix()@rot0).tolist(),fovy_deg=float(fov0+x[6]),right_gripper_opening_scale=float(a.fixed_gripper_scale if a.fixed_gripper_scale is not None else 1+x[7]),right_gripper_opening_offset_m=0.)
    def render(fit,row):
        bridge.set_kinematic_pose(row['state14']);apply_registration(bridge,fit)
        renderer.update_scene(data,camera='right_wrist',scene_option=opt)
        seg=renderer.render();return fill_envelope(np.isin(seg[:,:,0],white_ids))
    evaluations=0
    def objective(z):
        nonlocal evaluations;evaluations+=1;fit=decode(z)
        score=np.mean([silhouette_metrics(render(fit,r['row']),r['target'])['score'] for r in train])
        return float(score+.04*np.dot(z,z))
    rng=np.random.default_rng(101);solutions=[]
    try:
        for i in range(a.starts):
            center=np.zeros(len(scales))
            if a.initial_profile:
                prior=json.loads(a.initial_profile.read_text());center[:3]=(np.asarray(prior['model_camera_local_position_m'])-pos0)/scales[:3];center[3:6]=Rotation.from_matrix(np.asarray(prior['model_camera_local_rotation_mujoco'])@rot0.T).as_rotvec()/scales[3:6];center[6]=(prior['fovy_deg']-fov0)/scales[6]
                if a.fixed_gripper_scale is None:center[7]=(prior['right_gripper_opening_scale']-1)/scales[7]
            initial=center if i==0 else center+rng.normal(0,.35,len(scales))
            initial=np.clip(initial,np.asarray(bounds)[:,0]+1e-6,np.asarray(bounds)[:,1]-1e-6)
            result=minimize(objective,initial,method='Powell',bounds=bounds,options=dict(maxfev=a.max_evaluations,xtol=.01,ftol=.0002))
            fit=decode(result.x)
            metrics={split:[dict(id=r['row']['id'],**silhouette_metrics(render(fit,r['row']),r['target'])) for r in subset] for split,subset in [('train',train),('validation',validation)]}
            solutions.append(dict(parameters=result.x.tolist(),profile=fit,objective=float(result.fun),success=bool(result.success),nfev=int(result.nfev),metrics=metrics))
            print('START',i,'train',np.mean([m['score'] for m in metrics['train']]),'val',np.mean([m['score'] for m in metrics['validation']]),flush=True)
        selected=min(solutions,key=lambda s:np.mean([m['score'] for m in s['metrics']['validation']]))
        fit=dict(selected['profile']);fit.update(schema_version=1,kind='own_right_finger_silhouette_registration_hypothesis',real_calibrated=False,metric_accuracy_validated=False,
            fixed_gripper_scale=a.fixed_gripper_scale,parameter_bounds_dimensionless=bounds,parameters_at_bounds=[i for i,(v,(lo,hi)) in enumerate(zip(selected['parameters'],bounds)) if min(abs(v-lo),abs(v-hi))<.03],per_frame_finger_opening_m={r['id']:r['state14'][13]*.02*fit['right_gripper_opening_scale'] for r in rows},mesh_mask_semantics='Blackmount/pads/rightrobot occlusion rendered; mask exact disconnected white_digit component from original source STL facets; rearblackmount separatevisualmesh; collision/inertia/joints unchanged',selected_by='mean independent validation mask chamfer+20*(1-IoU); no pin/carrier/test pixels',starts=solutions,
            baseline_metrics={split:[dict(id=r['row']['id'],**silhouette_metrics(render(decode(np.zeros(len(scales))),r['row']),r['target'])) for r in subset] for split,subset in [('train',train),('validation',validation)]},
            selected_metrics=selected['metrics'],labels_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),environment=bridge.provenance(),
            assumptions=['Only two near-identical gripper apertures observed; camera/width/depth parameters may be confounded despite silhouette overlap. Fixedgripperscale whenprovided is a physicalcarriergrasp prior, not estimated from these masks.',
             'Manual white-finger outer envelopes fill rib slots in BOTH masks. Black pads, left gripper, carrier excluded.',
             'Native arm angles unchanged; right gripper radians-to-opening endpoint scale is fitted nuisance, not independently measured jaw calibration.',
             'Camera mount/FOV remains conditional CAD image registration, not full intrinsics/distortion or metric calibration.'])
        # Export CV camera pose in right tool, K for renderers.
        bridge.set_kinematic_pose(rows[0]['state14']);apply_registration(bridge,fit)
        tool=np.eye(4);tool[:3,:3]=data.site_xmat[bridge.sites[1]].reshape(3,3);tool[:3,3]=data.site_xpos[bridge.sites[1]]
        camera=np.eye(4);camera[:3,:3]=data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);camera[:3,3]=data.cam_xpos[cid]
        fit['T_right_tool_camera_opencv']=(np.linalg.inv(tool)@camera).tolist();focal=240/np.tan(np.deg2rad(fit['fovy_deg']/2));fit['K']=[[focal,0,320],[0,focal,240],[0,0,1]];fit['distortion']=[0]*5
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(fit,indent=2)+'\n')
        panels=[]
        for item in prepared:
            original=cv2.resize(cv2.imread(item['row']['image_path']),(width,height));pred=render(fit,item['row']);target=item['target'];overlay=original.copy()
            overlay[target]=(overlay[target]*.5+[0,100,0]).astype(np.uint8);overlay[pred]=(overlay[pred]*.5+[100,0,100]).astype(np.uint8)
            cv2.putText(overlay,item['row']['split'],(5,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(overlay)
        cv2.imwrite(str(a.output.with_suffix('.overlay.jpg')),np.hstack(panels))
    finally:renderer.close();bridge.close()

if __name__=='__main__':main()
