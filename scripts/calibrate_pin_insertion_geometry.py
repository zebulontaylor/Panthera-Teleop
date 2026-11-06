#!/usr/bin/env python3
"""Offline, prior-regularized camera/grasp hypotheses from saved real pixels.

No hardware is accessed. Reprojection fits are not independent metric validation.
Transforms T_A_B map B coordinates into A. Camera coordinates are OpenCV.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def rigid(rotation=None, translation=None):
    t=np.eye(4)
    if rotation is not None: t[:3,:3]=rotation
    if translation is not None: t[:3,3]=translation
    return t


def validate_transform(value):
    t=np.asarray(value,dtype=float)
    if (t.shape!=(4,4) or not np.isfinite(t).all() or
        not np.allclose(t[3],[0,0,0,1],atol=1e-7) or
        not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or
        not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-5)):
        raise ValueError('Expected finite proper rigid transform')
    return t


def pin_camera_points(left_tools,right_tools,camera_tool,base_left,pin_offsets):
    """Right-base/left-base pose chain with no interchangeable base frames."""
    left=np.asarray(left_tools);right=np.asarray(right_tools)
    points=np.einsum('nij,nj->ni',left[:,:3,:3],np.asarray(pin_offsets))+left[:,:3,3]
    points=points@base_left[:3,:3].T+base_left[:3,3]
    camera=np.asarray(right)@camera_tool
    return np.einsum('nji,nj->ni',camera[:,:3,:3],points-camera[:,:3,3])


def project_opencv(points,K,distortion):
    p=np.asarray(points); z=np.maximum(p[:,2],1e-5)
    xy=p[:,:2]/z[:,None]
    x,y=xy.T;r2=np.minimum(x*x+y*y,1e4)
    k1,k2,p1,p2,k3=distortion
    radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x)
    yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    return np.column_stack([K[0,0]*xd+K[0,2],K[1,1]*yd+K[1,2]])


def numerical_jacobian(function,x,step=1e-5):
    columns=[]
    for i in range(len(x)):
        d=np.zeros(len(x));d[i]=step
        columns.append((function(x+d)-function(x-d))/(2*step))
    return np.stack(columns,axis=1)


def jacobian_report(jacobian,names):
    _,singular,vh=np.linalg.svd(jacobian,full_matrices=True)
    threshold=max(float(singular[0]) if len(singular) else 0.,1.)*1e-6
    rank=int(np.sum(singular>threshold))
    directions=[]
    for vector in vh[rank:rank+5]:
        order=np.argsort(abs(vector))[-5:][::-1]
        directions.append({names[i]:float(vector[i]) for i in order})
    return dict(rows=int(jacobian.shape[0]),columns=int(jacobian.shape[1]),rank=rank,
        relative_singular_threshold=1e-6,singular_values=singular.tolist(),
        condition_number=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>1e-12 else None,
        null_or_weak_directions=directions,prior_rows_excluded=True,
        parameter_scaling='dimensionless prior-standard-deviation units')


def linearized_uncertainty(jacobian,sigmas,names):
    # Unit-variance whitened pixel error; priors explicitly excluded.
    _,singular,vh=np.linalg.svd(jacobian,full_matrices=False)
    inverse=np.where(singular>max(singular[0],1.)*1e-9,1/np.maximum(singular,1e-15)**2,0)
    covariance=(vh.T*inverse)@vh
    std=np.sqrt(np.maximum(np.diag(covariance),0))*sigmas
    return dict(assumption='Independent pixel annotation sigma from each label; local linearization, priors excluded; not a confidence guarantee or metric validation',
        one_sigma_parameter_std={n:float(v) for n,v in zip(names,std)},
        nullspace_not_bounded=bool(np.linalg.matrix_rank(jacobian)<len(names)))


def default_prior(scene,rows=None):
    from real2sim_bridge import Real2SimBridge
    b=Real2SimBridge(scene)
    sid=b.sites[1];cid=b.model.camera('right_wrist').id
    world_tool=rigid(b.data.site_xmat[sid].reshape(3,3),b.data.site_xpos[sid])
    world_camera=rigid(b.data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]),b.data.cam_xpos[cid])
    world_base=rigid(b.data.xmat[b.base_ids[1]].reshape(3,3),b.data.xpos[b.base_ids[1]])
    fy=240/np.tan(np.deg2rad(b.model.cam_fovy[cid]/2))
    camera_tool=np.linalg.inv(world_tool)@world_camera
    base_left=np.linalg.inv(world_base)
    provenance='MuJoCo camera mount and FOV prior, NOT measured real calibration'
    if rows:
        # Parallel-base initialization is a hypothesis, not a measured transform.
        # Estimate translation from train pixels at an assumed 0.12m optical depth.
        base_left=np.eye(4);estimates=[]
        for row in rows:
            if row['split']!='train':continue
            ray=np.r_[(row['xy']-[320,240])/fy,1.]*.12
            cam=row['right']@camera_tool
            p_right=cam[:3,:3]@ray+cam[:3,3]
            estimates.append(p_right-row['left'][:3,3])
        base_left[:3,3]=np.median(estimates,axis=0)
        provenance+='; parallel-base orientation and translation from training rays at assumed0.12m depth'
    return dict(T_right_tool_camera_opencv=camera_tool.tolist(),
        T_right_base_left_base=base_left.tolist(),pin_tip_left_tool_m=[0,0,0],
        K=[[fy,0,320],[0,fy,240],[0,0,1]],distortion=[0]*5,
        provenance=provenance)


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_observations(path,recordings_root):
    """Accept ready transform rows or assembled manual labels; pair exact saved FK."""
    obj=json.loads(Path(path).read_text())
    raw=obj if isinstance(obj,list) else obj.get('frames',obj.get('observations',obj.get('rows',[])))
    result=[]; cache={}; rejected=[]
    for index,row in enumerate(raw):
        eid=str(row.get('episode_id',row.get('episode','')))
        split=row.get('split','train')
        if split=='val': split='validation'
        if split not in ('train','validation','test'): raise ValueError('Explicit train/validation/test split required')
        ready='T_left_base_tool' in row
        lm=row.get('labels',{}).get('views',{}).get('wrist_port2',{}).get('landmarks',{}).get('pin_tip',{})
        xy=row.get('pin_tip_px',lm.get('xy_px'))
        if xy is None or (not ready and lm.get('visibility') not in ('visible','partially_occluded','occluded')):
            rejected.append(dict(index=index,reason='no_pin_pixel'));continue
        if not ready:
            grasp=row.get('labels',{}).get('grasp',{}).get('left',{})
            if grasp.get('object')!='pin':
                rejected.append(dict(index=index,reason='left_pin_grasp_not_labeled'));continue
            episode=Path(recordings_root)/eid
            if eid not in cache:
                meta=json.loads((episode/'metadata.json').read_text())
                states=[json.loads(line) for line in (episode/'states.jsonl').read_text().splitlines() if line.strip()]
                cache[eid]=(meta,states,np.array([r['monotonic_ns'] for r in states],dtype=np.int64))
            meta,states,times=cache[eid]
            image=row['images']['wrist_port2']
            stamp=int(image['monotonic_ns']);i=int(np.argmin(abs(times-stamp)))
            skew=abs(int(times[i])-stamp)/1e9
            if skew>.03:
                rejected.append(dict(index=index,reason='state_image_skew_over_30ms'));continue
            tools=[]
            for side in ('left','right'):
                pose=states[i]['arms'][side]['tool_pose_base']
                tools.append(rigid(pose['rotation'],pose['position_m']))
        else:
            tools=[np.asarray(row['T_left_base_tool']),np.asarray(row['T_right_base_tool'])];skew=abs(row.get('state_image_skew_s',row.get('camera_sample_skew_ms',0)/1000))
        tools=[validate_transform(t) for t in tools]
        point=np.asarray(xy,dtype=float)
        if point.shape!=(2,) or not np.isfinite(point).all(): raise ValueError('Invalid pin pixel')
        uncertainty=row.get('uncertainty_px',lm.get('uncertainty_px',3)) or 3
        if isinstance(uncertainty,list): uncertainty=max(uncertainty)
        result.append(dict(episode_id=eid,grasp_group_id=str(row.get('grasp_group_id',row.get('session_id',eid))),split=split,
            observation_id=row.get('observation_id',row.get('id',str(index))),left=tools[0],right=tools[1],xy=point,
            sigma=max(float(uncertainty),1.),state_image_skew_s=float(skew),width=int(row.get('width',640)),height=int(row.get('height',480)),
            carrier_bores_px=row.get('carrier_bores_px')))
    if not result: raise ValueError('No usable labeled pin observations')
    return result,rejected


def pose_excitation(rows):
    result={}
    for side in ('left','right'):
        poses=np.array([r[side] for r in rows])
        translations=poses[:,:3,3]
        rotations=Rotation.from_matrix(np.einsum('ij,njk->nik',poses[0,:3,:3].T,poses[:,:3,:3])).as_rotvec()
        result[side]=dict(translation_axis_span_m=np.ptp(translations,axis=0).tolist(),
            centered_translation_singular_values=np.linalg.svd(translations-translations.mean(axis=0),compute_uv=False).tolist(),
            relative_rotation_singular_values_rad=np.linalg.svd(rotations,compute_uv=False).tolist())
    result['grasp_frame_counts']={g:sum(r['grasp_group_id']==g for r in rows) for g in sorted({r['grasp_group_id'] for r in rows})}
    result['max_state_image_skew_s']=max((r.get('state_image_skew_s',0) for r in rows),default=0)
    return result


def fit(rows,prior,*,fit_intrinsics=False,fit_distortion=False,fit_base=False,fit_camera=True,fit_grasp=True,planar_base=False,base_translation_bound_m=None,base_rotation_bound_deg=None,starts=4,seed=42):
    if starts<1:raise ValueError('starts must be positive')
    train=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']!='train']
    if len(train)<6: raise ValueError('Require at least six training pin observations')
    train_e={r['episode_id'] for r in train}; held_e={r['episode_id'] for r in held}
    if train_e & held_e: raise ValueError('Episode leakage between training and heldout')
    if len({(r['width'],r['height']) for r in rows})!=1: raise ValueError('All pixel labels must use same image size')
    cam0=validate_transform(prior['T_right_tool_camera_opencv'])
    base0=validate_transform(prior['T_right_base_left_base'])
    if planar_base:
        yaw=np.arctan2(base0[1,0],base0[0,0]);base0=rigid(Rotation.from_rotvec([0,0,yaw]).as_matrix(),[base0[0,3],base0[1,3],0.])
    pin0=np.asarray(prior['pin_tip_left_tool_m'],float)
    k0=np.asarray(prior['K'],float);dist0=np.asarray(prior.get('distortion',[0]*5),float)
    names=[];sigmas=[]
    def block(label,values):
        start=len(names);names.extend([f'{label}_{i}' for i in range(len(values))]);sigmas.extend(values);return slice(start,len(names))
    cam_r=block('camera_rotation',[.35]*3) if fit_camera else None
    cam_t=block('camera_translation',[.04]*3) if fit_camera else None
    pin_s=block('mean_pin_offset',[.025]*3) if fit_grasp else None
    groups=sorted({r['grasp_group_id'] for r in train})
    group_s={g:block('grasp_'+g,[.01]*3) for g in groups[:-1]} if fit_grasp and len(groups)>1 else {}
    base_r=block('base_yaw',[.5]) if fit_base and planar_base else block('base_rotation',[.5]*3) if fit_base else None
    base_t=block('base_xy',[.15]*2) if fit_base and planar_base else block('base_translation',[.15]*3) if fit_base else None
    k_s=block('intrinsics',[.15,.15,30,30]) if fit_intrinsics else None
    d_s=block('distortion',[.1,.1,.02,.02,.1]) if fit_distortion else None
    sigmas=np.asarray(sigmas)
    lower=np.full(len(names),-4.);upper=np.full(len(names),4.)
    if fit_base and base_translation_bound_m is not None:
        lower[base_t]=-float(base_translation_bound_m)/sigmas[base_t];upper[base_t]=-lower[base_t]
    if fit_base and base_rotation_bound_deg is not None:
        lower[base_r]=-np.deg2rad(float(base_rotation_bound_deg))/sigmas[base_r];upper[base_r]=-lower[base_r]
    def decode(z):
        x=z*sigmas
        camera=rigid(Rotation.from_rotvec(x[cam_r]).as_matrix()@cam0[:3,:3],cam0[:3,3]+x[cam_t]) if fit_camera else cam0.copy()
        base=base0.copy()
        if fit_base:
            base=rigid(Rotation.from_rotvec(np.r_[0.,0.,x[base_r][0]] if planar_base else x[base_r]).as_matrix()@base0[:3,:3],base0[:3,3]+(np.r_[x[base_t],0.] if planar_base else x[base_t]))
        k=k0.copy()
        if fit_intrinsics:
            k[0,0]*=np.exp(x[k_s][0]);k[1,1]*=np.exp(x[k_s][1]);k[0,2]+=x[k_s][2];k[1,2]+=x[k_s][3]
        distortion=dist0.copy()
        if fit_distortion: distortion+=x[d_s]
        return camera,base,(pin0+x[pin_s] if fit_grasp else pin0.copy()),k,distortion,x
    def group_delta(group,x):
        if group in group_s:return x[group_s[group]]
        if group_s and group==groups[-1]:return -sum((x[s] for s in group_s.values()),start=np.zeros(3))
        return np.zeros(3)
    def predict(z,data):
        camera,base,pin,k,distortion,x=decode(z)
        offsets=np.array([pin+group_delta(r['grasp_group_id'],x) for r in data])
        points=pin_camera_points([r['left'] for r in data],[r['right'] for r in data],camera,base,offsets)
        return project_opencv(points,k,distortion),points[:,2]
    def image_residual(z,data):
        pred,_=predict(z,data)
        return ((pred-np.array([r['xy'] for r in data]))/np.array([r['sigma'] for r in data])[:,None]).ravel()
    def objective(z):
        _,depth=predict(z,train)
        return np.r_[image_residual(z,train),z,np.maximum(.005-depth,0)*1000]
    rng=np.random.default_rng(seed); solutions=[]
    for i in range(starts):
        initial=np.zeros(len(names)) if i==0 else rng.normal(0,.5,len(names))
        initial=np.clip(initial,lower+1e-8,upper-1e-8)
        solution=least_squares(objective,initial,bounds=(lower,upper),loss='soft_l1',f_scale=2,max_nfev=1500)
        solutions.append(solution)
    best=min(solutions,key=lambda s:s.cost)
    camera,base,pin,k,distortion,x=decode(best.x)
    image_jac=numerical_jacobian(lambda z:image_residual(z,train),best.x)
    rank=jacobian_report(image_jac,names)
    def errors(data,z):
        if not data:return dict(samples=0)
        pred,depth=predict(z,data);e=np.linalg.norm(pred-np.array([r['xy'] for r in data]),axis=1)
        return dict(samples=len(data),mean_px=float(e.mean()),median_px=float(np.median(e)),p95_px=float(np.percentile(e,95)),max_px=float(e.max()),
            nonpositive_depth=int(np.sum(depth<=0)),rows=[dict(observation_id=r['observation_id'],episode_id=r['episode_id'],split=r['split'],
                predicted_pin_tip_px=p.tolist(),labeled_pin_tip_px=r['xy'].tolist(),error_px=float(v)) for r,p,v in zip(data,pred,e)])
    params=np.array([s.x for s in solutions]);spread=params.std(axis=0)*sigmas
    sensitivities=sorted([dict(parameter=n,std_across_starts=float(v)) for n,v in zip(names,spread)],key=lambda r:r['std_across_starts'],reverse=True)
    result=dict(schema_version=1,fit_mode='base_only_fixed_camera_and_grasp' if (fit_base and not fit_camera and not fit_grasp and not fit_intrinsics and not fit_distortion) else 'joint_hypothesis_fit',kind='prior_regularized_camera_grasp_hypothesis',real_calibrated=False,metric_accuracy_validated=False,
        image_width=int(rows[0]['width']),image_height=int(rows[0]['height']),transform_convention='T_A_B maps B into A; camera OpenCV +x right,+y down,+z forward',
        T_right_tool_camera_opencv=camera.tolist(),T_right_base_left_base=base.tolist(),pin_tip_left_tool_m=pin.tolist(),K=k.tolist(),distortion=distortion.tolist(),
        per_grasp_pin_tip_left_tool_m={g:(pin+group_delta(g,x)).tolist() for g in groups}, grasp_parameterization='shared mean plus zero-sum train-grasp deviations, avoiding redundant mean/offset gauge' if fit_grasp else 'fixed supplied physically observed tool-to-pin offset, no per-grasp optimization',
        fixed_parameters=dict(base_planar=planar_base,grasp=not fit_grasp,camera=not fit_camera,intrinsics=not fit_intrinsics,distortion=not fit_distortion,base=not fit_base),
        fit_settings=dict(starts=starts,seed=seed,loss='soft_l1',max_prior_deviations=4,planar_base=planar_base,base_translation_bound_m=base_translation_bound_m,base_rotation_bound_deg=base_rotation_bound_deg,training_episodes=sorted(train_e),heldout_episodes=sorted(held_e)),
        prior=prior,training_pose_excitation=pose_excitation(train),training=errors(train,best.x),heldout=errors(held,best.x),validation=errors([r for r in held if r['split']=='validation'],best.x),test=errors([r for r in held if r['split']=='test'],best.x),data_only_identifiability=rank,
        multistart=[dict(cost=float(s.cost),success=bool(s.success),nfev=int(s.nfev),train_mean_px=errors(train,s.x)['mean_px'],heldout_mean_px=errors(held,s.x).get('mean_px')) for s in solutions],
        data_only_linearized_uncertainty=linearized_uncertainty(image_jac,sigmas,names),parameter_sensitivity=sensitivities,parameter_names=names,parameter_prior_standard_deviations=sigmas.tolist(),parameters_at_prior_bound=[n for i,(n,v) in enumerate(zip(names,best.x)) if abs(v-lower[i])<1e-4 or abs(v-upper[i])<1e-4],optimizer_converged=bool(best.success),
        limitations=['Camera-carrier bores alone constrain only combined camera-to-carrier pose; separate camera mount and carrier grasp are not identifiable without extra constraints.',
            'With a stationary right wrist only the combined camera/base mapping is identifiable; diverse right wrist rotations and an external base constraint are needed to separate it.',
            'Inter-base registration and focal scale are prior assumptions unless explicitly released; photographic setup profiles are not measured calibration.',
            'A pin endpoint gives no pin orientation or physical contact measurement; fixed rigid no-slip grasp per group assumed.',
            'Per-training-grasp offsets fit training pixels only; unseen heldout grasps use shared mean, not heldout labels.',
            'Robust priors stabilize numerically but do not establish observability; rank excludes all prior rows.',
            'Data-only local rank and multistart spread do not establish global identifiability or independent metric accuracy.',
            'Manual pin pixel error, tool FK convention, image timing and lens distortion may dominate submillimeter nudges.'])
    if rank['rank']<rank['columns']:result['limitations'].append('Data-only Jacobian is rank deficient; exported transforms are nonunique prior-conditioned hypotheses.')
    if not held:result['limitations'].append('No heldout observations supplied; out-of-demo generalization is untested.')
    result['carrier_pose_hypotheses']=carrier_hypotheses(rows,camera,k,distortion)
    return result


def carrier_hypotheses(rows,camera_tool,K,distortion):
    from real2sim_object_pose import fit_carrier_pose
    results=[]
    for row in rows:
        if row['split']!='train':continue
        points=row.get('carrier_bores_px')
        if points is None or np.asarray(points).shape!=(4,2):continue
        pose=fit_carrier_pose(points,K,distortion,calibrated=False,max_reprojection_error_px=8)
        candidates=[]
        for h in pose['hypotheses'][:8]:
            tc=rigid(h['rotation_matrix'],h['translation_m'])@rigid(translation=[0,0,-.01725])
            candidates.append(dict(T_camera_carrier=tc.tolist(),T_right_tool_carrier=(camera_tool@tc).tolist(),reprojection_rms_px=h['reprojection_rms_px']))
        results.append(dict(observation_id=row['observation_id'],grasp_group_id=row['grasp_group_id'],split=row['split'],candidates=candidates,ambiguous=True,fit_kind='independent planar PnP of four LARGE OUTER counterbores, no camera mount constraint',yaw_symmetry_degrees=90,normal_sign_ambiguous=True))
    return results


def teacher_projections(manifest,result):
    """Project unlabeled TRAIN pins only; no promotion to measured ground truth."""
    raw=manifest if isinstance(manifest,list) else manifest.get('frames',[])
    output=[]
    camera=np.asarray(result['T_right_tool_camera_opencv']);base=np.asarray(result['T_right_base_left_base'])
    k=np.asarray(result['K']);dist=np.asarray(result['distortion'])
    for row in raw:
        if row.get('split')!='train' or row.get('pin_tip_px') is not None:continue
        if 'T_left_base_tool' not in row or 'T_right_base_tool' not in row:continue
        group=str(row.get('grasp_group_id',row.get('session_id',row.get('episode_id',''))))
        offset=result['per_grasp_pin_tip_left_tool_m'].get(group,result['pin_tip_left_tool_m'])
        point=pin_camera_points([validate_transform(row['T_left_base_tool'])],[validate_transform(row['T_right_base_tool'])],camera,base,[offset])
        xy=project_opencv(point,k,dist)[0]
        output.append(dict(id=row.get('id',row.get('observation_id')),episode_id=row.get('episode_id'),session_id=group,split='train',
            pin_tip_px=xy.tolist(),depth_m=float(point[0,2]),in_frame=bool(point[0,2]>0 and 0<=xy[0]<row.get('width',640) and 0<=xy[1]<row.get('height',480)),
            source_kind='calibration_teacher_hypothesis_not_ground_truth',visibility='UNOBSERVED',
            caveat='Rigid grasp assumed; projected endpoint may be occluded, absent after release, or wrong. Not eligible as independent calibration or test truth.'))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recordings-root',type=Path,default=Path('data/kept'))
    p.add_argument('--scene',type=Path,default=Path('/home/robotics/Desktop/RobotArmTraining/sim/actuator/scene_bimanual.xml'))
    p.add_argument('--planar-base',action='store_true');p.add_argument('--fix-grasp',action='store_true');p.add_argument('--base-translation-bound-m',type=float);p.add_argument('--base-rotation-bound-deg',type=float);p.add_argument('--fix-camera',action='store_true');p.add_argument('--teacher-output',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--starts',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fit-intrinsics',action='store_true');p.add_argument('--fit-distortion',action='store_true');p.add_argument('--fit-base',action='store_true',default=True);p.add_argument('--fix-base',dest='fit_base',action='store_false')
    a=p.parse_args();rows,rejected=load_observations(a.labels,a.recordings_root)
    prior=json.loads(a.prior.read_text()) if a.prior else default_prior(a.scene,rows)
    result=fit(rows,prior,fit_intrinsics=a.fit_intrinsics,fit_distortion=a.fit_distortion,fit_base=a.fit_base,fit_camera=not a.fix_camera,fit_grasp=not a.fix_grasp,planar_base=a.planar_base,base_translation_bound_m=a.base_translation_bound_m,base_rotation_bound_deg=a.base_rotation_bound_deg,starts=a.starts,seed=a.seed)
    if a.teacher_output:
        teacher=teacher_projections(json.loads(a.labels.read_text()),result)
        a.teacher_output.parent.mkdir(parents=True,exist_ok=True)
        a.teacher_output.write_text(json.dumps(dict(kind='teacher_hypotheses_not_ground_truth',frames=teacher),indent=2)+'\n')
    result['source']=dict(labels=str(a.labels.resolve()),sha256=_sha(a.labels),script_sha256=_sha(__file__),rejected=rejected)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('kind','training','heldout','data_only_identifiability') if k in result},indent=2))

if __name__=='__main__':main()
