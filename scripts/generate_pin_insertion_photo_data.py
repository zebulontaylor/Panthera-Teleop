#!/usr/bin/env python3
"""Audit snapshots of the existing physical photo-setup pin-insertion environment.

This generator never writes robot/part poses, changes optics, bypasses reset
checks, or teleports during rollout. Only the existing environment reset and
bounded environment.step actions move objects. Default output is diagnostic,
not approved for training. No robot, motor, or live camera interface is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np


CARRIER_MOUTH_Z_M=.01725
BORE_RADIUS_M=.00412  # CAD nominal bore radius; .00017m is task axis-error criterion, not clearance.
PIN_RADIUS_M=.004


def project(points,camera_position,camera_rotation,fovy,width,height):
    camera=(np.asarray(points)-camera_position)@camera_rotation
    depth=-camera[...,2]
    focal=.5*height/np.tan(np.deg2rad(fovy)*.5)
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=np.stack((.5*width+focal*camera[...,0]/depth,.5*height-focal*camera[...,1]/depth),axis=-1)/[width,height]
    return uv,depth


def landmarks(env):
    """Pin-2 bottom-axis center and its TRUE carrier upper bore mouth."""
    parts=env.sim.manifest['parts']
    local=np.asarray(parts[env.pin_name]['assembled_origin'])-np.asarray(parts['large_carrier']['assembled_origin'])
    local=local.copy();local[2]=CARRIER_MOUTH_Z_M
    carrier_rotation=env.data.xmat[env.carrier_id].reshape(3,3)
    hole=env.data.xpos[env.carrier_id]+carrier_rotation@local
    tip=env.data.xpos[env.pin_id].copy()
    relative=carrier_rotation.T@(tip-hole)
    return np.stack((tip,hole)),relative


def contact_summary(env):
    model,data=env.model,env.data
    external=[]
    import mujoco
    force=np.empty(6)
    for i,contact in enumerate(data.contact):
        b1,b2=model.geom_bodyid[[contact.geom1,contact.geom2]]
        if env.pin_id not in (b1,b2):continue
        other=int(b2 if b1==env.pin_id else b1)
        if other not in (env.carrier_id,env.gear_id):continue
        mujoco.mj_contactForce(model,data,i,force)
        external.append({'other_body':model.body(other).name,'distance_m':float(contact.dist),'normal_force_n':float(max(0.,force[0]))})
    return external


def classify_snapshot(relative,contacts,info):
    lateral=float(np.linalg.norm(relative[:2]));height=float(relative[2])
    carrier_contact=any(c['other_body']=='large_carrier' and c['normal_force_n']>1e-5 for c in contacts)
    supported=bool(info.get('pinched') and info.get('holder_pinched') and info.get('carrier_supported'))
    if info.get('failure'):return 'terminated_failure',False
    if carrier_contact and abs(height)<=.0015 and supported:
        if lateral>=BORE_RADIUS_M+PIN_RADIUS_M:return 'carrier_surface_outside_hole',True
        if lateral>BORE_RADIUS_M-PIN_RADIUS_M:return 'carrier_rim_contact',True
        return 'aligned_contact',False
    if height>.0015:return 'hover_above_carrier',False
    return 'unclassified_geometry',False


def visibility(uv,depth,depth_image,segmentation,pin_geom_ids,tolerance_m=.001):
    h,w=depth_image.shape
    in_frame=(depth>0)&np.isfinite(uv).all(axis=1)&np.all((uv>=0)&(uv<1),axis=1)
    direct=np.zeros(2,bool);own_pin=np.zeros(2,bool)
    for j in range(2):
        if not in_frame[j]:continue
        x,y=np.minimum((uv[j]*[w,h]).astype(int),[w-1,h-1])
        direct[j]=depth_image[y,x]>=depth[j]-tolerance_m
        # MuJoCo segmentation channels are object id, object type.
        own_pin[j]=int(segmentation[y,x,0]) in pin_geom_ids and int(segmentation[y,x,1])==5
    position_valid=np.array([in_frame[0] and (direct[0] or own_pin[0]),in_frame[1] and direct[1]])
    amodal_valid=in_frame.copy()
    return direct,position_valid,amodal_valid,own_pin


def aperture_visibility_fraction(env,hole,camera,width,height,depth_image):
    """Visible area proxy from the bore interior, separate from center visibility."""
    angle=np.linspace(0,2*np.pi,24,endpoint=False)
    local=np.concatenate((np.zeros((1,3)),np.array([[radius*np.cos(a),radius*np.sin(a),0.] for radius in (BORE_RADIUS_M*.45,BORE_RADIUS_M*.85) for a in angle])))
    rotation=env.data.xmat[env.carrier_id].reshape(3,3)
    world=hole+local@rotation.T
    uv,depth=project(world,env.data.cam_xpos[camera],env.data.cam_xmat[camera].reshape(3,3),float(env.model.cam_fovy[camera]),width,height)
    inside=(depth>0)&np.isfinite(uv).all(axis=1)&np.all((uv>=0)&(uv<1),axis=1)
    visible=np.zeros(len(world),bool)
    for i in np.flatnonzero(inside):
        x,y=(uv[i]*[width,height]).astype(int)
        visible[i]=depth_image[y,x]>=depth[i]-.001
    return float(visible.mean())


def bounded_action(policy,step,info,max_action=.2):
    if not np.isfinite(max_action) or not 0<=max_action<=.25:raise ValueError('action bound must lie in [0,.25]')
    action=np.zeros(14,np.float32)
    if policy in ('descend','descend-to-contact'):
        # Existing env XYZ action units: left 1mm, right .25mm per control step.
        # Only lower toward the upper mouth; no lateral correction or gripping changes.
        if info.get('pinched') and info.get('holder_pinched') and info.get('carrier_supported') and not (policy=='descend-to-contact' and info.get('_descent_contact_latched')):
            action[2]=-max_action
    elif policy!='hold':raise ValueError('policy must be hold, descend, or descend-to-contact')
    return action


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def apply_camera_registration_checked(env,path,helper):
    """Reject any registration helper that changes physics or part rest geometry."""
    protected={}
    for owner,names in ((env.data,('qpos','qvel','ctrl','act')),(env.model,('body_pos','body_quat','geom_pos','geom_quat','geom_rgba'))):
        for name in names:protected[(id(owner),name)]=np.asarray(getattr(owner,name)).copy()
    result=helper(env,path)
    for owner,names in ((env.data,('qpos','qvel','ctrl','act')),(env.model,('body_pos','body_quat','geom_pos','geom_quat','geom_rgba'))):
        for name in names:
            if not np.array_equal(getattr(owner,name),protected[(id(owner),name)]):raise ValueError(f'camera registration changed protected physics/rest field: {name}')
    return result


def generate(args):
    if args.episodes<1 or not 0<=args.steps<=400 or args.stride<1 or args.height<32:raise ValueError('invalid episode/step/image configuration')
    if not np.isfinite(args.max_action) or not 0<=args.max_action<=.25:raise ValueError('bounded action magnitude must be <=.25')
    import mujoco
    from pin_insertion_environment import create_photo_environment,PhotoSetupBridge
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    env=create_photo_environment(render_mode=None,setup=getattr(args,'setup','registered'),
        registration_profile=getattr(args,'registration_profile',None))
    bridge=PhotoSetupBridge(environment=env)
    camera_registration=None
    if getattr(args,'camera_registration',None):
        # Only the validated shared photo helper may apply an optical registration.
        from pin_insertion_environment import apply_photo_camera_registration
        camera_registration=apply_camera_registration_checked(env,args.camera_registration,apply_photo_camera_registration)
    initial_provenance=jsonable(bridge.provenance())
    width=round(args.height*4/3)
    renderer=None;records=[];audits=[]
    try:
        camera=env.model.camera('right_wrist').id
        renderer=mujoco.Renderer(env.model,height=args.height,width=width)
        options=mujoco.MjvOption();options.geomgroup[3]=0;options.sitegroup[:]=0
        # Optics are read once and checked after every real environment step.
        optics=(env.model.cam_pos[camera].copy(),env.model.cam_quat[camera].copy(),float(env.model.cam_fovy[camera]))
        pin_geoms=set(np.flatnonzero(env.model.geom_bodyid==env.pin_id).tolist())
        for episode in range(args.episodes):
            seed=args.seed+episode
            _,info=env.reset(seed=seed,options={'randomize':not args.nominal})
            initial_time=float(env.data.time);steps=[];episode_done=False;contact_latched=False;contact_events=[]
            for step in range(args.steps+1):
                if step%args.stride==0 or episode_done:
                    points,relative=landmarks(env)
                    uv,depth=project(points,env.data.cam_xpos[camera],env.data.cam_xmat[camera].reshape(3,3),float(env.model.cam_fovy[camera]),width,args.height)
                    renderer.update_scene(env.data,camera='right_wrist',scene_option=options)
                    rgb=renderer.render().copy()
                    renderer.enable_depth_rendering();depth_image=renderer.render().copy();renderer.disable_depth_rendering()
                    renderer.enable_segmentation_rendering();segmentation=renderer.render().copy();renderer.disable_segmentation_rendering()
                    direct,valid,amodal,own_pin=visibility(uv,depth,depth_image,segmentation,pin_geoms)
                    aperture_fraction=aperture_visibility_fraction(env,points[1],camera,width,args.height,depth_image)
                    contacts=contact_summary(env);regime,geometric_active=classify_snapshot(relative,contacts,info)
                    right=env.model.body('right_base').id
                    right_r=env.data.xmat[right].reshape(3,3)
                    delta=right_r.T@(points[1]-points[0]);delta*=min(1.,.004/max(np.linalg.norm(delta),1e-30))
                    # Gate labels are diagnostic geometric categories, not safe-to-act labels.
                    finite_uv=np.where(np.isfinite(uv),uv,.5).clip(0,1)
                    carrier=env.data.xpos[env.carrier_id];camera_r=env.data.cam_xmat[camera].reshape(3,3)
                    records.append({'images':rgb,'state':bridge.native_state(),
                        'qpos':env.data.qpos.copy(),'qvel':env.data.qvel.copy(),'ctrl':env.data.ctrl.copy(),
                        'act':env.data.act.copy(),'qacc_warmstart':env.data.qacc_warmstart.copy(),'keypoints':finite_uv.flatten(),
                        'keypoints_valid':np.repeat(valid,2),'amodal_keypoints_valid':np.repeat(amodal,2),'direct_visible':direct,
                        'own_pin_occlusion':own_pin,'hole_aperture_visible_fraction':[aperture_fraction],
                        'hole_appearance':'open' if aperture_fraction>.9 else 'partial_aperture' if aperture_fraction>.05 else 'occluded_aperture','delta_xyz_m':delta,'active':[float(geometric_active)],'gate_valid':[False],
                        'group_ids':f'photo_environment_seed_{seed}','episode_seed':seed,'step':step,'sim_time_s':float(env.data.time),
                        'regime':regime,'relative_pin_to_mouth_carrier_m':relative,'carrier_in_right_camera_m':camera_r.T@(carrier-env.data.cam_xpos[camera]),
                        'hole_memory_eligible':[bool(direct[1] and info.get('holder_pinched'))]})
                    if not records[-1]['state'].shape==(14,) or not np.isfinite(records[-1]['state']).all():raise ValueError('bridge did not return finite native14 physics state')
                    if len(records)<=8:
                        from PIL import Image
                        Image.fromarray(rgb).save(output.parent/(output.stem+f'-sample-{len(records):03d}.png'))
                if step==args.steps or episode_done:break
                physical_contacts=contact_summary(env)
                positive_contact=any(c['normal_force_n']>.05 for c in physical_contacts)
                positive_peak=float(info.get('pin_contact_load_n',0.))>.05
                if args.policy=='descend-to-contact' and not contact_latched and (positive_contact or positive_peak):
                    contact_latched=True
                    contact_events.append({'event':'descent_stopped_on_physical_contact','step':step,'sim_time_s':float(env.data.time),
                        'contacts':physical_contacts,'peak_pin_external_load_n':float(info.get('pin_contact_load_n',0.)),
                        'source':'current pin-carrier/gear contact force or previous control interval measured external pin-contact peak'})
                action=bounded_action(args.policy,step,dict(info,_descent_contact_latched=contact_latched),args.max_action)
                before=float(env.data.time)
                _,reward,terminated,truncated,info=env.step(action)
                if env.data.time<=before:raise RuntimeError('physics time did not advance')
                if not np.array_equal(env.model.cam_pos[camera],optics[0]) or not np.array_equal(env.model.cam_quat[camera],optics[1]) or float(env.model.cam_fovy[camera])!=optics[2]:raise RuntimeError('environment changed wrist camera optics')
                steps.append({'step':step+1,'sim_time_s':float(env.data.time),'action':action.tolist(),'reward':float(reward),
                    'terminated':bool(terminated),'truncated':bool(truncated),'contact_latched':contact_latched,'pin_carrier_gear_contacts_after_step':contact_summary(env),'info':jsonable(info)})
                episode_done=bool(terminated or truncated)
            audits.append({'seed':seed,'initial_sim_time_s':initial_time,'final_sim_time_s':float(env.data.time),'reset':jsonable(env.reset_parameters),'contact_events':contact_events,'steps':steps})
        if not records:raise RuntimeError('no snapshots captured')
        arrays={key:np.asarray([row[key] for row in records]) for key in records[0]}
        for key in ('state','keypoints','keypoints_valid','amodal_keypoints_valid','delta_xyz_m','active'):arrays[key]=arrays[key].astype(np.float32)
        metadata={'schema_version':1,'generator':'generate_pin_insertion_photo_data.py','environment':'sim.pin_insertion_env.BimanualPinInsertionEnv',
            'physics_simulated':True,'mujoco_version':mujoco.__version__,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'diagnostic_only':True,'training_use_approved':False,'real_calibrated':False,
            'setup_profile':'photo_20261027','camera':'right_wrist','image_height':args.height,'image_width':width,
            'objects':{'pin':'pin_2','carrier':'large_carrier','gear':'gear_3'},'controlled_arm':'left','controlled_object':'pin','delta_frame':'right_native_base','delta_units':'meters',
            'keypoints_order':['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v'],'carrier_mouth_z_local_m':CARRIER_MOUTH_Z_M,
            'individual_bore_radius_m':BORE_RADIUS_M,'bore_geometry_semantics':'Nominal8.24mm bore diameter verified original visual/collision sections; faceted/decomposed minimumradius~4.111mm. Task0.17mm axis-error success criterion is NOT geometric clearance. Earlier4.17mm proxy artifacts retained as superseded labeling approximation.','pin_radius_m':PIN_RADIUS_M,'state_order':'left joints1..6,gripper,right joints1..6,gripper (radians)',
            'state_origin':'current physical qpos via PhotoSetupBridge.native_state; joint positions are not commanded targets; gripper radians use endpoint-only bridge approximation',
            'gate_semantics':'diagnostic contact geometry only; gate_valid is false for every snapshot; not motor/safety supervision',
            'keypoint_visibility':'1mm external depth tolerance; pin-axis bottom center may be hidden by its own pin geometry; hole position loss direct-visible only',
            'hole_aperture_visibility':'49 depth-tested interior samples distinguish partial aperture from center visibility; approximate projected-area proxy, not a learned confidence',
            'amodal_semantics':'separate privileged geometric coordinates/validity; inference memory may initialize only through explicit visible-evidence protocol',
            'motion_semantics':'part poses assigned only inside existing env.reset; generator never writes body poses or qpos; rollouts only env.step bounded actions',
            'bridge_provenance':jsonable(bridge.provenance()),'initial_camera_and_profile_provenance':initial_provenance,
            'initial_right_wrist_optics':{'position_parent_m':optics[0].tolist(),'quaternion_parent':optics[1].tolist(),'fovy_deg':optics[2]},
            'camera_registration':jsonable(camera_registration),'camera_registration_file_sha256':hashlib.sha256(Path(args.camera_registration).read_bytes()).hexdigest() if getattr(args,'camera_registration',None) else None,
            'full_state_capture':'qpos,qvel,ctrl,act,qacc_warmstart at every snapshot; reconstruct same model/profile before mj_forward for rendering',
            'model_dimensions':{'nq':env.model.nq,'nv':env.model.nv,'nu':env.model.nu,'na':env.model.na},'arguments':vars(args),'rollout_audits':audits,
            'category_counts':{str(k):int(v) for k,v in zip(*np.unique(arrays['regime'],return_counts=True))}}
        arrays['metadata_json']=np.asarray(json.dumps(metadata,allow_nan=False));np.savez_compressed(output,**arrays)
        output.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        return metadata
    finally:
        if renderer is not None:renderer.close()
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=1);p.add_argument('--steps',type=int,default=5)
    p.add_argument('--stride',type=int,default=1);p.add_argument('--height',type=int,default=480);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--setup',choices=['registered','nominal'],default='registered',help='environment setup; nominal explicitly opts out of registered camera/base setup')
    p.add_argument('--registration-profile',help='optional registered setup descriptor forwarded unchanged to the photo factory')
    p.add_argument('--camera-registration',help='validated photo camera registration applied once before rollout; no physics state changes')
    p.add_argument('--policy',choices=['hold','descend','descend-to-contact'],default='hold');p.add_argument('--max-action',type=float,default=.2);p.add_argument('--nominal',action='store_true',help='disable reset randomization only; use --setup nominal to select the baseline environment')
    print(json.dumps(generate(p.parse_args()),indent=2))

if __name__=='__main__':main()
