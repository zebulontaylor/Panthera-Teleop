#!/usr/bin/env python3
"""Inspect the existing photograph setup, never fit camera/base/grasp parameters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT.parent/'RobotArmTraining'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(TRAIN)]
import numpy as np


def rigid(p,r):
    t=np.eye(4);t[:3,3]=p;t[:3,:3]=r;return t


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--carrier-visual',choices=['original','reproduced'],default='reproduced',help='Documented v4 appearance reproduction changes only visual geom')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'work/pin-insertion-real/manifest.json')
    p.add_argument('--sequence-episodes',nargs='*',help='Heldout saved forward episodes to compare native-q kinematic replay')
    p.add_argument('--sequence-output',type=Path)
    p.add_argument('--sequence-fps',type=float,default=7.5,help='Display every30Hz source frame at quarter speed by default')
    p.add_argument('--carrier-grasp',type=Path,help='Train-only fixed camera-conditioned carrier/tool rigid hypothesis')
    p.add_argument('--base-registration',type=Path,help='Base-only registration under fixed physical pin grasp and own camera')
    p.add_argument('--camera-registration',type=Path,help='Own-right-finger image registration candidate, rendered separately from nominal baseline')
    p.add_argument('--table-camera',type=Path,help='Optional independently fit table camera comparison; never overwrite nominal images')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    import cv2,mujoco
    from scripts.pin_insertion_environment import create_photo_environment,PhotoSetupBridge
    from scripts.pin_insertion_sim import _place
    args.output_dir.mkdir(parents=True,exist_ok=True)
    env=create_photo_environment(setup="nominal",carrier_visual_profile=args.carrier_visual=="reproduced")
    _,info=env.reset(seed=args.seed,options={'randomize':False})
    model,data=env.model,env.data
    renderer=mujoco.Renderer(model,height=480,width=640)
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0];option.sitegroup[:]=0
    def render(name):
        renderer.update_scene(data,camera=name,scene_option=option);return renderer.render().copy()
    def save(name,rgb):cv2.imwrite(str(args.output_dir/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    def tool(side):
        site=env.sim.arms[side].ee_site
        return rigid(data.site_xpos[site],data.site_xmat[site].reshape(3,3))
    def part(name):
        bid=model.body(name).id
        return rigid(data.xpos[bid],data.xmat[bid].reshape(3,3))
    snapshots={camera:render(camera) for camera in ('overhead','left_wrist','right_wrist')}
    for camera,image in snapshots.items():save('reset-'+camera+'.png',image)
    pin_grasp=np.linalg.inv(tool('left'))@part(env.pin_name)
    carrier_grasp=np.linalg.inv(tool('right'))@part('large_carrier')
    reset_carrier_grasp=carrier_grasp.copy()
    gear_carrier=np.linalg.inv(part('large_carrier'))@part('gear_3')
    carrier_grasp_profile=None
    if args.carrier_grasp:
        carrier_grasp_profile=json.loads(args.carrier_grasp.read_text())
        if args.camera_registration is None or carrier_grasp_profile['camera_sha256']!=hashlib.sha256(args.camera_registration.read_bytes()).hexdigest():raise ValueError('Carrier profile camera provenance mismatch')
        carrier_grasp=np.asarray(carrier_grasp_profile['T_right_tool_carrier'])
    base_registration_profile=None
    if args.base_registration:
        base_registration_profile=json.loads(args.base_registration.read_text())
        left_world=part('left_base');right_world=left_world@np.linalg.inv(np.asarray(base_registration_profile['T_right_base_left_base']))
        _place(env.sim,'right_base',right_world[:3,3],right_world[:3,:3]);mujoco.mj_setConst(model,data);mujoco.mj_forward(model,data)
    manifest=json.loads(args.manifest.read_text())
    representative=next(r for r in manifest['frames'] if r['split']=='train' and r.get('pin_tip_px') is not None and r['video_frame']==0)
    real=cv2.cvtColor(cv2.imread(representative['image_path']),cv2.COLOR_BGR2RGB)
    save('representative-real-train.png',real)
    def panel(image,label):
        canvas=np.zeros((515,640,3),dtype=np.uint8);canvas[35:]=image
        cv2.putText(canvas,label,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA);return canvas
    overview=np.hstack([panel(real,'REAL TRAIN: clear approach; recorded frame'),panel(snapshots['right_wrist'],'EXISTING PHOTO ENV: physical reset; nominal wrist camera')])
    save('real-versus-existing-reset.png',overview)
    q=np.asarray(representative['state14'],dtype=float)
    for i,side in enumerate(('left','right')):
        arm=env.sim.arms[side];armq=q[i*7:i*7+6]
        data.qpos[arm.arm_qadr]=armq
        opening=q[i*7+6]/2*.04;data.qpos[arm.finger_qadr]=[opening,-opening]
    data.qvel[:]=0;mujoco.mj_forward(model,data)
    max_q_error=max(float(np.max(np.abs(data.qpos[env.sim.arms[side].arm_qadr]-q[i*7:i*7+6]))) for i,side in enumerate(('left','right')))
    if max_q_error>1e-12:raise ValueError('Native arm joints changed')
    pin=tool('left')@pin_grasp;carrier=tool('right')@carrier_grasp;gear=carrier@gear_carrier
    for name,pose in ((env.pin_name,pin),('large_carrier',carrier),('gear_3',gear)):_place(env.sim,name,pose[:3,3],pose[:3,:3])
    mujoco.mj_forward(model,data)
    matched={camera:render(camera) for camera in ('overhead','left_wrist','right_wrist')}
    for camera,image in matched.items():save('native-q-replay-'+camera+'.png',image)
    comparison=np.hstack([panel(real,'REAL TRAIN: '+representative['episode_id'][:12]+' frame 0'),panel(matched['right_wrist'],'SIM: identical native arm q; existing reset grasp prior')])
    save('real-versus-native-q-replay.png',comparison)
    # Also retain a three-view exact existing physical reset overview; replay
    # images are kinematic placement, not claimed physically settled contact.
    save('existing-reset-three-cameras.png',np.hstack([panel(snapshots[c],c+' | original photograph profile reset') for c in snapshots]))
    nominal_cameras={c:{'parent_body':model.body(int(model.cam_bodyid[model.camera(c).id])).name,'position_local_m':model.cam_pos[model.camera(c).id].tolist(),'quaternion_local':model.cam_quat[model.camera(c).id].tolist(),'fovy_deg':float(model.cam_fovy[model.camera(c).id])} for c in snapshots}
    # Reconstruct only supplied CAD assembly relationships. The central shaft /
    # bearing-looking unit is the existing small_carrier mesh, never invented.
    cad=json.loads((TRAIN/'sim/actuator/manifest.json').read_text())['parts']
    assembled_names=('small_carrier','gear_1','gear_2','gear_3','gear_4')
    assembled_offsets={}
    for name in assembled_names:
        offset=np.asarray(cad[name]['assembled_origin'])-np.asarray(cad['large_carrier']['assembled_origin'])
        assembled_offsets[name]=offset.tolist()
        _place(env.sim,name,carrier[:3,3]+carrier[:3,:3]@offset,carrier[:3,:3])
    hidden_fixture_helpers=[]
    for geom_id in range(model.ngeom):
        name=model.geom(geom_id).name or ''
        if name=='fixture' or name.endswith('_depth_stop'):
            model.geom_group[geom_id]=3;hidden_fixture_helpers.append(name)
    mujoco.mj_forward(model,data)
    assembly_position_error=max(float(np.linalg.norm(data.xpos[model.body(name).id]-(carrier[:3,3]+carrier[:3,:3]@np.asarray(assembled_offsets[name])))) for name in assembled_names)
    if assembly_position_error>1e-10:raise ValueError('CAD assembly transform disagreement')
    complete={camera:render(camera) for camera in ('overhead','left_wrist','right_wrist')}
    for camera,image in complete.items():save('complete-cad-assembly-'+camera+'.png',image)
    save('real-versus-complete-cad-assembly.png',np.hstack([panel(real,'REAL TRAIN: held assembly'),panel(complete['right_wrist'],'SIM: supplied complete CAD; TRAIN-only carrier grasp')]))
    # Isolated CAD view makes supplied assembly outline/central unit inspectable
    # without pretending its viewpoint is calibrated to the real photograph.
    isolated=mujoco.MjvOption();isolated.geomgroup[:]=0;isolated.geomgroup[4]=1;isolated.sitegroup[:]=0
    selected_geoms=[model.geom(name+'_visual').id for name in ('large_carrier',*assembled_names)]
    old_groups=model.geom_group[selected_geoms].copy();model.geom_group[selected_geoms]=4
    camera_free=mujoco.MjvCamera();camera_free.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera_free.lookat[:]=carrier[:3,3]+carrier[:3,:3]@np.array([0,0,.009])
    camera_free.distance=.12;normal=carrier[:3,2]
    camera_free.azimuth=float(np.rad2deg(np.arctan2(normal[1],normal[0])))
    camera_free.elevation=float(-np.rad2deg(np.arcsin(np.clip(normal[2],-1,1))))
    renderer.update_scene(data,camera=camera_free,scene_option=isolated);save('canonical-complete-cad-unit.png',renderer.render().copy())
    model.geom_group[selected_geoms]=old_groups
    registered_camera_report=None
    if args.camera_registration:
        registration=json.loads(args.camera_registration.read_text());cid=model.camera('right_wrist').id
        model.cam_pos[cid]=registration['model_camera_local_position_m']
        quaternion=np.empty(4);mujoco.mju_mat2Quat(quaternion,np.asarray(registration['model_camera_local_rotation_mujoco']).ravel())
        model.cam_quat[cid]=quaternion;model.cam_fovy[cid]=registration['fovy_deg']
        right_arm=env.sim.arms['right']
        nominal_opening=float(data.qpos[right_arm.finger_qadr[0]])
        registered_opening=nominal_opening*float(registration.get('right_gripper_opening_scale',1))+float(registration.get('right_gripper_opening_offset_m',0))
        data.qpos[right_arm.finger_qadr]=[registered_opening,-registered_opening]
        mujoco.mj_forward(model,data);registered=render('right_wrist')
        save('complete-cad-assembly-registered-right_wrist.png',registered)
        diagnostic=registered.copy();camera_rotation=data.cam_xmat[cid].reshape(3,3);camera_position=data.cam_xpos[cid]
        focal=240/np.tan(np.deg2rad(model.cam_fovy[cid])/2);part_audit={}
        for pin_name in ('pin_1','pin_2','pin_3','pin_4'):
            pose=part(pin_name);center=pose[:3,3]+pose[:3,:3]@np.array([0,0,.00894])
            point=camera_rotation.T@(center-camera_position);depth=-point[2]
            pixel=[320+focal*point[0]/depth,240-focal*point[1]/depth]
            part_audit[pin_name]={'center_px':pixel,'camera_depth_m':float(depth),'held':pin_name==env.pin_name,'world_position_m':pose[:3,3].tolist()}
            if depth>0:
                cv2.putText(diagnostic,('HELD ' if pin_name==env.pin_name else 'TABLE SPARE ')+pin_name,tuple(np.rint(pixel).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,30,30) if pin_name==env.pin_name else (255,255,0),1,cv2.LINE_AA)
        current_tool_pin=np.linalg.inv(tool('left'))@part(env.pin_name)
        pin_attachment_max_error=float(np.max(np.abs(current_tool_pin-pin_grasp)))
        if pin_attachment_max_error>1e-10:raise ValueError('Physical-reset tool-to-pin attachment changed')
        pin_bounds=np.asarray(json.loads((TRAIN/'sim/actuator/manifest.json').read_text())['parts'][env.pin_name]['bounds'])
        corners=np.array([[x,y,z] for x in pin_bounds[:,0] for y in pin_bounds[:,1] for z in pin_bounds[:,2]])
        reset_pin_tool_corners=corners@pin_grasp[:3,:3].T+pin_grasp[:3,3]
        current_pin_tool_corners=corners@current_tool_pin[:3,:3].T+current_tool_pin[:3,3]
        pin_tool_extent_audit={'physical_reset_min_m':reset_pin_tool_corners.min(0).tolist(),'physical_reset_max_m':reset_pin_tool_corners.max(0).tolist(),
            'replay_min_m':current_pin_tool_corners.min(0).tolist(),'replay_max_m':current_pin_tool_corners.max(0).tolist(),
            'max_extent_disagreement_m':float(np.max(np.abs(reset_pin_tool_corners-current_pin_tool_corners))),
            'scope':'Supplied CAD bounding corners transformed through the actual physical-reset tool-to-pin matrix; no fitted or shifted pin offset'}
        save('registered-pin-identities-diagnostic.png',diagnostic)
        save('real-nominal-registered-threeway.png',np.hstack([panel(real,'REAL TRAIN: representative approach'),panel(complete['right_wrist'],'Nominal photo camera; complete assembly'),panel(registered,'Own-right registration; fixed opening-scale prior')]))
        geom=model.geom('large_carrier_visual').id;mesh_id=int(model.geom_dataid[geom])
        vertices=model.mesh_vert[model.mesh_vertadr[mesh_id]:model.mesh_vertadr[mesh_id]+model.mesh_vertnum[mesh_id]]
        world_vertices=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        y_axis=tool('right')[:3,1];carrier_width=float(np.ptp(world_vertices@y_axis))
        registered_pad_gap=2*registered_opening-.001
        nominal_pad_gap=2*nominal_opening-.001
        save('real-versus-registered-complete-assembly.png',np.hstack([panel(real,'REAL TRAIN: representative approach'),panel(registered,'Own-finger image registration + complete CAD assembly')]))
        registered_camera_report={'conditional_registration_only':True,'source_rejected_as_final':bool(registration.get('rejected_as_final',False)),'path':str(args.camera_registration.resolve()),'sha256':hashlib.sha256(args.camera_registration.read_bytes()).hexdigest(),'scope':'Own-finger silhouette camera registration hypothesis with explicit per-finger opening scale/offset nuisance; mesh unchanged',
            'pin_identities':part_audit,'held_pin_tool_attachment_max_error':pin_attachment_max_error,
            'held_pin_tool_extent_audit':pin_tool_extent_audit,'held_pin_tool_transform_used':current_tool_pin.tolist(),
            'nominal_pad_inner_gap_m':nominal_pad_gap,'registered_pad_inner_gap_m':registered_pad_gap,
            'carrier_mesh_extent_along_jaw_axis_m':carrier_width,'registered_gap_exceeds_carrier_extent':bool(registered_pad_gap>carrier_width),
            'nominal_right_finger_opening_m':nominal_opening,'registered_right_finger_opening_m':registered_opening,
            'right_gripper_opening_scale':float(registration.get('right_gripper_opening_scale',1)),
            'right_gripper_opening_offset_m':float(registration.get('right_gripper_opening_offset_m',0))}
    table_camera_report=None
    if args.table_camera:
        camera=json.loads(args.table_camera.read_text());cam=model.camera('right_wrist').id
        world=tool('right')@np.asarray(camera['T_right_tool_camera_opencv'])
        world[:3,:3]=world[:3,:3]@np.diag([1,-1,-1])
        parent=int(model.cam_bodyid[cam]);parent_pose=rigid(data.xpos[parent],data.xmat[parent].reshape(3,3))
        local=np.linalg.inv(parent_pose)@world;quat=np.empty(4);mujoco.mju_mat2Quat(quat,local[:3,:3].ravel())
        model.cam_pos[cam]=local[:3,3];model.cam_quat[cam]=quat
        k=np.asarray(camera['K']);model.cam_fovy[cam]=np.rad2deg(2*np.arctan(480/(2*k[1,1])))
        mujoco.mj_forward(model,data);table_image=render('right_wrist')
        save('native-q-replay-table-camera-ideal.png',table_image)
        save('nominal-versus-table-camera.png',np.hstack([panel(complete['right_wrist'],'Nominal photo camera; complete supplied CAD assembly'),panel(table_image,'Table-fit camera pose + fy; ideal optics (distortion not applied)')]))
        table_camera_report={'path':str(args.table_camera.resolve()),'sha256':hashlib.sha256(args.table_camera.read_bytes()).hexdigest(),'distortion_applied':False,'scope':'Pose and fy comparison only; principal-point,fx and distortion omitted in this diagnostic ideal render'}
    report={'carrier_visual_profile':getattr(env,'photo_carrier_visual_profile',None),'carrier_visual_selection':args.carrier_visual,'visual_partition':getattr(env,'photo_visual_partition',None),'mujoco_version':mujoco.__version__,'registered_camera_comparison':registered_camera_report,'table_camera_comparison':table_camera_report,'environment':'sim.pin_insertion_env.BimanualPinInsertionEnv','setup_profile':'photo_20261027',
      'setup_profile_path':str(TRAIN/'config/real_setup_20261027.json'),'setup_profile_sha256':hashlib.sha256((TRAIN/'config/real_setup_20261027.json').read_bytes()).hexdigest(),
      'scene_layout':'side_by_side','camera':'nominal existing photo profile; no table-camera/refit overrides',
      'render_only_complete_assembly':{'parts':assembled_names,'hidden_render_only_fixture_helpers':hidden_fixture_helpers,'offsets_relative_large_carrier_m':assembled_offsets,'orientation':'Shared CAD assembly orientation; canonical assembled origins only','central_unit_source':cad['small_carrier']['source_name'],'physics_validated':False,'max_canonical_assembly_position_error_m':assembly_position_error,'manifest_sha256':hashlib.sha256((TRAIN/'sim/actuator/manifest.json').read_bytes()).hexdigest()},
      'reset_seed':args.seed,'reset_parameters':env.reset_parameters,'reset_info':info,
      'representative_real':{k:representative[k] for k in ('id','episode_id','split','video_frame','image_path','state14')},
      'matched_replay_max_arm_q_error_rad':max_q_error,'pin_name':env.pin_name,
      'base_registration':{'path':str(args.base_registration.resolve()),'sha256':hashlib.sha256(args.base_registration.read_bytes()).hexdigest()} if args.base_registration else None,
      'carrier_grasp_registration':{'path':str(args.carrier_grasp.resolve()),'sha256':hashlib.sha256(args.carrier_grasp.read_bytes()).hexdigest(),'scope':'TRAIN-only canonical bore PnP constant grasp hypothesis'} if args.carrier_grasp else None,
      'pin_tool_grasp_from_existing_physical_reset':pin_grasp.tolist(),'carrier_tool_grasp_from_existing_physical_reset':reset_carrier_grasp.tolist(),'carrier_tool_grasp_used':carrier_grasp.tolist(),
      'base_transforms':{s:rigid(data.xpos[model.body(s+'_base').id],data.xmat[model.body(s+'_base').id].reshape(3,3)).tolist() for s in ('left','right')},
      'nominal_cameras':nominal_cameras,
      'limitations':['Existing photo profile is approximate visual reconstruction, not physical camera calibration.',
        'Physical reset is a different pose from real reference; matched native-q replay transfers existing reset grasps as explicit hypotheses.',
        'No camera/base/grasp parameters fit from this representative image or test labels.',
        'Arm native radians preserved exactly; gripper radians-to-jaw metres endpoint map remains approximate.',
        'Complete assembly render attaches CAD parts kinematically; all four gears + supplied small_carrier unit, not a physically validated assembly state.',
        'Matched replay has no contact physics rollout; source reset was generated using existing contact-driven task.']}
    def serialize(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    (args.output_dir/'provenance.json').write_text(json.dumps(report,indent=2,default=serialize)+'\n')
    if args.sequence_episodes:
        if args.sequence_output is None or args.camera_registration is None or args.carrier_grasp is None or args.base_registration is None:
            raise ValueError('Sequence requires output and explicit camera/carrier/base profiles')
        from scripts.evaluate_pin_insertion_cnn import eligible_samples
        import imageio_ffmpeg,subprocess
        test_ids={r['episode_id'] for r in manifest['episodes'] if r['split']=='test'}
        if not set(args.sequence_episodes)<=test_ids:raise ValueError('Only frozen heldout test episodes allowed in comparison')
        if set(args.sequence_episodes)&{r['episode_id'] for r in manifest['episodes'] if r['split'] in ('train','validation')}:raise ValueError('Sequence split overlap')
        args.sequence_output.parent.mkdir(parents=True,exist_ok=True);raw=args.sequence_output.with_suffix('.raw.mp4')
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),args.sequence_fps,(1280,516))
        if not writer.isOpened():raise ValueError('Comparison writer failed')
        sequence=[];max_sequence_q_error=0.;max_pin_attachment_error=0.
        try:
            for eid in args.sequence_episodes:
                ep=ROOT/'data/pin-alignment-forward'/eid;metadata,samples,_=eligible_samples(ep,stride=1,max_skew_ms=100)
                cap=cv2.VideoCapture(str(ep/metadata['videos']['right_wrist']))
                try:
                    for row,skew in samples:
                        source_q=np.asarray(row['state'])
                        for side_index,side in enumerate(('left','right')):
                            arm=env.sim.arms[side];data.qpos[arm.arm_qadr]=source_q[side_index*7:side_index*7+6]
                            opening=source_q[side_index*7+6]/2*.04
                            if side=='right':opening=opening*float(registration.get('right_gripper_opening_scale',1))+float(registration.get('right_gripper_opening_offset_m',0))
                            data.qpos[arm.finger_qadr]=[opening,-opening]
                        data.qvel[:]=0;mujoco.mj_forward(model,data)
                        qerror=max(float(np.max(np.abs(data.qpos[env.sim.arms[side].arm_qadr]-source_q[i*7:i*7+6]))) for i,side in enumerate(('left','right')))
                        max_sequence_q_error=max(max_sequence_q_error,qerror)
                        if qerror>1e-12:raise ValueError('Native sequence arm angles changed')
                        pin_pose=tool('left')@pin_grasp;carrier_pose=tool('right')@carrier_grasp
                        _place(env.sim,env.pin_name,pin_pose[:3,3],pin_pose[:3,:3]);_place(env.sim,'large_carrier',carrier_pose[:3,3],carrier_pose[:3,:3])
                        for name in assembled_names:
                            offset=np.asarray(assembled_offsets[name]);_place(env.sim,name,carrier_pose[:3,3]+carrier_pose[:3,:3]@offset,carrier_pose[:3,:3])
                        mujoco.mj_forward(model,data)
                        attachment_error=float(np.max(np.abs(np.linalg.inv(tool('left'))@part(env.pin_name)-pin_grasp)))
                        max_pin_attachment_error=max(max_pin_attachment_error,attachment_error)
                        if attachment_error>1e-10:raise ValueError('Pin detached from unchanged physical-reset grasp')
                        sim_image=render('right_wrist')
                        cap.set(cv2.CAP_PROP_POS_FRAMES,row['video_frame']);ok,bgr=cap.read()
                        if not ok:raise ValueError('Cannot decode real source frame')
                        canvas=np.zeros((516,1280,3),dtype=np.uint8);canvas[36:,:640]=bgr;canvas[36:,640:]=cv2.cvtColor(sim_image,cv2.COLOR_RGB2BGR)
                        elapsed=(row['monotonic_ns']-metadata['started_monotonic_ns'])/1e9
                        caption=f'REAL HELDOUT {eid[:10]} frame{row["video_frame"]} t={elapsed:.3f}s'
                        cv2.putText(canvas,caption,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                        cv2.putText(canvas,('SIM: NEW v4 visual reproduction; same q; no contact physics' if args.carrier_visual=='reproduced' else 'SIM: same q; pin_2 held, other pins on table; no contact physics'),(648,24),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
                        writer.write(canvas);sequence.append({'episode_id':eid,'video_frame':row['video_frame'],'time_s':elapsed,'native_state14':source_q.tolist(),'camera_sample_skew_ms':skew})
                finally:cap.release()
        finally:writer.release()
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(args.sequence_output)],check=True);raw.unlink()
        sequence_report={'output':str(args.sequence_output.resolve()),'frames':len(sequence),'fps':args.sequence_fps,'duration_s':len(sequence)/args.sequence_fps,
            'carrier_visual_selection':args.carrier_visual,'carrier_visual_mesh_sha256':getattr(env,'photo_carrier_visual_profile',{}).get('mesh_sha256') if getattr(env,'photo_carrier_visual_profile',None) else None,
            'new_visual_reproduction_not_original_mesh':args.carrier_visual=='reproduced','max_native_arm_error_rad':max_sequence_q_error,'max_held_pin_tool_transform_error':max_pin_attachment_error,
            'held_pin_body':env.pin_name,'background_table_spare_pins':['pin_1','pin_3','pin_4'],'setup_provenance':str((args.output_dir/'provenance.json').resolve()),'source_frames':sequence,
            'profile_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('camera',args.camera_registration),('carrier',args.carrier_grasp),('planar_base',args.base_registration)]},
            'limitations':['Kinematic native-angle replay with explicitly attached complete CAD assembly; no physics/contact success claim.',
                           'Same source frames at quarter-speed display by default; calibration/grasp profiles fit TRAIN only and selected by independent validation.',
                           'Pin grasp fixed from original physical reset; carrier grasp is TRAIN bore-PnP hypothesis; own camera and opening are conditional image registration.',
                           'Selected planar base preserves common tabletop Z/upright constraints; optical distortion and photoreal materials remain approximate.']}
        args.sequence_output.with_suffix('.json').write_text(json.dumps(sequence_report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in sequence_report.items() if k!='source_frames'},indent=2))
    renderer.close();env.close()
    print(json.dumps({'output_dir':str(args.output_dir.resolve()),'max_arm_q_error_rad':max_q_error,'representative_real':representative['id']},indent=2))

if __name__=='__main__':main()
