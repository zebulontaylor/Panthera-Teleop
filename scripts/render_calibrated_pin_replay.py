#!/usr/bin/env python3
"""Historical rejected-grasp/legacy-scene replay; use render_pin_insertion_photo_setup.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np


def transform(position,rotation):
    t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=position;return t


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-prototype',action='store_true',help='Reproduce the superseded legacy scene and rejected grasp hypothesis')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--camera',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int,default=150)
    p.add_argument('--pin-axis-tool',type=float,nargs=3,help='Explicit CAD +Z axis in left tool; default negative shared tip offset normalized (grip-site-as-center prior)')
    p.add_argument('--episodes',nargs='*');p.add_argument('--fps',type=float,default=7.5)
    args=p.parse_args()
    if not args.legacy_prototype:
        p.error('Superseded scene: use render_pin_insertion_photo_setup.py. --legacy-prototype is only for historical reproduction.')
    import cv2,mujoco,torch,imageio_ffmpeg
    from real2sim_bridge import Real2SimBridge,DEFAULT_SCENE,ARM_COLUMNS
    from pin_insertion_sim import _place
    from app.pin_insertion_cnn import PinInsertionPredictor
    from scripts.render_pin_insertion_heatmaps import infer,compose_frame,assert_render_disjoint
    from scripts.evaluate_pin_insertion_cnn import eligible_samples
    torch.set_num_threads(4)
    fit=json.loads(args.calibration.read_text());cam=json.loads(args.camera.read_text());manifest=json.loads(args.manifest.read_text())
    if not np.allclose(fit['T_right_tool_camera_opencv'],cam['T_right_tool_camera_opencv'],atol=1e-12):raise ValueError('Refit did not fix table camera')
    test={r['episode_id'] for r in manifest['episodes'] if r['split']=='test'}
    # Default to test episodes with independently visible pin labels; gear-only
    # negatives would require a different workpiece model, never invent a pin there.
    episodes=args.episodes or sorted({r['episode_id'] for r in manifest['frames'] if r['split']=='test' and r.get('pin_tip_px') is not None})
    if not set(episodes)<=test:raise ValueError('Only explicitly reserved test episodes allowed')
    predictor=PinInsertionPredictor(args.checkpoint);assert_render_disjoint(episodes,predictor.metadata)
    if set(episodes)&set(fit['fit_settings']['training_episodes']):raise ValueError('Replay overlaps calibration fit episodes')
    candidates=[(c,h) for h in fit['carrier_pose_hypotheses'] if h['split']=='train' for c in h['candidates']]
    if not candidates:raise ValueError('No train-only carrier hypothesis available')
    carrier,carrier_source=min(candidates,key=lambda pair:pair[0]['reprojection_rms_px'])
    tool_carrier=np.asarray(carrier['T_right_tool_carrier']);tool_camera=np.asarray(cam['T_right_tool_camera_opencv'])
    bridge=Real2SimBridge();model,data=bridge.model,bridge.data
    right_world=np.linalg.inv(np.asarray(fit['T_right_base_left_base']))
    rb=model.body('right_base').id;model.body_pos[rb]=right_world[:3,3]
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,right_world[:3,:3].ravel());model.body_quat[rb]=quat
    for name in ('small_carrier','pin_2','pin_3','pin_4','gear_1','gear_2','gear_3','gear_4'):_place(bridge,name,[3,3,3],np.eye(3))
    model.geom_pos[model.geom('table_top').id,2]-=.05
    mujoco.mj_setConst(model,data)
    width,height=320,240;k=np.asarray(cam['K'])*.5;k[2,2]=1
    render_width,render_height=width*2,height*2
    ideal=np.array([[k[1,1],0,render_width/2],[0,k[1,1],render_height/2],[0,0,1.]])
    pixels=np.stack(np.meshgrid(np.arange(width),np.arange(height)),axis=-1).astype(np.float32)
    remap=cv2.undistortPoints(pixels.reshape(-1,1,2),k,np.asarray(cam['distortion']),P=ideal).reshape(height,width,2)
    cid=model.camera('right_wrist').id;model.cam_fovy[cid]=np.rad2deg(2*np.arctan(render_height/(2*k[1,1])))
    renderer=mujoco.Renderer(model,height=render_height,width=render_width)
    option=mujoco.MjvOption();option.sitegroup[:]=0;option.geomgroup[:]=[1,1,1,0,0,0]
    args.output.parent.mkdir(parents=True,exist_ok=True);raw=args.output.with_suffix('.raw.mp4')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(1440,506))
    if not writer.isOpened():raise RuntimeError('MP4 writer failed')
    pose_labels={(r['episode_id'],r['video_frame']):r for r in manifest['frames']}
    frames=[];fk_errors=[];q_errors=[];pin_offset=np.asarray(fit['pin_tip_left_tool_m'])
    axis_tool=np.asarray(args.pin_axis_tool,dtype=float) if args.pin_axis_tool else -pin_offset.copy()
    if not np.isfinite(axis_tool).all() or np.linalg.norm(axis_tool)<1e-8:raise ValueError('Nonzero finite pin-axis prior required')
    axis_tool/=np.linalg.norm(axis_tool)
    try:
        for episode in episodes:
            ep=ROOT/'data/pin-alignment-forward'/episode
            _,samples,_=eligible_samples(ep,stride=1,max_skew_ms=100)
            for row,skew in samples:
                q=np.asarray(row['state']);poses=bridge.set_kinematic_pose(q)
                qerror=float(np.max(np.abs(bridge.native_state()[ARM_COLUMNS]-q[ARM_COLUMNS])));q_errors.append(qerror)
                if qerror>1e-12:raise ValueError('Raw arm joints changed')
                label=pose_labels.get((episode,row['video_frame']))
                if label:
                    for side in ('left','right'):
                        expected=np.asarray(label['T_'+side+'_base_tool'])
                        error=float(np.linalg.norm(poses[side]['position']-expected[:3,3]));fk_errors.append(error)
                        if error>1e-5 or not np.allclose(poses[side]['rotation'],expected[:3,:3],atol=1e-4):raise ValueError('App vs MuJoCo native FK disagreement')
                left=transform(data.site_xpos[bridge.sites[0]].copy(),data.site_xmat[bridge.sites[0]].reshape(3,3).copy())
                right=transform(data.site_xpos[bridge.sites[1]].copy(),data.site_xmat[bridge.sites[1]].reshape(3,3).copy())
                camera_world=right@tool_camera;camera_world[:3,:3]=camera_world[:3,:3]@np.diag([1,-1,-1])
                parent=int(model.cam_bodyid[cid]);parent_world=transform(data.xpos[parent],data.xmat[parent].reshape(3,3))
                camera_local=np.linalg.inv(parent_world)@camera_world
                model.cam_pos[cid]=camera_local[:3,3];mujoco.mju_mat2Quat(quat,camera_local[:3,:3].ravel());model.cam_quat[cid]=quat
                carrier_world=right@tool_carrier;_place(bridge,'large_carrier',carrier_world[:3,3],carrier_world[:3,:3])
                tip=left[:3,:3]@pin_offset+left[:3,3]
                z=left[:3,:3]@axis_tool
                reference=left[:3,1] if abs(z@left[:3,1])<.9 else left[:3,0]
                x=reference-z*(reference@z);x/=np.linalg.norm(x)
                y=np.cross(z,x);pin_rotation=np.column_stack((x,y,z))
                _place(bridge,'pin_1',tip,pin_rotation);mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera='right_wrist',scene_option=option);rgb=renderer.render().copy()
                rgb=cv2.remap(rgb,remap[:,:,0],remap[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                out=infer(predictor,rgb,q)
                subtitle=f'{episode[:12]} frame {row["video_frame"]} | shared TRAIN grasp; pin-axis prior; ambiguous TRAIN carrier PnP'
                writer.write(compose_frame(rgb,out,'SIM | calibrated-hypothesis kinematic replay; no contact physics',subtitle,xyz_disabled=True))
                frames.append({'episode_id':episode,'video_frame':row['video_frame'],'raw_state14':q.tolist()})
                if len(frames)>=args.max_frames:break
            if len(frames)>=args.max_frames:break
    finally:writer.release();renderer.close()
    if not frames:raise ValueError('No frames')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],check=True);raw.unlink()
    provenance={'output':str(args.output.resolve()),'frames':len(frames),'fps':args.fps,'duration_s':len(frames)/args.fps,
       'camera_sha256':sha(args.camera),'calibration_sha256':sha(args.calibration),'manifest_sha256':sha(args.manifest),'checkpoint_sha256':sha(args.checkpoint),
       'episodes':episodes,'max_raw_arm_q_error_rad':max(q_errors),'app_fk_checks':len(fk_errors),'max_app_fk_position_error_m':max(fk_errors,default=None),
       'carrier_training_observation':carrier_source['observation_id'],'carrier_reprojection_rms_px':carrier['reprojection_rms_px'],
       'pin_axis_left_tool_prior':axis_tool.tolist(),'pin_axis_fitted':False,
       'shared_train_mean_pin_tip_left_tool_m':pin_offset.tolist(),'real_calibrated':False,'contact_physics':False,
       'priors':['Fixed-camera fit from independent table texture hypothesis; inter-base and pin offset remain prior-conditioned.',
                 'Pin CAD +Z is configurable prior, default negative shared tip offset normalized assuming grip-site at cylinder center. Endpoint fit does not fit axis; orientation is uncertain.',
                 'Single best TRAIN carrier PnP used across heldout grasps; planar normal and 90-degree yaw ambiguous.',
                 'No test labels set object or camera poses. Native arm joints unchanged; gripper endpoint mapping approximate.',
                 'Table lowered 50mm as existing scene prior; gears parked outside view, not reconstructed.',
                 'Kinematic replay only; not insertion rollout or successful-contact validation. Quarter-speed display of source samples.'],
       'source_frames':frames}
    args.output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({k:v for k,v in provenance.items() if k!='source_frames'},indent=2))

if __name__=='__main__':main()
