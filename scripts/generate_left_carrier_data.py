#!/usr/bin/env python3
"""Preview measured native-q approach/insertion replay before any training.

No camera aiming, pose randomization, physics rollout, or hardware access.
"""
import os,sys,json,argparse,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np

def preview(output,manifest):
 import cv2,mujoco
 from PIL import Image,ImageDraw
 import pin_insertion_environment as pe
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 # Preserve the corrected visual carrier while reusing shared measured-state FK.
 original=pe.create_photo_environment
 def corrected(**kw):return original(**dict(kw,carrier_visual_profile=True))
 pe.create_photo_environment=corrected
 try:replay=pe.NativeQPhotoReconstruction()
 finally:pe.create_photo_environment=original
 e=replay.environment;m,d=e.model,e.data;renderer=mujoco.Renderer(m,height=480,width=640)
 opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0];opt.sitegroup[:]=0
 entries=json.loads(Path(manifest).read_text())['frames'];selected=[]
 for episode in dict.fromkeys(x['episode_id'] for x in entries):
  rows=[x for x in entries if x['episode_id']==episode]
  for row in [rows[0],rows[-1]]:
   if row not in selected:selected.append(row)
  if len(selected)>=12:break
 panels=[];audit=[]
 for row in selected:
  state=np.asarray(row['state14']);desc=replay.set_native_pose(state)
  renderer.update_scene(d,camera='left_wrist',scene_option=opt);sim=renderer.render().copy()
  video=Path(row['source_episode_path'])/'left_wrist.avi';cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,row['video_frame']);ok,frame=cap.read();cap.release()
  if not ok:raise RuntimeError(f'No left frame {video}:{row["video_frame"]}')
  real=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  sid=row['id'];Image.fromarray(real).save(out/f'{sid}-real.png');Image.fromarray(sim).save(out/f'{sid}-sim.png')
  tile=Image.new('RGB',(640,265));draw=ImageDraw.Draw(tile);draw.text((5,4),f'{row["episode_id"][:12]} frame {row["video_frame"]}: REAL LEFT / SIM LEFT same measured q',fill='white');tile.paste(Image.fromarray(real).resize((320,240)),(0,25));tile.paste(Image.fromarray(sim).resize((320,240)),(320,25));panels.append(tile)
  audit.append(dict(id=sid,episode_id=row['episode_id'],source_split=row['split'],video_frame=row['video_frame'],state14=state.tolist(),source_left_video=str(video),reconstruction=desc))
 sheet=Image.new('RGB',(1280,265*((len(panels)+1)//2)))
 for i,panel in enumerate(panels):sheet.paste(panel,((i%2)*640,(i//2)*265))
 sheet.save(out/'measured-demo-contact-sheet.jpg');renderer.close();e.close()
 report=dict(status='PREVIEW ONLY; no accepted training checkpoint',camera='left_wrist',native_shape=[640,480],camera_perturbation=False,pose_perturbation=False,physical_rollout=False,real_calibrated=False,carrier_grasp='existing train-only right-tool/carrier PnP hypothesis',pin_grasp='actual nominal-reset left-tool/pin prior',left_camera='unchanged existing photo setup, not real calibrated',records=audit)
 (out/'measured-demo-preview.json').write_text(json.dumps(report,indent=2));print(out/'measured-demo-contact-sheet.jpg',flush=True)

def generate_measured(output,size=224):
 import mujoco
 from PIL import Image,ImageDraw
 import pin_insertion_environment as pe
 from generate_pin_insertion_photo_data import project
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 anchors=[];manifest=json.loads((ROOT/'work/pin-insertion-real/manifest.json').read_text())
 for episode in manifest['episodes']:
  path=ROOT/'data/pin-alignment-forward'/episode['episode_id'];rows=[json.loads(s) for s in (path/'samples.jsonl').read_text().splitlines() if s.strip()]
  # Actual full trajectory anchors, not sparse landmark-annotated frames.
  for row in rows[::max(1,len(rows)//20)]:
   anchors.append(dict(state14=row['state'],group=episode['episode_id'],split=episode['split'],source=str(path),frame=row['video_frame'],kind='independent_approach_recording'))
 split=json.loads((ROOT/'work/pin-insertion-newdemo/split.json').read_text());path=Path(split['source_path']);frames=[json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines() if s.strip()];frames=[f for f in frames if f['camera']=='wrist_port3'];states=[json.loads(s) for s in (path/'states.jsonl').read_text().splitlines() if s.strip()];times=np.array([s['monotonic_ns'] for s in states])
 for block in split['blocks']:
  chosen=[f for f in frames if block['start_s']<=f['episode_time_s']<=block['end_s']]
  for f in chosen[::5]:
   j=int(np.argmin(np.abs(times-f['monotonic_ns'])));state=states[j];q=[]
   for side in ('left','right'):q.extend(state['arms'][side]['position_rad']);q.append(state['arms'][side]['gripper']['position'])
   anchors.append(dict(state14=q,group=split['source_recording_id']+'::'+block['block_id'],split=block['split'],source=str(path),frame=f['frame_index'],kind='newdemo_pose_block_same_grasp',skew_ms=float((times[j]-f['monotonic_ns'])/1e6)))
 original=pe.create_photo_environment
 def corrected(**kw):return original(**dict(kw,carrier_visual_profile=True))
 pe.create_photo_environment=corrected
 try:replay=pe.NativeQPhotoReconstruction()
 finally:pe.create_photo_environment=original
 e=replay.environment;m,d=e.model,e.data;assert e.sim.layout=='side_by_side';width,height=320,240;renderer=mujoco.Renderer(m,height=height,width=width);opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0];opt.sitegroup[:]=0;cam=m.camera('left_wrist').id;optics=(m.cam_pos[cam].copy(),m.cam_quat[cam].copy(),float(m.cam_fovy[cam]));pin_geoms=set(np.flatnonzero(m.geom_bodyid==e.pin_id).tolist());rgba=m.geom_rgba.copy();diff=m.light_diffuse.copy();amb=m.light_ambient.copy()
 local=np.array([[0,0,.01725]]+[[r*np.cos(a),r*np.sin(a),.01725] for r in (.0179625,.02649) for a in np.arange(4)*np.pi/2]);records=[];audit=[]
 for anchor_index,anchor in enumerate(anchors):
  for variant in range(4 if anchor['split']=='train' else 1):
   rng=np.random.default_rng(43000+anchor_index*10+variant);q=np.array(anchor['state14']);jitter=variant==3
   if jitter:q[np.r_[0:6,7:13]]+=rng.uniform(-.003,.003,12)
   replay.set_native_pose(q);m.geom_rgba[:]=rgba;m.light_diffuse[:]=diff;m.light_ambient[:]=amb
   if variant:
    m.geom_rgba[:,:3]*=rng.uniform(.85,1.1,(m.ngeom,1));m.geom_rgba[:,:3]=m.geom_rgba[:,:3].clip(0,1);m.light_diffuse[:]=diff*rng.uniform(.7,1.3,(m.nlight,3));m.light_ambient[:]=amb*rng.uniform(.7,1.3,(m.nlight,3))
   mujoco.mj_forward(m,d);world=local@d.xmat[e.carrier_id].reshape(3,3).T+d.xpos[e.carrier_id];world=np.vstack((world,d.xpos[e.pin_id]));uv,dep=project(world,d.cam_xpos[cam],d.cam_xmat[cam].reshape(3,3),m.cam_fovy[cam],width,height)
   renderer.update_scene(d,camera='left_wrist',scene_option=opt);rgb=renderer.render().copy();renderer.enable_depth_rendering();depth=renderer.render().copy();renderer.disable_depth_rendering();renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering()
   visible=(dep>0)&np.isfinite(uv).all(1)&(uv>.005).all(1)&(uv<.995).all(1)
   for k in np.flatnonzero(visible):
    xx,yy=(uv[k]*[width,height]).astype(int);visible[k]=(depth[yy,xx]>=dep[k]-.001) or (k==9 and int(seg[yy,xx,0]) in pin_geoms and int(seg[yy,xx,1])==5)
   visible[:9]&=np.dot(d.xmat[e.carrier_id].reshape(3,3)[:,2],d.cam_xpos[cam]-world[0])>0
   records.append(dict(images=np.asarray(Image.fromarray(rgb).resize((size,size),Image.Resampling.BILINEAR)),keypoints=np.nan_to_num(uv).astype(np.float32),visible=visible,groups=anchor['group'],splits=anchor['split'],seeds=43000+anchor_index*10+variant,anchor_index=anchor_index,variant=variant));audit.append(dict(anchor_index=anchor_index,variant=variant,joint_jitter_rad=.003 if jitter else 0,**anchor))
  if anchor_index%100==0:print('measured anchor',anchor_index,'/',len(anchors),'frames',len(records),flush=True)
 assert np.array_equal(m.cam_pos[cam],optics[0]) and np.array_equal(m.cam_quat[cam],optics[1]) and float(m.cam_fovy[cam])==optics[2]
 renderer.close();e.close();arrays={k:np.asarray([r[k] for r in records]) for k in records[0]};np.savez_compressed(out/'dataset.npz',**arrays);(out/'source-poses.json').write_text(json.dumps(audit))
 metadata=dict(count=len(records),anchors=len(anchors),size=size,camera='left_wrist',layout='side_by_side',native_render_size=[320,240],runtime_native_size=[640,480],camera_perturbation=False,split_unit='independent approach recording; newdemo stationary pose blocks B1/B3 train B2 validation B4/B5 test',augmentation='train only:4 variants,3 exact native anchors,1 <=.003rad jitter;3 variants mild material/light',physical_rollout=False,real_calibrated=False,grasp_hypothesis='fixed tool-carrier TRAIN PnP; fixed left-pin nominal-reset prior',real_images_used_for_training=False,hidden_landmark_location_loss=False)
 (out/'dataset.json').write_text(json.dumps(metadata,indent=2));sheet=Image.new('RGB',(size*6,size*3))
 for j,idx in enumerate(np.linspace(0,len(records)-1,18).astype(int)):
  im=Image.fromarray(arrays['images'][idx]);draw=ImageDraw.Draw(im)
  for k,p in enumerate(arrays['keypoints'][idx]*size):
   if arrays['visible'][idx,k]:draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),outline='lime');draw.text(tuple(p),str(k),fill='red')
  sheet.paste(im,((j%6)*size,(j//6)*size))
 sheet.save(out/'samples.jpg');print(metadata,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='work/left-carrier-cnn');p.add_argument('--manifest',default='work/pin-insertion-real/manifest.json');p.add_argument('--generate-dataset',action='store_true');p.add_argument('--size',type=int,default=224);a=p.parse_args()
 if a.generate_dataset:generate_measured(a.output,a.size)
 else:preview(str(Path(a.output)/'measured-demo-preview'),a.manifest)
