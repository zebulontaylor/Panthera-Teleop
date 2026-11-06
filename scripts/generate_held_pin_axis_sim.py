#!/usr/bin/env python3
"""Synthetic varied-grasp hypotheses with unchanged native arms/camera.
These poses are not contact-certified. No hardware or rollout.
"""
import os,sys,json,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import numpy as np

def run(output):
 import mujoco
 from PIL import Image,ImageDraw
 import pin_insertion_environment as pe
 from generate_pin_insertion_photo_data import project
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 original=pe.create_photo_environment;pe.create_photo_environment=lambda **kw:original(**dict(kw,carrier_visual_profile=True))
 try:r=pe.NativeQPhotoReconstruction()
 finally:pe.create_photo_environment=original
 e=r.environment;m,d=e.model,e.data;renderer=mujoco.Renderer(m,height=480,width=640);opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0];opt.sitegroup[:]=0;c=m.camera('left_wrist').id
 rows=[a for a in json.load(open(ROOT/'work/left-carrier-cnn/source-poses.json')) if a['variant']==0];row=next(a for a in rows if 'B3' in a['group'] and 605<=a['frame']<=615)
 panels=[];audit=[]
 for i,(ty,dep,tilt) in enumerate([(None,None,None),(310,.145,5),(310,.145,20),(310,.145,35),(310,.18,35),(310,.115,35)]):
  r.set_native_pose(row['state14']);cr=d.cam_xmat[c].reshape(3,3).copy();cp=d.cam_xpos[c].copy();f=240/np.tan(np.deg2rad(m.cam_fovy[c]/2));prior=d.xpos[e.pin_id].copy()
  if ty is not None:
   top=np.array([(335-320)*dep/f,-(ty-240)*dep/f,-dep]);a=np.deg2rad(tilt);z=np.array([0,np.sin(a),np.cos(a)]);x=np.array([1.,0,0]);y=np.cross(z,x);rot=cr@np.stack((x,y,z),axis=1);bottom=cp+cr@(top-z*.01788);q=np.empty(4);mujoco.mju_mat2Quat(q,rot.ravel());e.sim.set_part_pose('pin_2',bottom,q);mujoco.mj_forward(m,d)
  world=np.array([[0,0,.01788],[0,0,0]])@d.xmat[e.pin_id].reshape(3,3).T+d.xpos[e.pin_id];uv,zdep=project(world,cp,cr,m.cam_fovy[c],640,480);renderer.update_scene(d,camera='left_wrist',scene_option=opt);rgb=renderer.render().copy();im=Image.fromarray(rgb);im.save(out/f'preview-{i}-native.png');dr=ImageDraw.Draw(im)
  for k,col in [(0,'lime'),(1,'red')]:
   x0,y0=uv[k]*[640,480];dr.ellipse((x0-3,y0-3,x0+3,y0+3),outline=col,width=1)
  tile=Image.new('RGB',(640,510));tile.paste(im,(0,30));ImageDraw.Draw(tile).text((5,5),'NOMINAL prior' if ty is None else f'cap y={ty} depth={dep} tilt={tilt}deg',fill='white');panels.append(tile);audit.append(dict(case=i,top_y_target=ty,depth_m=dep,tilt_deg=tilt,top_uv=uv[0].tolist(),bottom_uv=uv[1].tolist(),pin_bottom_delta_from_prior_m=(d.xpos[e.pin_id]-prior).tolist(),render=str((out/f'preview-{i}-native.png').resolve())))
 sheet=Image.new('RGB',(1920,1020))
 for i,p in enumerate(panels):sheet.paste(p,(i%3*640,i//3*510))
 sheet.save(out/'flex-preview.jpg');renderer.close();r.close();(out/'preview-audit.json').write_text(json.dumps(dict(status='PREVIEW ONLY; fit not accepted',fixed_camera=True,fixed_actual_arm_state=row,cap_landmark='proximal cap face center',pin_length_m=.01788,top_target_px=[335,310],cases=audit),indent=2)+'\n')
 print(out/'flex-preview.jpg',flush=True)

def generate(output,count=None,variants_per_anchor=12):
 import mujoco,cv2,hashlib,time
 from PIL import Image,ImageDraw
 import pin_insertion_environment as pe
 from generate_pin_insertion_photo_data import project
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 original=pe.create_photo_environment;pe.create_photo_environment=lambda **kw:original(**dict(kw,carrier_visual_profile=True))
 try:r=pe.NativeQPhotoReconstruction()
 finally:pe.create_photo_environment=original
 e=r.environment;m,d=e.model,e.data;renderer=mujoco.Renderer(m,height=240,width=320);opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0];opt.sitegroup[:]=0;c=m.camera('left_wrist').id
 source=ROOT/'work/left-carrier-cnn/source-poses.json';rows=[a for a in json.load(open(source)) if a['variant']==0];gear_manifest=json.loads((ROOT/'work/pin-insertion-real/manifest.json').read_text());gear_groups={f['episode_id'] for f in gear_manifest['frames'] if 'gear-like' in f.get('note','').lower()};rows=[a for a in rows if a['state14'][6]<=.25 and a['group'] not in gear_groups];rows=rows[:count] if count else rows
 if variants_per_anchor < 1:raise ValueError('variants_per_anchor must be positive')
 if not rows:raise ValueError('No eligible anchors')
 group_splits={}
 for row in rows:
  if row['group'] in group_splits and group_splits[row['group']]!=row['split']:raise ValueError('Source group crosses splits')
  group_splits[row['group']]=row['split']
 opt0=(m.cam_pos[c].copy(),m.cam_quat[c].copy(),m.cam_fovy[c]);records=[];audit=[];local=np.array([[0,0,.01788],[0,0,0]]);pin_geoms=set(np.flatnonzero(m.geom_bodyid==e.pin_id).tolist());colors=m.geom_rgba.copy();diff=m.light_diffuse.copy();amb=m.light_ambient.copy();light_dir=m.light_dir.copy();light_directional=m.light_type.copy();light_shadow=m.light_castshadow.copy();head_ambient=m.vis.headlight.ambient.copy();head_diffuse=m.vis.headlight.diffuse.copy();head_specular=m.vis.headlight.specular.copy();sheet=Image.new('RGB',(1280,1020));start=time.time()
 samples=[(row,v) for v in range(variants_per_anchor) for row in rows]
 for i,(row,variant) in enumerate(samples):
  seed=int.from_bytes(hashlib.sha256(f"held-pin-varied-v2:{row['anchor_index']}:{variant}".encode()).digest()[:8],'little');rng=np.random.default_rng(seed);r.set_native_pose(row['state14']);cr=d.cam_xmat[c].reshape(3,3).copy();cp=d.cam_xpos[c].copy();rot0=d.xmat[e.pin_id].reshape(3,3).copy();bottom0=d.xpos[e.pin_id].copy();center=bottom0+rot0[:,2]*.00894
  # Synthetic grasp hypotheses with native arms/camera and zero optical X shift.
  translation_cam=np.array([0,rng.uniform(-.005,.005),rng.uniform(-.005,.005)]);axial_depth=float(rng.uniform(-.005,.005));translation_cam[2]+=axial_depth;angle=np.deg2rad(rng.uniform(-25,25));rr=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]]);rot=cr@rr@cr.T@rot0;bottom=center+cr@translation_cam-rot[:,2]*.00894;q=np.empty(4);mujoco.mju_mat2Quat(q,rot.ravel());e.sim.set_part_pose('pin_2',bottom,q)
  # Reset lights per sample; independently randomize physical shadows on every split.
  m.geom_rgba[:]=colors;m.light_diffuse[:]=diff;m.light_ambient[:]=amb;m.light_dir[:]=light_dir;m.light_type[:]=light_directional;m.light_castshadow[:]=light_shadow
  lighting_domain=['ordinary','hard_shadow','extreme_shadow'][(variant+int(row['anchor_index']))%3]
  ambient_scale=float(rng.uniform(.65,1.2) if lighting_domain=='ordinary' else rng.uniform(.03,.15) if lighting_domain=='hard_shadow' else rng.uniform(.001,.015))
  diffuse_scale=float(rng.uniform(.7,1.4));azimuth=float(rng.uniform(-np.pi,np.pi));elevation=float(rng.uniform(.12,.65) if lighting_domain=='extreme_shadow' else rng.uniform(.3,1.4))
  direction=np.array([np.cos(azimuth)*np.cos(elevation),np.sin(azimuth)*np.cos(elevation),-np.sin(elevation)])
  m.vis.headlight.ambient[:]=head_ambient*ambient_scale;m.vis.headlight.diffuse[:]=head_diffuse*(1. if lighting_domain=='ordinary' else ambient_scale);m.vis.headlight.specular[:]=head_specular*(1. if lighting_domain=='ordinary' else ambient_scale)
  m.light_ambient[:]=amb*ambient_scale;m.light_diffuse[:]=diff*diffuse_scale
  if m.nlight:
   m.light_type[:]=1;m.light_castshadow[:]=1;m.light_dir[:]=direction
   if lighting_domain!='ordinary' and m.nlight>1:m.light_diffuse[1:]=0
  m.geom_rgba[:,:3]=np.clip(colors[:,:3]*rng.uniform(.87,1.13,(m.ngeom,1)),0,1)
  mujoco.mj_forward(m,d);world=local@rot.T+bottom;uv,dep=project(world,cp,cr,m.cam_fovy[c],320,240);renderer.update_scene(d,camera='left_wrist',scene_option=opt);rgb=renderer.render().copy();renderer.enable_depth_rendering();depth=renderer.render().copy();renderer.disable_depth_rendering();renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering();visible=(dep>0)&np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<1).all(1)
  for k in np.flatnonzero(visible):
   xx,yy=(uv[k]*[320,240]).astype(int);visible[k]=(depth[yy,xx]>=dep[k]-.001) or (int(seg[yy,xx,0]) in pin_geoms and int(seg[yy,xx,1])==5)
  # Identical whole-RGB and endpoint affine. Native registration +15,-51px.
  # Validation/test keep deterministic registration; flex seed remains unique.
  dy=rng.uniform(-15,15) if row['split']=='train' else 0.;scale=rng.uniform(.98,1.02) if row['split']=='train' else 1.;M=np.array([[scale,0,160*(1-scale)+7.5],[0,scale,120*(1-scale)+(-51+dy)*.5]],dtype=np.float32)
  rgb=cv2.warpAffine(rgb,M,(320,240),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);uv2=(np.c_[uv*[320,240],np.ones(2)]@M.T)/[320,240];visible&=(uv2>=0).all(1)&(uv2<1).all(1)
  records.append(dict(images=rgb,top_y=uv2[0,1],bottom_y=uv2[1,1],top_visible=visible[0],bottom_visible=visible[1],top_uv=uv2[0],bottom_uv=uv2[1],groups=row['group'],splits=row['split'],anchor_index=row['anchor_index'],variant=variant,lighting_domain=lighting_domain))
  audit.append(dict(anchor_index=row['anchor_index'],group=row['group'],split=row['split'],variant=variant,seed=seed,axial_depth_camera_z_m=axial_depth,lighting=dict(domain=lighting_domain,ambient_scale=ambient_scale,diffuse_scale=diffuse_scale,direction_world=direction.tolist(),directional=True,cast_shadow=True),pin_midpoint_translation_camera_m=translation_cam.tolist(),pin_rotation_optical_x_deg=float(np.rad2deg(angle)),registration_native_xy_px=[15,-51+dy],registration_scale=scale,registration_affine_320=M.tolist(),top_bottom_gap_native_y_px=float((uv2[1,1]-uv2[0,1])*480)))
  if i<16:
   tile=Image.new('RGB',(320,255));tile.paste(Image.fromarray(rgb),(0,15));dr=ImageDraw.Draw(tile);dr.text((2,1),f'{row["split"]} {lighting_domain} {np.rad2deg(angle):.1f}deg',fill='white')
   for k,color in [(0,'lime'),(1,'red')]:
    px,py=uv2[k]*[320,240];py+=15;dr.ellipse((px-3,py-3,px+3,py+3),outline=color);dr.text((px+3,py),['TOP','BOTTOM'][k],fill=color)
   sheet.paste(tile,(i%4*320,i//4*255))
   if i==min(15,len(samples)-1):sheet.save(out/'registered-flex-preview.jpg');print('PREVIEW',out/'registered-flex-preview.jpg',flush=True)
  if i%100==0:print(i,len(samples),f'{time.time()-start:.1f}s',flush=True)
 assert np.array_equal(m.cam_pos[c],opt0[0]) and np.array_equal(m.cam_quat[c],opt0[1]) and m.cam_fovy[c]==opt0[2]
 assert len({a['seed'] for a in audit})==len(audit),'Duplicate sample seeds'
 renderer.close();r.close();arrays={k:np.asarray([a[k] for a in records]) for k in records[0]};np.savez_compressed(out/'sim_endpoints.npz',**arrays);(out/'source-flex.json').write_text(json.dumps(audit,indent=2)+'\n')
 def stats(a):return dict(mean=float(np.mean(a)),std=float(np.std(a)),min=float(np.min(a)),max=float(np.max(a)))
 meta=dict(schema_version=2,count=len(records),variants_per_anchor=variants_per_anchor,anchor_count=len(rows),lighting_domains=['ordinary','hard_shadow','extreme_shadow'],native_render_size=[320,240],source_anchor_manifest=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),camera='left_wrist',camera_modified=False,arm_angles_modified=False,physical_rollout=False,pin_length_m=.01788,pin_radius_m=.004,physical_flex='Synthetic grasp hypotheses: +/-5mm optical Y/Z plus independent +/-5mm optical Z depth, +/-25deg optical X tilt; no mesh deformation or optical X shift',registration='TRAINING IMAGE DOMAIN ALIGNMENT ONLY. Whole RGB and endpoint affine +15,-51 native pixels; train-only +/-15px Y jitter and .98–1.02 scale. Val/test fixed registration.',registration_evidence='Real proximal TOP cap-face centers and own-finger relative framing; never real BOTTOM ground truth.',axis_x_fraction_runtime=.5234375,annotations='Projected simulator physical TOP localz=.01788 and BOTTOM localz=0; synthetic amodal coordinates remain valid when visually occluded; visibility separately depth/seg tested',top_y_native_px=stats(arrays['top_y']*480),bottom_y_native_px=stats(arrays['bottom_y']*480),gap_native_y_px=stats((arrays['bottom_y']-arrays['top_y'])*480),split_counts={s:int(np.sum(arrays['splits']==s)) for s in ['train','validation','test']},real_bottom_ground_truth=False,limitations=['Kinematic near-jaw synthetic hypothesis, not validated contact dynamics or calibrated physical grasp.','Whole-image registration changes training appearance only; no runtime camera/nudge geometry changed.','Native320 RGB; resampling cannot create native640 fine detail.','Real BOTTOM accuracy remains unknown; synthetic heldout accuracy does not certify real distal localization.'])
 assert meta['top_y_native_px']['std']>1 and meta['gap_native_y_px']['std']>.1
 for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(arrays['groups'][arrays['splits']==a])&set(arrays['groups'][arrays['splits']==b])
 (out/'sim_endpoints.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='work/held-pin-cnn/flex-sim');p.add_argument('--generate',action='store_true');p.add_argument('--count',type=int);p.add_argument('--variants-per-anchor',type=int,default=12);a=p.parse_args();generate(a.output,a.count,a.variants_per_anchor) if a.generate else run(a.output)
