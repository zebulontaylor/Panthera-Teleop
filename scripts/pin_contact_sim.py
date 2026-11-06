#!/usr/bin/env python3
"""Superseded legacy-scene contact hypotheses, for historical reproduction only.

Use generate_pin_insertion_photo_data.py for the existing photo environment.
"""
import argparse,json,os,struct,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import cv2,mujoco,numpy as np
from scipy.spatial.transform import Rotation
from real2sim_bridge import Real2SimBridge,DEFAULT_SCENE
from pin_insertion_sim import _place,correction
from real2sim_object_pose import carrier_bore_points


def stl_triangles(path):
    blob=Path(path).read_bytes();count=struct.unpack('<I',blob[80:84])[0]
    return np.ndarray(count,dtype=[('normal','<f4',3),('vertices','<f4',(3,3)),('attr','<u2')],buffer=blob,offset=84)['vertices'].astype(float)


def surface_heights(triangles,xy):
    """Exact triangle vertical intersections; max visible CAD surface, NaN in void."""
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    valid=abs(den)>1e-15;den=np.where(valid,den,1)
    out=[]
    for x,y in np.asarray(xy):
        u=((b[:,1]-c[:,1])*(x-c[:,0])+(c[:,0]-b[:,0])*(y-c[:,1]))/den
        v=((c[:,1]-a[:,1])*(x-c[:,0])+(a[:,0]-c[:,0])*(y-c[:,1]))/den
        hit=valid&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
        z=u*a[:,2]+v*b[:,2]+(1-u-v)*c[:,2]
        out.append(float(np.max(z[hit])) if hit.any() else np.nan)
    return np.asarray(out)


def disk_support(triangles,center_xy,radius=.004,rotation=None):
    """Sample pin footprint over CAD top surface, including rim and center."""
    angles=np.arange(32)*2*np.pi/32
    cap=np.vstack([np.zeros((1,2)),np.c_[np.cos(angles),np.sin(angles)]*radius,np.c_[np.cos(angles),np.sin(angles)]*radius*.5])
    disk=np.c_[cap,np.zeros(len(cap))]@(np.eye(3) if rotation is None else rotation).T
    points=disk[:,:2]+center_xy
    heights=surface_heights(triangles,points)-disk[:,2];finite=np.isfinite(heights)
    if not finite.any():return dict(supported=False,stable=False,z=None,fraction=0.)
    z=float(np.nanmax(heights));contact=finite&(abs(heights-z)<.00005)
    hull=cv2.convexHull(points[contact].astype(np.float32)) if contact.sum()>=3 else None
    stable=bool(hull is not None and cv2.pointPolygonTest(hull,tuple(map(float,center_xy)),False)>=0)
    return dict(supported=bool(contact.any()),stable=stable,z=z,fraction=float(contact.mean()))


def transform(p,r):
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;return t


def align_axis(rotation,axis,target):
    source=rotation@axis;cross=np.cross(source,target);s=np.linalg.norm(cross);c=np.clip(source@target,-1,1)
    if s<1e-8:return rotation if c>0 else Rotation.from_rotvec(np.pi*rotation[:,1]).as_matrix()@rotation
    return Rotation.from_rotvec(cross/s*np.arctan2(s,c)).as_matrix()@rotation


def solve_left(bridge,position,rotation,iterations=160):
    model,data=bridge.model,bridge.data;site=bridge.sites[0]
    joints=[model.joint(f'joint{i}').id for i in range(1,7)]
    qadr=model.jnt_qposadr[joints];dadr=model.jnt_dofadr[joints]
    for _ in range(iterations):
        mujoco.mj_forward(model,data)
        dp=position-data.site_xpos[site]
        dr=Rotation.from_matrix(rotation@data.site_xmat[site].reshape(3,3).T).as_rotvec()
        if np.linalg.norm(dp)<.0002 and np.linalg.norm(dr)<.01:return True
        jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,jp,jr,site)
        J=np.vstack([jp[:,dadr],jr[:,dadr]*.15]);error=np.r_[dp,dr*.15]
        dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,error)
        data.qpos[qadr]=np.clip(data.qpos[qadr]+np.clip(dq,-.10,.10),bridge.arm_ranges[0,:,0],bridge.arm_ranges[0,:,1])
    return False


def interarm_penetration(bridge):
    model,data=bridge.model,bridge.data
    left=[];right=[]
    for gid in range(model.ngeom):
        if model.geom_group[gid]!=3:continue
        name=model.body(int(model.geom_bodyid[gid])).name
        if name in ('link6','L_finger','R_finger'):left.append(gid)
        elif name in ('right_link6','right_L_finger','right_R_finger'):right.append(gid)
    maximum=0.
    for a in left:
        for b in right:
            maximum=max(maximum,-float(mujoco.mj_geomDistance(model,data,a,b,.005,None)))
    return maximum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-prototype', action='store_true', help='Explicitly reproduce the superseded opposed-base scene; not approved training data')
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--per-category',type=int,default=32)
    parser.add_argument('--grasp-prior',choices=['cad-pad','fitted-review-only'],default='cad-pad');parser.add_argument('--seed',type=int,default=901);parser.add_argument('--image-height',type=int,default=128)
    args=parser.parse_args()
    if not args.legacy_prototype:
        parser.error('Legacy scene is superseded. Use generate_pin_insertion_photo_data.py; --legacy-prototype is for historical reproduction only.')
    rng=np.random.default_rng(args.seed)
    manifest=json.loads(args.manifest.read_text());fit=json.loads(args.calibration.read_text())
    source=[r for r in manifest['frames'] if r['split']=='train' and r.get('state14') is not None]
    if not source:raise ValueError('No TRAIN-only measured joint seeds')
    bridge=Real2SimBridge();model,data=bridge.model,bridge.data
    parts=json.loads((DEFAULT_SCENE.parent/'manifest.json').read_text())['parts']
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    right_world=np.linalg.inv(np.asarray(fit['T_right_base_left_base']));rb=model.body('right_base').id
    model.body_pos[rb]=right_world[:3,3];quat=np.empty(4);mujoco.mju_mat2Quat(quat,right_world[:3,:3].ravel());model.body_quat[rb]=quat
    for name in ('small_carrier','pin_2','pin_3','pin_4','gear_1','gear_2','gear_3','gear_4'):_place(bridge,name,[3,3,3],np.eye(3))
    model.geom_pos[model.geom('table_top').id,2]-=.05;mujoco.mj_setConst(model,data)
    candidates=[(c,h) for h in fit['carrier_pose_hypotheses'] for c in h['candidates'] if h['split']=='train']
    front=[p for p in candidates if np.asarray(p[0]['T_camera_carrier'])[:3,2]@(-np.asarray(p[0]['T_camera_carrier'])[:3,3])>0]
    carrier_prior,_=min(front or candidates,key=lambda pair:pair[0]['reprojection_rms_px'])
    tc=np.asarray(carrier_prior['T_right_tool_carrier']);camera_tool=np.asarray(fit['T_right_tool_camera_opencv'])
    meshdir=DEFAULT_SCENE.parent/'meshes';triangles=stl_triangles(meshdir/'large_carrier/visual.stl')
    height=args.image_height;width=round(height*4/3);scale=height/480
    K=np.asarray(fit['K'])*scale;K[2,2]=1;dist=np.asarray(fit['distortion'])
    rw,rh=width*2,height*2;ideal=np.array([[K[1,1],0,rw/2],[0,K[1,1],rh/2],[0,0,1.]])
    pixels=np.stack(np.meshgrid(np.arange(width),np.arange(height)),axis=-1).astype(np.float32)
    remap=cv2.undistortPoints(pixels.reshape(-1,1,2),K,dist,P=ideal).reshape(height,width,2)
    cid=model.camera('right_wrist').id;model.cam_fovy[cid]=np.rad2deg(2*np.arctan(rh/(2*K[1,1])))
    renderer=mujoco.Renderer(model,height=rh,width=rw);option=mujoco.MjvOption();option.sitegroup[:]=0;option.geomgroup[:]=[1,1,1,0,0,0]
    categories=['supported_outside','supported_crescent','aligned_bore','hover','away']
    results=[];rejects={};attempts=0
    try:
        for category in categories:
            accepted=0
            while accepted<args.per_category and attempts<args.per_category*len(categories)*100:
                attempts+=1;row=source[int(rng.integers(len(source)))];bridge.set_kinematic_pose(row['state14'])
                right=transform(data.site_xpos[bridge.sites[1]],data.site_xmat[bridge.sites[1]].reshape(3,3));left=transform(data.site_xpos[bridge.sites[0]],data.site_xmat[bridge.sites[0]].reshape(3,3))
                carrier=right@tc;normal=carrier[:3,2]
                pin_offset=np.array([.010,0,0]) if args.grasp_prior=='cad-pad' else np.asarray(fit['pin_tip_left_tool_m'])
                axis=np.array([-1.,0,0]) if args.grasp_prior=='cad-pad' else -pin_offset/np.linalg.norm(pin_offset)
                if args.grasp_prior=='cad-pad':
                    adjusted=bridge.native_state();adjusted[6]=.225;bridge.set_kinematic_pose(adjusted)
                bore=carrier_bore_points()[int(rng.integers(4))];angle=rng.uniform(-np.pi,np.pi)
                radius={'supported_outside':(.005,.009),'supported_crescent':(.001,.004),'aligned_bore':(0,.00012),'hover':(.002,.009),'away':(.012,.019)}[category]
                offset=rng.uniform(*radius)*np.array([np.cos(angle),np.sin(angle)])
                xy=bore[:2]+offset
                pin_rotation=left[:3,:3]@Rotation.from_rotvec([0,-np.pi/2,0]).as_matrix() if args.grasp_prior=='cad-pad' else left[:3,:3]@Rotation.align_vectors([axis],[[0.,0,1.]])[0].as_matrix()
                tilt=np.arccos(np.clip(pin_rotation[:,2]@normal,-1,1))
                if tilt>np.deg2rad(85):rejects['axis_not_above_surface']=rejects.get('axis_not_above_surface',0)+1;continue
                support=disk_support(triangles,xy,rotation=carrier[:3,:3].T@pin_rotation)
                if category.startswith('supported') and not support['supported']:rejects['unsupported_or_unstable']=rejects.get('unsupported_or_unstable',0)+1;continue
                z=support['z'] if support['z'] is not None else .01725
                if category=='aligned_bore':z=.01725
                if category=='hover':z+=rng.uniform(.008,.025)
                tip=carrier[:3,:3]@np.r_[xy,z]+carrier[:3,3]
                orientation=left[:3,:3].copy()
                if not solve_left(bridge,tip-orientation@pin_offset,orientation):rejects['ik_failed']=rejects.get('ik_failed',0)+1;continue
                penetration=interarm_penetration(bridge)
                if penetration>.003:rejects['interarm_overlap_over3mm']=rejects.get('interarm_overlap_over3mm',0)+1;continue
                state=bridge.native_state();actual_tip=data.site_xpos[bridge.sites[0]]+data.site_xmat[bridge.sites[0]].reshape(3,3)@pin_offset
                if np.linalg.norm(actual_tip-tip)>.0003:raise RuntimeError('IK grasp mismatch')
                _place(bridge,'large_carrier',carrier[:3,3],carrier[:3,:3])
                for gear in ('gear_1','gear_2','gear_3','gear_4'):
                    offset_gear=np.asarray(parts[gear]['assembled_origin'])-parts['large_carrier']['assembled_origin'];_place(bridge,gear,carrier[:3,3]+carrier[:3,:3]@offset_gear,carrier[:3,:3])
                _place(bridge,'pin_1',tip,pin_rotation)
                world_cam=right@camera_tool;cam_mj=world_cam.copy();cam_mj[:3,:3]=world_cam[:3,:3]@np.diag([1,-1,-1]);parent=model.cam_bodyid[cid]
                local=np.linalg.inv(transform(data.xpos[parent],data.xmat[parent].reshape(3,3)))@cam_mj
                model.cam_pos[cid]=local[:3,3];mujoco.mju_mat2Quat(model.cam_quat[cid],local[:3,:3].ravel());mujoco.mj_forward(model,data)
                holes=carrier[:3,3]+np.c_[carrier_bore_points()[:,:2],np.full(4,.01725)]@carrier[:3,:3].T
                hole=holes[np.argmin(np.linalg.norm(holes-tip,axis=1))]
                points=np.array([tip,hole]);pc=(points-world_cam[:3,3])@world_cam[:3,:3]
                uv=cv2.projectPoints(pc,np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)
                if np.any(pc[:,2]<=0) or np.any(uv<0) or np.any(uv>=[width,height]):rejects['out_of_view']=rejects.get('out_of_view',0)+1;continue
                renderer.update_scene(data,camera='right_wrist',scene_option=option);rgb=cv2.remap(renderer.render(),remap[:,:,0],remap[:,:,1],cv2.INTER_LINEAR)
                renderer.enable_segmentation_rendering();seg=cv2.remap(renderer.render().astype(np.float32),remap[:,:,0],remap[:,:,1],cv2.INTER_NEAREST).astype(np.int32);renderer.disable_segmentation_rendering()
                ix=np.minimum(uv.astype(int),[width-1,height-1]);pinid=model.geom('pin_1_visual').id
                pin_support=seg[ix[0,1],ix[0,0],0]==pinid;hole_pin=seg[ix[1,1],ix[1,0],0]==pinid
                renderer.enable_depth_rendering();depth=cv2.remap(renderer.render(),remap[:,:,0],remap[:,:,1],cv2.INTER_NEAREST);renderer.disable_depth_rendering()
                carrier_visible=np.count_nonzero(seg[:,:,0]==model.geom('large_carrier_visual').id)>=20
                hole_direct=bool(depth[ix[1,1],ix[1,0]]>=pc[1,2]-.001)
                hole_valid=hole_direct or (hole_pin and carrier_visible)
                active=category.startswith('supported') and pin_support and hole_valid
                angles=np.arange(32)*2*np.pi/32
                aperture=np.vstack([np.zeros((1,2)),np.c_[np.cos(angles),np.sin(angles)]*.004, np.c_[np.cos(angles),np.sin(angles)]*.002])
                aperture_world=hole+np.c_[aperture,np.zeros(len(aperture))]@carrier[:3,:3].T
                aperture_pc=(aperture_world-world_cam[:3,3])@world_cam[:3,:3]
                aperture_uv=cv2.projectPoints(aperture_pc,np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)
                ap=np.clip(aperture_uv.astype(int),[0,0],[width-1,height-1]);own=seg[ap[:,1],ap[:,0],0]==pinid;clear=depth[ap[:,1],ap[:,0]]>=aperture_pc[:,2]-.001
                visible_fraction=float(clear.mean());pin_fraction=float(own.mean());external_fraction=float((~clear & ~own).mean())
                visibility_category='external_occluded' if external_fraction>.15 else 'own_pin_full' if pin_fraction>.9 else 'own_pin_partial' if pin_fraction>.05 else 'direct_visible'
                delta=correction(tip,hole,data.xmat[rb].reshape(3,3),'pin') if active else np.zeros(3)
                results.append(dict(images=rgb,state=state,keypoints=(uv/[width,height]).ravel(),keypoints_valid=[pin_support]*2+[hole_valid]*2,active=[float(active)],delta_xyz_m=delta,
                    group_ids=row['episode_id'],category=category,contact_support_fraction=support['fraction'],supported=category.startswith('supported'),stable_support_geometry=support['stable'],visibility_category=visibility_category,hole_visible_fraction=visible_fraction,hole_pin_occlusion_fraction=pin_fraction,hole_external_occlusion_fraction=external_fraction,hole_occluded_by_pin=bool(hole_pin),source_id=row['id'],interarm_penetration_m=penetration,tool_pin_rotation=orientation.T@pin_rotation,pin_tilt_deg=np.rad2deg(tilt)))
                accepted+=1
                print(f'{category} accepted{accepted} attempts{attempts}',flush=True)
            if accepted<args.per_category:raise RuntimeError(f'Could not balance category{category}:accepted{accepted},rejects{rejects}')
    finally:renderer.close()
    arrays={k:np.asarray([r[k] for r in results]) for k in results[0]}
    for k in ['state','keypoints','keypoints_valid','active','delta_xyz_m']:arrays[k]=arrays[k].astype(np.float32)
    groups=sorted({str(r['session_id']) for r in source});validation_group=groups[-1]
    episode_groups={r['episode_id']:r['session_id'] for r in source}
    arrays['split']=np.asarray(['validation' if episode_groups[e]==validation_group else 'train' for e in arrays['group_ids']])
    meta=dict(diagnostic_only=True,training_use_approved=False,grasp_prior=args.grasp_prior,grasp_real_calibrated=False,simulation_gripper_native_rad=.225,generator='pin_contact_sim.py',camera='right_wrist',delta_units='meters',state_order='left joints1..6,gripper,right joints1..6,gripper (radians)',keypoints_order=['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v'],kind='CAD_supported_contact_hypotheses',schema_version=1,seed=args.seed,category_counts={k:int(np.sum(arrays['category']==k)) for k in categories},visibility_counts={str(k):int(v) for k,v in zip(*np.unique(arrays['visibility_category'],return_counts=True))},active_count=int(arrays['active'].sum()),attempts=attempts,rejections=rejects,
        sources=[dict(id=e,session_id=episode_groups[e],split='train') for e in sorted(episode_groups)],sim_validation_session_id=validation_group,sim_split_definition='last lexicographic realTRAIN session held out from simulation training; realvalidation/test never synthesized',controlled_arm='left',controlled_object='pin',delta_frame='right_native_base',real_calibrated=False,physics_simulated=False,
        assumptions=['Exact CAD triangle footprint sampling establishes only geometric contact/stability, not forces, friction or successful nudge.',
        'Four bore CENTER pitch-circle radius17.9625mm; individualbore~8.34mm and pin8mm diameter, radialclearance~0.17mm.',
        'Camera,base,carriergrasp,pingrasp hypotheses from train-only calibration; carrierPnp has unresolved90degree symmetry/normal ambiguity.',
        'Seedtoolorientationpreserved; synthetic uncalibrated pin-in-jaws orientation varied, pinbasecap tilted, its origin raised until actual CADtop first touches sampledcap rim, not falsely forced to mouthplane.',
        'Left6DofIK from measuredTRAIN seeds enforces hypothetical rigid pin grasp and joint limits; resultingstates are simulationIK, not originalmeasuredjoints.',
        'CAD-pad graspprior fixed allsamples: freetip tool[+10,0,0]mm, cylinder+Z=-toolX, rearend tool[-7.88,0,0]mm overlaps fingertip pads; SIMjawgap8mm via nativegripper.225rad endpointapproximation. This replaces physicallyinfeasible fittedoffset for localizationaugmentation only; NOT realgraspcalibration.',
        'Contact baseheight is maximum actualmesh height across65sample footprint; tiltedcap contact may be unstable without leftarm support; stable_support_geometry reports sampledcontacthull containment only. No settling/dynamics.',
        'Aligned bore negative is unsupported mouth-plane hypothesis; hoverandaway inactive; no insertion successlabels.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**arrays,metadata_json=np.asarray(json.dumps(meta)));args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(width*5,height*min(args.per_category,4)))
    for c,category in enumerate(categories):
        ids=np.flatnonzero(arrays['category']==category)[:4]
        for j,i in enumerate(ids):
            im=Image.fromarray(arrays['images'][i]);ImageDraw.Draw(im).text((2,2),category,fill='red');sheet.paste(im,(c*width,j*height))
    sheet.save(args.output.with_suffix('.png'));print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
