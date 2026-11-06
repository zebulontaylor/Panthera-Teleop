#!/usr/bin/env python3
"""Offline FK-constrained natural-texture plane calibration hypothesis.

Only saved images/FK are read. Pairwise SIFT/RANSAC tracks on masked tabletop
texture constrain metric plane homographies. The z=0 plane and rigid camera
mount are explicit assumptions; texture tracking is not independent metrology.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from calibrate_pin_insertion_geometry import (rigid,validate_transform,default_prior,
    numerical_jacobian,jacobian_report,linearized_uncertainty,pose_excitation)

LOWER_TABLE_POLYGON=np.array([[.40,.66],[.63,.66],[.66,.86],[.40,.86]])
MASK_POLYGON=np.array([[.30,.09],[.91,.09],[.95,.45],[.79,.62],[.50,.57],[.34,.40]])


def texture_mask(shape,polygon=MASK_POLYGON):
    h,w=shape[:2];mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[(np.asarray(polygon)*[w,h]).astype(np.int32)],255)
    return mask


def wood_mask(image,roi=MASK_POLYGON):
    polygon=texture_mask(image.shape,roi)
    if image.ndim!=3:return polygon
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    wood=cv2.inRange(hsv,np.array([7,40,45]),np.array([40,245,255]))
    return cv2.bitwise_and(polygon,wood)


def match_table_features(image_a,image_b,max_points=70,roi_a=MASK_POLYGON,roi_b=MASK_POLYGON):
    sift=cv2.SIFT_create(nfeatures=1500,contrastThreshold=.025)
    ma,mb=wood_mask(image_a,roi_a),wood_mask(image_b,roi_b)
    coverage_color=min(np.count_nonzero(ma)/max(np.count_nonzero(texture_mask(image_a.shape,roi_a)),1),np.count_nonzero(mb)/max(np.count_nonzero(texture_mask(image_b.shape,roi_b)),1))
    if image_a.ndim==3 and coverage_color<.55:return None,dict(reason='not_wood_dominated_polygon',wood_fraction=coverage_color)
    ga=cv2.cvtColor(image_a,cv2.COLOR_BGR2GRAY) if image_a.ndim==3 else image_a
    gb=cv2.cvtColor(image_b,cv2.COLOR_BGR2GRAY) if image_b.ndim==3 else image_b
    ka,da=sift.detectAndCompute(ga,ma)
    kb,db=sift.detectAndCompute(gb,mb)
    if da is None or db is None:return None,dict(reason='no_descriptors')
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
    if len(good)<15:return None,dict(reason='fewer_than15_ratio_matches',ratio_matches=len(good))
    pa=np.array([ka[m.queryIdx].pt for m in good]);pb=np.array([kb[m.trainIdx].pt for m in good])
    H,inlier=cv2.findHomography(pa,pb,cv2.RANSAC,2.0,maxIters=4000,confidence=.999)
    if H is None:return None,dict(reason='homography_failed')
    keep=inlier.ravel().astype(bool);pa,pb=pa[keep],pb[keep]
    motion=float(np.median(np.linalg.norm(pa-pb,axis=1)))
    if len(pa)<15 or motion<2:return None,dict(reason='insufficient_inlier_motion',inliers=len(pa),median_motion_px=motion)
    # Spatial coverage rejects a tiny isolated moving part dominating RANSAC.
    extent=np.ptp(pa,axis=0);area=float(np.prod(extent)/np.prod(image_a.shape[:2]))
    if area<.006:return None,dict(reason='insufficient_spatial_coverage',coverage=area)
    select=np.linspace(0,len(pa)-1,min(max_points,len(pa)),dtype=int)
    return (pa[select],pb[select]),dict(inliers=len(pa),ratio_matches=len(good),median_motion_px=motion,coverage=area,homography=H.tolist())


def read_episode(path,stride=6,max_frames=80):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text())
    if not meta.get('complete') or meta.get('label')!='kept':raise ValueError('Require complete kept real recording')
    frames=[json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines() if s.strip()]
    all_frames=[r for r in frames if r['camera']=='wrist_port2']
    total_frames=len(all_frames)
    frames=all_frames[::stride][:max_frames]
    states=[json.loads(s) for s in (path/'states.jsonl').read_text().splitlines() if s.strip()]
    times=np.array([r['monotonic_ns'] for r in states],np.int64)
    result=[]
    for frame in frames:
        index=int(np.argmin(abs(times-int(frame['monotonic_ns']))));skew=abs(int(times[index])-int(frame['monotonic_ns']))/1e9
        if skew>.025:continue
        pose=states[index]['arms']['right']['tool_pose_base']
        result.append(dict(path=str(path/frame['path']),pose=validate_transform(rigid(pose['rotation'],pose['position_m'])),
            frame_index=frame['frame_index'],table_roi=LOWER_TABLE_POLYGON if frame['frame_index']<24 or frame['frame_index']>=total_frames-30 else MASK_POLYGON,skew_s=skew,episode_id=meta['id']))
    return result,meta


def extract_pairs(frames,heldout_every=5,max_points=70,independent_validation=False):
    images=[cv2.imread(r['path']) for r in frames]
    if any(im is None for im in images):raise ValueError('Missing saved image')
    pairs=[];audit=[]
    for gap in (1,3):
        for i in range(len(frames)-gap):
            j=i+gap
            # Reserved frames never enter training feature residuals.
            split='validation' if independent_validation or i%heldout_every==0 or j%heldout_every==0 else 'train'
            matched,report=match_table_features(images[i],images[j],max_points,frames[i]['table_roi'],frames[j]['table_roi'])
            audit.append(dict(i=i,j=j,split=split,**report))
            if matched is not None:
                pairs.append(dict(a=frames[i],b=frames[j],pa=matched[0],pb=matched[1],split=split))
    return pairs,audit


def plane_transfer(pa,poses_a,poses_b,camera_tool,K,distortion,table_z=0.):
    """Anchor source rays on fixed metric table; project into destination camera."""
    rays=np.c_[cv2.undistortPoints(np.asarray(pa,dtype=np.float64).reshape(-1,1,2),K,distortion).reshape(-1,2),np.ones(len(pa))]
    ca=np.asarray(poses_a)@camera_tool;cb=np.asarray(poses_b)@camera_tool
    direction=np.einsum('nij,nj->ni',ca[:,:3,:3],rays)
    denom=direction[:,2];safe=np.where(abs(denom)<1e-5,np.where(denom<0,-1e-5,1e-5),denom)
    distance=(table_z-ca[:,2,3])/safe
    world=ca[:,:3,3]+distance[:,None]*direction
    point=np.einsum('nji,nj->ni',cb[:,:3,:3],world-cb[:,:3,3])
    # cv2 handles the actual radial model (k1,k2,p1=0,p2=0,k3=0).
    projection=cv2.projectPoints(point,np.zeros(3),np.zeros(3),K,distortion)[0].reshape(-1,2)
    return projection,distance,point[:,2],world


def _stack(pairs):
    return dict(pa=np.concatenate([p['pa'] for p in pairs]),pb=np.concatenate([p['pb'] for p in pairs]),
        a=np.concatenate([np.repeat(p['a']['pose'][None],len(p['pa']),axis=0) for p in pairs]),
        b=np.concatenate([np.repeat(p['b']['pose'][None],len(p['pa']),axis=0) for p in pairs]))


def fit_plane(pairs,prior,table_z=0.,starts=3,seed=42):
    train=[p for p in pairs if p['split']=='train'];validation=[p for p in pairs if p['split']=='validation']
    if len(train)<6:raise ValueError('Fewer than six usable training table feature pairs')
    tr=_stack(train);va=_stack(validation) if validation else None
    cam0=np.asarray(prior['T_right_tool_camera_opencv']);k0=np.asarray(prior['K'])
    sigmas=np.array([.5]*3+[.08]*3+[.2,.2,50,50,.15,.15])
    names=['camera_rx','camera_ry','camera_rz','camera_tx','camera_ty','camera_tz','log_fx','log_fy','cx_px','cy_px','k1','k2']
    def decode(z):
        x=z*sigmas;camera=rigid(Rotation.from_rotvec(x[:3]).as_matrix()@cam0[:3,:3],cam0[:3,3]+x[3:6])
        k=k0.copy();k[0,0]*=np.exp(x[6]);k[1,1]*=np.exp(x[7]);k[0,2]+=x[8];k[1,2]+=x[9]
        return camera,k,np.r_[x[10:12],0.,0.,0.]
    def predict(z,data):
        c,k,d=decode(z);return plane_transfer(data['pa'],data['a'],data['b'],c,k,d,table_z)
    def image_residual(z,data):return (predict(z,data)[0]-data['pb']).ravel()/1.5
    def objective(z):
        _,distance,depth,_=predict(z,tr)
        return np.r_[image_residual(z,tr),z,np.minimum(distance,0)*100,np.minimum(depth,0)*100]
    rng=np.random.default_rng(seed);solutions=[]
    for i in range(starts):
        initial=np.zeros(12) if i==0 else rng.normal(0,.2,12)
        solutions.append(least_squares(objective,initial,bounds=(-4,4),loss='soft_l1',f_scale=2,max_nfev=500))
    best=min(solutions,key=lambda s:s.cost);c,k,d=decode(best.x)
    jac=numerical_jacobian(lambda z:image_residual(z,tr),best.x)
    def metrics(data,z=None):
        if data is None:return dict(points=0)
        projection,distance,depth,_=predict(best.x if z is None else z,data);errors=np.linalg.norm(projection-data['pb'],axis=1)
        return dict(points=len(errors),mean_px=float(errors.mean()),median_px=float(np.median(errors)),p95_px=float(np.percentile(errors,95)),
            max_px=float(errors.max()),rays_behind_plane=int(np.sum(distance<=0)),negative_destination_depth=int(np.sum(depth<=0)))
    # Table coordinates from paired rays are latent, fit-dependent hypotheses.
    _,_,_,xyz=predict(best.x,tr)
    return dict(schema_version=1,kind='stationary_texture_metric_plane_camera_hypothesis',real_calibrated=False,metric_accuracy_validated=False,
        T_right_tool_camera_opencv=c.tolist(),K=k.tolist(),distortion=d.tolist(),table_z_m=table_z,
        camera_coordinate_convention='OpenCV +xright,+ydown,+zforward',training=metrics(tr),validation=metrics(va),
        validation_by_episode={episode:metrics(_stack([p for p in validation if p['a'].get('episode_id','synthetic')==episode])) for episode in sorted({p['a'].get('episode_id','synthetic') for p in validation})},training_pairs=len(train),validation_pairs=len(validation),data_only_identifiability=jacobian_report(jac,names),
        data_only_linearized_uncertainty=linearized_uncertainty(jac,sigmas,names),
        multistart=[dict(cost=float(s.cost),converged=bool(s.success),nfev=int(s.nfev),training=metrics(tr,s.x),validation=metrics(va,s.x),T_right_tool_camera_opencv=decode(s.x)[0].tolist(),K=decode(s.x)[1].tolist(),distortion=decode(s.x)[2].tolist()) for s in solutions],
        multistart_parameter_std=(np.std([s.x for s in solutions],axis=0)*sigmas).tolist(),parameter_names=names,
        parameters_at_prior_bound=[n for n,v in zip(names,best.x) if abs(v)>3.99],prior=prior,
        latent_train_table_xy_m=xyz[:,:2].tolist(),
        assumptions=['Each retained SIFT/RANSAC match is the SAME stationary physical point on planar tabletop z=table_z_m in RIGHT native base.',
            'Natural wood features are unmeasured; source-ray plane intersection supplies latent table XY, not independent world coordinates.',
            'Known measured URDF FK, image timestamps, table plane and rigid camera mount are correct; arm flex and rolling shutter neglected.',
            'Conservative image polygon and RANSAC reject many moving objects but do not guarantee every match is tabletop; visual audit required.',
            'Planar scene and limited wrist rotation can still leave camera parameters weakly observable; numerical convergence is not metric validation.',
            'Camera transfer to a different date/setup requires independent mount/profile verification; matching device path alone proves nothing.'])


def evaluate_camera(pairs,configuration):
    data=_stack(pairs)
    projection,distance,depth,_=plane_transfer(data['pa'],data['a'],data['b'],np.asarray(configuration['T_right_tool_camera_opencv']),np.asarray(configuration['K']),np.asarray(configuration['distortion']),configuration.get('table_z_m',0))
    errors=np.linalg.norm(projection-data['pb'],axis=1)
    return dict(pairs=len(pairs),points=len(errors),mean_px=float(errors.mean()),median_px=float(np.median(errors)),p95_px=float(np.percentile(errors,95)),max_px=float(errors.max()),rays_behind_plane=int(np.sum(distance<=0)),negative_destination_depth=int(np.sum(depth<=0)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--validation-episode',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=6);p.add_argument('--max-frames',type=int,default=80)
    p.add_argument('--starts',type=int,default=3);p.add_argument('--table-z-m',type=float,default=0.)
    p.add_argument('--scene',type=Path,default=Path('/home/robotics/Desktop/RobotArmTraining/sim/actuator/scene_bimanual.xml'))
    a=p.parse_args();frames,meta=read_episode(a.episode,a.stride,a.max_frames);pairs,audit=extract_pairs(frames)
    sources=[dict(episode=meta['id'],role='training_with_reserved_frames',metadata_sha256=hashlib.sha256((a.episode/'metadata.json').read_bytes()).hexdigest())]
    if a.validation_episode:
        val,vm=read_episode(a.validation_episode,a.stride,a.max_frames);vp,va=extract_pairs(val,independent_validation=True)
        pairs+=vp;audit+=va;sources.append(dict(episode=vm['id'],role='independent_validation'))
    result=fit_plane(pairs,default_prior(a.scene),a.table_z_m,a.starts)
    result['sources']=sources;result['feature_pair_audit']=audit;result['mask_polygon_normalized']=MASK_POLYGON.tolist();result['early_late_table_roi']=dict(polygon=LOWER_TABLE_POLYGON.tolist(),first_frames=24,last_frames=30,provenance='manually audited saved images exclude room walls in camera-upward phases');result['wood_color_mask_hsv']=dict(lower=[7,40,45],upper=[40,245,255],minimum_polygon_wood_fraction=.55)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    # Visual audit of raw image polygon and retained stationary-texture tracks.
    tiles=[]
    for pair in pairs[::max(1,len(pairs)//8)][:8]:
        im=cv2.imread(pair['a']['path']);cv2.polylines(im,[(pair['a']['table_roi']*[im.shape[1],im.shape[0]]).astype(np.int32)],True,(0,255,255),2)
        for x,y in pair['pa']:cv2.circle(im,(round(x),round(y)),3,(0,255,0),1)
        cv2.putText(im,f"{pair['a']['episode_id']} frame{pair['a']['frame_index']}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
        tiles.append(cv2.resize(im,(320,240)))
    if len(tiles)==8:cv2.imwrite(str(a.output.with_suffix('.tracks.jpg')),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
    print(json.dumps({k:result[k] for k in ['training','validation','training_pairs','validation_pairs','data_only_identifiability','parameters_at_prior_bound']},indent=2))

if __name__=='__main__':main()
