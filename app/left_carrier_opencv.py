"""RGB-only real LEFT carrier bore registration and fresh local rim evidence.

TRAIN references supply appearance and approximate cyclic geometry. Every visible
bore needs a new dark-ellipse/rim verification in the current frame. No state,
CNN, held-out label, future frame or stationary pixel hold enters prediction.
Canonical cyclic slots are image-reference identities, not verified CAD rotation.
"""
from __future__ import annotations
import hashlib,json,itertools,time
from pathlib import Path
import cv2
import numpy as np
from scripts.prepare_left_carrier_opencv import refine_bore
from app.carrier_aux_holes import detect_aux_holes

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BANK=ROOT/'work/left-carrier-opencv/runtime/reference-bank.json'
PITCH_M=.0179625
CANONICAL=np.array([[PITCH_M,0],[0,PITCH_M],[-PITCH_M,0],[0,-PITCH_M]],np.float64)

def transform(points,H):
    p=np.c_[np.asarray(points),np.ones(len(points))]@H.T
    if not np.isfinite(p).all() or (np.abs(p[:,2])<1e-8).any():raise ValueError('invalid projective geometry')
    return p[:,:2]/p[:,2,None]

def cyclic_layout(points):
    """Image-clockwise ring order; infer missing vertex only from a clear opposite pair."""
    p=np.asarray(points,float)
    if p.shape not in ((3,2),(4,2)) or not np.isfinite(p).all():raise ValueError('three or four finite train bore centers required')
    if len(p)==3:
        pairs=sorted([(np.linalg.norm(p[i]-p[j]),i,j) for i,j in itertools.combinations(range(3),2)],reverse=True)
        if pairs[0][0]<1.2*pairs[1][0]:raise ValueError('three centers do not establish a clear opposite pair')
        _,i,j=pairs[0];k=next(k for k in range(3) if k not in (i,j));p=np.vstack((p,p[i]+p[j]-p[k]))
    center=p.mean(0);angle=np.arctan2(p[:,1]-center[1],p[:,0]-center[0]);order=np.argsort(angle);p=p[order]
    # Define slot zero as upper-left angular quadrant, then clockwise in image.
    start=np.argmin(p[:,0]+p[:,1]);p=np.roll(p,-int(start),axis=0)
    if abs(cv2.contourArea(p.astype(np.float32)))<400:raise ValueError('degenerate carrier layout')
    return p

def weighted_affine_refine(H,main_xy,main_px,aux_xy,aux_px,*,bearing_px=None,aux_weight=.25):
    """Regularized pixel-affine IRLS; small holes have quarter anchor influence.

    The initial three-anchor map is a prior, perspective is never freed, and a
    candidate is accepted only if small-hole residual improves while large-bore
    residual stays <=2 px. Gross auxiliary outliers are excluded before fitting.
    """
    H=np.asarray(H,float);main_xy=np.asarray(main_xy,float).reshape(-1,2);main_px=np.asarray(main_px,float).reshape(-1,2)
    aux_xy=np.asarray(aux_xy,float).reshape(-1,2);aux_px=np.asarray(aux_px,float).reshape(-1,2)
    main_before=np.linalg.norm(transform(main_xy,H)-main_px,axis=1)
    aux_before=np.linalg.norm(transform(aux_xy,H)-aux_px,axis=1) if len(aux_xy) else np.array([])
    keep=aux_before<=8
    diagnostics={'accepted':False,'weights':{'large_bore':1.,'bearing':.5,'inner_threaded':float(aux_weight),'raised_boss':0.,'edge_pose':0.},
        'large_before_px':main_before.tolist(),'auxiliary_before_px':aux_before.tolist(),'auxiliary_outlier_count':int((~keep).sum()),
        'perspective_parameters_freed':False,
        'bearing_before_px':float(np.linalg.norm(transform([[0,0]],H)[0]-bearing_px)) if bearing_px is not None else None}
    if keep.sum()<2:return H,diagnostics
    xy=[*main_xy];target=[*main_px];weights=[1.]*len(main_xy)
    if bearing_px is not None:xy.append([0,0]);target.append(bearing_px);weights.append(.5)
    xy.extend(aux_xy[keep]);target.extend(aux_px[keep]);weights.extend([aux_weight]*int(keep.sum()))
    predicted=transform(np.asarray(xy),H);target=np.asarray(target,float);weights=np.asarray(weights)
    center=transform([[0,0]],H)[0];scale=max(20.,float(np.median(np.linalg.norm(predicted[:len(main_xy)]-center,axis=1))))
    X=np.c_[(predicted-center)/scale,np.ones(len(predicted))];Y=(target-center)/scale
    prior=np.array([[1.,0.,0.],[0.,1.,0.]]).T;beta=prior.copy()
    for _ in range(6):
        residual=np.linalg.norm((X@beta-Y)*scale,axis=1)
        robust=np.minimum(1.,2./np.maximum(residual,1e-8));effective=weights*robust
        beta=np.linalg.solve(X.T@(effective[:,None]*X)+.10*np.eye(3),X.T@(effective[:,None]*Y)+.10*prior)
    D=np.eye(3);D[:2,:2]=beta[:2].T;D[:2,2]=center-D[:2,:2]@center+scale*beta[2]
    candidate=D@H
    for alpha in (1.,.5,.25,.125):
        correction=np.eye(3)+alpha*(D-np.eye(3));trial=correction@H
        large_after=np.linalg.norm(transform(main_xy,trial)-main_px,axis=1)
        aux_after=np.linalg.norm(transform(aux_xy[keep],trial)-aux_px[keep],axis=1)
        singular=np.linalg.svd(correction[:2,:2],compute_uv=False)
        if np.linalg.det(correction[:2,:2])<=0 or singular.min()<.9 or singular.max()>1.1:continue
        if large_after.max(initial=0)>2. or np.mean(aux_after)>=np.mean(aux_before[keep])-1e-4:continue
        if bearing_px is not None and np.linalg.norm(transform([[0,0]],trial)[0]-bearing_px)>2.5:continue
        diagnostics.update(accepted=True,large_after_px=large_after.tolist(),auxiliary_after_px=aux_after.tolist(),
                           auxiliary_inlier_count=int(keep.sum()),accepted_step=alpha,
                           auxiliary_mean_before_px=float(aux_before[keep].mean()),auxiliary_mean_after_px=float(aux_after.mean()),
                           bearing_after_px=float(np.linalg.norm(transform([[0,0]],trial)[0]-bearing_px)) if bearing_px is not None else None)
        return trial,diagnostics
    return H,diagnostics

class LeftCarrierOpenCVPredictor:
    def __init__(self,reference_bank=None,*,ratio=.78,min_inliers=6,max_rmse=2.5,enable_aux=True):
        self.path=Path(reference_bank or DEFAULT_BANK);bank=json.loads(self.path.read_text())
        self.metadata={'schema':'left-carrier-opencv-real-references-v1','camera':'left_wrist','reference_bank':str(self.path),
            'reference_bank_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),'training_only_references':True,
            'canonical_physical_slot_identity_verified':False,'geometry_kind':'affine_face_layout',
            'reference_slot_convention':'image clockwise from upper-left; approximate planar four-bore symmetry',
            'motor_permission':False,'state_inputs':False}
        self.enable_aux=bool(enable_aux)
        self.profile=json.loads((ROOT/'config/pin-insertion-photo/carrier-real-face-v4-profile.json').read_text())
        self.ratio=ratio;self.min_inliers=min_inliers;self.max_rmse=max_rmse
        self.sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.012,edgeThreshold=12)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2);self.references=[];self.load_issues=[]
        rows=bank.get('frames',bank.get('references',[]))
        for row in rows:
            if row.get('split')!='train':raise ValueError('OpenCV reference bank must contain TRAIN frames only')
            try:
                image_path=Path(row['image_path']);image_path=image_path if image_path.is_absolute() else self.path.parent/image_path
                rgb=cv2.imread(str(image_path));gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
                if row.get('image_sha256') and hashlib.sha256(image_path.read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('reference SHA mismatch')
                raw=np.array(row.get('unordered_visible_bores_px',row.get('bores_px')),float)
                raw=raw[np.isfinite(raw).all(1)]
                radii=row.get('bore_radii_px',[8.5]*len(raw));refined=[refine_bore(gray,p,r,max_shift=6) for p,r in zip(raw,radii)]
                # A failed local verifier does not erase a manually reviewed train geometry anchor.
                points=np.array([r['center'] if r else p for p,r in zip(raw,refined)])
                full=cyclic_layout(points)
                H=cv2.getPerspectiveTransform(CANONICAL.astype(np.float32),full.astype(np.float32))
                center=transform([[0,0]],H)[0]
                angles=np.linspace(0,2*np.pi,64,endpoint=False)
                circle=np.column_stack((np.cos(angles),np.sin(angles)))*.02649
                face_polygon=np.asarray(row.get('carrier_polygon_px',transform(circle,H).tolist()))
                aux=detect_aux_holes(gray,face_polygon,points.tolist(),H) if self.enable_aux else []
                mask=np.zeros(gray.shape,np.uint8)
                # Only neighborhoods around directly annotated visible bores supply
                # reference features: no table, whole gripper, or hidden-bore patch.
                for p,r in zip(points,refined):cv2.circle(mask,tuple(np.rint(p).astype(int)),int(round((r['radius'] if r else 8.5)*2.1)),255,-1)
                for a in aux:
                    if a['type'] in ('inner_threaded','central_bearing'):
                        cv2.circle(mask,tuple(np.rint(a['center']).astype(int)),int(round(a['radius']*1.7+3)),255,-1)
                if row.get('carrier_polygon_px'):
                    outline=np.zeros_like(mask);cv2.fillPoly(outline,[np.rint(row['carrier_polygon_px']).astype(np.int32)],255);mask=cv2.bitwise_and(mask,outline)
                for polygon in row.get('exclude_polygons_px',[]):cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],0)
                kp,desc=self.sift.detectAndCompute(gray,mask)
                if desc is None or len(kp)<min_inliers:raise ValueError('insufficient visible-face features')
                self.references.append({'id':row['id'],'points':np.array([k.pt for k in kp],np.float32),'descriptors':self._root(desc),
                    'full':full,'H':H,'center':center,'radius':float(np.median([r['radius'] for r in refined if r] or [8.5])),
                    'image_path':str(image_path),'aux':aux,'face_polygon':face_polygon,'feature_mask_area':int((mask>0).sum()),'feature_count':len(kp)})
            except (ValueError,TypeError,cv2.error) as exc:self.load_issues.append({'id':row.get('id'),'reason':str(exc)})
        if not self.references:raise ValueError('no usable real left TRAIN references')
        self._calibrate_inner_template()

    def _calibrate_inner_template(self):
        samples=[]
        for ref in self.references:
            inner=[a for a in ref['aux'] if a['type']=='inner_threaded']
            if inner:
                xy=transform([a['center'] for a in inner],np.linalg.inv(ref['H']))
                samples.extend((q,ref['id']) for q in xy)
        phases=[]
        for phase in (-22.5,22.5):
            bins=[[] for _ in range(4)];total=0.;count=0
            expected=np.arange(4)*np.pi/2+np.deg2rad(phase)
            for xy,identity in samples:
                angle=np.arctan2(xy[1],xy[0]);error=np.abs(np.angle(np.exp(1j*(angle-expected))))
                slot=int(np.argmin(error))
                if error[slot]<np.deg2rad(25):bins[slot].append((xy,identity));count+=1;total+=float(error[slot])
            phases.append((count,-total,phase,bins))
        _,_,phase,bins=max(phases,key=lambda item:(item[0],item[1]))
        points=[];evidence=[];pitch=self.profile['inner_threaded_holes']['pitch_radius_mm']/1000
        for i,values in enumerate(bins):
            prior=np.array([np.cos(i*np.pi/2+np.deg2rad(phase)),np.sin(i*np.pi/2+np.deg2rad(phase))])*pitch
            point=np.median([v[0] for v in values],axis=0) if values else prior
            points.append(point);evidence.append({'slot':i,'train_reference_count':len(set(v[1] for v in values)),
                'sample_count':len(values),'position_mm':(point*1000).tolist(),
                'source':'real_train_auxiliary_median' if values else 'photo_profile_symmetry_prior',
                'spread_mm':float(np.median(np.linalg.norm(np.array([v[0] for v in values])-point,axis=1))*1000) if values else None})
        self.inner_template=np.asarray(points);self.inner_template_phase=phase
        self.metadata['inner_template_calibration']={'train_only':True,'phase_hypothesis_deg':phase,'landmarks':evidence,
            'source':'native real TRAIN dark-hole centers inverse mapped by train large-bore affine geometry; coordinate medians; no attachment or eval points'}

    def _match_inner_template(self,H,observations):
        inner=[a for a in observations if a['type']=='inner_threaded'];options=[]
        for handedness in (1,-1):
            xy=self.inner_template.copy();xy[:,1]*=handedness;projected=transform(xy,H);used=set();matches=[]
            for i,p in enumerate(projected):
                candidates=[(np.linalg.norm(np.asarray(q['center'])-p),j) for j,q in enumerate(inner) if j not in used]
                if candidates:
                    distance,j=min(candidates)
                    if distance<6:used.add(j);matches.append((i,j,float(distance)))
            options.append({'phase':self.inner_template_phase*handedness,'points':xy,'matches':matches,
                'mean':float(np.mean([m[2] for m in matches])) if matches else 100.,'handedness':handedness})
        options.sort(key=lambda o:(-len(o['matches']),o['mean']));chosen,other=options
        chosen['orientation_verified']=bool(len(chosen['matches'])>=2 and (len(chosen['matches'])>len(other['matches']) or chosen['mean']+1<other['mean']))
        return chosen,inner

    def _weighted_current_fit(self,H,gray,best,visible,pixels):
        angles=np.linspace(0,2*np.pi,64,endpoint=False);outline=transform(np.column_stack((np.cos(angles),np.sin(angles)))*.02649,H)
        observations=best.get('aux_observations') or detect_aux_holes(gray,outline,[p for p,v in zip(pixels,visible) if v],H)
        best['aux_observations']=observations
        chosen,inner=self._match_inner_template(H,observations)
        if not chosen['orientation_verified']:return H,{'accepted':False,'reason':'fewer than two unambiguous real-template small-hole matches'}
        xy=np.array([chosen['points'][i] for i,j,e in chosen['matches']]);current=np.array([inner[j]['center'] for i,j,e in chosen['matches']])
        ids=np.flatnonzero(visible);large=np.array([pixels[i] for i in ids]);bearing=best['bearing']['center'] if best.get('bearing') else None
        refined,fit=weighted_affine_refine(H,CANONICAL[ids],large,xy,current,bearing_px=bearing)
        fit['template_source']='real_train_auxiliary_medians';fit['template_matches']=len(chosen['matches'])
        fit['profile_prior_auxiliary_mean_px']=float(np.mean(np.linalg.norm(transform(np.column_stack((np.cos(np.arange(4)*np.pi/2+np.deg2rad(chosen['phase'])),np.sin(np.arange(4)*np.pi/2+np.deg2rad(chosen['phase']))))*self.profile['inner_threaded_holes']['pitch_radius_mm']/1000,H)[[i for i,j,e in chosen['matches']]]-current,axis=1)))
        return refined,fit

    @staticmethod
    def _root(desc):return np.sqrt(desc/np.maximum(desc.sum(1,keepdims=True),1e-12)).astype(np.float32)

    def _candidate(self,ref,current,descriptors,gray):
        pairs=self.matcher.knnMatch(ref['descriptors'],descriptors,k=2)
        reverse=self.matcher.match(descriptors,ref['descriptors']);back={m.queryIdx:m.trainIdx for m in reverse}
        matches=[p[0] for p in pairs if len(p)==2 and p[0].distance<self.ratio*p[1].distance and back.get(p[0].trainIdx)==p[0].queryIdx]
        if len(matches)<self.min_inliers:return None
        src=ref['points'][[m.queryIdx for m in matches]];dst=current[[m.trainIdx for m in matches]]
        A,inlier=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1500,confidence=.995)
        if A is None or inlier is None:return None
        selected=inlier.ravel().astype(bool);n=int(selected.sum())
        if n<self.min_inliers or n/len(matches)<.45:return None
        M=np.vstack((A,[0,0,1]));sing=np.linalg.svd(A[:,:2],compute_uv=False)
        if np.linalg.det(A[:,:2])<=0 or sing.min()<.65 or sing.max()>1.5:return None
        spread=np.linalg.svd(src[selected]-src[selected].mean(0),compute_uv=False)
        if spread[1]<8:return None
        rmse=float(np.sqrt(np.mean(np.sum((transform(src[selected],M)-dst[selected])**2,axis=1))))
        if rmse>self.max_rmse:return None
        proposed=transform(ref['full'],M);center=transform([ref['center']],M)[0]
        h,w=gray.shape
        if not (0<center[0]<w and 0<center[1]<h):return None
        radius=ref['radius']*float(np.sqrt(np.linalg.det(A[:,:2])))
        evidence=[]
        for p in proposed:
            r=refine_bore(gray,p,radius,max_shift=8)
            evidence.append(r)
        ids=[i for i,r in enumerate(evidence) if r]
        # Distinct physical bores cannot collapse onto a single local dark region.
        if any(np.linalg.norm(np.array(evidence[i]['center'])-evidence[j]['center'])<radius*2 for i,j in itertools.combinations(ids,2)):return None
        if len(ids)<2:return None
        aux=[];matched_aux=[];aux_H=None
        if self.enable_aux:
            polygon=transform(ref['face_polygon'],M)
            aux=detect_aux_holes(gray,polygon,[r['center'] for r in evidence if r],M@ref['H'])
            used=set()
            for ai,a in enumerate(ref['aux']):
                if a['type'] not in ('inner_threaded','central_bearing'):continue
                expected=transform([a['center']],M)[0]
                options=[(np.linalg.norm(np.asarray(q['center'])-expected),j,q) for j,q in enumerate(aux) if j not in used and q['type']==a['type'] and .7<q['radius']/(a['radius']*np.sqrt(np.linalg.det(A[:,:2])))<1.4]
                if options:
                    distance,j,q=min(options,key=lambda item:item[0])
                    if distance<4.5:
                        used.add(j);q['reference_landmark_id']=ref['id']+'_aux_'+str(ai);q['reference_correspondence_verified']=True
                        matched_aux.append((a,q))
            inner=[(a,q) for a,q in matched_aux if a['type']=='inner_threaded']
            if len(ids)>=2 and len(inner)>=2:
                aux_xy=transform([a['center'] for a,q in inner],np.linalg.inv(ref['H']))
                aux_pixels=np.array([q['center'] for a,q in inner])
                base_H=M@ref['H'];large_pixels=np.array([evidence[i]['center'] for i in ids])
                proposal,fit=weighted_affine_refine(base_H,CANONICAL[ids],large_pixels,aux_xy,aux_pixels)
                initial_aux=np.linalg.norm(transform(aux_xy,base_H)-aux_pixels,axis=1)
                initial_large=np.linalg.norm(transform(CANONICAL[ids],base_H)-large_pixels,axis=1)
                if fit['accepted'] or (initial_aux.max()<2.5 and initial_large.max()<2.):
                    if np.linalg.norm(transform(CANONICAL,proposal)-proposed,axis=1).max()<6:
                        aux_H=proposal
                        for a,q in inner:q['metric_homography_eligible']=True
        score=20*len(ids)+n/(1+rmse)
        return {'reference_id':ref['id'],'proposed':proposed,'evidence':evidence,'inliers':n,'matches':len(matches),
                'rmse':rmse,'score':score,'reference_H':M@ref['H'],'scale':radius/ref['radius'],
                'aux_observations':aux,'matched_aux_count':len(matched_aux),'aux_H':aux_H}

    def _constellation_candidate(self,gray):
        """Cold start from actual concentric bearing and a three-large-bore ring."""
        from app.carrier_aux_holes import global_central_bearing_candidates
        bearing_candidates=global_central_bearing_candidates(gray)
        if not bearing_candidates:return None
        edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),35,95)
        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        bores=[]
        for contour in contours:
            if len(contour)<16:continue
            (x,y),(a,b),_=cv2.fitEllipse(contour)
            radius=float(np.sqrt(a*b)/2)
            if not 6<radius<14 or min(a,b)/max(a,b)<.55:continue
            if any(np.linalg.norm(np.array([x,y])-q['center'])<4 for q in bores):continue
            bore=refine_bore(gray,[x,y],radius,max_shift=2.)
            if bore and not any(np.linalg.norm(np.asarray(bore['center'])-q['center'])<4 for q in bores):bores.append(bore)
        best=None
        for bearing in bearing_candidates:
            center=np.asarray(bearing['center'])
            nearby=[q for q in bores if 2.8<np.linalg.norm(np.asarray(q['center'])-center)/q['radius']<6.2]
            for triple in itertools.combinations(nearby,3):
                points=np.array([q['center'] for q in triple]);radii=np.array([q['radius'] for q in triple])
                if radii.max()/radii.min()>1.45:continue
                distances=np.linalg.norm(points-center,axis=1)
                if distances.std()/distances.mean()>.20:continue
                pairs=sorted([(np.linalg.norm(points[i]-points[j]),i,j) for i,j in itertools.combinations(range(3),2)],reverse=True)
                if pairs[0][0]<1.20*pairs[1][0]:continue
                _,i,j=pairs[0];k=next(k for k in range(3) if k not in (i,j))
                inferred_center=(points[i]+points[j])/2
                if np.linalg.norm(inferred_center-center)>7:continue
                u=points[i]-center;v=points[j]-center;z=points[k]-center
                if np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))>-.80:continue
                if abs(np.dot(u,z)/(np.linalg.norm(u)*np.linalg.norm(z)))>.45:continue
                try:full=cyclic_layout(points)
                except ValueError:continue
                evidence=[None]*4
                for bore in triple:
                    index=int(np.argmin(np.linalg.norm(full-np.asarray(bore['center']),axis=1)));evidence[index]=bore
                # Search the fourth current rim as well; never mark symmetry inference visible.
                radius=float(np.median(radii))
                for index,q in enumerate(full):
                    if evidence[index] is None:evidence[index]=refine_bore(gray,q,radius,max_shift=4.)
                visible=sum(q is not None for q in evidence)
                score=20*visible+sum(q['confidence'] for q in evidence if q)+float(bearing.get('concentric_support',1))
                if best is None or score>best['score']:
                    H=cv2.getPerspectiveTransform(CANONICAL.astype(np.float32),full.astype(np.float32))
                    best={'reference_id':'cold_start_constellation','proposed':full,'evidence':evidence,
                          'inliers':visible,'matches':visible,'rmse':None,'score':score,'reference_H':H,
                          'scale':1.,'bearing':bearing,'cold_start':True}
        # Two adjacent insertion bores plus a genuinely observed concentric
        # bearing center are three non-collinear physical landmarks. They fully
        # constrain an approximate affine face map, without a reference image.
        for bearing in bearing_candidates:
            center=np.asarray(bearing['center']);nearby=[q for q in bores if 2.8<np.linalg.norm(np.asarray(q['center'])-center)/q['radius']<6.2]
            for one,two in itertools.combinations(nearby,2):
                pair=np.array([one['center'],two['center']]);vectors=pair-center
                distances=np.linalg.norm(vectors,axis=1);radii=np.array([one['radius'],two['radius']])
                if radii.max()/radii.min()>1.45 or distances.max()/distances.min()>1.30:continue
                if abs(np.dot(vectors[0],vectors[1])/np.prod(distances))>.40:continue
                if not .30<bearing['radius']/distances.mean()<.75:continue
                full=cyclic_layout(np.vstack((pair,2*center-pair)))
                evidence=[None]*4
                for bore in (one,two):evidence[int(np.argmin(np.linalg.norm(full-bore['center'],axis=1)))]=bore
                for index,q in enumerate(full):
                    if evidence[index] is None:evidence[index]=refine_bore(gray,q,float(np.median(radii)),max_shift=4.)
                n=sum(q is not None for q in evidence)
                score=20*n+sum(q['confidence'] for q in evidence if q)+float(bearing.get('concentric_support',1))
                if best is None or score>best['score']:
                    H=cv2.getPerspectiveTransform(CANONICAL.astype(np.float32),full.astype(np.float32))
                    best={'reference_id':'cold_start_constellation','proposed':full,'evidence':evidence,'inliers':n+1,
                          'matches':n+1,'rmse':None,'score':score,'reference_H':H,'scale':1.,'bearing':bearing,
                          'cold_start':True,'geometry_from_bearing':True}
        return best

    def _face_outputs(self,H,gray,best,visible,pixels):
        if H is None:return {'face_landmarks':[],'face_outline_px':None,'layout_holes':[],
                             'layout_orientation_verified':False,'registration_landmark_count':sum(visible)}
        angles=np.linspace(0,2*np.pi,72,endpoint=False)
        unit=np.column_stack((np.cos(angles),np.sin(angles)))
        outline=transform(unit*self.profile['plate_outer_diameter_mm']/2000,H)
        observations=list(best.get('aux_observations',[]))
        if self.enable_aux and not observations:
            observations=detect_aux_holes(gray,outline,[p for p,v in zip(pixels,visible) if v],H)
        if best.get('bearing'):
            bearing=dict(best['bearing']);bearing.update(type='central_bearing',plane='face',observed=True)
            observations=[a for a in observations if not (a['type']=='central_bearing' and np.linalg.norm(np.asarray(a['center'])-bearing['center'])<5)]
            observations.append(bearing)
        chosen,inner=self._match_inner_template(H,observations)
        orientation=chosen['orientation_verified']
        matched_inner={tuple(np.round(inner[j]['center'],5)):chosen['points'][slot] for slot,j,error in chosen['matches']} if orientation else {}
        matched_slots={tuple(np.round(inner[j]['center'],5)):slot for slot,j,error in chosen['matches']} if orientation else {}
        landmarks=[]
        for i,a in enumerate(observations):
            kind=a['type'];canonical_xy=matched_inner.get(tuple(np.round(a['center'],5))) if kind=='inner_threaded' else np.zeros(2) if kind=='central_bearing' else None
            eligible=bool(canonical_xy is not None and (kind=='inner_threaded' or a.get('concentric_support',0)>=2))
            stable_id=f'inner_template_{matched_slots[tuple(np.round(a["center"],5))]}' if kind=='inner_threaded' and tuple(np.round(a['center'],5)) in matched_slots else 'bearing_center' if kind=='central_bearing' else a.get('reference_landmark_id',f'current_{kind}_{i}')
            landmarks.append(dict(a,id=stable_id,kind=kind,
                px=a['center'],observed=True,inferred=False,plane='raised_boss' if kind=='boss_mounting' else 'main_face',
                insertion_target=False,classification_verified=False,
                metric_homography_eligible=eligible,canonical_xy_m=canonical_xy.tolist() if canonical_xy is not None else None,
                point_weight=.25 if kind=='inner_threaded' else .5 if kind=='central_bearing' else 0.))
        definitions=[('central_bearing',np.zeros(2),self.profile['central_upper_bore_diameter_mm']/2000,0.,False)]
        for i,xy in enumerate(CANONICAL):definitions.append((f'insertion_bore_{i}',xy,self.profile['insertion_bore_diameter_mm']/2000,0.,True))
        for i,xy in enumerate(chosen['points']):definitions.append((f'inner_threaded_{i}',xy,self.profile['inner_threaded_holes']['diameter_mm']/2000,0.,False))
        bosses=self.profile['raised_mounting_bosses'];boss_angles=np.arange(4)*np.pi/2+np.pi/4
        # CAD recovered boss centers are (+/-16.475588,+/-16.475588)mm:
        # exactly the 23.3mm ring at 45 degrees, invariant under handedness swap.
        for i,a in enumerate(boss_angles):definitions.append((f'boss_mounting_{i}',np.array([np.cos(a),np.sin(a)])*bosses['pitch_radius_mm']/1000,bosses['mounting_hole_diameter_mm']/2000,bosses['height_above_plate_mm'],False))
        layout=[]
        for name,xy,radius,height,is_target in definitions:
            px=transform([xy],H)[0];kind='insertion_bore' if is_target else 'inner_threaded' if name.startswith('inner') else 'boss_mounting' if name.startswith('boss') else name
            observed=False;actual=None
            if is_target:
                slot=int(name.rsplit('_',1)[1]);observed=bool(visible[slot]);actual=pixels[slot] if observed else None
            elif kind!='boss_mounting' or height==0:
                nearest=[(np.linalg.norm(np.asarray(a['px'])-px),a) for a in landmarks if a['kind']==kind]
                if nearest:
                    distance,a=min(nearest,key=lambda z:z[0])
                    if distance<6:observed=True;actual=a['px']
            layout.append({'id':name,'kind':kind,'px':actual if observed else px.tolist(),
                'profile_px':px.tolist(),'outline_px':transform(xy+unit*radius,H).tolist(),
                'observed':observed,'inferred':not observed,'insertion_target':is_target,
                'plane':'raised_boss' if height else 'main_face','height_above_face_mm_hypothesis':height,
                'projection_kind':'coplanar approximation for raised boss' if height else 'approximate planar face projection',
                'physical_slot_identity_verified':False,'layout_orientation_verified':orientation})
        return {'face_landmarks':landmarks,'face_outline_px':outline.tolist(),'layout_holes':layout,
                'layout_orientation_verified':orientation,'inner_layout_phase_deg':chosen['phase'],
                'registration_landmark_count':sum(visible)+best.get('matched_aux_count',0)+(1 if best.get('bearing') else 0),
                'layout_inner_phase_evidence_count':len(chosen['matches']),
                'inner_template_calibration':self.metadata['inner_template_calibration'],
                'geometry_source':'real_train_calibrated_inner_template_plus_large_bore_photo_profile',
                'profile_source':'config/pin-insertion-photo/carrier-real-face-v4-profile.json',
                'boss_phase_source':'recovered CAD boss centers, 45 degree phase at 23.3mm ring'}

    def predict(self,rgb,state=None):
        image=np.asarray(rgb)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<32:raise ValueError('full-frame RGB uint8 required')
        start=time.perf_counter();gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);h,w=gray.shape
        kp,desc=self.sift.detectAndCompute(gray,None);candidates=[]
        if desc is not None and len(kp)>=self.min_inliers:
            current=np.array([k.pt for k in kp],np.float32);desc=self._root(desc)
            for ref in self.references:
                c=self._candidate(ref,current,desc,gray)
                if c:candidates.append(c)
        candidates.sort(key=lambda c:c['score'],reverse=True)
        best=candidates[0] if candidates else None;reason='no fresh registered carrier/rim evidence'
        if best:
            # Repeated bores may match a different location. Reject equally strong
            # references that disagree in unordered layout, rather than picking one.
            for alt in candidates[1:]:
                if alt['score']<best['score']*.85:continue
                distance=np.mean(np.min(np.linalg.norm(best['proposed'][:,None]-alt['proposed'][None,:],axis=-1),axis=1))
                if distance>10:best=None;reason='ambiguous carrier registration';break
        if best is None or sum(r is not None for r in best['evidence'])<3:
            cold=self._constellation_candidate(gray)
            if cold is not None:best=cold
        holes=[];visible=[False]*4;pixels=[None]*4;H=None;geometry_valid=False
        if best:
            visible=[r is not None for r in best['evidence']];ids=np.flatnonzero(visible)
            if best.get('aux_H') is not None:
                H=best['aux_H'];projected=transform(CANONICAL,H)
                geometry_valid=bool(cv2.isContourConvex(projected.astype(np.float32)) and 400<abs(cv2.contourArea(projected.astype(np.float32)))<50000)
            elif len(ids)>=3:
                observed=np.array([best['evidence'][i]['center'] for i in ids])
                if len(ids)==3:
                    affine=cv2.getAffineTransform(CANONICAL[ids].astype(np.float32),observed.astype(np.float32));H=np.vstack((affine,[0,0,1]))
                else:H=cv2.getPerspectiveTransform(CANONICAL.astype(np.float32),observed.astype(np.float32))
                projected=transform(CANONICAL,H)
                geometry_valid=bool(cv2.isContourConvex(projected.astype(np.float32)) and 400<abs(cv2.contourArea(projected.astype(np.float32)))<50000)
            elif len(ids)>=2 and best.get('geometry_from_bearing'):
                H=best['reference_H'];projected=transform(CANONICAL,H)
                geometry_valid=bool(cv2.isContourConvex(projected.astype(np.float32)) and 400<abs(cv2.contourArea(projected.astype(np.float32)))<50000)
            else:projected=best['proposed']
            if not geometry_valid:H=None
            for i,r in enumerate(best['evidence']):
                if r:pixels[i]=r['center']
                elif geometry_valid and (projected[i]>=0).all() and (projected[i]<[w,h]).all():pixels[i]=projected[i].tolist()
            reason='fresh real rim detections with approximate four-bore layout' if geometry_valid else 'visible rims only; hidden layout unsupported'
        for i in range(4):
            r=best['evidence'][i] if best else None
            holes.append({'id':f'insertion_bore_{i}','index':i,'uv':(np.asarray(pixels[i])/[w,h]).tolist() if pixels[i] is not None else None,
                          'confidence':r['confidence'] if r else 0.,'observed':visible[i],
                          'inferred':bool(pixels[i] is not None and not visible[i]),'rim_evidence':r,
                          'canonical_xy_m':CANONICAL[i].tolist(),'plane':'main_face','metric_homography_eligible':visible[i],
                          'point_weight':1.})
        point_fit={'accepted':False}
        if H is not None and best and self.enable_aux:
            H,point_fit=self._weighted_current_fit(H,gray,best,visible,pixels)
            projected=transform(CANONICAL,H)
            for i in range(4):
                if not visible[i] and pixels[i] is not None:pixels[i]=projected[i].tolist()
        face=self._face_outputs(H,gray,best,visible,pixels) if best else {'face_landmarks':[],'face_outline_px':None,'layout_holes':[],'layout_orientation_verified':False,'registration_landmark_count':0}
        face['primary_registration_landmark_count']=sum(visible)+(1 if best and best.get('bearing') else 0)
        face['secondary_registration_landmark_count']=int(point_fit.get('auxiliary_inlier_count',0)) if point_fit.get('accepted') else 0
        if point_fit.get('accepted'):
            face['registration_landmark_count']=face['primary_registration_landmark_count']+face['secondary_registration_landmark_count']
        for i,hole in enumerate(holes):
            if hole['inferred'] and pixels[i] is not None:hole['uv']=(np.asarray(pixels[i])/[w,h]).tolist()
        edge={}
        if H is not None:
            from app.carrier_face_edge import refine_face_outline
            edge=refine_face_outline(gray,H)
            edge['edge_contour_source']=edge.get('shape_template_source','approximate_photo_profile_circle')
            point_fit['edge_pose_influence_weight']=0.
            point_fit['edge_appearance_secondary_weight']=.05
            face['profile_circle_outline_px']=face['face_outline_px']
            face['face_outline_px']=edge.get('face_outline_px',face['face_outline_px'])
        return {**face,**edge,'point_fit':point_fit,'holes_px':pixels,'visible':visible,'holes':holes,'supported':geometry_valid,'geometry_valid':geometry_valid,
                'homography':H.tolist() if H is not None else None,'face_homography':None,
                'geometry_kind':('reference_aux_face_layout' if best and best.get('aux_H') is not None else 'affine_three_total_landmarks' if best and best.get('geometry_from_bearing') and sum(visible)==2 else 'affine_face_layout' if sum(visible)==3 else 'reference_planar_face_layout'),
                'registration_kind':('affine_aux_landmarks' if best and best.get('aux_H') is not None else 'affine3landmark' if geometry_valid and sum(visible)==2 else 'affine3bore' if sum(visible)==3 else 'projective4bore' if sum(visible)==4 else 'unsupported'),
                                'bearing_observed':bool((best and best.get('bearing')) or any(a['kind']=='central_bearing' for a in face['face_landmarks'])),
                'canonical_physical_slot_identity_verified':False,'homography_coordinate_frame':'image-cyclic reference face XY metres, physical CAD rotation unverified',
                'reference_id':best['reference_id'] if best else None,'inliers':best['inliers'] if best else 0,
                'reprojection_rmse_px':best['rmse'] if best else None,'method':('current_rgb_bearing_bore_constellation' if best and best.get('cold_start') else 'real_train_reference_sift_plus_local_ellipse'),
                'reason':reason,'camera':'left_wrist','motor_permission':False,
                'latency_ms':(time.perf_counter()-start)*1000,'pixel_hold':False}
