"""Causal demo-reference carrier tracking for a hole pixel location.

TRAIN reference carrier features locate the current carrier; no CNN prediction,
block identity, joint angle, test label, or future frame enters estimation. The
hole neighborhood and annotated moving-pin regions are excluded from reference
features. Results are geometric diagnostics, never contact/motor permission.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np


class DemoHoleCache:
    def __init__(self,reference_bank_path,*,mode='affine_partial',ratio_test=.72,min_matches=8,min_inliers=8,
                 min_inlier_ratio=.5,ransac_threshold_px=3.,max_reprojection_rmse_px=3.,min_coverage=.06,
                 min_axis_ratio=.015,reference_ambiguity_px=12.,ambiguity_score_ratio=.8,max_step_px=60.,
                 max_hold_ns=500_000_000,max_input_age_ns=250_000_000,max_flow_gap_ns=1_000_000_000,
                 max_flow_unanchored_ns=3_000_000_000,enable_flow=True,max_features=2500,
                 uncertainty_growth_px_per_s=12.):
        if mode not in ('affine_partial','homography'):raise ValueError('mode must be affine_partial or homography')
        if not 0<ratio_test<1 or not 0<min_inlier_ratio<=1 or not 0<min_coverage<=1 or not 0<min_axis_ratio<1 or not 0<ambiguity_score_ratio<=1:raise ValueError('invalid matching/coverage ratios')
        for name,value in (('min_matches',min_matches),('min_inliers',min_inliers),('max_features',max_features)):
            if type(value) is not int or value<4:raise ValueError(f'{name} must be an integer >=4')
        for value in (ransac_threshold_px,max_reprojection_rmse_px,reference_ambiguity_px,max_step_px):
            if not math.isfinite(value) or value<=0:raise ValueError('pixel thresholds must be positive finite')
        for value in (max_hold_ns,max_input_age_ns,max_flow_gap_ns,max_flow_unanchored_ns):
            if type(value) is not int or value<0:raise ValueError('time limits must be nonnegative integer nanoseconds')
        if type(enable_flow) is not bool or not math.isfinite(uncertainty_growth_px_per_s) or uncertainty_growth_px_per_s<0:raise ValueError('invalid flow/uncertainty configuration')
        self.config={key:value for key,value in locals().copy().items() if key not in ('self','reference_bank_path','name','value')}
        self.mode=mode;self.ratio=ratio_test;self.min_matches=min_matches;self.min_inliers=min_inliers
        self.min_inlier_ratio=min_inlier_ratio;self.ransac=ransac_threshold_px;self.max_rmse=max_reprojection_rmse_px
        self.min_coverage=min_coverage;self.min_axis_ratio=min_axis_ratio;self.ambiguity_px=reference_ambiguity_px
        self.ambiguity_ratio=ambiguity_score_ratio;self.max_step=max_step_px;self.max_hold=max_hold_ns
        self.max_input_age=max_input_age_ns;self.max_flow_gap=max_flow_gap_ns;self.max_flow_age=max_flow_unanchored_ns
        self.enable_flow=enable_flow;self.uncertainty_growth=uncertainty_growth_px_per_s
        self.sift=cv2.SIFT_create(nfeatures=max_features,contrastThreshold=.02)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)
        path=Path(reference_bank_path);self.reference_bank_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        bank=json.loads(path.read_text());rows=bank.get('references')
        if not isinstance(rows,list) or not rows:raise ValueError('reference bank requires nonempty references')
        self.references=[];self.reference_load_issues=[];ids=set()
        for row in rows:
            if row.get('split')!='train':raise ValueError('reference features and hole labels must be TRAIN only')
            identity=row.get('id')
            if not isinstance(identity,str) or not identity or identity in ids:raise ValueError('unique reference IDs required')
            ids.add(identity)
            image_path=Path(row['image_path'])
            if not image_path.is_absolute():image_path=path.resolve().parent/image_path
            if hashlib.sha256(image_path.read_bytes()).hexdigest()!=row.get('image_sha256'):raise ValueError('reference image SHA mismatch')
            bgr=cv2.imread(str(image_path))
            if bgr is None:raise ValueError('reference image cannot be decoded')
            h,w=bgr.shape[:2];polygon=self._polygon(row['carrier_polygon_px'],w,h)
            hole=np.asarray(row['hole_px'],dtype=float)
            if hole.shape!=(2,) or not np.isfinite(hole).all() or cv2.pointPolygonTest(polygon.astype(np.float32),tuple(hole),False)<0:raise ValueError('reference hole must lie within annotated carrier face')
            radius=row.get('hole_exclusion_radius_px',14.)
            if isinstance(radius,bool) or not isinstance(radius,(int,float)) or not math.isfinite(radius) or radius<=0:raise ValueError('positive hole exclusion radius required')
            mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
            for excluded in row.get('exclude_polygons_px',[]):
                excluded=self._polygon(excluded,w,h);cv2.fillPoly(mask,[np.rint(excluded).astype(np.int32)],0)
            cv2.circle(mask,tuple(np.rint(hole).astype(int)),int(math.ceil(radius)),0,-1)
            supplied_mask=row.get('mask_path',row.get('carrier_mask_path'))
            if supplied_mask:
                mask_path=Path(supplied_mask)
                if not mask_path.is_absolute():mask_path=path.resolve().parent/mask_path
                if hashlib.sha256(mask_path.read_bytes()).hexdigest()!=row.get('mask_sha256',row.get('carrier_mask_sha256')):raise ValueError('reference mask SHA mismatch')
                supplied=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
                if supplied is None or supplied.shape!=(h,w):raise ValueError('reference mask dimensions mismatch')
                mask[supplied==0]=0
            uncertainty=row.get('uncertainty_px')
            if uncertainty is not None and (isinstance(uncertainty,bool) or not isinstance(uncertainty,(int,float)) or not math.isfinite(uncertainty) or uncertainty<0):raise ValueError('reference uncertainty must be nonnegative scalar pixels')
            keypoints,descriptors=self.sift.detectAndCompute(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),mask)
            if descriptors is None or len(keypoints)<min_matches:
                self.reference_load_issues.append({'reference_id':identity,'reason':'insufficient masked carrier features','features':len(keypoints)})
                continue
            self.references.append({'id':identity,'hole':hole,'polygon':polygon,'uncertainty':uncertainty,
                'points':np.asarray([kp.pt for kp in keypoints],np.float32),'descriptors':self._rootsift(descriptors),
                'image_sha256':row['image_sha256'],'shape':(h,w),'provenance':row.get('annotation_provenance')})
        if not self.references:raise ValueError('no usable masked carrier references')
        self.reference_ids=[r['id'] for r in self.references]
        self.reset()

    @classmethod
    def from_references(cls,path,config=None):return cls(path,**(config or {}))

    @staticmethod
    def _polygon(value,width,height):
        polygon=np.asarray(value,dtype=float)
        if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():raise ValueError('polygon requires finite pixel vertices')
        if (polygon<0).any() or (polygon>=[width,height]).any() or abs(cv2.contourArea(polygon.astype(np.float32)))<16:raise ValueError('polygon outside image or degenerate')
        if not cv2.isContourConvex(polygon.astype(np.float32)):raise ValueError('carrier/exclusion polygon must be convex')
        return polygon

    @staticmethod
    def _rootsift(descriptors):
        return np.sqrt(descriptors/np.maximum(descriptors.sum(axis=1,keepdims=True),1e-12)).astype(np.float32)

    @staticmethod
    def _transform(points,matrix):
        points=np.asarray(points,dtype=float).reshape(-1,2)
        homogeneous=np.c_[points,np.ones(len(points))]@matrix.T
        if not np.isfinite(homogeneous).all() or (np.abs(homogeneous[:,2])<1e-8).any():raise ValueError('singular projective geometry')
        return homogeneous[:,:2]/homogeneous[:,2:]

    def reset(self):
        self._context=None;self._last_seen_ns=None;self._last_good_ns=None;self._reference_ns=None
        self._good=None;self._gray=None;self._points=None

    def _fit(self,source,target):
        if len(source)<self.min_inliers:return None
        if self.mode=='homography':
            matrix,inliers=cv2.findHomography(source,target,cv2.RANSAC,self.ransac,maxIters=2500,confidence=.995)
        else:
            affine,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=self.ransac,maxIters=2500,confidence=.995,refineIters=10)
            matrix=None if affine is None else np.vstack((affine,[0.,0.,1.]))
        if matrix is None or inliers is None or not np.isfinite(matrix).all():return None
        selected=inliers.reshape(-1).astype(bool)
        return matrix,selected

    def _quality(self,source,target,matrix,inliers,polygon,hole,width,height):
        count=int(inliers.sum());matches=len(source)
        if count<self.min_inliers or count/max(matches,1)<self.min_inlier_ratio:return None,'insufficient geometric inliers'
        reference=source[inliers];current=target[inliers]
        covariance=np.linalg.eigvalsh(np.cov(reference.T))
        if covariance[0]/max(covariance[-1],1e-12)<self.min_axis_ratio:return None,'degenerate near-collinear features'
        coverage=abs(cv2.contourArea(cv2.convexHull(reference.astype(np.float32))))/max(abs(cv2.contourArea(polygon.astype(np.float32))),1.)
        if coverage<self.min_coverage:return None,'insufficient spatial carrier coverage'
        try:
            projected=self._transform(reference,matrix);outline=self._transform(polygon,matrix);point=self._transform([hole],matrix)[0]
            # Local Jacobian around the hole bounds extreme/sheared/mirrored hypotheses.
            basis=self._transform([hole,hole+[1,0],hole+[0,1]],matrix)
            jacobian=np.stack((basis[1]-basis[0],basis[2]-basis[0]),axis=1)
            singular=np.linalg.svd(jacobian,compute_uv=False)
            if np.linalg.det(jacobian)<=0 or singular.min()<.3 or singular.max()>3 or singular.max()/singular.min()>4:return None,'implausible orientation/scale/shear'
            if not cv2.isContourConvex(outline.astype(np.float32)) or abs(cv2.contourArea(outline.astype(np.float32)))<32:return None,'invalid carrier polygon transform'
            if not (0<=point[0]<width and 0<=point[1]<height):return None,'transformed hole outside current image'
            error=np.linalg.norm(projected-current,axis=1);rmse=float(np.sqrt(np.mean(error**2)))
            if rmse>self.max_rmse:return None,'high carrier reprojection residual'
        except (ValueError,np.linalg.LinAlgError):return None,'invalid projective transform'
        return {'hole':point,'polygon':outline,'matrix':matrix,'inliers':count,'matches':matches,'rmse':rmse,
            'coverage':float(coverage),'scale':float(np.sqrt(abs(np.linalg.det(jacobian)))),
            'points':current.astype(np.float32),'score':count*min(1.,coverage/.2)/(1+rmse)},None

    def _reference_candidates(self,keypoints,descriptors,width,height):
        current=np.asarray([kp.pt for kp in keypoints],np.float32);candidates=[];rejections=[]
        if descriptors is None or len(descriptors)<2:return [],[{'reason':'no current carrier-compatible visual features'}]
        for ref in self.references:
            forward=self.matcher.knnMatch(ref['descriptors'],descriptors,k=2)
            reverse=self.matcher.match(descriptors,ref['descriptors'])
            reverse_best={m.queryIdx:m.trainIdx for m in reverse}
            matches=[pair[0] for pair in forward if len(pair)==2 and pair[0].distance<self.ratio*pair[1].distance and reverse_best.get(pair[0].trainIdx)==pair[0].queryIdx]
            if len(matches)<self.min_matches:
                rejections.append({'reference_id':ref['id'],'matches':len(matches),'reason':'insufficient mutual ratio matches'});continue
            source=ref['points'][[m.queryIdx for m in matches]];target=current[[m.trainIdx for m in matches]]
            fitted=self._fit(source,target)
            if fitted is None:rejections.append({'reference_id':ref['id'],'reason':'robust transform failed'});continue
            matrix,inliers=fitted
            result,reason=self._quality(source,target,matrix,inliers,ref['polygon'],ref['hole'],width,height)
            if result is None:rejections.append({'reference_id':ref['id'],'reason':reason});continue
            # An almost-as-good distinct transform is a symmetry ambiguity, not
            # evidence that the chosen bore identity is correct.
            remaining=~inliers
            alternative=self._fit(source[remaining],target[remaining]) if remaining.sum()>=self.min_inliers else None
            if alternative is not None:
                alt,_=self._quality(source[remaining],target[remaining],alternative[0],alternative[1],ref['polygon'],ref['hole'],width,height)
                if alt is not None and alt['score']>=self.ambiguity_ratio*result['score'] and np.linalg.norm(alt['hole']-result['hole'])>self.ambiguity_px:
                    rejections.append({'reference_id':ref['id'],'reason':'ambiguous repeated/symmetric carrier geometry'});continue
            result.update(reference=ref,method='reference_sift');candidates.append(result)
        return candidates,rejections

    def _flow_candidate(self,gray,timestamp,width,height):
        if not self.enable_flow or self._good is None or self._points is None or len(self._points)<self.min_inliers:return None
        if timestamp-self._last_good_ns>self.max_flow_gap or timestamp-self._reference_ns>self.max_flow_age:return None
        before=self._points.reshape(-1,1,2)
        after,status,_=cv2.calcOpticalFlowPyrLK(self._gray,gray,before,None,winSize=(21,21),maxLevel=3)
        if after is None:return None
        backward,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self._gray,after,None,winSize=(21,21),maxLevel=3)
        if backward is None:return None
        selected=status.reshape(-1).astype(bool)&back_status.reshape(-1).astype(bool)&(np.linalg.norm(backward-before,axis=2).reshape(-1)<1.5)
        source=before.reshape(-1,2)[selected];target=after.reshape(-1,2)[selected]
        fitted=self._fit(source,target)
        if fitted is None:return None
        result,_=self._quality(source,target,fitted[0],fitted[1],self._good['polygon'],self._good['hole'],width,height)
        if result is None:return None
        result['matrix']=result['matrix']@self._good['matrix']
        result.update(reference=self._good['reference'],method='carrier_optical_flow')
        return result

    def update(self,rgb,*,timestamp_ns,now_ns,sequence_id,camera_id,grasp_id):
        output={'hole_px':None,'mode':'lost','method':None,'reference_id':None,'matches':0,'match_count':0,'inliers':0,
            'reprojection_rmse_px':None,'coverage':0.,'source_ns':None,'reference_source_ns':None,'age_ns':None,
            'uncertainty_px':None,'uncertainty_kind':'uncalibrated reference/registration heuristic',
            'transform':None,'carrier_polygon_px':None,'reason':None,'diagnostics':[],
            'gate_positive':False,'motor_action':False}
        try:
            if any(isinstance(t,bool) or not isinstance(t,(int,np.integer)) or t<=0 for t in (timestamp_ns,now_ns)):raise ValueError('positive monotonic integer timestamps required')
            if not 0<=now_ns-timestamp_ns<=self.max_input_age:raise ValueError('stale or future observation')
            if any(not isinstance(value,str) or not value for value in (sequence_id,camera_id,grasp_id)):raise ValueError('explicit sequence/camera/grasp identity required')
            image=np.asarray(rgb)
            if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<32:raise ValueError('RGB uint8 full-frame HWC image required')
            context=(sequence_id,camera_id,grasp_id,image.shape[:2]);reset_reason=None
            if self._context is not None and context!=self._context:self.reset();reset_reason='sequence/camera/grasp/image geometry changed'
            if self._last_seen_ns is not None and timestamp_ns<=self._last_seen_ns:raise ValueError('timestamp regressed or repeated')
            self._context=context;self._last_seen_ns=int(timestamp_ns)
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);height,width=gray.shape
            keypoints,descriptors=self.sift.detectAndCompute(gray,None)
            descriptors=None if descriptors is None else self._rootsift(descriptors)
            candidates,rejections=self._reference_candidates(keypoints,descriptors,width,height)
            output['diagnostics']=rejections
            candidates.sort(key=lambda r:r['score'],reverse=True)
            best=candidates[0] if candidates else None
            if best is not None:
                conflicting=[r for r in candidates[1:] if r['score']>=self.ambiguity_ratio*best['score'] and np.linalg.norm(r['hole']-best['hole'])>self.ambiguity_px]
                if conflicting:best=None;output['reason']='conflicting carrier references / possible symmetry'
            flow=self._flow_candidate(gray,int(timestamp_ns),width,height)
            if best is not None and self._good is not None and timestamp_ns-self._last_good_ns<=self.max_flow_gap:
                expected=flow['hole'] if flow is not None else self._good['hole']
                limit=self.ambiguity_px if flow is not None else self.max_step
                if np.linalg.norm(best['hole']-expected)>limit:
                    best=None;output['reason']='reference transform jumps against causal carrier motion'
            if best is None and flow is not None and output['reason'] is None:best=flow
            if best is not None:
                self._good=best;self._gray=gray.copy();self._points=best['points'].copy();self._last_good_ns=int(timestamp_ns)
                if best['method']=='reference_sift':self._reference_ns=int(timestamp_ns)
                output['mode']='tracked';output['reason']=reset_reason
            elif self._good is not None and now_ns-self._last_good_ns<=self.max_hold:
                output['mode']='held';output['reason']=output['reason'] or 'brief visual drop; last carrier location held'
            else:
                # After loss, no stale image-space assumption constrains a fresh
                # robust TRAIN-reference reacquisition on a later observation.
                self._good=None;self._gray=None;self._points=None;self._last_good_ns=None;self._reference_ns=None
                output['reason']=output['reason'] or 'no unambiguous supported carrier transform'
                return output
            good=self._good;age=int(now_ns-self._last_good_ns);ref=good['reference']
            uncertainty=max(good['rmse'],float(ref['uncertainty'])*good['scale']) if ref['uncertainty'] is not None else None
            if uncertainty is not None:uncertainty+=self.uncertainty_growth*age/1e9
            output.update(hole_px=good['hole'].tolist(),reference_id=ref['id'],matches=good['matches'],match_count=good['matches'],
                inliers=good['inliers'],reprojection_rmse_px=good['rmse'],coverage=good['coverage'],source_ns=self._last_good_ns,
                reference_source_ns=self._reference_ns,age_ns=age,uncertainty_px=uncertainty,
                transform=good['matrix'].tolist(),carrier_polygon_px=good['polygon'].tolist(),
                method=good['method'] if output['mode']=='tracked' else 'stationary_hold')
        except (ValueError,TypeError,KeyError,cv2.error,np.linalg.LinAlgError) as exc:
            self.reset();output['reason']=str(exc)
        return output


def merge_with_prediction(prediction,cache_result,image_size_wh):
    """Keep the current CNN pin and fuse only an available reference-cache hole.

    Lost/invalid cache falls back to the raw CNN hole with an explicit unverified
    source flag. Gate and XYZ fields are copied unchanged; this helper creates no
    motor permission and cannot override checkpoint/runtime control restrictions.
    Neither input is mutated.
    """
    import copy
    merged=copy.deepcopy(prediction)
    original=prediction.get('keypoints')
    value=np.asarray(original,dtype=float)
    size=np.asarray(image_size_wh,dtype=float)
    if value.shape!=(4,) or not np.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('prediction requires four finite normalized keypoints')
    if size.shape!=(2,) or not np.isfinite(size).all() or (size<1).any():raise ValueError('image_size_wh requires positive width,height')
    keypoints=list(original)
    merged['raw_cnn_hole_normalized']=copy.deepcopy(keypoints[2:])
    merged['hole_source']='cnn_unverified_fallback'
    merged['cache_valid']=False
    merged['hole_cache_status']=cache_result.get('mode','lost')
    merged['hole_cache']=copy.deepcopy(cache_result)
    merged['gate_permission_from_cache']=False
    if cache_result.get('mode') in ('tracked','held'):
        try:
            hole=np.asarray(cache_result.get('hole_px'),dtype=float)
            valid=hole.shape==(2,) and np.isfinite(hole).all() and bool(np.all((hole>=0)&(hole<size)))
        except (TypeError,ValueError):valid=False
        if valid:
            keypoints[2:]=(hole/size).tolist()
            merged['cache_valid']=True
            merged['hole_source']='demo_reference_'+cache_result['mode']
        else:
            merged['hole_cache_status']='lost'
            merged['hole_cache']['reason']='invalid/out-of-frame cache result; raw CNN hole retained explicitly'
    merged['keypoints']=keypoints
    return merged
