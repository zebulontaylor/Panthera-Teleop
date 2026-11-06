"""RGB evidence for auxiliary carrier landmarks, never insertion targets.

Coordinates are native image pixels. Optional face_homography maps face XY in
metres to pixels. Approximate photo profile dimensions are hypotheses;
raised boss observations must never be used as coplanar metric correspondences.
"""
import cv2
import numpy as np


def refine_aperture(gray, center, radius, max_shift=3., min_radius=1.4,
                    min_contrast=16., min_coverage=.65, face_mask=None):
    """Refine a dark aperture ellipse with resolved surrounding bright rim."""
    gray=np.asarray(gray)
    if gray.ndim!=2:raise ValueError('Expected grayscale image')
    x,y=map(float,center);radius=float(radius)
    margin=int(np.ceil(radius*1.8+max_shift+2));ix,iy=int(round(x)),int(round(y))
    if radius<min_radius or min(ix,iy)<margin or ix+margin>=gray.shape[1] or iy+margin>=gray.shape[0]:return None
    crop=gray[iy-margin:iy+margin+1,ix-margin:ix+margin+1]
    smooth=cv2.GaussianBlur(crop,(3,3),.55)
    edges=cv2.Canny(smooth,25,70); contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    best=None
    for contour in contours:
        if len(contour)<8:continue
        (cx,cy),(a,b),angle=cv2.fitEllipse(contour);lo,hi=min(a,b)/2,max(a,b)/2;r=np.sqrt(lo*hi)
        q=np.array([cx+ix-margin,cy+iy-margin]);shift=np.linalg.norm(q-[x,y])
        if shift>min(max_shift,max(1.,radius*.4)) or r<min_radius or not .65*radius<r<1.45*radius or lo/hi<.5:continue
        theta=np.linspace(0,2*np.pi,48,endpoint=False);ar=np.deg2rad(angle)
        def xy(scale):
            return cx+scale*(a/2*np.cos(theta)*np.cos(ar)-b/2*np.sin(theta)*np.sin(ar)),cy+scale*(a/2*np.cos(theta)*np.sin(ar)+b/2*np.sin(theta)*np.cos(ar))
        def sample(scale):
            xx,yy=xy(scale);return cv2.remap(smooth,xx.astype('float32')[None],yy.astype('float32')[None],cv2.INTER_LINEAR)[0].astype(float)
        inner=sample(.45);outer=sample(1.45);contrast=outer-inner
        coverage=float(np.mean(contrast>10));difference=float(np.mean(contrast))
        # Dark centers plus bright surroundings reject highlights and isolated ink/noise.
        if coverage<min_coverage or difference<min_contrast or np.mean(inner)>np.mean(outer)*.83:continue
        if face_mask is not None:
            xx,yy=xy(1.45);xx=np.round(xx+ix-margin).astype(int);yy=np.round(yy+iy-margin).astype(int)
            if not np.all(face_mask[yy,xx]>0):continue
        score=coverage+min(difference/140,1)-shift/10
        item=dict(center=q.tolist(),radius=float(r),ellipse_axes_px=[float(a),float(b)],ellipse_angle_deg=float(angle),confidence=float(min(1,.55*coverage+difference/180)),rim_coverage=coverage,contrast=difference,evidence='local_dark_aperture_bright_rim',insertion_target=False)
        if best is None or score>best[0]:best=(score,item)
    return None if best is None else best[1]


def _project(H,points):
    return cv2.perspectiveTransform(np.asarray(points,np.float32)[None],np.asarray(H,float))[0]


def detect_aux_holes(gray,face_polygon,large_holes=None,face_homography=None):
    """Return verified candidates tagged inner_threaded, boss_mounting or bearing.

    Without an independently fitted face H, radial class assignments are tentative
    image-layout hypotheses. No observation is automatically a metric H anchor.
    """
    gray=np.asarray(gray);poly=np.asarray(face_polygon,np.int32)
    if len(poly)<3:return []
    mask=np.zeros(gray.shape,np.uint8);cv2.fillPoly(mask,[poly],255)
    mom=cv2.moments(poly);center=np.array([mom['m10']/max(mom['m00'],1),mom['m01']/max(mom['m00'],1)])
    scale=np.sqrt(max(cv2.contourArea(poly),1)/np.pi)/26.49
    large=[]
    for p in ([] if large_holes is None else large_holes):
        large.append(np.asarray(p.get('center',p.get('center_px')) if isinstance(p,dict) else p,float))
    if len(large)>=3:
        # Opposite pair is furthest apart on the insertion pitch circle.
        pairs=[(np.linalg.norm(a-b),(a+b)/2) for i,a in enumerate(large) for b in large[i+1:]]
        span,center=max(pairs,key=lambda item:item[0]);scale=span/(2*17.9625)
    if face_homography is not None:
        center=_project(face_homography,[[0,0]])[0];basis=_project(face_homography,[[0,0],[.001,0],[0,.001]]);scale=np.sqrt(abs(np.linalg.det(np.stack([basis[1]-basis[0],basis[2]-basis[0]]))))
    scale=max(.6,float(scale))
    blurred=cv2.GaussianBlur(gray,(3,3),.55);edges=cv2.Canny(blurred,25,70);edges[mask==0]=0
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);observations=[]
    for contour in contours:
        if len(contour)<8:continue
        (x,y),(a,b),angle=cv2.fitEllipse(contour);r=np.sqrt(a*b)/2
        if min(a,b)/max(a,b)<.5 or not 1.4<r<15*scale:continue
        p=np.array([x,y]);radial=np.linalg.norm(p-center)/scale
        if face_homography is not None:
            radial=float(np.linalg.norm(_project(np.linalg.inv(np.asarray(face_homography,float)),[p])[0])*1000)
        if any(np.linalg.norm(p-q)<max(5,3.7*scale) for q in large):continue
        kind=None
        if radial<5 and 3.5*scale<r<13*scale:kind='central_bearing'
        elif 10<radial<15 and 1.4<r<2.4*scale:kind='inner_threaded'
        elif 18.5<radial<29 and 1.4<r<2.6*scale:kind='boss_mounting'
        if kind is None:continue
        result=refine_aperture(gray,p,r,max_shift=1.6,face_mask=mask)
        if result is None:continue
        if any(np.linalg.norm(np.array(result['center'])-o['center'])<max(2.5,min(result['radius'],o['radius'])) for o in observations):continue
        result.update(landmark_class=kind,type=kind,observed=True,plane='raised_boss' if kind=='boss_mounting' else 'face',height_above_face_mm_hypothesis=4. if kind=='boss_mounting' else 0.,classification_verified=False,layout_source='photo_profile_approximate',metric_homography_eligible=False,radial_distance_mm_hypothesis=float(radial))
        observations.append(result)
    return observations


def globalcentralbearingcandidates(gray):
    """Global concentric resolved ellipse evidence; no face ROI assumptions."""
    gray=np.asarray(gray);smooth=cv2.GaussianBlur(gray,(3,3),.6)
    contours,_=cv2.findContours(cv2.Canny(smooth,22,65),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);ellipses=[]
    for contour in contours:
        if len(contour)<28:continue
        (x,y),(a,b),angle=cv2.fitEllipse(contour);r=np.sqrt(a*b)/2
        if not 6<r<32 or min(a,b)/max(a,b)<.6:continue
        # An open arc can fit an arbitrary circle: require broad angular support.
        pts=contour[:,0,:].astype(float)-[x,y];angular=np.mod(np.arctan2(pts[:,1],pts[:,0]),2*np.pi);bins=np.unique((angular/(2*np.pi)*16).astype(int))
        if len(bins)<13:continue
        ellipses.append(dict(center=np.array([x,y]),radius=float(r),axes=[a,b]))
    results=[]
    for e in ellipses:
        cluster=[q for q in ellipses if np.linalg.norm(e['center']-q['center'])<3.5]
        radii=sorted(q['radius'] for q in cluster);distinct=[]
        for r in radii:
            if not distinct or r-distinct[-1]>1.5:distinct.append(r)
        if len(distinct)<2 or max(distinct)<12:continue
        center=np.mean([q['center'] for q in cluster],axis=0);x,y=np.round(center).astype(int);r=max(distinct)
        if min(x,y)<r+2 or x+r+2>=gray.shape[1] or y+r+2>=gray.shape[0]:continue
        yy,xx=np.ogrid[:gray.shape[0],:gray.shape[1]];d=np.hypot(xx-center[0],yy-center[1]);core=gray[d<min(distinct)*.5];ring=gray[(d>r*.75)&(d<r*1.15)]
        contrast=float(np.percentile(ring,75)-np.median(core))
        if contrast<12:continue
        result=dict(center=center.tolist(),radius=float(r),concentric_support=len(distinct),confidence=float(min(1,.3+len(distinct)*.12+contrast/300)),contrast=contrast,observed=True,type='central_bearing',insertion_target=False)
        if any(np.linalg.norm(np.array(o['center'])-center)<6 for o in results):continue
        results.append(result)
    return sorted(results,key=lambda q:q['confidence'],reverse=True)

# Conventional snake-case alias for integrations.
global_central_bearing_candidates=globalcentralbearingcandidates
