"""Left-wrist RGB landmarks and explicit, support-gated planar extrapolation.

No joint states, right camera, simulator state, or direct motor commands are inputs.
Landmark coordinates are normalized full-frame image coordinates. Hidden landmarks
never contribute to the location loss; inferred holes come only from fitted geometry.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'carrier-face-left-rgb-v1'
LANDMARK_NAMES = ('face_center', *(f'bore_{i}' for i in range(4)), *(f'rim_{i}' for i in range(4)), 'pin_tip')
_angles = np.arange(4) * np.pi / 2
FACE_POINTS_MM = np.concatenate((np.zeros((1, 2)),
    np.stack((np.cos(_angles), np.sin(_angles)), 1)*17.9625,
    np.stack((np.cos(_angles), np.sin(_angles)), 1)*26.49)).astype(np.float32)

class CarrierFaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.ReLU(),
            nn.Conv2d(24,48,3,2,1),nn.ReLU(),nn.Conv2d(48,64,3,1,1),nn.ReLU(),
            nn.Conv2d(64,64,3,1,2,dilation=2),nn.ReLU(),
            nn.Conv2d(64,64,3,1,4,dilation=4),nn.ReLU())
        self.score = nn.Conv2d(64,10,1)
        self.visibility = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,10))

    def forward(self,image,*,return_logits=False):
        visual=self.vision(image);logits=self.score(visual)
        b,k,h,w=logits.shape
        p=logits.flatten(2).softmax(-1).reshape(b,k,h,w)
        xs=(torch.arange(w,device=image.device,dtype=image.dtype)+.5)/w
        ys=(torch.arange(h,device=image.device,dtype=image.dtype)+.5)/h
        points=torch.stack(((p.sum(2)*xs).sum(-1),(p.sum(3)*ys).sum(-1)),-1)
        out={'keypoints':points,'visibility_logits':self.visibility(visual)}
        if return_logits:out['keypoint_logits']=logits
        return out

def carrier_face_loss(out,keypoints,visible,*,sigma=.025):
    """Visible-only spatial CE and coordinates; visibility learns all examples."""
    logits=out['keypoint_logits'];b,k,h,w=logits.shape
    visible=visible.to(logits.dtype)
    # Sanitize masked coordinates before arithmetic: NaN hidden labels are allowed.
    target=torch.where(visible[...,None].bool(),keypoints,torch.zeros_like(keypoints))
    yy,xx=torch.meshgrid((torch.arange(h,device=logits.device)+.5)/h,
                         (torch.arange(w,device=logits.device)+.5)/w,indexing='ij')
    squared=(xx-target[...,0,None,None])**2+(yy-target[...,1,None,None])**2
    distribution=(-squared/(2*sigma*sigma)).flatten(2).softmax(-1)
    ce=-(distribution*logits.flatten(2).log_softmax(-1)).sum(-1)
    denominator=visible.sum().clamp_min(1)
    location=(ce*visible).sum()/denominator
    coordinate=(((out['keypoints']-target)**2).sum(-1)*visible).sum()/denominator
    visibility=F.binary_cross_entropy_with_logits(out['visibility_logits'],visible)
    total=location+20*coordinate+visibility
    return total,{'heatmap':location.detach(),'coordinate':coordinate.detach(),'visibility':visibility.detach()}

def project_face(homography,points=FACE_POINTS_MM):
    points=np.asarray(points,dtype=float)
    projected=np.column_stack((points,np.ones(len(points))))@np.asarray(homography).T
    if np.any(np.abs(projected[:,2])<1e-9):raise ValueError('face crosses projective horizon')
    return projected[:,:2]/projected[:,2,None]

def fit_face_geometry(points,confidence,*,threshold=.6,max_residual=.035):
    """RANSAC homography from observed landmarks, rejecting weak/degenerate support."""
    import itertools
    points=np.asarray(points,float);confidence=np.asarray(confidence,float)
    mask=(confidence>=threshold)&np.isfinite(points).all(1)&(points>=0).all(1)&(points<=1).all(1)
    ids=np.flatnonzero(mask)
    if len(ids)<4:return None
    source=FACE_POINTS_MM[ids].astype(np.float64)
    if np.linalg.svd(source-source.mean(0),compute_uv=False)[1]<4:return None
    def dlt(selected):
        rows=[]
        for (x,y),(u,v) in zip(FACE_POINTS_MM[selected]/26.49,points[selected]):
            rows.extend(([-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]))
        rows=np.asarray(rows)
        if np.linalg.matrix_rank(rows,tol=1e-8)<8:return None
        _,_,vt=np.linalg.svd(rows,full_matrices=True)
        return vt[-1].reshape(3,3) @ np.diag([1/26.49,1/26.49,1.])
    best=None;best_score=(-1,-np.inf)
    for subset in itertools.combinations(ids,4):
        candidate=dlt(list(subset))
        if candidate is None:continue
        try:errors=np.linalg.norm(project_face(candidate)-points,axis=1)
        except ValueError:continue
        selected=ids[errors[ids]<=max_residual]
        score=(len(selected),-float(errors[selected].sum()))
        if score>best_score:best_score=score;best=selected
    if best is None or len(best)<max(4,int(np.ceil(.7*len(ids)))):return None
    selected=best;H=dlt(selected)
    if H is None:return None
    # A collinear triple among exactly four correspondences leaves DLT deficient.
    rows=[]
    for (x,y),(u,v) in zip(FACE_POINTS_MM[selected]/26.49,points[selected]):
        rows.extend(([-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]))
    if np.linalg.matrix_rank(np.asarray(rows),tol=1e-8)<8:return None
    denominator=np.column_stack((FACE_POINTS_MM,np.ones(9)))@H[2]
    if not ((denominator>1e-8).all() or (denominator < -1e-8).all()):return None
    try: projected=project_face(H)
    except ValueError:return None
    if not np.isfinite(projected).all():return None
    residual=float(np.max(np.linalg.norm(projected[selected]-points[selected],axis=1)))
    # Reject collapsed or wildly extrapolated projected faces.
    polygon=projected[5:9].astype(np.float32)
    area=abs(float(np.dot(polygon[:,0],np.roll(polygon[:,1],-1))-np.dot(polygon[:,1],np.roll(polygon[:,0],-1))))/2
    edges=np.roll(polygon,-1,axis=0)-polygon
    turns=edges[:,0]*np.roll(edges[:,1],-1)-edges[:,1]*np.roll(edges[:,0],-1)
    convex=(turns>0).all() or (turns<0).all()
    if not convex or residual>max_residual or not .0002<area<1.5 or (np.abs(projected) > 3).any():return None
    return {'homography':H.tolist(),'projected':projected.tolist(),'support':selected.tolist(),'residual':residual}

class CarrierFacePredictor:
    def __init__(self,checkpoint,device='cpu'):
        self.device=torch.device(device)
        payload=torch.load(checkpoint,map_location=self.device,weights_only=True)
        self.metadata=payload['metadata']
        if self.metadata.get('schema')!=SCHEMA or self.metadata.get('camera')!='left_wrist':
            raise ValueError('left-wrist RGB carrier schema required')
        if self.metadata.get('model_variant')=='legacy-adapted-left-local-visibility-v1':
            from app.carrier_face_legacy_cnn import LegacyLeftCarrierCNN
            self.model=LegacyLeftCarrierCNN().to(self.device)
        else:self.model=CarrierFaceCNN().to(self.device)
        self.model.load_state_dict(payload['model']);self.model.eval()

    def predict(self,image,state=None):
        rgb=np.asarray(image)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<16:
            raise ValueError('full-frame RGB uint8 image required')
        x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().to(self.device)/255
        x=F.interpolate(x,(self.metadata.get('image_size',128),)*2,mode='bilinear',align_corners=False)
        with torch.inference_mode():out=self.model(x)
        points=out['keypoints'][0].cpu().numpy();conf=out['visibility_logits'][0].sigmoid().cpu().numpy()
        threshold=float(self.metadata.get('visibility_threshold',.6))
        thresholds=np.asarray(self.metadata.get('visibility_thresholds',[threshold]*10),dtype=float)
        if thresholds.shape!=(10,):raise ValueError('ten visibility thresholds required')
        thresholds=np.maximum(thresholds,float(self.metadata.get('visibility_threshold_floor',0.)))
        geometry=fit_face_geometry(points[:9],conf[:9],threshold=thresholds[:9])
        holes=[]
        for i in range(4):
            observed=bool(conf[i+1]>=thresholds[i+1] and np.isfinite(points[i+1]).all() and
                          (points[i+1]>=0).all() and (points[i+1]<=1).all())
            if observed and geometry is not None:
                observed=bool(np.linalg.norm(points[i+1]-geometry['projected'][i+1]) <= .035)
            uv=points[i+1].tolist() if observed else (geometry['projected'][i+1] if geometry else None)
            holes.append({'index':i,'uv':uv,'confidence':float(conf[i+1]),'observed':observed,
                          'inferred':bool(not observed and geometry is not None)})
        height,width=rgb.shape[:2]
        Hpixels=(np.diag([width,height,1.]) @ np.asarray(geometry['homography']) @ np.diag([1000.,1000.,1.])).tolist() if geometry else None
        return {'holes_px':[(np.asarray(hole['uv'])*[width,height]).tolist() if hole['uv'] is not None else None for hole in holes],
                'visible':[hole['observed'] for hole in holes],
                'pin_px':(points[9]*[width,height]).tolist() if conf[9]>=thresholds[9] else None,
                'pin_confidence':float(conf[9]),'homography':Hpixels,'supported':geometry is not None,
                'holes':holes,'landmarks':points.tolist(),'visibility':conf.tolist(),
                'geometry_valid':geometry is not None,'face_homography':geometry['homography'] if geometry else None,
                'geometry_support':geometry['support'] if geometry else [],'camera':'left_wrist'}
