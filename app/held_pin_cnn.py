"""Left-wrist pin top/bottom Y localization on a configured gripper center axis.

Top means the near/proximal cap center, bottom the distal endpoint. Bottom is an
amodal estimate under the user-specified center-axis assumption, never presented
as an observed real distal-tip label. No image-dependent X coordinate is output.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
SCHEMA='held-pin-left-center-axis-y-v2'

class VerticalHeatmap(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision=nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.ReLU(),nn.Conv2d(24,48,3,2,1),nn.ReLU(),
            nn.Conv2d(48,64,3,1,1),nn.ReLU(),nn.Conv2d(64,64,3,1,2,dilation=2),nn.ReLU(),
            nn.Conv2d(64,64,3,1,4,dilation=4),nn.ReLU())
        self.score=nn.Conv2d(64,1,1)
        self.visibility=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,1))
        self.offset=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,image):
        f=self.vision(image);logits=self.score(f).mean(-1)[:,0];prob=logits.softmax(-1)
        ys=(torch.arange(logits.shape[-1],device=image.device,dtype=image.dtype)+.5)/logits.shape[-1]
        return {'y':(prob*ys).sum(-1),'logits':logits,'visibility_logits':self.visibility(f).flatten(),
                'offset':self.offset(f).flatten().tanh()*.20}

def extract_native_roi(images,centers,roi_size_px=160,output_size=128):
    b,_,h,w=images.shape
    scale=images.new_tensor([roi_size_px/w,roi_size_px/h])[None].expand(b,-1)
    origin=centers-scale/2
    theta=images.new_zeros((b,2,3));theta[:,0,0]=scale[:,0];theta[:,1,1]=scale[:,1]
    theta[:,:,2]=centers*2-1
    grid=F.affine_grid(theta,(b,3,output_size,output_size),align_corners=False)
    return F.grid_sample(images,grid,align_corners=False,padding_mode='border'),origin,scale

class HeldPinCNN(nn.Module):
    def __init__(self):
        super().__init__();self.coarse=VerticalHeatmap();self.refiner=VerticalHeatmap()
    def forward(self,images,*,roi_y=None,axis_x=.5,roi_size_px=160):
        coarse=self.coarse(F.interpolate(images,(120,160),mode='bilinear',align_corners=False))
        y=coarse['y'].detach() if roi_y is None else roi_y
        axis=torch.as_tensor(axis_x,device=images.device,dtype=images.dtype).expand(len(images))
        centers=torch.stack((axis,y),-1)
        crop,origin,scale=extract_native_roi(images,centers,roi_size_px)
        fine=self.refiner(crop);top=origin[:,1]+fine['y']*scale[:,1]
        return {'coarse':coarse,'fine':fine,'top_y':top,'bottom_y':top+fine['offset'],
                'origin':origin,'scale':scale}

def heatmap_loss(out,target,mask,sigma=.025):
    mask=mask.to(target.dtype);target=torch.where(mask.bool(),target,torch.zeros_like(target))
    h=out['logits'].shape[-1];ys=(torch.arange(h,device=target.device)+.5)/h
    distribution=(-(ys-target[:,None]).square()/(2*sigma*sigma)).softmax(-1)
    ce=-(distribution*out['logits'].log_softmax(-1)).sum(-1)
    return ((ce+20*(out['y']-target).square())*mask).sum()/mask.sum().clamp_min(1)

class HeldPinPredictor:
    def __init__(self,checkpoint,device='cpu'):
        self.device=torch.device(device);payload=torch.load(checkpoint,map_location=self.device,weights_only=True)
        self.metadata=payload['metadata']
        if self.metadata.get('schema')!=SCHEMA or self.metadata.get('camera')!='left_wrist':raise ValueError('left center-axis Y checkpoint required')
        self.model=HeldPinCNN().to(self.device);self.model.load_state_dict(payload['model']);self.model.eval()
    def predict(self,image):
        rgb=np.asarray(image)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<32:raise ValueError('native full-frame RGB uint8 required')
        x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().to(self.device)/255
        axis=float(self.metadata.get('axis_x_fraction',.5))
        with torch.inference_mode():out=self.model(x,axis_x=axis,roi_size_px=self.metadata.get('roi_size_px',160))
        confidence=float(torch.minimum(out['coarse']['visibility_logits'].sigmoid(),out['fine']['visibility_logits'].sigmoid())[0])
        top=float(out['top_y'][0]);bottom=float(out['bottom_y'][0]);h,w=rgb.shape[:2]
        supported=bool(confidence>=self.metadata.get('visibility_threshold',.6) and np.isfinite([top,bottom]).all() and 0<=top<=1 and 0<=bottom<=1)
        top_px=[axis*w,top*h] if supported else None;bottom_px=[axis*w,bottom*h] if supported else None
        return {'top_px':top_px,'bottom_px':bottom_px,'pin_px':bottom_px,'top_observed':False,'top_detected':supported,'top_estimated':supported,'top_visibility_verified':False,
                'bottom_inferred':supported,'confidence':confidence,'pin_confidence':confidence,'visible':supported,
                'center_axis_x':axis*w,'camera':'left_wrist','bottom_assumption':'pin lies on configured gripper center axis; distal endpoint inferred from sim learned projected length',
                'bottom_observed':False,'confidence_calibrated':False,'visibility_gate_validated':bool(self.metadata.get('visibility_gate_validated',False)),
                'automatic_nudge_eligible':bool(self.metadata.get('automatic_nudge_eligible',False))}
