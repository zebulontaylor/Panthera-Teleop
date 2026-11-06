"""LEFT-only reuse of the original adapted multiview CNN's visual weights.

Only a single RGB frame is accepted. Fixed legacy LEFT identity is a constant,
not another camera input. Cross-view/context and proprio inputs are removed.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

VARIANT='legacy-adapted-left-local-visibility-v1'
DEFAULT_SOURCE=Path(__file__).resolve().parents[2]/'RobotArmTraining/outputs/panthera-assembly-cnns/pin_perception/carrier_v4/adapted.pt'

def _block(inside,outside,stride=1):
 groups=min(8,outside)
 while outside%groups:groups-=1
 return nn.Sequential(nn.Conv2d(inside,outside,3,stride,1,bias=False),nn.GroupNorm(groups,outside),nn.SiLU(),nn.Conv2d(outside,outside,3,padding=1,bias=False),nn.GroupNorm(groups,outside),nn.SiLU())

class LegacyLeftCarrierCNN(nn.Module):
 def __init__(self):
  super().__init__();self.stem=_block(8,16,2);self.encoder_quarter=_block(16,32,2);self.encoder_eighth=_block(32,64,2);self.encoder_sixteenth=_block(64,96,2);self.cross_view=nn.Sequential(nn.Linear(96,96),nn.SiLU());self.decoder_eighth=_block(256,64);self.decoder_quarter=_block(96,32);self.heatmap=nn.Conv2d(32,10,1);self.landmark_embedding=nn.Embedding(10,16);self.local_visibility=nn.Sequential(nn.Linear(32+96+16,64),nn.SiLU(),nn.Linear(64,1))
 def load_original_adapted(self,path=DEFAULT_SOURCE):
  path=Path(path);payload=torch.load(path,map_location='cpu',weights_only=True);source=payload['model_state'];state=self.state_dict();copied=[]
  for key,value in source.items():
   if key in state and state[key].shape==value.shape and not key.startswith('cross_view'):
    state[key]=value.clone();copied.append(key)
  # The learned model's LEFT branch is index1; no right/overhead/proprio data.
  state['cross_view.0.weight']=source['cross_view.0.weight'][:,96:192].clone();state['cross_view.0.bias']=source['cross_view.0.bias'].clone();copied+=['cross_view.0.weight[left96columns]','cross_view.0.bias']
  for new,old in [(1,2),(2,3),(3,4),(4,5),(9,0)]:
   state['heatmap.weight'][new]=source['heatmap.weight'][old];state['heatmap.bias'][new]=source['heatmap.bias'][old]
  self.load_state_dict(state)
  return dict(source_checkpoint=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),copied_parameters=copied,fixed_legacy_camera_identity='left_wrist index1',removed_inputs=['overhead','right_wrist','end_effector_positions'],hole_initialization='Old unordered bore slots copied then relearned with canonical simulator labels; physical hole ID remains rotationally ambiguous.')
 def forward(self,image,*,return_logits=False):
  b,_,height,width=image.shape;yy,xx=torch.meshgrid(torch.linspace(-1,1,height,device=image.device,dtype=image.dtype),torch.linspace(-1,1,width,device=image.device,dtype=image.dtype),indexing='ij');coordinates=torch.stack((xx,yy)).expand(b,-1,-1,-1);identity=torch.tensor([0.,1.,0.],device=image.device,dtype=image.dtype)[None,:,None,None].expand(b,-1,height,width);x=torch.cat(((image-.5)/.25,coordinates,identity),1);half=self.stem(x);quarter=self.encoder_quarter(half);eighth=self.encoder_eighth(quarter);sixteenth=self.encoder_sixteenth(eighth);context=self.cross_view(sixteenth.mean((-2,-1)));deep=F.interpolate(sixteenth,size=eighth.shape[-2:],mode='bilinear',align_corners=False);fused=context[:,:,None,None].expand(-1,-1,*eighth.shape[-2:]);x=self.decoder_eighth(torch.cat((eighth,deep,fused),1));x=F.interpolate(x,size=quarter.shape[-2:],mode='bilinear',align_corners=False);x=self.decoder_quarter(torch.cat((quarter,x),1));logits=self.heatmap(x);_,k,h,w=logits.shape;prob=logits.flatten(2).softmax(-1).reshape(b,k,h,w);xs=(torch.arange(w,device=x.device,dtype=x.dtype)+.5)/w;ys=(torch.arange(h,device=x.device,dtype=x.dtype)+.5)/h;points=torch.stack(((prob.sum(2)*xs).sum(-1),(prob.sum(3)*ys).sum(-1)),-1)
  # Sample a local3x3 patch around the predicted landmark; coordinates detached
  # so visibility cannot move a keypoint into a convenient visible patch.
  offsets=torch.tensor([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]],device=x.device,dtype=x.dtype)*torch.tensor([2/w,2/h],device=x.device,dtype=x.dtype);grid=points.detach()[:,:,None,:]*2-1+offsets[None,None];patch=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False).mean(-1).permute(0,2,1);embedding=self.landmark_embedding(torch.arange(k,device=x.device))[None].expand(b,-1,-1);visible=self.local_visibility(torch.cat((patch,context[:,None].expand(-1,k,-1),embedding),-1)).squeeze(-1);out=dict(keypoints=points,visibility_logits=visible)
  if return_logits:out['keypoint_logits']=logits
  return out

def balanced_visibility_loss(logits,visible):
 target=visible.to(logits.dtype);loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none');positive=target.sum(0);negative=(1-target).sum(0);pos=(loss*target).sum(0)/positive.clamp_min(1);neg=(loss*(1-target)).sum(0)/negative.clamp_min(1);classes=(positive>0).to(logits.dtype)+(negative>0).to(logits.dtype);return ((pos*(positive>0)+neg*(negative>0))/classes.clamp_min(1)).mean()

def targeted_occlusion(images,points,visible,generator=None):
 """Force varied TRAIN occlusion at known visible face points; no hidden XY loss."""
 x=images.clone();v=visible.bool().clone();b,_,h,w=x.shape
 def rand(*shape):return torch.rand(shape,device=x.device,dtype=x.dtype,generator=generator)
 choose=torch.randint(0,9,(b,),device=x.device,generator=generator);enabled=rand(b)<.65;size=.025+.07*rand(b,2);center=points[torch.arange(b,device=x.device),choose].detach();lo=center-size/2;hi=center+size/2;yy,xx=torch.meshgrid((torch.arange(h,device=x.device)+.5)/h,(torch.arange(w,device=x.device)+.5)/w,indexing='ij');mask=(xx[None]>=lo[:,0,None,None])&(xx[None]<=hi[:,0,None,None])&(yy[None]>=lo[:,1,None,None])&(yy[None]<=hi[:,1,None,None])&enabled[:,None,None];fill=.03+.65*rand(b,3,1,1);x=torch.where(mask[:,None],fill,x);covered=(points[:,:,0]>=lo[:,0,None])&(points[:,:,0]<=hi[:,0,None])&(points[:,:,1]>=lo[:,1,None])&(points[:,:,1]<=hi[:,1,None])&enabled[:,None];return x,points,v&~covered
