"""Offline paired-reference landmark model. No robot or motor interfaces."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from app.pin_insertion_cnn import HeatmapPinInsertionCNN, joint_features

SCHEMA = 'pin-insertion-paired-reference-v1'

class PinReferenceCNN(HeatmapPinInsertionCNN):
    def __init__(self):
        super().__init__()
        self.vision[0] = nn.Conv2d(9,24,5,2,2)

    @classmethod
    def from_rgb_model(cls, source):
        model=cls()
        state=copy.deepcopy(source.state_dict())
        weight=state['vision.0.weight']
        if weight.shape[1]!=3: raise ValueError('RGB heatmap ancestor required')
        expanded=weight.new_zeros((weight.shape[0],9,*weight.shape[2:]))
        expanded[:,:3]=weight
        state['vision.0.weight']=expanded
        model.load_state_dict(state)
        return model


def paired_tensor(current, reference, size, device='cpu'):
    """Normalize RGB pairs and append signed current-minus-reference in [-1,1]."""
    arrays=[]
    for value in (current,reference):
        value=np.asarray(value)
        if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3 or min(value.shape[:2])<16:
            raise ValueError('full-frame uint8 RGB pairs required')
        arrays.append(value)
    if arrays[0].shape!=arrays[1].shape: raise ValueError('aligned pair dimensions must match')
    tensors=[F.interpolate(torch.from_numpy(v.copy()).permute(2,0,1)[None].float()/255,
                           (size,size),mode='bilinear',align_corners=False) for v in arrays]
    return torch.cat((tensors[0],tensors[1],tensors[0]-tensors[1]),1).to(device)

class PinReferencePredictor:
    """Explicitly paired predictor; incompatible with the live RGB-only loader."""
    def __init__(self, checkpoint, device='cpu'):
        self.device=torch.device(device)
        payload=torch.load(checkpoint,map_location=self.device,weights_only=True)
        self.metadata=payload['metadata']
        if self.metadata.get('schema')!=SCHEMA: raise ValueError('paired-reference schema required')
        self.model=PinReferenceCNN().to(self.device)
        self.model.load_state_dict(payload['model']);self.model.eval()

    def predict(self,current,reference,state):
        x=paired_tensor(current,reference,self.metadata['image_size'],self.device)
        q=torch.tensor(joint_features(state),device=self.device)[None]
        with torch.inference_mode():out=self.model(x,q,return_logits=True)
        return {k:v[0].cpu().numpy().tolist() for k,v in out.items()}
