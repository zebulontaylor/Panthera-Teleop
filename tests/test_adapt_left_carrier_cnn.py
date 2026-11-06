import numpy as np
import torch
from scripts.adapt_left_carrier_cnn import masked_loss,assign

def test_unlabeled_real_points_have_no_visibility_or_location_gradient():
 logits=torch.zeros(1,10,8,8,requires_grad=True);visibility=torch.zeros(1,10,requires_grad=True);pred=torch.ones(1,10,2,requires_grad=True)*.5;target=torch.zeros(1,10,2);target[0,1]=torch.tensor([.3,.4]);visible=torch.zeros(1,10,dtype=torch.bool);visible[0,1]=True;known=visible.clone();known[0,2]=True;loss=masked_loss({'keypoints':pred,'keypoint_logits':logits,'visibility_logits':visibility},target,visible,known);loss.backward();assert visibility.grad[0,1]<0;assert visibility.grad[0,2]>0;assert torch.equal(visibility.grad[0,[0,3,4,5,6,7,8,9]],torch.zeros(8));assert torch.equal(logits.grad[0,0],torch.zeros(8,8));assert logits.grad[0,1].abs().sum()>0

def test_unordered_partial_bores_match_without_inventing_missing_or_other_labels():
 pred=torch.zeros(1,10,2);pred[0,1:5]=torch.tensor([[.2,.2],[.8,.2],[.8,.8],[.2,.8]]);items=[{'points':np.array([[.81,.79],[.19,.21],[.79,.19]])}];target,known=assign(pred,items);assert known.sum()==3;assert known[0,1] and known[0,2] and known[0,3];assert not known[0,4];assert not known[0,0];assert not known[0,5:].any();assert torch.allclose(target[0,1],torch.tensor([.19,.21]))
