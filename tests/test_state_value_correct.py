"""Correction objectives and leakage guards, independent of image model training."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from state_value_correct import objective,read_pairs,anchors


def test_preference_gradient_has_label_direction():
    a=torch.tensor([0.,0.],requires_grad=True);b=torch.tensor([0.,0.],requires_grad=True)
    loss=objective(a,b,torch.tensor([1.,0.]));loss.backward()
    assert a.grad[0]<0 and b.grad[0]>0
    assert a.grad[1]>0 and b.grad[1]<0
    assert objective(torch.tensor([3.]),torch.tensor([-3.]),torch.tensor([1.])) < objective(torch.tensor([-3.]),torch.tensor([3.]),torch.tensor([1.]))


def write_pairs(tmp_path,rows):
    path=tmp_path/'pairs.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows));return path


def pair(tmp_path,identity='e',split='train'):
    paths=[]
    for i in range(6):
        p=tmp_path/f'{identity}-{i}.png';p.touch();paths.append(str(p))
    return dict(id=identity,episode_id=identity,session_id=identity,split=split,label=1,a=paths[:3],b=paths[3:],t_a=2,t_b=1)


def test_session_leakage_rejected(tmp_path):
    a=pair(tmp_path);b=pair(tmp_path,'other','val');b['session_id']=a['session_id']
    with pytest.raises(ValueError,match='session_id split leakage'):read_pairs(write_pairs(tmp_path,[a,b]))


def test_image_leakage_rejected(tmp_path):
    a=pair(tmp_path);b=pair(tmp_path,'other','test');b['a']=a['a']
    with pytest.raises(ValueError,match='image split leakage'):read_pairs(write_pairs(tmp_path,[a,b]))


def test_nonfinite_timestamp_rejected(tmp_path):
    a=pair(tmp_path);a['t_a']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):read_pairs(write_pairs(tmp_path,[a]))


def test_time_does_not_define_label(tmp_path):
    a=pair(tmp_path);assert read_pairs(write_pairs(tmp_path,[a]))[0]['label']==1


def test_fusion_training_keeps_backbone_frozen():
    from torch import nn
    from state_value_correct_representation import BackboneEncoder,FusionValue
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.feature_dim=2;self.backbone=nn.Sequential(nn.Flatten(),nn.Linear(12,2));self.fuse=nn.Linear(6,4);self.value=nn.Linear(4,1)
    model=Tiny()
    for p in model.parameters():p.requires_grad_(False)
    encoder=BackboneEncoder(model);head=FusionValue(model)
    with torch.no_grad():z=encoder.encode(torch.randn(4,3,3,2,2))
    before=[p.clone() for p in model.backbone.parameters()]
    for p in head.parameters():p.requires_grad_(True)
    opt=torch.optim.Adam(head.parameters(),lr=.1)
    loss=objective(head(z[:2]).flatten(),head(z[2:]).flatten(),torch.ones(2));loss.backward();opt.step()
    assert all(torch.equal(a,b) for a,b in zip(before,model.backbone.parameters()))
    assert all(p.grad is not None for p in model.fuse.parameters())


def test_backbone_adaptation_freezes_early_layers_and_bn():
    from torch import nn
    from state_value_correct_backbone import trainable_layers
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=nn.Module();self.backbone.layer1=nn.Linear(2,2);self.backbone.layer4=nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2));self.fuse=nn.Linear(2,2);self.value=nn.Linear(2,1)
    model=Tiny();trainable_layers(model)
    assert not any(p.requires_grad for p in model.backbone.layer1.parameters())
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())
    assert all(p.requires_grad for p in model.fuse.parameters())
    assert all(p.requires_grad for p in model.value.parameters())
    assert not model.backbone.layer4[1].training
