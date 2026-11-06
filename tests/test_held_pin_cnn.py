"""Center-axis Y, native-detail crop and real-top-only supervision contracts."""
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from app.held_pin_cnn import HeldPinCNN,extract_native_roi,heatmap_loss,HeldPinPredictor,SCHEMA
from scripts.train_held_pin_cnn import load_real

def test_native_roi_preserves_pixel_geometry():
    torch.set_num_threads(2)
    yy,xx=torch.meshgrid((torch.arange(80)+.5)/80,(torch.arange(100)+.5)/100,indexing='ij')
    image=torch.stack((xx,yy,xx))[None];center=torch.tensor([[.65,.45]])
    crop,origin,scale=extract_native_roi(image,center,40,40)
    expected=origin[:,None,None,:]+torch.stack(torch.meshgrid((torch.arange(40)+.5)/40,(torch.arange(40)+.5)/40,indexing='ij'),-1).flip(-1)[None]*scale[:,None,None,:]
    torch.testing.assert_close(crop[:,0],expected[...,0],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(crop[:,1],expected[...,1],atol=1e-6,rtol=1e-6)

def test_hidden_nan_labels_cannot_affect_top_position_loss():
    model=HeldPinCNN();out=model(torch.rand(2,3,80,100));target=torch.rand(2);mask=torch.tensor([True,False])
    first=heatmap_loss(out['coarse'],target,mask);target[1]=float('nan')
    second=heatmap_loss(out['coarse'],target,mask);torch.testing.assert_close(first,second)
    second.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.coarse.parameters() if p.grad is not None)

def test_real_bottom_labels_and_group_leakage_are_rejected(tmp_path):
    row={'id':'x','top_supervision_valid':True,'group_id':'same','split':'train','bottom_supervision_valid':True}
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'frames':[row]}))
    with pytest.raises(ValueError,match='bottom ground truth'):load_real(p)
    row['bottom_supervision_valid']=False
    p.write_text(json.dumps({'frames':[row,dict(row,split='validation')]}))
    with pytest.raises(ValueError,match='group leakage'):load_real(p)

def test_predictor_fixed_x_and_inferred_bottom(tmp_path):
    model=HeldPinCNN();path=tmp_path/'axis.pt';meta={'schema':SCHEMA,'camera':'left_wrist','axis_x_fraction':.5234375,'visibility_threshold':0.}
    torch.save({'model':model.state_dict(),'metadata':meta},path)
    result=HeldPinPredictor(path).predict(np.zeros((80,100,3),np.uint8))
    assert result['top_px'][0]==52.34375 and result['bottom_px'][0]==52.34375
    assert result['pin_px']==result['bottom_px'] and result['bottom_inferred'] and not result['bottom_observed']
    assert result['top_detected'] and result['top_estimated'] and not result['top_observed']
    assert not result['top_visibility_verified'] and not result['automatic_nudge_eligible']
    meta['visibility_threshold']=1.;torch.save({'model':model.state_dict(),'metadata':meta},path)
    result=HeldPinPredictor(path).predict(np.zeros((80,100,3),np.uint8))
    assert result['pin_px'] is None and not result['top_observed']
    meta['camera']='right_wrist';torch.save({'model':model.state_dict(),'metadata':meta},path)
    with pytest.raises(ValueError):HeldPinPredictor(path)
