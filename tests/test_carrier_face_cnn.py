"""RGB-only carrier contract and visible-only geometry/loss checks."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from app.carrier_face_cnn import (CarrierFaceCNN,carrier_face_loss,fit_face_geometry,
    project_face,FACE_POINTS_MM,CarrierFacePredictor,SCHEMA)

def test_hidden_coordinates_do_not_supervise_locations():
    torch.set_num_threads(2)
    model=CarrierFaceCNN()
    out=model(torch.rand(2,3,48,48),return_logits=True)
    visible=torch.zeros(2,10);visible[:,1]=1
    points=torch.rand(2,10,2)
    first,_=carrier_face_loss(out,points,visible)
    points[visible==0]=float('nan')
    second,_=carrier_face_loss(out,points,visible)
    torch.testing.assert_close(first,second)
    second.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

def test_supported_geometry_extrapolates_only_known_layout():
    H=np.array([[.009,.001,.5],[.001,.007,.4],[.001,-.002,1.]])
    points=project_face(H)
    confidence=np.ones(9);confidence[2:4]=0
    geometry=fit_face_geometry(points,confidence)
    assert geometry is not None
    np.testing.assert_allclose(geometry['projected'],points,atol=1e-6)
    assert 2 not in geometry['support']
    assert fit_face_geometry(points,np.array([1,1,1,0,0,0,0,0,0])) is None

def test_predictor_rejects_other_camera_and_uses_rgb_only(tmp_path):
    path=tmp_path/'carrier.pt';model=CarrierFaceCNN()
    metadata={'schema':SCHEMA,'camera':'right_wrist','image_size':32}
    torch.save({'metadata':metadata,'model':model.state_dict()},path)
    with pytest.raises(ValueError):CarrierFacePredictor(path)
    metadata['camera']='left_wrist';metadata['visibility_threshold']=1.
    torch.save({'metadata':metadata,'model':model.state_dict()},path)
    predictor=CarrierFacePredictor(path)
    result=predictor.predict(np.zeros((48,64,3),np.uint8),state='ignored')
    assert result['holes_px']==[None]*4
    assert not result['supported'] and result['pin_px'] is None
    assert result['camera']=='left_wrist'

def test_collinear_triple_and_projective_horizon_are_rejected():
    H=np.array([[.009,0,.5],[0,.007,.4],[0,0,1.]])
    points=project_face(H)
    # center, +X bore, +X rim lie on one line, fourth point cannot rescue DLT.
    confidence=np.zeros(9);confidence[[0,1,5,2]]=1
    assert fit_face_geometry(points,confidence) is None
    points[:]=.5
    assert fit_face_geometry(points,np.ones(9)) is None
