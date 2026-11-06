"""Pure CPU contracts: no robot or camera required."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from app.pin_insertion_cnn import BoundedPinCorrection, PinInsertionCNN, SpatialPool, joint_features
from scripts.train_pin_insertion_cnn import group_split


class Predictor:
    def predict(self, image, state):
        return {'keypoints':[.2,.3,.4,.5], 'delta_xyz_m':[.01,0,0], 'active_logit':[8.]}


def test_spatial_pool_matches_adaptive_averages_and_gradient():
    value = torch.randn(2, 3, 10, 11, requires_grad=True)
    expected = torch.nn.functional.adaptive_avg_pool2d(value, (4, 4))
    actual = SpatialPool()(value)
    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    gradient = value.grad.clone()
    value.grad = None
    actual.sum().backward()
    torch.testing.assert_close(value.grad, gradient)


def test_joint_layout_and_model():
    assert joint_features(np.arange(14)).tolist() == list(range(6))+list(range(7,13))
    out=PinInsertionCNN()(torch.zeros(2,3,64,64),torch.zeros(2,12))
    assert out['keypoints'].shape == (2,4)
    assert out['delta_xyz_m'].shape == (2,3)
    assert out['active_logit'].shape == (2,1)


def test_group_split_never_leaks():
    groups=np.repeat(np.arange(10),5)
    train,val=group_split(groups,2)
    assert not set(groups[train]) & set(groups[val])
    assert len(train)+len(val)==len(groups)
    assert np.array_equal(val,group_split(groups,2)[1])


def test_shadow_persistence_cap_and_stale():
    adapter=BoundedPinCorrection(Predictor())
    image=np.zeros((32,32,3),dtype=np.uint8)
    for stamp in (10_000_000,20_000_000,30_000_000):
        result=adapter.propose(image,np.zeros(14),frame_ns=stamp,measured_state_ns=stamp,now_ns=stamp)
    assert result['eligible']
    assert result['delta_xyz_m']==[0.,0.,0.]
    assert np.linalg.norm(result['proposed_delta_xyz_m']) <= .0005
    assert result['controlled_arm']=='left'
    result=adapter.propose(image,np.zeros(14),frame_ns=40_000_000,measured_state_ns=40_000_000,now_ns=400_000_000)
    assert not result['eligible']
    assert result['delta_xyz_m']==[0.,0.,0.]


def test_invalid_predictions_fail_closed():
    predictor=Predictor()
    predictor.predict=lambda *_: {'keypoints':[float('nan')]*4,'delta_xyz_m':[0]*3,'active_logit':[8]}
    adapter=BoundedPinCorrection(predictor,enabled=True,persistence=1)
    result=adapter.propose(None,np.zeros(14),frame_ns=1,measured_state_ns=1,now_ns=1)
    assert not result['eligible']
    assert result['delta_xyz_m']==[0.,0.,0.]
    with pytest.raises(ValueError): BoundedPinCorrection(Predictor(),max_delta_m=.001)


def test_both_arms_required_and_nonfinite_left_rejected():
    with pytest.raises(ValueError): joint_features(np.zeros(6))
    state=np.zeros(14); state[0]=np.nan
    adapter=BoundedPinCorrection(Predictor(),enabled=True,persistence=1)
    result=adapter.propose(None,state,frame_ns=1,measured_state_ns=1,now_ns=1)
    assert not result['eligible']
    assert result['delta_xyz_m']==[0.,0.,0.]


def test_dataset_frame_and_control_semantics():
    from scripts.train_pin_insertion_cnn import validate_dataset_metadata
    metadata={'schema_version':1,'generator':'pin_insertion_sim.py','camera':'right_wrist',
        'controlled_object':'pin','controlled_arm':'left','delta_frame':'right_native_base',
        'delta_units':'meters','state_order':'left joints1..6,gripper,right joints1..6,gripper (radians)',
        'keypoints_order':['pin_tip_u','pin_tip_v','target_hole_u','target_hole_v']}
    validate_dataset_metadata(metadata)
    for key,value in [('controlled_object','carrier'),('controlled_arm','right'),
                      ('delta_frame','left_base'),('delta_units','mm'),('camera','left_wrist')]:
        with pytest.raises(ValueError,match='semantic mismatch'):
            validate_dataset_metadata(dict(metadata,**{key:value}))


def test_group_split_balances_validation_when_feasible():
    groups=np.repeat(np.arange(10),4)
    labels=np.tile([0,0,1,1],10)
    train,val=group_split(groups,labels=labels)
    assert set(labels[train])=={0,1}
    assert set(labels[val])=={0,1}
    assert not set(groups[train]) & set(groups[val])


def test_heatmap_shape_gradient_and_masking():
    from app.pin_insertion_cnn import build_model
    from scripts.train_pin_insertion_cnn import heatmap_supervision_loss
    model=build_model('heatmap')
    image=torch.zeros(2,3,48,64)
    joints=torch.zeros(2,12)
    out=model(image,joints,return_logits=True)
    assert out['keypoints'].shape==(2,4)
    assert out['keypoint_logits'].shape==(2,2,12,16)
    assert out['delta_xyz_m'].shape==(2,3)
    assert out['active_logit'].shape==(2,1)
    targets=torch.full((2,4),.5)
    loss=heatmap_supervision_loss(out['keypoint_logits'],targets,torch.ones(2,4))
    loss.backward()
    assert model.score[-1].weight.grad is not None
    assert torch.isfinite(model.score[-1].weight.grad).all()
    assert heatmap_supervision_loss(out['keypoint_logits'].detach(),targets,torch.zeros(2,4)).item()==0
    assert set(model(image,joints))=={'keypoints','delta_xyz_m','active_logit'}


def test_predictor_rejects_silent_square_crop():
    from app.pin_insertion_cnn import PinInsertionPredictor
    predictor = object.__new__(PinInsertionPredictor)
    predictor.metadata = {'dataset_metadata': {'image_width': 256, 'image_height': 192}}
    with pytest.raises(ValueError, match='uncropped'):
        predictor.predict(np.zeros((256, 256, 3), dtype=np.uint8), np.zeros(14))


def test_localization_only_checkpoint_never_exposes_motion():
    predictor = Predictor()
    predictor.metadata = {'direct_xyz_disabled': True}
    adapter = BoundedPinCorrection(predictor, enabled=True, persistence=1)
    result = adapter.propose(None, np.zeros(14), frame_ns=1, measured_state_ns=1, now_ns=1)
    assert not result['eligible']
    assert result['delta_xyz_m'] == result['proposed_delta_xyz_m'] == [0., 0., 0.]
    assert 'xyz_disabled' in result['reason']
    assert 'keypoints' in result
