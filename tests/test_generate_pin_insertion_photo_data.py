from types import SimpleNamespace
import numpy as np
import pytest
from scripts.generate_pin_insertion_photo_data import (project,landmarks,visibility,bounded_action,classify_snapshot)


def test_projection_preserves_vertical_fov_and_aspect():
    uv,depth=project([[0,0,-1],[1,1,-1]],np.zeros(3),np.eye(3),90,200,100)
    np.testing.assert_allclose(uv,[[.5,.5],[.75,0]],atol=1e-10)
    np.testing.assert_allclose(depth,[1,1])


def test_target_is_pin2_specific_upper_bore_and_transforms_with_carrier():
    rotation=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    sim=SimpleNamespace(manifest={'parts':{'pin_2':{'assembled_origin':[1.01,2.02,3.]},'large_carrier':{'assembled_origin':[1,2,3]}}})
    position=np.array([[.2,.3,.4],[.5,.6,.7]])
    env=SimpleNamespace(sim=sim,pin_name='pin_2',pin_id=0,carrier_id=1,data=SimpleNamespace(xpos=position,xmat=np.stack([np.eye(3).flatten(),rotation.flatten()])))
    points,relative=landmarks(env)
    np.testing.assert_allclose(points[0],position[0])
    np.testing.assert_allclose(points[1],position[1]+rotation@np.array([.01,.02,.01725]))
    np.testing.assert_allclose(relative,rotation.T@(points[0]-points[1]))


def test_visibility_keeps_own_pin_amodal_tip_separate_from_hidden_hole():
    depth_image=np.full((10,10),.9)
    seg=np.zeros((10,10,2),int);seg[5,5]=[7,5]
    uv=np.array([[.5,.5],[.5,.5]])
    direct,valid,amodal,own=visibility(uv,np.array([1.,1.]),depth_image,seg,{7})
    assert not direct.any()
    assert valid.tolist()==[True,False]
    assert amodal.tolist()==[True,True]
    assert own.all()


def test_motion_is_bounded_hold_or_left_descent_only():
    info={'pinched':True,'holder_pinched':True,'carrier_supported':True}
    np.testing.assert_array_equal(bounded_action('hold',0,info),np.zeros(14))
    action=bounded_action('descend',0,info,.2)
    assert action[2]==pytest.approx(-.2)
    assert np.count_nonzero(action)==1
    assert not bounded_action('descend',0,{}).any()
    with pytest.raises(ValueError):bounded_action('descend',0,info,.3)


def test_no_contact_positive_from_proximity_or_nonphysical_support():
    info={'pinched':True,'holder_pinched':True,'carrier_supported':True}
    assert not classify_snapshot(np.array([.005,0,0]),[],info)[1]
    contact=[{'other_body':'large_carrier','normal_force_n':1.}]
    regime,active=classify_snapshot(np.array([.005,0,0]),contact,info)
    assert regime=='carrier_rim_contact' and active
    assert not classify_snapshot(np.array([.005,0,0]),contact,{})[1]
    assert not classify_snapshot(np.array([.005,0,.01]),contact,info)[1]


def test_aperture_visibility_is_separate_from_center_point_label():
    from scripts.generate_pin_insertion_photo_data import aperture_visibility_fraction
    env=SimpleNamespace(carrier_id=0,data=SimpleNamespace(xmat=np.array([np.eye(3).flatten()]),
        cam_xpos=np.zeros((1,3)),cam_xmat=np.array([np.eye(3).flatten()])),model=SimpleNamespace(cam_fovy=np.array([90.])))
    assert aperture_visibility_fraction(env,np.array([0,0,-1.]),0,640,480,np.full((480,640),2.))==1
    assert aperture_visibility_fraction(env,np.array([0,0,-1.]),0,640,480,np.full((480,640),.5))==0


def test_descend_to_contact_latches_hold_without_right_or_gripper_motion():
    info={'pinched':True,'holder_pinched':True,'carrier_supported':True}
    action=bounded_action('descend-to-contact',0,info,.2)
    assert action[2]==pytest.approx(-.2)
    info['_descent_contact_latched']=True
    assert not bounded_action('descend-to-contact',1,info,.2).any()


def test_camera_registration_cannot_change_physics_or_part_rest_poses():
    from scripts.generate_pin_insertion_photo_data import apply_camera_registration_checked
    env=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(2),qvel=np.zeros(2),ctrl=np.zeros(2),act=np.zeros(0)),
        model=SimpleNamespace(body_pos=np.zeros((1,3)),body_quat=np.zeros((1,4)),geom_pos=np.zeros((1,3)),
            geom_quat=np.zeros((1,4)),geom_rgba=np.ones((1,4)),cam_pos=np.zeros((1,3))))
    def optics_only(env,path):env.model.cam_pos[0,0]=.1;return {'path':path}
    assert apply_camera_registration_checked(env,'registration.json',optics_only)=={'path':'registration.json'}
    def bad(env,path):env.data.qpos[0]=1
    with pytest.raises(ValueError,match='protected physics'):apply_camera_registration_checked(env,'bad.json',bad)


def test_regime_proxy_uses_verified_824mm_bore_not_success_criterion():
    from scripts.generate_pin_insertion_photo_data import BORE_RADIUS_M,PIN_RADIUS_M
    assert BORE_RADIUS_M==pytest.approx(.00412)
    info={'pinched':True,'holder_pinched':True,'carrier_supported':True}
    contact=[{'other_body':'large_carrier','normal_force_n':1.}]
    #0.15mm lies outside nominal0.12mm radial gap, despite being below the
    #separate0.17mm task-success axis-error criterion. Contact evidence required.
    assert classify_snapshot(np.array([.00015,0,0]),contact,info)==('carrier_rim_contact',True)
    assert classify_snapshot(np.array([.00010,0,0]),contact,info)==('aligned_contact',False)
    assert classify_snapshot(np.array([.00813,0,0]),contact,info)==('carrier_surface_outside_hole',True)
