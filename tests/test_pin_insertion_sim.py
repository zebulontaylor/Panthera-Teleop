"""Pure geometry tests for projection and correction semantics."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
pytest.importorskip('mujoco')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pin_insertion_sim import project, correction, alignment_gate, depth_visibility, hole_gate_support


def test_legacy_scene_cannot_silently_generate_new_training_data(tmp_path):
    from pin_insertion_sim import generate
    with pytest.raises(ValueError, match='superseded'):
        generate([], tmp_path/'data.npz')


def test_camera_projection_axes_and_aspect():
    uv,depth=project([[0,0,-1],[1,1,-1]],np.zeros(3),np.eye(3),90,200,100)
    np.testing.assert_allclose(uv,[[.5,.5],[.75,0]],atol=1e-10)
    np.testing.assert_allclose(depth,[1,1])


def test_object_sign_and_right_base_frame():
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    pin=np.array([.003,0,0]);hole=np.zeros(3)
    np.testing.assert_allclose(correction(pin,hole,rotation,'pin'),[0,.003,0])
    np.testing.assert_allclose(correction(pin,hole,rotation,'carrier'),[0,-.003,0])


def test_step_bound_and_zero():
    np.testing.assert_allclose(correction([.1,0,0],[0,0,0],np.eye(3),'pin'),[-.004,0,0])
    np.testing.assert_array_equal(correction([0,0,0],[0,0,0],np.eye(3),'pin'),np.zeros(3))
    with pytest.raises(ValueError): correction([0,0,0],[0,0,0],np.eye(3),'robot')


def test_gate_requires_plane_height_and_visibility():
    hole=np.zeros(3);normal=np.array([0,0,1])
    assert alignment_gate([.005,0,0],hole,normal,[True,True])[0]
    assert not alignment_gate([.005,0,.008],hole,normal,[True,True])[0]
    assert not alignment_gate([.005,0,-.002],hole,normal,[True,True])[0]
    assert not alignment_gate([.0001,0,0],hole,normal,[True,True])[0]
    assert not alignment_gate([.02,0,0],hole,normal,[True,True])[0]
    assert not alignment_gate([.005,0,0],hole,normal,[True,False])[0]
    # Plane test is independent of world orientation.
    assert alignment_gate([0,.005,0],hole,[1,0,0],[True,True])[0]
    active,height,lateral=alignment_gate([.01,.005,0],hole,[1,0,0],[True,True])
    assert not active
    assert height == .01
    assert lateral == .005


def test_visibility_rejects_foreground_occlusion():
    actual=depth_visibility([.1,.1,.1,.1],[.1,.0995,.098,.094],[True,True,True,True])
    np.testing.assert_array_equal(actual,[True,True,False,False])
    assert not depth_visibility(.1,.1,False)
    assert depth_visibility(.1,.12,True)  # empty bore behind the mouth plane


def test_visibility_same_object_requires_explicit_segmentation_support():
    assert depth_visibility(.1,.09,True,same_object=True)
    assert not depth_visibility(.1,.09,True,same_object=False)
    assert not depth_visibility(.1,.09,False,same_object=True)


def test_gate_submillimeter_nudge_and_configurable_deadband():
    assert alignment_gate([.0003,0,0],[0,0,0],[0,0,1],True)[0]
    assert not alignment_gate([.00015,0,0],[0,0,0],[0,0,1],True)[0]
    assert not alignment_gate([.0003,0,0],[0,0,0],[0,0,1],True,min_nudge_m=.0004)[0]


def test_hole_gate_support_only_expected_pin_occlusion():
    assert hole_gate_support(True,False,True)
    assert hole_gate_support(False,True,True)
    assert not hole_gate_support(False,True,False)
    assert not hole_gate_support(False,False,True)
