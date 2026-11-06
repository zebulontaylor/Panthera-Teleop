import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('mujoco')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pin_contact_sim import surface_heights,disk_support


def annulus(inner=.00417,outer=.02,height=.01725):
    angle=np.arange(65)*2*np.pi/64
    a=np.c_[np.cos(angle)*inner,np.sin(angle)*inner,np.full(65,height)]
    b=np.c_[np.cos(angle)*outer,np.sin(angle)*outer,np.full(65,height)]
    return np.array([triangle for i in range(64) for triangle in [[a[i],b[i],b[i+1]],[a[i],b[i+1],a[i+1]]]])


def test_actual_height_and_empty_bore():
    mesh=annulus()
    z=surface_heights(mesh,[[0,0],[.01,0],[.03,0]])
    assert np.isnan(z[0]) and np.isnan(z[2])
    assert abs(z[1]-.01725)<1e-10


def test_support_distinguishes_void_rim_and_stable_plate():
    mesh=annulus()
    assert not disk_support(mesh,[0,0])['supported']
    rim=disk_support(mesh,[.0005,0])
    assert rim['supported'] and not rim['stable']
    face=disk_support(mesh,[.007,0])
    assert face['supported'] and face['stable']


def test_tilted_cap_origin_raised_until_rim_touches_surface():
    square=np.array([[[-.1,-.1,.01725],[.1,-.1,.01725],[.1,.1,.01725]],[[-.1,-.1,.01725],[.1,.1,.01725],[-.1,.1,.01725]]])
    theta=np.deg2rad(30);rotation=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    support=disk_support(square,[0,0],rotation=rotation)
    assert support['supported']
    assert abs(support['z']-(.01725+.004*np.sin(theta)))<1e-10
    assert not support['stable']
