import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from pin_alignment_motion import NativeToolKinematics, plan_left_nudge
from pin_regrasp_motion import rotation_log


def state():
    q = [.2,1.,1.4,.3,.4,.2]
    return np.r_[q,.6,q,.7]


def plan(delta=(.0005,0,0), **kwargs):
    return plan_left_nudge(kwargs.pop('state14',state()),delta,
        kwargs.pop('lower14',[-5]*14),kwargs.pop('upper14',[5]*14),**kwargs)


def test_fk_delta_orientation_and_held_channels():
    initial = state()
    delta = np.array([.0003,-.0004,0.])
    result = plan(delta)
    assert result['valid'],result
    target = np.array(result['target14'])
    kin = NativeToolKinematics()
    p0,r0 = kin.pose(initial[:6])
    p1,r1 = kin.pose(target[:6])
    np.testing.assert_allclose(p1-p0,delta,atol=2e-5)
    assert np.linalg.norm(rotation_log(r1@r0.T)) < 2e-4
    np.testing.assert_array_equal(target[6:],initial[6:])
    np.testing.assert_array_equal(initial,state())
    assert kin.jacobian(initial[:6]).shape == (6,6)
    assert result['metadata']['requires_backend_path_validation']


def test_limits_and_unreachable_local_target_refused():
    initial = state()
    result = plan(lower14=initial-1e-8,upper14=initial+1e-8)
    assert not result['valid']
    assert 'limit' in result['reason'].lower()
    assert not plan(max_joint_delta_rad=.00001)['valid']
    assert not plan(state14=initial+10)['valid']


def test_singularity_rejected_not_rescued_by_gripper():
    initial = np.zeros(14)
    initial[2] = 2.8875376884422113
    initial[6] = .8
    result = plan(state14=initial)
    assert not result['valid']
    assert result['reason'] == 'kinematic_singularity'


def test_finite_inputs_and_hard_step_bounds():
    for delta in ([float('nan'),0,0],[0,float('inf'),0],[.006,0,0],[0,0,0]):
        assert not plan(delta)['valid']
    assert not plan(max_step_m=.011)['valid']
    assert not plan(max_step_m=float('nan'))['valid']
    assert not plan(state14=[0]*13)['valid']
    assert not plan(lower14=[0]*14,upper14=[0]*14)['valid']


def test_explicit_five_mm_nudge_uses_dense_cartesian_path():
    initial=state()
    out=plan([.005,0,0],max_step_m=.005)
    assert out['valid'],out
    kin=NativeToolKinematics()
    p0,r0=kin.pose(initial[:6])
    p1,r1=kin.pose(np.asarray(out['target14'])[:6])
    np.testing.assert_allclose(p1-p0,[.005,0,0],atol=2e-5)
    assert np.linalg.norm(rotation_log(r1@r0.T)) < 2e-4
    assert out['metadata']['max_path_translation_error_m'] <= .0005
    assert out['metadata']['joint_delta_norm_rad'] <= .2
    # Dense Cartesian waypoints remove the old full-span joint interpolation
    # curvature for the same five millimetre request.
    curved=plan([0,.005,0],max_step_m=.005)
    assert curved['valid'],curved
    assert curved['metadata']['max_path_translation_error_m'] < .00001
    assert not plan([.01001,0,0],max_step_m=.01)['valid']
    assert not plan([.006,0,0])['valid']  # default is capped at 5mm
    assert not plan(max_joint_delta_rad=.201)['valid']


def test_nonlinear_kinematics_bowing_compensated_by_cartesian_waypoints():
    from pin_regrasp_motion import rotation_exp
    class BowedKinematics:
        def pose(self,q):
            point=np.asarray(q[:3]).copy()
            point[1]+=.001*np.sin(np.pi*q[0]/.01)
            return point,rotation_exp(np.asarray(q[3:]))
        def jacobian(self,q):
            j=np.eye(6)
            j[1,0]=.001*np.pi/.01*np.cos(np.pi*q[0]/.01)
            return j
    result=plan([.005,0,0],max_step_m=.005,state14=np.zeros(14),kinematics=BowedKinematics())
    assert result['valid'], result
    assert result['metadata']['max_path_translation_error_m'] < .00001


def test_xy_path_locks_height_orientation_and_every_held_channel():
    initial=state()
    kin=NativeToolKinematics()
    anchor,rotation=kin.pose(initial[:6])
    result=plan([0,.005,0],max_step_m=.005)
    assert result['valid'],result
    waypoints=np.asarray(result['waypoints14'])
    assert len(waypoints)>=21
    np.testing.assert_array_equal(waypoints[0],initial)
    np.testing.assert_array_equal(waypoints[-1],result['target14'])
    for start,end in zip(waypoints,waypoints[1:]):
        for fraction in np.linspace(0,1,11):
            q=start+fraction*(end-start)
            p,r=kin.pose(q[:6])
            assert abs(p[2]-anchor[2])<1e-5
            assert np.linalg.norm(rotation_log(r@rotation.T))<1e-5
            np.testing.assert_array_equal(q[6:],initial[6:])
    assert result['metadata']['height_locked']
    np.testing.assert_allclose(result['metadata']['anchor_position_m'],anchor)
    np.testing.assert_allclose(result['metadata']['anchor_rotation'],rotation)
    assert not plan([0,0,.0001])['valid']
