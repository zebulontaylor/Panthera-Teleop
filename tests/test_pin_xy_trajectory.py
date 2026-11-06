"""Offline emitted-quintic height checks; never imports or opens motor interfaces."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from pin_xy_trajectory import HeightLockedNudgeTrajectory,MAX_HEIGHT_ERROR_M
from pin_alignment_motion import NativeToolKinematics,plan_left_nudge
from pin_regrasp_motion import rotation_log
from remote_inference import CONTRACT


def native_plan(delta=(0,.005,0)):
    q=np.array([.2,1.,1.4,.3,.4,.2]);state=np.r_[q,.6,q,.7]
    plan=plan_left_nudge(state,delta,np.full(14,-5.),np.full(14,5.))
    assert plan['valid'],plan
    kin=NativeToolKinematics();p,r=kin.pose(state[:6])
    return state,np.array(plan['waypoints14']),kin,p,r

@pytest.mark.parametrize('timesteps',[(.005,),(.02,),(.05,),(.1,),(.005,.017,.083,.011,.1,.023)])
def test_five_mm_native_quintic_emissions_hold_height_and_speed(timesteps):
    state,path,kin,p0,r0=native_plan();trajectory=HeightLockedNudgeTrajectory(path,10.,kin,p0,r0)
    assert trajectory.preflight_metadata['spline_samples']==(len(path)-1)*21
    assert trajectory.preflight_metadata['max_spline_height_error_m']<=MAX_HEIGHT_ERROR_M
    now=10.;target=state.copy();done=False;count=0
    while not done:
        dt=timesteps[count%len(timesteps)];now+=dt;previous=target.copy()
        target,done=trajectory.tick(now,dt,target);p,r=kin.pose(target[:6])
        assert abs(p[2]-p0[2])<=MAX_HEIGHT_ERROR_M+1e-12
        assert np.linalg.norm(rotation_log(r@r0.T))<=.01
        np.testing.assert_array_equal(target[6:],state[6:])
        assert np.all(np.abs(target-previous)<=np.minimum(trajectory.speed*dt,CONTRACT['max_target_jump_rad'])+1e-12)
        count+=1;assert count<5000
    np.testing.assert_allclose(target,path[-1],atol=1e-10,rtol=0)
    assert trajectory.remaining(now)==0


def test_anchor_stale_origin_and_held_channel_changes_are_refused():
    state,path,kin,p,r=native_plan()
    with pytest.raises(ValueError,match='height_violation'):
        HeightLockedNudgeTrajectory(path,0,kin,p+[0,0,.0001],r)
    bad=path.copy();bad[3,6]+=.01
    with pytest.raises(ValueError,match='held_channels'):
        HeightLockedNudgeTrajectory(bad,0,kin,p,r)
    trajectory=HeightLockedNudgeTrajectory(path,0,kin,p,r)
    state[0]+=.0001
    with pytest.raises(RuntimeError,match='origin_or_command_changed'):trajectory.tick(.01,.01,state)


class CartesianKinematics:
    def pose(self,q):return np.asarray(q[:3],float),np.eye(3)


def test_off_plane_endpoint_and_ten_mm_nudge_are_rejected():
    origin=np.zeros(14);end=origin.copy();end[0]=.005;kin=CartesianKinematics()
    off=end.copy();off[2]=.00002
    with pytest.raises(ValueError,match='height_violation'):
        HeightLockedNudgeTrajectory([origin,off],0,kin,[0,0,0],np.eye(3))
    end[0]=.010
    with pytest.raises(ValueError,match='travel_violation'):
        HeightLockedNudgeTrajectory([origin,end],0,kin,[0,0,0],np.eye(3))


def test_actual_bezier_path_is_checked_not_only_endpoint_poses():
    class BowedKinematics:
        def pose(self,q):
            return np.array([q[0],q[1],q[2]+.0001*np.sin(np.pi*q[0]/.005)]),np.eye(3)
    origin=np.zeros(14);end=origin.copy();end[0]=.005
    with pytest.raises(ValueError,match='height_violation'):
        HeightLockedNudgeTrajectory([origin,end],0,BowedKinematics(),[0,0,0],np.eye(3))


def test_runtime_checks_emitted_command_before_return(monkeypatch):
    state,path,kin,p,r=native_plan();trajectory=HeightLockedNudgeTrajectory(path,0,kin,p,r)
    bad=state.copy();bad[2]+=.01
    monkeypatch.setattr(trajectory._smooth,'tick',lambda now,dt,target:(bad.copy(),False))
    with pytest.raises(RuntimeError,match='height_violation'):trajectory.tick(.01,.01,state)


def test_start_reset_reuses_exact_preflight_path_and_dt_contract():
    state,path,kin,p,r=native_plan();trajectory=HeightLockedNudgeTrajectory(path,0,kin,p,r)
    controls=trajectory._smooth.controls.copy();trajectory.reset_started(30.)
    assert trajectory.started==30. and trajectory.remaining(30.)==pytest.approx(trajectory.duration)
    np.testing.assert_array_equal(controls,trajectory._smooth.controls)
    emitted,done=trajectory.tick(30.,0,state)
    np.testing.assert_array_equal(emitted,state)
    assert not done
    with pytest.raises(RuntimeError,match='deadline'):trajectory.tick(30.11,.11,emitted)

def test_nonbinary_decimal_held_channels_snap_roundoff_not_real_motion(monkeypatch):
    origin=np.zeros(14);origin[6:]=[.6,.2,1.,1.4,.3,.4,.2,.7]
    path=np.repeat(origin[None],21,axis=0);path[:,0]=np.linspace(0,.005,21)
    trajectory=HeightLockedNudgeTrajectory(path,0,CartesianKinematics(),[0,0,0],np.eye(3))
    # Exercise central validation as well as the preflight/tick outer guards.
    rounded=origin.copy();rounded[6:]=np.nextafter(rounded[6:],np.inf)
    trajectory._check_command(rounded)
    np.testing.assert_array_equal(rounded[6:],origin[6:])
    real_change=origin.copy();real_change[6]+=1.1e-10
    with pytest.raises(ValueError,match='held_channels'):trajectory._check_command(real_change)
    emitted=origin.copy();emitted[0]=.0001;emitted[6:]=np.nextafter(emitted[6:],np.inf)
    monkeypatch.setattr(trajectory._smooth,'tick',lambda now,dt,target:(emitted.copy(),False))
    command,_=trajectory.tick(.01,.01,origin)
    np.testing.assert_array_equal(command[6:],origin[6:])
