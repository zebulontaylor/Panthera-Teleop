import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from pin_alignment_force import CartesianLoadController, CartesianLoadConfig
from pin_regrasp_motion import rotation_exp


class LinearKinematics:
    def pose(self, q):
        return np.asarray(q[:3]), rotation_exp(np.asarray(q[3:]))

    def jacobian(self, q):
        return np.eye(6)


class LowLeverageKinematics(LinearKinematics):
    # Consistent linear plant with enough leverage for multi-newton forces
    # while retaining the unchanged joint torque caps.
    def pose(self, q):
        return .1 * np.asarray(q[:3]), rotation_exp(np.asarray(q[3:]))

    def jacobian(self, q):
        return np.diag([.1, .1, .1, 1., 1., 1.])


def controller(**kwargs):
    return CartesianLoadController(np.zeros(14), now=0., lower6=[-2.]*6,
        upper6=[2.]*6, kinematics=kwargs.pop('kinematics', LinearKinematics()), **kwargs)


def compute(c, q=None, dq=None, gravity=None, target=None, now=.1, **kwargs):
    return c.compute(np.zeros(6) if q is None else q,
        np.zeros(6) if dq is None else dq, np.zeros(6) if gravity is None else gravity,
        [0., 0.] if target is None else target, now, **kwargs)


def test_downward_ramp_jacobian_transpose_and_gravity_sum():
    c = controller()
    a = compute(c, target=[.001, -.002], gravity=[.1]*6)
    assert a['valid'] and not a['force_regulation'] and not a['physically_validated']
    np.testing.assert_allclose(a['wrench_left_base'], np.r_[np.array([.6, -1.2])+a['diagnostics']['xy_integral_force_n'], -.2, 0, 0, 0])
    np.testing.assert_allclose(a['assist_torque_nm'], a['wrench_left_base'])
    np.testing.assert_allclose(a['full_torque_nm'], np.array(a['assist_torque_nm']) + .1)
    assert compute(c, now=.2)['commanded_force_n'] == pytest.approx(.4)
    assert compute(c, now=.3)['commanded_force_n'] == pytest.approx(.5)


def test_no_vertical_position_spring_and_vertical_damping_sign():
    c = controller()
    for i in range(1,11):
        assert compute(c,q=[0,0,i*.0004,0,0,0],now=i*.1)['valid']
    displaced = compute(c,q=[0,0,.004,0,0,0],now=1.1)
    assert displaced['wrench_left_base'][2] == pytest.approx(-.5)
    descending = compute(controller(), q=[0,0,-.0005,0,0,0])
    assert descending['wrench_left_base'][2] == pytest.approx(-.15)


def test_only_explicit_bounded_contact_approach_allows_extra_vertical_travel():
    config = dict(contact_approach=True, max_vertical_travel_m=.020,
                  max_lateral_travel_m=.005, max_duration_s=30.)
    c = controller(config=config)
    for tick in range(1, 171):
        now = tick*.05
        u = min(1., now)
        distance = .001*(u**3-.5*u**4+max(0., now-1.))
        assert compute(c, q=[0,0,-distance,0,0,0], now=now)['valid']
    assert compute(c, q=[0, 0, -.0201, 0, 0, 0], now=8.55)['reason']=='vertical_travel_exceeded'
    for change in [dict(force_n=.6), dict(max_lateral_travel_m=.006),
                   dict(max_duration_s=31.), dict(max_vertical_travel_m=.021)]:
        with pytest.raises(ValueError):
            CartesianLoadConfig(**dict(config, **change))


def approach_controller():
    return controller(config=dict(contact_approach=True, max_vertical_travel_m=.020,
        max_lateral_travel_m=.005, max_duration_s=30., force_slew_n_s=.15,
        vertical_damping_ns_m=20., xy_integral_gain_n_m_s=600.,
        xy_integral_limit_n=1.5, xy_integral_slew_n_s=.5))


def test_approach_never_uses_upward_position_recovery_and_stops_recorded_scale_sag():
    c = approach_controller()
    r = compute(c, q=[0,0,-.001,0,0,0], now=.01)
    assert r['valid']
    assert r['diagnostics']['approach_z_error_m'] > 0
    assert r['diagnostics']['approach_spring_force_n']==0
    # The observed run fell 8.8mm in 161ms before being pulled back up.
    # Terminate that unexpected sag before allowing upward recovery to build.
    r = compute(c, q=[0,0,-.003,0,0,0], now=.05)
    assert not r['valid'] and r['reason']=='approach_descent_tracking_lost'
    assert r['assist_torque_nm']==[0.]*6
    assert compute(c, now=.06)['reason']=='approach_descent_tracking_lost'


def test_approach_reference_starts_smoothly_and_every_joint_assist_is_slew_limited():
    c = approach_controller()
    previous = np.zeros(6)
    last_z = 0.
    for tick in range(1, 151):
        now = tick*.01
        # Early lateral sag and downward movement exercise stabilization even
        # during the velocity-history startup window.
        r = compute(c, q=[.0005,0,-.0003,0,0,0], now=now)
        assert r['valid'], r
        assist = np.asarray(r['assist_torque_nm'])
        assert np.max(np.abs(assist-previous)) <= .005+1e-12
        previous = assist
        z = r['diagnostics']['approach_z_target_m']
        assert 0 <= last_z-z <= .00001+1e-12
        if tick==1:
            assert abs(z)<1e-8
            assert np.linalg.norm(assist)>0  # No gravity-only startup gap.
        if tick==100:
            assert z == pytest.approx(-.0005)
        last_z = z


def test_slow_free_tracking_does_not_claim_contact_but_blocked_pin_does():
    c = approach_controller()
    q = np.zeros(6); dq = np.zeros(6)
    blocked = False
    loaded_after_block = False
    for tick in range(1, 801):
        r = compute(c, q=q, dq=dq, now=tick*.01)
        assert r['valid'], r
        if not blocked:
            assert not r['diagnostics']['approach_load_established']
        elif r['diagnostics']['approach_load_established']:
            loaded_after_block = True
        # Ideal half-kilogram translational plant with a carrier at -4mm.
        dq[2] += r['assist_torque_nm'][2]/.5*.01
        q[2] += dq[2]*.01
        if not blocked:
            assert abs(dq[2]) < .002
        if q[2] <= -.004:
            q[2] = -.004; dq[2] = 0.; blocked = True
    assert blocked and loaded_after_block


def test_continuous_loaded_transfer_preserves_support_history_and_slews_rebound_response():
    c = approach_controller()
    for tick in range(1, 401):
        r = compute(c, now=tick*.01)
        assert r['valid']
    before = np.asarray(r['assist_torque_nm'])
    history = list(c._pose_history)
    integral = c.xy_integral_n.copy()
    force = c.ramped_force_n
    c.begin_loaded(np.zeros(14), 4., dict(force_n=.5, force_slew_n_s=.15,
        vertical_damping_ns_m=20., max_duration_s=120.))
    assert list(c._pose_history)==history
    np.testing.assert_array_equal(c.xy_integral_n, integral)
    np.testing.assert_array_equal(c._previous_assist, before)
    assert c.ramped_force_n==force and not c.config.contact_approach
    # First loaded frame keeps the previous support rather than reprimes at zero.
    r = compute(c, now=4.01)
    assert r['valid'] and r['reason']=='ok'
    assert r['assist_torque_nm'][2] < -.45
    np.testing.assert_allclose(r['assist_torque_nm'], before, atol=.00500001, rtol=0)
    previous = np.asarray(r['assist_torque_nm'])
    for tick in range(2, 151):
        # Small upward rebound stays inside the fixed loaded envelope and gets
        # a smoothly changing corrective torque throughout the transfer.
        r = compute(c, q=[0,0,.0026,0,0,0], now=4.+tick*.01)
        assert r['valid'], r
        assert np.max(np.abs(np.asarray(r['assist_torque_nm'])-previous))<=.00500001
        previous = np.asarray(r['assist_torque_nm'])
    assert r['diagnostics']['load_transfer_fraction']==1.
    with pytest.raises(ValueError):
        c.begin_loaded(np.zeros(14), 5.5, dict(force_n=.5))


def test_fixed_rotation_restores_anchor_and_damps_motion():
    r = compute(controller(), q=[0, 0, 0, .02, 0, 0], dq=[0, 0, 0, .02, 0, 0])
    assert r['valid']
    assert r['wrench_left_base'][3] == pytest.approx(-.06)


@pytest.mark.parametrize('diagonal', [[1,1,1,1,1,0], [1,1,1,1,1,9e-5],
    [2,1,1,1,1,.001]])
def test_singular_or_ill_conditioned_jacobian_latches_stop(diagonal):
    class DegenerateKinematics(LinearKinematics):
        def jacobian(self, q): return np.diag(diagonal)
    c = CartesianLoadController(np.zeros(14), lower6=[-2.]*6, upper6=[2.]*6,
        kinematics=DegenerateKinematics())
    assert compute(c)['reason'] == 'kinematic_singularity'
    c.kinematics = LinearKinematics()
    assert compute(c)['reason'] == 'kinematic_singularity'


@pytest.mark.parametrize('changes,reason', [
    ({'q': [.0301,0,0,0,0,0]}, 'lateral_travel_exceeded'),
    ({'target': [.0301,0]}, 'xy_target_travel_exceeded'),
    ({'q': [0,0,.0051,0,0,0]}, 'vertical_travel_exceeded'),
    ({'q': [0,0,0,.101,0,0]}, 'orientation_deviation_exceeded'),
    ({'q': [1.99,0,0,0,0,0]}, 'joint_limit_margin'),
    ({'now': -.01}, 'invalid_monotonic_time'),
    ({'now': .11}, 'update_gap_exceeded'),
    ({'gravity': [5.01,0,0,0,0,0]}, 'gravity_plus_assist_torque_limit'),
    ({'gravity': [np.nan]*6}, 'invalid_gravity6'),
])
def test_fail_closed_and_latched_stop(changes, reason):
    c = controller(); r = compute(c, **changes)
    assert not r['valid'] and r['stop'] and r['reason'] == reason
    assert r['full_torque_nm'] is None
    assert compute(c)['reason'] == reason


def test_non_xy_assist_torque_never_silently_clipped():
    class VerticalLeverage(LinearKinematics):
        def pose(self, q):
            return np.asarray(q[:3])*[1.,1.,40.], rotation_exp(np.asarray(q[3:]))
        def jacobian(self, q):
            return np.diag([1.,1.,40.,1.,1.,1.])
    r = compute(controller(kinematics=VerticalLeverage()))
    assert r['reason'] == 'assist_torque_limit'
    assert abs(r['diagnostics']['assist_torque_nm'][2]) > 6.
    assert r['diagnostics']['assist_torque_limit_nm'] == [3.,6.,6.,1.5,1.2,1.2]
    assert r['full_torque_nm'] is None


def test_larger_assistance_accepts_previously_refused_xy_force():
    r = compute(controller(), target=[.009, 0])
    assert r['valid'], r
    assert 1.5 < r['assist_torque_nm'][0] <= 3.


def test_xy_force_rises_substantially_faster_without_increasing_downward_load():
    previous = controller(kinematics=LowLeverageKinematics(), config=dict(
        xy_stiffness_n_m=150., xy_damping_ns_m=5.,
        xy_integral_gain_n_m_s=2400., xy_integral_limit_n=6., xy_integral_slew_n_s=4.))
    stronger = controller(kinematics=LowLeverageKinematics())
    old = compute(previous, target=[.003,0])
    new = compute(stronger, target=[.003,0])
    assert old['valid'] and new['valid']
    assert old['wrench_left_base'][0] == pytest.approx(.85)
    assert new['wrench_left_base'][0] == pytest.approx(3.)
    assert new['wrench_left_base'][0] > 3*old['wrench_left_base'][0]
    np.testing.assert_allclose(new['wrench_left_base'][2:],old['wrench_left_base'][2:])


@pytest.mark.parametrize('gravity_x,expected_assist', [(0.,3.), (4.9,.1)])
def test_large_xy_spring_allocates_headroom_without_changing_downward_load(gravity_x, expected_assist):
    r = compute(controller(), target=[.02,0], gravity=[gravity_x,0,0,0,0,0])
    assert r['valid'], r
    assert r['diagnostics']['xy_drive_scale'] < 1.
    assert r['assist_torque_nm'][0] == pytest.approx(expected_assist)
    assert r['wrench_left_base'][2] == pytest.approx(-.2)
    assert r['diagnostics']['xy_requested_proportional_force_n'] == [12.,0.]
    assert r['full_torque_nm'][0] <= 5.
    # Stored integral equals the applied component; no hidden windup.
    assert r['diagnostics']['xy_integral_force_n'][0] < 1.2


def test_duration_and_actual_limits_required():
    c = controller(config={'max_duration_s': .2})
    assert compute(c)['valid']
    assert compute(c, now=.2)['reason'] == 'duration_exceeded'
    missing = CartesianLoadController(np.zeros(14), kinematics=LinearKinematics())
    assert compute(missing)['reason'] == 'invalid_actual_lower6'


def test_native14_api_uses_left_six_only_and_requires_gravity():
    c = controller(); q = np.r_[np.zeros(6), .5, np.ones(7)]
    result = c.step(q, np.zeros(14), [.001,0], .1, gravity6=np.zeros(6))
    assert result['valid'] and result['assist_torque_nm'][0] == pytest.approx(.6 + result['diagnostics']['xy_integral_force_n'][0])
    assert controller().step(q, np.zeros(14), [0,0], .1)['reason'] == 'invalid_gravity6'


def test_zero_observed_motion_ignores_moderate_sdk_velocity_for_damping():
    c = controller()
    for t in np.arange(.01,.101,.01):
        r = compute(c,dq=[.04,0,-.03,0,0,0],now=float(t))
        assert r['valid']
        assert r['diagnostics']['tool_speed_m_s'] == 0
        assert r['tool_velocity_left'][:3] == [0.,0.,0.]
        assert r['wrench_left_base'][0] == 0
    assert r['diagnostics']['raw_sdk_tool_speed_m_s'] == pytest.approx(.05)


def test_recorded_scale_encoder_jump_during_startup_is_guarded_zero_assist():
    c = controller()
    r = compute(c,q=[0,0,-.000544,0,0,0],now=.00668)
    assert r['valid'] and r['reason'] == 'priming_velocity'
    assert r['assist_torque_nm'] == [0.]*6
    assert r['commanded_force_n'] == 0
    assert r['diagnostics']['tool_speed_m_s'] == 0
    assert r['diagnostics']['raw_pose_tool_speed_m_s'] > .08
    # Speed alone does not stop loading, and priming still supplies zero assist.
    r = compute(c,q=[.0015,0,0,0,0,0],now=.01336)
    assert r['valid'] and r['assist_torque_nm'] == [0.]*6
    assert r['diagnostics']['speed_guards_enabled'] is False


def test_continuous_one_mm_per_second_spiral_motion_keeps_damping():
    c = controller()
    for t in np.arange(.01,.201,.01):
        x = .001*t
        r = compute(c,q=[x,0,0,0,0,0],target=[x,0],now=float(t))
        assert r['valid']
    assert r['tool_velocity_left'][0] == pytest.approx(.001)
    assert r['wrench_left_base'][0] == pytest.approx(-.01)
    assert r['diagnostics']['tool_speed_m_s'] == 0  # Guard noise budget only.


@pytest.mark.parametrize('changes', [
    {'q': [.006,0,0,0,0,0], 'target': [.006,0]},
    {'dq': [.5,0,0,0,0,0]},
    {'dq': [0,0,0,1.1,0,0]},
    {'q': [0,0,-.0018,0,0,0]},
    {'q': [0,0,0,.05,0,0]},
])
def test_speed_alone_does_not_stop_load(changes):
    r = compute(controller(config={'max_down_speed_m_s':.01}), **changes)
    assert r['valid'], r
    assert r['diagnostics']['speed_guards_enabled'] is False
    assert r['diagnostics']['max_tool_speed_m_s'] is None
    assert r['diagnostics']['max_down_speed_m_s'] is None


def test_ten_mm_xy_target_circle_at_native_pose_respects_torque_caps():
    # Native recorded-like follower pose; this verifies the pure model mapping,
    # not hardware stability or actual tracking of the requested circle.
    q = np.array([-.543,1.458,.684,.0804,.3965,.4738])
    state = np.r_[q,1.2,q,1.2]
    gravity = [0.,-2.8,3.46,.697,.071,.0013]
    for angle in np.linspace(0,2*np.pi,12,endpoint=False):
        c = CartesianLoadController(state, lower6=[-3]*6, upper6=[3]*6)
        target = c.anchor_position[:2]+.01*np.array([np.cos(angle),np.sin(angle)])
        for t in [.1,.2,.3]:
            result = c.step(state,np.zeros(14),target,t,gravity6=gravity)
            assert result['valid'],result
        assert result['commanded_force_n'] == pytest.approx(.5)
        np.testing.assert_array_less(np.abs(result['assist_torque_nm']), np.array([3.,6.,6.,1.5,1.2,1.2])+1e-12)


@pytest.mark.parametrize('config', [{'force_n': 0}, {'force_n': 3.01}, {'max_duration_s': 301},
    {'max_vertical_travel_m': .006}, {'max_lateral_travel_m':.031},
    {'max_tool_speed_m_s':.051}, {'force_n': True}, {'xy_stiffness_n_m': float('nan')}])
def test_configuration_hard_bounds(config):
    with pytest.raises(ValueError): CartesianLoadConfig(**config)


def test_five_mm_manual_xy_target_preserves_downward_load_and_vertical_bounds():
    # Compare identical measured motion/gravity histories. Only the manual XY
    # target differs: changing that target must never introduce a Z position
    # command, reset force ramp, or alter orientation/gravity compensation.
    baseline=controller()
    shifted=controller()
    gravity=np.array([.1,.2,.3,.04,.05,.06])
    for now in [.1,.2,.3]:
        before=compute(baseline,gravity=gravity,now=now)
        moved=compute(shifted,gravity=gravity,now=now)
        assert before['valid'] and moved['valid']
    anchor=shifted.anchor_position.copy()
    rotation=shifted.anchor_rotation.copy()
    config=shifted.config
    for now in [.4,.5,.6]:
        before=compute(baseline,gravity=gravity,now=now)
        moved=compute(shifted,target=[.003,.004],gravity=gravity,now=now)
        assert before['valid'] and moved['valid']
        assert moved['commanded_force_n']==pytest.approx(before['commanded_force_n'])
        np.testing.assert_allclose(moved['wrench_left_base'][2:],before['wrench_left_base'][2:])
        np.testing.assert_allclose(moved['gravity_torque_nm'],before['gravity_torque_nm'])
        np.testing.assert_allclose(moved['full_torque_nm'][2:],before['full_torque_nm'][2:])
        np.testing.assert_allclose(moved['wrench_left_base'][:2],np.array(moved['diagnostics']['xy_proportional_force_n'])+moved['diagnostics']['xy_integral_force_n'])
    np.testing.assert_array_equal(shifted.anchor_position,anchor)
    np.testing.assert_array_equal(shifted.anchor_rotation,rotation)
    assert shifted.config is config
    assert shifted.config.max_vertical_travel_m==.005
    # Vertical compliance still has the same absolute travel stop after a nudge.
    assert compute(shifted,target=[.003,.004],q=[0,0,.0051,0,0,0],now=.7)['reason']=='vertical_travel_exceeded'


def test_loaded_manual_xy_does_not_override_vertical_damping():
    baseline=controller()
    shifted=controller()
    for now in [.1,.2,.3]:
        assert compute(baseline,now=now)['valid']
        assert compute(shifted,now=now)['valid']
    q=[0,0,-.0002,0,0,0]
    before=compute(baseline,q=q,now=.4)
    moved=compute(shifted,q=q,target=[.003,.004],now=.4)
    assert before['valid'] and moved['valid']
    assert moved['wrench_left_base'][2]==pytest.approx(before['wrench_left_base'][2])
    assert moved['wrench_left_base'][2]>-.5  # damping opposes measured descent


def test_xy_integral_priming_deadband_slew_cap_and_unwind():
    c = controller(kinematics=LowLeverageKinematics())
    for now in [.01,.02,.03]:
        r=compute(c,target=[0,.003],now=now)
        assert r['valid'] and r['diagnostics']['xy_integral_force_n']==[0.,0.]
        assert r['diagnostics']['xy_proportional_force_n']==[0.,0.]
    previous=np.zeros(2)
    for now in np.arange(.04,4.05,.01):
        r=compute(c,target=[0,.003],now=float(now))
        assert r['valid'],r
        integral=np.array(r['diagnostics']['xy_integral_force_n'])
        assert np.linalg.norm(integral-previous)<=.12+1e-12
        assert np.linalg.norm(integral)<=12.+1e-12
        previous=integral
    assert previous[1]==pytest.approx(12.)
    # No hidden windup: an opposite error immediately reduces stored force.
    r=compute(c,target=[0,-.003],now=4.06)
    assert r['diagnostics']['xy_integral_force_n'][1]<12.
    retained=r['diagnostics']['xy_integral_force_n']
    r=compute(c,target=[.0001,0],now=4.07)
    assert r['diagnostics']['xy_integral_force_n']==retained


def test_bounded_xy_integral_overcomes_loaded_fake_static_friction():
    # The plant stays fixed below 1.5 N, then slides with viscous drag.
    # At 1 mm, the old 1 N integral cap plus .15 N spring cannot move it.
    def simulate(config, breakaway_n=1.5):
        c=controller(config=config, kinematics=LowLeverageKinematics())
        x=0.
        peak_force=0.
        for now in np.arange(.01,8.01,.01):
            r=compute(c,q=[10*x,0,0,0,0,0],target=[.001,0],now=float(now))
            assert r['valid'],r
            force=r['wrench_left_base'][0]
            peak_force=max(peak_force, abs(force))
            if abs(force)>breakaway_n:
                x += .01*np.sign(force)*(abs(force)-breakaway_n)/20.
            assert r['wrench_left_base'][2]==pytest.approx(-r['commanded_force_n'])
            np.testing.assert_allclose(r['wrench_left_base'][3:],0.)
        return x, peak_force

    old_x, old_peak=simulate(CartesianLoadConfig(
        xy_stiffness_n_m=150., xy_integral_gain_n_m_s=300., xy_integral_limit_n=1.,
        xy_integral_slew_n_s=.5))
    new_x, new_peak=simulate(CartesianLoadConfig())
    assert old_x==0.
    assert old_peak==pytest.approx(1.15)
    assert new_peak>1.5
    assert new_x>.0005
    # Stronger loaded contact exceeds the previous default's available force.
    previous_x, previous_peak=simulate(CartesianLoadConfig(
        xy_stiffness_n_m=150., xy_integral_gain_n_m_s=1200., xy_integral_limit_n=4.,
        xy_integral_slew_n_s=2.), breakaway_n=4.5)
    stronger_x, stronger_peak=simulate(CartesianLoadConfig(), breakaway_n=4.5)
    assert previous_x==0.
    assert previous_peak==pytest.approx(4.15)
    assert stronger_peak>4.5
    assert stronger_x>.0005
    # The immediately previous defaults still cannot break this contact free.
    stuck_x, stuck_peak=simulate(CartesianLoadConfig(
        xy_stiffness_n_m=150., xy_damping_ns_m=5.,
        xy_integral_gain_n_m_s=2400., xy_integral_limit_n=6.,
        xy_integral_slew_n_s=4.), breakaway_n=8.)
    moved_x, moved_peak=simulate(CartesianLoadConfig(), breakaway_n=8.)
    assert stuck_x==0.
    assert stuck_peak==pytest.approx(6.15)
    assert moved_peak>8.
    assert moved_x>.0005


@pytest.mark.parametrize('config', [
    {'xy_integral_gain_n_m_s':9600.1}, {'xy_integral_limit_n':12.01},
    {'xy_integral_slew_n_s':12.01}, {'xy_integral_deadband_m':.000251}])
def test_xy_integral_hard_bounds(config):
    with pytest.raises(ValueError): CartesianLoadConfig(**config)


def test_xy_integral_limit_is_radial_for_diagonal_error():
    c=controller(kinematics=LowLeverageKinematics())
    for now in np.arange(.1,8.01,.1):
        r=compute(c,target=[.001,.001],now=float(now))
        assert r['valid'],r
    integral=np.array(r['diagnostics']['xy_integral_force_n'])
    np.testing.assert_allclose(integral,[12/np.sqrt(2)]*2)
    assert np.linalg.norm(integral)==pytest.approx(12.)


def test_native_stuck_fifteen_mm_axial_target_respects_torque_headroom_without_stop():
    q=np.array([-.543,1.458,.684,.0804,.3965,.4738])
    state=np.r_[q,1.2,q,1.2]
    gravity=np.array([0.,-2.8,3.46,.697,.071,.0013])
    largest_integral=0.
    for angle in np.linspace(0,2*np.pi,8,endpoint=False):
        c=CartesianLoadController(state, lower6=[-3]*6, upper6=[3]*6)
        target=c.anchor_position[:2]+.015*np.array([np.cos(angle),np.sin(angle)])
        for now in np.arange(.01,6.01,.01):
            r=c.step(state,np.zeros(14),target,float(now),gravity6=gravity)
            assert r['valid'],r
            np.testing.assert_array_less(np.abs(r['assist_torque_nm']),
                np.array([3.,6.,6.,1.5,1.2,1.2])+1e-12)
            np.testing.assert_array_less(np.abs(r['full_torque_nm']),
                np.array([5.,8.,8.,3.,2.,2.])+1e-12)
            largest_integral=max(largest_integral,
                np.linalg.norm(r['diagnostics']['xy_integral_force_n']))
        assert r['commanded_force_n']==pytest.approx(.5)
    assert largest_integral>1.
