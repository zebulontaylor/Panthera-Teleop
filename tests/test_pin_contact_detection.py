import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.pin_contact_detection import PinContactDetector


class Trace:
    def __init__(self, enabled=True):
        self.d=PinContactDetector(dict(enabled=enabled));self.t=0.
    def tick(self, grip=.1,z=.04,region=True,frozen=True,eligible=True,dt=.05,allow_trigger=True):
        self.t+=dt;q=[0.]*14;q[6]=grip
        left=[.29,-.25,z];right=[.29,.35,z-.03] if region else [.5,.5,.5]
        return self.d.update(self.t,q,left,right,eligible,frozen,allow_trigger=allow_trigger)
    def pickup(self):
        self.tick(grip=.8,z=.02,region=False)
        for _ in range(5):self.tick(z=.02,region=False)
        assert self.d.phase=='pin_grasped'
        self.tick(z=.07,region=False)
        assert self.d.phase=='transporting_pin'


def test_causal_pickup_lift_descent_stall_triggers_once():
    t=Trace();t.pickup()
    for z in [.065,.06,.055,.05]:
        assert not t.tick(z=z)['triggered']
    result=None
    for _ in range(5):
        r=t.tick(z=.05)
        if r['triggered']:result=r
    assert result and result['contact_candidate']
    assert not result['contact_confirmed'] and not result['motor_permission']
    assert not t.tick(z=.05)['triggered']
    assert not t.d.record()['triggered']


def test_missing_open_grip_or_lift_never_triggers():
    t=Trace()
    for _ in range(30):assert not t.tick()['triggered']
    t.tick(grip=.8,z=.02)
    for _ in range(20):assert not t.tick(z=.02)['triggered']
    assert t.d.phase=='pin_grasped'


@pytest.mark.parametrize('flag', ['region','frozen','eligible'])
def test_missing_required_context_never_triggers(flag):
    t=Trace();t.pickup()
    for _ in range(12):assert not t.tick(z=.05,**{flag:False})['triggered']


def test_continuing_descent_hands_off_once_for_guarded_contact_approach():
    t=Trace();t.pickup()
    results=[t.tick(z=.055-i*.001) for i in range(20)]
    assert sum(r['triggered'] for r in results)==1
    assert not any(r['contact_confirmed'] for r in results)


def test_region_entry_can_handoff_without_prior_descent():
    t=Trace();t.pickup()
    results=[t.tick(z=.07+i*.001) for i in range(10)]
    assert sum(r['triggered'] for r in results)==1
    assert any(r.get('heuristic')=='pin_pickup_lift_carrier_region_v2' for r in results)


def test_trigger_gate_preserves_causal_pickup_and_region_history():
    t=Trace();t.pickup()
    for _ in range(10):
        assert not t.tick(z=.07,allow_trigger=False)['triggered']
    assert t.d.phase=='checking_carrier_region'
    result=t.tick(z=.07,allow_trigger=True)
    assert result['triggered'] and not result['contact_confirmed']


def test_stall_dwell_restarts_after_measurement_gap():
    t=Trace();t.pickup()
    t.tick(z=.05);t.tick(z=.05)
    assert not t.tick(z=.05,dt=.5)['triggered']
    assert not t.tick(z=.05)['triggered']
    assert not t.tick(z=.05)['triggered']


def test_grip_loss_requires_new_pickup_sequence():
    t=Trace();t.pickup();t.tick(z=.05,grip=.3)
    for _ in range(10):assert not t.tick(z=.05)['triggered']
    assert t.d.phase=='waiting_for_open_grip'


def test_disabled_reset_rearm_and_invalid_data():
    t=Trace(False)
    assert not t.d.enabled
    assert t.tick()['status']=='disabled'
    t.d.set_enabled(True);t.pickup()
    assert not t.d.update(t.t+.01,[float('nan')]*14,[0,0,.05],[0,.6,.02],True,True)['triggered']
    t.d.complete()
    assert not t.tick()['triggered']
    t.d.rearm();assert t.d.phase=='waiting_for_open_grip'
    with pytest.raises(ValueError):t.d.configure(enabled=1)


def test_out_of_order_sample_clears_pending_dwell():
    t=Trace();t.pickup();t.tick(z=.05);t.tick(z=.05)
    q=[0.]*14;q[6]=.1
    assert not t.d.update(t.t-.01,q,[0,0,.05],[0,.6,.02],True,True)['triggered']
    assert not t.tick(z=.05)['triggered']


def test_direct_in_approach_uses_no_visual_misalignment_input():
    t=Trace();t.pickup()
    results=[t.tick(z=.05) for _ in range(8)]
    assert sum(r['triggered'] for r in results)==1


def test_latched_attempt_survives_ineligible_and_completion_updates_result():
    t=Trace();t.pickup()
    for _ in range(8):t.tick(z=.05)
    assert t.d.latched
    assert t.tick(eligible=False)['status']=='assisting'
    assert t.d.latched
    t.d.complete()
    assert t.d.result['status']=='complete'
    assert t.tick(eligible=False)['status']=='complete'


def test_small_real_pin_grip_remains_valid_after_pickup():
    t=Trace();t.pickup()
    outputs=[t.tick(grip=.011,z=.05) for _ in range(8)]
    assert sum(r['triggered'] for r in outputs)==1


def test_gear_area_low_pickup_does_not_arm_pin_stage():
    d=PinContactDetector(dict(enabled=True));q=[0.]*14;q[6]=.8
    d.update(0.,q,[.15,0.,.02],[.15,.6,-.01],True,True)
    q[6]=.1
    for i in range(1,20):
        result=d.update(i*.05,q,[.15,0.,.02],[.15,.6,-.01],True,True)
    assert result['phase']=='waiting_for_pin_grasp'


def test_smooth_pickup_can_settle_grip_during_early_lift():
    t=Trace();t.tick(grip=.8,z=.02,region=False)
    t.tick(grip=.1,z=.02,region=False)
    for z in [.025,.03,.04,.055,.07]:t.tick(grip=.1,z=z,region=False)
    assert t.d.phase in ('pin_grasped','transporting_pin')
    results=[t.tick(z=.05) for _ in range(8)]
    assert sum(r['triggered'] for r in results)==1


def test_negative_encoder_pin_grip_is_supported_with_full_stage_sequence():
    t=Trace();t.tick(grip=.8,z=.02,region=False)
    for _ in range(6):t.tick(grip=-.011,z=.02,region=False)
    t.tick(grip=-.011,z=.07,region=False)
    results=[t.tick(grip=-.011,z=.05) for _ in range(8)]
    assert sum(r['triggered'] for r in results)==1


@pytest.mark.parametrize('x,y,expected',[(.222,-.264,True),(.246,-.256,True),(.241,-.275,True),(.20,-.33,False),(.20,-.25,False)])
def test_relocated_pin_pickup_and_gear_release_region(x,y,expected):
    d=PinContactDetector(dict(enabled=True));q=[0.]*14;q[6]=.8
    d.update(0.,q,[x,y,.02],[x,y+.6,-.01],True,True)
    q[6]=-.011
    for i in range(1,8):
        result=d.update(i*.05,q,[x,y,.02],[x,y+.6,-.01],True,True)
    assert (result['phase']=='pin_grasped') is expected


def test_carrier_region_dwell_restarts_after_exit():
    t=Trace();t.pickup()
    t.tick(z=.07);t.tick(z=.07)
    assert not t.tick(z=.07,region=False)['triggered']
    assert not t.tick(z=.07)['triggered']
    assert not t.tick(z=.07)['triggered']
    results=[t.tick(z=.07) for _ in range(5)]
    assert sum(r['triggered'] for r in results)==1
