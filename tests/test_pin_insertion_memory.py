import numpy as np
import pytest
from app.pin_insertion_memory import PinInsertionHoleMemory


def observe(memory,stamp,kp=None,**overrides):
    args={'observation_ns':stamp,'now_ns':stamp,'episode_id':'episode','grasp_id':'grasp',
        'camera_id':'right_wrist','sequence_id':'rollout','image_size':(640,480),'grasp_locked':True,
        'hole_visible':True,'hole_trusted':True,'initialize':True}
    args.update(overrides)
    return memory.observe(kp if kp is not None else [.2,.3,.5,.6],**args)


def seed(memory):
    for i,u in enumerate((.50,.51,.49),1):result=observe(memory,i*1_000_000,[.2,.3,u,.6])
    return result


def test_retains_robust_hole_through_occlusion_and_pin_is_current():
    memory=PinInsertionHoleMemory(max_memory_age_ns=60_000_000_000)
    result=seed(memory)
    assert result['keypoints'][2:]==pytest.approx([.5,.6])
    result=observe(memory,35_000_000_000,[.7,.8,None,None],hole_visible=False,hole_trusted=False,initialize=False)
    assert result['memory_available']
    assert result['keypoints']==pytest.approx([.7,.8,.5,.6])
    assert result['source_ns']==3_000_000
    assert result['age_ns']==35_000_000_000-3_000_000
    assert result['hole_source']=='remembered_stationary_prior'
    assert not result['gate_positive_from_memory']
    assert result['uncertainty_px'] is None # Low spread is not calibrated model accuracy.


def test_no_silent_untrusted_or_undesignated_seed():
    memory=PinInsertionHoleMemory()
    for i in range(1,5):assert not observe(memory,i,initialize=False)['memory_available']
    for i in range(5,9):assert not observe(memory,i,hole_trusted=False)['memory_available']
    result=observe(memory,9,hole_visible=False,hole_trusted=True)
    assert not result['memory_available']
    assert 'trust requires' in result['reset_reason']


@pytest.mark.parametrize('change',[{'episode_id':'new'},{'grasp_id':'new'},{'camera_id':'new'},
    {'sequence_id':'new'},{'image_size':(320,240)},{'grasp_locked':False}])
def test_explicit_context_changes_reset_memory(change):
    memory=PinInsertionHoleMemory();seed(memory)
    result=observe(memory,4_000_000,initialize=False,hole_visible=False,hole_trusted=False,**change)
    assert not result['memory_available']
    assert result['reset_reason']


def test_invalid_timestamp_horizon_and_optional_gap_reset():
    memory=PinInsertionHoleMemory(max_memory_age_ns=10_000_000)
    seed(memory)
    result=observe(memory,3_000_000,initialize=False)
    assert not result['memory_available']
    assert 'timestamp' in result['reset_reason']
    seed(memory)
    result=observe(memory,20_000_000,initialize=False,hole_visible=False,hole_trusted=False)
    assert not result['memory_available']
    assert 'horizon' in result['reset_reason']
    memory=PinInsertionHoleMemory(max_frame_gap_ns=2_000_000);seed(memory)
    result=observe(memory,10_000_000,initialize=False,hole_visible=False,hole_trusted=False)
    assert not result['memory_available']
    assert 'gap' in result['reset_reason']
    result=observe(memory,20_000_000,now_ns=1_000_000_000)
    assert not result['memory_available']
    assert 'stale' in result['reset_reason']


def test_visible_disagreement_resets_and_uncertainty_is_explicit():
    memory=PinInsertionHoleMemory()
    for i in range(1,4):result=observe(memory,i,hole_uncertainty_px=6.)
    assert result['uncertainty_px']==6
    result=observe(memory,4,[.2,.3,.9,.9],initialize=False)
    assert not result['memory_available']
    assert 'disagrees' in result['reset_reason']
    assert not result['motor_action']
