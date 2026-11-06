import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pin_nudge_detection import PinNudgeDetector


def proposal(i=1, **changes):
    return dict(status='ready', localization_ready=True, nudge_valid=True,
        frame_id=i, analyzed_timestamp_ns=1_000_000_000+i*500_000_000,
        measured_state14=[0.]*14, pin_px=[100., 100.], hole_px=[110., 100.],
        pin_source='cnn', hole_source='left_carrier_opencv_observed',
        generation=1, target_hole_index=0, **changes)


def feed(detector, p, **flags):
    return detector.update(p, p['analyzed_timestamp_ns']+1, True, **flags)


def test_default_disabled_and_temporal_evidence():
    d=PinNudgeDetector()
    assert feed(d, proposal())['needs_nudge'] is None
    d.set_enabled(True)
    assert feed(d, proposal(1))['status']=='waiting'
    assert feed(d, proposal(2))['status']=='waiting'
    result=feed(d, proposal(3))
    assert result['status']=='needed' and result['needs_nudge'] is True
    assert result['reason']=='persistent visual misalignment'
    assert result['motor_permission'] is False


def test_distinct_frames_and_full_second_required():
    d=PinNudgeDetector(dict(enabled=True))
    p=proposal()
    feed(d,p)
    assert feed(d,p)['frame_count']==1
    p2=proposal(2);p2['analyzed_timestamp_ns']=p['analyzed_timestamp_ns']
    assert feed(d,p2)['status']=='unknown'
    assert feed(d,proposal(2))['status']=='waiting'
    assert feed(d,proposal(3))['status']=='waiting'
    assert feed(d,proposal(4))['needs_nudge'] is True


@pytest.mark.parametrize('change', [dict(generation=2),dict(target_hole_index=1),
    dict(context_id='new'),dict(measured_state14=[.02]+[0.]*13),
    dict(measured_state14=[0.]*7+[.02]+[0.]*6),dict(hole_px=[90.,100.])])
def test_context_pose_or_sign_change_restarts(change):
    d=PinNudgeDetector(dict(enabled=True))
    feed(d,proposal(1));feed(d,proposal(2))
    p=proposal(3);p.update(change)
    assert feed(d,p)['frame_count']==1


@pytest.mark.parametrize('change', [dict(nudge_valid=False),dict(localization_ready=False),
    dict(hole_source='cnn_unverified_fallback'),dict(stale_result=True),
    dict(pin_source='center_axis_inferred'),dict(measured_state14=[float('nan')]*14),
    dict(pin_px=[-1.,0.]),dict(frame_id=None),dict(status='error')])
def test_invalid_clears_evidence(change):
    d=PinNudgeDetector(dict(enabled=True))
    feed(d,proposal(1));feed(d,proposal(2))
    p=proposal(3);p.update(change)
    assert feed(d,p)['status']=='unknown'
    assert feed(d,proposal(4))['frame_count']==1


def test_stale_future_and_policy_rejected():
    d=PinNudgeDetector(dict(enabled=True));p=proposal()
    assert d.update(p,p['analyzed_timestamp_ns']+2_000_000_000,True)['status']=='unknown'
    assert d.update(p,p['analyzed_timestamp_ns']-1,True)['status']=='unknown'
    assert d.update(p,p['analyzed_timestamp_ns'],False)['status']=='unknown'
    assert feed(d,p,eligible=False)['status']=='unknown'


def test_aligned_requires_persistence_and_grippers_do_not_reset():
    d=PinNudgeDetector(dict(enabled=True))
    for i in range(1,4):
        p=proposal(i);p['hole_px']=[101.,100.];p['measured_state14'][6]=i
        result=feed(d,p)
    assert result['status']=='aligned' and result['needs_nudge'] is False


def test_config_validation_and_reset():
    d=PinNudgeDetector(dict(enabled=True));feed(d,proposal())
    for config in (dict(min_frames=2),dict(min_span_s=.9),dict(enabled=1),dict(unknown=1)):
        with pytest.raises(ValueError):d.configure(config)
    d.set_enabled(False);d.set_enabled(True)
    assert feed(d,proposal(2))['frame_count']==1


def test_long_gap_restarts_continuous_evidence():
    d=PinNudgeDetector(dict(enabled=True))
    feed(d,proposal(1));feed(d,proposal(2))
    assert feed(d,proposal(10))['frame_count']==1


@pytest.mark.parametrize('aligned', [False, True])
def test_repeated_current_frame_retains_decision_until_stale(aligned):
    d=PinNudgeDetector(dict(enabled=True))
    for i in range(1,4):
        p=proposal(i)
        if aligned:p['hole_px']=[101.,100.]
        result=feed(d,p)
    assert result['needs_nudge'] is (not aligned)
    assert feed(d,p)==result
    assert d.update(p,p['analyzed_timestamp_ns']+2_000_000_000,True)['status']=='unknown'


def test_regressing_frame_clears_decision_and_storage_is_constant():
    d=PinNudgeDetector(dict(enabled=True))
    for i in range(1,1001):feed(d,proposal(i))
    assert d._count==1000
    assert not any(isinstance(v,set) for v in vars(d).values())
    assert feed(d,proposal(999))['status']=='unknown'
    assert feed(d,proposal(1001))['frame_count']==1
