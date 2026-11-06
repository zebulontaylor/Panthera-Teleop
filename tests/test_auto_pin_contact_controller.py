"""Automatic contact handoff ownership and motor guards, without devices."""
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import Mock, patch
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import inference_dashboard as dashboard
from test_inference_dashboard import GuardedForceHardware, wait_for, fake_request, OperatorAlignmentNudgeTests


def test_arming_authorizes_automatic_handoff_but_does_not_fake_manual_contact():
    c=dashboard.Controller('unused')
    assert not c.snapshot()['auto_pin_contact']['enabled']
    with pytest.raises(ValueError): c.configure_auto_pin_contact(1)
    with pytest.raises(ValueError): c.configure_auto_pin_contact(True,dict(force_n=4))
    with pytest.raises(ValueError): c.force_load_config({})
    c.configure_auto_pin_contact(True,dict(force_n=.5))
    assert c.snapshot()['auto_pin_contact']['enabled']
    assert c._force_requested is None
    c.configure_auto_pin_contact(False)
    assert not c.snapshot()['auto_pin_contact']['enabled']


@pytest.mark.parametrize('disable', [True,False])
def test_aligned_contact_pauses_policy_and_starts_guarded_assist_once(disable):
    hardware=GuardedForceHardware()
    hardware.guarded_load_limits = dict(hardware.guarded_load_limits,
        contact_approach=dict(max_down_m=.020, max_xy_m=.005, max_duration_s=30.),
        loaded_insertion=dict(max_up_m=.005))
    def request_fn(base,path,body=None,timeout=3):
        if path=='/health':return fake_request(base,path,body,timeout)
        q=hardware.position.copy();q[0]+=.002
        return dict(sequence=body['sequence'],fps=20,checkpoint='saved',mock=False,actions=[q.tolist()]*50)
    with tempfile.TemporaryDirectory() as directory, patch.object(dashboard,'request',side_effect=request_fn):
        c=dashboard.Controller('unused',lambda:hardware,log_path=Path(directory)/'steps.jsonl',data_dir=directory)
        c.connect(reset_pose=False)
        try:
            wait_for(lambda:c.snapshot()['mode']=='holding')
            c.heartbeat();c.request_right_arm_freeze(True)
            wait_for(lambda:c.snapshot()['right_arm_frozen'])
            monitor=Mock()
            monitor.status.side_effect=lambda: OperatorAlignmentNudgeTests.proposal(hardware.position,nudge_left_mm=[0,0,0])
            c.alignment=monitor
            c.configure_auto_pin_contact(True)
            def detection(*args,**kwargs):
                trigger=kwargs['eligible'] and c.state['mode']=='executing'
                return dict(enabled=True,status='triggered' if trigger else 'observing',phase='contact_candidate',
                    triggered=trigger,reason='test measured contact candidate',contact_confirmed=False)
            with patch.object(c.pin_contact_detector,'update',side_effect=detection):
                c.submit('play',15,0)
                wait_for(lambda:len(hardware.load_commands)>=3)
                s=c.snapshot()
                assert not s['playing'] and s['mode']=='force_auto'
                assert s['right_arm_frozen']
                assert s['alignment_force']['automatic_pin_contact']
                assert not c.pause_event.is_set()
                np.testing.assert_allclose(hardware.load_commands[-1]['target'][7:],s['right_arm_freeze_target'])
                if disable:c.configure_auto_pin_contact(False)
                else:c.request_alignment_force_stop()
                wait_for(lambda:c.snapshot()['mode']=='holding')
                n=len(hardware.load_commands);time.sleep(.08)
                assert len(hardware.load_commands)==n
                assert not c.snapshot()['playing']
            events=[json.loads(x) for x in c.log_path.read_text().splitlines()]
            handoffs=[x for x in events if x.get('event')=='auto_pin_contact_handoff']
            starts=[x for x in events if x.get('event')=='auto_pin_contact_force_start']
            assert len(handoffs)==len(starts)==1
            assert not starts[0]['operator_pin_on_carrier']
            assert not handoffs[0]['physical_contact_confirmed']
        finally:
            c.alignment=None;c.shutdown();c.worker.join(3)


def test_automatic_approach_descends_past_old_limit_before_nudges_and_reanchors_once():
    import pin_alignment_force
    import pin_alignment_motion

    class LinearKinematics:
        def pose(self, q):
            return np.asarray(q[:3]).copy(), np.eye(3)
        def jacobian(self, q):
            return np.eye(6)

    class DescendingHardware(GuardedForceHardware):
        guarded_load_limits = dict(GuardedForceHardware.guarded_load_limits,
            contact_approach=dict(max_down_m=.020, max_xy_m=.005, max_duration_s=30.),
        loaded_insertion=dict(max_up_m=.005))
        def __init__(self):
            super().__init__()
            self.origin = self.position[2]
            self.history = []
            self.approach_started = None
        def command_guarded_load(self, arms, target, assist, guard):
            super().command_guarded_load(arms, target, assist, guard)
            self.history.append(('load', dict(guard)))
            if guard.get('phase') == 'contact_approach':
                if self.approach_started is None:
                    self.approach_started = time.monotonic()
                elapsed = time.monotonic()-self.approach_started
                u = min(1., elapsed)
                distance = .01*(u**3-.5*u**4+max(0., elapsed-1.))
                self.position[2] = self.origin-min(.008, distance)
        def command(self, target):
            super().command(target)
            self.history.append(('hold', None))

    hardware = DescendingHardware()
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(dashboard, 'request', side_effect=fake_request), \
            patch.object(pin_alignment_motion, 'NativeToolKinematics', LinearKinematics), \
            patch.object(pin_alignment_force, 'NativeToolKinematics', LinearKinematics), \
            patch.object(pin_alignment_force, 'APPROACH_SPEED_M_S', .01):
        c = dashboard.Controller('unused', lambda:hardware, log_path=Path(directory)/'steps.jsonl', data_dir=directory)
        c.connect(reset_pose=False)
        try:
            wait_for(lambda:c.snapshot()['mode']=='holding')
            c.heartbeat(); c.request_right_arm_freeze(True)
            wait_for(lambda:c.snapshot()['right_arm_frozen'])
            c.alignment = Mock()
            c.alignment.status.return_value = {}
            nudge_after_contact = []
            def validate_after_contact(*args, **kwargs):
                kind, guard = hardware.history[-1]
                nudge_after_contact.append(kind=='hold' or guard.get('phase') != 'contact_approach')
                return np.zeros(3)
            c.validate_nudge_preview = Mock(side_effect=validate_after_contact)
            c.configure_auto_pin_contact(True)
            with c.lock:
                c._force_requested = dict(force_n=.5, load_config=c._auto_contact_load[1],
                    auto_insert=True, automatic_pin_contact=True, generation=c._nudge_generation)
            deadline = time.monotonic()+7
            while time.monotonic()<deadline:
                c.heartbeat()
                loads = hardware.load_commands
                if loads and loads[-1]['guard'].get('phase') != 'contact_approach':
                    break
                time.sleep(.01)
            else:
                pytest.fail(str(c.snapshot()))
            first = hardware.load_commands[0]['guard']
            last = hardware.load_commands[-1]['guard']
            assert first['phase']=='contact_approach' and first['max_down_m']==.020
            assert last['phase']=='loaded_insertion' and last['max_down_m']==.004
            assert last['max_up_m']==.005 and last['transition_from']==first['id']
            assert first['anchor_tool_position_m'][2]-last['anchor_tool_position_m'][2]==pytest.approx(.008)
            ids = {x['guard']['id'] for x in hardware.load_commands}
            assert len(ids)==2
            first_loaded = next(i for i,(kind,guard) in enumerate(hardware.history)
                if kind=='load' and guard['id']==last['id'])
            first_approach = next(i for i,(kind,guard) in enumerate(hardware.history) if kind=='load')
            assert all(kind=='load' for kind,_ in hardware.history[first_approach:first_loaded+1])
            np.testing.assert_allclose(hardware.load_commands[first_loaded-first_approach]['assist'],
                hardware.load_commands[first_loaded-first_approach-1]['assist'], atol=.011, rtol=0)
            assert not nudge_after_contact
            settled_deadline = time.monotonic()+2.5
            while not nudge_after_contact and time.monotonic()<settled_deadline:
                c.heartbeat(); time.sleep(.01)
            assert nudge_after_contact and all(nudge_after_contact)
            c.request_alignment_force_stop()
            wait_for(lambda:c.snapshot()['mode']=='holding')
        finally:
            c.alignment=None; c.shutdown(); c.worker.join(3)
