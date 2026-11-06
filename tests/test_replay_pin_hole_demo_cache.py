import json
from pathlib import Path
import pytest
from scripts.replay_pin_hole_demo_cache import causal_context, update_inputs, validate_replay_protocol, sha, ROOT


def test_pose_block_and_world_motion_do_not_reset_fixed_grasp():
    before = {'camera':'wrist_port2','recording_segment':0,'block_id':'B3','monotonic_ns':100,
              'right_arm_world_position':[0,0,0]}
    after = {**before,'block_id':'B4','monotonic_ns':200,'right_arm_world_position':[1,2,3]}
    assert causal_context(before,'episode') == causal_context(after,'episode')
    assert causal_context(before,'episode') != causal_context({**after,'recording_segment':1},'episode')
    assert causal_context(before,'episode') != causal_context({**after,'camera':'wrist_port3'},'episode')


def test_update_inputs_exclude_labels_future_references_and_block_identity():
    frame = {'camera':'wrist_port2','recording_segment':0,'monotonic_ns':1234,
             'block_id':'B4','hole_px':[300,200],'future_insertion_reference_frame':999}
    result = update_inputs(frame,'episode')
    assert set(result) == {'timestamp_ns','now_ns','sequence_id','camera_id','grasp_id'}
    assert result['timestamp_ns'] == result['now_ns'] == 1234
    assert 'B4' not in str(result) and '999' not in str(result)


def replay_fixture(tmp_path):
    paths = {}
    for name in ('bank','config','manifest','checkpoint','selected'):
        paths[name] = tmp_path/(name+'.json')
        paths[name].write_text('{}')
    paths['selected'].write_text(json.dumps({'mode':'affine_partial','max_hold_ns':2_000_000_000}))
    paths['config'].write_text(json.dumps({'mode':'affine_partial','max_hold_ns':30_000_000_000}))
    protocol = {'evaluation_protocol':'continuous_demonstration_replay','post_test_engineering_revision':True,
                'stationary_carrier_to_right_wrist':True,'reset_on_recording_segment_change':True,
                'fixed_grasp_assumption':True,'independent_unseen_test':False,
                'gt_or_future_test_reference_used_by_prediction':False,
                'selected_config_path':str(paths['selected']),'selected_config_sha256':sha(paths['selected']),
                'implementation_sha256':sha(ROOT/'app/pin_hole_demo_cache.py')}
    for key, name in [('references_sha256','bank'),('config_sha256','config'),
                      ('manifest_sha256','manifest'),('checkpoint_sha256','checkpoint')]:
        protocol[key] = sha(paths[name])
    paths['protocol'] = tmp_path/'protocol.json'
    paths['protocol'].write_text(json.dumps(protocol))
    return paths, protocol


def validate(paths):
    return validate_replay_protocol(*(paths[k] for k in ('protocol','bank','config','manifest','checkpoint')))


def test_replay_allows_hold_extension_but_not_new_feature_fitting(tmp_path):
    paths, protocol = replay_fixture(tmp_path)
    _, config = validate(paths)
    assert config['max_hold_ns'] == 30_000_000_000
    config['mode'] = 'homography'; paths['config'].write_text(json.dumps(config))
    protocol['config_sha256'] = sha(paths['config']); paths['protocol'].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='only stationary hold'):
        validate(paths)


def test_continuous_replay_cannot_be_relabelled_as_unseen_test(tmp_path):
    paths, protocol = replay_fixture(tmp_path)
    protocol['independent_unseen_test'] = True
    paths['protocol'].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='disclose reused'):
        validate(paths)
