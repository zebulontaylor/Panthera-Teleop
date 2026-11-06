import pytest
from scripts.evaluate_pin_insertion_independent_domain import audit_domain


def test_independent_domain_excludes_all_ancestor_recordings_and_sessions():
    training={'_split_audit':{'source_recording_ids':['new_recording']},'frames':[{'episode_id':'new_recording::B1','session_id':'new_grasp','split':'train'}]}
    baseline={'adaptation':{'train_episode_ids':['old_train'],'manifest_metadata':{'episodes':[{'episode_id':'old_train','session_id':'old_grasp','split':'train'}]}}}
    candidate={'adaptation':{'ancestor_optimization_and_selection_ids':['ancestor']}}
    row={'episode_id':'independent','session_id':'independent_grasp'}
    result=audit_domain([row],training,baseline,candidate)
    assert result['optimization_recording_overlap']==[]
    with pytest.raises(ValueError,match='recording source'):audit_domain([dict(row,episode_id='ancestor')],training,baseline,candidate)
    with pytest.raises(ValueError,match='recording source'):audit_domain([dict(row,source_recording_id='new_recording')],training,baseline,candidate)
    with pytest.raises(ValueError,match='session/grasp'):audit_domain([dict(row,session_id='old_grasp')],training,baseline,candidate)
