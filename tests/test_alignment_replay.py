"""Counterfactual replay must be checked against independent recorded motion."""
import numpy as np
from scripts.validate_alignment_replay import dls, evaluate, training_reference, temporal_pair_rows


def synthetic(reverse_heldout=False):
    rng=np.random.default_rng(22)
    samples=[]; split={}
    j=rng.normal(size=(6,4))*200
    for ep in range(17):
        name=f'trial-{ep:02d}'; split[name]='train' if ep<13 else 'validation'
        # Independent tiny random motion keeps every trial within the local pose.
        q=np.cumsum(rng.normal(size=(40,6))*.008,axis=0)
        error=q @ j
        if ep>=13 and reverse_heldout: error=-error
        for t,(pose,e) in enumerate(zip(q,error)):
            samples.append(dict(episode_id=name,time_s=t*.1,q_left=pose.tolist(),q_right=[0.]*6,error_px=e.tolist()))
    return samples,split


def test_heldout_direction_catches_reversed_actual_response():
    normal,split=synthetic(); reverse,_=synthetic(True)
    good=evaluate(normal,'gear',split)['summary']
    bad=evaluate(reverse,'gear',split)['summary']
    assert good['direction_accuracy']==1.
    assert bad['direction_accuracy']==0.
    assert bad['rmse_px']>bad['zero_motion_rmse_px']
    assert good['rmse_px']<1e-8


def test_objective_and_envelope_do_not_use_validation_outcomes():
    normal,split=synthetic(); reverse,_=synthetic(True)
    first=evaluate(normal,'gear',split)['folds'][0]
    other=evaluate(reverse,'gear',split)['folds'][0]
    assert first['training_episode_count']==13
    assert first['target_error_px_train_only']==other['target_error_px_train_only']
    assert first['reference_joint_positions_train_only']==other['reference_joint_positions_train_only']


def test_dls_cap_and_direction():
    j=np.eye(2,6)*100
    delta=dls(j,[20.,-10.])
    assert np.isclose(np.linalg.norm(delta),.006)
    assert delta[0]<0 and delta[1]>0
    assert np.linalg.norm(np.array([20.,-10.])+j @ delta)<np.linalg.norm([20.,-10.])


def test_repeated_joint_pose_preserves_original_frame_and_error():
    from scripts.alignment_calibration_fit import _pairs
    samples,split=synthetic()
    # Deliberately repeat the first validation pose at a much later frame,
    # with a distinct target/image error. Pose-key lookup would select it.
    episode='trial-13'
    rows=[r for r in samples if r['episode_id']==episode]
    rows[9]['q_left']=rows[0]['q_left'][:]
    rows[9]['error_px']=[1000.,2000.,3000.,4000.]
    paired=temporal_pair_rows(rows)
    fitter_pairs,_=_pairs(rows,.2,.5)
    assert len(paired)==len(fitter_pairs)
    for (start,end),(_,q0,q1,dy) in zip(paired,fitter_pairs):
        assert np.array_equal(q0,np.r_[start['q_left'],start['q_right']])
        assert np.array_equal(q1,np.r_[end['q_left'],end['q_right']])
        assert np.array_equal(dy,np.asarray(end['error_px'])-np.asarray(start['error_px']))
    fold=next(f for f in evaluate(samples,'gear',split)['folds'] if f['episode_id']==episode)
    pair=fold['evaluated_pairs'][0]
    assert pair['time_s']==0.
    assert pair['end_time_s']==.2
    np.testing.assert_array_equal(pair['start_error_px'],np.asarray(rows[0]['error_px'])[[1,3]])
    residual=np.asarray(pair['start_error_px'])-np.asarray(fold['target_error_px_train_only'])[[1,3]]
    observed=np.asarray(pair['observed_image_change_px'])
    assert np.isclose(pair['observed_squared_error_reduction_px2'],residual @ residual-(residual+observed) @ (residual+observed))


def load_tests(loader, tests, pattern):
    """Expose standalone regression checks to the project's unittest runner."""
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in (
        test_heldout_direction_catches_reversed_actual_response,
        test_objective_and_envelope_do_not_use_validation_outcomes,
        test_dls_cap_and_direction,
        test_repeated_joint_pose_preserves_original_frame_and_error,
    ))
