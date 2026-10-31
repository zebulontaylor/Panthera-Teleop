"""Offline regression tests with independent arm/camera motion and real confounds."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from alignment_calibration_fit import fit_local


def samples(*, coupled=False, stationary=False, outliers=False):
    rng = np.random.default_rng(38)
    j = np.array([[160, 30, 0, 0, 0, 0], [20, 220, 0, 0, 0, 0],
                  [-150, 20, 0, 0, 0, 0], [40, 180, 0, 0, 0, 0.]])
    nuisance = rng.normal(0, 90, (4, 6))
    rows = []
    for episode in range(6):
        for k in range(30):
            ql = rng.normal(0, .025, 6)
            qr = np.r_[rng.normal(0, .04, 2), np.zeros(4)]
            if coupled:
                qr = ql.copy()
            if stationary:
                qr *= 0
            err = j @ qr + nuisance @ ql + rng.normal(0, .08, 4)
            if outliers and k % 17 == 0:
                err += rng.normal(0, 30, 4)
            rows.append(dict(episode_id=f'trial{episode}', time_s=k*.25,
                             q_left=ql.tolist(), q_right=qr.tolist(), error_px=err.tolist()))
    return rows, j


def test_identifies_only_excited_subspace_and_controls_other_arm():
    rows, expected = samples()
    fit = fit_local(rows, 'right', 'pin')
    assert fit['candidate_usable'], fit['reasons']
    assert fit['verified'] is False
    assert fit['identifiable_rank'] == 2
    np.testing.assert_allclose(fit['image_jacobian_px_per_rad'], expected, atol=2)
    assert len(fit['heldout_trials']) == 6
    assert fit['validation']['improvement_over_nuisance'] > .9
    assert fit['validation']['conditional_directional_accuracy'] > .95


def test_both_arms_moving_identically_cannot_identify_controlled_arm():
    rows, _ = samples(coupled=True)
    fit = fit_local(rows, 'right', 'pin')
    assert not fit['candidate_usable']
    assert fit['identifiable_rank'] == 0
    assert 'insufficient_independent_controlled_arm_excitation' in fit['reasons']


def test_camera_motion_without_controlled_arm_motion_cannot_certify():
    rows, _ = samples(stationary=True)
    fit = fit_local(rows, 'right', 'pin')
    assert not fit['candidate_usable']
    assert fit['identifiable_rank'] == 0


def test_robust_fit_limits_cnn_outlier_influence():
    rows, expected = samples(outliers=True)
    fit = fit_local(rows, 'right', 'pin')
    np.testing.assert_allclose(fit['image_jacobian_px_per_rad'], expected, atol=7)
    assert fit['downweighted_pair_count'] > 0


def test_invalid_and_short_data_report_reasons_without_crash():
    fit = fit_local([{'episode_id': 'x', 'time_s': float('nan')}], 'left', 'gear')
    assert fit['rejected_samples'] == 1
    assert fit['reasons'] == ['no_valid_temporal_pairs']
    rows, _ = samples()
    short = fit_local(rows[:2], 'right', 'gear')
    assert not short['candidate_usable']
    assert short['reasons'] == ['insufficient_local_pairs']


def test_never_pair_across_trials_or_long_tracking_gap():
    rows, _ = samples()
    for i, row in enumerate(rows):
        row['time_s'] = i*2.
    fit = fit_local(rows, 'right', 'pin')
    assert fit['temporal_pair_count'] == 0


def test_bootstrap_intervals_and_rank_are_reported_and_required():
    rows, expected = samples()
    fit = fit_local(rows, 'right', 'pin')
    bootstrap = fit['trial_bootstrap']
    assert bootstrap['replicates'] == 64
    assert bootstrap['minimum_identifiable_rank'] == 2
    assert bootstrap['minimum_direction_consistency'] >= .9
    low = np.asarray(bootstrap['coefficient_lower_95_px_per_rad'])
    high = np.asarray(bootstrap['coefficient_upper_95_px_per_rad'])
    assert low.shape == high.shape == (4, 6)
    assert np.all(low <= high)
    unchecked = fit_local(rows, 'right', 'pin', bootstrap_replicates=0)
    assert not unchecked['candidate_usable']
    assert 'trial_bootstrap_response_not_stable' in unchecked['reasons']


def test_changing_visual_mapping_across_trials_fails_heldout_validation():
    rows, _ = samples()
    for row in rows:
        if int(row['episode_id'][-1]) % 2:
            row['error_px'] = (-np.asarray(row['error_px'])).tolist()
    fit = fit_local(rows, 'right', 'pin')
    assert not fit['candidate_usable']
    assert 'heldout_prediction_does_not_beat_baselines' in fit['reasons']



def load_tests(loader, tests, pattern):
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in globals().items()
                              if name.startswith('test_') and callable(test))


if __name__ == '__main__':
    import unittest
    unittest.main()
