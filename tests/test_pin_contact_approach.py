import numpy as np
import pytest
from app.pin_contact_approach import ContactApproach
from app.pin_alignment_search import LoadedDropMonitor


def test_free_descent_beyond_old_guard_then_settlement_and_fresh_insertion_origin():
    approach = ContactApproach(0., [0., 0., .05])
    for tick in range(1, 201):
        now = tick*.02
        z = .05-min(now*.004, .008)
        out = approach.update(now, [0., 0., z], load_established=now>=.3)
        assert out['stop_reason'] is None
        if now<2.3:
            assert not out['contact_ready']
        if out['contact_ready']:
            break
    assert out['contact_ready'] and out['downward_from_start_m'] == pytest.approx(.008)
    # Settling consumed 8mm, but loaded insertion gets its own fixed origin.
    loaded = LoadedDropMonitor(now, [0., 0., z])
    assert loaded.update(now, [0., 0., z], False, contact_candidate=True)['stop_reason'] is None


def test_stationary_without_established_load_never_claims_contact():
    approach = ContactApproach(0., [0., 0., .05])
    for tick in range(1, 151):
        assert not approach.update(tick*.02, [0., 0., .05], load_established=False)['contact_ready']


@pytest.mark.parametrize('cause', ['travel', 'time', 'gap'])
def test_approach_has_finite_limits_and_latches_failure(cause):
    approach = ContactApproach(0., [0., 0., .05])
    if cause == 'travel':
        out = approach.update(.02, [0., 0., .0299], load_established=True)
        expected = 'contact_approach_travel_limit'
    elif cause == 'gap':
        out = approach.update(.2, [0., 0., .05], load_established=True)
        expected = 'measurement_gap_stop'
    else:
        for tick in range(1, 1502):
            out = approach.update(tick*.02, [0., 0., .05], load_established=False)
        expected = 'contact_approach_timeout'
    assert out['stop_reason'] == expected and not out['contact_ready']
    assert approach.update(approach.last_time+.02, [0., 0., .05], load_established=True)['stop_reason'] == expected


def test_height_noise_restarts_contact_dwell():
    approach = ContactApproach(0., [0., 0., .05])
    for tick in range(1, 151):
        z = .05+(tick%2)*.001
        assert not approach.update(tick*.02, [0., 0., z], load_established=True)['contact_ready']


def test_slow_free_drift_inside_noise_band_is_not_contact_even_with_saturated_effort():
    approach = ContactApproach(0., [0., 0., .05])
    for tick in range(1, 201):
        now = tick*.02
        assert not approach.update(now, [0., 0., .05-now*.0004], load_established=True)['contact_ready']
