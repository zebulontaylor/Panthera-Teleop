import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from pin_alignment_search import SearchConfig, SpiralSearch, LoadedDropMonitor


def arm(monitor,start=0.,z=.1):
    for t in np.arange(start,start+.61,.1):
        out=monitor.update(float(t),[0,0,z],True)
    assert out['phase']=='armed'
    return out


def test_current_pose_center_outward_nominal_radius_speed_and_finish():
    config = SearchConfig(oscillation_amplitude_m=0)
    start = np.array([.021, -.013, .1])
    pattern = SpiralSearch(start, config)
    start[:] = 99  # Caller mutation must not move the captured center.
    first = pattern.sample(0)
    np.testing.assert_array_equal(first['target_xyz'], [.021, -.013, .1])
    assert first['center_source'] == 'current_pose'
    assert first['phase'] == 'spiral'
    np.testing.assert_array_equal(first['center_xyz'], first['target_xyz'])
    times = np.linspace(0, pattern.duration_s, 3000)
    samples = [pattern.sample(t) for t in times]
    points = np.array([sample['nominal_target_xyz'] for sample in samples])
    radius = np.linalg.norm(points[:, :2] - [.021, -.013], axis=1)
    assert np.all(np.diff(radius) >= -1e-10)
    assert radius[-1] == pytest.approx(config.radius_m)
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(times)
    assert np.max(speed) <= config.speed_m_s + 1e-10
    assert np.all(points[:, 2] == .1)
    np.testing.assert_allclose([s['target_xyz'] for s in samples], points)
    assert samples[-1]['done']
    assert samples[-1]['stop_reason'] == 'pattern_complete'


def test_pattern_timeout_bounds_and_invalid_samples():
    pattern = SpiralSearch([0, 0, 0], SearchConfig(max_duration_s=.1))
    assert pattern.sample(.1)['stop_reason'] == 'time_limit'
    np.testing.assert_array_equal(pattern.sample(.1)['target_xyz'],
                                  pattern.sample(100)['target_xyz'])
    for kwargs in [dict(radius_m=.021), dict(turns=11), dict(speed_m_s=.011),
                   dict(max_duration_s=301), dict(max_down_travel_m=.006),
                   dict(insertion_drop_m=.0001), dict(oscillation_amplitude_m=-1),
                   dict(oscillation_frequency_hz=float('nan'))]:
        with pytest.raises(ValueError):
            SearchConfig(**kwargs)
    for elapsed in [float('nan'), float('inf'), -.1]:
        with pytest.raises(ValueError):
            pattern.sample(elapsed)


def test_default_diagonal_oscillation_amplitude_frequency_and_unchanged_z():
    config = SearchConfig()
    assert config.oscillation_amplitude_m == .0002
    assert config.oscillation_frequency_hz == 15.
    pattern = SpiralSearch([.01, -.02, .3], config)
    period = 1 / config.oscillation_frequency_hz
    for cycles, sign in [(0, 0), (.25, 1), (.5, 0), (.75, -1), (1, 0), (1.25, 1)]:
        sample = pattern.sample(cycles * period)
        expected = np.array([1., 1., 0.]) * sign * config.oscillation_amplitude_m / np.sqrt(2)
        np.testing.assert_allclose(sample['oscillation_offset_xyz'], expected, atol=1e-15)
        np.testing.assert_allclose(np.array(sample['target_xyz']) - sample['nominal_target_xyz'],
                                   expected, atol=1e-15)
        assert sample['target_xyz'][2] == .3
        if sign:
            assert np.linalg.norm(sample['oscillation_offset_xyz']) == pytest.approx(config.oscillation_amplitude_m)
    # Non-synchronous samples check the full sine, rather than aliasing it at whole cycles.
    for elapsed in [.0137, .0271, .0433]:
        sample = pattern.sample(elapsed)
        offset = config.oscillation_amplitude_m / np.sqrt(2) * np.sin(2*np.pi*15*elapsed)
        np.testing.assert_allclose(sample['oscillation_offset_xyz'], [offset, offset, 0], atol=1e-15)


@pytest.mark.parametrize('settings', [dict(oscillation_amplitude_m=0), dict(oscillation_frequency_hz=0)])
def test_either_zero_setting_disables_oscillation(settings):
    pattern = SpiralSearch([.01, .02, .3], SearchConfig(**settings))
    for elapsed in [0., .017, .04, 1.]:
        sample = pattern.sample(elapsed)
        np.testing.assert_array_equal(sample['oscillation_offset_xyz'], [0, 0, 0])
        np.testing.assert_array_equal(sample['target_xyz'], sample['nominal_target_xyz'])


def test_initial_settling_is_not_insertion_and_stable_baseline_required():
    monitor=LoadedDropMonitor(0,[0,0,.1])
    for i in range(6):
        out=monitor.update(i*.1,[0,0,.1-i*.00015],True)
        assert not out['done']
        assert out['baseline_z_m'] is None
    arm(monitor,start=.6,z=.0995)
    assert monitor.baseline_z==.0995


def test_drop_debounce_reset_and_candidate_is_not_proof():
    m=LoadedDropMonitor(0,[0,0,.1]);arm(m)
    assert not m.update(.7,[0,0,.0969],True)['done']
    assert not m.update(.8,[0,0,.1],True)['done']
    assert not m.update(.9,[0,0,.0969],True)['done']
    assert not m.update(1.,[0,0,.0969],True)['done']
    out=m.update(1.11,[0,0,.0969],True)
    assert out['done'] and out['stop_reason']=='insertion_candidate'
    assert not out['insertion_confirmed']
    assert m.update(1.2,[0,0,.1],True)['stop_reason']=='insertion_candidate'


def test_independent_hard_travel_and_contact_loss_stale_data():
    m=LoadedDropMonitor(0,[0,0,.1])
    assert m.update(.1,[0,0,.095],False)['stop_reason']=='hard_down_travel_stop'
    m=LoadedDropMonitor(0,[0,0,.1]);arm(m)
    assert m.update(.7,[0,0,.1],False)['stop_reason']=='operator_contact_confirmation_missing'
    m=LoadedDropMonitor(0,[0,0,.1]);arm(m)
    assert m.update(1.,[0,0,.099],True)['stop_reason']=='measurement_gap_stop'
    m=LoadedDropMonitor(0,[0,0,.1])
    assert m.update(1.,[0,0,.099],True)['stop_reason']=='measurement_gap_stop'
    with pytest.raises(ValueError):m.update(.5,[0,0,.1],True)
    with pytest.raises(ValueError):m.update(1.1,[0,0,float('nan')],True)


def test_loading_drop_can_settle_without_stopping_or_becoming_insertion():
    m=LoadedDropMonitor(0,[0,0,.1],SearchConfig(insertion_drop_m=.0008))
    # Initial settling exceeds the insertion threshold for longer than debounce.
    for t in [.1,.2,.31,.4,.5,.61,.7,.8,.9]:
        out=m.update(t,[0,0,.099],True)
        assert not out['done']
    assert out['phase']=='armed'
    assert out['baseline_z_m']==pytest.approx(.099)
    # Early settling is excluded from subsequent candidate detection.
    for t in [1.,1.1,1.2]:
        assert not m.update(t,[0,0,.0986],True)['done']
    # A later sustained drop from the actual loaded height still stops.
    for t in [1.3,1.4,1.51]:
        out=m.update(t,[0,0,.098],True)
    assert out['stop_reason']=='insertion_candidate'
    assert not out['insertion_confirmed']


def test_initial_settling_still_counts_toward_hard_travel_limit():
    m=LoadedDropMonitor(0,[0,0,.1])
    for t,z in [(.1,.099),(.2,.098),(.3,.097)]:
        assert not m.update(t,[0,0,z],True)['done']
    assert m.update(.4,[0,0,.0959],True)['stop_reason']=='hard_down_travel_stop'


def test_confirmation_withdrawal_and_sample_gap_stop_before_baseline():
    m=LoadedDropMonitor(0,[0,0,.1])
    assert m.update(.1,[0,0,.1],False)['stop_reason']=='operator_contact_confirmation_missing'
    m=LoadedDropMonitor(0,[0,0,.1])
    assert m.update(.3,[0,0,.1],True)['stop_reason']=='measurement_gap_stop'
    assert SearchConfig().baseline_stability_m == .0002


def test_large_radius_turn_count_and_long_configurable_duration():
    assert SearchConfig().max_duration_s == 120.
    config = SearchConfig(radius_m=.020, turns=10, speed_m_s=.010,
                          max_duration_s=300, oscillation_amplitude_m=.002,
                          oscillation_frequency_hz=40)
    pattern = SpiralSearch([.1, -.1, .2], config)
    assert pattern.completion_s > 30
    assert pattern.completion_s < 300
    for elapsed in np.linspace(0, pattern.duration_s, 300):
        sample = pattern.sample(elapsed)
        nominal = np.array(sample['nominal_target_xyz'])
        assert np.linalg.norm(nominal[:2] - [.1, -.1]) <= .0200000001
        assert np.linalg.norm(np.array(sample['target_xyz'])[:2] - [.1, -.1]) <= .0220000001
    last = pattern.sample(pattern.duration_s)
    assert np.linalg.norm(np.array(last['nominal_target_xyz'])[:2] - [.1, -.1]) == pytest.approx(.020)
    assert last['done'] and last['stop_reason'] == 'pattern_complete'


def test_inferred_contact_baseline_does_not_claim_operator_confirmation():
    monitor=LoadedDropMonitor(0,[0,0,.1])
    for t in np.arange(0,.61,.1):
        out=monitor.update(float(t),[0,0,.1],contact_confirmed=False,contact_candidate=True)
    assert out['phase']=='armed'
    assert out['baseline_authority']=='inferred_contact_candidate'
    assert not out['insertion_confirmed']
    with pytest.raises(ValueError):
        monitor.update(.7,[0,0,.1],False,contact_candidate=1)
