import numpy as np
import pytest
from app.pin_alignment_insert import (
    AutoInsertion, AUTO_NUDGE_INTERVAL_S, AUTO_NUDGE_PASSES,
    AXIAL_RETRY_SPEED_M_S, AXIAL_RETRY_LIMIT_M,
    loaded_xy_nudge_goal, LoadedXYEnvelopeError,
)
from app.pin_alignment_search import SearchConfig


def preview(frame, now, delta=(.001, 0., 0.)):
    return dict(frame_id=frame, analyzed_timestamp_ns=int(now*1e9), delta=delta)


def validate(proposal, frame_id, *, applying=False):
    assert proposal['frame_id'] == frame_id
    assert applying  # Same current-pose validation as the loaded nudge button.
    return proposal['delta']


def test_three_button_style_nudges_then_axial_even_without_settlement_or_detection():
    start = np.array([.3, .4, .1])
    auto = AutoInsertion(start, SearchConfig(), 0.)
    # Tool deliberately stays stuck: no convergence, baseline, face-quiet,
    # aligned-preview or tracking-stall gate may block this sequence.
    for frame in range(1, 4):
        now = float(frame-1)
        out = auto.update(now, start, preview(frame, now), validate, baseline_armed=False)
        assert out['nudge_count'] == frame
        np.testing.assert_allclose(out['xy_goal'], loaded_xy_nudge_goal(start, [.001, 0, 0], start))
        assert out['phase'] == 'nudging'
        assert not out['preview_unavailable']
    out = auto.update(3., start, None, validate)
    assert out['phase'] == 'axial_retry'
    assert out['waiting_reason'] is None
    assert out['axial_retry_m'] == 0.
    origin = out['xy_goal'].copy()
    for tick in range(1, 1001):
        now = 3. + tick*.01
        out = auto.update(now, start, None, validate)
        distance = min(AXIAL_RETRY_LIMIT_M, tick*.01*AXIAL_RETRY_SPEED_M_S)
        np.testing.assert_allclose(out['xy_goal'], origin-np.array([.6, .8])*distance, atol=1e-12)
        assert out['stop_reason'] is None
        assert out['nudge_count'] == AUTO_NUDGE_PASSES
        assert not out['preview_unavailable']
        np.testing.assert_array_equal(out['dither_xy'], [0, 0])
    assert AXIAL_RETRY_SPEED_M_S == .002
    assert out['phase'] == 'axial_limit'
    assert out['axial_retry_m'] == .015


def test_pass_spacing_is_one_second_not_measured_settlement():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    auto.update(0., [.4, 0, 0], preview(1, 0), validate)
    out = auto.update(.99, [.4, 0, 0], preview(2, .99), validate)
    assert out['nudge_count'] == 1
    assert out['phase'] == 'nudging'
    out = auto.update(AUTO_NUDGE_INTERVAL_S, [.4, 0, 0], preview(2, .99), validate)
    assert out['nudge_count'] == 2


def test_zero_corrections_count_as_passes_and_cannot_wait_forever_in_deadband():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    for frame in range(3):
        out = auto.update(float(frame), [.4, 0, 0], preview(frame, frame, (0, 0, 0)), validate)
        assert out['nudge_count'] == frame+1
        np.testing.assert_array_equal(out['xy_goal'], [.4, 0])
    assert auto.update(3., [.4, 0, 0], None, validate)['phase'] == 'axial_retry'


def test_reused_frame_is_not_automatically_pressed_three_times():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    first = preview(1, 0)
    auto.update(0., [.4, 0, 0], first, validate)
    out = auto.update(1., [.4002, 0, 0], first, validate)
    assert out['nudge_count'] == 1
    assert out['phase'] == 'waiting_for_preview'
    assert out['waiting_reason'] == 'Waiting for a different nudge frame'
    np.testing.assert_allclose(out['xy_goal'], [.401, 0])


def test_zero_passes_preserve_previous_target_despite_encoder_drift():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    auto.update(0., [.4, 0, 0], preview(1, 0), validate)
    for frame in (2, 3):
        out = auto.update(float(frame-1), [.399, .001, 0], preview(frame, frame-1, (0, 0, 0)), validate)
        assert out['nudge_count'] == frame
        np.testing.assert_allclose(out['xy_goal'], [.401, 0])


def test_missing_or_rejected_preview_preserves_goal_and_reports_exact_reason():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    auto.update(0., [.4, 0, 0], preview(1, 0), validate)
    out = auto.update(1., [.4, 0, 0], None, validate)
    assert out['waiting_reason'] == 'Alignment preview unavailable'
    def refuse(*args, **kwargs):
        raise RuntimeError('Arm pose moved since the analyzed frame')
    out = auto.update(2., [.4, 0, 0], preview(2, 2), refuse)
    assert out['preview_unavailable']
    assert 'Arm pose moved' in out['waiting_reason']
    np.testing.assert_allclose(out['xy_goal'], [.401, 0])
    out = auto.update(2.1, [.4, 0, 0], preview(3, 2.1), validate)
    assert not out['preview_unavailable']
    assert out['waiting_reason'] is None
    assert out['nudge_count'] == 2


def test_axial_does_not_backfill_time_spent_waiting_for_third_frame():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    for now in (0., 1.):
        auto.update(now, [.4, 0, 0], preview(int(now), now), validate)
    auto.update(10., [.4, 0, 0], None, validate)
    auto.update(20., [.4, 0, 0], preview(3, 20), validate)
    out = auto.update(21., [.4, 0, 0], None, validate)
    assert out['axial_retry_m'] == 0.
    out = auto.update(21.1, [.4, 0, 0], None, validate)
    assert out['axial_retry_m'] == pytest.approx(.0002)


def test_axial_goal_never_reanchors_to_encoder_drift():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    for frame in range(3):
        auto.update(float(frame), [.4, 0, 0], preview(frame, frame, (0, 0, 0)), validate)
    auto.update(3., [.4, 0, 0], None, validate)
    out = auto.update(3.1, [.41, .01, 0], None, validate)
    np.testing.assert_allclose(out['xy_goal'], [.3998, 0])


def test_visual_and_axial_targets_share_original_travel_envelope():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    out = auto.update(0., [.429, 0, 0], preview(1, 0, (.002, 0, 0)), validate)
    assert out['stop_reason'] == 'auto_xy_envelope'
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    for frame in range(3):
        auto.update(float(frame), [.371, 0, 0], preview(frame, frame, (-.0001, 0, 0)), validate)
    auto.update(3., [.371, 0, 0], None, validate)
    auto.update(3.1, [.371, 0, 0], None, validate)
    for now in (3.2, 3.3, 3.4):
        assert auto.update(now, [.371, 0, 0], None, validate)['stop_reason'] is None
    assert auto.update(3.5, [.371, 0, 0], None, validate)['stop_reason'] == 'auto_xy_envelope'


def test_overall_timeout_and_invalid_measurement_still_latch_stop():
    auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
    assert auto.update(120., [.4, 0, 0], None, validate)['stop_reason'] == 'time_limit'
    assert auto.update(121., [.4, 0, 0], preview(1, 121), validate)['stop_reason'] == 'time_limit'
    for now, point in [(float('nan'), [.4, 0, 0]), (-1., [.4, 0, 0]), (0., [float('nan'), 0, 0])]:
        auto = AutoInsertion([.4, 0, 0], SearchConfig(), 0.)
        assert auto.update(now, point, None, validate)['stop_reason'] == 'invalid_auto_measurement'


@pytest.mark.parametrize('delta', [[.003, .004, 0], [-.005, 0, 0], [0, 0, 0]])
def test_shared_loaded_button_goal(delta):
    goal = loaded_xy_nudge_goal([.4, 0, .1], delta, [.4, 0, .1])
    np.testing.assert_allclose(goal, np.array([.4, 0])+np.array(delta[:2]))


@pytest.mark.parametrize('delta', [[.0031, .004, 0], [float('nan'), 0, 0], [0, 0]])
def test_shared_loaded_button_rejects_invalid_correction(delta):
    with pytest.raises(ValueError):
        loaded_xy_nudge_goal([.4, 0, .1], delta, [.4, 0, .1])


def test_shared_loaded_button_rejects_outside_original_envelope():
    with pytest.raises(LoadedXYEnvelopeError):
        loaded_xy_nudge_goal([.429, 0, .1], [.002, 0, 0], [.4, 0, .1])
