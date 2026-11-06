import base64
import time
import pytest
from app.pin_alignment_monitor import PinAlignmentMonitor, validated_config


def test_config_validation():
    assert validated_config({'gain': .2})['hold_s'] == 30
    assert validated_config({})['frequency_hz'] == 20.
    assert validated_config({'frequency_hz': 20.})['frequency_hz'] == 20.
    with pytest.raises(ValueError):
        validated_config({'frequency_hz': 20.1})
    for update in ({'gain': float('nan')}, {'gain': True}, {'unknown': 1}, {'hold_s': -1}, {'enabled': 1}):
        with pytest.raises(ValueError):
            validated_config(update)


def test_completed_result_publishes_despite_newer_pending_and_invalidates():
    seen = []
    monitor = PinAlignmentMonitor(callback=seen.append)
    try:
        monitor._request_id = 9  # A later submitted frame must not suppress frame 3.
        request = {'request_id': 3, 'generation': 0}
        result = dict(status='ready', ready=True, nudge_valid=True,
                      analyzed_timestamp_ns=time.monotonic_ns(),
                      nudge_left_mm=[1, 0, 0],
                      analyzed_jpeg_b64=base64.b64encode(b'exact-image').decode())
        monitor._publish(result, request, {}, time.monotonic())
        assert seen[-1]['frame_id'] == 3
        assert monitor.analyzed_frame() == (3, b'exact-image')
        epoch = monitor._cache_epoch
        monitor.configure(gain=.3)
        assert monitor._cache_epoch == epoch
        assert not monitor.status()['ready']
        assert monitor.analyzed_frame() == (None, None)
        monitor._publish(dict(status='ready', ready=True), request, {}, time.monotonic())
        assert len(seen) == 1
        monitor.reset()
        assert monitor._cache_epoch == epoch + 1
    finally:
        monitor.close()


def test_age_invalidates_proposal_without_hiding_analyzed_image():
    monitor = PinAlignmentMonitor()
    try:
        monitor._publish(dict(status='ready', ready=True, nudge_valid=True,
            analyzed_timestamp_ns=time.monotonic_ns()-6_000_000_000,
            nudge_left_mm=[1, 0, 0], analyzed_jpeg_b64=base64.b64encode(b'x').decode()),
            {'request_id': 1, 'generation': 0}, {}, time.monotonic())
        status = monitor.status()
        assert status['status'] == 'stale'
        assert status['analyzed_age_s'] >= 6
        assert not status['nudge_valid']
        assert status['nudge_left_mm'] == [0, 0, 0]
        assert monitor.analyzed_frame() == (1, b'x')
    finally:
        monitor.close()


def test_partial_current_config_and_initial_disabled_status():
    from app.pin_alignment_monitor import validate_config
    assert validate_config({'gain': .3}, current={})['gain'] == .3
    result = validate_config({'gain': .3}, current={'hold_s': 12.})
    assert result['hold_s'] == 12. and result['frequency_hz'] == 20.
    monitor = PinAlignmentMonitor(config={'enabled': False})
    try:
        assert monitor.status()['status'] == 'disabled'
        assert not monitor.status()['ready']
    finally:
        monitor.close()


def test_capture_requires_exact_fresh_supported_frame_and_reset_clears():
    monitor = PinAlignmentMonitor()
    try:
        def publish(source='demo_reference_tracked', stamp=None):
            monitor._publish(dict(status='ready', localization_ready=True,
                analyzed_timestamp_ns=stamp or time.monotonic_ns(), hole_source=source,
                hole_px=[300., 200.], cache={'mode': 'tracked'}),
                {'request_id': 7, 'generation': monitor._generation}, {}, time.monotonic())
        publish('cnn_unverified_fallback')
        with pytest.raises(ValueError): monitor.capture_hole(7)
        publish()
        with pytest.raises(ValueError): monitor.capture_hole(6)
        captured = monitor.capture_hole(7)
        assert captured['hole_px'] == [300., 200.]
        assert monitor.status()['captured_hole']['source_frame_id'] == 7
        assert not monitor.status()['nudge_valid']
        with pytest.raises(ValueError): monitor.capture_hole(7)
        monitor.configure(gain=.4)
        assert monitor.status()['captured_hole'] is not None
        monitor.reset()
        assert monitor.status()['captured_hole'] is None
        publish(stamp=time.monotonic_ns()-6_000_000_000)
        with pytest.raises(ValueError): monitor.capture_hole(7)
    finally:
        monitor.close()


def test_actual_step_hard_limit_config():
    assert validated_config({'max_step_mm': 5.})['max_step_mm'] == 5.
    with pytest.raises(ValueError): validated_config({'max_step_mm': 5.01})


def test_explicit_operator_capture_does_not_promote_cnn_fallback():
    monitor = PinAlignmentMonitor()
    try:
        monitor._publish(dict(status='waiting_for_hole', localization_ready=False,
            analyzed_timestamp_ns=time.monotonic_ns(), hole_source='cnn_unverified_fallback',
            hole_px=[10., 10.], cache={'mode': 'lost'}),
            {'request_id': 8, 'generation': 0}, {}, time.monotonic())
        with pytest.raises(ValueError): monitor.capture_hole(8)
        for invalid in ([float('nan'), 20], [-1, 20], [640, 20], [True, 20]):
            with pytest.raises(ValueError): monitor.capture_hole(8, invalid)
        with pytest.raises(ValueError): monitor.capture_hole(7, [300, 200])
        record = monitor.capture_hole(8, [300, 200])
        assert record['operator_selected'] is True
        assert record['source'] == 'captured_operator_reference'
        assert record['hole_px'] == [300, 200]
        assert record['source_cache']['mode'] == 'lost'
    finally:
        monitor.close()


def test_recent_exact_snapshot_survives_newer_result_and_captures_original():
    monitor = PinAlignmentMonitor(config={'frequency_hz': 2.})
    try:
        def publish(frame, x):
            monitor._publish(dict(status='ready', localization_ready=True,
                analyzed_timestamp_ns=time.monotonic_ns(), hole_source='demo_reference_tracked',
                hole_px=[x, 200.], measured_state14=[frame]*14,
                analyzed_jpeg_b64=base64.b64encode(bytes([frame])).decode()),
                {'request_id': frame, 'generation': monitor._generation}, {}, time.monotonic())
        publish(1, 100.)
        publish(2, 200.)
        assert monitor.status()['frame_id'] == 2
        assert monitor.status_for_frame(1)['measured_state14'] == [1]*14
        assert monitor.analyzed_frame(1) == (1, bytes([1]))
        captured = monitor.capture_hole(1)
        assert captured['hole_px'] == [100., 200.]
        assert captured['source_frame_id'] == 1
        assert monitor.status_for_frame(1) is None  # Capture changes generation.
        assert monitor.analyzed_frame(1) == (None, None)
        monitor.reset()
        for frame in range(3, 17): publish(frame, 100.)
        assert monitor.status_for_frame(3) is None  # At 2 Hz the window retains 13 frames.
        assert monitor.status_for_frame(4) is not None
        monitor.configure(gain=.3)
        assert monitor.status_for_frame(4) is None
    finally:
        monitor.close()


def test_expired_snapshot_cannot_be_retrieved_or_captured():
    monitor = PinAlignmentMonitor()
    try:
        monitor._publish(dict(status='ready', localization_ready=True,
            analyzed_timestamp_ns=time.monotonic_ns()-5_100_000_000,
            hole_source='demo_reference_tracked', hole_px=[100., 200.],
            analyzed_jpeg_b64=base64.b64encode(b'x').decode()),
            {'request_id': 1, 'generation': 0}, {}, time.monotonic())
        assert monitor.status_for_frame(1) is None
        assert monitor.analyzed_frame(1) == (None, None)
        with pytest.raises(ValueError): monitor.capture_hole(1)
    finally:
        monitor.close()


def test_manual_pin_persists_and_preserves_display_but_invalidates_action():
    monitor = PinAlignmentMonitor()
    try:
        monitor._publish(dict(status='waiting_for_hole',
            analyzed_timestamp_ns=time.monotonic_ns()-900_000_000,
            analyzed_jpeg_b64=base64.b64encode(b'image').decode()),
            {'request_id': 1, 'generation': 0}, {}, time.monotonic())
        record = monitor.capture_pin(1, [220., 180.])
        assert record['pin_px'] == [220., 180.]
        assert monitor.analyzed_frame() == (1, b'image')
        assert monitor.status()['display_frame_id'] == 1
        assert monitor.status_for_frame(1) is None
        monitor.configure(gain=.4)
        assert monitor.status()['captured_pin']['pin_px'] == [220., 180.]
        monitor.clear_pin()
        assert monitor.status()['captured_pin'] is None
        monitor.reset()
        assert monitor.status()['captured_pin'] is None
    finally:
        monitor.close()


def test_clear_points_independently_and_reset_clears_both():
    monitor = PinAlignmentMonitor()
    try:
        monitor._captured_pin = {'pin_px': [100, 100], 'captured_timestamp_ns': time.monotonic_ns()}
        monitor._captured_hole = {'hole_px': [200, 200], 'captured_timestamp_ns': time.monotonic_ns()}
        monitor.clear_hole()
        assert monitor.status()['captured_pin'] is not None
        assert monitor.status()['captured_hole'] is None
        monitor._captured_hole = {'hole_px': [200, 200], 'captured_timestamp_ns': time.monotonic_ns()}
        monitor.clear_pin()
        assert monitor.status()['captured_pin'] is None
        assert monitor.status()['captured_hole'] is not None
        monitor.reset()
        assert monitor.status()['captured_hole'] is None
    finally:
        monitor.close()


def test_freshness_window_independent_of_cadence_and_explicit_override_retained():
    assert validated_config()['max_frame_age_ms'] == 5000.
    for frequency in (.2, 2., 10.):
        assert validated_config({'frequency_hz': frequency})['max_frame_age_ms'] == 5000.
    assert validated_config({'frequency_hz': .2, 'max_frame_age_ms': 500.})['max_frame_age_ms'] == 500.
    assert validated_config({'max_frame_age_ms': 10000.})['max_frame_age_ms'] == 10000.
    assert validated_config({'frequency_hz': 10.}, {'max_frame_age_ms': 8000.})['max_frame_age_ms'] == 8000.
    for age in (99., 10001.):
        with pytest.raises(ValueError): validated_config({'max_frame_age_ms': age})


def test_snapshot_retention_covers_configured_window_and_is_bounded():
    monitor = PinAlignmentMonitor(config={'frequency_hz': 10., 'max_frame_age_ms': 10000.})
    try:
        stamp = time.monotonic_ns()-4_000_000_000
        for frame in range(1, 105):
            monitor._publish(dict(status='ready', nudge_valid=True,
                analyzed_timestamp_ns=stamp,
                analyzed_jpeg_b64=base64.b64encode(bytes([frame])).decode()),
                {'request_id': frame, 'generation': monitor._generation}, {}, time.monotonic())
        assert len(monitor._snapshots) == 103
        assert monitor.status_for_frame(1) is None
        assert monitor.status_for_frame(2)['nudge_valid']
        assert monitor.analyzed_frame(2) == (2, bytes([2]))
    finally:
        monitor.close()
