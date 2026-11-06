import base64
import time
import numpy as np
import pytest
from app.pin_alignment_worker import AlignmentWorker
from app.pin_alignment_monitor import validated_config
from app import pin_alignment_geometry as geometry


class Cache:
    mode = 'tracked'
    def reset(self): pass
    def update(self, image, **kw):
        self.last = kw
        return dict(mode=self.mode, hole_px=[300, 200] if self.mode == 'tracked' else None,
                    source_ns=kw['timestamp_ns'])


@pytest.fixture
def worker(monkeypatch):
    worker = AlignmentWorker.__new__(AlignmentWorker)
    worker.cache = Cache()
    worker.predictor = type('Predictor', (), {'predict': lambda self, rgb, state: {'keypoints': [.4, .4, .5, .5]}})()
    worker.epoch = None
    worker.checkpoint = 'fake'; worker.references = 'fake'
    calls = []
    def estimate(state, pin, hole, **kw):
        calls.append((pin, hole))
        return geometry.preview_from_geometry(pin, hole, np.diag([100., 100., 1.]),
            np.eye(4), [0, 0, 1], [0, 0, 1], np.eye(3), **kw)
    monkeypatch.setattr(geometry, 'estimate_preview_nudge', estimate)
    worker.calls = calls
    return worker


def request(age_ns=0):
    stamp = time.monotonic_ns()-age_ns
    return dict(config=validated_config(), timestamp_ns=stamp, measured_timestamp_ns=stamp,
        image=dict(kind='rgb', data=base64.b64encode(bytes(640*480*3)).decode()),
        state=[0.]*14, cache_epoch=0, context_id='same-grasp', camera_id='right', grasp_id='same')


def test_pixel_units_direction_bounded_step_and_jpeg(worker):
    import cv2
    result = worker.score(request())
    assert worker.calls == [([256., 192.], [300., 200.])]
    assert result['pixel_error_px'] > 3
    assert result['nudge_valid'] and result['localization_ready']
    assert result['nudge_left_mm'][0] > 0
    assert np.linalg.norm(result['nudge_left_mm']) == pytest.approx(5.)
    assert result['arrow_kind'] == 'alignment_direction'
    assert not result['motor_permission'] and not result['cnn_gate_permission']
    jpeg = np.frombuffer(base64.b64decode(result['analyzed_jpeg_b64']), np.uint8)
    assert cv2.imdecode(jpeg, cv2.IMREAD_COLOR).shape == (480, 640, 3)


def test_old_frame_updates_cache_causally_but_cannot_propose(worker):
    req = request(2_000_000_000)
    result = worker.score(req)
    assert worker.cache.last['now_ns'] == req['timestamp_ns']
    assert result['stale_result'] and not result['nudge_valid']
    assert result['arrow_kind'] == 'none'


@pytest.mark.parametrize('timestamp', [None, 'future'])
def test_missing_or_future_state_localizes_but_blocks_nudge(worker, timestamp):
    req = request()
    req['measured_timestamp_ns'] = None if timestamp is None else time.monotonic_ns()+10_000_000_000
    result = worker.score(req)
    assert result['localization_ready']
    assert not result['nudge_valid']
    assert result['nudge_left_mm'] == [0, 0, 0]
    assert result['nudge_blocked_reason']
    assert result['arrow_kind'] == 'none'


def test_unverified_cnn_hole_has_no_arrow_or_proposal(worker):
    worker.cache.mode = 'lost'
    result = worker.score(request())
    assert not worker.calls
    assert result['status'] == 'waiting_for_hole'
    assert not result['nudge_valid']
    assert result['hole_source'] == 'cnn_unverified_fallback'
    assert result['arrow_kind'] == 'none'


def test_deadband_localizes_without_arrow(worker):
    req = request()
    req['config']['deadband_px'] = 100.
    result = worker.score(req)
    assert result['nudge_valid']
    assert result['geometry']['reason'] == 'deadband'
    assert result['nudge_left_mm'] == [0, 0, 0]
    assert result['arrow_kind'] == 'none'


def test_captured_hole_retains_coordinate_without_visual_match_or_expiry(worker):
    worker.cache.mode = 'lost'
    req = request()
    req['captured_hole'] = dict(hole_px=[310., 210.],
        captured_timestamp_ns=time.monotonic_ns()-200_000_000_000,
        context={k:req[k] for k in ('camera_id', 'grasp_id', 'context_id')},
        assumption='rigid unchanged grasp')
    result = worker.score(req)
    assert result['hole_px'] == [310., 210.]
    assert result['hole_source'] == 'captured_demo_reference'
    assert result['nudge_valid']
    assert result['measured_state14'] == req['state']
    assert result['cache']['age_s'] >= 200
    assert result['visual_cache']['mode'] == 'lost'
    req['captured_hole']['context']['grasp_id'] = 'wrong'
    with pytest.raises(ValueError, match='context mismatch'): worker.score(req)


def test_operator_capture_distinct_authority_and_invalid_pin_rejected(worker):
    req = request()
    worker.cache.mode = 'lost'
    req['captured_hole'] = dict(hole_px=[310., 210.],
        captured_timestamp_ns=time.monotonic_ns(), source='captured_operator_reference',
        operator_selected=True, context={k:req[k] for k in ('camera_id','grasp_id','context_id')},
        assumption='operator selected; rigid unchanged grasp')
    result = worker.score(req)
    assert result['hole_source'] == 'captured_operator_reference'
    assert result['captured_hole']['operator_selected']
    assert result['nudge_valid']
    worker.predictor.predict = lambda *args: {'keypoints': [float('nan'), .4, .5, .5]}
    with pytest.raises(ValueError, match='pin localization invalid'): worker.score(req)


def test_operator_pin_survives_cnn_changes_and_coexists_with_hole(worker):
    req = request()
    context = {k:req[k] for k in ('camera_id','grasp_id','context_id')}
    req['captured_pin'] = dict(pin_px=[200., 180.], context=context)
    req['captured_hole'] = dict(hole_px=[310., 210.],
        captured_timestamp_ns=time.monotonic_ns(), source='captured_operator_reference',
        operator_selected=True, context=context, assumption='fixed')
    first = worker.score(req)
    worker.predictor.predict = lambda *args: {'keypoints': [float('nan'), .9, .5, .5]}
    second = worker.score(req)
    assert first['pin_px'] == second['pin_px'] == [200., 180.]
    assert second['pin_source'] == 'operator_selected'
    assert second['nudge_valid']
